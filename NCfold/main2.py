import os, gc
import random
import argparse

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
# Fix fastai bug to enable fp16 training with dictionaries

import fastai
from fastai.vision.all import Callback, L, to_float, CancelStepException, delegates, DataLoaders
from fastai.vision.all import SaveModelCallback, EarlyStoppingCallback, GradientClip, Learner

from .dataset import get_dataset
from .model import get_model, get_loss, myMetric, cal_metric_batch
from .util.yaml_config import write_yaml, get_config, update_config
from .util.postprocess import postprocess, apply_constraints
from .util.data_sampler import LenMatchBatchSampler, DeviceMultiDataLoader
from .util.RNA_kit import write_SS, arr2connects, remove_lone_pairs


def flatten(o):
    "Concatenate all collections and items as a generator"
    for item in o:
        if isinstance(o, dict): yield o[item]; continue
        elif isinstance(item, str): yield item; continue
        try: yield from flatten(item)
        except TypeError: yield item


@delegates(GradScaler)
class MixedPrecision(Callback):
    "Mixed precision training using Pytorch's `autocast` and `GradScaler`"
    order = 10
    def __init__(self, **kwargs): self.kwargs = kwargs
    def before_fit(self): 
        self.autocast,self.learn.scaler,self.scales = autocast(),GradScaler(**self.kwargs),L()
    def before_batch(self): self.autocast.__enter__()
    def after_pred(self):
        if next(flatten(self.pred)).dtype==torch.float16: self.learn.pred = to_float(self.pred)
    def after_loss(self): self.autocast.__exit__(None, None, None)                       
    def before_backward(self): self.learn.loss_grad = self.scaler.scale(self.loss_grad)
    def before_step(self):
        "Use `self` as a fake optimizer. `self.skipped` will be set to True `after_step` if gradients overflow. "
        self.skipped=True
        self.scaler.step(self)
        if self.skipped: raise CancelStepException()
        self.scales.append(self.scaler.get_scale())
    def after_step(self): self.learn.scaler.update()

    @property 
    def param_groups(self): 
        "Pretend to be an optimizer for `GradScaler`"
        return self.opt.param_groups
    def step(self, *args, **kwargs): 
        "Fake optimizer step to detect whether this batch was skipped from `GradScaler`"
        self.skipped=False
    def after_fit(self): self.autocast,self.learn.scaler,self.scales = None,None,None
fastai.callback.fp16.MixedPrecision = MixedPrecision
        

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train(opts):
    data_name = opts['dataset']['data_name']
    run_name = opts.basic.run_name
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # learning paras
    loss_func = get_loss(opts.learning.loss)(pos_weight=opts.learning.pos_weight, device=device) # pos_weight: weight of positive samples
    bs = opts.learning.batch_size
    num_workers = opts.learning.num_workers
    pin_memory = opts.learning.pin_memory
    metric_name = opts.learning.metric_name
    nfolds = opts.common.nfolds if opts.common.nfolds > 1 else 1
    fold_list = opts.learning.fold_list
    model_name = opts['model']['model_name']
    model_opts = opts['model'][model_name]
    data_opts = opts['dataset'][data_name]
    common_opts = opts['common']
    data_opts.update(common_opts)
    model_opts.update(common_opts)

    if fold_list is None:
        fold_list = list(range(nfolds))
    else:
        fold_list = [int(f) for f in fold_list if 0<=int(f)<nfolds]

    # model paras
    RNA_model = get_model(model_name)

    # data paras
    data_class = get_dataset(data_name)

    for fold in fold_list: # running multiple folds may cause OOM
        print(f'[Info] Training fold {fold}/{nfolds}')
        data_opts['fold'] = fold
        ds_train = data_class(phase='train', **data_opts)
        ds_train_len = data_class(phase='train', mask_only=True, **data_opts)
        sampler_train = torch.utils.data.RandomSampler(ds_train_len)
        len_sampler_train = LenMatchBatchSampler(sampler_train, batch_size=bs, drop_last=True)
        gc.collect()

        model = RNA_model(**model_opts)
        model = model.to(device)

        # load checkpoint
        begin_epoch = 0
        if opts.learning.load_checkpoint:
            ckpt_dir = os.path.join(run_name, 'models')
            pat = f'{model_name}_{fold+1}-{nfolds}'
            checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith(pat) and f.endswith('.pth')]
            if checkpoints:
                best_ckpt = max(checkpoints, key=lambda f: int(f[:f.rfind('.')].split('_')[-1]))
                begin_epoch = int(best_ckpt[:best_ckpt.rfind('.')].split('_')[-1])
                model.load_state_dict(torch.load(os.path.join(ckpt_dir, best_ckpt),map_location=torch.device('cpu')))

        print("model paras:",sum(p.numel() for p in model.parameters() if p.requires_grad))


        # set optimizer
        optim = torch.optim.Adam(model.parameters(), lr=opts.learning.lr, weight_decay=0.0001)
        
        checkpoint_dir = os.path.join(run_name, 'models')
        pbar = tqdm(range(begin_epoch, begin_epoch+opts.learning.epoch))
        avg_loss = 0
        model.train()  # important
        for epoch in pbar:
            cur_loss = 0
            dl_train_ori = torch.utils.data.DataLoader(ds_train, batch_sampler=len_sampler_train, num_workers=num_workers, persistent_workers=True, pin_memory=pin_memory)
            dl_train = DeviceMultiDataLoader([dl_train_ori], device, keywords=ds_train.to_device_keywords)
            batch_num = len(dl_train)
            for i, (data_dic, y) in enumerate(dl_train):
                pred = model(data_dic)
                loss = loss_func(pred, y)
                optim.zero_grad()
                cur_loss += loss.item()
                loss.backward()
                optim.step()
                pbar.set_description(f"[train] epoch:{epoch:>3d}/{begin_epoch+opts.learning.epoch}, batch:{i+1:>6d}/{batch_num:<6d}, train_loss/avg_loss:{loss.item():.6f}/{avg_loss:.6f}")
            avg_loss = cur_loss / batch_num
            save_name = f"{model_name}_{fold+1}-{nfolds}_{epoch}.pth"
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, save_name))

        # TODO : validate
        model.eval()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--phase', choices=['train', 'test'], default='train')
    # basic configuration
    parser.add_argument('-c', '--config', type=str, default='configs/config.yaml')
    parser.add_argument('-g', '--gpu', type=str, default='0')
    parser.add_argument('--ignore_fold', action='store_true')
    parser.add_argument('--run_name', type=str, default='dim256')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_epoch', nargs='*', help='epochs of checkpoints corr to fold at test stage')

    # dataset configuration
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--Lmax', type=int, help='max length of RNA seq, [Lmin, Lmax]')
    parser.add_argument('--Lmin', type=int, help='min length of RNA seq, [Lmin, Lmax]')
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--index_name', type=str)
    parser.add_argument('--method', choices=['EternaFold', 'CDPfold', 'Contrafold', 'ViennaRNA'])
    parser.add_argument('--trainall', action='store_true')
    parser.add_argument('--normalize_energy', action='store_true')
    parser.add_argument('--training_set', nargs='*', default=None)
    parser.add_argument('--test_set', nargs='*', default=None)

    # learning paras, training
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--earlystop', action='store_true')
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--fold_list', nargs='*', help='fold num list, 0 <= num < nfolds')
    parser.add_argument('--gradientclip', type=float)
    parser.add_argument('--load_checkpoint', action='store_true', help='load checkpoint when training')
    parser.add_argument('--loss', choices=['BCE', 'MSE'])
    parser.add_argument('--lr', type=float)
    parser.add_argument('--metric_name', type=str, default='F1', choices=['F1', 'INF', 'MCC',])

    parser.add_argument('--nfolds', type=int, help='nfolds<=1 for no kfold')
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--pos_weight', type=float)
    parser.add_argument('--save_freq', type=int)

    # model paras
    parser.add_argument('--model_name', choices=['NCfold'])
    parser.add_argument('--depth', type=int)
    parser.add_argument('--dim', type=int)
    parser.add_argument('--head_size', type=int)
    parser.add_argument('--not_slice', action='store_true')
    parser.add_argument('--positional_embedding', choices=['dyn', 'alibi'])
    
    parser.add_argument('--use_BPE', action='store_true')
    parser.add_argument('--use_BPP', action='store_true')
    # End

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    opts = get_config(args.config)
    update_config(opts, args)

    seed_everything(opts.common.seed)

    phase = args.phase
    run_name = opts.basic.run_name
    os.makedirs(run_name, exist_ok=True)
    write_yaml(os.path.join(run_name, f"config_{phase}.yaml"), opts)
    print(opts)
    if phase == 'train':
        train(opts)
