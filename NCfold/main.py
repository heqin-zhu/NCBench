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
    ckpt_dir = opts['basic']['ckpt_dir']
    model_name = opts['model']['model_name']
    model_opts = opts['model'][model_name]
    data_opts = opts['dataset'][data_name]
    data_opts.update(opts['common'])
    model_opts.update(opts['common'])

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
        dl_train_ori = torch.utils.data.DataLoader(ds_train, batch_sampler=len_sampler_train, num_workers=num_workers, persistent_workers=True, pin_memory=pin_memory)
        dl_train = DeviceMultiDataLoader([dl_train_ori], device, keywords=ds_train.to_device_keywords)

        ds_val = data_class(phase='validate', **data_opts)
        ds_val_len = data_class(phase='validate', mask_only=True, **data_opts)
        sampler_val = torch.utils.data.SequentialSampler(ds_val_len)
        len_sampler_val = LenMatchBatchSampler(sampler_val, batch_size=bs, drop_last=False)
        dl_val_ori = torch.utils.data.DataLoader(ds_val, batch_sampler=len_sampler_val, num_workers=num_workers, pin_memory=pin_memory)
        dl_val= DeviceMultiDataLoader([dl_val_ori], device, keywords=ds_train.to_device_keywords)
        gc.collect()

        model = RNA_model(**model_opts)
        model = model.to(device)

        # load checkpoint
        begin_epoch = 0
        if opts.learning.load_checkpoint:
            if ckpt_dir is None:
                ckpt_dir = os.path.join(run_name, 'models')
            pat = f'{model_name}_{fold+1}-{nfolds}'
            checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith(pat) and f.endswith('.pth')]
            if checkpoints: # load checkpoint of max-epoch
                best_ckpt = max(checkpoints, key=lambda f: int(f[:f.rfind('.')].split('_')[-1]))
                begin_epoch = int(best_ckpt[:best_ckpt.rfind('.')].split('_')[-1])
                model.load_state_dict(torch.load(os.path.join(ckpt_dir, best_ckpt),map_location=torch.device('cpu')))
        flag = '' if begin_epoch==0 else f'_begin{begin_epoch}'

        cbs_savemodel = SaveModelCallback(monitor=metric_name, comp=np.greater, min_delta=0.0, fname=f'{model_name}_{fold+1}-{nfolds}'+flag, every_epoch=opts.learning.save_freq, at_end=False, with_opt=False, reset_on_fit=True)
        cbs = [cbs_savemodel]
        if opts.learning.earlystop:
            cbs_earlystop = EarlyStoppingCallback(monitor='valid_loss', comp=None, min_delta=0.0, patience=10, reset_on_fit=True)
            cbs.append(cbs_earlystop)
        if opts.learning.gradientclip:
            cbs.append(GradientClip(opts.learning.gradientclip)) 

        print("model paras:",sum(p.numel() for p in model.parameters() if p.requires_grad))
        learn = Learner(DataLoaders(dl_train, dl_val), model, path=f'{run_name}', loss_func=loss_func, cbs=cbs, metrics=[myMetric(metric_name, device)]).to_fp16() 
        learn.fit_one_cycle(opts.learning.epoch, lr_max=opts.learning.lr, wd=0.05, pct_start=0.02)
        # torch.save(learn.model.state_dict(), os.path.join(run_name, f'{model_name}_{fold+1}-{nfolds}.pth'))
        gc.collect()

        if begin_epoch>0:
            for f in os.listdir(ckpt_dir):
                if 'begin' in f:
                    idx = f.find('begin')
                    begin_epoch = int(f[idx+5:].split('_')[0])
                    cur_epoch = int(f[:f.rfind('.')].split('_')[-1])
                    f_new = f[:idx] + str(1+begin_epoch+cur_epoch)+f[f.rfind('.'):]
                    os.rename(os.path.join(ckpt_dir, f), os.path.join(ckpt_dir, f_new))


def load_eval_checkpoints(ckpt_dir, RNA_model, model_opts, device, ckpt_names=None):
    if not os.path.exists(ckpt_dir):
        raise Exception(f'[Error] Checkpoint directory not exist: {ckpt_dir}')
    models = []
    if ckpt_names is None:
        ckpt_names = sorted(os.listdir(ckpt_dir))
    for ckpt_name in ckpt_names:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        print(f'Loading {ckpt_path}')
        model = RNA_model(**model_opts)
        model = model.to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
        model.eval()
        models.append(model)
    if models == []:
        raise Exception(f'[Error] No checkpoint found in {ckpt_dir}')
    return models


def test(opts):
    def get_model_name_epoch(ckpt_dir, fold_epoch=None):
        ''' return the k-fold checkpoints with biggest epoch '''
        files = [name for name in os.listdir(ckpt_dir) if (name.endswith('.pth') or name.endswith('.pt'))]
        if files:
            nfolds = None
            fold_name = {}
            for f in files:
                p = f.rfind('.')
                name = f[:p] if p!=-1 else f
                name = '_'.join([seg for seg in name.split('_') if 'begin' not in seg])
                name_segs = name.split('_')
                model_name, fold_and_nfold, epoch_and_suf = name_segs if len(name_segs)==3 else (*name_segs, None)
                fold, nfolds = fold_and_nfold.split('-')
                fold, nfolds = int(fold), int(nfolds)
                epoch = int(epoch_and_suf) if epoch_and_suf is not None else 10000
                if fold_epoch is None:
                    if fold not in fold_name or fold_name[fold]['epoch'] <= epoch:
                        fold_name[fold] = {'name': f, 'epoch': epoch}
                else:
                    if fold_epoch[fold] == epoch:
                        fold_name[fold] = {'name': f, 'epoch': epoch}
            if not opts.basic.ignore_fold:
                cur_folds = set(fold_name.keys())
                expected_folds = set(range(1, 1+int(nfolds)))
                assert expected_folds == cur_folds, f'Expected folds: {expected_folds}, Got {cur_folds}'
            return sorted([name_dic for name_dic in fold_name.values()], key=lambda d: d['name'])
        else:
            raise Exception('No checkpoint found.')
    data_name = opts['dataset']['data_name']
    run_name = opts.basic.run_name
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = opts['model']['model_name']
    model_opts = opts['model'][model_name]
    ep_list = opts.basic.test_epoch
    ckpt_dir = opts['basic']['ckpt_dir']
    save_contact_map = opts['basic']['save_contact_map']
    data_opts = opts['dataset'][data_name]
    learn_opts = opts['learning']
    data_opts.update(opts['common'])
    model_opts.update(opts['common'])

    RNA_model = get_model(model_name)
    data_class = get_dataset(data_name)

    # to device
    ds = data_class(phase='test', **data_opts)
    dl_ori = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, drop_last=False, num_workers=learn_opts.num_workers)
    dl = DeviceMultiDataLoader([dl_ori], device, keywords=ds.to_device_keywords)

    fold_epoch = None
    if ep_list is not None:
        if opts.common.nfolds!=len(ep_list):
            fold_epoch = {i+1: int(ep_list[0]) for i in range(opts.common.nfolds)}
        else:
            fold_epoch = {i+1: int(epoch) for i, epoch in enumerate(ep_list)}
    models = None
    pred_dir = None
    if ckpt_dir and os.path.exists(ckpt_dir):
        models = load_eval_checkpoints(ckpt_dir, RNA_model, model_opts, device)
        pred_dir = os.path.join(run_name, f'pred_results')
    else:
        ckpt_dir = os.path.join(run_name, 'models')
        if os.path.exists(ckpt_dir):
            name_epoch_list = get_model_name_epoch(ckpt_dir, fold_epoch)
            max_epoch = max([d['epoch'] for d in name_epoch_list])
            epoch_str = f'epoch-{max_epoch}'
            ckpt_names = [d['name'] for d in name_epoch_list]
            models = load_eval_checkpoints(ckpt_dir, RNA_model, model_opts, device, ckpt_names)
            pred_dir = os.path.join(run_name, f'pred_{epoch_str}') # save pred
        else:
            raise Exception("No checkpiont found, please specify argument '--ckpt_dir'.")

    os.makedirs(pred_dir, exist_ok=True)
    metric_data = {}
    for data_dic, _ in tqdm(dl): # batch-wise
        # nc: non-canonical
        with torch.no_grad(),torch.cuda.amp.autocast():
            # torch.nan_to_num
            # BS x forward_batch_Lmax x forward_batch_Lmax
            pred_batch = torch.stack([model(data_dic) for model in models], 0).mean(0)

            # remove `begin` and `end` tokens
            forward_batch_Lmax = data_dic['forward_mask'].sum(-1).max()
            batch_Lmax = forward_batch_Lmax-2
            pred_batch = pred_batch[:, 1:batch_Lmax+1, 1:batch_Lmax+1]
            mask = data_dic['mask'][:, 1:batch_Lmax+1, 1:batch_Lmax+1]
            seq_onehot = data_dic['seq_onehot'][:, 1:batch_Lmax+1, :]
            gt = data_dic['gt'][:, 1:batch_Lmax+1, 1:batch_Lmax+1]
            nc_map = data_dic['nc_map'][:, 1:batch_Lmax+1, 1:batch_Lmax+1]
            gt_nc = nc_map*gt

            # postprocess
            ret_pred, _, ret_score, _ = postprocess(pred_batch, seq_onehot, nc_map, return_nc=False, return_score=False)


            # save pred
            for i in range(len(ret_pred)):
                dataset = data_dic['dataset'][i]
                dataset = 'PDB_test' if dataset.startswith('PDB') else dataset
                pred_data_dir = os.path.join(pred_dir, dataset)
                os.makedirs(pred_data_dir, exist_ok=True)
                path = os.path.join(pred_data_dir, data_dic['name'][i]+'.bpseq')
                length = data_dic['length'][i]
                connects = arr2connects(ret_pred[i][mask[i]].reshape(length, length))
                connects = remove_lone_pairs(connects)
                
                write_SS(path, data_dic['ori_seq'][i], connects, out_type='bpseq')
                ## save contact maps before and after postprocessing
                if save_contact_map:
                    save_data_dir = os.path.join(run_name, 'score_'+os.path.basename(pred_dir), dataset)
                    os.makedirs(save_data_dir, exist_ok=True)
                    ## save numpy arr, before/after postprocessing
                    name = data_dic['name'][i]
                    mat = pred_batch[i][mask[i]].reshape(length, length).detach().cpu().numpy()
                    mat_post = ret_pred[i][mask[i]].reshape(length, length).detach().cpu().numpy()
                    np.save(os.path.join(save_data_dir, f'{name}.npy'), mat)
                    np.save(os.path.join(save_data_dir, f'{name}_post.npy'), mat_post)
                    post_before_th = apply_constraints(pred_batch[i:i+1], seq_onehot[i:i+1], 0.01, 0.1, 100, 1.6, True, 1.5)[0]
                    np.save(os.path.join(save_data_dir, f'{name}_post_before_th.npy'), post_before_th[mask[i]].reshape(length, length).detach().cpu().numpy())

            # cal_metric
            metric_dic = cal_metric_batch(ret_pred, gt, mask, data_dic['name'], data_dic['dataset'])
            for dataset_name, dic in metric_dic.items():
                if dataset_name not in metric_data:
                    metric_data[dataset_name] = {}
                metric_data[dataset_name].update(dic)
            
    # save metric
    write_yaml(os.path.join(run_name, f'test_metric_Lmax{data_opts["Lmax"]}.yaml'), metric_data)

    # display metric
    for dataset_name, dic in metric_data.items():
        print(f'{dataset_name}: {len(dic)}')
        for metric in ['INF', 'F1', 'P', 'R']:
            print(metric, np.mean([d[metric] for d in dic.values()]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--phase', choices=['train', 'test'], default='train')
    # basic configuration
    parser.add_argument('--ckpt_dir', type=str, help='ckpt_dir that continas checkpoints, default=os.path.join(run_name, "models")')
    parser.add_argument('-c', '--config', type=str, default='configs/config.yaml')
    parser.add_argument('-g', '--gpu', type=str, default='0')
    parser.add_argument('--ignore_fold', action='store_true')
    parser.add_argument('--run_name', type=str, default='dim256')
    parser.add_argument('--save_contact_map', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_epoch', nargs='*', help='epochs of checkpoints corr to fold at test stage')

    # dataset configuration
    parser.add_argument('--BPM_type', type=str, default='all', choices=['all', 'BPM_iH', 'BPM_iCB', 'BPM_oCB'])
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--index_name', type=str)
    parser.add_argument('--Lmax', type=int, help='max length of RNA seq, [Lmin, Lmax]')
    parser.add_argument('--Lmin', type=int, help='min length of RNA seq, [Lmin, Lmax]')
    parser.add_argument('--method', choices=['EternaFold', 'CDPfold', 'Contrafold', 'ViennaRNA'])
    parser.add_argument('--normalize_energy', action='store_true')
    parser.add_argument('--trainall', action='store_true')
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
    test(opts)
