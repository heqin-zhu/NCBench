import os
import time
import argparse
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from trainers import SeqClsTrainer
from .dataset.RNAdata import RNAdata
from .dataset.collator import NCfoldCollator
from .model.NCfold import NCfold_model
from .model.loss_and_metric import NCfoldLoss, compute_metrics
from .util.NCfold_kit import str2bool, str2list, count_para, get_config


MODELS = ['NCfold_model']
TASKS = ["NCfold"]


class BaseTrainer(object):
    def __init__(self,
                 args,
                 model,
                 pretrained_model=None,
                 indicator=None,
                 ensemble=None,
                 train_dataset=None,
                 eval_dataset=None,
                 data_collator=None,
                 loss_fn=None,
                 optimizer=None,
                 compute_metrics=None,
                 visual_writer=None):
        self.args = args
        self.model = model
        self.pretrained_model = pretrained_model
        self.indicator = indicator
        self.ensemble = ensemble
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.compute_metrics = compute_metrics
        # default name_pbar is the first metric
        self.name_pbar = self.compute_metrics.metrics[0]
        self.visual_writer = visual_writer
        self.max_metric = 0.
        self.max_model_dir = ""
        # init dataloaders
        self._prepare_dataloaders()

    def _get_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

    def _prepare_dataloaders(self):
        if self.train_dataset:
            self.train_dataloader = self._get_dataloader(self.train_dataset)

        if self.eval_dataset:
            self.eval_dataloader = self._get_dataloader(self.eval_dataset)

    def save_model(self, metrics_dataset, epoch):
        """
        Save model after epoch training in save_dir.
        Args:
            metrics_dataset: metrics of dataset
            epoch: training epoch number

        Returns:
            None
        """
        if metrics_dataset[self.name_pbar] > self.max_metric:
            self.max_metric = metrics_dataset[self.name_pbar]
            if os.path.exists(self.max_model_dir):
                print("Remove old max model dir:", self.max_model_dir)
                shutil.rmtree(self.max_model_dir)

            self.max_model_dir = osp.join(self.args.output_dir, f"epoch{epoch}_{self.max_metric:.4f}")
            os.makedirs(self.max_model_dir)
            save_model_path = osp.join(
                self.max_model_dir, "model_state.pdparams")
            torch.save(self.model.state_dict(), save_model_path)
            print("Model saved at:", save_model_path)

    def train(self, epoch):
        raise NotImplementedError("Must implement train method.")

    def eval(self, epoch):
        raise NotImplementedError("Must implement eval method.")


class SeqClsTrainer(BaseTrainer):
    def train(self, epoch):
        self.model.train()
        time_st = time.time()
        num_total, loss_total = 0, 0

        with tqdm(total=len(self.train_dataset), disable=self.args.disable_tqdm) as pbar:
            for i, data in enumerate(self.train_dataloader):
                input_ids = data["input_ids"].to(self.args.device)
                labels = data["labels"].to(self.args.device)
                mat = data["mat"].to(self.args.device)

                logits = self.model(input_ids, mat)
                loss = self.loss_fn(logits, labels)

                # clear grads
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # log to pbar
                num_total += self.args.batch_size
                loss_total += loss.item()

                # reset loss if too many steps
                if num_total >= self.args.logging_steps:
                    pbar.set_postfix(train_loss='{:.4f}'.format(
                        loss_total / num_total))
                    pbar.update(self.args.logging_steps)
                    num_total, loss_total = 0, 0

        time_ed = time.time() - time_st
        print('Train\tLoss: {:.6f}; Time: {:.4f}s'.format(loss.item(), time_ed))

    def eval(self, epoch):
        self.model.eval()
        time_st = time.time()
        num_total = 0
        with tqdm(total=len(self.eval_dataset), disable=self.args.disable_tqdm) as pbar:
            outputs_dataset, labels_dataset = [], []
            outputs_names, outputs_seqs = [], []
            for i, data in enumerate(self.eval_dataloader):
                input_ids = data["input_ids"].to(self.args.device)
                labels = data["labels"].to(self.args.device)
                mat = data["mat"].to(self.args.device)

                with torch.no_grad():
                    logits = self.model(input_ids, mat)

                num_total += self.args.batch_size
                outputs_dataset.append(logits)
                labels_dataset.append(labels)
                outputs_names += data['name']
                outputs_seqs += data['seq']

                if num_total >= self.args.logging_steps:
                    pbar.update(self.args.logging_steps)
                    num_total = 0

        outputs_dataset = torch.concat(outputs_dataset, axis=0)
        labels_dataset = torch.concat(labels_dataset, axis=0)
        # save best model
        metrics_dataset = self.compute_metrics(outputs_dataset, labels_dataset)
        self.save_model(metrics_dataset, epoch)
        pd.DataFrame()
        df = pd.DataFrame({
            "name": outputs_names,
            "seq": outputs_seqs,
            "true_label": labels_dataset.detach().cpu().numpy(),
            **{f"class_{i}_logit": outputs_dataset[:, i].detach().cpu().numpy() for i in range(outputs_dataset.shape[1])},
            "predicted_class": np.argmax(outputs_dataset.detach().cpu().numpy(), axis=1),
        })
        df.to_csv(os.path.join(self.args.output_dir, f'test_metric_{epoch}_{self.max_metric:.4f}.csv'), index=False)

        # log results to screen/bash
        results = {}
        log = 'Test\t' + self.args.dataset + "\t"
        # log results to visualdl
        tag_value = defaultdict(float)
        # extract results
        for k, v in metrics_dataset.items():
            log += k + ": {" + k + ":.4f}\t"
            results[k] = v
            tag = "eval/" + k
            tag_value[tag] = v

        time_ed = time.time() - time_st
        print(log.format(**results), "; Time: {:.4f}s".format(time_ed))


def get_args():
    parser = argparse.ArgumentParser('Implementation of RNA sequence classification.')
    # model args
    parser.add_argument('--model_name', type=str, default="NCfold_model", choices=MODELS)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--config_path', type=str, default="./configs/")
    parser.add_argument('--dataset_dir', type=str, default="../data")
    parser.add_argument('--dataset', type=str, default="NCfold", choices=TASKS)
    parser.add_argument('--replace_T', type=bool, default=True)
    parser.add_argument('--replace_U', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--batch_size', type=int, default=32, help='The number of samples used per step & per device.')
    parser.add_argument('--num_train_epochs', type=int, default=60, help='The number of epoch for training.')
    parser.add_argument('--metrics', type=str2list, default="F1s,Precision,Recall,Accuracy,Mcc",)
    # save checkpoint
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    assert args.replace_T ^ args.replace_U, "Only replace T or U."

    if args.model_name == "NCfold_model":
        model = NCfold_model(d_model=args.hidden_dim)
    else:
        raise ValueError("Unknown model name: {}".format(args.model_name))
    model.to(args.device)
    count_para(model)

    _loss_fn = NucClsLoss().to(args.device)
    if args.output_dir is None:
        run_name = os.path.basename(args.checkpoint_path)
        run_name = run_name[:run_name.rfind('.')]
        args.output_dir = os.path.join('outputs', run_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # ========== Prepare data
    dataset_train = NucClsDataset(data_dir=args.dataset_dir)
    dataset_eval = NucClsDataset(data_dir=args.dataset_dirtrain=False)
    print(f'dataset dir: {args.dataset_dir} {args.dataset}')
    print(f'dataset {args.dataset} train:test={len(dataset_train)}:{len(dataset_eval)}') 

    # ========== Create the data collator
    _collate_fn = NucClsCollator(max_seq_len=args.max_seq_len, replace_T=args.replace_T, replace_U=args.replace_U)

    # ========== Create the learning_rate scheduler (if need) and optimizer
    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)

    # ========== Create the metrics
    _metric = NucClsMetrics(metrics=args.metrics)

    # ========== Create the trainer
    seq_cls_trainer = SeqClsTrainer(
        args=args,
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        data_collator=_collate_fn,
        loss_fn=_loss_fn,
        optimizer=optimizer,
        compute_metrics=_metric,
    )
    if args.train:
        for i_epoch in range(args.num_train_epochs):
            print("Epoch: {}".format(i_epoch))
            seq_cls_trainer.train(i_epoch)
            seq_cls_trainer.eval(i_epoch)
