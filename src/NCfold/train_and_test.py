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

from .dataset.RNAdata import RNAdata
from .dataset.collator import NCfoldCollator
from .model.NCfold import NCfold_model
from .model.loss_and_metric import NCfoldLoss, compute_metrics
from .util.NCfold_kit import str2bool, str2list, count_para, get_config, edge_orient_to_basepair


MODELS = ['NCfold_model']
DATASETS = ["PDB_NC"]


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
        self.name_pbar = 'edge_orient_score' # TODO, modify compute_metrics
        self.visual_writer = visual_writer
        self.max_metric = 0.
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
        checkpoint_dir = os.path.join(self.args.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        if metrics_dataset[self.name_pbar] > self.max_metric:
            self.max_metric = metrics_dataset[self.name_pbar]
            save_model_path = os.path.join(self.checkpoint_dir, f"epoch{epoch}_{self.max_metric:.3f}.pth")
            torch.save(self.model.state_dict(), save_model_path)
            print(f"Epoch={epoch}:", save_model_path)

    def train(self, epoch):
        raise NotImplementedError("Must implement train method.")

    def eval(self, epoch):
        raise NotImplementedError("Must implement eval method.")


class NCfoldTrainer(BaseTrainer):
    def train(self, epoch):
        self.model.train()
        time_st = time.time()
        num_total, loss_total = 0, 0

        with tqdm(total=len(self.train_dataset)) as pbar:
            for i, data in enumerate(self.train_dataloader):
                input_ids = data["input_ids"].to(self.args.device)
                mat = data["mat"].to(self.args.device)
                label_edge = data["label_edge"].to(self.args.device)
                label_orient = data["label_orient"].to(self.args.device)
                pred_edge, pred_orient = self.model(input_ids, mat)
                loss, loss_edge, loss_orient = self.loss_fn(pred_edge, pred_orient, label_edge, label_orient)
                # clear grads
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # log to pbar
                num_total += self.args.batch_size
                loss_total += loss.item()

                # reset loss if too many steps
                if num_total >= self.args.logging_steps:
                    pbar.set_postfix(train_loss='{:.4f}'.format(loss_total / num_total))
                    pbar.update(self.args.logging_steps)
                    num_total, loss_total = 0, 0
        time_ed = time.time() - time_st
        print('Train\tLoss: {:.6f}; Time: {:.4f}s'.format(loss.item(), time_ed))


    def eval(self, epoch):
        self.model.eval()
        time_st = time.time()
        num_total = 0
        outputs_names, outputs_seqs = [], []
        pred_edges, pred_orients, gt_edges, gt_orients = [], [], [], []
        with tqdm(total=len(self.eval_dataset)) as pbar:
            for i, data in enumerate(self.eval_dataloader):
                input_ids = data["input_ids"].to(self.args.device)
                mat = data["mat"].to(self.args.device)
                with torch.no_grad():
                    pred_edge, pred_orient = self.model(input_ids, mat)
                num_total += self.args.batch_size
                # out_pred = edge_orient_to_basepair(pred_edge.detach().cpu().numpy(), pred_orient.detach().cpu().numpy())
                # out_gt = edge_orient_to_basepair(data["label_edge"], data["label_orient"])
                pred_edges.append(pred_edge)
                pred_orients.append(pred_orient)
                gt_edges.append(data["label_edge"])
                gt_orients.append(data["label_orient"])
                outputs_names += data['name']
                outputs_seqs += data['seq']
                if num_total >= self.args.logging_steps:
                    pbar.update(self.args.logging_steps)
                    num_total = 0

        xs = [pred_edges, pred_orients, gt_edges, gt_orients]
        for i in range(len(xs)):
             xs[i]= torch.concat(xs[i], axis=0)
        metrics_dataset = self.compute_metrics(*xs)
        self.save_model(metrics_dataset, epoch)
        df = pd.DataFrame({
            "name": outputs_names,
            "seq": outputs_seqs,
            # TODO
            # "true_label": labels_dataset.detach().cpu().numpy(),
            # **{f"class_{i}_logit": outputs_dataset[:, i].detach().cpu().numpy() for i in range(outputs_dataset.shape[1])},
            # "predicted_class": np.argmax(outputs_dataset.detach().cpu().numpy(), axis=1),
        })
        df.to_csv(os.path.join(self.args.output_dir, f'test_{epoch}_{self.max_metric:.4f}.csv'), index=False)

        # log results to screen/bash
        results = {}
        log = 'Test\t' + self.args.dataset + "\t"
        for k, v in metrics_dataset.items():
            if 'f1' in k:
                log += k + ": {" + k + ":.4f}\t"
                results[k] = v
        time_ed = time.time() - time_st
        print(log.format(**results), "; Time: {:.4f}s".format(time_ed))


def get_args():
    parser = argparse.ArgumentParser('Implementation of RNA sequence classification.')
    # save checkpoint
    parser.add_argument('--output_dir', type=str, default='.runs/tmp')
    # model args
    parser.add_argument('--model_name', type=str, default="NCfold_model", choices=MODELS)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=16)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--dataset_dir', type=str, default="data")
    parser.add_argument('--dataset', type=str, default="PDB_NC", choices=DATASETS)
    parser.add_argument('--replace_T', type=bool, default=True)
    parser.add_argument('--replace_U', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--batch_size', type=int, default=16, help='The number of samples used per step & per device.')
    parser.add_argument('--num_train_epochs', type=int, default=60, help='The number of epoch for training.')
    # loss weight
    parser.add_argument('--weight_edgeW', type=float, default=5)
    parser.add_argument('--weight_edgeH', type=float, default=20)
    parser.add_argument('--weight_edgeS', type=float, default=20)
    parser.add_argument('--weight_trans', type=float, default=20)
    parser.add_argument('--weight_cis', type=float, default=20)
    args = parser.parse_args()
    return args


def train_and_test():
    args = get_args()
    assert args.replace_T ^ args.replace_U, "Only replace T or U."

    if args.model_name == "NCfold_model":
        model = NCfold_model(seq_dim=args.hidden_dim, mat_channels=args.hidden_dim, num_blocks=args.num_blocks)
    else:
        raise ValueError("Unknown model name: {}".format(args.model_name))
    os.makedirs(args.output_dir, exist_ok=True)
    model.to(args.device)
    count_para(model)

    _loss_fn = NCfoldLoss(
        edge_weight=1.0, 
        orient_weight=1.0,
        edge_weights=torch.tensor([1.0, args.weight_edgeW, args.weight_edgeH, args.weight_edgeS]),
        orient_weights=torch.tensor([1.0, args.weight_trans, args.weight_cis]),
    ).to(args.device)

    dataset_train = RNAdata(data_dir=args.dataset_dir)
    dataset_eval = RNAdata(data_dir=args.dataset_dir, train=False)
    print(f'dataset dir: {args.dataset_dir} {args.dataset}')
    print(f'dataset {args.dataset} train:test={len(dataset_train)}:{len(dataset_eval)}') 
    ## max_seq_len == None, for setting batch_max_len
    _collate_fn = NCfoldCollator(max_seq_len=None, replace_T=args.replace_T, replace_U=args.replace_U)
    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)
    trainer = NCfoldTrainer(
        args=args,
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        data_collator=_collate_fn,
        loss_fn=_loss_fn,
        optimizer=optimizer,
        compute_metrics=compute_metrics,
    )
    if args.train:
        for i_epoch in range(args.num_train_epochs):
            print("Epoch: {}".format(i_epoch))
            trainer.train(i_epoch)
            trainer.eval(i_epoch)

if __name__ == "__main__":
    train_and_test()
