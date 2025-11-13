import os
import time
import json
import argparse
from datetime import datetime
from collections import defaultdict
import random
from pathlib import Path

from sqlalchemy import values
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset, Dataset 
from sklearn.model_selection import KFold 

from dataset.RNAdata import Kfold_RNAdata, RNAdata
from dataset.collator import NCfoldCollator
from dataset.LM_embeddings import LM_dim_dic
from model.RFM_model import RFM_net, SFM_net
from model.loss_and_metric import NCfoldLoss, compute_metrics
from model.AttnMatFusion_net import AttnMatFusion_net
from util.NCfold_kit import str2bool, str2list, count_para, get_config
from util.data_processing import extract_basepair_interaction, extract_basepair_interaction_gt
from util.utils_json import json_default
from util.mm_utils import *

import pickle

MODELS = ['AttnMatFusion_net']
DATASETS = ["PDB_NC"]
LMs = ['structRFM', 'rnaernie', 'rnafm', 'splicebert', 'utrlm-te_el', 'aido.rna-650m', 'rinalmo-micro']
SLMs = ['BPfold', 'RNAFold', 'MxFold2', 'ContraFold']
MM = ['RF', 'Gradient Boosting', 'XGBoost', 'SGD', 'Logistic Regression', 'MLP', 'SVM', 'KNN']


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class BaseTrainer(object):
    def __init__(
                 self,
                 args,
                 model,
                 pretrained_model=None,
                 indicator=None,
                 ensemble=None,
                 train_dataset=None,
                 eval_dataset=None,
                 test_dataset=None,
                 data_collator=None,
                 loss_fn=None,
                 optimizer=None,
                 compute_metrics=None,
                 visual_writer=None,
                 best_ckpt_path=None,
                ):
        self.args = args
        self.model_name = self.args.model_name 
        self.model = model
        self.pretrained_model = pretrained_model
        self.indicator = indicator
        self.ensemble = ensemble
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.data_collator = data_collator
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.compute_metrics = compute_metrics
        self.prime_metric = 'nc_score'
        self.visual_writer = visual_writer
        self.max_metric = 0.
        self.best_ckpt_path = best_ckpt_path
        if self.best_ckpt_path and os.path.exists(self.best_ckpt_path):
            self.model.load_state_dict(torch.load(self.best_ckpt_path, map_location=self.args.device))
            print(f'Loading checkpoint: {self.best_ckpt_path}')
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
            
        if self.test_dataset:
            self.test_dataloader = self._get_dataloader(self.test_dataset)

    def save_model(self, metrics_dataset, epoch, cur_k):
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
        if metrics_dataset[self.prime_metric] > self.max_metric:
            self.max_metric = metrics_dataset[self.prime_metric]
            save_model_path = os.path.join(checkpoint_dir, f"epoch{epoch}_{self.max_metric:.3f}.pth")
            torch.save(self.model.state_dict(), save_model_path)
            print("Save model:", save_model_path)
            
            ## save the best result 
            save_model_path = os.path.join(checkpoint_dir, f"best_epoch{epoch}_{self.max_metric:.3f}.pth")
            torch.save(self.model.state_dict(), save_model_path)
            fixed_path = os.path.join(checkpoint_dir, "best.pth")
            torch.save(self.model.state_dict(), fixed_path)
            self.best_ckpt_path = fixed_path
            print("Save best model:", save_model_path)

    def train(self, epoch):
        raise NotImplementedError("Must implement train method.")

    def eval(self, epoch):
        raise NotImplementedError("Must implement eval method.")
    
    def test(self):
        raise NotImplementedError("Must implement test method.")

## RFM and secondary structure prediction-based model 
class RFMBaselineTrainer(BaseTrainer):
    def train(self, epoch, cur_k):
        self.model.train()
        time_st = time.time()
        num_total, loss_total = 0, 0
        mean_loss = float('inf')

        with tqdm(total=len(self.train_dataset)) as pbar:
            for i, data in enumerate(self.train_dataloader):
                # move tensor to device
                input_ids = data["input_ids"].to(self.args.device)
                mat = data["mat"].to(self.args.device)
                mat_mask = data["mat_mask"].to(self.args.device)
                seq_mask = data["seq_mask"].to(self.args.device)
                label_edge = data["label_edge"].to(self.args.device)
                label_orient = data["label_orient"].to(self.args.device)
                LM_embed = data["LM_embed"].to(self.args.device) if 'LM_embed' in data else None
                
                pred_edge, pred_orient = self.model(input_ids, LM_embed, seq_mask, mat_mask)
                # pred_edge, pred_orient = self.model(input_ids, None, seq_mask, mat_mask)

                loss_dic = self.loss_fn(pred_edge, pred_orient, label_edge, label_orient)
                loss = loss_dic['loss']
                # clear grads
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                num_total += self.args.batch_size
                loss_total += loss.item()
                mean_loss = loss_total/num_total

                pbar.set_description(f'[train] epoch={epoch:>3d}, batch={i+1:>4d}/{len(self.train_dataloader):>4d}, best_score={self.max_metric:.3f}, train_loss={mean_loss:.4f}')
                # reset loss if too many steps
                if num_total >= self.args.logging_steps:
                    num_total, loss_total = 0, 0
        time_ed = time.time() - time_st
        print(f'[train] epoch={epoch:>3d}, train_loss={mean_loss:.4f}, time={time_ed:.4f}s')
   
    def save_pred_gt_results(self, raw_out_dir, names, seqs, pred_edge_np, pred_orient_np, gt_edge_np, gt_orient_np, extract_pred, extract_gt):
        for i in range(len(pred_edge_np)):
            name = names[i]
            np.savez(os.path.join(raw_out_dir, f'{name}_pred.npz'), edge=pred_edge_np[i], orient=pred_orient_np[i], )
            np.savez(os.path.join(raw_out_dir, f'{name}_gt.npz'), edge=gt_edge_np[i], orient=gt_orient_np[i], )
            for flag, data in [('pred', extract_pred), ('gt', extract_gt)]:
                with open(os.path.join(raw_out_dir, f'{name}_{flag}.ss'), 'w') as fp:
                    fp.write(f'name:{names[i]}\n')
                    fp.write(f'seq:{seqs[i]}\n')
                    fp.write(f'num interactions:{len(data[i])}\n')
                    for tup in data[i]:
                        fp.write(','.join([str(i) for i in tup])+'\n')
                        
    def eval(self, epoch, cur_k):
        self.model.eval()
        time_st = time.time()
        num_total, loss_total = 0, 0
        outputs_names, outputs_seqs = [], []
        pred_edges, pred_orients, gt_edges, gt_orients = [], [], [], []
        out_preds, out_gts = [], []
        raw_out_dir = os.path.join(self.args.output_dir, 'validate_results', f'epoch{epoch:03d}')
        os.makedirs(raw_out_dir, exist_ok=True)
        with tqdm(total=len(self.eval_dataset)) as pbar:
            for i, data in enumerate(self.eval_dataloader):
                input_ids = data["input_ids"].to(self.args.device)
                mat = data["mat"].to(self.args.device)
                mat_mask = data["mat_mask"].to(self.args.device)
                seq_mask = data["seq_mask"].to(self.args.device)
                LM_embed = data["LM_embed"].to(self.args.device) if 'LM_embed' in data else None
                label_edge = data["label_edge"].to(self.args.device)
                label_orient = data["label_orient"].to(self.args.device)
                with torch.no_grad():
                    # pred_edge, pred_orient = self.model(input_ids, LM_embed, seq_mask, mat_mask)
                    pred_edge, pred_orient = self.model(input_ids, None, seq_mask, mat_mask)
                    loss_dic = self.loss_fn(pred_edge, pred_orient, label_edge, label_orient)
                    loss = loss_dic['loss']
                    loss_total += loss.item()
                
                num_total += self.args.batch_size
                pred_edges += [b for b in pred_edge] # batch
                pred_orients += [b for b in pred_orient]
                gt_edges += [b for b in data["label_edge"]]
                gt_orients += [b for b in data["label_orient"]]
                pred_edge_np, pred_orient_np = pred_edge.detach().cpu().numpy(), pred_orient.detach().cpu().numpy()
                gt_edge_np, gt_orient_np = data["label_edge"].detach().cpu().numpy(), data["label_orient"].detach().cpu().numpy()
                extract_edge, extract_orient, extract_pred = extract_basepair_interaction(pred_edge_np, pred_orient_np, data['seq'])
                out_preds.append(extract_pred)
                extract_edge_gt, extract_orient_gt, extract_gt = extract_basepair_interaction_gt(gt_edge_np, gt_orient_np, data['seq'])
                out_gts.append(extract_gt)
                ## TODO: cal the loss, then update the best_val_loss 
                
                self.save_pred_gt_results(raw_out_dir, data['name'], data['seq'], pred_edge_np, pred_orient_np, gt_edge_np, gt_orient_np, extract_pred, extract_gt)
                outputs_names += data['name']
                outputs_seqs += data['seq']
                # if num_total >= self.args.logging_steps:
                #     num_total = 0
                pbar.set_description(f'[eval ] epoch={epoch:>3d}, batch={i+1:>4d}/{len(self.eval_dataloader):>4d}, best_score={self.max_metric:.3f}')

        xs = [pred_edges, pred_orients, gt_edges, gt_orients]
        metric_dic_list = [self.compute_metrics(*x) for x in zip(*xs)]
        df_data = []
        for name, seq, out_pred, out_gt, metric_dic in zip(outputs_names, outputs_seqs, out_preds, out_gts, metric_dic_list):
            df_data.append({
                            'name': name, 
                            'seq': seq,
                            'pred': out_pred,
                            'gt': out_gt,
                            **metric_dic,
                          })
        metrics_dataset = {}
        for metric_dic in metric_dic_list:
            for k, v in metric_dic.items():
                if k not in metrics_dataset:
                    metrics_dataset[k] = []
                metrics_dataset[k].append(v)
        metrics_dataset = {k: sum(v)/len(v) for k, v in metrics_dataset.items()}

        self.save_model(metrics_dataset, epoch, cur_k)
        metric_dir = os.path.join(self.args.output_dir, 'metrics')
        os.makedirs(metric_dir, exist_ok=True)
        pd.DataFrame(df_data).to_csv(os.path.join(metric_dir, f'test_{epoch}_{self.max_metric:.4f}.csv'), index=False)

        metric_str = ' '.join([f'{k}={v:.4f}' for k, v in metrics_dataset.items() if 'f1' in k])
        time_ed = time.time() - time_st
        print(f'[eval ] epoch={epoch:>3d}, {metric_str}, time={time_ed:.4f}s')
        epoch_metric_path = os.path.join(self.args.output_dir, 'epoch_metric.json')
        if os.path.exists(epoch_metric_path):
            with open(epoch_metric_path) as fp:
                all_epoch_metric = json.load(fp)
        else:
            all_epoch_metric = {}
        all_epoch_metric[epoch] = metrics_dataset
        with open(epoch_metric_path, 'w') as fp:
            json.dump(all_epoch_metric, fp)
     
            
    def test(self, cur_k):
        if self.test_dataset is None:
            return
        if self.best_ckpt_path and os.path.exists(self.best_ckpt_path):
            self.model.load_state_dict(torch.load(self.best_ckpt_path, map_location=self.args.device))
            print(f'Loading checkpoint: {self.best_ckpt_path}')
        else:
            print('Best checkpoint not found. Using the last epoch weights for test evalutation')  

        self.model.eval()
        time_st = time.time()
        num_total = 0
        outputs_names, outputs_seqs = [], []
        pred_edges, pred_orients, gt_edges, gt_orients = [], [], [], []
        out_preds, out_gts = [], []
        raw_out_dir = os.path.join(self.args.output_dir, 'test_results')
        os.makedirs(raw_out_dir, exist_ok=True)
        with tqdm(total=len(self.eval_dataset)) as pbar:
            for i, data in enumerate(self.test_dataloader):
                input_ids = data["input_ids"].to(self.args.device)
                mat = data["mat"].to(self.args.device)
                mat_mask = data["mat_mask"].to(self.args.device)
                seq_mask = data["seq_mask"].to(self.args.device)
                LM_embed = data["LM_embed"].to(self.args.device) if 'LM_embed' in data else None
                with torch.no_grad():
                    # pred_edge, pred_orient = self.model(input_ids, LM_embed=LM_embed, seq_mask=seq_mask, mat_mask=mat_mask)
                    pred_edge, pred_orient = self.model(input_ids, None, seq_mask=seq_mask, mat_mask=mat_mask)
                    
                num_total += self.args.batch_size
                pred_edges += [b for b in pred_edge] # batch
                pred_orients += [b for b in pred_orient]
                gt_edges += [b for b in data["label_edge"]]
                gt_orients += [b for b in data["label_orient"]]
                pred_edge_np, pred_orient_np = pred_edge.detach().cpu().numpy(), pred_orient.detach().cpu().numpy()
                gt_edge_np, gt_orient_np = data["label_edge"].detach().cpu().numpy(), data["label_orient"].detach().cpu().numpy()
                extract_edges, extract_orients, extract_pred = extract_basepair_interaction(pred_edge_np, pred_orient_np, data['seq'])
                out_preds.append(extract_pred)
                extract_edges_gt, extract_orients_gt, extract_gt = extract_basepair_interaction_gt(gt_edge_np, gt_orient_np, data['seq'])
                out_gts.append(extract_gt)
                self.save_pred_gt_results(raw_out_dir, data['name'], data['seq'], pred_edge_np, pred_orient_np, gt_edge_np, gt_orient_np, extract_pred, extract_gt)
                outputs_names += data['name']
                outputs_seqs += data['seq']
                if num_total >= self.args.logging_steps:
                    num_total = 0
                pbar.set_description(f'[test ], batch={i+1:>4d}/{len(self.test_dataloader):>4d}, best_score={self.max_metric:.3f}')

        xs = [pred_edges, pred_orients, gt_edges, gt_orients]
        metric_dic_list = [self.compute_metrics(*x) for x in zip(*xs)]

        # Aggregate metrics: build df_data and accumulate averages in one loop
        agg = {}           # {metric_name: running_sum}
        count = 0
        df_data = []
        for name, seq, out_pred, out_gt, metric_dic in zip(outputs_names, outputs_seqs, out_preds, out_gts, metric_dic_list):
            df_data.append({
                'name': name,
                'seq': seq,
                # Optional: store file paths instead of full arrays to avoid huge CSV
                'pred_file': os.path.join('out_npy', f'{name}_pred.npz'),
                'gt_file': os.path.join('out_npy', f'{name}_gt.npz'),
                **metric_dic,
            })
            for k, v in metric_dic.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            count += 1

        # Compute average metrics
        metrics_dataset = {k: (v / max(count, 1)) for k, v in agg.items()}

        # Save per-sample metrics to CSV
        metric_dir = os.path.join(self.args.output_dir, 'metrics', f'k_{cur_k}')
        os.makedirs(metric_dir, exist_ok=True)
        pd.DataFrame(df_data).to_csv(os.path.join(metric_dir, f'test_{self.max_metric:.4f}.csv'), index=False)

        # Display only f1-related metrics (or all if none found)
        metric_items = [(k, v) for k, v in metrics_dataset.items() if 'f1' in k]
        if not metric_items:
            metric_items = sorted(metrics_dataset.items())
        metric_str = ' '.join([f'{k}={v:.4f}' for k, v in metric_items])

        time_ed = time.time() - time_st
        print(f'[test ], {metric_str}, time={time_ed:.4f}s')

        # Save test results to JSON, append instead of overwrite
        test_metric_path = os.path.join(self.args.output_dir, f'k_{cur_k}_test_metric.json')
        record = {
            "max_metric": getattr(self, "max_metric", None),
            "avg_metrics": metrics_dataset,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        if os.path.exists(test_metric_path):
            try:
                with open(test_metric_path) as fp:
                    all_test_metric = json.load(fp)
                if not isinstance(all_test_metric, list):
                    all_test_metric = [all_test_metric]
            except Exception:
                all_test_metric = []
        else:
            all_test_metric = []

        # Append the current record
        all_test_metric.append(record)
        with open(test_metric_path, 'w') as fp:
            json.dump(all_test_metric, fp, indent=2)
            
        return metrics_dataset

    def run(self, phase='train', cur_k=None):
        if phase == 'k_fold_train': # first phase: choose the best hyperparameters and model architecture
            for epoch in range(self.args.num_train_epochs):
                self.train(epoch, cur_k)
                self.eval(epoch, cur_k)
                
            metrics_dataset = self.test(cur_k)
            return metrics_dataset



class NCfoldTrainer(BaseTrainer):
    def train(self, epoch, cur_k=None):
        self.model.train()
        time_st = time.time()
        num_total, loss_total = 0, 0
        mean_loss = float('inf')

        with tqdm(total=len(self.train_dataset)) as pbar:
            for i, data in enumerate(self.train_dataloader):
                # move tensor to device
                input_ids = data["input_ids"].to(self.args.device)
                mat = data["mat"].to(self.args.device)
                mat_mask = data["mat_mask"].to(self.args.device)
                seq_mask = data["seq_mask"].to(self.args.device)
                label_edge = data["label_edge"].to(self.args.device)
                label_orient = data["label_orient"].to(self.args.device)
                LM_embed = data["LM_embed"].to(self.args.device) if 'LM_embed' in data else None
                pred_edge, pred_orient = self.model(input_ids, mat, LM_embed, seq_mask, mat_mask)
                loss_dic = self.loss_fn(pred_edge, pred_orient, label_edge, label_orient)
                loss = loss_dic['loss']
                # clear grads
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                num_total += self.args.batch_size
                loss_total += loss.item()
                mean_loss = loss_total/num_total

                pbar.set_description(f'[train] epoch={epoch:>3d}, batch={i+1:>4d}/{len(self.train_dataloader):>4d}, best_score={self.max_metric:.3f}, train_loss={mean_loss:.4f}')
                # reset loss if too many steps
                if num_total >= self.args.logging_steps:
                    num_total, loss_total = 0, 0
        time_ed = time.time() - time_st
        print(f'[train] epoch={epoch:>3d}, train_loss={mean_loss:.4f}, time={time_ed:.4f}s')

    def save_pred_gt_results(self, raw_out_dir, names, seqs, pred_edge_np, pred_orient_np, gt_edge_np, gt_orient_np, extract_pred, extract_gt):
        for i in range(len(pred_edge_np)):
            name = names[i]
            np.savez(os.path.join(raw_out_dir, f'{name}_pred.npz'), edge=pred_edge_np[i], orient=pred_orient_np[i], )
            np.savez(os.path.join(raw_out_dir, f'{name}_gt.npz'), edge=gt_edge_np[i], orient=gt_orient_np[i], )
            for flag, data in [('pred', extract_pred), ('gt', extract_gt)]:
                with open(os.path.join(raw_out_dir, f'{name}_{flag}.ss'), 'w') as fp:
                    fp.write(f'name:{names[i]}\n')
                    fp.write(f'seq:{seqs[i]}\n')
                    fp.write(f'num interactions:{len(data[i])}\n')
                    for tup in data[i]:
                        fp.write(','.join([str(i) for i in tup])+'\n')
                                          
    def eval(self, epoch, cur_k=None):
        self.model.eval()
        time_st = time.time()
        num_total, loss_total = 0, 0
        outputs_names, outputs_seqs = [], []
        pred_edges, pred_orients, gt_edges, gt_orients = [], [], [], []
        out_preds, out_gts = [], []
        raw_out_dir = os.path.join(self.args.output_dir, 'validate_results', f'epoch{epoch:03d}')
        os.makedirs(raw_out_dir, exist_ok=True)
        with tqdm(total=len(self.eval_dataset)) as pbar:
            for i, data in enumerate(self.eval_dataloader):
                input_ids = data["input_ids"].to(self.args.device)
                mat = data["mat"].to(self.args.device)
                mat_mask = data["mat_mask"].to(self.args.device)
                seq_mask = data["seq_mask"].to(self.args.device)
                LM_embed = data["LM_embed"].to(self.args.device) if 'LM_embed' in data else None
                label_edge = data["label_edge"].to(self.args.device)
                label_orient = data["label_orient"].to(self.args.device)
                
                with torch.no_grad():
                    pred_edge, pred_orient = self.model(input_ids, mat, LM_embed, seq_mask, mat_mask)
                    loss_dic = self.loss_fn(pred_edge, pred_orient, label_edge, label_orient)
                    loss = loss_dic['loss']
                    loss_total += loss.item()
                num_total += self.args.batch_size
                pred_edges += [b for b in pred_edge] # batch
                pred_orients += [b for b in pred_orient]
                gt_edges += [b for b in data["label_edge"]]
                gt_orients += [b for b in data["label_orient"]]
                pred_edge_np, pred_orient_np = pred_edge.detach().cpu().numpy(), pred_orient.detach().cpu().numpy()
                gt_edge_np, gt_orient_np = data["label_edge"].detach().cpu().numpy(), data["label_orient"].detach().cpu().numpy()
                extract_edge, extract_orient, extract_pred = extract_basepair_interaction(pred_edge_np, pred_orient_np, data['seq'])
                out_preds.append(extract_pred)
                extract_edge_gt, extract_orient_gt, extract_gt = extract_basepair_interaction_gt(gt_edge_np, gt_orient_np, data['seq'])
                out_gts.append(extract_gt)
                self.save_pred_gt_results(raw_out_dir, data['name'], data['seq'], pred_edge_np, pred_orient_np, gt_edge_np, gt_orient_np, extract_pred, extract_gt)
                outputs_names += data['name']
                outputs_seqs += data['seq']
                # if num_total >= self.args.logging_steps:
                #     num_total = 0
                pbar.set_description(f'[eval ] epoch={epoch:>3d}, batch={i+1:>4d}/{len(self.eval_dataloader):>4d}, best_score={self.max_metric:.3f}')

        xs = [pred_edges, pred_orients, gt_edges, gt_orients]
        metric_dic_list = [self.compute_metrics(*x) for x in zip(*xs)]
        df_data = []
        for name, seq, out_pred, out_gt, metric_dic in zip(outputs_names, outputs_seqs, out_preds, out_gts, metric_dic_list):
            df_data.append({
                            'name': name, 
                            'seq': seq,
                            'pred': out_pred,
                            'gt': out_gt,
                            **metric_dic,
                          })
        metrics_dataset = {}
        for metric_dic in metric_dic_list:
            for k, v in metric_dic.items():
                if k not in metrics_dataset:
                    metrics_dataset[k] = []
                metrics_dataset[k].append(v)
        metrics_dataset = {k: sum(v)/len(v) for k, v in metrics_dataset.items()}

        self.save_model(metrics_dataset, epoch, cur_k)
        metric_dir = os.path.join(self.args.output_dir, 'metrics')
        os.makedirs(metric_dir, exist_ok=True)
        pd.DataFrame(df_data).to_csv(os.path.join(metric_dir, f'test_{epoch}_{self.max_metric:.4f}.csv'), index=False)

        metric_str = ' '.join([f'{k}={v:.4f}' for k, v in metrics_dataset.items() if 'f1' in k])
        time_ed = time.time() - time_st
        print(f'[eval ] epoch={epoch:>3d}, {metric_str}, time={time_ed:.4f}s')
        epoch_metric_path = os.path.join(self.args.output_dir, 'epoch_metric.json')
        if os.path.exists(epoch_metric_path):
            with open(epoch_metric_path) as fp:
                all_epoch_metric = json.load(fp)
        else:
            all_epoch_metric = {}
        all_epoch_metric[epoch] = metrics_dataset
        with open(epoch_metric_path, 'w') as fp:
            json.dump(all_epoch_metric, fp)

    def test(self, cur_k=None):
        if self.test_dataset is None:
            return
        if self.best_ckpt_path and os.path.exists(self.best_ckpt_path):
            self.model.load_state_dict(torch.load(self.best_ckpt_path, map_location=self.args.device))
            print(f'Loading checkpoint: {self.best_ckpt_path}')
        else:
            print('Best checkpoint not found. Using the last epoch weights for test evalutation')  

        self.model.eval()
        time_st = time.time()
        num_total = 0
        outputs_names, outputs_seqs = [], []
        pred_edges, pred_orients, gt_edges, gt_orients = [], [], [], []
        out_preds, out_gts = [], []
        raw_out_dir = os.path.join(self.args.output_dir, 'test_results')
        os.makedirs(raw_out_dir, exist_ok=True)
        with tqdm(total=len(self.eval_dataset)) as pbar:
            for i, data in enumerate(self.test_dataloader):
                input_ids = data["input_ids"].to(self.args.device)
                mat = data["mat"].to(self.args.device)
                mat_mask = data["mat_mask"].to(self.args.device)
                seq_mask = data["seq_mask"].to(self.args.device)
                LM_embed = data["LM_embed"].to(self.args.device) if 'LM_embed' in data else None
                with torch.no_grad():
                    pred_edge, pred_orient = self.model(input_ids, mat, LM_embed=LM_embed, seq_mask=seq_mask, mat_mask=mat_mask)
                num_total += self.args.batch_size
                pred_edges += [b for b in pred_edge] # batch
                pred_orients += [b for b in pred_orient]
                gt_edges += [b for b in data["label_edge"]]
                gt_orients += [b for b in data["label_orient"]]
                pred_edge_np, pred_orient_np = pred_edge.detach().cpu().numpy(), pred_orient.detach().cpu().numpy()
                gt_edge_np, gt_orient_np = data["label_edge"].detach().cpu().numpy(), data["label_orient"].detach().cpu().numpy()
                extract_edges, extract_orients, extract_pred = extract_basepair_interaction(pred_edge_np, pred_orient_np, data['seq'])
                out_preds.append(extract_pred)
                extract_edges_gt, extract_orients_gt, extract_gt = extract_basepair_interaction_gt(gt_edge_np, gt_orient_np, data['seq'])
                out_gts.append(extract_gt)
                self.save_pred_gt_results(raw_out_dir, data['name'], data['seq'], pred_edge_np, pred_orient_np, gt_edge_np, gt_orient_np, extract_pred, extract_gt)
                outputs_names += data['name']
                outputs_seqs += data['seq']
                if num_total >= self.args.logging_steps:
                    num_total = 0
                pbar.set_description(f'[test ], batch={i+1:>4d}/{len(self.test_dataloader):>4d}, best_score={self.max_metric:.3f}')

        xs = [pred_edges, pred_orients, gt_edges, gt_orients]
        
        
        metric_dic_list = [self.compute_metrics(*x) for x in zip(*xs)]

        # Aggregate metrics: build df_data and accumulate averages in one loop
        agg = {}           # {metric_name: running_sum}
        count = 0
        df_data = []
        for name, seq, out_pred, out_gt, metric_dic in zip(outputs_names, outputs_seqs, out_preds, out_gts, metric_dic_list):
            df_data.append({
                'name': name,
                'seq': seq,
                # Optional: store file paths instead of full arrays to avoid huge CSV
                'pred_file': os.path.join('out_npy', f'{name}_pred.npz'),
                'gt_file': os.path.join('out_npy', f'{name}_gt.npz'),
                **metric_dic,
            })
            for k, v in metric_dic.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            count += 1

        # Compute average metrics
        metrics_dataset = {k: (v / max(count, 1)) for k, v in agg.items()}

        # Save per-sample metrics to CSV
        metric_dir = os.path.join(self.args.output_dir, 'metrics', f'k_{cur_k}')
        os.makedirs(metric_dir, exist_ok=True)
        pd.DataFrame(df_data).to_csv(os.path.join(metric_dir, f'test_{self.max_metric:.4f}.csv'), index=False)

        # Display only f1-related metrics (or all if none found)
        metric_items = [(k, v) for k, v in metrics_dataset.items() if 'f1' in k]
        if not metric_items:
            metric_items = sorted(metrics_dataset.items())
        metric_str = ' '.join([f'{k}={v:.4f}' for k, v in metric_items])

        time_ed = time.time() - time_st
        print(f'[test ], {metric_str}, time={time_ed:.4f}s')

        # Save test results to JSON, append instead of overwrite
        test_metric_path = os.path.join(self.args.output_dir, f'k_{cur_k}_test_metric.json')
        record = {
            "max_metric": getattr(self, "max_metric", None),
            "avg_metrics": metrics_dataset,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        if os.path.exists(test_metric_path):
            try:
                with open(test_metric_path) as fp:
                    all_test_metric = json.load(fp)
                if not isinstance(all_test_metric, list):
                    all_test_metric = [all_test_metric]
            except Exception:
                all_test_metric = []
        else:
            all_test_metric = []

        # Append the current record
        all_test_metric.append(record)
        with open(test_metric_path, 'w') as fp:
            json.dump(all_test_metric, fp, indent=2)
            
        return metrics_dataset
    

    def run(self, phase='k_fold_train', cur_k=None):
        if phase == 'k_fold_train':
            for epoch in range(self.args.num_train_epochs):
                self.train(epoch, cur_k)
                self.eval(epoch, cur_k)
                
        metrics_dataset = self.test(cur_k)
        return metrics_dataset

# machine learning  
class EdgeOrientSklearnTrainer(BaseTrainer):
    """
    Two-head classical model trainer:
      - Edge (node-level):    4 classes, outputs [B, L, 4]
      - Orient (pair-level):  3 classes, outputs [B, 3, L, L]
    """

    def __init__(
        self,
        args,
        model_name: str,
        train_dataset=None,
        eval_dataset=None,
        test_dataset=None,
        data_collator=None,
        compute_metrics=None,
        best_ckpt_path=None,
        sample_pairs_ratio: float = 0.5,
    ):
        # Do not rely on torch model; we reuse BaseTrainer infra for loaders only
        self.args = args
        self.model = None
        self.pretrained_model = None
        self.indicator = None
        self.ensemble = None
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.data_collator = data_collator
        self.loss_fn = None
        self.optimizer = None
        self.compute_metrics = compute_metrics
        self.prime_metric = 'nc_score'
        self.visual_writer = None
        self.max_metric = 0.
        self.best_ckpt_path = best_ckpt_path
        self.model_name = model_name

        self._prepare_dataloaders()

        # Build classifiers for both heads
        self.edge_clf   = make_classifier(model_name, num_classes=4)  # node-level
        self.orient_clf = make_classifier(model_name, num_classes=3)  # pair-level

        # Optionally load checkpoint (pickled classifiers)
        if self.best_ckpt_path and os.path.exists(self.best_ckpt_path):
            with open(self.best_ckpt_path, "rb") as f:
                payload = pickle.load(f)
            self.edge_clf = payload["edge_clf"]
            self.orient_clf = payload["orient_clf"]
            print(f'Loading {model_name} checkpoint: {self.best_ckpt_path}')

        self.sample_pairs_ratio = sample_pairs_ratio

    # ---------- save helper (kept compatible with your saving conventions) ----------
    def save_pred_gt_results(self, raw_out_dir, names, seqs,
                             pred_edge_np, pred_orient_np, gt_edge_np, gt_orient_np,
                             extract_pred, extract_gt):
        for i in range(len(pred_edge_np)):
            name = names[i]
            np.savez(os.path.join(raw_out_dir, f'{name}_pred.npz'),
                     edge=pred_edge_np[i], orient=pred_orient_np[i])
            np.savez(os.path.join(raw_out_dir, f'{name}_gt.npz'),
                     edge=gt_edge_np[i], orient=gt_orient_np[i])
            for flag, data in [('pred', extract_pred), ('gt', extract_gt)]:
                with open(os.path.join(raw_out_dir, f'{name}_{flag}.ss'), 'w') as fp:
                    fp.write(f'name:{names[i]}\n')
                    fp.write(f'seq:{seqs[i]}\n')
                    fp.write(f'num interactions:{len(data[i])}\n')
                    for tup in data[i]:
                        fp.write(','.join([str(i) for i in tup])+'\n')

    # ------------------------------------- train -------------------------------------
    def train(self, epoch, cur_k):
        rng = np.random.RandomState(0)
        Xn_list, yn_list = [], []  # node (edge head)
        Xp_list, yp_list = [], []  # pair (orient head)

        with tqdm(total=len(self.train_dataset)) as pbar:
            for i, batch in enumerate(self.train_dataloader):
                # fetch numpy views
                input_ids = batch["input_ids"].cpu().numpy()                 # [B, L]
                seq_mask  = batch["seq_mask"].cpu().numpy().astype(bool)     # [B, L]
            
                seq_feat = input_ids[..., None].astype(np.float32)   # -> [B, L, 1]
                mat = np.einsum('blh,bmh->bhlm', seq_feat, seq_feat) # -> [B, 1, L, L]
                # edge labels: [B, L]
                y_edge = batch["label_edge"].cpu().numpy()                   # node labels (0..3 or -1)

                # orient labels: [B, 1, L, L]  → squeeze channel
                y_orient = batch["label_orient"].cpu().numpy()
                                
                B, L = input_ids.shape
                for b in range(B):
                    Lb = int(seq_mask[b].sum()) if seq_mask is not None else L
                    if Lb <= 0:
                        continue
                    seq_b = input_ids[b, :Lb].astype(np.int32)
                    M = mat[b, 0, :Lb, :Lb].astype(np.float32)

                    # --- node rows (edge head) ---
                    Xn_b = build_node_features_single(seq_b, M)              # [Lb, 8]
                    yn_b = y_edge[b, :Lb].astype(np.int32)                   # [Lb]
                    m = (yn_b >= 0)
                    if m.any():
                        Xn_list.append(Xn_b[m]); yn_list.append(yn_b[m])

                    # --- pair rows (orient head), upper-tri with subsampling ---
                    Xp_b, yp_b = build_pair_features_upper(
                        seq_b, M, y_orient[b, :Lb, :Lb], sample_ratio=self.sample_pairs_ratio, rng=rng
                    )
                    if Xp_b.size:
                        Xp_list.append(Xp_b); yp_list.append(yp_b)

                pbar.set_description(f'[train-{self.model_name}] epoch={epoch:>3d}, node_chunks={len(Xn_list)}, pair_chunks={len(Xp_list)}')

        # concat
        Xn = np.concatenate(Xn_list, axis=0) if Xn_list else np.zeros((0, 8), dtype=np.float32)
        yn = np.concatenate(yn_list, axis=0) if yn_list else np.zeros((0,), dtype=np.int32)
        Xp = np.concatenate(Xp_list, axis=0) if Xp_list else np.zeros((0,16), dtype=np.float32)
        yp = np.concatenate(yp_list, axis=0) if yp_list else np.zeros((0,), dtype=np.int32)

        if Xn.shape[0] == 0 or Xp.shape[0] == 0:
            raise ValueError("No valid training rows for node or pair head (check masks/labels).")

        # fit
        self.edge_clf.fit(Xn, yn)     # 4 classes
        self.orient_clf.fit(Xp, yp)   # 3 classes

        print(f'[train-{self.model_name}] fitted: node N={Xn.shape[0]}, pair N={Xp.shape[0]}')

    # -------------------------------- inference helper --------------------------------
    def _predict_batch_probs(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          edge_pred   : [B, L, 4]     (node classes)
          orient_pred : [B, 3, L, L]  (pair classes)
        """
        input_ids = batch["input_ids"].cpu().numpy()                 # [B, L]
        mat       = batch["mat"].cpu().numpy()                       # [B, 1, L, L]
        seq_mask  = batch["seq_mask"].cpu().numpy().astype(bool)     # [B, L]
        B, L = input_ids.shape

        edge_probs_B   = np.zeros((B, L, 4), dtype=np.float32)       # [B, L, 4]
        orient_probs_B = np.zeros((B, L, L, 3), dtype=np.float32)    # [B, L, L, 3]

        for b in range(B):
            Lb = int(seq_mask[b].sum()) if seq_mask is not None else L
            if Lb <= 0:
                continue
            seq_b = input_ids[b, :Lb].astype(np.int32)
            M = mat[b, 0, :Lb, :Lb].astype(np.float32)

            # node head
            Xn_b = build_node_features_single(seq_b, M)              # [Lb, 8]
            pn_raw = self.edge_clf.predict_proba(Xn_b)               # [Lb, C?]
            pn = align_proba_fixed_classes(pn_raw, self.edge_clf.classes_, 4)  # [Lb,4]
            edge_probs_B[b, :Lb, :] = pn

            # pair head
            Xp_b = build_pair_features_single(seq_b, M)              # [Lb*Lb, 16]
            po_raw = self.orient_clf.predict_proba(Xp_b)             # [Lb*Lb, C?]
            po = align_proba_fixed_classes(po_raw, self.orient_clf.classes_, 3).reshape(Lb, Lb, 3)  # [Lb,Lb,3]
            orient_probs_B[b] = pad_square_probs(po, L, 3)           # [L,L,3]

        # convert to tensors with requested shapes
        edge_pred   = torch.from_numpy(edge_probs_B).to(self.args.device)                              # [B,L,4]
        orient_pred = torch.from_numpy(np.transpose(orient_probs_B, (0, 3, 1, 2))).to(self.args.device) # [B,3,L,L]
        return edge_pred, orient_pred

    # ------------------------------------ eval / test / run ------------------------------------
    def save_model(self, metrics_dataset, epoch, cur_k):
        checkpoint_dir = os.path.join(self.args.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        score = metrics_dataset.get(self.prime_metric, 0.0)
        if score > self.max_metric:
            self.max_metric = score
            payload = {"edge_clf": self.edge_clf, "orient_clf": self.orient_clf}
            save_path = os.path.join(checkpoint_dir, f"{self.model_name}_cur_k{cur_k}_epoch{epoch}_{self.max_metric:.3f}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(payload, f)
            best_path = os.path.join(checkpoint_dir, f"best_cur_k{cur_k}_{self.model_name}.pkl")
            with open(best_path, "wb") as f:
                pickle.dump(payload, f)
            self.best_ckpt_path = best_path
            print(f"Save {self.model_name} model for cur_k={cur_k}:", save_path, "and", best_path, f'score: {score}')


    def eval(self, epoch, cur_k):
        # identical structure to your existing eval, but using our inference helper
        time_st = time.time()
        outputs_names, outputs_seqs = [], []
        pred_edges, pred_orients, gt_edges, gt_orients = [], [], [], []
        out_preds, out_gts = [], []
        raw_out_dir = os.path.join(self.args.output_dir, 'validate_results', f'epoch{epoch:03d}')
        os.makedirs(raw_out_dir, exist_ok=True)

        with tqdm(total=len(self.eval_dataset)) as pbar:
            for i, batch in enumerate(self.eval_dataloader):
                with torch.no_grad():
                    pred_edge, pred_orient = self._predict_batch_probs(batch)

                pred_edges   += [b for b in pred_edge]          # b: [L,4]
                pred_orients += [b for b in pred_orient]        # b: [3,L,L]
                gt_edges     += [b for b in batch["label_edge"]]         # expected [L]
                gt_orients   += [b for b in batch["label_orient"]]       # expected [1,L,L] or squeeze later

                # numpy copies for your extract/save utils
                pred_edge_np, pred_orient_np = pred_edge.cpu().numpy(), pred_orient.cpu().numpy()
                gt_edge_np,   gt_orient_np   = batch["label_edge"].cpu().numpy(), batch["label_orient"].cpu().numpy()

                # If your extract_basepair_interaction expects pairwise edge logits, skip or adapt it.
                # Here we pass node-edge and pair-orient; adjust your extractor accordingly if needed.
                extract_edge, extract_orient, extract_pred = extract_basepair_interaction(pred_edge_np, pred_orient_np, batch['seq'])
                extract_edge_gt, extract_orient_gt, extract_gt = extract_basepair_interaction_gt(gt_edge_np, gt_orient_np, batch['seq'])
                out_preds.append(extract_pred); out_gts.append(extract_gt)

                self.save_pred_gt_results(raw_out_dir, batch['name'], batch['seq'],
                                          pred_edge_np, pred_orient_np, gt_edge_np, gt_orient_np,
                                          extract_pred, extract_gt)
                outputs_names += batch['name']; outputs_seqs += batch['seq']
                pbar.set_description(f'[eval-{self.model_name}] epoch={epoch:>3d}, batch={i+1:>4d}/{len(self.eval_dataloader):>4d}, best={self.max_metric:.3f}')

        xs = [pred_edges, pred_orients, gt_edges, gt_orients]
        metric_dic_list = [self.compute_metrics(*x) for x in zip(*xs)]

        df_data, agg = [], {}
        for name, seq, out_pred, out_gt, metric_dic in zip(outputs_names, outputs_seqs, out_preds, out_gts, metric_dic_list):
            df_data.append({'name': name, 'seq': seq, 'pred': out_pred, 'gt': out_gt, **metric_dic})
            for k, v in metric_dic.items():
                agg[k] = agg.get(k, 0.0) + float(v)
        metrics_dataset = {k: agg[k] / max(len(metric_dic_list), 1) for k in agg}

        self.save_model(metrics_dataset, epoch, cur_k)

        metric_dir = os.path.join(self.args.output_dir, 'metrics')
        os.makedirs(metric_dir, exist_ok=True)
        pd.DataFrame(df_data).to_csv(os.path.join(metric_dir, f'eval_{cur_k}_{epoch}_{self.max_metric:.4f}.csv'), index=False)

        time_ed = time.time() - time_st
        metric_str = ' '.join([f'{k}={v:.4f}' for k, v in metrics_dataset.items() if 'f1' in k])
        print(f'[eval-{self.model_name}] epoch={epoch:>3d}, {metric_str}, time={time_ed:.4f}s')

        return metrics_dataset
    

    def test(self, cur_k):
        if self.test_dataset is None:
            return
        if self.best_ckpt_path and os.path.exists(self.best_ckpt_path):
            with open(self.best_ckpt_path, "rb") as f:
                payload = pickle.load(f)
            self.edge_clf = payload["edge_clf"]
            self.orient_clf = payload["orient_clf"]
            print(f'Loading {self.model_name} checkpoint: {self.best_ckpt_path}')
        else:
            print(f'Best {self.model_name} checkpoint not found. Using current classifiers.')

        time_st = time.time()
        outputs_names, outputs_seqs = [], []
        pred_edges, pred_orients, gt_edges, gt_orients = [], [], [], []
        out_preds, out_gts = [], []
        raw_out_dir = os.path.join(self.args.output_dir, 'test_results')
        os.makedirs(raw_out_dir, exist_ok=True)

        with tqdm(total=len(self.test_dataset)) as pbar:
            for i, batch in enumerate(self.test_dataloader):
                with torch.no_grad():
                    pred_edge, pred_orient = self._predict_batch_probs(batch)

                pred_edges   += [b for b in pred_edge]
                pred_orients += [b for b in pred_orient]
                gt_edges     += [b for b in batch["label_edge"]]
                gt_orients   += [b for b in batch["label_orient"]]

                pred_edge_np, pred_orient_np = pred_edge.cpu().numpy(), pred_orient.cpu().numpy()
                gt_edge_np,   gt_orient_np   = batch["label_edge"].cpu().numpy(), batch["label_orient"].cpu().numpy()

                extract_edges, extract_orients, extract_pred = extract_basepair_interaction(pred_edge_np, pred_orient_np, batch['seq'])
                extract_edges_gt, extract_orients_gt, extract_gt = extract_basepair_interaction_gt(gt_edge_np, gt_orient_np, batch['seq'])
                out_preds.append(extract_pred); out_gts.append(extract_gt)

                self.save_pred_gt_results(raw_out_dir, batch['name'], batch['seq'],
                                          pred_edge_np, pred_orient_np, gt_edge_np, gt_orient_np,
                                          extract_pred, extract_gt)
                outputs_names += batch['name']; outputs_seqs += batch['seq']
                pbar.set_description(f'[test-{self.model_name}], batch={i+1:>4d}/{len(self.test_dataloader):>4d}, best={self.max_metric:.3f}')

        xs = [pred_edges, pred_orients, gt_edges, gt_orients]
        metric_dic_list = [self.compute_metrics(*x) for x in zip(*xs)]

        agg, df_data = {}, []
        for name, seq, out_pred, out_gt, metric_dic in zip(outputs_names, outputs_seqs, out_preds, out_gts, metric_dic_list):
            df_data.append({'name': name, 'seq': seq,
                            'pred_file': os.path.join('out_npy', f'{name}_pred.npz'),
                            'gt_file': os.path.join('out_npy', f'{name}_gt.npz'),
                            **metric_dic})
            for k, v in metric_dic.items():
                agg[k] = agg.get(k, 0.0) + float(v)
        metrics_dataset = {k: (v / max(len(metric_dic_list), 1)) for k, v in agg.items()}

        metric_dir = os.path.join(self.args.output_dir, 'metrics', f'k_{cur_k}')
        os.makedirs(metric_dir, exist_ok=True)
        pd.DataFrame(df_data).to_csv(os.path.join(metric_dir, f'test_{self.max_metric:.4f}.csv'), index=False)

        metric_items = [(k, v) for k, v in metrics_dataset.items() if 'f1' in k] or sorted(metrics_dataset.items())
        metric_str = ' '.join([f'{k}={v:.4f}' for k, v in metric_items])

        time_ed = time.time() - time_st
        print(f'[test-{self.model_name}], {metric_str}, time={time_ed:.4f}s')

        test_metric_path = os.path.join(self.args.output_dir, f'k_{cur_k}_test_metric.json')
        record = {"max_metric": getattr(self, "max_metric", None),
                  "avg_metrics": metrics_dataset,
                  "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
        if os.path.exists(test_metric_path):
            try:
                with open(test_metric_path) as fp:
                    all_test_metric = json.load(fp)
                if not isinstance(all_test_metric, list):
                    all_test_metric = [all_test_metric]
            except Exception:
                all_test_metric = []
        else:
            all_test_metric = []
        all_test_metric.append(record)
        with open(test_metric_path, 'w') as fp:
            json.dump(all_test_metric, fp, indent=2)
            
        return metrics_dataset

    def run(self, phase='train', cur_k=None):
        if phase == 'k_fold_train':
            for epoch in range(self.args.num_train_epochs):
                self.train(epoch, cur_k)
                self.eval(epoch, cur_k)
            metrics_dataset = self.test(cur_k) # return the test metrics for this fold
            return metrics_dataset



# ============================ Argument parsing ============================

def get_args():
    parser = argparse.ArgumentParser('Implementation of RNA sequence classification.')
    parser.add_argument('--kfold_phase', type=str, default="k_fold_train", choices=['k_fold_train', 'test'])
    
    # data
    parser.add_argument('--dataset_dir', type=str, default="data")
    parser.add_argument('--dataset', type=str, default="PDB_NC", choices=DATASETS)
    parser.add_argument('--filter_fasta', type=str)
    parser.add_argument('--output_dir', type=str, default='.runs/tmp')
    parser.add_argument('--include_canonical', action='store_true')
    parser.add_argument('--use_RFdiff_data', action='store_true')
    parser.add_argument('--LM_list', nargs='*', default=[], choices=LMs)
    parser.add_argument('--LM_checkpoint_dir', type=str, default='LM_checkpoint', help='LM checkpoint_dir, each LM is placed in a subdir of same name.')

    ## k-fold data 
    parser.add_argument('--seed', type=int, default=42, help='random seed for data spliting.')
    parser.add_argument('--k_fold', type=int, default=4, help='k for k-fold cross validation. 0 means no k-fold.')
    parser.add_argument('--raw_trainval_file', type=str, default='src/NCfold/data/NCpair_trainval_data_raw.json')
    parser.add_argument('--raw_test_file', type=str, default='src/NCfold/data/NCpair_test_data_raw.json')
    parser.add_argument('--fold_out_dir', type=str, default='data/folds')
    
    ## Fuse LM embeddings
    parser.add_argument('--top_k', type=int, default=0)

    # model args
    # MM = ['RF', 'Gradient Boosting', 'XGBoost', 'SGD', 'Logistic Regression', 'MLP', 'SVM', 'KNN']
    # SLMs = ['BPfold', 'RNAFold', 'MxFold2', 'ContraFold']
    # MODELS = ['AttnMatFusion_net']
    # LMs = ['structRFM', 'rnaernie', 'rnafm', 'splicebert', 'utrlm-te_el', 'aido.rna-650m', 'rinalmo-micro']
    parser.add_argument('--model_name', type=str, default='structRFM', choices=MODELS)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_blocks', type=int, default=6)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--use_BPM', type=bool, default=False)
    parser.add_argument('--replace_T', type=bool, default=True)
    parser.add_argument('--replace_U', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--batch_size', type=int, default=4, help='The number of samples used per step & per device.')
    parser.add_argument('--num_train_epochs', type=int, default=60, help='The number of epoch for training.')
    parser.add_argument('--logging_steps', type=int)
    parser.add_argument('--best_ckpt_path', type=str)

    # loss weight
    parser.add_argument('--weight_edgeW', type=float, default=5)
    parser.add_argument('--weight_edgeH', type=float, default=20)
    parser.add_argument('--weight_edgeS', type=float, default=20)
    parser.add_argument('--weight_trans', type=float, default=20)
    parser.add_argument('--weight_cis', type=float, default=20)
    
    parser.add_argument('--label_smoothing', type=float, default=0.05, help='label smoothing for cross entropy loss.')
    parser.add_argument('--use_uncertainty_weighting', type=bool, default=False, help='use uncertainty weighting for multi-task loss.')
    args = parser.parse_args()
    return args


def train_and_test():
    print(f'Begin time: {datetime.now()}')
    args = get_args()
    
    # set seed 
    set_seed(42)

    assert args.replace_T ^ args.replace_U, "Only replace T or U."
    if args.logging_steps is None:
        args.logging_steps = 1500//args.batch_size
    print(args)

    if args.model_name in LMs:
        model = RFM_net(
                        max_seq_len=args.max_seq_len, 
                        out_dim=4, 
                        out_channels=3,
                        LM_embed_chan= args.top_k if args.LM_list else None,
                                 ) 
    elif args.model_name in MM:
        # Classical models (RF/GBDT/XGBoost/SGD/LogReg/SVM/KNN):
        # nothing to build here; trainers will create sklearn/xgboost classifiers inside.
        pass 
    elif args.model_name in MODELS:
        model = AttnMatFusion_net(
            max_seq_len=args.max_seq_len, 
            out_dim=4, 
            out_channels=3,
            dim=args.hidden_dim, 
            depth=args.num_blocks, 
            positional_embedding='rope', 
            use_BPM=args.use_BPM,
            LM_embed_chan= args.top_k if args.LM_list else None,
            )
    else:
        raise ValueError("Unknown model name: {}".format(args.model_name))
    
    args.output_dir = args.output_dir.rstrip('/') + 'MLP' + '_blocks'+str(args.num_blocks) + '_bz'+str(args.batch_size) + '_topk'+str(args.top_k) + 'only_iscore' 
    os.makedirs(args.output_dir, exist_ok=True)
    if args.model_name in LMs or args.model_name in MODELS:
        model.to(args.device)
        count_para(model)

        _loss_fn = NCfoldLoss(
        edge_weight=1.0, 
        orient_weight=1.0,
        edge_weights=torch.tensor([1.0, args.weight_edgeW, args.weight_edgeH, args.weight_edgeS]),
        orient_weights=torch.tensor([1.0, args.weight_trans, args.weight_cis]),
        label_smoothing=args.label_smoothing,          # new 
        use_uncertainty_weighting=args.use_uncertainty_weighting # new
        ).to(args.device)
    
        optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)
        
        
    dataset_trainval = Kfold_RNAdata(args.dataset_dir, args.max_seq_len, filter_fasta=args.filter_fasta, phase='k_fold_train', include_canonical=args.include_canonical, use_RFdiff_data=args.use_RFdiff_data)
    dataset_test = Kfold_RNAdata(args.dataset_dir, args.max_seq_len, filter_fasta=args.filter_fasta, phase='test', include_canonical=args.include_canonical)
    _collate_fn = NCfoldCollator(max_seq_len=None, replace_T=args.replace_T, replace_U=args.replace_U, LM_list=args.LM_list, LM_checkpoint_dir=args.LM_checkpoint_dir, top_k=args.top_k)
    
    # k_fold 
    kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
    if args.k_fold > 1:
        os.makedirs(args.fold_out_dir, exist_ok=True)
    indices = np.arange(len(dataset_trainval))
    
    all_test_metric_dic = {}

    for k, (train_idx, val_idx) in enumerate(kf.split(indices), start=1):
        # If test metric for this fold already exists, skip to next fold
        test_metric_path = os.path.join(args.output_dir, f'k_{k}_test_metric.json')
        
        if test_metric_path and os.path.exists(test_metric_path):
            print(f"Skip fold {k} as test metric exists: {test_metric_path}")
            with open(test_metric_path) as fp:
                fold_metrics = json.load(fp)
            if isinstance(fold_metrics, list) and len(fold_metrics) > 0:
                all_test_metric_dic[k] = fold_metrics[-1].get('avg_metrics', {})
                print(all_test_metric_dic[k])
            continue
        
        print(f"\n===== FOLD {k}/{args.k_fold} =====")
       
        dataset_train = Subset(dataset_trainval, train_idx)
        dataset_eval = Subset(dataset_trainval, val_idx)  
          
        ## save the k-fold train/val set 
        fold_dir = Path(args.fold_out_dir) / f'fold{k}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        with open(fold_dir/f"fold{k}_train.json", "w", encoding='utf-8') as fp:
            json.dump([dataset_trainval[i] for i in train_idx], fp, indent=2, default=json_default)

        with open(fold_dir/f"fold{k}_val.json", "w", encoding='utf-8') as fp:
            json.dump([dataset_trainval[i] for i in val_idx], fp, indent=2, default=json_default)

        if args.model_name in LMs or args.model_name in SLMs: 
            trainer = RFMBaselineTrainer(
                args=args,
                model=model,
                train_dataset=dataset_train,
                eval_dataset=dataset_eval,
                test_dataset=dataset_test,
                data_collator=_collate_fn,
                loss_fn=_loss_fn,
                optimizer=optimizer,
                compute_metrics=compute_metrics,
                best_ckpt_path=args.best_ckpt_path,
            )
            best_val_metric = trainer.run(args.kfold_phase, k)
            all_test_metric_dic[k] = best_val_metric
            print(f'Fold {k} best val score: {best_val_metric}') 
                   
        elif args.model_name in MM: 
            # "RF", "Gradient Boosting", "XGBoost", "SGD", "Logistic Regression", "SVM", "KNN"
            trainer = EdgeOrientSklearnTrainer(
            args=args,
            model_name=args.model_name,
            train_dataset=dataset_train,
            eval_dataset=dataset_eval,
            test_dataset=dataset_test,
            data_collator=_collate_fn,
            compute_metrics=compute_metrics,
            best_ckpt_path=args.best_ckpt_path,
            sample_pairs_ratio=0.5,  # tune for speed vs. accuracy
            )
            best_val_score = trainer.run(args.kfold_phase, k)
            all_test_metric_dic[k] = best_val_score
            print(f'Fold {k} best val score: {best_val_score}')
            
        elif args.model_name in MODELS:
            trainer = NCfoldTrainer(
                args=args,
                model=model,
                train_dataset=dataset_train,
                eval_dataset=dataset_eval,
                test_dataset=dataset_test,
                data_collator=_collate_fn,
                loss_fn=_loss_fn,
                optimizer=optimizer,
                compute_metrics=compute_metrics,
                best_ckpt_path=args.best_ckpt_path,
            )
            best_val_metric = trainer.run(args.kfold_phase, k)
            all_test_metric_dic[k] = best_val_metric
            print(f'Fold {k} best val score: {best_val_metric}')
            
        else:
            print('Unknown model name: {}'.format(args.model_name))


    # Calculate mean of all avg_metrics across folds
    print('all test metric dic:', all_test_metric_dic)
    all_fold_out_dir = os.path.join(args.output_dir, 'metric', 'all_folds')
    os.makedirs(all_fold_out_dir, exist_ok=True)
    all_test_metric_path = os.path.join(all_fold_out_dir, 'all_test_metric.json')

    with open(all_test_metric_path, 'w') as fp:
        json.dump(all_test_metric_dic, fp, indent=2)
    print(f"Saved all test metrics to {all_test_metric_path}")
    
    
    values = defaultdict(list)
    for result in all_test_metric_dic.values():
        for k, v in result.items():
            values[k].append(v)
    mean_results = {f"mean_{k}": sum(vs)/len(vs) for k, vs in values.items()}

    ##   TODO : check
    print(f'All avg_metrics across folds: {mean_results}')


    # Save the results to mean_metric.json
    mean_metric_path = os.path.join(args.output_dir, 'metric', 'mean_metric.json')
    with open(mean_metric_path, 'w') as fp:
        json.dump(mean_results, fp, indent=2)
    print(f"Saved mean metrics to {mean_metric_path}")

    print(f'End time: {datetime.now()}')


if __name__ == "__main__":
    train_and_test()
