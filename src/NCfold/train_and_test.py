import os
import time
import json
import argparse
from datetime import datetime
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .dataset.RNAdata import RNAdata
from .dataset.collator import NCfoldCollator
from .dataset.LM_embeddings import LM_dim_dic
from .model.AttnMatFusion_net import AttnMatFusion_net
from .model.SeqMatFusion_net import SeqMatFusion_net
from .model.loss_and_metric import NCfoldLoss, compute_metrics
from .util.NCfold_kit import str2bool, str2list, count_para, get_config
from .util.data_processing import edge_orient_to_basepair_batch


MODELS = ['AttnMatFusion_net']
DATASETS = ["PDB_NC"]
LMs = ['structRFM', 'aido.rna-650m', 'aido.rna-1.6b', 'ernierna', 'ernierna-ss', 'rinalmo-giga', 'rinalmo-mega', 'rinalmo-micro', 'rnabert', 'rnaernie', 'rnafm', 'mrnafm', 'rnamsm', 'splicebert', 'splicebert.510', 'splicebert-human.510', 'utrbert-3mer', 'utrbert-4mer', 'utrbert-5mer', 'utrbert-6mer', 'utrlm-te_el', 'utrlm-mrl']


class BaseTrainer(object):
    def __init__(self,
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
                 visual_writer=None):
        self.args = args
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
        if metrics_dataset[self.prime_metric] > self.max_metric:
            self.max_metric = metrics_dataset[self.prime_metric]
            save_model_path = os.path.join(checkpoint_dir, f"epoch{epoch}_{self.max_metric:.3f}.pth")
            torch.save(self.model.state_dict(), save_model_path)
            print(f"Save model:", save_model_path)
            
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


class NCfoldTrainer(BaseTrainer):
    def train(self, epoch):
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


    def save_pred_gt_results(self, raw_out_dir, names, seqs, pred_edge, pred_orient, gt_edge_np, gt_orient_np):
        pred_edge_np, pred_orient_np = pred_edge.detach().cpu().numpy(), pred_orient.detach().cpu().numpy()
        for i in range(len(pred_edge_np)):
            name = names[i]
            np.savez(os.path.join(raw_out_dir, f'{name}_pred.npz'), edge=pred_edge_np[i], orient=pred_orient_np[i], )
            np.savez(os.path.join(raw_out_dir, f'{name}_gt.npz'), edge=gt_edge_np[i], orient=gt_orient_np[i], )
            with open(os.path.join(raw_out_dir, f'{name}_pred.ss'), 'w') as fp:
                fp.write('seq:' + seqs[i]+'\n')
                gt_edge_str = ','.join([str(x) for x in gt_edge_np[i].tolist()])
                fp.write('gt_edge:'+gt_edge_str+'\n')
                edge_str = ','.join([str(x) for x in pred_edge_np[i].tolist()])
                fp.write('pred_edge:'+edge_str+'\n')

    def eval(self, epoch):
        self.model.eval()
        time_st = time.time()
        num_total = 0
        outputs_names, outputs_seqs = [], []
        pred_edges, pred_orients, gt_edges, gt_orients = [], [], [], []
        out_preds, out_gts = [], []
        raw_out_dir = os.path.join(self.args.output_dir, 'out_npy')
        os.makedirs(raw_out_dir, exist_ok=True)
        with tqdm(total=len(self.eval_dataset)) as pbar:
            for i, data in enumerate(self.eval_dataloader):
                input_ids = data["input_ids"].to(self.args.device)
                mat = data["mat"].to(self.args.device)
                mat_mask = data["mat_mask"].to(self.args.device)
                seq_mask = data["seq_mask"].to(self.args.device)
                LM_embed = data["LM_embed"].to(self.args.device) if 'LM_embed' in data else None
                with torch.no_grad():
                    pred_edge, pred_orient = self.model(input_ids, mat, LM_embed, seq_mask, mat_mask)
                num_total += self.args.batch_size
                pred_edges += [b for b in pred_edge] # batch
                pred_orients += [b for b in pred_orient]
                gt_edges += [b for b in data["label_edge"]]
                gt_orients += [b for b in data["label_orient"]]
                pred_edge_np, pred_orient_np = pred_edge.detach().cpu().numpy(), pred_orient.detach().cpu().numpy()
                gt_edge_np, gt_orient_np = data["label_edge"].detach().cpu().numpy(), data["label_orient"].detach().cpu().numpy()
                self.save_pred_gt_results(raw_out_dir, data['name'], data['seq'], pred_edge, pred_orient, gt_edge_np, gt_orient_np)

                out_preds.append(edge_orient_to_basepair_batch(pred_edge_np, pred_orient_np))
                out_gts.append(edge_orient_to_basepair_batch(gt_edge_np, gt_orient_np))
                outputs_names += data['name']
                outputs_seqs += data['seq']
                if num_total >= self.args.logging_steps:
                    num_total = 0
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

        self.save_model(metrics_dataset, epoch)
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

    def test(self):
        self.model.eval()
        time_st = time.time()
        num_total = 0
        outputs_names, outputs_seqs = [], []
        pred_edges, pred_orients, gt_edges, gt_orients = [], [], [], []
        out_preds, out_gts = [], []
        raw_out_dir = os.path.join(self.args.output_dir, 'out_npy')
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
                self.save_pred_gt_results(raw_out_dir, data['name'], data['seq'], pred_edge, pred_orient, gt_edge_np, gt_orient_np)

                out_preds.append(edge_orient_to_basepair_batch(pred_edge_np, pred_orient_np))
                out_gts.append(edge_orient_to_basepair_batch(gt_edge_np, gt_orient_np))
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
        metric_dir = os.path.join(self.args.output_dir, 'metrics')
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
        test_metric_path = os.path.join(self.args.output_dir, 'test_metric.json')
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

          
    def fit(self):
        for epoch in range(self.args.num_train_epochs):
            self.train(epoch)
            self.eval(epoch)
            
        if self.test_dataset is not None:
            if self.best_ckpt_path and os.path.exists(self.best_ckpt_path):
                self.model.load_state_dict(torch.load(self.best_ckpt_path, map_location=self.args.device))
            else:
                print('Best checkpoint not found. Using the last epoch weights for test evalutation')  
            self.test()
     


def get_args():
    parser = argparse.ArgumentParser('Implementation of RNA sequence classification.')
    # data
    parser.add_argument('--dataset_dir', type=str, default="data")
    parser.add_argument('--dataset', type=str, default="PDB_NC", choices=DATASETS)
    parser.add_argument('--filter_fasta', type=str)
    parser.add_argument('--output_dir', type=str, default='.runs/tmp')
    parser.add_argument('--include_canonical', action='store_true')
    parser.add_argument('--use_RFdiff_data', action='store_true')
    parser.add_argument('--LM_list', nargs='*', default=['structRFM'], choices=LMs)
    parser.add_argument('--LM_checkpoint_dir', type=str, default='LM_checkpoint', help='LM checkpoint_dir, each LM is placed in a subdir of same name.')

    ## Fuse LM embeddings
    parser.add_argument('--top_k', type=int, default=2)

    # model args
    parser.add_argument('--model_name', type=str, default="AttnMatFusion_net", choices=MODELS)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_blocks', type=int, default=12)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--use_BPM', type=bool, default=False)
    parser.add_argument('--replace_T', type=bool, default=True)
    parser.add_argument('--replace_U', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--batch_size', type=int, default=64, help='The number of samples used per step & per device.')
    parser.add_argument('--num_train_epochs', type=int, default=60, help='The number of epoch for training.')
    parser.add_argument('--logging_steps', type=int)
    # loss weight
    parser.add_argument('--weight_edgeW', type=float, default=5)
    parser.add_argument('--weight_edgeH', type=float, default=20)
    parser.add_argument('--weight_edgeS', type=float, default=20)
    parser.add_argument('--weight_trans', type=float, default=20)
    parser.add_argument('--weight_cis', type=float, default=20)
    args = parser.parse_args()
    return args


def train_and_test():
    print(f'Begin time: {datetime.now()}')
    args = get_args()
    assert args.replace_T ^ args.replace_U, "Only replace T or U."
    if args.logging_steps is None:
        args.logging_steps = 1500//args.batch_size
    print(args)

    if args.model_name == "AttnMatFusion_net":
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
    elif args.model_name == "SeqMatFusion_net":
        model = SeqMatFusion_net(seq_dim=args.hidden_dim, mat_channels=args.hidden_dim, num_blocks=args.num_blocks)
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

    dataset_train = RNAdata(args.dataset_dir, args.max_seq_len, filter_fasta=args.filter_fasta, phase='train', include_canonical=args.include_canonical, use_RFdiff_data=args.use_RFdiff_data)
    dataset_eval = RNAdata(args.dataset_dir, args.max_seq_len, filter_fasta=args.filter_fasta, phase='validate', include_canonical=args.include_canonical)
    dataset_test = RNAdata(args.dataset_dir, args.max_seq_len, filter_fasta=args.filter_fasta, phase='test', include_canonical=args.include_canonical)

    print(f'dataset_dir={args.dataset_dir}, filter={args.filter_fasta}: train:val:test={len(dataset_train)}:{len(dataset_eval)}:{len(dataset_test)}') 
    ## max_seq_len == None, for setting batch_max_len
    _collate_fn = NCfoldCollator(max_seq_len=None, replace_T=args.replace_T, replace_U=args.replace_U, LM_list=args.LM_list, LM_checkpoint_dir=args.LM_checkpoint_dir, top_k=args.top_k)
    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)
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
    )
    # if args.train:
    #     for i_epoch in range(args.num_train_epochs):
    #         trainer.train(i_epoch)
    #         trainer.eval(i_epoch)

    trainer.fit()

    print(f'End time: {datetime.now()}')

if __name__ == "__main__":
    train_and_test()
