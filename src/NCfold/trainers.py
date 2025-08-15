import os
import time
from tqdm import tqdm

import torch

from base_classes import BaseTrainer
from collections import defaultdict

import pandas as pd
import numpy as np
from metrics import compare_bpseq


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
