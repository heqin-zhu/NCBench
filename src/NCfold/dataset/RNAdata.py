import os
import json
import os.path as osp

import numpy as np
import pandas as pd
from Bio import SeqIO
from torch.utils.data import Dataset

from BPfold.util.RNA_kit import connects2mat, dbn2connects


class RNAData(Dataset):
    def __init__(self, data_dir, seed=42, train=True):
        super(RNAData, self).__init__()
        self.data_dir = data_dir
        file_name = "train.json" if train else "test.josn"
        with open(os.path.join(data_dir, file_name)) as fp:
            data = json.load(fp)


    def __getitem__(self, idx):
        seq = self.df.loc[idx, 'seq']
        name = self.df.loc[idx, 'id']
        label = self.df.loc[idx, 'label']
        ret = {"seq": seq, "label": label, 'name': name}
        return ret

    def __len__(self):
        return len(self.df)


class GenerateRRInterTrainTest:
    def __init__(self,
                 rr_dir,
                 dataset,
                 split=0.8,
                 seed=42):
        csv_path = osp.join(rr_dir, dataset) + ".csv"
        self.data = pd.read_csv(csv_path, sep=",").values.tolist()

        self.split_index = int(len(self.data) * split)

        np_rng = np.random.RandomState(seed=seed)
        np_rng.shuffle(self.data)

    def get(self):
        return RRInterDataset(self.data[:self.split_index]), RRInterDataset(self.data[self.split_index:])


def bpseq2dotbracket(bpseq):
    dotbracket = []
    for i, x in enumerate(bpseq):
        if x == 0:
            dotbracket.append('.')
        elif x > i:
            dotbracket.append('(')
        else:
            dotbracket.append(')')
    return ''.join(dotbracket)


class BPseqDataset(Dataset):

    def __init__(self, data_root, bpseq_list):
        super().__init__()
        self.data_root = data_root
        with open(bpseq_list) as f:
            self.file_path = f.readlines()
            self.file_path = [x.replace("\n", "") for x in self.file_path]

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        file_path = osp.join(self.data_root, self.file_path[idx])
        return self.load_bpseq(file_path)

    def load_bpseq(self, filename):
        with open(filename) as f:
            p = [0]
            s = ['']
            for line in f:
                line = line.rstrip('\n').split()
                idx, c, pair = line
                idx, pair = int(idx), int(pair)
                s.append(c)
                p.append(pair)

        seq = ''.join(s)
        return {"name": filename, "seq": seq, "pairs": np.array(p)}
