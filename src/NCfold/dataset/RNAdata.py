import os
import json
import os.path as osp

import numpy as np
import pandas as pd
from Bio import SeqIO
from torch.utils.data import Dataset

from BPfold.util.RNA_kit import connects2mat, dbn2connects

from ..util.data_processing import prepare_dataset_RNAVIEW_pickle


class RNAData(Dataset):
    def __init__(self, data_dir, seed=42, train=True):
        super(RNAData, self).__init__()
        self.data_dir = data_dir
        file_name = "train.json" if train else "test.josn"
        path = os.path.join(data_dir, file_name)
        self.data = prepare_dataset_RNAVIEW_pickle(path)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


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
