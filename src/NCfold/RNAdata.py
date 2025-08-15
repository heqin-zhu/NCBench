import os.path as osp

import numpy as np
import pandas as pd
from Bio import SeqIO

from torch.utils.data import Dataset

from BPfold.util.RNA_kit import connects2mat, dbn2connects


class SeqClsDataset(Dataset):
    def __init__(self, data_dir, prefix, seed=42, train=True):
        super(SeqClsDataset, self).__init__()

        self.data_dir = data_dir
        self.prefix = prefix

        # SS: dbn
        if prefix.startswith('nRCS'):
            file_name = "nRCS_512_train_SS.csv" if train else "nRCS_512_test_SS.csv"
            path = osp.join(osp.join(data_dir, prefix), file_name)
            self.df = pd.read_csv(path, index_col=False)
        else:
            file_name = "train.fa" if train else "test.fa"
            fasta = osp.join(osp.join(data_dir, prefix), file_name)
            records = list(SeqIO.parse(fasta, "fasta"))
            data = [(str(x.seq), *x.description.split(" ")[0:2], self.get_SS(x.seq)) for x in records]
            self.df = pd.DataFrame([dict(seq=seq, id=name, family=label, SS=SS) for seq, name, label, SS in data])

    def get_SS(self, seq):
        # TODO, use BPfold to predict
        L = len(seq)
        mat = np.zeros((L, L))
        SS = '.'* L
        return SS


    def __getitem__(self, idx):
        seq = self.df.loc[idx, 'seq']
        name = self.df.loc[idx, 'id']
        label = self.df.loc[idx, 'family']
        mat = connects2mat(dbn2connects(self.df.loc[idx, 'SS']))
        ret = {"seq": seq, "label": label, 'name': name, 'mat': mat}
        return ret

    def __len__(self):
        return len(self.df)


class NucClsDataset(Dataset):
    def __init__(self, data_dir, seed=42, train=True):
        super(NucClsDataset, self).__init__()
        self.data_dir = data_dir
        file_name = "train.csv" if train else "test.csv"
        self.df = pd.read_csv(os.path.join(data_dir, file_name), index_col=False)

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


class RRInterDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, idx):

        instance = self.data[idx]
        return {
            "a_name": instance[0],
            "a_seq": instance[1],
            "b_name": instance[2],
            "b_seq": instance[3],
            "label": instance[4],
        }

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
