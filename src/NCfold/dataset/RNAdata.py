import os

from Bio import SeqIO
from torch.utils.data import Dataset

from BPfold.util.RNA_kit import read_fasta

from ..util.data_processing import load_dataset_RNAVIEW


class RNAdata(Dataset):
    def __init__(self, data_dir, filter_fasta=None, train=True):
        '''
            filter_fasta: fasta file
        '''
        super(RNAdata, self).__init__()
        self.data_dir = data_dir
        file_name = "train.json" if train else "test.json"
        path = os.path.join(data_dir, file_name)
        self.data = load_dataset_RNAVIEW(path)
        if filter_fasta is not None:
            names = {name for name, seq in read_fasta(os.path.join(data_dir, filter_fasta))}
            self.data = [d for d in self.data if d['name'] in names]


    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
