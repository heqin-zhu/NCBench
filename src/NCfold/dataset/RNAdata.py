import os

from Bio import SeqIO
from torch.utils.data import Dataset

from ..util.data_processing import load_dataset_RNAVIEW


class RNAdata(Dataset):
    def __init__(self, data_dir, seed=42, train=True):
        super(RNAdata, self).__init__()
        self.data_dir = data_dir
        file_name = "train.json" if train else "test.json"
        path = os.path.join(data_dir, file_name)
        self.data = load_dataset_RNAVIEW(path)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
