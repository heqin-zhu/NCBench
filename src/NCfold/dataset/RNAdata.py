import os

from Bio import SeqIO
from torch.utils.data import Dataset

from ..util.data_processing import prepare_dataset_RNAVIEW_pickle


class RNAdata(Dataset):
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
