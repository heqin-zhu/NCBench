import os
from torch.utils.data import Dataset
from util.data_processing import load_dataset_RNAVIEW


class RNAdata(Dataset):
    def __init__(self, data_dir, max_seq_len=512, filter_fasta=None, phase='train', include_canonical=False, use_RFdiff_data=False):
        '''
            filter_fasta: fasta file
        '''
        super(RNAdata, self).__init__()
        self.data_dir = data_dir
        filter_path = None if filter_fasta is None else os.path.join(data_dir, filter_fasta)
        if phase=='train':
            file_name = "NCpair_train_data.json"
            self.data = load_dataset_RNAVIEW(os.path.join(data_dir, file_name), max_seq_len, filter_path, include_canonical=include_canonical)
            if use_RFdiff_data:
                RFdiff_path = 'RFdiff_data.json'
                RFdiff_data = load_dataset_RNAVIEW(os.path.join(data_dir, RFdiff_path), max_seq_len, include_canonical=include_canonical)
                print(f'Using Rfdiff data for augmentation: {RFdiff_path}={len(RFdiff_data)}')
                self.data += RFdiff_data
        else:
            if phase=='validate':
                file_name = "NCpair_validation_data.json"
            else:
                file_name = "NCpair_test_data.json"
            self.data = load_dataset_RNAVIEW(os.path.join(data_dir, file_name), max_seq_len, filter_path, include_canonical=include_canonical)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class Kfold_RNAdata(Dataset):
    def __init__(self, data_dir, max_seq_len=512, filter_fasta=None, phase='k_fold_train', include_canonical=False, use_RFdiff_data=False):
        '''
            filter_fasta: fasta file
        '''
        super(Kfold_RNAdata, self).__init__()
        self.data_dir = data_dir
        filter_path = None if filter_fasta is None else os.path.join(data_dir, filter_fasta)
        if phase=='k_fold_train':
            file_name = "NCpair_trainval_data_raw.json"
            self.data = load_dataset_RNAVIEW(os.path.join(data_dir, file_name), max_seq_len, filter_path, include_canonical=include_canonical)
        else:
            file_name = "NCpair_test_data_raw.json"
            self.data = load_dataset_RNAVIEW(os.path.join(data_dir, file_name), max_seq_len, filter_path, include_canonical=include_canonical)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
