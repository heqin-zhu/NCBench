import os

from torch.utils.data import Dataset

from ..util.data_processing import load_dataset_RNAVIEW
from .LM_embeddings import get_LM_models, get_LM_embedding, get_LM_embedding_score


class RNAdata(Dataset):
    def __init__(self, data_dir, max_seq_len=512, filter_fasta=None, phase='train', include_canonical=False, use_RFdiff_data=False, LM_list=None, LM_checkpoint_dir='LM_checkpoint', rerun=False, ):
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
        self.rerun = rerun
        self.LM_cache_dir = os.path.join(LM_checkpoint_dir, 'embeddings')
        if LM_list is not None:
            self.LM_list = LM_list
            self.LM_dic = get_LM_models(LM_checkpoint_dir, self.LM_list)
        else:
            self.LM_list = []
        self.prepare_embedding_and_score()


    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def prepare_embedding_and_score(self):
        for raw_data in self.data:
            seq = raw_data["seq"]
            name = raw_data['name']
            L = len(seq)
            seq = seq.upper()
            seq = seq.replace("T", "U")
            ## LM embed
            scores = []
            embeddings = []
            for LM_name in self.LM_list:
                scores.append(get_LM_embedding_score(self.LM_dic[LM_name], LM_name, name, seq, cache_dir=self.LM_cache_dir, rerun=self.rerun))
                embeddings.append(get_LM_embedding(self.LM_dic[LM_name], LM_name, name, seq, cache_dir=self.LM_cache_dir, rerun=self.rerun))
            raw_data['scores'] = scores
            raw_data['embeddings'] = embeddings
