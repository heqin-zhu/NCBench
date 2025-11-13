import os

import torch
import numpy as np
import torch.nn.functional as F

from BPfold.util.base_pair_motif import BPM_energy

from util.NCfold_kit import Stack
from .LM_embeddings import get_LM_models, get_LM_embedding, get_LM_embedding_score


class BaseCollator(object):
    def __init__(self):
        self.stack_fn = Stack()

    def __call__(self, raw_data_b):
        raise NotImplementedError("Must implement __call__ method.")


class NCfoldCollator(BaseCollator):
    def __init__(self, max_seq_len=None, replace_T=True, replace_U=False, LM_list=None, LM_checkpoint_dir='LM_checkpoint', top_k=1, rerun=False, *args, **kargs):
        super(NCfoldCollator, self).__init__()
        self.rerun = rerun
        self.max_seq_len = max_seq_len
        # only replace T or U
        assert replace_T ^ replace_U, "Only replace T or U."
        self.replace_T = replace_T
        self.replace_U = replace_U
        self.base2idx = {'A': 0, 'U':1, 'G':2, 'C':3, 'N':4, 'PAD':5}
        self.BPM = BPM_energy()
        self.LM_cache_dir = os.path.join(LM_checkpoint_dir, 'embeddings')
        if LM_list is not None:
            self.LM_list = LM_list
            self.LM_dic = get_LM_models(LM_checkpoint_dir, self.LM_list)
        else:
            self.LM_list = []
        self.top_k = top_k

    def __call__(self, raw_data_b):
        input_ids_b = []
        label_edge = []
        label_orient = []
        seq_b = []
        name_b = []
        mat_b = []
        seq_mask = []
        mat_mask = []
        LM_embed_list = []
        for idx, raw_data in enumerate(raw_data_b):
            seq = raw_data["seq"]
            name = raw_data['name']
            L = len(seq)
            seq = seq.upper()
            seq = seq.replace("T", "U") if self.replace_T else seq.replace("U", "T")
            seq_b.append(seq)

            ## LM embed
            if self.LM_list:
                scores = []
                for LM_name in self.LM_list:
                     
                    scores.append(get_LM_embedding_score(self.LM_dic[LM_name], LM_name, name, seq, cache_dir=self.LM_cache_dir, rerun=self.rerun))

                top_idx = np.argsort(scores)[-self.top_k:]
                fused_embeddings = []
                for i in top_idx:
                    LM_name = self.LM_list[i]
                    fused_embeddings.append(get_LM_embedding(self.LM_dic[LM_name], LM_name, name, seq, cache_dir=self.LM_cache_dir, rerun=self.rerun))
                fused_embedding = np.stack([(em @ em.T).detach().cpu().numpy()  for em in fused_embeddings], axis=0) # top_k x L x L
                LM_embed_list.append(fused_embedding)

            mat_b.append(self.BPM.get_energy(seq))
            name_b.append(name)
            input_text = [base if base in 'AUGC' else 'N' for base in seq]
            input_ids = [self.base2idx[base] for base in input_text]
            input_ids_b.append(input_ids)
            label_dic = raw_data["labels"]['pair']
            label_edge.append(label_dic['edge_arr'])
            label_orient.append(label_dic['orient_mat'])
            mat_mask.append(np.ones((L, L)))
            seq_mask.append(np.ones(L))
        batch_seq_len = max([len(x) for x in input_ids_b])
        if self.max_seq_len:
            batch_seq_len = min(self.max_seq_len, batch_seq_len)

        for i_batch in range(len(input_ids_b)):
            L = len(input_ids_b[i_batch])
            if L > batch_seq_len:
                input_ids_b[i_batch] = input_ids_b[i_batch][:batch_seq_len]
                label_edge[i_batch] = label_edge[i_batch][:batch_seq_len]
                label_orient[i_batch] = label_edge[i_batch][:batch_seq_len, :batch_seq_len]
                mat_b[i_batch] = mat_b[i_batch][:, :batch_seq_len, :batch_seq_len]
                mat_mask[i_batch] = mat_mask[i_batch][:batch_seq_len, :batch_seq_len]
                seq_mask[i_batch] = seq_mask[i_batch][:batch_seq_len]
                if LM_embed_list:
                    LM_embed_list[i_batch] = LM_embed_list[i_batch][:, :batch_seq_len, :batch_seq_len]
            elif L < batch_seq_len:
                pad_len = (batch_seq_len - L)
                input_ids_b[i_batch] += [self.base2idx['PAD']] * pad_len
                label_edge[i_batch] = np.pad(label_edge[i_batch], (0, pad_len), constant_values=-1)
                label_orient[i_batch] = np.pad(label_orient[i_batch], ((0, pad_len), (0, pad_len)), constant_values=-1)
                mat_b[i_batch] = np.pad(mat_b[i_batch], ((0, 0), (0, pad_len), (0, pad_len)), constant_values=0)
                mat_mask[i_batch] = np.pad(mat_mask[i_batch], ((0, pad_len), (0, pad_len)), constant_values=0)
                seq_mask[i_batch] = np.pad(seq_mask[i_batch], ((0, pad_len)), constant_values=0)
                if LM_embed_list:
                    LM_embed_list[i_batch] = np.pad(LM_embed_list[i_batch], ((0, 0), (0, pad_len), (0, pad_len)), constant_values=0)

        data_dic = {
            'mat_mask': torch.from_numpy(self.stack_fn(mat_mask)).bool(),
            'seq_mask': torch.from_numpy(self.stack_fn(seq_mask)).bool(),
            "input_ids": torch.from_numpy(self.stack_fn(input_ids_b)).long(),
            'mat': torch.FloatTensor(self.stack_fn(mat_b)),
            "label_edge": torch.from_numpy(self.stack_fn(label_edge)).long(),
            "label_orient": torch.from_numpy(self.stack_fn(label_orient)).long(),
            'name': name_b,
            'seq': seq_b,
            }
        if LM_embed_list:
            data_dic['LM_embed'] = torch.FloatTensor(self.stack_fn(LM_embed_list))
        return data_dic


