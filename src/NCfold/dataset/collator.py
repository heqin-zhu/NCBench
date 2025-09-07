import os

import torch
import numpy as np
import torch.nn.functional as F

from BPfold.util.base_pair_motif import BPM_energy

from ..util.NCfold_kit import Stack


class BaseCollator(object):
    def __init__(self):
        self.stack_fn = Stack()

    def __call__(self, raw_data_b):
        raise NotImplementedError("Must implement __call__ method.")


def outer_product_mean(embedding):
    '''
        (B) x L x D -> (B) x L x L x D x D -> (B) x L x L
    '''
    shape = embedding.shape
    embedding1 = embedding.unsqueeze(-2).unsqueeze(-1)
    embedding2 = embedding.unsqueeze(-3).unsqueeze(-2)
    outer = embedding1 * embedding2 
    outer_mean = outer.reshape(*shape[:-1], shape[-2], -1).mean(dim=-1)
    return outer_mean


class NCfoldCollator(BaseCollator):
    def __init__(self, max_seq_len=None, replace_T=True, replace_U=False, top_k=1, *args, **kargs):
        super(NCfoldCollator, self).__init__()
        self.max_seq_len = max_seq_len
        # only replace T or U
        assert replace_T ^ replace_U, "Only replace T or U."
        self.replace_T = replace_T
        self.replace_U = replace_U
        self.base2idx = {'A': 0, 'U':1, 'G':2, 'C':3, 'N':4, 'PAD':5}
        self.BPM = BPM_energy()
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

            scores = raw_data['scores']
            embeddings = raw_data['embeddings']

            ## LM embed
            if scores and embeddings:
                top_idx = np.argsort(scores)[-self.top_k:]
                fused_embeddings = []
                for i in top_idx:
                    fused_embeddings.append(embeddings[i])
                fused_embedding = np.stack([outer_product_mean(em).detach().cpu().numpy()  for em in fused_embeddings], axis=0) # top_k x L x L
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
