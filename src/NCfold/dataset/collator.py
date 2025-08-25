import os

import torch
import numpy as np
import torch.nn.functional as F

from BPfold.util.base_pair_motif import BPM_energy

from ..util.NCfold_kit import Stack

from structRFM.infer import structRFM_infer


class BaseCollator(object):
    def __init__(self):
        self.stack_fn = Stack()

    def __call__(self, raw_data_b):
        raise NotImplementedError("Must implement __call__ method.")


class NCfoldCollator(BaseCollator):
    def __init__(self, max_seq_len=None, replace_T=True, replace_U=False, LM_list=None, LM_checkpoint_dir='LM_checkpoint'):
        super(NCfoldCollator, self).__init__()
        self.max_seq_len = max_seq_len
        # only replace T or U
        assert replace_T ^ replace_U, "Only replace T or U."
        self.replace_T = replace_T
        self.replace_U = replace_U
        self.base2idx = {'A': 0, 'U':1, 'G':2, 'C':3, 'N':4, 'PAD':5}
        self.BPM = BPM_energy()
        self.LM_list = LM_list
        self.LM_checkpoint_dir = LM_checkpoint_dir
        if self.LM_list is not None:
            self.LM_dic = self.init_LM(self.LM_list)
            # self.cache_dir = os.path.join(LM_checkpoint_dir, 'embeddings') TODO
        else:
            self.LM_list = []

    def __call__(self, raw_data_b):
        input_ids_b = []
        label_edge = []
        label_orient = []
        seq_b = []
        name_b = []
        mat_b = []
        seq_mask = []
        mat_mask = []
        LM_embed_dic = {name: [] for name in self.LM_list}
        for idx, raw_data in enumerate(raw_data_b):
            seq = raw_data["seq"]
            L = len(seq)
            seq = seq.upper()
            seq = seq.replace("T", "U") if self.replace_T else seq.replace("U", "T")
            seq_b.append(seq)
            for LM_name in self.LM_list:
                LM_embed_dic[LM_name].append(self.get_LM_embedding(seq, LM_name))
            mat_b.append(self.BPM.get_energy(seq)) # TODO, N, unknown?
            name_b.append(raw_data['name'])
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
                for k, v in LM_embed_dic.items():
                    LM_embed_dic[k][i_batch] = v[i_batch][:batch_sqe_len,:]
            elif L < batch_seq_len:
                pad_len = (batch_seq_len - L)
                input_ids_b[i_batch] += [self.base2idx['PAD']] * pad_len
                label_edge[i_batch] = np.pad(label_edge[i_batch], (0, pad_len), constant_values=-1)
                label_orient[i_batch] = np.pad(label_orient[i_batch], ((0, pad_len), (0, pad_len)), constant_values=-1)
                mat_b[i_batch] = np.pad(mat_b[i_batch], ((0, 0), (0, pad_len), (0, pad_len)), constant_values=0)
                mat_mask[i_batch] = np.pad(mat_mask[i_batch], ((0, pad_len), (0, pad_len)), constant_values=0)
                seq_mask[i_batch] = np.pad(seq_mask[i_batch], ((0, pad_len)), constant_values=0)
                for k, v in LM_embed_dic.items():
                    LM_embed_dic[k][i_batch] = F.pad(v[i_batch], pad=(0, 0, 0, pad_len), mode='constant', value=0)

        for k, v in LM_embed_dic.items():
            LM_embed_dic[k] = torch.stack(v, dim=0)  # [LxD] -> BxLxD
        data_dic = {
            'mat_mask': torch.from_numpy(self.stack_fn(mat_mask)).bool(),
            'seq_mask': torch.from_numpy(self.stack_fn(seq_mask)).bool(),
            "input_ids": torch.from_numpy(self.stack_fn(input_ids_b)).long(),
            'mat': torch.FloatTensor(self.stack_fn(mat_b)),
            "label_edge": torch.from_numpy(self.stack_fn(label_edge)).long(),
            "label_orient": torch.from_numpy(self.stack_fn(label_orient)).long(),
            'name': name_b,
            'seq': seq_b,
            'LM_embed_dic': LM_embed_dic,
            }
        # for k, v in data_dic.items():
        #     print(k, v[0], v.shape if k not in ['name', 'seq'] else len(v))
        return data_dic

    def get_LM_embedding(self, seq, LM_name):
        model = self.LM_dic[LM_name]
        if LM_name == 'structRFM':
            return model.extract_feature(seq)['seq_feat']
        else:
            raise Exception(f'Unknown LM name: {LM_name}')
    

    def init_LM(self, LM_list):
        dic = {}
        for LM_name in LM_list:
            if LM_name == 'structRFM':
                model = structRFM_infer(os.path.join(self.LM_checkpoint_dir, 'structRFM'), max_length=514)
                dic['structRFM'] = model
            else:
                raise Exception(f'Unknown LM name: {LM_name}')
        return dic
