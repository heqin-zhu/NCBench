import torch
import numpy as np

from ..util.NCfold_kit import Stack


class BaseCollator(object):
    def __init__(self):
        self.stack_fn = Stack()

    def __call__(self, raw_data_b):
        raise NotImplementedError("Must implement __call__ method.")


class NCfoldCollator(BaseCollator):
    def __init__(self, label2id, replace_T=True, replace_U=False, max_seq_len=None):
        super(NCfoldCollator, self).__init__()
        self.max_seq_len = max_seq_len
        self.label2id = label2id
        # only replace T or U
        assert replace_T ^ replace_U, "Only replace T or U."
        self.replace_T = replace_T
        self.replace_U = replace_U
        self.base2idx = {'A': 0, 'U':1, 'G':2, 'C':3, 'N':4, 'PAD':5}

    def __call__(self, raw_data_b):
        input_ids_b = []
        label_edge = []
        label_orient = []
        seq_b = []
        name_b = []
        for idx, raw_data in enumerate(raw_data_b):
            seq = raw_data["seq"]
            seq = seq.upper()
            seq = seq.replace("T", "U") if self.replace_T else seq.replace("U", "T")
            seq_b.append(seq)
            name_b.append(raw_data['name'])
            input_text = [base if base in 'AUGC' else 'N' for base in seq]
            input_ids = [self.base2idx[base] for base in input_text]
            input_ids_b.append(input_ids)
            label_dic = raw_data["labels"]['pair']
            label_edge.append(label_dic['edge_arr'])
            label_orient.append(label_dic['orient_mat'])
            # if idx == 0:
            #     print(input_text)
            #     print(input_ids)
        batch_seq_len = max([len(x) for x in input_ids_b])
        if self.max_seq_len:
            batch_seq_len = min(self.max_seq_len, batch_seq_len)

        for i_batch in range(len(input_ids_b)):
            L = len(input_ids_b[i_batch])

            if L > batch_seq_len:
                input_ids_b[i_batch] = input_ids_b[i_batch][:batch_seq_len]
                label_edge[i_batch] = label_edge[i_batch][:batch_seq_len]
                label_orient[i_batch] = label_edge[i_batch][:batch_sesq_len, :batch_seq_len]
            elif L < batch_seq_len:
                pad_len = (batch_seq_len - L)
                input_ids_b[i_batch] += [self.base2idx['PAD']] * pad_len
                label_edge[i_batch] += [-1] * pad_len
                label_orient[i_batch] = np.pad(label_orient[i_batch], ((0, pad_len), (0, pad_len)), constant_value=-1)

        return {
            "input_ids": torch.from_numpy(self.stack_fn(input_ids_b)),
            "label_edge": torch.from_numpy(self.stack_fn(label_edge)),
            "label_orient": torch.from_numpy(self.stack_fn(label_orient)),
            'name': name_b,
            'seq': seq_b,
               }
