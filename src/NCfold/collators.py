import torch
import numpy as np

from base_classes import BaseCollator


class SeqClsCollator(BaseCollator):
    def __init__(self, label2id, replace_T=True, replace_U=False, max_seq_len=None):
        super(SeqClsCollator, self).__init__()
        self.max_seq_len = max_seq_len
        self.label2id = label2id
        # only replace T or U
        assert replace_T ^ replace_U, "Only replace T or U."
        self.replace_T = replace_T
        self.replace_U = replace_U
        self.base2idx = {'A': 0, 'U':1, 'G':2, 'C':3, 'N':4, 'PAD':5}

    def __call__(self, raw_data_b):
        input_ids_b = []
        label_b = []
        seq_b = []
        name_b = []
        mat_b = []
        for idx, raw_data in enumerate(raw_data_b):
            seq = raw_data["seq"]
            seq = seq.upper()
            seq = seq.replace("T", "U") if self.replace_T else seq.replace("U", "T")
            seq_b.append(seq)
            name_b.append(raw_data['name'])
            input_text = [base if base in 'AUGC' else 'N' for base in seq]
            input_ids = [self.base2idx[base] for base in input_text]
            input_ids_b.append(input_ids)
            label = raw_data["label"]
            label_b.append(self.label2id[label])
            mat_b.append(raw_data['mat'])
            # if idx == 0:
            #     print(input_text)
            #     print(input_ids)
        batch_seq_len = max([len(x) for x in input_ids_b])
        if self.max_seq_len:
            batch_seq_len = min(self.max_seq_len, batch_seq_len)

        input_ids_stack = []
        labels_stack = []
        ss_stack = []
        for i_batch in range(len(input_ids_b)):
            input_ids = input_ids_b[i_batch]
            label = label_b[i_batch]

            if len(input_ids) > batch_seq_len:
                input_ids = input_ids[:batch_seq_len]

            pad_len = (batch_seq_len - len(input_ids))
            input_ids += [self.base2idx['PAD']] * pad_len
            input_ids_stack.append(input_ids)
            labels_stack.append(label)

            mat = mat_b[i_batch]
            if len(mat) > batch_seq_len:
                mat = mat[:batch_seq_len, :batch_seq_len]

            mat = np.pad(mat, ((0, pad_len), (0, pad_len)), constant_values=0)
            ss_stack.append(mat)

        return {
            "input_ids": torch.from_numpy(self.stack_fn(input_ids_stack)),
            "mat": torch.FloatTensor(self.stack_fn(ss_stack)),
            "labels": torch.from_numpy(self.stack_fn(labels_stack)),
            'name': name_b,
            'seq': seq_b,
               }

class NucClsCollator(BaseCollator):
    def __init__(self, label2id, replace_T=True, replace_U=False, max_seq_len=None):
        super(NucClsCollator, self).__init__()
        self.max_seq_len = max_seq_len
        self.label2id = label2id
        # only replace T or U
        assert replace_T ^ replace_U, "Only replace T or U."
        self.replace_T = replace_T
        self.replace_U = replace_U
        self.base2idx = {'A': 0, 'U':1, 'G':2, 'C':3, 'N':4, 'PAD':5}

    def __call__(self, raw_data_b):
        input_ids_b = []
        label_b = []
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
            label = raw_data["label"]
            label_b.append(self.label2id[label])
            # if idx == 0:
            #     print(input_text)
            #     print(input_ids)
        batch_seq_len = max([len(x) for x in input_ids_b])
        if self.max_seq_len:
            batch_seq_len = min(self.max_seq_len, batch_seq_len)

        input_ids_stack = []
        labels_stack = []
        ss_stack = []
        for i_batch in range(len(input_ids_b)):
            input_ids = input_ids_b[i_batch]
            label = label_b[i_batch]

            if len(input_ids) > batch_seq_len:
                input_ids = input_ids[:batch_seq_len]

            pad_len = (batch_seq_len - len(input_ids))
            input_ids += [self.base2idx['PAD']] * pad_len
            input_ids_stack.append(input_ids)
            labels_stack.append(label)

        return {
            "input_ids": torch.from_numpy(self.stack_fn(input_ids_stack)),
            "labels": torch.from_numpy(self.stack_fn(labels_stack)),
            'name': name_b,
            'seq': seq_b,
               }
