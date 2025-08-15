import os
import json


from BPfold.util.RNA_kit import write_fasta

if __name__ == '__main__':
    path = 'data/NC_data_index.json'

    with open(path,) as fp:
        data = json.load(fp)

    name_seq_pairs = []
    print(len(data))
    for d in data:
        if len(d['pair_types'])!=0:
            name_seq_pairs.append((d['name'], d['seq']))
    print(len(name_seq_pairs))
    write_fasta('data/NC_seq.fasta', name_seq_pairs)
