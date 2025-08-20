import os
import re
import json
import shutil
import subprocess

from tqdm import tqdm
import pandas as pd
import numpy as np

from BPfold.util.RNA_kit import read_fasta


orient_type_dic = {
                   # no-pair: 0
                   'tran': 1,
                   'cis': 2,
                  }
edge_type_dic = {
                # no-edge: 0
               'W': 1,
               '+': 1,
               '-': 1,
               'H': 2,
               'S': 3,
                }
orient_index_dic = {
                    1: 'tran',
                    2: 'cis',
                  }
edge_index_dic = {
                  1: 'W',
                  2: 'H',
                  3: 'S',
                }

def edge_orient_to_basepair(edge, orient):
    ret = []
    L = len(edge)
    for i in range(L):
        for j in range(i+2, L):
            if orient[i][j]>0:
                ret.append((i, j, edge_index_dic[edge[i]], edge_index_dic[edge[j]], orient_index_dic[orient[i][j]]))
    return ret


def edge_orient_to_basepair_batch(edge, orient):
    def get_pred_edge(edge_arr):
        # Lx4
        return np.argmax(edge_arr[:, 1:], axis=-1)+1 # ignore class 0, TODO

    shape = np.array(edge).shape
    if len(shape)==1:
        return edge_orient_to_basepair(edge, orient)
    elif len(shape)==2:
        if shape[-1] == 4:
            return edge_orient_to_basepair(get_pred_edge(edge), np.argmax(orient, axis=0))
        else:
            return [edge_orient_to_basepair(edge[i], orient[i]) for i in range(shape[0])]
    elif len(edge.shape)==3:
        return [edge_orient_to_basepair(get_pred_edge(edge[i]), np.argmax(orient[i], axis=0)) for i in range(shape[0])]
    else:
        raise Exception(f'[Error] edge shape:{edge.shape}')


def get_seq_and_SS_from_PDB_by_onepiece(pdb_path):
    tmp_dir = 'tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, 'tmp_op.ss')
    op_src = '/public2/home/heqinzhu/gitrepo/RNA/RNA3d/onePiece/src'
    os.system(f'java -cp {op_src} Zhu_onepiece {pdb_path} {tmp_file} > log_op 2>&1')
    with open(tmp_file) as fp:
        head, seq, ss = fp.read().split('\n')
        return seq, ss


def run_RNAVIEW(dest, data_dir):
    os.makedirs(dest, exist_ok=True)
    for f in os.listdir(data_dir):
        if len(f)==8 and f.endswith('.pdb'):
            name = f[:f.rfind('.')]
            src_path = os.path.join(data_dir, f)
            dest_path = os.path.join(dest, name+'.txt')
            if not os.path.exists(dest_path) or os.path.getsize(dest_path)==0:
                cmd = f'rnaview {src_path} > {dest_path} 2>&1'
                print(cmd)
                continue
                os.system(cmd)


def get_connSS_from_PDB_by_RNAVIEW(pdb_path):
    '''
        will generate files in data_dir, awful
    '''
    pattern = r'pair-type\s+\d+\s+(?P<left>\d+)\s+(?P<right>\d+) [\-\+][\-\+]c'

    length_pattern = r'The total base pairs =\s+\d+ \(from\s+(?P<length>\d+) bases\)'
    tmp_dir = '.tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    dest_path = os.path.join(tmp_dir, os.path.basename(pdb_path))
    if os.path.exists(dest_path):
        os.remove(dest_path)
    shutil.copy(pdb_path, dest_path)
    CMD = f'rnaview {dest_path}'
    res = subprocess.Popen(CMD, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    text = res.stdout.read()
    res.stdout.close()

    length = None
    for L in re.findall(length_pattern, text):
        length = int(L)
    connects = [0]*length
    for left, right in re.findall(pattern, text):
        left, right = int(left), int(right)
        connects[left-1] = right-1
        connects[right-1] = left-1
    return connects


def parse_RNAVIEW_out(pdb_path, dest='.', save_info=False):
    '''
        pdb_path: *.pdb
        file_path: *.pdb.out, after run `RNAVIEW *.pdb`
    '''
    pattern = r'\s+(?P<left>\d+)_(?P<right>\d+),.*?(?P<pair>[augctAUGCT]\-[augctAUGCT]).*?: (?P<edge_type>[\+\-WHS]/[\+\-WHS]|\s+) +(?P<orientation>cis|tran|stacked).*?(?P<class_id>[IVX]+|n/a|!.*?\(.*?\)|)\n'

    length_pattern = r'The total base pairs =\s+\d+ \(from\s+(?P<length>\d+) bases\)'
    file_path = pdb_path + '.out'
    if not os.path.exists(file_path):
        CMD = f'rnaview {pdb_path}'
        res = subprocess.Popen(CMD, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # text = res.stdout.read()
        # res.stdout.close()
    with open(file_path) as fp:
        text = fp.read()
    length = None
    for L in re.findall(length_pattern, text):
        length = int(L)
    info = []
    for left, right, pair, edge_type, orientation, class_id  in re.findall(pattern, text):
        info.append({
                     'left': int(left),
                     'right': int(right),
                     'edge_type': edge_type,
                     'orientation': orientation,
                     'class_id': class_id,
                     'pair': pair,
                    })
    if len(info)!=0 and save_info:
        os.makedirs(dest, exist_ok=True)
        name = os.path.basename(pdb_path)
        name = name[:name.rfind('.')]
        dest_path = os.path.join(dest, name+'.json')
        with open(dest_path, 'w') as fp:
            json.dump(info, fp)
    return info


def cif2pdb(src, dest):
    from Bio.PDB import MMCIFParser, PDBIO
    structure_id = "example"
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(structure_id, src)
    io = PDBIO()
    io.set_structure(structure)
    io.save(dest)



def prepare_dataset_onepiece(dest, data_dir, rerun=False):
    os.makedirs(dest, exist_ok=True)
    dest_path = os.path.join(dest, 'PDB_SS.csv')
    data = []

    if not os.path.exists(dest_path) or rerun:
        for f in os.listdir(data_dir):
            if len(f)==8 and f.endswith('.pdb'):
                src_path = os.path.join(data_dir, f)
                seq, ss = get_seq_and_SS_from_PDB_by_onepiece(src_path)
                data.append({'seq': seq, 'SS': ss})
        df = pd.DataFrame(data)
        df.to_csv(dest_path, index=False)
    else:
        df = pd.read_csv(dest_path, index_col=False)
    return df


def prepare_dataset_RNAVIEW_json(dest, pdb_dir):
    if not dest.endswith('.json'):
        dest = dest+'.json'
    if not os.path.exists(dest):
        data = []
        files = [f for f in os.listdir(pdb_dir) if len(f)==8 and f.endswith('.pdb')]
        for f in tqdm(files):
            name = f[:f.rfind('.')]
            pdb_path = os.path.join(pdb_dir, f)
            seq, ss = get_seq_and_SS_from_PDB_by_onepiece(pdb_path)
            info = parse_RNAVIEW_out(pdb_path)
            if len(info)!=0:
                data.append({
                              'name': name,
                              'seq': seq,
                              'pair_info': info,
                             })
        with open(dest, 'w') as fp:
            json.dump(data, fp)


def construct_RNAVIEW_labels(data_dic, verbose=False):
    '''
        data_dic:
            seq: str
            name: str
            pair_info: dict
    '''
    info = data_dic['pair_info']
    seq = data_dic['seq'].upper()
    L = len(seq)
    labels = {
            # LxL orient matrix: 0 for no-pair, 1 for trans, 2 for cis
            # 1xL edge array: 0 for no-edge, 1 for Watson-crick, 2 for Hoogesteen, 3 for Sugar.
            'pair': {'orient_mat': np.zeros((L, L), dtype=int), 'edge_arr': np.zeros(L, dtype=int)},
            'stack': np.zeros((L, L), dtype=int), # optional
           }
    pair_types = []
    for dic in info:
        left, right = dic['left']-1, dic['right']-1 # 1-indexed
        orientation = dic['orientation']
        class_id = dic['class_id']
        pair = dic['pair'].upper()
        if not (left < L and right < L and seq[left] == pair[0] and seq[right] == pair[2]):
            if verbose:
                print(f'[Error] Bases dismatch, {pair},  seq={seq}, left={left}, right={right}')
            return None
        if not dic['pair']:
            return None
        if class_id.startswith('!'): # 3D interaction, ignored
            pass
        elif orientation == 'stacked': # stacking interaction
            labels['stack'][left][right] = labels['stack'][right][left] = 1
        else: # pair intercation
            left_edge, _, right_edge = dic['edge_type']
            if orientation in ['cis', 'tran'] and all(edge in 'WHS+-' for edge in [left_edge, right_edge]):
                labels['pair']['orient_mat'][left][right] = labels['pair']['orient_mat'][right][left] = orient_type_dic[orientation]
                labels['pair']['edge_arr'][left] = edge_type_dic[left_edge]
                labels['pair']['edge_arr'][right] = edge_type_dic[right_edge]
            else:
                raise Exception(f'[Warning] Unknown info: {dic}')
            pair_types.append(left_edge+right_edge+orientation[0])
    return {'labels': labels, 'pair_types': pair_types}


def load_dataset_RNAVIEW(data_path, max_seq_len=None, filter_fasta=None):
    with open(data_path) as fp:
        json_data = json.load(fp)
    index_dest = data_path[:data_path.rfind('.')]+'_index.json'
    total_ct = 0
    error_ct = 0
    data_list = []
    index_data = []
    lengths = []
    for dic in json_data:
        name = dic['name']
        seq = dic['seq']
        d = construct_RNAVIEW_labels(dic)
        if d is None or len(seq)==0 or len(d['labels'])==0:
            error_ct+=1
        else:
            labels = d['labels']
            pair_types = d['pair_types']
            total_ct+=1
            lengths.append(len(seq))
            data_list.append({
                                 'seq': seq,
                                  'name': name,
                                 'labels': labels,
                                })
            index_data.append({
                               'name': name,
                               'seq': seq,
                               'pair_types': pair_types,
                              })
            # 1xL edge array: 0 for Watson-crick, 1 for Hoogesteen, 2 for Sugar.
            # print(labels['pair']['edge_arr']) # L, 
            # LxL orient matrix: 0 for no-pair, 1 for trans, 2 for cis.
            # print(labels['pair']['orient_mat'])

    with open(index_dest, 'w') as fp:
        json.dump(index_data, fp)
    if filter_fasta is not None:
        names = {name for name, seq in read_fasta(filter_fasta)}
        data_list = [d for d in data_list if d['name'] in names]

    print(f'Processing {data_path}: valid={total_ct}, invalid={error_ct}')
    print(f'Lengths: min={np.min(lengths)}, max={np.max(lengths)}')
    if max_seq_len is not None:
        data_list = [d for d in data_list if len(d['seq'])<=max_seq_len]
        print(f'  L<={max_seq_len} = {len([l for l in lengths if l<=max_seq_len])}')
    return data_list


if __name__ == '__main__':
    data_dir = 'data/PDB_download'
    json_path = 'data/NC_data.json'

    # df = prepare_dataset_onepiece(dest, data_dir, rerun=False)
    # print(df)

    # run_RNAVIEW(dest, data_dir) # deprecated
    prepare_dataset_RNAVIEW_json(json_path, data_dir)
    load_dataset_RNAVIEW(json_path)

    ## Example
    # pdb_path = os.path.join(data_dir, '3Q51.pdb')
    # info = parse_RNAVIEW_out(pdb_path, dest='.')
    # print(info, len(info))
