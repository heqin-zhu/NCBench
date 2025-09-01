import os
import re
import json
import shutil
import subprocess

from tqdm import tqdm
import pandas as pd
import numpy as np

from BPfold.util.RNA_kit import read_fasta
from BPfold.util.misc import get_file_name


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

def get_seq_and_SS_from_PDB_by_onepiece(pdb_path):
    tmp_dir = 'tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, 'tmp_op.ss')
    op_src = '/public2/home/heqinzhu/gitrepo/RNA/RNA3d/onePiece/src'
    os.system(f'java -cp {op_src} Zhu_onepiece {pdb_path} {tmp_file} > log_op 2>&1')
    with open(tmp_file) as fp:
        head, seq, ss = fp.read().split('\n')
        return seq, ss


def get_seq_from_xml_by_RNAVIEW(path):
    pattern = r'<seq-data>\s+(?P<seq>[A-Za-z ]+)\s+</seq-data>'
    with open(path) as fp:
        text = fp.read()
        for f in re.findall(pattern, text):
            return f.replace(' ', '')


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


def run_RNAVIEW(pdb_path):
    CMD = f'rnaview {pdb_path}'
    p = subprocess.Popen(CMD, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # text = p.stdout.read()
    # p.stdout.close()
    return p


def parse_RNAVIEW_out(file_path, save_info=False, run_failed=False):
    '''
        file_path: *.pdb.out, after run `RNAVIEW *.pdb`
        gen: *.pdb.json
    '''
    if not os.path.exists(file_path):
        if run_failed:
            p = run_RNAVIEW(file_path[:-4])
            p.join()
        else:
            return []
    name = get_file_name(file_path)
    dest = os.path.dirname(file_path)
    dest_path = os.path.join(dest, name+'.json')
    if os.path.exists(dest_path):
        with open(dest_path) as fp:
            info = json.load(fp)
            return info
    elif run_failed:
        pattern = r'\s+(?P<left>\d+)_(?P<right>\d+),.*?(?P<pair>[augctAUGCT]\-[augctAUGCT]).*?: (?P<edge_type>[\+\-WHS]/[\+\-WHS]|\s+) +(?P<orientation>cis|tran|stacked).*?(?P<class_id>[IVX]+|n/a|!.*?\(.*?\)|)\n'

        length_pattern = r'The total base pairs =\s+\d+ \(from\s+(?P<length>\d+) bases\)'
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
            with open(dest_path, 'w') as fp:
                json.dump(info, fp)
        return info
    else:
        return []


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


def prepare_dataset_RNAVIEW_json(dest, pdb_dir, filter_fasta=None, rerun=False):
    if not dest.endswith('.json'):
        dest = dest+'.json'
    if not os.path.exists(dest) or rerun:
        data = []
        pdb_paths = [os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if (not f.endswith('_tmp.pdb')) and f.endswith('.pdb')]
        # ps = []
        # for pdb_path in tqdm(pdb_paths):
        #     ps.append(run_RNAVIEW(pdb_path))
        # for p in ps:
        #     if p is not None:
        #         p.wait()
        filter_names = None
        if filter_fasta:
            filter_names = [name for name, seq in read_fasta(filter_fasta)]
        for pdb_path in tqdm(pdb_paths):
            name = get_file_name(pdb_path)
            if filter_names and name in filter_names:
                seq = get_seq_from_xml_by_RNAVIEW(pdb_path+'.xml')
                info = parse_RNAVIEW_out(pdb_path+'.out', save_info=True)
                if len(info)!=0:
                    data.append({
                                  'name': name,
                                  'seq': seq,
                                  'pair_info': info,
                                 })
        with open(dest, 'w') as fp:
            json.dump(data, fp)


def construct_RNAVIEW_labels(data_dic, verbose=False, include_canonical=False):
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
            if not include_canonical and all(edge in 'W+-' for edge in [left_edge, right_edge]) and orientation=='tran': ## canonical
                continue
            if orientation in ['cis', 'tran'] and all(edge in 'WHS+-' for edge in [left_edge, right_edge]):
                labels['pair']['orient_mat'][left][right] = labels['pair']['orient_mat'][right][left] = orient_type_dic[orientation]
                labels['pair']['edge_arr'][left] = edge_type_dic[left_edge]
                labels['pair']['edge_arr'][right] = edge_type_dic[right_edge]
            else:
                raise Exception(f'[Warning] Unknown info: {dic}')
            pair_types.append(left_edge+right_edge+orientation[0])
    return {'labels': labels, 'pair_types': pair_types}


def load_dataset_RNAVIEW(data_path, max_seq_len=None, filter_fasta=None, include_canonical=False):
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
        d = construct_RNAVIEW_labels(dic, include_canonical=include_canonical)
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
        print(f'  L less than {max_seq_len}: {len([l for l in lengths if l<=max_seq_len])}')
    return data_list


def extract_basepair_interaction(pred_edge_np, pred_orient_np, seqs=None):
    '''
        pred_edge_np: numpy.ndarray, (B)xLx4
        pred_orient_np: numpy.ndarray, (B)x3xLxL
    '''
    shape = pred_edge_np.shape
    if len(shape)>3 or len(shape)<2 or shape[-1]!=4:
        raise Exception('Error format of pred_edge: {shape}, should be in shape of Lx4 or BxLx4')
    elif len(shape)==3:
        out_edges, out_pairs, out_preds = [], [], []
        for i in range(len(pred_edge_np)):
            out_edge, out_pair, out_pred = extract_basepair_interaction(pred_edge_np[i], pred_orient_np[i], seqs[i] if seqs else None)
            out_edges.append(out_edge)
            out_pairs.append(out_pair)
            out_preds.append(out_pred)
        return out_edges, out_pairs, out_preds
    pred_orient_np = (pred_orient_np + np.transpose(pred_orient_np, (0, -1, -2)))/2
    L3, D = pred_edge_np.shape
    C, L1, L2 = pred_orient_np.shape
    assert C==3 and D==4 and L1==L2==L3
    L = len(seqs) if seqs else L1
    if L1>L:
        pred_edge_np = pred_edge_np[:L, :]
        pred_orient_np = pred_orient_np[:, :L, :L]

    pairs = []
    
    candidates = []
    orient_softmax = pred_orient_np.argmax(axis=0)
    for i in range(L):
        for j in range(i + 1, L):
            cls = orient_softmax[i, j]
            if cls>0:
                candidates.append((i, j, cls, pred_orient_np[cls, i, j]))
    
    candidates.sort(key=lambda x: x[-1], reverse=True)
    
    used = set()
    ## greedy search
    for i, j, cls, score in candidates:
        if i not in used and j not in used:
            pairs.append((i, j, cls))
            used.add(i)
            used.add(j)

    edges = [0]*L
    final_data = []
    ## process edge
    for i, j, cls in pairs:
        edges[i] = pred_edge_np[i, 1:].argmax() + 1
        edges[j] = pred_edge_np[j, 1:].argmax() + 1
        final_data.append((i, j, edge_index_dic[edges[i]], edge_index_dic[edges[j]], orient_index_dic[cls]))
    return edges, pairs, final_data


def extract_basepair_interaction_gt(gt_edge_np, gt_orient_np, seqs=None):
    '''
        gt_edge_np: (B)xL
        gt_orient_np: (B)xLxL
    '''
    shape = gt_edge_np.shape
    if len(shape)>2:
        raise Exception('Error format of gt_edge: {shape}, should be in shape of L or BxL')
    elif len(shape)==2:
        out_edges, out_pairs, out_gts = [], [], []
        for i in range(len(gt_edge_np)):
            out_edge, out_pair, out_gt = extract_basepair_interaction_gt(gt_edge_np[i], gt_orient_np[i], seqs[i] if seqs else None)
            out_edges.append(out_edge)
            out_pairs.append(out_pair)
            out_gts.append(out_gt)
        return out_edges, out_pairs, out_gts
    edges = [i for i in gt_edge_np]
    L1 = len(gt_edge_np)
    L2, L3 = gt_orient_np.shape
    assert L1==L2==L3
    L = len(seqs) if seqs else L1
    if L1>L:
        gt_edge_np = gt_edge_np[:L]
        gt_orient_np = gt_orient_np[:L, :L]
    pairs = []
    final_data = []
    for i in range(L):
        for j in range(i+1, L):
            cls = gt_orient_np[i, j]
            if cls>0:
                pairs.append((i, j, cls))

                final_data.append((i, j, edge_index_dic[edges[i]], edge_index_dic[edges[j]], orient_index_dic[cls]))
    return edges, pairs, final_data


if __name__ == '__main__':
    data_dir = 'data/PDB_download'
    json_path = 'data/NC_data.json'

    # df = prepare_dataset_onepiece(dest, data_dir, rerun=False)
    # print(df)

    prepare_dataset_RNAVIEW_json(json_path, data_dir)
    load_dataset_RNAVIEW(json_path)

    ## Example
    # pdb_path = os.path.join(data_dir, '3Q51.pdb')
    # info = parse_RNAVIEW_out(pdb_path+'.out')
    # print(info, len(info))
