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


def download_PDB_RNA3DB(dest='PDB_RNA3DB', rna3db_split='rna3db-jsons/split.json', fmt='cif'):
    os.makedirs(dest, exist_ok=True)
    DOWNLOAD_CMD = 'wget https://files.rcsb.org/download/{name}.{fmt} -P {dest}'
    sufs = ['.pdb', '.cif']

    with open(rna3db_split) as fp:
        split_data = json.loads(fp.read())
    names = set()
    for k, v in split_data.items():
        num_chain = 0
        comp_names = set()
        for comp, d in v.items():
            for name_chain in d:
                name = name_chain.split('_')[0]
                comp_names.add(name.upper())
            num_chain += len(d)
        names = names.union(comp_names)
        print(f'{k}: components={len(v)}, comp_names={len(comp_names)}, chains={num_chain}')
    missing_names = []

    for name in names:
        for suf in sufs:
            path = os.path.join(dest, name+suf)
            if os.path.exists(path):
                break
        else:
            cmd = DOWNLOAD_CMD.format(name=name, dest=dest, fmt=fmt)
            ret = os.system(cmd)
            if ret:
                print(name, ret)
                missing_names.append(name)
    print('total names', len(names))
    print('missing    ', len(missing_names))


def get_seq_and_SS_from_PDB_by_onepiece(pdb_path):
    tmp_dir = 'tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, 'tmp_op.ss')
    op_src = '~/gitrepo/RNA/RNA3d/onePiece/src'
    os.system(f'java -cp {op_src} Zhu_onepiece {pdb_path} {tmp_file} > log_op 2>&1')
    with open(tmp_file) as fp:
        head, seq, ss = fp.read().split('\n')
        return seq, ss


def get_seq_from_xml_by_RNAVIEW(path):
    # deprecated: no chainID info
    pattern = r'<seq-data>\s+(?P<seq>[A-Za-z ]+)\s+</seq-data>'
    chains = []
    with open(path) as fp:
        text = fp.read()
        for f in re.findall(pattern, text):
            chain = f.replace(' ', '')
            chains.append(chain)
    return chains


def get_seq_from_torsion_by_RNAVIEW(path, separate_chain=True):
    separate_chain_dic = {}
    single_chain_seq = []
    with open(path) as fp:
        lines = list(fp.readlines())
        i = 0
        n = len(lines)
        while i<n:
            if lines[i].startswith('Ch Res'):
                break
            i+=1
        for idx in range(i+1, n):
            parts = [part for part in lines[idx].split(' ') if part]
            if len(parts)>=3:
                chain, base, num = parts[:3]
                num = int(num)
                if chain not in separate_chain_dic:
                    separate_chain_dic[chain] = {'seq': [], 'idx_map': {}}
                separate_chain_dic[chain]['seq'].append(base)
                separate_chain_dic[chain]['idx_map'][num] = len(separate_chain_dic[chain]['seq'])
                single_chain_seq.append(base)
    separate_chain_dic = {chain: {'seq': ''.join(dic['seq']), 'idx_map': dic['idx_map']} for chain, dic in separate_chain_dic.items()}
    if separate_chain:
        return separate_chain_dic
    else:
        return ''.join(single_chain_seq)


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
    fmt = pdb_path[pdb_path.rfind('.')+1:]
    assert fmt in ['pdb', 'cif']
    flag = '' if fmt == 'pdb' else '--cif'
    CMD = f'rnaview {flag} {pdb_path}'
    p = subprocess.Popen(CMD, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # text = p.stdout.read()
    # p.stdout.close()
    return p


def parse_RNAVIEW_out(file_path, save_info=False, run_failed=True, rerun=False, chain_idx_map={}):
    '''
        file_path: *.pdb.out, after run `RNAVIEW *.pdb`
        gen: *.pdb.json
    '''
    if not os.path.exists(file_path):
        if run_failed:
            p = run_RNAVIEW(file_path[:-4])
            p.join()
        else:
            return {}
    name = get_file_name(file_path)
    dest = os.path.dirname(file_path)
    dest_path = os.path.join(dest, name+'.json')
    if os.path.exists(dest_path) and not rerun:
        with open(dest_path) as fp:
            info = json.load(fp)
            return info
    else:
        pattern = r'\s+(?P<global_left>\d+)_(?P<global_right>\d+), (?P<left_chain>.*?):\s+(?P<left>\d+) (?P<pair>[augctAUGCNT]\-[augctAUGCNT])\s+(?P<right>\d+) (?P<right_chain>.*?): (?P<edge_type>[\+\-WHS]/[\+\-WHS]|\s+) +(?P<orientation>cis|tran|stacked).*?(?P<class_id>[IVX]+|n/a|!.*?\(.*?\)|)\n'

        length_pattern = r'The total base pairs =\s+\d+ \(from\s+(?P<length>\d+) bases\)'
        with open(file_path) as fp:
            text = fp.read()
        length = None
        for L in re.findall(length_pattern, text):
            length = int(L)
        info = {}
        for global_left, global_right, left_chain, left, pair, right, right_chain, edge_type, orientation, class_id in re.findall(pattern, text):
            if chain_idx_map: # separate_chain
                if left_chain!=right_chain: # pair in different chains, ignored
                    continue
                left, right = int(left), int(right)
                left = chain_idx_map[left_chain].get(left, left)
                right = chain_idx_map[right_chain].get(right, right)
                cur_chain = left_chain
            else:
                left = int(global_left)
                right = int(global_right)
                cur_chain = 'single'
            if cur_chain not in info:
                info[cur_chain] = []
            info[cur_chain].append({
                         'left': left,
                         'right': right,
                         'edge_type': edge_type,
                         'orientation': orientation,
                         'pair': pair,
                         'class_id': class_id,
                        })
        if len(info)!=0 and save_info:
            with open(dest_path, 'w') as fp:
                json.dump(info, fp)
        return info


def cif2pdb(src, dest):
    from Bio.PDB import MMCIFParser, PDBIO
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('structure', src)
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


def prepare_dataset_RNAVIEW_json(dest, pdb_dir, filter_fasta=None, separate_chain=True, rerun=True):
    if not dest.endswith('.json'):
        dest = dest+'.json'
    if not os.path.exists(dest) or rerun:
        data = []
        sufs = ['.pdb', '.cif']
        pdb_names = [f[:f.rfind('.')] for f in os.listdir(pdb_dir) if any(f.endswith(suf) for suf in sufs) and len(f)==8]
        for pdb_name in pdb_names:
            if all(not os.path.exists(os.path.join(pdb_dir, pdb_name+suf+'_torsion.out')) for suf in sufs):
                data_path = os.path.join(pdb_dir, pdb_name+'.cif')
                print('run rnaview', data_path)
                run_RNAVIEW(data_path)
        # ps = []
        # for pdb_path in tqdm(pdb_paths):
        #     ps.append(run_RNAVIEW(pdb_path))
        # for p in ps:
        #     if p is not None:
        #         p.wait()
        filter_names = None
        if filter_fasta:
            filter_names = [name for name, seq in read_fasta(filter_fasta)]
        failed_names = set()
        for pdb_name in tqdm(pdb_names):
            pdb_path = os.path.join(pdb_dir, pdb_name+'.cif')
            name = get_file_name(pdb_path)
            if (filter_names and name in filter_names) or not filter_names:
                torsion_path = pdb_path+'_torsion.out'
                out_path = pdb_path+'.out'
                if all(os.path.exists(path) and os.path.getsize(path)>0 for path in [torsion_path, out_path]):
                    chain_seq = get_seq_from_torsion_by_RNAVIEW(torsion_path, separate_chain=separate_chain)
                    chain_idx_map = {k: dic['idx_map'] for k, dic in chain_seq.items()} if separate_chain else {}
                    chain_info_dic = parse_RNAVIEW_out(out_path, save_info=True, rerun=rerun, chain_idx_map=chain_idx_map)
                    if separate_chain:
                        for chain, info in chain_info_dic.items():
                            if len(info)==0:
                                print('no info', name, chain)
                            elif chain in chain_seq:
                                data.append({
                                              'name': f'{name}_{chain}',
                                              'seq': chain_seq[chain]['seq'],
                                              'pair_info': info,
                                             })
                            else:
                                print(f'chain {chain}, found no seq, {chain_seq_dic.keys()}')
                                failed_names.add(f'{name}_{chain}')
                    else:
                        info_list = []
                        for chain, info in chain_info_dic.items():
                            info_list += info
                        data.append({
                                      'name': f'{name}',
                                      'seq': chain_seq,
                                      'pair_info': info_list,
                                     })
                else:
                    failed_names.add(name)
        with open(dest, 'w') as fp:
            json.dump(data, fp)
        with open('failed.json', 'w') as fp:
            json.dump(list(failed_names), fp)


def construct_RNAVIEW_labels(data_dic, include_canonical=False):
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
            print(data_dic["name"], left, right, pair, len(seq))
            # print(f'[Error] Bases dismatch, {pair}, seq[{left}]={seq[left]}, seq[{right}]={seq[right]}\nseq={seq}, {data_dic["name"]}')
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
    error_label = error_seq = error_dlabel = 0
    for dic in json_data:
        name = dic['name']
        seq = dic['seq']
        d = construct_RNAVIEW_labels(dic, include_canonical=include_canonical)
        if d is None or len(seq)==0 or len(d['labels'])==0:
            if d is None:
                error_label +=1
            else:
                if len(d['labels'])==0:
                    error_dlabel +=1
            if len(seq)==0:
                error_seq +=1
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
            # 1xL edge array: 0 for Watson-crick, 1 for Hoogsteen, 2 for Sugar.
            # print(labels['pair']['edge_arr']) # L, 
            # LxL orient matrix: 0 for no-pair, 1 for trans, 2 for cis.
            # print(labels['pair']['orient_mat'])
    print(error_label, error_seq, error_dlabel)
    with open(index_dest, 'w') as fp:
        json.dump(index_data, fp)
    if filter_fasta is not None:
        names = {name for name, seq in read_fasta(filter_fasta)}
        data_list = [d for d in data_list if d['name'] in names]

    print(f'Processing {data_path}: valid={total_ct}, invalid={error_ct}, min_len={np.min(lengths)}, max_len={np.max(lengths)}', end='')
    if max_seq_len is not None:
        data_list = [d for d in data_list if len(d['seq'])<=max_seq_len]
        print(f', len<={max_seq_len}: {len([l for l in lengths if l<=max_seq_len])}')
    else:
        print()
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
    # download_PDB_RNA3DB(dest='data/PDB_RNA3DB', rna3db_split='data/rna3db_split.json', fmt='cif')

    data_dir = 'data/PDB_download'
    data_dir = 'data/PDB_RNA3DB'
    json_path = 'data/NC_data.json'

    # df = prepare_dataset_onepiece(dest, data_dir, rerun=False)
    # print(df)

    print('Parsing RNAVIEW output and saving as json...')
    prepare_dataset_RNAVIEW_json(json_path, data_dir, separate_chain=False, rerun=rerun)
    print('Convert json to labels...')
    load_dataset_RNAVIEW(json_path, include_canonical=True)
