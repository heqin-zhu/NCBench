import json
import pickle

import numpy as np


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


def prepare_dataset_RNAVIEW_pickle(dest, data_path):
    with open(data_path) as fp:
        json_data = json.load(fp)
    index_dest = dest[:dest.rfind('.')]+'_index.json'
    total_ct = 0
    error_ct = 0
    save_data = {}
    index_data = []
    for dic in json_data:
        name = dic['name']
        seq = dic['seq']
        d = construct_RNAVIEW_labels(dic)
        if d is None:
            error_ct+=1
        else:
            labels = d['labels']
            pair_types = d['pair_types']
            total_ct+=1
            save_data[name] = {
                                 'seq': seq,
                                 'labels': labels,
                                }
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
    print(f'valid={total_ct}, Bases dismatch={error_ct}')
    if not dest.endswith('.pkl') and not dest.endswith('.pickle'):
        dest = dest[:dest.rfind('.')]+'.pickle'
    with open(dest, 'wb') as fp:
        pickle.dump(save_data, fp)


if __name__ == '__main__':
    data_path = 'data/NC_data.json'
    dest = 'data/NC_data.pickle'
    prepare_dataset_RNAVIEW_pickle(dest, data_path)
