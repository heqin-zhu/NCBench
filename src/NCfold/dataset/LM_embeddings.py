import os

import torch

from structRFM.infer import structRFM_infer
    

def get_LM_models(LM_checkpoint_dir, LM_list):
    dic = {}
    for LM_name in LM_list:
        if LM_name == 'structRFM':
            model = structRFM_infer(os.path.join(LM_checkpoint_dir, 'structRFM'), max_length=514)
            dic['structRFM'] = model
        else:
            raise Exception(f'Unknown LM name: {LM_name}')
    return dic


def get_LM_embedding(model, LM_name, name, seq, cache_dir='LM_embeddings'):
    cache_dir = os.path.join(cache_dir, LM_name)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, name+'.pt')
    if os.path.exists(cache_path):
        return torch.load(cache_path)
    else:
        if LM_name == 'structRFM':
            feat =  model.extract_feature(seq)['seq_feat']
        else:
            raise Exception(f'Unknown LM name: {LM_name}')
        torch.save(feat, cache_path)
        return feat


if __name__ == '__main__':
    LM_checkpoint_dir = 'LM_checkpoint'
    LM_list = ['structRFM', 'RNA-FM']
    LM_list = ['structRFM']
    model_dic = get_LM_models(LM_checkpoint_dir, LM_list)
    name = 'tmp'
    seq = 'AUGUAUGAUGCCCGU'
    LM_embed_dic = {}
    print(seq, len(seq))
    for LM_name in LM_list:
        if LM_name not in LM_embed_dic:
            LM_embed_dic[LM_name] = []
        feat = get_LM_embedding(model_dic[LM_name], LM_name, name, seq, cache_dir='tmp')
        LM_embed_dic[LM_name].append(feat)
        print(LM_name, feat.shape)
