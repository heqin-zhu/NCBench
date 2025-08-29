 # pip install structRFM
 # pip install rna-fm

import os

import torch

import fm
from structRFM.infer import structRFM_infer
from LM_models.RNABERT.rnabert import BertModel
# from LM_models.RNAMSM.model import MSATransformer
    

def get_LM_models(LM_checkpoint_dir, LM_list):
    dic = {}
    for LM_name in LM_list:
        if LM_name == 'structRFM':
            model = structRFM_infer(os.path.join(LM_checkpoint_dir, 'structRFM'), max_length=514)
            dic[LM_name] = model
        elif LM_name == 'RNAMSM':
            model_config = {
                      "vocab_size": 12,
                      "embed_dim": 768,
                      "num_attention_heads": 12,
                      "num_layers": 10,
                      "embed_positions_msa": "True",
                      "dropout": 0.1,
                      "attention_dropout": 0.1,
                      "activation_dropout": 0.1,
                      "max_tokens_per_msa": 16384,
                      "max_seqlen": 1024
                           }
            model = MSATransformer(**model_config)
            para_dic = torch.load(os.path.join(LM_checkpoint_dir, 'RNAMSM.pth'))
            print(para_dic.keys())
            model.load_state_dict(para_dic)
        elif LM_name == 'RNABERT':
            pass # TODO
        elif LM_name == 'RNAFM':
            model, alphabet = fm.pretrained.rna_fm_t12()
            model.eval()
            batch_converter = alphabet.get_batch_converter()
            dic[LM_name] = model, batch_converter
        else:
            raise Exception(f'Unknown LM name: {LM_name}')
    return dic


def get_LM_embedding(model, LM_name, name, seq, cache_dir='LM_embeddings', rerun=False):
    cache_dir = os.path.join(cache_dir, LM_name)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, name+'.pt')
    if os.path.exists(cache_path) and not rerun:
        return torch.load(cache_path)
    else:
        if LM_name == 'structRFM':
            feat =  model.extract_feature(seq)['seq_feat']
        elif LM_name == 'RNAFM':
            model, batch_converter = model
            data = [(name, seq)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[12])
            feat = results["representations"][12][0][1:-1]
        else:
            raise Exception(f'Unknown LM name: {LM_name}')
        torch.save(feat, cache_path)
        return feat


if __name__ == '__main__':
    LM_checkpoint_dir = 'LM_checkpoint'
    LM_list = ['structRFM', 'RNAFM', 'RNAMSM']
    LM_list = ['structRFM', 'RNAFM']
    model_dic = get_LM_models(LM_checkpoint_dir, LM_list)
    name = 'tmp'
    seq = 'AUGUAUGAUGCCCGU'
    LM_embed_dic = {}
    print(seq, len(seq))
    for LM_name in LM_list:
        if LM_name not in LM_embed_dic:
            LM_embed_dic[LM_name] = []
        feat = get_LM_embedding(model_dic[LM_name], LM_name, name, seq, cache_dir='tmp', rerun=True)
        LM_embed_dic[LM_name].append(feat)
        print(LM_name, feat.shape)
