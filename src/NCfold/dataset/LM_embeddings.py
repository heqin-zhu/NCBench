import os

import torch
from structRFM.infer import structRFM_infer
from multimolecule import RnaTokenizer, AidoRnaModel, ErnieRnaModel, RiNALMoModel, RnaBertModel, RnaErnieModel, RnaFmModel, RnaMsmModel, SpliceBertModel, UtrBertModel, UtrLmModel


multimolecule_LMs = ['aido.rna-650m', 'aido.rna-1.6b', 'ernierna', 'ernierna-ss', 'rinalmo-giga', 'rinalmo-mega', 'rinalmo-micro', 'rnabert', 'rnaernie', 'rnafm', 'mrnafm', 'rnamsm', 'splicebert', 'splicebert.510', 'splicebert-human.510', 'utrbert-3mer', 'utrbert-4mer', 'utrbert-5mer', 'utrbert-6mer', 'utrlm-te_el', 'utrlm-mrl']

multimolecule_LM_classes = {
         'aido.rna-650m': AidoRnaModel, 
         'aido.rna-1.6b': AidoRnaModel, 
         'ernierna': ErnieRnaModel, 
         'ernierna-ss': ErnieRnaModel, 
         'rinalmo-giga': RiNALMoModel, 
         'rinalmo-mega': RiNALMoModel, 
         'rinalmo-micro': RiNALMoModel, 
         'rnabert': RnaBertModel, 
         'rnaernie': RnaErnieModel, 
         'rnafm': RnaFmModel, 
         'mrnafm': RnaFmModel, 
         'rnamsm': RnaMsmModel, 
         'splicebert': SpliceBertModel, 
         'splicebert.510': SpliceBertModel, 
         'splicebert-human.510': SpliceBertModel, 
         'utrbert-3mer': UtrBertModel, 
         'utrbert-4mer': UtrBertModel, 
         'utrbert-5mer': UtrBertModel, 
         'utrbert-6mer': UtrBertModel, 
         'utrlm-te_el': UtrLmModel, 
         'utrlm-mrl': UtrLmModel,
        
        }

LM_dim_dic = {
         'structRFM': 768,
         'aido.rna-650m': 1280, 
         'aido.rna-1.6b': 2048, 
         'ernierna': 768, 
         'ernierna-ss': 768, 
         'rinalmo-giga': 1280, 
         'rinalmo-mega': 640, 
         'rinalmo-micro': 480, 
         'rnabert': 120, 
         'rnaernie': 768, 
         'rnafm': 640, 
         'mrnafm': 1280, 
         'rnamsm': 768, 
         'splicebert': 512, 
         'splicebert.510': 512, 
         'splicebert-human.510': 512, 
         'utrbert-3mer': 768, 
         'utrbert-4mer': 768, 
         'utrbert-5mer': 768, 
         'utrbert-6mer': 768, 
         'utrlm-te_el': 128, 
         'utrlm-mrl': 128,
        }

MAX_SEQ_LEN_DIC = {
               'rnabert': 438,
              }


def get_LM_models(LM_checkpoint_dir, LM_list):
    dic = {}
    for LM_name in LM_list:
        if LM_name == 'structRFM':
            model = structRFM_infer(os.path.join(LM_checkpoint_dir, 'structRFM'), max_length=514)
            dic[LM_name] = model
        elif LM_name.lower() in multimolecule_LMs:
            tokenizer = RnaTokenizer.from_pretrained(f"multimolecule/{LM_name.lower()}")
            model = multimolecule_LM_classes[LM_name.lower()].from_pretrained(f"multimolecule/{LM_name.lower()}")
            dic[LM_name] = model, tokenizer
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
        # elif LM_name == 'RNAFM':
        #     model, batch_converter = model
        #     data = [(name, seq)]
        #     batch_labels, batch_strs, batch_tokens = batch_converter(data)
        #     with torch.no_grad():
        #         results = model(batch_tokens, repr_layers=[12])
        #     feat = results["representations"][12][0][1:-1]
        elif LM_name.lower() in multimolecule_LMs:
            model, tokenizer = model
            if LM_name.lower() in MAX_SEQ_LEN_DIC:
                max_len = MAX_SEQ_LEN_DIC[LM_name.lower()]
                if len(seq)>max_len:
                    seq = seq[:max_len]
            inputs = tokenizer(seq, return_tensors='pt')
            output = model(**inputs, output_hidden_states=True)
            feat = output.hidden_states[-1][0, 1:-1,:]
        else:
            raise Exception(f'Unknown LM name: {LM_name}')
        torch.save(feat, cache_path)
        return feat


if __name__ == '__main__':
    LM_list = ['structRFM', 'RNABERT']
    LM_checkpoint_dir = 'LM_checkpoint'
    model_dic = get_LM_models(LM_checkpoint_dir, LM_list)
    name = 'tmp'
    cache_dir = 'tmp'
    seq = 'AUGUAUGAUGCCCGU'
    LM_embed_dic = {}
    print(seq, len(seq))
    for LM_name in LM_list:
        if LM_name not in LM_embed_dic:
            LM_embed_dic[LM_name] = []
        feat = get_LM_embedding(model_dic[LM_name], LM_name, name, seq, cache_dir=cache_dir, rerun=True)
        LM_embed_dic[LM_name].append(feat)
        print(LM_name, feat.shape)
