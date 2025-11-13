from typing import List, Dict 

import numpy as np
from tomlkit import string
import torch
import torch.nn as nn

# from ..RNAFoldAssess.RNAFoldAssess.models.predictors.contrafold import ContraFold
# from ..RNAFoldAssess.RNAFoldAssess.models.predictors.mxfold2 import MXFold2

from .positional_embedding import *
from util.NCfold_kit import ids_to_strings

from .RNAFoldAssess.models.predictors.contrafold import ContraFold
from .RNAFoldAssess.models.predictors.mxfold2 import MXFold2
from .RNAFoldAssess.models.predictors.rna_fold import RNAFold

class SE_Block(nn.Module):
    def __init__(self, c, r=1):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)          
        

class ResConv2dSimple(nn.Module):
    def __init__(self, 
                 in_c, 
                 out_c,
                 kernel_size=3,
                 use_SE = False,
                ):  
        super().__init__()
        if use_SE:
            self.conv = nn.Sequential(
                # b c w h
                nn.Conv2d(in_c,
                          out_c, 
                          kernel_size=kernel_size, 
                          padding="same", 
                          bias=False),
                # b w h c#
                nn.BatchNorm2d(out_c),
                SE_Block(out_c),
                nn.GELU(),
                # b c e 
            )
            
        else:
            self.conv = nn.Sequential(
                # b c w h
                nn.Conv2d(in_c,
                          out_c, 
                          kernel_size=kernel_size, 
                          padding="same", 
                          bias=False),
                # b w h c#
                nn.BatchNorm2d(out_c),
                nn.GELU(),
                # b c e 
            )
        
        if in_c == out_c:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(
                nn.Conv2d(in_c,
                          out_c, 
                          kernel_size=1, 
                          bias=False)
            )

    def forward(self, x):
        # b e s 
        h = self.conv(x)
        x = self.res(x) + h
        return x
    

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, mid_dim=None, dropout=0, activation=nn.GELU):
        super(MLP, self).__init__()
        mid_dim = mid_dim or out_dim
        self.layers = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, mid_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, out_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.layers(x)
 

class RFM_net(nn.Module):
    def __init__(self,  
                 max_seq_len=512,
                 out_dim=4,
                 out_channels=3,
                 dim=256, 
                 use_SE=True,
                 use_BPM=True,
                 LM_embed_chan=None,
                 *args,
                 **kargs,
                 ):
        super().__init__()
        self.conv_in_chan = 0
        if LM_embed_chan:
            self.conv_in_chan += LM_embed_chan
        ## LM embed # channel
        self.LM_embed_chan = LM_embed_chan

        ## token embed
        self.seq_input_embed = nn.Embedding(6, dim)

        self.final_seq_proj = MLP(dim, out_dim, dim//2)
        self.final_mat_proj = ResConv2dSimple(1, out_c=out_channels, kernel_size=3, use_SE=use_SE)

            
    def forward(self, seq, LM_embed=None, seq_mask=None, mat_mask=None):
        # token embed
        seq = self.seq_input_embed(seq)
        edge = self.final_seq_proj(seq)
        if LM_embed is not None:
            orient = self.final_mat_proj(LM_embed)
        else:
            orient = self.final_mat_proj(torch.einsum('blh,bmh->blm', seq, seq).unsqueeze(1))
            
        orient = (orient + orient.transpose(-1, -2))/2
        
        return edge, orient


def dbn_to_partner_map(dbn: str, bracket_pairs: str = "()[]{}<>") -> Dict[int, int]:
    """
    Parse a dot-bracket string into a symmetric partner map i -> j (0-based).
    Supports multiple bracket types, e.g., () [] {} <> for pseudoknots.
    """
    assert len(bracket_pairs) % 2 == 0, "bracket_pairs length must be even."
    partner: Dict[int, int] = {}
    # one stack per bracket type
    stacks = {(bracket_pairs[i], bracket_pairs[i+1]): [] for i in range(0, len(bracket_pairs), 2)}

    for i, ch in enumerate(dbn.strip()):
        for op, cl in stacks.keys():
            if ch == op:               # opening bracket
                stacks[(op, cl)].append(i)
                break
            elif ch == cl:             # closing bracket
                st = stacks[(op, cl)]
                if st:
                    j = st.pop()
                    partner[i] = j
                    partner[j] = i
                break
        # dots/others are ignored
    return partner

# ------------------------ single sequence -> torch tensor ------------------------

def structure_to_mat01_torch(
    sequence: str,
    dbn: str,
    *,
    bracket_pairs: str = "()[]{}<>",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert (sequence, dot-bracket) to a binary contact matrix as torch.Tensor.
    Returns:
        mat01: tensor with shape [1, L, L], where 1.0 means i is paired with j.
            The matrix is symmetric and has zero diagonal.
    """
    L = len(dbn)
    mat01 = torch.zeros((1, L, L), device=device, dtype=dtype)
    partner = dbn_to_partner_map(dbn, bracket_pairs=bracket_pairs)

    # set symmetric ones for each paired (i, j)
    for i, j in partner.items():
        if i < j:
            mat01[0, i, j] = 1.0
            mat01[0, j, i] = 1.0

    # ensure zero diagonal (operate on the [L,L] slice)
    mat01[0].fill_diagonal_(0.0)
    return mat01  # [1, L, L]

# ------------------------ batch with padding -> [B,1,L,L] ------------------------

def batch_structure_to_mat01_torch(
    sequences: List[str],
    dbns: List[str],
    seq_mask,  # torch.BoolTensor or numpy array with shape [B, L]
    *,
    bracket_pairs: str = "()[]{}<>",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert a batch of (sequence, dot-bracket) to a padded binary contact tensor.

    Args:
        sequences: list of RNA strings (only used for length alignment; not required for parsing)
        dbns:      list of dot-bracket strings, same order as sequences
        seq_mask:  [B, L] mask; each row's sum is the effective length Lb
                (torch.BoolTensor or numpy array)
    Returns:
        mat01_B: torch tensor of shape [B, 1, L, L]
    """
    if not torch.is_tensor(seq_mask):
        seq_mask = torch.as_tensor(seq_mask, dtype=torch.bool)
    else:
        seq_mask = seq_mask.bool()

    B, L = seq_mask.shape
    mat01_B = torch.zeros((B, 1, L, L), device=device, dtype=dtype)

    for b in range(B):
        Lb = int(seq_mask[b].sum().item())
        if Lb <= 0:
            continue
        dbn_b = dbns[b][:Lb]
        # build [1, Lb, Lb] then pad into [1, L, L]
        m01 = structure_to_mat01_torch(sequences[b][:Lb], dbn_b,
                                    bracket_pairs=bracket_pairs,
                                    device=device, dtype=dtype)  # [1, Lb, Lb]
        mat01_B[b, 0, :Lb, :Lb] = m01[0]

    return mat01_B  # [B, 1, L, L]
    
    
class SFM_net(nn.Module):
    def __init__(self,  
                 model_name,
                 max_seq_len=512,
                 out_dim=4,
                 out_channels=3,
                 dim=256, 
                 use_BPM=True,
                 *args,
                 **kargs,
                 ):
        super().__init__()
        
        self.base2idx = {'A': 0, 'U':1, 'G':2, 'C':3, 'N':4, 'PAD':5}
        self.model_name = model_name
        if model_name == 'BPfold':
            pass # no predictor 
        elif model_name == 'RNAFold':
            self.secondary_predictor = RNAFold() 
        elif model_name == 'MxFold2':
            self.secondary_predictor = MXFold2()
        elif model_name == 'ContraFold':
            self.secondary_predictor = ContraFold() 
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        self.conv_in_chan = 1
        if use_BPM:
            self.conv_in_chan = 2
        ## token embed
        self.seq_input_embed = nn.Embedding(6, dim)

        self.final_seq_proj = MLP(dim, out_dim, dim//2)
        self.final_mat_proj = ResConv2dSimple(self.conv_in_chan, out_c=out_channels, kernel_size=3, use_SE=False)

    
    # 3) Convert dot-bracket to mat1 (torch) with padding to [B, 1, L, L]
    def dbn_to_partner_map(self, dbn: str, bracket_pairs: str = "()[]{}<>"):
        partner = {}
        stacks = {(bracket_pairs[i], bracket_pairs[i+1]): [] for i in range(0, len(bracket_pairs), 2)}
        for i, ch in enumerate(dbn.strip()):
            for op, cl in stacks:
                if ch == op:
                    stacks[(op, cl)].append(i); break
                elif ch == cl:
                    st = stacks[(op, cl)]
                    if st: j = st.pop(); partner[i]=j; partner[j]=i
                    break
        return partner

    def structure_to_mat01_torch_from_dbn(self, dbn: str, device="cpu", dtype=torch.float32):
        L = len(dbn)
        M = torch.zeros((1, L, L), device=device, dtype=dtype)
        for i, j in dbn_to_partner_map(dbn).items():
            if i < j:
                M[0, i, j] = 1.0; M[0, j, i] = 1.0
        M[0].fill_diagonal_(0.0)
        return M

    def batch_structure_to_mat01_torch_from_dbn(self, dbns: List[str], seq_mask, device="cpu", dtype=torch.float32):
        if not torch.is_tensor(seq_mask):
            seq_mask = torch.as_tensor(seq_mask, dtype=torch.bool)
        B, L = seq_mask.shape
        out = torch.zeros((B, 1, L, L), device=device, dtype=dtype)
        for b in range(B):
            Lb = int(seq_mask[b].sum().item())
            if Lb > 0:
                M = self.structure_to_mat01_torch_from_dbn(dbns[b][:Lb], device=device, dtype=dtype)  # [1,Lb,Lb]
                out[b, 0, :Lb, :Lb] = M[0]
        return out

    def forward(self, input_ids, matrix, seq_mask, mat_mask):
        
        if self.model_name == 'BPfold':
            mat = matrix
        else:
            print(input_ids)
            
            seqs = ids_to_strings(input_ids, seq_mask, self.base2idx)
        
            # obtain the sequence
            dbns = []
            for s in seqs:
                self.secondary_predictor.execute(seq_file=s)
                secondary_structure = self.secondary_predictor.get_ss_prediction()
                
                print(secondary_structure)
                dbns.append(secondary_structure)                
                
            mat = self.batch_structure_to_mat01_torch_from_dbn(dbns, seq_mask, device=input_ids.device, dtype=torch.float32)
         
        seq_emb = self.seq_input_embed(input_ids)
        edge = self.final_seq_proj(seq_emb)
        orient = self.final_mat_proj(mat)
        orient = (orient + orient.transpose(-1, -2))/2
        return edge, orient
