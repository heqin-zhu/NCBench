from typing import ClassVar

import numpy as np
import torch
import torch.nn as nn

from .positional_embedding import *


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
                 kernel_size=7,
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


class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 positional_embedding:str='rope',
                 num_heads: int = None,
                # k_dim: int = None,
                # v_dim: int = None,
                 dropout: float = 0.10, 
                 bias: bool = True,
                 temperature: float = 1,
                 use_SE = False,
                ):
        super().__init__()
        
        assert positional_embedding in ("dyn", "alibi", 'rope')
        self.positional_embedding = positional_embedding
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads or 1
        self.head_size = hidden_dim//self.num_heads
        assert hidden_dim == self.head_size*self.num_heads, "hidden_dim must be divisible by num_heads"
        self.dropout = dropout
        self.bias = bias
        self.temperature = temperature
        self.use_SE = use_SE
        
        if self.positional_embedding == "dyn":
            self.dynpos = DynamicPositionBias(dim = hidden_dim//4, heads = num_heads, depth = 2)
        elif self.positional_embedding == "alibi":
            alibi_heads = num_heads // 2 + (num_heads % 2 == 1)
            self.alibi = AlibiPositionalBias(alibi_heads, self.num_heads)
            
        self.dropout_layer = nn.Dropout(dropout)
        self.weights = nn.Parameter(torch.empty(self.hidden_dim, 3 * self.hidden_dim)) # QKV
        self.out_w = nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim)) # QKV
        torch.nn.init.xavier_normal_(self.weights)
        torch.nn.init.xavier_normal_(self.out_w)

        if self.bias:
            self.out_bias = nn.Parameter(torch.empty(1, 1, self.hidden_dim)) # QKV
            self.in_bias = nn.Parameter(torch.empty(1, 1, 3*self.hidden_dim)) # QKV
            torch.nn.init.constant_(self.out_bias, 0.)
            torch.nn.init.constant_(self.in_bias, 0.)
        if not use_SE:
            self.gamma = nn.Parameter(torch.ones(self.num_heads).view(1, -1, 1, 1))

    def forward(self, x, adj=None, seq_mask=None, mat_mask=None, return_attn_weights=False):
        '''
            x: sequence feature: BxLxD
            adj: adj_matrix feature: Bx1xLxL
        '''
        b, l, h = x.shape
        x = x @ self.weights # b, l, 3*hidden
        if self.bias:
            x = x + self.in_bias
        Q, K, V = x.view(b, l, self.num_heads, -1).permute(0,2,1,3).chunk(3, dim=3) # b, a, l, head
        
        norm = self.head_size**0.5
        attention = (Q @ K.transpose(2,3)/self.temperature/norm)
        
        if self.positional_embedding == "dyn":
            i, j = map(lambda t: t.shape[-2], (Q, K))
            attn_bias = self.dynpos(i, j).unsqueeze(0)
            attention = attention + attn_bias
        elif self.positional_embedding == "alibi":
            i, j = map(lambda t: t.shape[-2], (Q, K))
            attn_bias = self.alibi(i, j).unsqueeze(0)
            attention = attention + attn_bias
        elif self.positional_embedding == "xpos":
            self.xpos = XPOS(self.head_size)

        if adj is not None:
            if not self.use_SE:
                adj = self.gamma * adj
            attention = attention + adj

        if mat_mask is not None:
            valid_mask = mat_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # (b, num_heads, l, l)
            fill_value = torch.finfo(attention.dtype).min
            attention = attention.masked_fill(~valid_mask, fill_value)

        attention = attention.softmax(dim=-1) # b, a, l, l, softmax won't change shape
        out = attention @ V  # b, a, l, head
        out = out.permute(0,2,1,3).flatten(2,3) # b, a, l, head -> b, l, (a, head) -> b, l, hidden
        # out_w is defined as Parameter, use matrix op instead of linear layer: (self.out_proj(out))
        out = out @ self.out_w  # linear transformation, fuse multi-head feature
        if self.bias:
            out = out + self.out_bias
        if return_attn_weights:
            return out, attention
        else:
            return out           


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 positional_embedding: str='alibi',
                 num_heads: int = None,
                 dropout: float = 0.10,
                 ffn_size: int = None,
                 activation: nn.Module = nn.GELU,
                 temperature: float = 1.,
                 use_SE = False,
                ):
        super().__init__()
        if num_heads is None:
            num_heads = 1
        if ffn_size is None:
            ffn_size = hidden_dim*4
        self.in_norm = nn.LayerNorm(hidden_dim)
        self.mhsa = MultiHeadSelfAttention(hidden_dim=hidden_dim,
                                           num_heads=num_heads,
                                           positional_embedding=positional_embedding,
                                           dropout=dropout,
                                           bias=True,
                                           temperature=temperature,
                                           use_SE=use_SE,
                                          )
        self.dropout_layer = nn.Dropout(dropout)
        self.ffn = MLP(hidden_dim, hidden_dim, ffn_size, dropout)
        

    def forward(self, x, adj=None, seq_mask=None, mat_mask=None, return_attn_weights = False):
        '''
            x: sequence feature: BxLXD
            adj: adj_matrix feature: Bx1xLxL
        '''
        x_in = x
        if return_attn_weights:
            x, attn_w = self.mhsa(self.in_norm(x), adj=adj, seq_mask=seq_mask, mat_mask=mat_mask, return_attn_weights=return_attn_weights)
        else:
            x = self.mhsa(self.in_norm(x), adj=adj, seq_mask=seq_mask, mat_mask=mat_mask, return_attn_weights = False)
        x = self.dropout_layer(x) + x_in
        x = self.ffn(x) + x

        if return_attn_weights:
            return x, attn_w
        else:
            return x
        

class FuseFormer(nn.Module):
    def __init__(self,
                 dim: int  = 256,
                 positional_embedding: str='alibi',
                 head_size: int = 32,
                 dropout: float = 0.10,
                 dim_feedforward: int = 192 * 4,
                 activation: nn.Module = nn.GELU,
                 temperature: float = 1.,
                 num_layers: int = 12,
                 num_adj_convs: int =2,
                 use_SE = False,
                 conv_in_chan=0,
                ):
        super().__init__()
        num_heads, rest = divmod(dim, head_size)
        assert rest == 0
        self.num_heads = num_heads
        
        self.layers = nn.Sequential(
            *[TransformerEncoderLayer(hidden_dim=dim,
                                     num_heads=num_heads,
                                     positional_embedding=positional_embedding,
                                     dropout=dropout,
                                     ffn_size=dim_feedforward,
                                     activation=activation,
                                     temperature=temperature,
                                     use_SE=use_SE,
                                    ) 
             for i in range(num_layers)]
        )
        self.conv_in_chan = conv_in_chan
        if conv_in_chan>0:
            self.conv_layers = nn.ModuleList()
            for i in range(num_layers):
                self.conv_layers.append(ResConv2dSimple(in_c=conv_in_chan if i == 0 else num_heads, out_c=num_heads, kernel_size=3, use_SE=use_SE))
            
            
    def forward(self, x, adj=None, seq_mask=None, mat_mask=None):
        '''
            x: sequence feature: BxLxD
            adj: adj_matrix feature: Bx1xLxL
        '''
        for ind, mod in enumerate(self.layers):
            if self.conv_in_chan>0:
                adj = self.conv_layers[ind](adj)
            x, adj = mod(x, adj=adj, seq_mask=seq_mask, mat_mask=mat_mask, return_attn_weights=True)
        return x, adj


class AttnMatFusion_net(nn.Module):
    def __init__(self,  
                 max_seq_len=512,
                 out_dim=4,
                 out_channels=3,
                 dim=256, 
                 depth=12,
                 positional_embedding: str='alibi',
                 head_size=32,
                 use_SE=True,
                 use_BPM=True,
                 LM_embed_dim=None,
                 *args,
                 **kargs,
                 ):
        super().__init__()
        conv_in_chan = 0
        if use_BPM:
            conv_in_chan = 2
        self.use_BPM = use_BPM
        ## token embed
        self.seq_input_embed = nn.Embedding(6, dim)

        ## LM embed
        self.LM_embed_dim = LM_embed_dim
        if LM_embed_dim:
            self.LM_linear = nn.Linear(LM_embed_dim, dim)
            self.fuse_LM_seq_linear = nn.Linear(2*dim, dim)

        ## positional embedding
        if positional_embedding=='rope':
            self.seq_rope = RoPE(dim=dim, max_seq_len=max_seq_len)
            self.mat_2d_rope = Matrix2DRoPE(dim=dim, max_seq_len=max_seq_len)

        self.positional_embedding=positional_embedding,
        self.transformer = FuseFormer(
            num_layers=depth,
            dim=dim,
            head_size=head_size,
            positional_embedding=positional_embedding,
            use_SE=use_SE,
            conv_in_chan=conv_in_chan,
        )

        self.final_seq_proj = MLP(dim, out_dim, dim//2)
        # self.final_mat_proj = MLP(dim, out_channels, dim//2)
        num_heads, rest = divmod(dim, head_size)
        self.final_mat_proj = ResConv2dSimple(num_heads, out_c=out_channels, kernel_size=3, use_SE=use_SE)

            
    def forward(self, seq, mat=None, seq_mask=None, mat_mask=None, LM_embed=None):
        # token embed
        seq = self.seq_input_embed(seq)
        ## fuse LM
        if self.LM_embed_dim:
            seq = self.fuse_LM_seq_linear(torch.cat([seq, self.LM_linear(LM_embed)], dim=-1))

        # positional embedding
        if self.positional_embedding == 'rope':
            seq = self.seq_rope(seq)
            if self.use_BPM:
                mat = self.mat_2d_rope(mat)
        seq_feat, mat_feat = self.transformer(seq, mat, seq_mask, mat_mask)

        edge = self.final_seq_proj(seq_feat)
        orient = self.final_mat_proj(mat_feat)
        # orient = torch.einsum('blh,bmh->bhlm', seq_feat, seq_feat)
        return edge, orient


if __name__ == '__main__':
    torch.manual_seed(42)
    kargs = {
           'depth': 12,
           'dim': 256,
           'head_size': 32,
           'positional_embedding': 'alibi', # 'dyn', # alibi
           'use_SE': True,
           'out_dim': 4,
           'out_channels': 3,
           'max_seq_len': 512,
              }
    B, L = 5, 513
    # Create random inputs
    seq = torch.randint(0, 6, (B, L)).long()
    mat = torch.randn(B, 2, L, L)
    mat_mask = torch.ones(B, L, L, dtype=bool)
    
    # Initialize model
    model = AttnMatFusion_net(**kargs)
    # Forward pass
    out_seq, out_mat = model(seq, mat, seq_mask=None, mat_mask=mat_mask)
    
    # Check output shapes
    print(f"Input sequence shape: {seq.shape}")
    print(f"Input matrix shape: {mat.shape}")
    print(f"Output sequence shape: {out_seq.shape}")  # Should be BxLxhidden_dim
    print(f"Output matrix shape: {out_mat.shape}")    # Should be Bxhidden_dimxLxL
    
    # Verify output shapes
    assert out_seq.shape == (B, L, kargs['out_dim']), f"Sequence shape mismatch: {out_seq.shape} vs {(B, L, kargs['out_dim'])}"
    assert out_mat.shape == (B, kargs['out_channels'], L, L), f"Matrix shape mismatch: {out_mat.shape} vs {(B, kargs['out_channels'], L, L)}"
    print("All tests passed!")
    import sys
    sys.path.append('../util')
    from NCfold_kit import count_para
    count_para(model)
