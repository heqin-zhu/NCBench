import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .positional_embedding import *


class SeqBackbone(nn.Module):
    """Backbone for processing RNA sequence features"""
    def __init__(self, input_dim, hidden_dim, num_layers=2, nhead=4):
        super(SeqBackbone, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Transformer encoder for sequence modeling
        encoder_layers = TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.nhead = nhead
        self.transformer = TransformerEncoder(encoder_layers, num_layers=num_layers)
        
    def forward(self, x, attention_mask=None):
        # x: BxLxD
        B, L, D = x.shape
        x = self.input_proj(x)
        x = self.layer_norm(x)
        x = self.transformer(x)
        return x


class MatBackbone(nn.Module):
    """Backbone for processing pairwise matrix features"""
    def __init__(self, input_channels, hidden_channels, num_layers=2):
        super(MatBackbone, self).__init__()
        layers = []
        
        # Initial projection
        layers.append(nn.Conv2d(input_channels, hidden_channels, kernel_size=1))
        layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.GELU())
        
        # Convolutional layers
        for _ in range(num_layers):
            layers.extend([
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.GELU(),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
                nn.BatchNorm2d(hidden_channels),
                nn.GELU()
            ])
        
        self.conv_layers = nn.Sequential(*layers)
        # self.conv_layers = nn.Sequential(
        #     nn.Conv2d(mat_channels, self.hidden_channels, kernel_size=1),
        #     nn.BatchNorm2d(self.hidden_channels),
        #     nn.GELU(),
        #     nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(self.hidden_channels),
        #     nn.GELU()
        # )
        
        # Residual connection
        self.residual_proj = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)
        
    def forward(self, x, attention_mask=None):
        # x: BxCxLxL
        residual = self.residual_proj(x)
        x = self.conv_layers(x)
        x = x + residual
        if attention_mask is not None:
            x *= attention_mask
        return x


class SeqMatFusion(nn.Module):
    """
    Seq-matrix fusion block
    Integrated fusion block containing both feature extraction and fusion capabilities
    Can be used standalone or in cascaded configurations
    """
    def __init__(self, seq_dim, mat_channels, hidden_dim=None, hidden_channels=None):
        super(SeqMatFusion, self).__init__()
        self.hidden_dim = hidden_dim or seq_dim
        self.hidden_channels = hidden_channels or mat_channels

        self.seq_extractor = SeqBackbone(seq_dim, hidden_dim)
        self.mat_extractor = MatBackbone(mat_channels, hidden_channels)
        
        # Fusion attention mechanism
        in_chan = self.hidden_channels*2
        self.fusion_attn = nn.Sequential(
            nn.Linear(in_chan, in_chan // 2),
            nn.GELU(),
            nn.Linear(in_chan // 2, 2),
        )
        
        # Output projection layers
        self.seq_output_proj = nn.Linear(self.hidden_dim+self.hidden_channels, self.hidden_dim)
        self.mat_output_proj = nn.Conv2d(self.hidden_channels+self.hidden_dim, self.hidden_channels, kernel_size=1)
        
    def forward(self, seq, mat, seq_mask=None, mat_mask=None):
        # Input shapes: seq (BxLxD), mat (BxCxLxL)
        B, L, D = seq.shape
        
        # Feature extraction with residual connections
        seq_feat = self.seq_extractor(seq, seq_mask)
        mat_feat = self.mat_extractor(mat, mat_mask)
        
        # ---------------------------
        # Matrix-to-sequence fusion
        # ---------------------------
        # Matrix feature projection and pooling
        mat_seq_i = torch.mean(mat_feat, dim=-2).transpose(1, 2)  # BxHxLxL -> BxHxL -> BxLxH
        mat_seq_j = torch.mean(mat_feat, dim=-1).transpose(1, 2)  # BxHxLxL -> BxHxL -> BxLxH
        attn_input = torch.cat([mat_seq_i, mat_seq_j], dim=-1)  # BxLx(2H)
        attn_weight = self.fusion_attn(attn_input)  # BxLx2
        if mat_mask is not None:
            attn_weight = attn_weight.masked_fill(seq_mask.unsqueeze(-1).repeat(1, 1, 2)==0, float('-inf'))
        attn_weight = attn_weight.softmax(dim=-1)
        mat_seq = mat_seq_i * attn_weight[...,0:1] + mat_seq_j * attn_weight[...,1:2]

        fused_seq = torch.cat([seq_feat, mat_seq], dim=-1)
        out_seq = self.seq_output_proj(fused_seq)
        
        # ---------------------------
        # Sequence-to-matrix fusion
        # ---------------------------
        # feat1 = seq_feat.unsqueeze(-2).unsqueeze(-1)
        # feat2 = seq_feat.unsqueeze(-3).unsqueeze(-2)
        # (feat1 * feat2).reshape(B, L, L, -1)
        seq_mat = torch.einsum('blh,bmh->bhlm', seq_feat, seq_feat)  # BxHxLxL
        fused_mat = torch.cat([mat_feat, seq_mat], dim=1)  # Residual-style fusion
        out_mat = self.mat_output_proj(fused_mat)
        if mat_mask is not None:
            out_mat = out_mat * mat_mask
        return out_seq, out_mat


class SeqMatFusion_net(nn.Module):
    '''
    Cascaded Sequence-Matrix Fusion Network (CSMF-Net)
    '''

    def __init__(self, LM_embed_chan=None, use_BPM=False, seq_dim=32, mat_channels=32, max_seq_len=512, out_dim=4, out_channels=3, num_blocks=16):
        super(SeqMatFusion_net, self).__init__()
        out_dim = out_dim or seq_dim
        out_channels = out_channels or mat_channels

        mat_in_chan = 0
        if use_BPM:
            mat_in_chan += 2
        if LM_embed_chan:
            mat_in_chan += LM_embed_chan
        self.use_BPM = use_BPM
        self.LM_embed_chan = LM_embed_chan

        ## token embed
        self.seq_input_embed = nn.Embedding(6, seq_dim)
        self.mat_input_proj = nn.Conv2d(mat_in_chan, mat_channels, kernel_size=1)

        ## positional embedding
        self.seq_rope = RoPE(dim=seq_dim, max_seq_len=max_seq_len)
        self.mat_2d_rope = Matrix2DRoPE(dim=mat_channels, max_seq_len=max_seq_len)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(SeqMatFusion(seq_dim, mat_channels, seq_dim, mat_channels))
        
        self.final_seq_proj = nn.Linear(seq_dim, out_dim)
        self.final_mat_proj = nn.Conv2d(mat_channels, out_channels, kernel_size=1)
        
    def forward(self, seq, BPM=None, LM_embed=None, seq_mask=None, mat_mask=None):
        # token embed
        seq_feat = self.seq_input_embed(seq)

        mat = []
        if self.use_BPM:
            mat.append(BPM)
        if self.LM_embed_chan:
            mat.append(LM_embed)
        if mat:
            mat = torch.cat(mat, dim=1)
        mat_feat = self.mat_input_proj(mat)

        # positional embedding
        seq_feat = self.seq_rope(seq_feat)
        mat_feat = self.mat_2d_rope(mat_feat)
        
        # Cascaded processing
        for block in self.blocks:
            seq_feat, mat_feat = block(seq_feat, mat_feat, seq_mask, mat_mask)
        
        # Final output
        edge = self.final_seq_proj(seq_feat)
        orient = self.final_mat_proj(mat_feat)
        orient = (orient + orient.transpose(-2, -1))/2
        return edge, orient


if __name__ == "__main__":
    torch.manual_seed(42)
    import sys
    sys.path.append('../util')
    from NCfold_kit import count_para
    # Hyperparameters
    B, L = 5, 513  # Batch size, sequence length
    seq_dim, mat_channels = 32, 32  # 4-divided 0.9M
    seq_dim, mat_channels = 64, 64  # 4-divided # 3.5M
    seq_dim, mat_channels = 8, 8  # 4-divided # 3.5M
    out_dim, out_channels = 4, 3  # Input dimensions
    num_blocks = 16  # Number of cascaded blocks
    max_seq_len = 512
    
    # Create random inputs
    seq = torch.randint(0, 6, (B, L)).long()
    mat = torch.randn(B, 2, L, L)
    
    # Initialize model
    model = SeqMatFusion_net(
        mat_in_chan=2,
        seq_dim=seq_dim,
        mat_channels=mat_channels,
        max_seq_len = max_seq_len,
        out_dim=out_dim,
        out_channels=out_channels,
        num_blocks=num_blocks,
    )
    count_para(model)
    
    # Forward pass
    out_seq, out_mat = model(seq, mat)
    
    # Check output shapes
    print(f"Input sequence shape: {seq.shape}")
    print(f"Input matrix shape: {mat.shape}")
    print(f"Output sequence shape: {out_seq.shape}")  # Should be BxLxhidden_dim
    print(f"Output matrix shape: {out_mat.shape}")    # Should be Bxhidden_dimxLxL
    
    # Verify output shapes
    assert out_seq.shape == (B, L, out_dim), f"Sequence shape mismatch: {out_seq.shape} vs {(B, L, out_dim)}"
    assert out_mat.shape == (B, out_channels, L, L), f"Matrix shape mismatch: {out_mat.shape} vs {(B, out_channels, L, L)}"
    print("All tests passed!")
