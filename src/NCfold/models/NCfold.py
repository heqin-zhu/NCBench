import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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
        self.transformer = TransformerEncoder(encoder_layers, num_layers=num_layers)
        # self.transformer = nn.Sequential(
        #     nn.Linear(seq_dim, self.hidden_dim),
        #     nn.GELU(),
        #     nn.LayerNorm(self.hidden_dim),
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     nn.GELU()
        # )
        
        # Residual connection projection
        self.residual_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # x: BxLxD
        residual = x
        x = self.input_proj(x)
        x = self.layer_norm(x)
        x = self.transformer(x)
        residual = self.residual_proj(residual)
        x = x + residual
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
        
    def forward(self, x):
        # x: BxCxLxL
        residual = self.residual_proj(x)
        x = self.conv_layers(x)
        x = x + residual
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
            nn.Softmax(dim=-1)
        )
        
        # Output projection layers
        self.seq_output_proj = nn.Linear(self.hidden_dim+self.hidden_channels, self.hidden_dim)
        self.mat_output_proj = nn.Conv2d(self.hidden_channels+self.hidden_dim, self.hidden_channels, kernel_size=1)
        
    def forward(self, seq, mat):
        # Input shapes: seq (BxLxD), mat (BxCxLxL)
        B, L, _ = seq.shape
        
        # Feature extraction with residual connections
        seq_feat = self.seq_extractor(seq)
        mat_feat = self.mat_extractor(mat)
        
        # ---------------------------
        # Matrix-to-sequence fusion
        # ---------------------------
        # Matrix feature projection and pooling
        mat_seq_i = torch.mean(mat_feat, dim=-2).transpose(1, 2)  # BxHxLxL -> BxHxL -> BxLxH
        mat_seq_j = torch.mean(mat_feat, dim=-1).transpose(1, 2)  # BxHxLxL -> BxHxL -> BxLxH
        attn_input = torch.cat([mat_seq_i, mat_seq_j], dim=-1)  # BxLx(2H)
        attn_weight = self.fusion_attn(attn_input)  # BxLx1
        mat_seq = mat_seq_i * attn_weight[...,0:1] + mat_seq_j * attn_weight[...,1:2]

        fused_seq = torch.cat([seq_feat, mat_seq], dim=-1)
        out_seq = self.seq_output_proj(fused_seq)
        
        # ---------------------------
        # Sequence-to-matrix fusion
        # ---------------------------
        seq_mat = torch.einsum('blh,bmh->bhlm', seq_feat, seq_feat)  # BxHxLxL
        fused_mat = torch.cat([mat_feat, seq_mat], dim=1)  # Residual-style fusion
        out_mat = self.mat_output_proj(fused_mat)
        
        return out_seq, out_mat


class NCfold(nn.Module):
    '''
    Cascaded Sequence-Matrix Fusion Network (CSMF-Net)
    Rationale:
        Cascaded: Highlights the key design of stacking multiple fusion blocks to enable progressive feature refinement.
        Sequence-Matrix: Explicitly indicates the dual-modal nature of the input data (sequence features + matrix features).
        Fusion: Emphasizes the core functionality of integrating these two data types.
        Network: Signifies the deep learning architecture.
    '''

    def __init__(self, seq_dim, mat_channels, out_dim=None, out_channels=None, hidden_dim=32, hidden_channels=32, num_blocks=8):
        super(NCfold, self).__init__()
        out_dim = out_dim or seq_dim
        out_channels = out_channels or mat_channels
        hidden_dim = hidden_dim or seq_dim
        hidden_channels = hidden_channels or mat_channels

        self.seq_input_proj = nn.Linear(seq_dim, hidden_dim)
        self.mat_input_proj = nn.Conv2d(mat_channels, hidden_channels, kernel_size=1)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(SeqMatFusion(hidden_dim, hidden_channels, hidden_dim, hidden_channels))
        
        self.final_seq_proj = nn.Linear(hidden_dim, out_dim)
        self.final_mat_proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        
    def forward(self, seq, mat):
        # Input projection
        seq_feat = self.seq_input_proj(seq)
        mat_feat = self.mat_input_proj(mat)
        
        # Cascaded processing
        for block in self.blocks:
            seq_feat, mat_feat = block(seq_feat, mat_feat)
        
        # Final output
        out_seq = self.final_seq_proj(seq_feat)
        out_mat = self.final_mat_proj(mat_feat)
        
        return out_seq, out_mat


if __name__ == "__main__":
    torch.manual_seed(42)
    import sys
    sys.path.append('../utils')
    from NCfold_kit import count_para
    # Hyperparameters
    B, L = 5, 500  # Batch size, sequence length
    hidden_dim, hidden_channels = 32, 32  # Input dimensions, 4-divided  # 0.9M
    hidden_dim, hidden_channels = 64, 64  # Input dimensions, 4-divided  # 3.5M
    seq_dim, mat_channels = hidden_dim, 2  # Input dimensions
    out_dim, out_channels = 4, 3  # Input dimensions
    num_blocks = 16  # Number of cascaded blocks
    
    # Create random inputs
    seq = torch.randn(B, L, seq_dim)
    mat = torch.randn(B, mat_channels, L, L)
    
    # Initialize model
    model = NCfold(
        seq_dim=seq_dim,
        mat_channels=mat_channels,
        hidden_dim=hidden_dim,
        hidden_channels=hidden_channels,
        out_dim=out_dim,
        out_channels=out_channels,
        num_blocks=num_blocks
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
