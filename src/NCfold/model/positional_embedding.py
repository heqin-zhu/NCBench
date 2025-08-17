import math

import torch
import torch.nn as nn


class RoPE(nn.Module):
    """
    Rotary Position Embedding for sequence features
    Encodes positional information using rotation matrices to preserve relative position relationships
    """
    def __init__(self, dim, max_seq_len=512, base=10000):
        super().__init__()
        self.dim = dim  # Feature dimension (must be even)
        self.base = base
        self.max_seq_len = max_seq_len
        
        # Precompute frequency parameters with unique buffer name
        inv_freq = 1.0 / (base **(torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('rope_inv_freq', inv_freq)
        
    def forward(self, x):
        """
        Args:
            x: Sequence features with shape (B, L, D) where:
               - B = batch size
               - L = sequence length
               - D = feature dimension (must match self.dim and be even)
        Returns:
            Features with rotary positional embedding applied, shape (B, L, D)
        """
        B, L, D = x.shape
        assert D == self.dim, f"Input dimension {D} mismatch with ROPE dimension {self.dim}"
        assert D % 2 == 0, "ROPE requires even feature dimension"
        
        # Generate position indices (0 to L-1)
        pos = torch.arange(L, device=x.device).unsqueeze(1)  # Shape: (L, 1)
        
        # Calculate frequencies and phase embeddings using renamed buffer
        freqs = pos * self.rope_inv_freq  # Shape: (L, D/2)
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)  # Shape: (L, D)
        emb = emb.unsqueeze(0).repeat(B, 1, 1)  # Shape: (B, L, D)
        
        # Apply rotary encoding
        return self.apply_rotary(x, emb)
    
    @staticmethod
    def apply_rotary(x, emb):
        """Applies rotary transformation to features using precomputed embeddings"""
        B, L, D = x.shape
        x1 = x[..., :D//2]  # Real part
        x2 = x[..., D//2:]  # Imaginary part
        
        # Extract sine and cosine components from embeddings
        sin = emb[..., :D//2]
        cos = emb[..., D//2:]
        
        # Rotary transformation formulas
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin
        return torch.cat([out1, out2], dim=-1)


class Matrix2DRoPE(nn.Module):
    """
    2D Rotary Position Embedding for matrix features
    Encodes both row and column positional information for 2D matrix structures
    """
    def __init__(self, dim, max_seq_len=512, base=10000):
        super().__init__()
        self.dim = dim  # Channel dimension (must be divisible by 4)
        self.base = base
        self.max_seq_len = max_seq_len
        
        # Use different frequency bases for rows and columns with unique buffer names
        inv_freq_row = 1.0 / (base** (torch.arange(0, dim//2, 2).float() / (dim//2)))
        inv_freq_col = 1.0 / ((base * 2) **(torch.arange(0, dim//2, 2).float() / (dim//2)))
        
        self.register_buffer('rope_inv_freq_row', inv_freq_row)
        self.register_buffer('rope_inv_freq_col', inv_freq_col)
        
    def forward(self, x):
        """
        Args:
            x: Matrix features with shape (B, C, L, L) where:
               - B = batch size
               - C = channel dimension (must match self.dim and be divisible by 4)
               - L = matrix dimensions (height and width)
        Returns:
            Matrix features with 2D rotary positional embedding applied, shape (B, C, L, L)
        """
        B, C, L, _ = x.shape
        assert C == self.dim, f"Input channels {C} mismatch with 2D ROPE dimension {self.dim}"
        assert C % 4 == 0, "2D ROPE requires channel dimension divisible by 4"
        
        # Generate row and column position indices
        row_pos = torch.arange(L, device=x.device).unsqueeze(1)  # Shape: (L, 1)
        col_pos = torch.arange(L, device=x.device).unsqueeze(1)  # Shape: (L, 1)
        
        # Calculate row position embeddings using renamed buffer
        row_freqs = row_pos * self.rope_inv_freq_row  # Shape: (L, C/4)
        row_emb = torch.cat([row_freqs.sin(), row_freqs.cos()], dim=-1)  # Shape: (L, C/2)
        
        # Calculate column position embeddings using renamed buffer
        col_freqs = col_pos * self.rope_inv_freq_col  # Shape: (L, C/4)
        col_emb = torch.cat([col_freqs.sin(), col_freqs.cos()], dim=-1)  # Shape: (L, C/2)
        
        # Expand embeddings to match matrix dimensions (B, L, L, C)
        row_emb = row_emb.unsqueeze(0).unsqueeze(2).repeat(B, 1, L, 1)  # (B, L, L, C/2)
        col_emb = col_emb.unsqueeze(0).unsqueeze(1).repeat(B, L, 1, 1)  # (B, L, L, C/2)
        emb = torch.cat([row_emb, col_emb], dim=-1)  # (B, L, L, C)
        emb = emb.permute(0, 3, 1, 2)  # (B, C, L, L) to match input dimensions
        
        # Apply 2D rotary encoding
        return self.apply_2d_rotary(x, emb)
    
    @staticmethod
    def apply_2d_rotary(x, emb):
        """Applies 2D rotary transformation to matrix features"""
        B, C, L, _ = x.shape
        half_dim = C // 2
        quarter_dim = C // 4
        
        # Split channels for row and column rotations
        x_row = x[:, :half_dim, :, :]  # Channels for row rotation (B, C/2, L, L)
        x_col = x[:, half_dim:, :, :]  # Channels for column rotation (B, C/2, L, L)
        
        # Row rotation components
        x_row1 = x_row[:, :quarter_dim, :, :]
        x_row2 = x_row[:, quarter_dim:, :, :]
        row_sin = emb[:, :quarter_dim, :, :]
        row_cos = emb[:, quarter_dim:half_dim, :, :]
        row_rot1 = x_row1 * row_cos - x_row2 * row_sin
        row_rot2 = x_row2 * row_cos + x_row1 * row_sin
        row_out = torch.cat([row_rot1, row_rot2], dim=1)
        
        # Column rotation components
        x_col1 = x_col[:, :quarter_dim, :, :]
        x_col2 = x_col[:, quarter_dim:, :, :]
        col_sin = emb[:, half_dim:half_dim+quarter_dim, :, :]
        col_cos = emb[:, half_dim+quarter_dim:, :, :]
        col_rot1 = x_col1 * col_cos - x_col2 * col_sin
        col_rot2 = x_col2 * col_cos + x_col1 * col_sin
        col_out = torch.cat([col_rot1, col_rot2], dim=1)
        
        return torch.cat([row_out, col_out], dim=1)


if __name__ == "__main__":
    # Configuration
    B, L = 2, 100  # Batch size and sequence/matrix length
    seq_dim = 64    # Sequence feature dimension (must be even)
    mat_channels = 64  # Matrix channel dimension (must be divisible by 4)
    max_seq_len = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate test data
    seq_feat = torch.randn(B, L, seq_dim).to(device)  # Sequence features
    mat_feat = torch.randn(B, mat_channels, L, L).to(device)  # Matrix features
    
    # Test sequence ROPE
    seq_rope = RoPE(dim=seq_dim, max_seq_len=max_seq_len).to(device)
    seq_with_rope = seq_rope(seq_feat)
    print(f"Sequence ROPE output shape: {seq_with_rope.shape}")  # Should be (B, L, seq_dim)
    
    # Test matrix 2D ROPE
    mat_rope = Matrix2DRoPE(dim=mat_channels, max_seq_len=max_seq_len).to(device)
    mat_with_rope = mat_rope(mat_feat)
    print(f"Matrix 2D ROPE output shape: {mat_with_rope.shape}")  # Should be (B, mat_channels, L, L)
    
    # Verify relative position property (consistent encoding for same relative distances)
    def check_relative_position_consistency(rope, dim):
        x = torch.zeros(1, 5, dim).to(device)
        x[0, 2, :] = 1.0  # Set unit vector at position 2
        x_rot = rope(x)
        # Relative distance between positions 2&0 should match 3&1 (both = 2)
        diff1 = x_rot[0, 2] - x_rot[0, 0]
        diff2 = x_rot[0, 3] - x_rot[0, 1]
        print(f"Relative position consistency (should be near 0): {torch.norm(diff1 - diff2):.6f}")
    
    check_relative_position_consistency(seq_rope, seq_dim)
