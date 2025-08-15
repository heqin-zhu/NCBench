import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Config


class TransformerBranch(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        config = DebertaV2Config(
            hidden_size=d_model,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=4*d_model,
            max_position_embeddings=512,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.deberta = DebertaV2Model(config)

    def forward(self, x, attention_mask=None):
        outputs = self.deberta(
            inputs_embeds=x,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.last_hidden_state

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # B H W C
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # B C H W
        return input + x

class ConvBranch(nn.Module):
    def __init__(self, in_channels=1, dims=[64, 128, 256], out_channels=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, dims[0]),
        )
        self.blocks = nn.ModuleList()
        current_dim = dims[0]
        for dim in dims[1:]:
            self.blocks.append(ConvNeXtBlock(current_dim))
            self.blocks.append(nn.Conv2d(current_dim, dim, kernel_size=2, stride=2))
            current_dim = dim
        self.blocks.append(ConvNeXtBlock(current_dim))
        self.norm = nn.LayerNorm(current_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(current_dim, out_channels)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 3, 1)  # B H W C
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # B C H W
        print(x.shape)
        print(self.pool(x).shape)
        x = self.pool(x).flatten(1)
        print(x.shape)
        exit()
        # TODO
        return self.fc(x)


class NCpair_model(nn.Module):
    def __init__(self, d_model=64, conv_out=None, from_pretrained=None):
        super().__init__()
        conv_out = conv_out or d_model
        # 6 tokens: AUGCN PAD
        self.embed = nn.Embedding(6, d_model)
        self.transformer_branch = TransformerBranch(d_model)
        self.conv_branch = ConvBranch(in_channels=1, out_channels=conv_out)
        # 4 edges: no-edge, W, H, S
        self.edge_classifier = nn.Linear(d_model + conv_out, 4)
        # 3 orient: non-pair, trans, cis
        self.orient_classifier = nn.Conv2d(1, 3, 3)

        if from_pretrained:
            self.load_state_dict(torch.load(from_pretrained))

    def forward(self, rna_seq, contact_map, attention_mask=None):
        rna_seq_embed = self.embed(rna_seq)
        contact_map = contact_map.unsqueeze(dim=1)
        transformer_out = self.transformer_branch(rna_seq_embed, attention_mask)
        # seq pooling, consider padding
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(transformer_out).float()
            sum_embeddings = torch.sum(transformer_out * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            transformer_pooled = sum_embeddings / sum_mask
        else:
            transformer_pooled = transformer_out.mean(dim=1)

        # conv
        conv_pooled = self.conv_branch(contact_map)
        print('conv', conv_pooled.shape)
        exit()

        # feature fusion
        # BxLxD
        combined = torch.cat([transformer_pooled, conv_pooled], dim=1)
        # BxLxL
        x = combined @ combined.T

        # BxLxLx3
        print(combined.shape, x.shape)
        exit()
        orient = self.orient_classifier(x.unsqueeze(1)).permute(0, 2, 3, 1)
        return {
                'edge': self.edge_classifier(combined),
                'orient': orient,
               }


if __name__ == "__main__":
    from utils import count_para
    batch = 20
    dim = 64
    length = 100
    model = NCpair_model(d_model=dim, conv_out=dim)
    rna_seq = torch.randint(0, 6, (batch, length), dtype=torch.long)
    contact_map = torch.randn(batch, length, length)
    attention_mask = torch.ones(batch, length) # suppose no padding
    
    outputs = model(rna_seq, contact_map, attention_mask)
    print(outputs['edge'].shape)  # torch.Size([batch, L, 4])
    print(outputs['orient'].shape)  # torch.Size([batch, L, L, 3])
    count_para(model)
