import torch
import torch.nn as nn
import numpy as np
import objaverse.xl as oxl
import trimesh


class PointEmbed(nn.Module):
    def __init__(self, num_frequencies=8, dim=128):
        super().__init__()

        frequencies = torch.pow(2, torch.arange(num_frequencies)).float() * torch.pi
        fourier_dim = num_frequencies * 6
        zeros = torch.zeros(num_frequencies)

        fourier_basis = torch.stack(
            (
                torch.cat((frequencies, zeros, zeros)),
                torch.cat((zeros, frequencies, zeros)),
                torch.cat((zeros, zeros, frequencies)),
            )
        )

        self.register_buffer("fourier_basis", fourier_basis)
        self.proj = nn.Linear(fourier_dim + 3, dim)

    @staticmethod
    def embed(input, fourier_basis):
        # B x N x 3 * 3 x 24 = B x N x 24
        projections = input @ fourier_basis
        # B x N x 48
        return torch.cat((projections.sin(), projections.cos()), dim=2)

    def forward(self, input):
        # input: B x N x 3
        fourier_features = self.embed(input, self.fourier_basis)
        return self.proj(torch.cat((fourier_features, input), dim=2))


class Attention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, channels=64):
        super().__init__()
        '''
        self.apply_query_matrix = nn.Linear(input_dim, query_dim, bias=False)
        self.apply_kv = nn.Linear(data_dim, query_dim + value_dim, bias=False)
        self.query_dim=query_dim
        scale = query_dim ** -0.5
        '''
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.apply_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
        # TODO: DropPath?

    def forward(self, input, data) -> torch.Tensor:
        q = self.apply_query_matrix(input)

        # kv: [B, N, Dq + Dv]
        kv = self.apply_kv(data)
        k = kv[:, :, : self.query_dim]
        v = kv[:, :, self.query_dim :]

        # softmax(QKt / sqrt(Dq)) * V
        attn = torch.softmax((q @ torch.transpose(k, -2, -1)) * self.scale, dim=-1)
        out = attn @ v

        return out

class DiffusionTransformerLayer(nn.Module):
    def __init__(self, model_dim=768, num_heads=12, model_channels=64):
        super().__init__()
        assert model_dim % num_heads == 0

        self.self_attn = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=True)

        self.norm0 = nn.LayerNorm(model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        #self.in_proj = nn.Linear(model_channels, model_dim)
        #self.out_proj = nn.Linear(model_dim, model_channels)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim), 
            nn.GELU(),
            nn.Linear(4 * model_dim, model_dim)
        )

    def forward(self, z) -> torch.Tensor:
        # z: B x N x 64
        # TODO: go from 64 -> 768 before this layer
        normalized_input = self.norm0(z)
        attn, _ = self.self_attn(normalized_input, normalized_input, normalized_input, need_weights=False)
        output0 = attn + z

        # TODO: use a pretrained language model to get textual features c
        c = None
        x = self.norm1(output0)
        attn, _ = self.cross_attn(x, c, c, need_weights=False)
        output1 = attn + output0

        x = self.norm2(output1)
        x = self.ffn(x)
        output2 = x + output1

        return output2 


def main():
    # annotations = oxl.get_annotations()
    # oxl.download_objects(annotations, "data")
    mesh = trimesh.load("cow-nonormals.obj")
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)


if __name__ == "__main__":
    main()
