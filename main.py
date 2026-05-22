import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import objaverse.xl as oxl
import trimesh
import flash_attn


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


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        pass

class SelfMultiheadAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, use_bias: bool = True):
        super().__init__()

        assert model_dim % num_heads == 0

        self.num_heads = num_heads
        self.to_qkv = nn.Linear(model_dim, 3 * model_dim, bias=use_bias)
        self.to_result = nn.Linear(model_dim, model_dim, bias=use_bias)

    def forward(self, input):
        if input.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError(f"SelfMultiheadAttention expects fp16/bf16 input, got {input.dtype}")
        if not input.is_cuda:
            raise ValueError("flash-attn requires CUDA tensors")

        batch_size, seq_len, channels = input.shape
        qkv = self.to_qkv(input)
        # qkv: (batch_size, seqlen, 3, nheads, headdim)
        qkv = torch.reshape(qkv, (batch_size, seq_len, 3, self.num_heads, -1))
        # out: (batch_size, seqlen, nheads, model_dim)
        out = flash_attn.flash_attn_qkvpacked_func(qkv)
        out = out.reshape(batch_size, seq_len, -1)
        out = self.to_result(out)

        return out

class CrossMultiheadAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, use_bias: bool = True):
        super().__init__()

        assert model_dim % num_heads == 0

        self.num_heads = num_heads
        self.use_bias = use_bias

        self.to_q = nn.Linear(model_dim, model_dim, bias=use_bias)
        self.to_kv = nn.Linear(model_dim, 2 * model_dim, bias=use_bias)
        self.to_result = nn.Linear(model_dim, model_dim)

    def forward(self, input: torch.Tensor, context):
        if input.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError(f"CrossMultiheadAttention expects fp16/bf16 input, got {input.dtype}")
        if context.dtype != input.dtype:
            raise TypeError(f"context dtype must match input dtype, got {context.dtype} vs {input.dtype}")
        if not input.is_cuda or not context.is_cuda:
            raise ValueError("flash-attn requires CUDA tensors")

        batch_size, seq_len, channels = input.shape
        q = self.to_q(input)
        kv = self.to_kv(context)
        q = torch.reshape(q, (batch_size, seq_len, self.num_heads, -1))
        kv = torch.reshape(kv, (batch_size, context.shape[1], 2, self.num_heads, -1))

        out = flash_attn.flash_attn_kvpacked_func(q, kv)
        out = out.reshape(batch_size, seq_len, -1)
        out = self.to_result(out)

        return out

class DiffusionTransformerLayer(nn.Module):
    def __init__(self, model_dim=768, num_heads=12):
        super().__init__()
        assert model_dim % num_heads == 0

        self.num_heads = num_heads
        self.self_attn = SelfMultiheadAttention(model_dim, num_heads)
        self.cross_attn = CrossMultiheadAttention(model_dim, num_heads)

        self.norm0 = nn.LayerNorm(model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        self.ffn = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Linear(4 * model_dim, model_dim),
        )

    def forward(self, z) -> torch.Tensor:
        # z: B x N x D
        # TODO: go from 64 -> 768 before this layer
        normalized_input = self.norm0(z)
        attn = self.self_attn(normalized_input)
        output0 = attn + z

        # TODO: use a pretrained language model to get textual features c
        c = None
        x = self.norm1(output0)
        attn = self.cross_attn(x, c)
        output1 = attn + output0

        x = self.norm2(output1)
        x = self.ffn(x)
        output2 = x + output1

        return output2

class DiffusionTransformer(nn.Module):
    def __init__(self, model_dim=768, num_heads=12, model_channels=64):
        super().__init__()
        assert model_dim % num_heads == 0

        self.in_proj = nn.Linear(model_channels, model_dim)


def main():
    # annotations = oxl.get_annotations()
    # oxl.download_objects(annotations, "data")
    mesh = trimesh.load("cow-nonormals.obj")
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)


if __name__ == "__main__":
    main()
