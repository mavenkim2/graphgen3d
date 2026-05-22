import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import objaverse.xl as oxl
import trimesh
import flash_attn
import torch_cluster

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

class ResidualAttentionBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, is_self: bool=False, context_dim: int=0, use_bias:bool=True,):
        super().__init__()
        self.is_self = is_self
        self.norm = nn.LayerNorm(model_dim)
        if self.is_self:
            self.attn = SelfMultiheadAttention(model_dim, num_heads, use_bias)
        else:
            context_dim = model_dim if context_dim == 0 else context_dim
            self.cross_norm = nn.LayerNorm(context_dim)
            self.attn = CrossMultiheadAttention(model_dim, num_heads, use_bias) 
    def forward(self, input, context=None):
        x = self.norm(input)
        if self.is_self:
            x = self.attn(x)
        else: 
            assert context is not None
            y = self.cross_norm(context)
            x = self.attn(x, y)
        x = input + x
        return x


class FeedForward(nn.Module):
    def __init__(self, model_dim, mult: int=4):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, mult * model_dim),
            nn.GELU(),
            nn.Linear(mult * model_dim, model_dim)
        )
    def forward(self, input):
        x = self.norm(input)
        x = self.ffn(x)
        x = x + input
        return x


class VAEGeometryEncoder(nn.Module):
    def __init__(self, model_dim=512, num_heads=8):
        super().__init__()
        self.cross_attn_block = ResidualAttentionBlock(model_dim, num_heads, is_self=False)
        self.ffn_block = FeedForward(model_dim)
        self.positional_encoding = PointEmbed(dim=model_dim)

    def forward(self, point_cloud):
        ratio = 0.25

        B, N, D = point_cloud.shape

        flattened_point_cloud = point_cloud.view(B * N, D)
        batch = torch.arange(B).to(point_cloud.device)
        batch = torch.repeat_interleave(batch, N)
        downsampled_pc_idx = torch_cluster.fps(src=flattened_point_cloud, batch=batch, ratio=ratio)
        downsampled_pc = flattened_point_cloud[downsampled_pc_idx]
        downsampled_pc = downsampled_pc.view(B, -1, 3)

        encoded_pc = self.positional_encoding(point_cloud)
        encoded_downsampled_pc = self.positional_encoding(downsampled_pc)

        output = self.cross_attn_block(encoded_downsampled_pc, encoded_pc)
        output = self.ffn_block(output)
        return output


class DiffusionTransformerLayer(nn.Module):
    def __init__(self, model_dim=768, num_heads=12):
        super().__init__()
        assert model_dim % num_heads == 0

        self.self_attn_block = ResidualAttentionBlock(model_dim=model_dim, num_heads=num_heads, is_self=True)
        self.cross_attn_block = ResidualAttentionBlock(model_dim=model_dim, num_heads=num_heads, is_self=False)
        self.ffn_block = FeedForward(model_dim)

    def forward(self, z) -> torch.Tensor:
        # z: B x N x D
        # TODO: go from 64 -> 768 before this layer
        output = self.self_attn_block(z)

        # TODO: use a pretrained language model to get textual features c
        c = None
        output = self.cross_attn_block(output, c)
        output = self.ffn_block(output)

        return output

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
