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
                torch.cat(
                    (frequencies, zeros, zeros)
                ),
                torch.cat(
                    (zeros, frequencies, zeros)
                ),
                torch.cat(
                    (zeros, zeros, frequencies)
                ),
            )
        )

        self.register_buffer("fourier_basis", fourier_basis)
        self.proj = nn.Linear(fourier_dim + 3, dim)

    @staticmethod
    def embed(input, fourier_basis):
        # B x N x 3 * 3 x 24 = B x N x 24
        projections = torch.einsum("bnd,de->bne", input, fourier_basis)
        # B x N x 48
        return torch.cat((projections.sin(), projections.cos()), dim=2)

    def forward(self, input):
        # input: B x N x 3
        fourier_features = self.embed(input, self.fourier_basis)
        return self.proj(torch.cat((fourier_features, input), dim=2))


def main():
    # annotations = oxl.get_annotations()
    # oxl.download_objects(annotations, "data")
    mesh = trimesh.load("cow-nonormals.obj")
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)


if __name__ == "__main__":
    main()
