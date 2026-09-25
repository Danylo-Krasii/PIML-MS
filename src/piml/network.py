"""The localization network and the cube-symmetry augmentation used to train it.

`LocalizationNet` maps the 21 standardized stiffness channels of every voxel to the 36
components of A(x). `Octahedral` applies the 48 symmetries of the cube to a batch of fields.
"""

import numpy as np
import torch
import torch.nn as nn

from .microstructure import channel_maps, octahedral_group, spatial_action


class VoxelNorm(nn.Module):
    """Normalize each voxel's channel vector independently, then scale and shift per channel.

    Unlike InstanceNorm3d this has zero spatial extent, so a network using it sees no further
    than its receptive field.
    """

    def __init__(self, c, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, c, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, c, 1, 1, 1))

    def forward(self, x):
        m = x.mean(1, keepdim=True)
        v = x.var(1, keepdim=True, unbiased=False)
        return (x - m) / torch.sqrt(v + self.eps) * self.weight + self.bias


class LocalizationNet(nn.Module):
    """Pooling-free, stride 1, 3x3x3 kernels. The receptive field is 1 + 2L, so L sets it.

    Residual additions join blocks 2 to L. The 1x1x1 skip from the input learns the part of
    A(x) that is local in C(x), with its own scale.
    """

    def __init__(self, layers, width=64, cin=21, cout=36, linear=False,
                 padding_mode="zeros", norm="instance"):
        """padding_mode "circular" is the true neighbour on a periodic cell.

        norm selects what the nonlinear arm puts after each convolution:
          "instance"  InstanceNorm3d, whose statistics span the whole cell, so every voxel
                      sees global information at a nominally local receptive field.
          "none"      no normalization, so reach is exactly the receptive field. The released
                      checkpoints use it.
          "voxel"     VoxelNorm, which normalizes without widening the reach.
        """
        super().__init__()
        self.blocks = nn.ModuleList()
        c = cin
        for _ in range(layers):
            conv = nn.Conv3d(c, width, 3, padding=1, padding_mode=padding_mode)
            if linear:
                self.blocks.append(conv)
            else:
                act = nn.LeakyReLU(0.2, inplace=True)
                if norm == "instance":
                    self.blocks.append(nn.Sequential(conv, nn.InstanceNorm3d(width, affine=True), act))
                elif norm == "voxel":
                    self.blocks.append(nn.Sequential(conv, VoxelNorm(width), act))
                elif norm == "none":
                    self.blocks.append(nn.Sequential(conv, act))
                else:
                    raise ValueError(f"unknown norm {norm!r}")
            c = width
        self.head = nn.Conv3d(width, cout, 1)
        self.skip = nn.Conv3d(cin, cout, 1)

    def forward(self, x):
        h = x
        for i, b in enumerate(self.blocks):
            y = b(h)
            h = y + h if (i > 0 and y.shape == h.shape) else y
        return self.head(h) + self.skip(x)


def to_nchw(a):
    return torch.from_numpy(np.ascontiguousarray(a.transpose(0, 4, 1, 2, 3)))


class Octahedral:
    """The 48 cube symmetries as torch operations on (B, C, n, n, n) fields.

    Fields are transformed raw, before standardization, since standardization is not
    equivariant. Standardization statistics are then averaged over the channel orbits of the
    group so that the standardized field transforms the same way. `transpose=True` pairs the
    spatial action with the transposed tensor action, which is not a symmetry; it is a control
    that validates the convention (a model trained with it should do worse).
    """

    def __init__(self, dev="cuda", transpose=False):
        self.ops = []
        for Q in octahedral_group():
            inv, flips = spatial_action(Q)
            M21, M36 = channel_maps(Q.T if transpose else Q)
            self.ops.append((inv, flips, torch.tensor(M21, dtype=torch.float32, device=dev),
                             torch.tensor(M36, dtype=torch.float32, device=dev)))
        self.orbits21 = self._orbits([o[2].cpu().numpy() for o in self.ops])
        self.orbits36 = self._orbits([o[3].cpu().numpy() for o in self.ops])

    @staticmethod
    def _orbits(mats):
        k = mats[0].shape[0]
        adj = np.zeros((k, k), dtype=bool)
        for M in mats:
            adj |= np.abs(M) > 1e-9
        seen, orbits = set(), []
        for i in range(k):
            if i in seen:
                continue
            stack, orb = [i], set()
            while stack:
                a = stack.pop()
                if a in orb:
                    continue
                orb.add(a)
                stack.extend(np.nonzero(adj[a] | adj[:, a])[0].tolist())
            seen |= orb
            orbits.append(sorted(orb))
        return orbits

    @staticmethod
    def orbit_stats(x, orbits):
        """Per-channel mean and std, then equalized within each orbit."""
        mu = x.mean((0, 2, 3, 4)); var = x.var((0, 2, 3, 4))
        for orb in orbits:
            mu[orb] = mu[orb].mean(); var[orb] = var[orb].mean()
        return mu.view(1, -1, 1, 1, 1), var.sqrt().view(1, -1, 1, 1, 1) + 1e-8

    def apply(self, X, Y, g):
        inv, flips, M21, M36 = self.ops[g]
        ax = [2 + k for k in flips]
        if ax:
            X = torch.flip(X, ax); Y = torch.flip(Y, ax)
        perm = [0, 1] + [2 + k for k in inv]
        X = X.permute(perm).contiguous(); Y = Y.permute(perm).contiguous()
        X = torch.einsum("ij,bjxyz->bixyz", M21, X)
        Y = torch.einsum("ij,bjxyz->bixyz", M36, Y)
        return X, Y
