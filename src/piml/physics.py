"""Physics operators on the stress-localization field: the equilibrium ratio and the projections.

The equilibrium ratio is the solver's own convergence metric (`equilibrium_error` in
`fft_homog.hpp`): unnormalized forward FFT, xi_j = m_j / n with signed integer m, Nyquist
retained, numerator summed over xi != 0, denominator the Mandel 6-norm of the xi = 0 mode.
Stored Mandel order is xx, yy, zz, xy, yz, xz.

Array conventions. A 36-channel field is either channels-last (..., 36), the layout of
`periodic.load()`, or channels-first (B, 36, n, n, n), the layout the network emits; both
flatten the stored (n, n, n, 6, 6) array in C order, so channel 6 i + k is [i, k] = stress
component i per unit load k. The operators below take A66 as (B, n, n, n, 6, 6) torch tensors.

    python -m piml.physics labels --which cu --seeds 1160-1199

`labels` prints the equilibrium ratio of the stored labels, one line per volume.
"""

import argparse

import h5py
import numpy as np
import torch

from . import paths

S2 = 1.0 / np.sqrt(2.0)


def c21_to_c6(C21):
    """(..., 21) -> (..., 6, 6) symmetric, inverse of microstructure.upper_triangle.

    Accepts a numpy array or a torch tensor and returns the same kind.
    """
    i, j = np.triu_indices(6)
    if isinstance(C21, torch.Tensor):
        out = C21.new_zeros(C21.shape[:-1] + (6, 6))
        out[..., i, j] = C21
        out[..., j, i] = C21
        return out
    out = np.zeros(C21.shape[:-1] + (6, 6), dtype=C21.dtype)
    out[..., i, j] = C21
    out[..., j, i] = C21
    return out


def a36_to_cols(A):
    """36-channel field -> (..., n, n, n, 6, 6) with [i, k] = stress i per unit load k.

    Channels-last (..., 36) is reshaped in place; channels-first (B, 36, n, n, n) is moved to
    channels-last first. Works on numpy arrays and torch tensors.
    """
    last = A.shape[-1] == 36
    first = A.ndim == 5 and A.shape[1] == 36
    if last and first:
        raise ValueError(f"shape {tuple(A.shape)} is ambiguous between channel layouts")
    if first:
        A = A.permute(0, 2, 3, 4, 1) if isinstance(A, torch.Tensor) else A.transpose(0, 2, 3, 4, 1)
    elif not last:
        raise ValueError(f"shape {tuple(A.shape)} has no 36-channel axis")
    return A.reshape(A.shape[:-1] + (6, 6))


def cols_to_a36(A66):
    """(B, n, n, n, 6, 6) -> (B, 36, n, n, n), the network's output layout."""
    return A66.reshape(A66.shape[:-2] + (36,)).permute(0, 4, 1, 2, 3)


def _xi(n, dtype, device):
    kk = torch.fft.fftfreq(n, d=1.0 / n, dtype=dtype, device=device)
    return [k / n for k in torch.meshgrid(kk, kk, kk, indexing="ij")]


def _to_tensor(sh):
    """Mandel (..., 6) Fourier coefficients -> (..., 3, 3), shear divided by sqrt 2."""
    s = [sh[..., c] for c in range(6)]
    return torch.stack([
        torch.stack([s[0], s[3] * S2, s[5] * S2], -1),
        torch.stack([s[3] * S2, s[1], s[4] * S2], -1),
        torch.stack([s[5] * S2, s[4] * S2, s[2]], -1)], -2)


def _from_tensor(t):
    """(..., 3, 3) -> Mandel (..., 6), shear multiplied by sqrt 2."""
    r2 = np.sqrt(2.0)
    return torch.stack([t[..., 0, 0], t[..., 1, 1], t[..., 2, 2],
                        t[..., 0, 1] * r2, t[..., 1, 2] * r2, t[..., 0, 2] * r2], -1)


def equilibrium_ratio_sq(A66):
    """Squared equilibrium ratio r_k^2 per volume and load column, shape (B, 6).

    Differentiable. The field is divided by its detached mean absolute value first, which
    leaves the ratio unchanged and keeps float32 finite at 1e11 Pa.
    """
    n = A66.shape[1]
    s = A66 / A66.detach().abs().mean().clamp_min(torch.finfo(A66.dtype).tiny)
    sh = torch.fft.fftn(s.transpose(-1, -2), dim=(1, 2, 3))          # (B, n, n, n, 6 loads, 6)
    t = _to_tensor(sh)                                                # (B, n, n, n, 6, 3, 3)
    xi = _xi(n, A66.dtype, A66.device)
    d = sum(t[..., :, j] * xi[j][..., None, None] for j in range(3))  # (div sigma)_i
    num = torch.view_as_real(d).pow(2).sum((-1, -2))                  # (B, n, n, n, 6)
    mask = torch.ones((n, n, n), dtype=A66.dtype, device=A66.device)
    mask[0, 0, 0] = 0.0
    num = (num * mask[..., None]).sum((1, 2, 3))
    den = torch.view_as_real(sh[:, 0, 0, 0]).pow(2).sum((-1, -2))
    return num / den


def equilibrium_ratio(A66):
    """r_k per volume and load column, shape (B, 6): the square root of equilibrium_ratio_sq."""
    return equilibrium_ratio_sq(A66).sqrt()


def undo_octahedral(oct_, Y, g):
    """Map a (B, 36, n, n, n) field acted on by `oct_.apply(..., g)` back to the data frame.

    The ratio keeps the Nyquist plane, where a flipped axis does not send xi to -xi, so a label
    field after one or two flips scores 1e-3 instead of 1e-5. The loss is taken in the data frame.
    """
    from .microstructure import octahedral_group
    Qs = octahedral_group()
    j = next(i for i, Q in enumerate(Qs) if np.array_equal(Q, Qs[g].T))
    inv, flips, _, M36 = oct_.ops[j]
    if flips:
        Y = torch.flip(Y, [2 + k for k in flips])
    Y = Y.permute([0, 1] + [2 + k for k in inv])
    return torch.einsum("ij,bjxyz->bixyz", M36.to(Y.dtype), Y)


def project_equilibrium(A66):
    """For every xi != 0 replace each column's stress tensor s by P s P, P = I - projector onto span{xi, xi'}.

    xi' is the stored wavevector of the conjugate mode -m. Off the Nyquist planes xi' = -xi and P is
    I - xi xi^T / |xi|^2. On a Nyquist plane the two differ, and annihilating both keeps the real
    part of the inverse transform divergence-free under the solver's metric. The xi = 0 mode is
    untouched. Returns the real part of the inverse transform.
    """
    n = A66.shape[1]
    sh = torch.fft.fftn(A66.transpose(-1, -2), dim=(1, 2, 3))
    t = _to_tensor(sh)
    xi = torch.stack(_xi(n, A66.dtype, A66.device), -1)         # (n, n, n, 3)
    xc = torch.roll(xi.flip((0, 1, 2)), shifts=(1, 1, 1), dims=(0, 1, 2))
    eye = torch.eye(3, dtype=xi.dtype, device=xi.device)
    e1 = xi / (xi ** 2).sum(-1, keepdim=True).sqrt().clamp_min(1e-30)
    w = xc - (xc * e1).sum(-1, keepdim=True) * e1
    w2 = (w ** 2).sum(-1, keepdim=True)
    e2 = torch.where(w2 > 1e-12, w / w2.sqrt().clamp_min(1e-30), torch.zeros_like(w))
    P = eye - e1[..., :, None] * e1[..., None, :] - e2[..., :, None] * e2[..., None, :]
    P[0, 0, 0] = eye
    P = P[:, :, :, None].to(t.dtype)
    tp = P @ t @ P
    out = torch.fft.ifftn(_from_tensor(tp), dim=(1, 2, 3)).real
    return out.transpose(-1, -2).contiguous()


def average_strain_mean(A66, C66):
    """M = <C^-1 A> per volume, shape (B, 6, 6)."""
    return torch.linalg.solve(C66, A66).mean((1, 2, 3))


def project_average_strain(A66, C66):
    """A + C (I - <C^-1 A>), which makes the volume-mean strain localization exactly I."""
    M = average_strain_mean(A66, C66)
    eye = torch.eye(6, dtype=A66.dtype, device=A66.device)
    return A66 + C66 @ (eye - M)[:, None, None, None]


def average_strain_deviation(A66, C66):
    """||<C^-1 A> - I||_F per volume, shape (B,). corpus.py divides this by sqrt 6."""
    M = average_strain_mean(A66, C66)
    return torch.linalg.matrix_norm(M - torch.eye(6, dtype=A66.dtype, device=A66.device))


def read_volume(which, seed):
    """(C21 float32 (n,n,n,21), A float32 (n,n,n,6,6), C_grain[voxels] float64) from one HDF5 file."""
    from .microstructure import upper_triangle
    from .periodic import CORPUS
    with h5py.File(paths.DATA / CORPUS[which] / f"sve_{seed}.hdf5", "r") as f:
        vox = f["voxels"][...]
        Cg = f["C_grain"][...]
        A = f["A"][...]
    return upper_triangle(Cg[vox]).astype(np.float32), A, Cg[vox]


def labels(a):
    torch.set_num_threads(a.threads)
    lo, hi = (int(v) for v in a.seeds.split("-"))
    means, maxes = [], []
    print(f"equilibrium ratio of the stored labels, {a.which}, seeds {lo}..{hi}, float64")
    for s in range(lo, hi + 1):
        _, A, _ = read_volume(a.which, s)
        r = equilibrium_ratio(torch.from_numpy(A.astype(np.float64))[None])[0].numpy()
        means.append(r.mean()); maxes.append(r.max())
        print(f"  sve_{s}: mean {r.mean():.4e}  max {r.max():.4e}", flush=True)
    means, maxes = np.array(means), np.array(maxes)
    print(f"{len(means)} volumes x 6 loads: mean of means {means.mean():.4e}, worst load {maxes.max():.4e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["labels"])
    ap.add_argument("--which", default="cu", choices=["cu", "z5", "z8"])
    ap.add_argument("--seeds", default="1160-1199")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    labels(a)


if __name__ == "__main__":
    main()
