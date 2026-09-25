"""Local stress sigma(x) = A(x) : E_bar and the scalar measures read from it.

    from piml.fields import stress_field, measure, summary
    sigma = stress_field(A, E_bar)          # (n1, n2, n3, 3, 3) in Pa
    vm = measure(sigma, "vm")               # von Mises stress, (n1, n2, n3) in Pa

Measures by name: the components xx, yy, zz, xy, yz, xz; vm, the von Mises stress sqrt(3/2 s:s) of the
deviator s; p, the mean stress trace / 3, positive in tension (the negative of the pressure); s1, s2
and s3, the principal stresses in descending order; tau, the maximum shear stress (s1 - s3) / 2.
grain_mean replaces a field by its average over each grain of a voxel map, and summary returns the
statistics that piml.infer prints.
"""

import numpy as np

from .loading import symmetric
from .microstructure import MANDEL_F, MANDEL_MAP, to_mandel2, von_mises

MEASURES = ("xx", "yy", "zz", "xy", "yz", "xz", "vm", "p", "s1", "s2", "s3", "tau")
COMPONENT = {"xx": (0, 0), "yy": (1, 1), "zz": (2, 2), "xy": (0, 1), "yz": (1, 2), "xz": (0, 2)}
ROWS, COLS = np.array(MANDEL_MAP).T
PERCENTILES = (50, 99, 99.9)
LABELS = {**{c: f"sigma_{c}" for c in COMPONENT}, "vm": "von Mises stress", "p": "mean stress",
          "s1": "largest principal stress", "s2": "middle principal stress",
          "s3": "smallest principal stress", "tau": "maximum shear stress"}


def stress_field(A, E_bar):
    """sigma (n1, n2, n3, 3, 3) float64 in Pa under the symmetric macroscopic strain tensor E_bar (3, 3)."""
    E_bar = symmetric(E_bar, "E_bar")
    s = (np.asarray(A, dtype=np.float64) @ to_mandel2(E_bar)) / MANDEL_F
    sigma = np.empty(s.shape[:-1] + (3, 3))
    sigma[..., ROWS, COLS] = s
    sigma[..., COLS, ROWS] = s
    return sigma


def measure(sigma, name):
    """One scalar field of shape sigma.shape[:-2], in Pa; name is one of MEASURES."""
    if name in COMPONENT:
        return sigma[..., COMPONENT[name][0], COMPONENT[name][1]]
    if name == "vm":
        return von_mises(sigma)
    if name == "p":
        return np.trace(sigma, axis1=-2, axis2=-1) / 3
    if name in ("s1", "s2", "s3", "tau"):
        ev = np.linalg.eigvalsh(sigma)[..., ::-1]
        return (ev[..., 0] - ev[..., 2]) / 2 if name == "tau" else ev[..., int(name[1]) - 1]
    raise ValueError(f"unknown measure {name!r}; use one of {MEASURES}")


def grain_mean(field, voxels):
    """field with every voxel replaced by the mean over its grain; voxels holds grain ids 0 to G - 1."""
    voxels = np.asarray(voxels)
    if np.shape(field)[:voxels.ndim] != voxels.shape:
        raise ValueError(f"field shape {np.shape(field)} does not start with the voxel grid {voxels.shape}")
    ids = voxels.ravel()
    flat = np.asarray(field, dtype=np.float64).reshape(ids.size, -1)
    count = np.maximum(np.bincount(ids), 1)
    means = np.stack([np.bincount(ids, weights=flat[:, k]) for k in range(flat.shape[1])], -1)
    return (means / count[:, None])[ids].reshape(np.shape(field))


def relative_error(sigma, sigma_ref):
    """||sigma - sigma_ref|| / ||sigma_ref||, Frobenius norm over every voxel and tensor component."""
    return float(np.linalg.norm(sigma - sigma_ref) / np.linalg.norm(sigma_ref))


def summary(sigma):
    """Mean stress (xx, yy, zz, xy, yz, xz) and the von Mises mean, percentiles and maximum, in Pa."""
    vm = von_mises(sigma).ravel()
    out = {"mean": sigma.reshape(-1, 3, 3).mean(0)[ROWS, COLS], "vm_mean": float(vm.mean()),
           "vm_max": float(vm.max())}
    out.update({f"vm_p{q:g}": float(v) for q, v in zip(PERCENTILES, np.percentile(vm, PERCENTILES))})
    return out
