"""Macroscopic loads for a localization field A(x): a strain applied directly, or a stress through <A>.

    from piml.loading import macro_strain
    E_bar = macro_strain(A, case="uniaxial-stress-x", magnitude=100e6)

A(x) maps a macroscopic strain to the local stress, so every load enters as a strain E_bar. A load given
as a macroscopic stress Sigma_bar is converted through the effective stiffness of the volume,
C_eff = <A(x)>, the volume average of A: the mean of sigma(x) = A(x) : E_bar is C_eff : E_bar, so
E_bar = C_eff^-1 : Sigma_bar returns Sigma_bar as the mean stress of the field A it was computed from.
Tensors are symmetric (3, 3) with tensor shear components; Mandel vectors carry sqrt 2 on shear, as in
the corpus.

Named cases, with the magnitude m in Pa for stress cases and as a strain for strain cases:
    uniaxial-stress-x|y|z    Sigma_ii = m, every other stress component zero
    uniaxial-strain-x|y|z    E_ii = m, every other strain component zero
    shear-xy|yz|xz           Sigma_ij = Sigma_ji = m
    biaxial-xy|yz|xz         Sigma_ii = Sigma_jj = m
    hydrostatic              Sigma = m I
"""

import numpy as np

from .microstructure import MANDEL_F, MANDEL_MAP, to_mandel2

AXIS = {"x": 0, "y": 1, "z": 2}
PAIR = {"xy": (0, 1), "yz": (1, 2), "xz": (0, 2)}
LOAD_CASES = ([f"uniaxial-stress-{a}" for a in AXIS] + [f"uniaxial-strain-{a}" for a in AXIS]
              + [f"shear-{p}" for p in PAIR] + [f"biaxial-{p}" for p in PAIR] + ["hydrostatic"])


def from_mandel2(v):
    """Mandel (..., 6) -> symmetric (..., 3, 3), shear divided by sqrt 2."""
    v = np.asarray(v, dtype=np.float64)
    if v.shape[-1:] != (6,):
        raise ValueError(f"a Mandel vector has 6 components, got shape {v.shape}")
    T = np.empty(v.shape[:-1] + (3, 3))
    for I, (i, j) in enumerate(MANDEL_MAP):
        T[..., i, j] = T[..., j, i] = v[..., I] / MANDEL_F[I]
    return T


def symmetric(T, what="tensor"):
    """T as a float64 (3, 3) array, or ValueError when it is not a finite, symmetric 3x3 tensor."""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (3, 3):
        raise ValueError(f"{what} must be a (3, 3) tensor, got shape {T.shape}")
    if not np.isfinite(T).all() or np.abs(T - T.T).max() > 1e-6 * max(np.abs(T).max(), 1e-300):
        raise ValueError(f"{what} must be a finite, symmetric (3, 3) tensor, got {T.tolist()}")
    return T


def effective_stiffness(A):
    """C_eff = <A(x)>, (6, 6) Mandel in Pa: the volume average of A of shape (n1, n2, n3, 6, 6)."""
    return np.asarray(A).reshape(-1, 6, 6).mean(0, dtype=np.float64)


def load_case(case, magnitude):
    """("stress" or "strain", the symmetric (3, 3) tensor) of a named case of LOAD_CASES."""
    T = np.zeros((3, 3))
    name, _, part = case.rpartition("-")
    if case == "hydrostatic":
        return "stress", magnitude * np.eye(3)
    if name in ("uniaxial-stress", "uniaxial-strain") and part in AXIS:
        T[AXIS[part], AXIS[part]] = magnitude
        return name.split("-")[1], T
    if name in ("shear", "biaxial") and part in PAIR:
        i, j = PAIR[part]
        if name == "shear":
            T[i, j] = T[j, i] = magnitude
        else:
            T[i, i] = T[j, j] = magnitude
        return "stress", T
    raise ValueError(f"unknown load case {case!r}; use one of {LOAD_CASES}")


def strain_for_stress(C_eff, Sigma):
    """Symmetric (3, 3) macroscopic strain E_bar with C_eff : E_bar = Sigma, C_eff (6, 6) Mandel."""
    e = np.linalg.solve(np.asarray(C_eff, dtype=np.float64), to_mandel2(symmetric(Sigma, "stress")))
    return from_mandel2(e)


def macro_strain(A=None, case=None, magnitude=None, strain=None, stress=None):
    """E_bar (3, 3) from exactly one of: a named case and its magnitude, a strain tensor or a stress tensor.

    A stress, given directly or by a stress case, is converted with the effective stiffness of A, the
    field whose mean stress should equal it; A is not needed for a strain.
    """
    if [case is not None, strain is not None, stress is not None].count(True) != 1:
        raise ValueError("give exactly one of case, strain or stress")
    if case is not None:
        if magnitude is None:
            raise ValueError(f"load case {case!r} needs a magnitude")
        kind, T = load_case(case, float(magnitude))
        strain, stress = (T, None) if kind == "strain" else (None, T)
    if strain is not None:
        return symmetric(strain, "strain")
    if A is None:
        raise ValueError("a stress load needs A for the effective stiffness")
    return strain_for_stress(effective_stiffness(A), stress)
