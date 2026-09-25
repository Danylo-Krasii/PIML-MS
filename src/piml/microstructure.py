"""Crystal stiffness, orientations and Mandel notation.

Builds the per-voxel stiffness field C(x) from grain orientations, converts between tensor and
Mandel forms (component order xx, yy, zz, xy, yz, xz, shear scaled by sqrt 2), and supplies the
48 cube symmetries that the training augmentation uses.

C11, C12 and C44 are the cubic constants of copper, the constants Yabansu, Patel & Kalidindi
(2014) use, with Zener ratio 3.21.
"""

import numpy as np

C11, C12, C44 = 168.4e9, 121.4e9, 75.4e9   # copper, Pa


def cubic_stiffness(c11=C11, c12=C12, c44=C44):
    """Fourth-order cubic stiffness in the crystal frame."""
    C = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    if i == j == k == l:
                        C[i, j, k, l] = c11
                    elif i == j and k == l:
                        C[i, j, k, l] = c12
                    elif (i == k and j == l) or (i == l and j == k):
                        C[i, j, k, l] = c44
    return C


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def _Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def _Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def orientation_matrix(angles, source="ansys"):
    """Crystal orientation g from a row of `local_cs`.

    MatViz3D's two solvers write different things into the same `local_cs` field, and
    nothing in the file says which. The wrong choice gives a wrong C(x) without an error.

      source="ansys"  files from MatViz3D's ANSYS solver.
                      (THXY, THYZ, THZX) in DEGREES, written by
                      TextureLibrary::matrixToAnsys.
                          g = (Ry(THZX) Rx(THYZ) Rz(THXY))^T

      source="fft"    files written by the FFT solver, i.e. anything from
                      --solver fft. Bunge (phi1, Phi, phi2) in RADIANS, see
                      fft_solver_session.hpp line 344.
                          g = (Rz(phi2) Rx(Phi) Rz(phi1))^T

    Tell them apart by range: radians land in [0, 2*pi], degrees in [-180, 180].
    """
    a = np.asarray(angles, dtype=float)
    if source == "ansys":
        t1, t2, t3 = np.radians(a)
        return (_Ry(t3) @ _Rx(t2) @ _Rz(t1)).T
    if source == "fft":
        p1, P, p2 = a                      # already radians
        return (_Rz(p2) @ _Rx(P) @ _Rz(p1)).T
    raise ValueError(f"unknown source {source!r}")


def detect_source(local_cs):
    """Guess the convention from the value range. Radians never exceed 2*pi."""
    return "fft" if np.abs(np.asarray(local_cs)).max() <= 2 * np.pi + 1e-6 else "ansys"


def stiffness_field(voxels, local_cs, C_crystal=None, source=None):
    """C(x) of shape (n, n, n, 3, 3, 3, 3) in the sample frame.

    `local_cs` is indexed by grain id, so row 0 is unused and row g belongs to
    grain g. Grain ids in `voxels` run from 1.
    """
    if C_crystal is None:
        C_crystal = cubic_stiffness()
    if source is None:
        source = detect_source(local_cs)
    per_grain = np.array([
        np.einsum('pi,qj,rk,sl,ijkl->pqrs', g, g, g, g, C_crystal)
        for g in (orientation_matrix(a, source) for a in local_cs)
    ])
    return per_grain[voxels]


def applied_strain(eps_as_loading):
    """The (3, 3) macroscopic strain tensor. Stored values are tensor shear."""
    e = eps_as_loading
    return np.array([[e[0], e[3], e[5]],
                     [e[3], e[1], e[4]],
                     [e[5], e[4], e[2]]], dtype=float)


def von_mises(T):
    d = T - np.eye(3) * np.trace(T, axis1=-2, axis2=-1)[..., None, None] / 3
    return np.sqrt(1.5 * np.einsum('...ij,...ij->...', d, d))


MANDEL_MAP = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]
MANDEL_F = np.array([1.0, 1.0, 1.0, np.sqrt(2), np.sqrt(2), np.sqrt(2)])


def to_mandel4(C):
    """(..., 3, 3, 3, 3) -> (..., 6, 6). Mandel, so the 6x6 rotation is orthogonal."""
    out = np.empty(C.shape[:-4] + (6, 6))
    for I, (i, j) in enumerate(MANDEL_MAP):
        for J, (k, l) in enumerate(MANDEL_MAP):
            out[..., I, J] = MANDEL_F[I] * MANDEL_F[J] * C[..., i, j, k, l]
    return out


def to_mandel2(T):
    """(..., 3, 3) -> (..., 6)."""
    return np.stack([MANDEL_F[I] * T[..., i, j] for I, (i, j) in enumerate(MANDEL_MAP)], -1)


def upper_triangle(C6):
    """The 21 independent Mandel components, (..., 6, 6) -> (..., 21)."""
    i, j = np.triu_indices(6)
    return C6[..., i, j]


def octahedral_group():
    """The 48 signed permutation matrices of the cube, proper and improper.

    Elasticity is even-rank, so reflections are symmetries too. Returned as a
    list of integer (3, 3) arrays with the identity first.
    """
    import itertools
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            Q = np.zeros((3, 3), dtype=int)
            for j in range(3):
                Q[perm[j], j] = signs[j]
            out.append(Q)
    out.sort(key=lambda Q: -int(np.array_equal(Q, np.eye(3, dtype=int))))
    return out


def mandel_rotation(Q):
    """The (6, 6) matrix R with mandel2(Q T Q^T) = R mandel2(T) for symmetric T.

    Orthogonal in Mandel form, which is the reason for the convention: a
    stiffness or localization tensor then transforms as R M R^T.
    """
    R = np.zeros((6, 6))
    for I, (i, j) in enumerate(MANDEL_MAP):
        T = np.zeros((3, 3))
        if i == j:
            T[i, i] = 1.0
        else:
            T[i, j] = T[j, i] = 1.0 / np.sqrt(2)
        R[:, I] = to_mandel2(Q @ T @ Q.T)
    return R


def channel_maps(Q):
    """Linear maps on the 21 upper-triangle and 36 full channels for C' = R C R^T."""
    R = mandel_rotation(Q)
    iu, ju = np.triu_indices(6)
    M21 = np.zeros((21, 21))
    for k, (i, j) in enumerate(zip(iu, ju)):
        E = np.zeros((6, 6)); E[i, j] = E[j, i] = 1.0
        M21[:, k] = (R @ E @ R.T)[iu, ju]
    M36 = np.zeros((36, 36))
    for k in range(36):
        E = np.zeros(36); E[k] = 1.0
        M36[:, k] = (R @ E.reshape(6, 6) @ R.T).reshape(36)
    return M21, M36


def spatial_action(Q):
    """Axis permutation and flips that move a voxel field by Q: F'(x) = F(Q^T x).

    With Q[p[j], j] = s[j], the value at index i lands from index s_j * i_{p(j)}
    on axis j, so output axis k reads input axis p^-1(k), flipped where s < 0.
    Apply the flips to the INPUT axes first, then transpose: `np.transpose(
    np.flip(F, flips), inv)`. Flipping after the transpose flips the wrong axes.
    """
    p = [int(np.argmax(np.abs(Q[:, j]))) for j in range(3)]
    s = [int(Q[p[j], j]) for j in range(3)]
    inv = [p.index(k) for k in range(3)]
    flips = [j for j in range(3) if s[j] < 0]
    return inv, flips
