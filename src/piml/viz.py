"""Pictures of a stress field: mid-plane slices, the distribution over voxels, a 3D cube, and a VTK file.

    from piml.viz import plot_slices, plot_distribution, plot_cube, write_vti
    plot_slices({"label": vm_label, "prediction": vm}, "out/slices_vm.png", reference="label")
    write_vti("out/volume.vti", {"sigma": sigma, "vm": vm, "grain": voxels})

Scalar fields are (n1, n2, n3) arrays indexed [x, y, z] as in the corpus, in Pa; the plots draw them in
MPa. plot_slices puts every field of one call on one colour scale and, given a reference, draws the
signed difference of each other field from it in a second row. Fields are drawn in the nine bands of the
ANSYS Mechanical legend, blue at the bottom of the scale and red at its top; the scale runs from zero when
the field is nowhere negative and from its minimum otherwise. The difference row keeps a diverging scale
centred on zero. The plot
functions return the matplotlib Figure and also save it when given a path; they use matplotlib's object
interface, so they leave the caller's backend alone. write_vti writes VTK XML image data in ASCII with
one cell per voxel, which ParaView opens directly; a (n1, n2, n3, 3, 3) array becomes a 9-component array
in row-major order, and the first such array is declared the active tensor.
"""

from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D

INK, INK2 = "#0b0b0b", "#52514e"
FIELD = LinearSegmentedColormap.from_list("field", ["#f3f0f7", "#8a6fc0", "#3b2a6e", "#150b2e"])
ERROR = LinearSegmentedColormap.from_list("error", ["#8c5a1e", "#f2f1ec", "#2a8f8a"])
SIGNED = colormaps["PuOr_r"]
ANSYS = ListedColormap(["#0000ff", "#00b2ff", "#00ffff", "#00ffb2", "#00ff00", "#b2ff00", "#ffff00", "#ffb200",
                        "#ff0000"], name="ansys")
SERIES = ["#0b0b0b", "#2a78d6", "#eb6834", "#2a8f8a", "#8a6fc0", "#8c5a1e"]
PLANE_AXES = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}
DEPTH = (0.42, 0.32)


def plane(field, axis="z", index=None):
    """(section, index): the 2D cut of a (n1, n2, n3) field normal to axis, rows along the later axis.

    The cut is at index, by default the middle; its columns run along the earlier of the two in-plane
    axes and its rows along the later, as PLANE_AXES names them.
    """
    if axis not in PLANE_AXES:
        raise ValueError(f"axis must be x, y or z, got {axis!r}")
    k = "xyz".index(axis)
    index = field.shape[k] // 2 if index is None else index
    return np.take(field, index, axis=k).T, index


def _colorbar(fig, mappable, ax, unit, shrink=0.9):
    cb = fig.colorbar(mappable, ax=ax, shrink=shrink, pad=0.02)
    if mappable.get_cmap() is ANSYS:
        ticks = np.linspace(*mappable.get_clim(), ANSYS.N + 1)
        cb.set_ticks(ticks, labels=[f"{t:.5g}" for t in ticks])
    cb.set_label(unit, color=INK2, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=INK2)


def _scale(lo, hi, cmap):
    """(lo, hi, cmap): the ANSYS bands from zero, or from lo when the field goes negative, up to hi."""
    if cmap is not None:
        return lo, hi, cmap
    return min(lo, 0.0), hi, ANSYS


def _save(fig, path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    return fig


def plot_slices(fields, path=None, axis="z", index=None, reference=None, title=None, scale=1e-6, unit="MPa",
                cmap=None):
    """One panel per field on a shared colour scale, and the difference from the reference field below."""
    names = list(fields)
    if not names:
        raise ValueError("no field to plot")
    if reference is not None and (reference not in fields or len(names) < 2):
        raise ValueError(f"reference {reference!r} must be one of at least two fields, got {names}")
    cuts = {k: plane(np.asarray(v, dtype=np.float64), axis, index) for k, v in fields.items()}
    idx = next(iter(cuts.values()))[1]
    img = {k: c[0] * scale for k, c in cuts.items()}
    lo, hi, cmap = _scale(min(v.min() for v in img.values()), max(v.max() for v in img.values()), cmap)
    rows = 2 if reference is not None else 1
    fig = Figure(figsize=(2.2 * len(names) + 0.9, 2.35 * rows + 0.35), layout="constrained")
    axes = fig.subplots(rows, len(names), squeeze=False)
    h, v = PLANE_AXES[axis]
    for c, k in enumerate(names):
        ax = axes[0, c]
        im = ax.imshow(img[k], origin="lower", cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(k, color=INK, fontsize=9)
        ax.set_xlabel(f"{h}, voxel", color=INK2, fontsize=7.5)
        if c == 0:
            ax.set_ylabel(f"{v}, voxel", color=INK2, fontsize=7.5)
        ax.tick_params(labelsize=7, colors=INK2)
    _colorbar(fig, im, list(axes[0]), unit)
    if reference is not None:
        diff = {k: img[k] - img[reference] for k in names if k != reference}
        m = max(np.abs(d).max() for d in diff.values()) or 1.0
        first = next(k for k in names if k != reference)
        for c, k in enumerate(names):
            ax = axes[1, c]
            if k == reference:
                ax.axis("off")
                continue
            im = ax.imshow(diff[k], origin="lower", cmap=ERROR, vmin=-m, vmax=m, interpolation="nearest")
            ax.set_title(f"{k} - {reference}", color=INK, fontsize=9)
            ax.set_xlabel(f"{h}, voxel", color=INK2, fontsize=7.5)
            if k == first:
                ax.set_ylabel(f"{v}, voxel", color=INK2, fontsize=7.5)
            ax.tick_params(labelsize=7, colors=INK2)
        _colorbar(fig, im, list(axes[1]), unit)
    fig.suptitle(f"{title + ', ' if title else ''}plane {axis} = {idx}", color=INK, fontsize=10)
    return _save(fig, path)


def plot_distribution(samples, path=None, xlabel="von Mises stress", scale=1e-6, unit="MPa", bins=120,
                      colors=None):
    """Density of each sample over its voxels on common bins, log scale, with the 1st and 99th percentiles."""
    data = {k: np.ravel(v).astype(np.float64) * scale for k, v in samples.items()}
    lo = min(d.min() for d in data.values())
    hi = max(d.max() for d in data.values())
    if not hi > lo:
        raise ValueError(f"the samples take one value, {lo:g} {unit}; there is no distribution to draw")
    edges = np.linspace(lo, hi, bins + 1)
    fig = Figure(figsize=(4.6, 3.0), layout="constrained")
    ax = fig.subplots()
    for i, (k, d) in enumerate(data.items()):
        color = (colors or {}).get(k, SERIES[i % len(SERIES)])
        dens, _ = np.histogram(d, edges, density=True)
        ax.stairs(dens, edges, color=color, lw=1.2, label=k)
        for q in (1, 99):
            ax.axvline(np.percentile(d, q), color=color, lw=0.7, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel(f"{xlabel}, {unit}", color=INK, fontsize=8.5)
    ax.set_ylabel(f"density, 1/{unit}", color=INK, fontsize=8.5)
    ax.tick_params(labelsize=7.5, colors=INK2)
    ax.legend(frameon=False, fontsize=7.5, title="dotted: 1st and 99th percentiles", title_fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return _save(fig, path)


def plot_cube(field, path=None, title=None, scale=1e-6, unit="MPa", lo=None, hi=None, cmap=None):
    """The field on the three visible faces of its volume: front y = 0, top z = n3 - 1, side x = n1 - 1.

    The faces keep the proportions of the grid; lo and hi set the colour range in the plotted unit.
    """
    f = np.asarray(field, dtype=np.float64) * scale
    lo, hi, cmap = _scale(f.min() if lo is None else lo, f.max() if hi is None else hi, cmap)
    sx, sy, sz = np.array(f.shape, dtype=float) / max(f.shape)
    dx, dy = DEPTH[0] * sy, DEPTH[1] * sy
    faces = [(f[:, 0, :].T, Affine2D.from_values(sx, 0, 0, sz, 0, 0)),
             (f[:, :, -1].T, Affine2D.from_values(sx, 0, dx, dy, 0, sz)),
             (f[-1, :, :].T, Affine2D.from_values(dx, dy, 0, sz, sx, 0))]
    fig = Figure(figsize=(3.4, 3.0), layout="constrained")
    ax = fig.subplots()
    for face, tr in faces:
        ax.imshow(face, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, vmin=lo, vmax=hi,
                  interpolation="nearest", transform=tr + ax.transData)
    for pts in ([(0, 0), (sx, 0), (sx, sz), (0, sz)], [(0, sz), (sx, sz), (sx + dx, sz + dy), (dx, sz + dy)],
                [(sx, 0), (sx + dx, dy), (sx + dx, sz + dy), (sx, sz)]):
        ax.add_patch(Polygon(pts, closed=True, fill=False, edgecolor=INK2, lw=0.8))
    ax.set_xlim(-0.04, sx + dx + 0.04)
    ax.set_ylim(-0.04, sz + dy + 0.04)
    ax.set_aspect("equal")
    ax.axis("off")
    _colorbar(fig, ScalarMappable(Normalize(lo, hi), cmap), ax, unit, shrink=0.8)
    if title:
        ax.set_title(title, color=INK, fontsize=9)
    return _save(fig, path)


def write_vti(path, cells, spacing=1.0):
    """VTK XML image data, ASCII, one cell per voxel; cells maps a name to a (n1, n2, n3, ...) array."""
    arrays, shape, tensor = [], None, None
    for name, arr in cells.items():
        a = np.asarray(arr)
        shape = shape or a.shape[:3]
        if a.ndim < 3 or a.shape[:3] != shape:
            raise ValueError(f"{name}: shape {a.shape} does not start with the grid {shape}")
        k = int(np.prod(a.shape[3:], dtype=int))
        tensor = tensor or (name if k == 9 else None)
        flat = a.reshape(shape + (k,)).transpose(2, 1, 0, 3).reshape(-1, k)
        integer = np.issubdtype(a.dtype, np.integer)
        text = "\n".join(" ".join(map(str, row)) for row in flat.astype(np.int64)) if integer else \
            "\n".join(" ".join(f"{x:.9g}" for x in row) for row in flat.astype(np.float32))
        kind = "Int64" if integer else "Float32"
        arrays.append(f'        <DataArray type="{kind}" Name={quoteattr(str(name))} '
                      f'NumberOfComponents="{k}" format="ascii">\n{text}\n        </DataArray>')
    n1, n2, n3 = shape
    extent = f"0 {n1} 0 {n2} 0 {n3}"
    active = f" Tensors={quoteattr(str(tensor))}" if tensor else ""
    xml = ('<?xml version="1.0"?>\n<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n'
           f'  <ImageData WholeExtent="{extent}" Origin="0 0 0" Spacing="{spacing} {spacing} {spacing}">\n'
           f'    <Piece Extent="{extent}">\n      <CellData{active}>\n' + "\n".join(arrays) +
           "\n      </CellData>\n    </Piece>\n  </ImageData>\n</VTKFile>\n")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(xml)
    return Path(path)
