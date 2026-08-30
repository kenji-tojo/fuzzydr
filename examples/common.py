# examples/common.py
#
# Shared helpers for the fuzzydr examples.
#
# Nothing here is part of the fuzzydr API: these are small, dependency-light
# utilities (torch, numpy, imageio) that keep the example scripts focused on
# the renderer itself.
#
# Sections
# --------
#   1. Paths and devices  - REPO_ROOT, DEFAULT_MESH, select_device, make_out_dir
#   2. Mesh I/O           - load_obj_mesh, save_obj_with_vertex_colors
#   3. Camera math        - look_at, perspective, SceneCamera,
#                           spatial_lr_scale
#   4. Geometry           - vertex_normals, normals_to_rgb
#   5. Output helpers     - to_u8, save_png, open_video, save_loss_curve,
#                           save_loss_plot
#   6. Primitive sets     - make_random_triangles, make_random_lines,
#                           make_random_points
#   7. Checkpoint I/O     - save_checkpoint, load_checkpoint

from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch


# =============================================================================
# 1. Paths and devices
# =============================================================================

# Resolved from this file rather than the working directory, so the examples
# find their default asset no matter where they are launched from.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MESH = REPO_ROOT / "assets" / "bunny.obj"


def select_device(name: str, gpu_id: int = 0) -> torch.device:
    """Return a ``torch.device``, falling back to CPU when CUDA is missing.

    fuzzydr renders on Vulkan either way; the torch device only decides where
    the tensors live, and whether the rasterizer uses CUDA/Vulkan interop or
    stages through host memory.
    """
    if name.lower() == "cuda":
        if not torch.cuda.is_available():
            print("[torch] CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu")
        torch.cuda.set_device(int(gpu_id))
        return torch.device("cuda")
    return torch.device("cpu")


def make_out_dir(path: str | Path) -> Path:
    """Create (if needed) and return an output directory."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


# =============================================================================
# 2. Mesh I/O
# =============================================================================

def load_obj_mesh(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load positions and triangle indices from a Wavefront OBJ file.

    Only the ``v`` and ``f`` records are read - the examples shade meshes from
    geometric normals, so texture coordinates and stored normals are ignored.
    Faces with more than three corners are triangulated as a fan.

    Returns
    -------
    verts : float32 [N, 3]
    faces : uint32  [M, 3]
    """
    positions: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []

    with open(path, "r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            if line.startswith("v "):
                x, y, z = line.split()[1:4]
                positions.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                # Every corner form ("v", "v/vt", "v//vn", "v/vt/vn") starts
                # with the position index, which is all this loader needs.
                corners = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                # OBJ indices are 1-based; negative ones count back from the
                # vertices declared so far.
                corners = [i - 1 if i > 0 else len(positions) + i for i in corners]
                for k in range(1, len(corners) - 1):
                    triangles.append((corners[0], corners[k], corners[k + 1]))

    if not positions or not triangles:
        raise ValueError(f"{path}: no triangles found")

    verts = torch.tensor(positions, dtype=torch.float32)
    faces = torch.tensor(triangles, dtype=torch.int64)
    if int(faces.min()) < 0 or int(faces.max()) >= verts.shape[0]:
        raise ValueError(f"{path}: a face references a vertex outside the file")

    return verts.contiguous(), faces.to(torch.uint32).contiguous()


def save_obj_with_vertex_colors(
    path: str | Path,
    verts: torch.Tensor,    # float32 [N, 3]
    faces: torch.Tensor,    # uint32  [M, 3]
    colors: torch.Tensor,   # float32 [N, 3]  RGB in [0, 1]
) -> None:
    """Write an OBJ using the ``v x y z r g b`` vertex-colour extension."""
    verts = verts.detach().cpu()
    faces = faces.detach().cpu()
    colors = colors.detach().cpu()

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("verts must have shape [N, 3]")
    if colors.shape != verts.shape:
        raise ValueError("colors must have shape [N, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [M, 3]")

    with open(path, "w", encoding="utf-8") as fp:
        fp.write("# fuzzydr example output\n")
        for (x, y, z), (r, g, b) in zip(verts.tolist(), colors.tolist()):
            fp.write(f"v {x:.9g} {y:.9g} {z:.9g} {r:.6g} {g:.6g} {b:.6g}\n")
        for i0, i1, i2 in (faces.to(torch.int64) + 1).tolist():
            fp.write(f"f {i0} {i1} {i2}\n")
    print(f"Wrote {path}")


# =============================================================================
# 3. Camera math
# =============================================================================

def look_at(
    eye: torch.Tensor,
    target: torch.Tensor,
    up: torch.Tensor | None = None,
) -> torch.Tensor:
    """Right-handed look-at view matrix, float32 [4, 4] on CPU."""
    eye = eye.to(device="cpu", dtype=torch.float32)
    target = target.to(device="cpu", dtype=torch.float32)
    if up is None:
        up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    def _unit(v: torch.Tensor) -> torch.Tensor:
        return v / torch.linalg.norm(v).clamp_min(1e-12)

    f = _unit(target - eye)
    r = _unit(torch.cross(f, up.to(torch.float32), dim=0))
    u = torch.cross(r, f, dim=0)

    m = torch.eye(4, dtype=torch.float32)
    m[0, :3], m[0, 3] = r, -torch.dot(r, eye)
    m[1, :3], m[1, 3] = u, -torch.dot(u, eye)
    m[2, :3], m[2, 3] = -f, torch.dot(f, eye)
    return m


def perspective(fovy_deg: float, aspect: float, z_near: float, z_far: float) -> torch.Tensor:
    """Right-handed perspective matrix in Vulkan conventions, float32 [4, 4].

    Depth maps to [0, 1] and the Y axis is flipped, so the result can be fed
    straight to ``fuzzydr.rasterize`` as (part of) ``viewproj``.
    """
    f = 1.0 / math.tan(0.5 * math.radians(fovy_deg))
    m = torch.zeros((4, 4), dtype=torch.float32)
    m[0, 0] = f / aspect
    m[1, 1] = -f                                   # Vulkan Y-flip
    m[2, 2] = z_far / (z_near - z_far)
    m[2, 3] = (z_far * z_near) / (z_near - z_far)
    m[3, 2] = -1.0
    return m


@dataclasses.dataclass(frozen=True)
class SceneCamera:
    """A perspective camera framing a scene's bounding box.

    Every matrix and vector it hands out is a CPU float32 tensor, which is what
    ``fuzzydr.rasterize`` wants for ``viewproj`` and ``campos``.  The near and
    far planes are baked into ``proj`` from the framing distance, so all the
    eye positions produced here must stay on the ``dist`` sphere around
    ``center`` - which is what :meth:`orbit_eye` and :meth:`sphere_eyes` do.
    """

    center: torch.Tensor    # float32 [3]  bounding-box centre
    radius: float           # half the bounding-box diagonal
    dist: float             # eye distance from the centre
    proj: torch.Tensor      # float32 [4, 4]

    @classmethod
    def frame(
        cls,
        verts: torch.Tensor,
        width: int,
        height: int,
        *,
        fovy_deg: float = 45.0,
        margin: float = 0.96,
    ) -> "SceneCamera":
        """Fit a camera to the bounding box of ``verts``.

        The distance follows from the field of view - ``radius / sin(fovy/2)``
        is where the bounding sphere exactly fills the frame - so narrowing
        ``fovy_deg`` backs the camera off by the matching amount and keeps the
        subject the same size.  Lowering the field of view is therefore the one
        knob for a flatter, more orthographic-looking render.  ``margin``
        scales that distance; below 1 the box overfills slightly, which is fine
        because a real mesh does not fill its bounding sphere.
        """
        v = verts.detach().to(device="cpu", dtype=torch.float32)
        lo = v.min(dim=0).values
        hi = v.max(dim=0).values
        center = 0.5 * (lo + hi)
        radius = max(0.5 * float(torch.linalg.norm(hi - lo)), 1e-3)

        dist = margin * radius / math.sin(math.radians(fovy_deg) * 0.5)
        proj = perspective(
            fovy_deg,
            float(width) / float(height),
            z_near=max(1e-3, dist - radius),
            z_far=dist + radius,
        )
        return cls(center=center.contiguous(), radius=radius, dist=dist, proj=proj)

    def orbit_eye(self, azimuth_rad: float, *, elevation: float = 0.3) -> torch.Tensor:
        """Eye position on the framing sphere at the given azimuth.

        ``elevation`` tilts the view above the horizon; it is the tangent of
        the elevation angle rather than the angle itself, so 0 looks level and
        0.3 looks slightly down at the scene.
        """
        d = torch.tensor(
            [math.sin(azimuth_rad), float(elevation), math.cos(azimuth_rad)],
            dtype=torch.float32,
        )
        return (self.center + self.dist * d / torch.linalg.norm(d)).contiguous()

    def sphere_eyes(self, count: int) -> torch.Tensor:
        """``count`` eye positions spread over the framing sphere.

        Uses a Fibonacci lattice, which is far more even than sampling the
        azimuth/elevation rectangle.

        Returns
        -------
        eyes : float32 [count, 3]
        """
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        dirs = torch.empty((count, 3), dtype=torch.float32)
        for i in range(count):
            y = 1.0 - 2.0 * (i + 0.5) / count
            r = math.sqrt(max(0.0, 1.0 - y * y))
            t = golden_angle * i
            dirs[i] = torch.tensor([math.cos(t) * r, y, math.sin(t) * r])
        return (self.center + self.dist * dirs).contiguous()

    def view_proj(self, eye: torch.Tensor) -> torch.Tensor:
        """Row-major view-projection matrix for an eye position, float32 [4, 4]."""
        return (self.proj @ look_at(eye, self.center)).contiguous()


def spatial_lr_scale(eyes: torch.Tensor) -> float:
    """Scene scale for position learning rates, as 3DGS defines it.

    1.1 times the largest distance from a camera to the camera centroid.  The
    fitting examples quote their position learning rate against this, so the
    same number works whatever size the scene is.
    """
    return 1.1 * float((eyes - eyes.mean(dim=0)).norm(dim=1).max())


# =============================================================================
# 4. Geometry
# =============================================================================

def vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area-weighted per-vertex normals, L2-normalised.

    Parameters
    ----------
    verts : float32 [N, 3]
    faces : uint32  [M, 3]

    Returns
    -------
    normals : float32 [N, 3]
    """
    v = verts.to(torch.float32)
    f = faces.to(torch.int64)
    # Un-normalised face normals have length proportional to twice the
    # triangle area, which is exactly the area weighting we want.
    face_n = torch.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], dim=1)

    n = torch.zeros_like(v)
    for k in range(3):
        n.index_add_(0, f[:, k], face_n)
    return (n / torch.linalg.norm(n, dim=1, keepdim=True).clamp_min(1e-12)).contiguous()


def normals_to_rgb(normals: torch.Tensor) -> torch.Tensor:
    """Map unit normals from [-1, 1] to RGB in [0, 1]."""
    return torch.clamp(normals.to(torch.float32) * 0.5 + 0.5, 0.0, 1.0).contiguous()


# =============================================================================
# 5. Output helpers
# =============================================================================

def to_u8(img: torch.Tensor) -> torch.Tensor:
    """Clamp a float image in [0, 1] and round it to uint8."""
    return (torch.clamp(img, 0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)


def save_png(path: str | Path, img: torch.Tensor) -> None:
    """Write ``img`` ([H, W, 3], float in [0, 1] or uint8) as a PNG."""
    if img.dtype != torch.uint8:
        img = to_u8(img)
    imageio.imwrite(str(path), img.detach().cpu().numpy())
    print(f"Wrote {path}")


def open_video(path: str | Path, *, fps: int = 30):
    """Open an H.264 MP4 writer with the settings shared by the examples.

    Frame dimensions should be multiples of 16; libx264 silently rescales
    anything else, which is why the examples pick resolutions that already are.
    """
    return imageio.get_writer(
        str(path), fps=fps, codec="libx264", pixelformat="yuv420p", quality=8
    )


def save_loss_curve(path: str | Path, losses: list[float]) -> None:
    """Write a tab-separated ``iteration<TAB>loss`` table."""
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("# iteration\tloss\n")
        for i, value in enumerate(losses, start=1):
            fp.write(f"{i}\t{value:.9g}\n")
    print(f"Wrote {path}")


def save_loss_plot(path: str | Path, losses: list[float]) -> None:
    """Write a log-scale loss curve as a PNG.

    The examples draw one view per iteration, so the raw curve mostly records
    which view an iteration happened to use.  A running mean over roughly one
    sweep of the views is drawn on top, which is the part worth reading.
    """
    # Imported here rather than at module scope: matplotlib is slow to import
    # and only the fitting examples plot anything.
    import matplotlib
    matplotlib.use("Agg")          # no display, so this works over SSH too
    import matplotlib.pyplot as plt

    xs = range(1, len(losses) + 1)
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.plot(xs, losses, linewidth=0.7, alpha=0.35, label="per iteration")

    window = max(1, len(losses) // 100)
    if window > 1:
        cumulative = torch.tensor(losses, dtype=torch.float64).cumsum(0)
        smoothed = (cumulative[window:] - cumulative[:-window]) / window
        ax.plot(range(window + 1, len(losses) + 1), smoothed.numpy(),
                linewidth=1.4, label=f"mean of {window}")
        ax.legend(frameon=False)

    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


# =============================================================================
# 6. Primitive sets
#
# Random initialisations for the inverse-rendering examples.  Each builder
# returns unshared vertices plus a trivial index buffer, so a primitive can be
# dropped by removing one row of the index buffer.
# =============================================================================

def _uniform_in_box(count: int, bbox_min: torch.Tensor, bbox_max: torch.Tensor) -> torch.Tensor:
    lo = bbox_min.to(device="cpu", dtype=torch.float32)
    hi = bbox_max.to(device="cpu", dtype=torch.float32)
    return torch.rand((count, 3), dtype=torch.float32) * (hi - lo) + lo


def make_random_triangles(
    num_tris: int,
    size: float,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random triangles with centres uniform in the box and corners at ``size``.

    Returns
    -------
    verts : float32 [num_tris * 3, 3]
    faces : uint32  [num_tris, 3]   sequential triples [0,1,2], [3,4,5], ...
    """
    centers = _uniform_in_box(num_tris, bbox_min, bbox_max)
    offsets = torch.randn((num_tris, 3, 3), dtype=torch.float32)
    offsets = offsets / offsets.norm(dim=2, keepdim=True).clamp_min(1e-8) * float(size)

    verts = (centers[:, None, :] + offsets).reshape(num_tris * 3, 3).contiguous()
    faces = _sequential_indices(num_tris, 3)
    return verts, faces


def make_random_lines(
    num_lines: int,
    length: float,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random segments with midpoints uniform in the box and unit-random directions.

    Returns
    -------
    verts : float32 [num_lines * 2, 3]
    lines : uint32  [num_lines, 2]   sequential pairs [0,1], [2,3], ...
    """
    mids = _uniform_in_box(num_lines, bbox_min, bbox_max)
    dirs = torch.randn((num_lines, 3), dtype=torch.float32)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8) * (0.5 * float(length))

    verts = torch.stack([mids - dirs, mids + dirs], dim=1).reshape(num_lines * 2, 3).contiguous()
    lines = _sequential_indices(num_lines, 2)
    return verts, lines


def make_random_points(
    num_points: int,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random points uniform in the box.

    Returns
    -------
    verts  : float32 [num_points, 3]
    points : uint32  [num_points, 1]  the identity index buffer

    The index buffer keeps a trailing axis so that faces, lines and points can
    share the same ``[num_prims, arity]`` handling; ``fuzzydr.rasterize_points``
    wants it flat, so reshape before the call.
    """
    verts = _uniform_in_box(num_points, bbox_min, bbox_max).contiguous()
    return verts, _sequential_indices(num_points, 1)


def _sequential_indices(num_prims: int, arity: int) -> torch.Tensor:
    """``[[0..arity-1], [arity..2*arity-1], ...]`` as uint32 [num_prims, arity]."""
    idx = torch.arange(num_prims * arity, dtype=torch.int32).reshape(num_prims, arity)
    # torch has no int32 -> uint32 cast, but the bit patterns already agree.
    return idx.view(torch.uint32).contiguous()


# =============================================================================
# 7. Checkpoint I/O
#
# Compressed NumPy archive holding one set of fitted primitives:
#
#   verts          float32 [N, 3]   world-space vertex positions
#   colors         float32 [N, 3]   per-vertex RGB in [0, 1]
#   radius         float32 [N]      per-vertex world-space radius (quad lines)
#   prims          uint32  [M, A]   index buffer; A is 3 (faces), 2 (lines)
#                                   or 1 (points)
#   opacity_logit  float32 [M]      raw logits; sigmoid gives the opacity
#   meta_json      str              JSON metadata, see below
#
# Storing raw logits rather than opacities lets a reader re-threshold the
# result without retraining.
# =============================================================================

def save_checkpoint(
    path: str | Path,
    *,
    primitive: str,                     # "faces" | "lines" | "points"
    verts: torch.Tensor,                # float32 [N, 3]
    colors: torch.Tensor,               # float32 [N, 3]
    prims: torch.Tensor,                # uint32  [M, A]
    opacity_logit: torch.Tensor,        # float32 [M]
    radius: torch.Tensor | None = None,  # float32 [N]
    meta: dict | None = None,
) -> None:
    """Save a fitted primitive set so it can be re-loaded or viewed later."""
    if primitive not in ("faces", "lines", "points"):
        raise ValueError(f"primitive must be faces, lines or points; got {primitive!r}")

    n_verts = int(verts.shape[0])
    if radius is None:
        radius = torch.zeros(n_verts, dtype=torch.float32)

    header = {
        "format_version": 1,
        "primitive": primitive,
        "n_verts": n_verts,
        "n_prims": int(prims.shape[0]),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if meta:
        header.update(meta)

    def _np(t: torch.Tensor, dtype) -> np.ndarray:
        return t.detach().cpu().to(dtype).numpy()

    np.savez_compressed(
        str(path),
        verts=_np(verts, torch.float32),
        colors=_np(colors, torch.float32),
        radius=_np(radius, torch.float32),
        prims=_np(prims, torch.uint32),
        opacity_logit=_np(opacity_logit, torch.float32),
        meta_json=np.array(json.dumps(header)),
    )
    print(f"Wrote {path} ({header['n_verts']} verts, "
          f"{header['n_prims']} {primitive})")


def load_checkpoint(path: str | Path) -> dict:
    """Load a checkpoint written by :func:`save_checkpoint`.

    Returns a dict of NumPy arrays plus a parsed ``"meta"`` entry.
    """
    data = np.load(str(path), allow_pickle=False)
    out = {key: data[key] for key in data.files}
    out["meta"] = json.loads(str(out.pop("meta_json")))
    return out
