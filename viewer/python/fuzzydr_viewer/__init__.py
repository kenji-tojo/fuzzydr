# fuzzydr_viewer/__init__.py
from __future__ import annotations

import os

import numpy as np
from . import _core


# This unnatural ordering is historical; refactoring it would mean keeping the
# backend and the original enums in sync.
_AA_NAMES = {
    "hw_msaa_4x":    0,   # 4x hardware MSAA
    "gaussian_msaa": 1,   # 2x supersample + 6x6 sigma=0.5 Gaussian resolve
    "none":          2,   # 1-sample target, no AA - fastest
    "hw_msaa_2x":    3,   # 2x hardware MSAA
}

# launch() only.  The interactive viewer maps its three UI entries onto these:
#   "none"       -> No AA            (hardware path pinned to 1 sample)
#   "gaussian_msaa" -> Gaussian MSAA 4x (2x grid per axis, count locked)
# launch() accepts the same names as the offline table above.  Each entry is
# (aa_mode, initial sample count); unlike offline, where aa_mode pins one exact
# configuration, the count here is only a starting point since the viewer's
# panel can change it at runtime.  The C++ mode ids differ from the offline
# ones, which is why this is a separate table -- the *names* are what must
# agree, and they do.
_LAUNCH_AA_NAMES = {
    "none":          (2, 1),
    "hw_msaa_2x":    (0, 2),
    "hw_msaa_4x":    (0, 4),
    "gaussian_msaa": (1, 1),
}


def _resolve_launch_aa(aa_mode: str) -> tuple[int, int]:
    if not isinstance(aa_mode, str):
        raise TypeError(
            "launch aa_mode must be a string - the sample count is part of the "
            f"mode name, so an integer would be ambiguous; got {aa_mode!r}. "
            f"Use one of {list(_LAUNCH_AA_NAMES)}."
        )
    if aa_mode not in _LAUNCH_AA_NAMES:
        raise ValueError(
            f"launch aa_mode must be one of {list(_LAUNCH_AA_NAMES)}; got {aa_mode!r}"
        )
    return _LAUNCH_AA_NAMES[aa_mode]

def _resolve_aa_mode(aa_mode: str | int) -> int:
    if isinstance(aa_mode, str):
        name = aa_mode
        if name not in _AA_NAMES:
            raise ValueError(
                f"aa_mode must be one of {list(_AA_NAMES)} or 0..3; got {aa_mode!r}"
            )
        return _AA_NAMES[name]
    i = int(aa_mode)
    if i not in set(_AA_NAMES.values()):
        raise ValueError(f"aa_mode integer must be one of {sorted(_AA_NAMES.values())}; got {aa_mode}")
    return i


# viewer_sh_eval.comp hardcodes 16 coefficients per channel and indexes
# sh[(c * 16 + k) * nVerts + i], so 48 is the only column count it can read.
# Anything else would run past the end of the buffer, so it is rejected here
# rather than left to the shader.
_SH_COLS = 48


def _check_sh_cols(sh_cols: int) -> None:
    if sh_cols != _SH_COLS:
        raise ValueError(
            f"sh_coeffs must have {_SH_COLS} columns (degree-3 SH, [N, 3, 16]); "
            f"got {sh_cols}.  Lower SH degrees are not allowed."
        )


def launch(
    positions: np.ndarray,
    *,
    colors: np.ndarray | None = None,
    radius: np.ndarray | None = None,
    faces: np.ndarray | None = None,
    lines: np.ndarray | None = None,
    sh_coeffs: np.ndarray | None = None,
    width: int = 1280,
    height: int = 720,
    aa_mode: str | int = "gaussian_msaa",
    line_topology: str = "strip",
    background: tuple[float, float, float] | None = None,
    flip_up: bool = False,
    screenshot_dir: str | None = None,
) -> None:
    """Open an interactive viewer window. Blocks until the window is closed.

    Unlike :func:`benchmark` and :class:`OffscreenRenderer`, which take the
    packed ``[N, 7]`` ``vert_attrs`` layout shared with ``fuzzydr.pack_attrs``,
    this function takes the components separately and packs them internally.
    The offline entry points stream pre-packed frames, so packing there is a
    per-frame cost; here it happens once, before the window opens.

    Parameters
    ----------
    positions:
        Float32 array of shape (N, 3) - world-space vertex positions.
    colors:
        Float32 array of shape (N, 3) - per-vertex RGB.  Required unless
        ``sh_coeffs`` is given, in which case colour is evaluated on the GPU
        each frame and this must be omitted.
    radius:
        Optional float32 array of shape (N,) - per-vertex world-space radius.
        Pass it only for checkpoints saved with ``bresen_lines=False``; its
        presence enables the viewer's "Quads - per-vertex radius" line style
        and opens the window in it.  When omitted the viewer opens in
        Bresenham and drives quad width from its own slider.
    faces:
        Optional uint32 array of shape (F, 3) - CCW triangle indices.
    lines:
        Optional uint32 array of shape (L, 2) - line segment index pairs.
    sh_coeffs:
        Optional float32 array of shape (N, 48) or (N, 3, 16) - degree-3 SH
        coefficients, re-evaluated on the GPU from the camera position.
    width, height:
        Initial window size in pixels.
    background:
        Initial clear colour as an RGB triple in [0, 1].  Training renders
        against either white or black (fuzzydr's ``white_bg``); scripts that
        train on black record ``"bg_color"`` in the checkpoint metadata, so
        pass that through when present.  ``None`` (default) means white,
        which is what the checkpoints omitting the field were trained on.
    screenshot_dir:
        Directory the panel's screenshot button writes into, resolved to an
        absolute path so it does not depend on the process working directory.
        ``None`` (default) means the current directory.
    flip_up:
        Start with the camera's up reference pointing down -Z.  Datasets
        differ in which way their Z axis points, so some scenes load upside
        down; this affects only the view matrix, never the geometry, and can
        be toggled at runtime from the viewer's panel.
    aa_mode:
        ``"gaussian_msaa"`` (default) renders on a 2x grid in each axis - 4
        samples per output pixel - and resolves with a 6x6 sigma=0.5 Gaussian;
        its sample count is fixed.  ``"hw_msaa_2x"`` / ``"hw_msaa_4x"`` use
        hardware MSAA at that starting count, which the panel can then change.
        ``"none"`` disables antialiasing.  These are the same names
        :func:`benchmark` accepts.
        All three are switchable at runtime from the ImGui panel.
    line_topology:
        ``"strip"`` (default) rebakes the segment list into LINE_STRIP runs,
        which costs the Bresenham line style roughly half the vertex-shader
        invocations; ``"list"`` draws the segments as given.  The quad line
        styles always read the segment list and are unaffected.  Pixels are
        identical either way - the draws are opaque and depth-resolved, so
        segment order does not reach the image.  Same names
        :func:`benchmark` accepts.
    """
    pos = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1, 3)
    N = pos.shape[0]
    if N == 0:
        raise ValueError("positions must not be empty")

    if sh_coeffs is not None and np.asarray(sh_coeffs).size > 0:
        if colors is not None:
            raise ValueError(
                "pass either colors or sh_coeffs, not both: with sh_coeffs the "
                "rgb columns are overwritten by the GPU each frame"
            )
        _sh = np.ascontiguousarray(sh_coeffs, dtype=np.float32)
        if _sh.ndim == 3:
            _sh = _sh.reshape(N, -1)
        if _sh.ndim != 2 or _sh.shape[0] != N:
            raise ValueError(
                f"sh_coeffs must have shape ({N}, C) or ({N}, ..., C), "
                f"got {np.asarray(sh_coeffs).shape}"
            )
        sh_cols = _sh.shape[1]
        _check_sh_cols(sh_cols)
    else:
        if colors is None:
            raise ValueError("colors is required when sh_coeffs is not given")
        _sh = np.empty(0, dtype=np.float32)
        sh_cols = 0

    # [x, y, z, radius, r, g, b] - the layout fuzzydr.pack_attrs produces.
    va = np.zeros((N, 7), dtype=np.float32)
    va[:, :3] = pos
    if radius is not None:
        rad = np.ascontiguousarray(radius, dtype=np.float32).reshape(-1)
        if rad.shape[0] != N:
            raise ValueError(f"radius must have shape ({N},); got {rad.shape}")
        va[:, 3] = rad
    if colors is not None:
        va[:, 4:] = np.ascontiguousarray(colors, dtype=np.float32).reshape(-1, 3)

    _faces = (np.ascontiguousarray(faces, dtype=np.uint32).reshape(-1, 3)
              if faces is not None and np.asarray(faces).size > 0
              else np.empty((0, 3), dtype=np.uint32))

    _lines = (np.ascontiguousarray(lines, dtype=np.uint32).reshape(-1, 2)
              if lines is not None and np.asarray(lines).size > 0
              else np.empty((0, 2), dtype=np.uint32))

    if _faces.shape[0] == 0 and _lines.shape[0] == 0:
        raise ValueError("At least one of 'faces' or 'lines' must be non-empty")

    # The Bresenham style draws from the strip rebake, the quad styles from
    # the pair list; both are uploaded so the panel can switch freely.  This
    # runs once, before the window opens.
    if _lines.shape[0] > 0 and line_topology == "strip":
        _strip, _ = lines_to_strips(_lines)
    elif line_topology != "list":
        raise ValueError(
            f"line_topology must be one of {list(_LINE_TOPOLOGIES)}; "
            f"got {line_topology!r}")
    else:
        _strip = np.empty(0, dtype=np.uint32)

    if background is None:
        bg = (1.0, 1.0, 1.0)
    else:
        bg = tuple(float(c) for c in background)
        if len(bg) != 3:
            raise ValueError(f"background must be an RGB triple; got {background!r}")

    _core._launch(
        va.tobytes(),      N,
        _faces.tobytes(),  _faces.shape[0],
        _lines.tobytes(),  _lines.shape[0],
        _strip.tobytes(),  _strip.size,
        _sh.tobytes(),     sh_cols,
        width, height,
        *_resolve_launch_aa(aa_mode),
        radius is not None,
        *bg,
        bool(flip_up),
        os.path.abspath(screenshot_dir) if screenshot_dir is not None else "",
    )


# LINE_LIST -> LINE_STRIP rebake.
#
# Greedy maximal-walk decomposition of the line graph: starting from a
# degree-1 vertex when one is available (so a simple polyline becomes a
# single strip), walk along incident unused edges until no edge can extend
# the current walk.  Emit each walk as a contiguous run of indices; walks
# are separated by Vulkan/D3D primitive-restart sentinel UINT32_MAX.
#
# For polyline-soup data (most CCs are simple paths), this halves vertex-
# shader invocations: 2 x N_segments -> ~N_segments + N_polylines.  Y-junc-
# tions and branches still decompose correctly but emit one extra strip
# per branch; the asymptotic bound is "indices <= 2*segments" so the worst
# case matches LINE_LIST exactly.
_RESTART = np.uint32(0xFFFFFFFF)


def lines_to_strips(lines: np.ndarray) -> tuple[np.ndarray, int]:
    """Rebake a LINE_LIST index buffer into LINE_STRIP-with-restart indices.

    ``lines`` : uint32 [L, 2]  endpoint index pairs.

    Returns ``(flat_indices, n_strips)`` where ``flat_indices`` is a flat
    uint32 array suitable as a primitive-restart-enabled LINE_STRIP index
    buffer, and ``n_strips`` is the number of strips it encodes.
    """
    L = int(lines.shape[0])
    if L == 0:
        return np.empty(0, dtype=np.uint32), 0

    lines64 = np.ascontiguousarray(lines, dtype=np.int64).reshape(-1, 2)
    a = lines64[:, 0]
    b = lines64[:, 1]
    n_verts = int(max(a.max(), b.max())) + 1

    # Build undirected adjacency as CSR.  Each undirected edge contributes
    # two directed records (a->b and b->a) so we can walk in either direction.
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    eid = np.tile(np.arange(L, dtype=np.int64), 2)

    order = np.argsort(src, kind="stable")
    src = src[order]
    dst = dst[order]
    eid = eid[order]

    offsets = np.zeros(n_verts + 1, dtype=np.int64)
    np.add.at(offsets, src + 1, 1)
    np.cumsum(offsets, out=offsets)

    cursor = offsets[:-1].copy()
    upper  = offsets[1:].copy()
    used   = np.zeros(L, dtype=bool)

    # Walk-start order: odd-degree (== path endpoints) first so each simple
    # polyline traverses its entire chain in one walk.
    degree = upper - cursor
    odd  = np.flatnonzero(degree & 1)
    even = np.flatnonzero(((degree & 1) == 0) & (degree > 0))
    start_order = np.concatenate([odd, even]).tolist()

    out_chunks: list[np.ndarray] = []
    walk_buf: list[int] = []

    cursor_l = cursor
    upper_l  = upper
    used_l   = used
    eid_l    = eid
    dst_l    = dst

    for start in start_order:
        # Each `start` may seed multiple walks (e.g. a degree-3 vertex that
        # still has unused incident edges after the first walk finishes).
        while True:
            # Advance cursor past edges that were consumed via the other
            # endpoint of some prior walk.
            i = int(cursor_l[start])
            up = int(upper_l[start])
            while i < up and used_l[int(eid_l[i])]:
                i += 1
            cursor_l[start] = i
            if i >= up:
                break

            walk_buf.clear()
            walk_buf.append(int(start))
            v = int(start)
            while True:
                i  = int(cursor_l[v])
                up = int(upper_l[v])
                while i < up and used_l[int(eid_l[i])]:
                    i += 1
                cursor_l[v] = i
                if i >= up:
                    break
                e = int(eid_l[i])
                used_l[e] = True
                v = int(dst_l[i])
                walk_buf.append(v)

            out_chunks.append(np.asarray(walk_buf, dtype=np.uint32))

    if not out_chunks:
        return np.empty(0, dtype=np.uint32), 0

    # Interleave chunks with UINT32_MAX restart sentinels.
    n_strips = len(out_chunks)
    total = sum(c.size for c in out_chunks) + (n_strips - 1)
    flat = np.empty(total, dtype=np.uint32)
    pos = 0
    for i, c in enumerate(out_chunks):
        if i > 0:
            flat[pos] = _RESTART
            pos += 1
        flat[pos:pos + c.size] = c
        pos += c.size
    return flat, n_strips


_LINE_TOPOLOGIES = {"list": 0, "strip": 1}


def benchmark(
    vert_attrs: np.ndarray,
    viewprojs: np.ndarray,
    eyes: np.ndarray,
    *,
    lines: np.ndarray,
    sh_coeffs: np.ndarray | None = None,
    compute_shader: str = "",
    width: int = 1920,
    height: int = 1080,
    warmup: int = 20,
    measure: int = 30,
    capture_screenshots: bool = False,
    measure_nosh: bool = False,
    aa_mode: str | int = "gaussian_msaa",
    line_topology: str = "list",
) -> dict:
    """Benchmark the viewer's Vulkan rendering path at a list of camera poses.

    Stands up an offscreen Vulkan context (no window, no vsync) that
    reproduces the viewer's SH compute + Bresenham rasterization pipeline.
    SH compute is dispatched on every frame - there is no camera-dirty
    caching, so reported timings include full inference cost.

    Line-only: there is no `faces` parameter - the benchmark path does not
    support face rendering.  Use ``fuzzydr_viewer.launch`` for an interactive viewer
    that handles faces.

    Parameters
    ----------
    vert_attrs : float32 [N, 7]  (x, y, z, radius, r, g, b)
    viewprojs  : float32 [V, 4, 4]  row-major view-projection matrices
    eyes       : float32 [V, 3]     world-space camera positions
    lines      : uint32 index buffer.  Shape depends on ``line_topology``:
                 - ``"list"``  (default): ``[L, 2]`` endpoint index pairs.
                 - ``"strip"``: flat 1-D index buffer with ``UINT32_MAX``
                   (``0xFFFFFFFF``) used as the primitive-restart sentinel
                   between maximal-walk strips.  Caller is responsible for
                   the rebake.
    sh_coeffs  : float32 [N, C] or [N, 3, K] - per-vertex data buffer
    compute_shader : SPV filename override; empty -> "viewer_sh_eval.comp.spv"
    width, height  : frame size in pixels (defaults 1920x1080)
    warmup, measure: frames per view; the returned times array contains
                     all ``warmup + measure`` frames so callers can eyeball
                     the transient and slice ``times[:, warmup:]`` for
                     steady-state stats.
    capture_screenshots : when True, render one extra *untimed* frame per
                     view and read back the final resolved image.
    measure_nosh : when True, run a second timed loop per view that skips
                     the SH compute dispatch (reusing evalBuf from the
                     full-pipeline loop), isolating the graphics cost.
    aa_mode : one of
                     ``"hw_msaa_4x"``     - 4x hardware MSAA
                     ``"gaussian_msaa"``  - 2x supersample + 6x6 sigma=0.5 Gauss resolve (default)
                     ``"none"``           - 1-sample, no AA - fastest
                     ``"hw_msaa_2x"``     - 2x hardware MSAA
                     All modes rasterize lines with Bresenham; they differ only
                     in sample count and resolve filter.  Same names as
                     :func:`launch`.
                     Integer forms ``0``-``3`` are also accepted.

    Returns
    -------
    dict with keys:
      - ``times``      : float32 [V, warmup + measure] seconds per frame
      - ``times_nosh`` : float32 [V, warmup + measure] or None
      - ``images``     : uint8 [V, H, W, 4] RGBA if capture_screenshots
      - ``warmup``, ``measure`` : int echoed for convenience
    """
    va = np.ascontiguousarray(vert_attrs, dtype=np.float32)
    if va.ndim == 1:
        if va.size % 7 != 0:
            raise ValueError("vert_attrs flat array length must be a multiple of 7")
        va = va.reshape(-1, 7)
    if va.ndim != 2 or va.shape[1] != 7:
        raise ValueError("vert_attrs must have shape (N, 7)")
    N = va.shape[0]

    if lines is None or np.asarray(lines).size == 0:
        raise ValueError("benchmark is line-only; `lines` is required and must be non-empty")
    if line_topology not in _LINE_TOPOLOGIES:
        raise ValueError(
            f"line_topology must be one of {list(_LINE_TOPOLOGIES)}; got {line_topology!r}")
    if line_topology == "list":
        _lines = np.ascontiguousarray(lines, dtype=np.uint32).reshape(-1, 2)
    else:  # "strip"
        _lines = np.ascontiguousarray(lines, dtype=np.uint32).reshape(-1)
    _n_indices = int(_lines.size)

    sh_cols = 0
    _sh = np.empty(0, dtype=np.float32)
    if sh_coeffs is not None and np.asarray(sh_coeffs).size > 0:
        _sh = np.ascontiguousarray(sh_coeffs, dtype=np.float32)
        if _sh.ndim == 3:
            _sh = _sh.reshape(N, -1)
        if _sh.ndim != 2 or _sh.shape[0] != N:
            raise ValueError(
                f"sh_coeffs must have shape ({N}, C) or ({N}, ..., C), "
                f"got {np.asarray(sh_coeffs).shape}"
            )
        sh_cols = _sh.shape[1]
        _check_sh_cols(sh_cols)

    vps = np.ascontiguousarray(viewprojs, dtype=np.float32)
    if vps.ndim != 3 or vps.shape[1:] != (4, 4):
        raise ValueError(f"viewprojs must have shape (V, 4, 4); got {vps.shape}")
    V = vps.shape[0]
    # Shader reads `mat4 viewproj` column-major in GLSL; callers pass
    # row-major numpy so transpose per-matrix before feeding the byte stream.
    vps_cm = np.ascontiguousarray(vps.transpose(0, 2, 1))

    es = np.ascontiguousarray(eyes, dtype=np.float32)
    if es.shape != (V, 3):
        raise ValueError(f"eyes must have shape ({V}, 3); got {es.shape}")

    if warmup < 0 or measure <= 0:
        raise ValueError("warmup must be >= 0 and measure must be > 0")

    times_flat, times_nosh_flat, images_bytes = _core._benchmark(
        va.tobytes(),      N,
        _lines.tobytes(),  _n_indices,
        _sh.tobytes(),     sh_cols,
        str(compute_shader),
        vps_cm.tobytes(),  V,
        es.tobytes(),
        int(width), int(height),
        int(warmup), int(measure),
        bool(capture_screenshots),
        bool(measure_nosh),
        _resolve_aa_mode(aa_mode),
        _LINE_TOPOLOGIES[line_topology],
    )

    total = int(warmup) + int(measure)
    times = np.asarray(times_flat, dtype=np.float32).reshape(V, total)

    times_nosh: np.ndarray | None = None
    if measure_nosh:
        times_nosh = np.asarray(times_nosh_flat, dtype=np.float32).reshape(V, total)

    images: np.ndarray | None = None
    if capture_screenshots:
        images = (np.frombuffer(images_bytes, dtype=np.uint8)
                    .reshape(V, int(height), int(width), 4)
                    .copy())

    return {
        "times":       times,
        "times_nosh":  times_nosh,
        "images":      images,
        "warmup":      int(warmup),
        "measure":     int(measure),
    }


class OffscreenRenderer:
    """Persistent-context offscreen renderer for streamed-vertex playback.

    Reuses one Vulkan context across many ``frame()`` calls.  Static geometry
    (``sh_coeffs``, ``lines``) is uploaded once at construction; each frame
    pushes only the updated ``[N, 7]`` ``vert_attrs`` through a permanently-
    mapped host-visible staging buffer.  One ``vkQueueSubmit`` per frame; an
    optional second untimed submit reads the resolved RGBA8 image back to
    host.

    Line-only - there is no `faces` parameter.  Use ``fuzzydr_viewer.launch`` for an
    interactive viewer that handles faces.

    Parameters
    ----------
    vert_attrs : float32 [N, 7]   initial positions+radius+rgb
    lines      : uint32 index buffer.  Shape depends on ``line_topology``:
                 - ``"list"``  (default): ``[L, 2]`` endpoint index pairs.
                 - ``"strip"``: flat 1-D index buffer with ``UINT32_MAX``
                   (``0xFFFFFFFF``) used as the primitive-restart sentinel
                   between maximal-walk strips.  Closes the 1-pixel
                   diamond-exit gaps that LINE_LIST leaves at shared
                   vertices - needed for parity with the eval renderer.
    sh_coeffs  : float32 [N, C] or [N, 3, K]
    width, height : int
    aa_mode    : str | int      see :func:`benchmark`
    line_topology : ``"list"`` (default) or ``"strip"``
    compute_shader : str        SPV filename override for the SH eval pass

    Use as a context manager so the Vulkan context tears down promptly::

        with fuzzydr_viewer.OffscreenRenderer(va0, lines=L, sh_coeffs=sh,
                                      width=W, height=H) as rdr:
            for t in range(T):
                out = rdr.frame(va_t, viewproj, eye, capture=True)
                writer.append_data(out["image"][..., :3])
    """

    def __init__(
        self,
        vert_attrs: np.ndarray,
        *,
        lines: np.ndarray,
        sh_coeffs: np.ndarray | None = None,
        compute_shader: str = "",
        width: int = 1920,
        height: int = 1080,
        aa_mode: str | int = "gaussian_msaa",
        line_topology: str = "list",
    ) -> None:
        va = np.ascontiguousarray(vert_attrs, dtype=np.float32)
        if va.ndim == 1:
            if va.size % 7 != 0:
                raise ValueError("vert_attrs flat array length must be a multiple of 7")
            va = va.reshape(-1, 7)
        if va.ndim != 2 or va.shape[1] != 7:
            raise ValueError("vert_attrs must have shape (N, 7)")
        self._N = int(va.shape[0])

        if lines is None or np.asarray(lines).size == 0:
            raise ValueError("OffscreenRenderer is line-only; `lines` is required and must be non-empty")
        if line_topology not in _LINE_TOPOLOGIES:
            raise ValueError(
                f"line_topology must be one of {list(_LINE_TOPOLOGIES)}; got {line_topology!r}")
        if line_topology == "list":
            _lines = np.ascontiguousarray(lines, dtype=np.uint32).reshape(-1, 2)
        else:  # "strip"
            _lines = np.ascontiguousarray(lines, dtype=np.uint32).reshape(-1)
        _n_indices = int(_lines.size)

        sh_cols = 0
        _sh = np.empty(0, dtype=np.float32)
        if sh_coeffs is not None and np.asarray(sh_coeffs).size > 0:
            _sh = np.ascontiguousarray(sh_coeffs, dtype=np.float32)
            if _sh.ndim == 3:
                _sh = _sh.reshape(self._N, -1)
            if _sh.ndim != 2 or _sh.shape[0] != self._N:
                raise ValueError(
                    f"sh_coeffs must have shape ({self._N}, C) or ({self._N}, ..., C), "
                    f"got {np.asarray(sh_coeffs).shape}"
                )
            sh_cols = _sh.shape[1]
            _check_sh_cols(sh_cols)

        self._impl = _core._OffscreenRenderer(
            va.tobytes(),       self._N,
            _lines.tobytes(),   _n_indices,
            _sh.tobytes(),      sh_cols,
            str(compute_shader),
            int(width), int(height),
            _resolve_aa_mode(aa_mode),
            _LINE_TOPOLOGIES[line_topology],
        )
        self._W = int(width)
        self._H = int(height)

    @property
    def width(self) -> int:  return self._W
    @property
    def height(self) -> int: return self._H
    @property
    def num_verts(self) -> int: return self._N

    def frame(
        self,
        vert_attrs: np.ndarray,
        viewproj: np.ndarray,
        eye: np.ndarray,
        *,
        capture: bool = False,
    ) -> dict:
        """Render one frame.

        Parameters
        ----------
        vert_attrs : float32 [N, 7]   updated positions+radius+rgb (must
                                       match the constructor's N).  Copied
                                       into the mapped staging buffer.
        viewproj   : float32 [4, 4]   row-major (numpy convention); the C++
                                       side transposes to GLSL column-major.
        eye        : float32 [3]      world-space camera position
        capture    : bool             when True, read resolved RGBA8 back
                                       to host (one extra untimed submit)

        Returns
        -------
        dict with keys
          - ``time_ms`` : float - wall-clock around the timed render submit
          - ``image``   : uint8 [H, W, 4] RGBA, or None if not captured
        """
        if self._impl is None:
            raise RuntimeError("OffscreenRenderer used after close()")

        va = np.ascontiguousarray(vert_attrs, dtype=np.float32)
        if va.ndim == 1:
            if va.size != self._N * 7:
                raise ValueError(
                    f"vert_attrs flat must have length {self._N * 7}; got {va.size}"
                )
        elif va.shape != (self._N, 7):
            raise ValueError(
                f"vert_attrs must have shape ({self._N}, 7); got {va.shape}"
            )

        vp = np.ascontiguousarray(viewproj, dtype=np.float32).reshape(4, 4)
        # GLSL reads mat4 column-major; numpy is row-major.
        vp_cm = np.ascontiguousarray(vp.T)

        ey = np.ascontiguousarray(eye, dtype=np.float32).reshape(3)

        time_s, img_b = self._impl.frame(
            va.tobytes(), vp_cm.tobytes(), ey.tobytes(), bool(capture)
        )

        image: np.ndarray | None = None
        if capture and len(img_b) > 0:
            image = (np.frombuffer(img_b, dtype=np.uint8)
                       .reshape(self._H, self._W, 4)
                       .copy())

        return {"time_ms": float(time_s) * 1000.0, "image": image}

    def close(self) -> None:
        """Tear down the Vulkan context immediately.  Idempotent."""
        self._impl = None

    def __enter__(self) -> "OffscreenRenderer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
