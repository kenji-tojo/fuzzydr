# fuzzydr/__init__.py
from __future__ import annotations

import torch
from ._core import __version__
from . import _core
from . import optimize


def init(device_index: int = -1) -> None:
    """Initialize the global Vulkan context (idempotent).

    ``device_index`` selects the Vulkan physical device by enumeration index.
    ``<0`` (default) picks the first suitable device, or the value of the
    ``FUZZYDR_DEVICE_INDEX`` environment variable if it is set. On multi-GPU
    machines, set this (or the env var) so the Vulkan device matches the CUDA
    device used for interop; otherwise the interop can fail with
    ``invalid device ordinal``.
    """
    _core.init(device_index)


def shutdown() -> None:
    """Destroy the global Vulkan context (safe to call multiple times)."""
    _core.shutdown()


# ---------------------------------------------------------------------------
# render_options - compact uint8[4] CPU tensor packing boolean render flags.
#
# These index constants mirror the OPT_* values in rasterize.h so that both
# sides stay in sync without requiring a round-trip through the C++ layer.
# ---------------------------------------------------------------------------
_OPT_WHITE_BG         = 0   # clear background to white; else black
_OPT_BRESEN_LINES     = 1   # 1-px Bresenham LINE_LIST mode
_OPT_CULL_BACKFACE    = 2   # discard CW-wound faces; lines unaffected
_OPT_AUX_BUFFERS      = 3   # download prim_id + bary_depth to host/CUDA
_NUM_RENDER_OPTIONS   = 4   # number of flags above


def make_render_options(
    *,
    white_bg: bool = True,
    bresen_lines: bool = False,
    cull_back_faces: bool = False,
    aux_buffers: bool = False,
) -> torch.Tensor:
    """Pack boolean render flags into a CPU ``uint8[4]`` tensor.

    This is the canonical way to build the ``render_options`` argument.
    The resulting tensor is always on CPU regardless of the device used for
    geometry data.

    Parameters
    ----------
    white_bg : bool
        Clear the background to white ``(1,1,1,1)``; otherwise black
        ``(0,0,0,1)``.  The alpha channel is padded to 1 and ignored by
        the MSAA filter (which reads RGB only).
    bresen_lines : bool
        Render lines with ``VK_PRIMITIVE_TOPOLOGY_LINE_LIST`` (1-px,
        implementation-defined Bresenham).  The per-vertex ``radius``
        slot is unused in this mode.
    cull_back_faces : bool
        Discard clockwise-wound (back-facing) triangles. Has no effect on
        line primitives.
    aux_buffers : bool
        Download ``prim_id`` and ``bary_depth`` auxiliary buffers to
        host/CUDA after the forward pass. When False (default), those
        buffers are still written on the GPU (for use by backward passes)
        but the host download is skipped, saving PCIe bandwidth.

    Note: with stochastic opacity masking (``tau == -1``) the threshold is
    drawn once per primitive.
    """
    opts = torch.zeros(_NUM_RENDER_OPTIONS, dtype=torch.uint8)
    opts[_OPT_WHITE_BG]         = int(white_bg)
    opts[_OPT_BRESEN_LINES]     = int(bresen_lines)
    opts[_OPT_CULL_BACKFACE]    = int(cull_back_faces)
    opts[_OPT_AUX_BUFFERS]      = int(aux_buffers)
    return opts


def _require_contig(x, *, dtype: torch.dtype) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    if t.dtype != dtype:
        raise ValueError(f"Expected dtype={dtype}, got {t.dtype}.")
    if not t.is_contiguous():
        raise ValueError("Expected C-contiguous tensor.")
    return t


def _require_same_device(*tensors: torch.Tensor) -> torch.device:
    dev = None
    for t in tensors:
        if t is None:
            continue
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t)
        if dev is None:
            dev = t.device
        elif t.device != dev:
            raise ValueError("All tensors must be on the same device (all CPU or same CUDA device).")
    # Default to CPU when no tensors were given.
    return dev if dev is not None else torch.device("cpu")


def _empty_uint32(shape, device: torch.device) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.uint32, device=device).contiguous()


def _empty_float32(shape, device: torch.device) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.float32, device=device).contiguous()


class _RasterizeFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        vert_attrs: torch.Tensor,      # float32 [num_verts,7]
        num_faces: int,                # explicit face count (may differ from faces.shape[0] when caching)
        faces: torch.Tensor,           # uint32  [num_faces,3] or empty [0,3] for cached topology
        face_opacity: torch.Tensor,    # float32 [num_faces]
        num_lines: int,                # explicit line count (may differ from lines.shape[0] when caching)
        lines: torch.Tensor,           # uint32  [num_lines,2] or empty [0,2] for cached topology
        line_opacity: torch.Tensor,    # float32 [num_lines]
        num_points: int,               # explicit point count (may differ from points.shape[0] when caching)
        points: torch.Tensor,          # uint32  [num_points] or empty [0] for cached topology
        point_opacity: torch.Tensor,   # float32 [num_points]
        viewproj: torch.Tensor,        # float32 [4,4] row-major (always CPU)
        campos: torch.Tensor,          # float32 [3] (always CPU)
        width: int,
        height: int,
        tau: float,
        seed: int,
        render_options: torch.Tensor,  # uint8   [_NUM_RENDER_OPTIONS] (always CPU)
    ):
        vert_attrs = _require_contig(vert_attrs, dtype=torch.float32)
        faces = _require_contig(faces, dtype=torch.uint32)
        face_opacity = _require_contig(face_opacity, dtype=torch.float32)
        lines = _require_contig(lines, dtype=torch.uint32)
        line_opacity = _require_contig(line_opacity, dtype=torch.float32)
        points = _require_contig(points, dtype=torch.uint32)
        point_opacity = _require_contig(point_opacity, dtype=torch.float32)
        device = _require_same_device(vert_attrs, faces, face_opacity, lines, line_opacity, points, point_opacity)

        # viewproj and campos are always CPU, validated separately
        viewproj_cpu = _require_contig(viewproj, dtype=torch.float32).cpu().contiguous()
        campos_cpu = _require_contig(campos, dtype=torch.float32).cpu().contiguous()
        if viewproj_cpu.shape != (4, 4):
            raise ValueError("viewproj must have shape [4,4]")
        if campos_cpu.shape != (3,):
            raise ValueError("campos must have shape [3]")

        # render_options is always CPU
        render_options_cpu = _require_contig(render_options, dtype=torch.uint8).cpu().contiguous()
        if render_options_cpu.shape != (_NUM_RENDER_OPTIONS,):
            raise ValueError(f"render_options must have shape [{_NUM_RENDER_OPTIONS}]")

        aux_buffers = bool(render_options_cpu[_OPT_AUX_BUFFERS].item())

        if vert_attrs.ndim != 2 or vert_attrs.shape[1] != 7:
            raise ValueError("vert_attrs must have shape [num_verts,7]")
        if faces.shape[0] > 0 and (faces.ndim != 2 or faces.shape[1] != 3):
            raise ValueError("faces must have shape [num_faces,3]")
        if face_opacity.ndim != 1 or face_opacity.shape[0] != num_faces:
            raise ValueError("face_opacity must have shape [num_faces]")
        if lines.shape[0] > 0 and (lines.ndim != 2 or lines.shape[1] != 2):
            raise ValueError("lines must have shape [num_lines,2]")
        if line_opacity.ndim != 1 or line_opacity.shape[0] != num_lines:
            raise ValueError("line_opacity must have shape [num_lines]")
        if points.shape[0] > 0 and points.ndim != 1:
            raise ValueError("points must have shape [num_points]")
        if point_opacity.ndim != 1 or point_opacity.shape[0] != num_points:
            raise ValueError("point_opacity must have shape [num_points]")
        if num_faces == 0 and num_lines == 0 and num_points == 0:
            raise ValueError("at least one of faces, lines, or points must be non-empty")
        if not (isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0):
            raise ValueError("width and height must be positive ints")

        tau = float(tau)
        if tau != -1.0 and not (0.0 <= tau <= 1.0):
            raise ValueError("tau must be -1 or in [0,1]")

        seed = int(seed)
        is_cuda = bool(vert_attrs.is_cuda)

        # When aux_buffers is False, pass empty tensors so the C++
        # binding feeds nullptr to the rasterizer, skipping host downloads.
        if aux_buffers:
            prim_id = torch.empty((height, width), dtype=torch.uint32, device=device).contiguous()
            bary_depth = torch.empty((height, width, 4), dtype=torch.float32, device=device).contiguous()
        else:
            prim_id = _empty_uint32((0, 0), device)
            bary_depth = _empty_float32((0, 0, 4), device)

        rgba = torch.empty((height, width, 4), dtype=torch.float32, device=device).contiguous()

        _core.rasterize(
            vert_attrs,
            num_faces,
            faces,
            face_opacity,
            num_lines,
            lines,
            line_opacity,
            num_points,
            points,
            point_opacity,
            viewproj_cpu,
            campos_cpu,
            tau,
            seed,
            prim_id,
            bary_depth,
            rgba,
            render_options_cpu,
            is_cuda,
        )

        prim_counts = torch.tensor([num_faces, num_lines, num_points], dtype=torch.uint32)
        ctx.save_for_backward(vert_attrs, prim_counts)

        ctx.mark_non_differentiable(prim_id)
        ctx.mark_non_differentiable(bary_depth)
        return prim_id, bary_depth, rgba

    @staticmethod
    def backward(ctx, grad_prim_id, grad_bary_depth, grad_rgba):
        vert_attrs, prim_counts = ctx.saved_tensors
        num_faces  = prim_counts[0].item()
        num_lines  = prim_counts[1].item()
        num_points = prim_counts[2].item()

        if grad_rgba is None:
            return (None,) * 17

        grad_rgba = _require_contig(grad_rgba, dtype=torch.float32)
        if grad_rgba.ndim != 3 or grad_rgba.shape[2] != 4:
            raise ValueError("grad_rgba must have shape [height,width,4]")

        is_cuda = bool(vert_attrs.is_cuda)

        grad_vert_attrs = torch.empty_like(vert_attrs).contiguous()

        _core.edge_grad(
            num_faces,
            num_lines,
            num_points,
            grad_rgba,
            grad_vert_attrs,
            is_cuda,
        )

        # vert_attrs, num_faces, faces, face_opacity, num_lines, lines, line_opacity,
        # num_points, points, point_opacity,
        # viewproj, campos, width, height, tau, seed, render_options
        return (grad_vert_attrs,) + (None,) * 16


def rasterize(
    vert_attrs: torch.Tensor,                             # float32 [num_verts,7]
    *,
    viewproj: torch.Tensor,                               # float32 [4,4] row-major (always CPU)
    campos: torch.Tensor | None = None,                   # float32 [3] (always CPU)
    faces: torch.Tensor | int | None = None,              # uint32 [num_faces,3], or int for cached topology, or None/0 for no faces
    face_opacity: torch.Tensor | None = None,             # float32 [num_faces] (default=ones)
    lines: torch.Tensor | int | None = None,              # uint32 [num_lines,2], or int for cached topology, or None/0 for no lines
    line_opacity: torch.Tensor | None = None,             # float32 [num_lines] (default=ones)
    width: int,
    height: int,
    tau: float = 0.5,
    seed: int = 0,
    white_bg: bool = True,
    bresen_lines: bool = False,
    cull_back_faces: bool = False,
    aux_buffers: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
    """
    Differentiable Vulkan rasterizer supporting triangles and/or lines.

    At least one of ``faces`` or ``lines`` must be provided and non-empty.

    Parameters
    ----------
    vert_attrs : float32 [num_verts,7]
        Packed vertex attributes (x,y,z,radius,r,g,b).  (x,y,z) are
        world-space positions; (r,g,b) are vertex colors.  The ``radius``
        slot is used only by quad lines (``bresen_lines=False``) as a
        world-space cylinder radius driving the screen-space quad width;
        it is ignored for Bresenham lines, faces, and points (pack any
        value, e.g. 0).
    viewproj : float32 [4,4]
        Row-major view-projection matrix.  Converted to column-major
        internally for GLSL.  Always a CPU tensor.
    campos : float32 [3], optional
        Camera position in world space (always CPU).  Used by the
        quad line vertex shader for screen-space radius projection.
        Ignored in Bresenham mode.  Defaults to [0,0,0].
    faces : uint32 [num_faces,3] | int | None, optional
        Triangle index buffer.  Pass an int to reuse previously cached
        topology with that face count (topology upload is skipped).
        Pass ``None`` or ``0`` when there are no faces.
    face_opacity : float32 [num_faces], optional
        Per-face opacity (defaults to all ones).
    lines : uint32 [num_lines,2] | int | None, optional
        Line index buffer.  Pass an int to reuse previously cached
        topology with that line count (topology upload is skipped).
        Pass ``None`` or ``0`` when there are no lines.
    line_opacity : float32 [num_lines], optional
        Per-line opacity (defaults to all ones).
    width, height : int
        Output image dimensions.
    tau : float
        Opacity threshold (-1 for stochastic, or [0,1]).
    seed : int
        RNG seed for stochastic masking.
    white_bg : bool
        If True, background is white; otherwise black.
    bresen_lines : bool
        If True, render lines with ``VK_PRIMITIVE_TOPOLOGY_LINE_LIST``
        (1-px width, implementation-defined Bresenham rasterization).
        The radius vertex attribute is not read in this mode, and no
        radius gradient is computed by the backward pass.
    cull_back_faces : bool
        If True, discard clockwise-wound (back-facing) triangles.
        Has no effect on line primitives.
    aux_buffers : bool
        If True, return ``(prim_id, bary_depth, rgba)`` with all three
        buffers downloaded.  If False (default), skip downloading the
        auxiliary GPU buffers and return only ``rgba``, saving PCIe
        bandwidth when per-pixel primitive IDs and interpolation weights
        are not needed.

    Returns
    -------
    If aux_buffers is True:
        prim_id    : uint32  [height,width]     (no grad)
        bary_depth : float32 [height,width,4]   (no grad)
        rgba       : float32 [height,width,4]   (grad wrt vert_attrs via edge_grad)
    If aux_buffers is False:
        rgba       : float32 [height,width,4]   (grad wrt vert_attrs via edge_grad)
    """
    device = vert_attrs.device

    # viewproj is required; campos defaults to origin
    viewproj = _require_contig(viewproj, dtype=torch.float32)
    if viewproj.shape != (4, 4):
        raise ValueError("viewproj must have shape [4,4]")
    if campos is None:
        campos = torch.zeros(3, dtype=torch.float32)
    else:
        campos = _require_contig(campos, dtype=torch.float32)
        if campos.shape != (3,):
            raise ValueError("campos must have shape [3]")

    # Normalise faces: int -> cached topology, None -> no faces, Tensor -> regular
    if isinstance(faces, int):
        num_faces = faces
        faces = _empty_uint32((0, 3), device)
    elif faces is None:
        num_faces = 0
        faces = _empty_uint32((0, 3), device)
    else:
        num_faces = int(faces.shape[0])

    if face_opacity is None:
        if num_faces > 0:
            face_opacity = torch.ones((num_faces,), dtype=torch.float32, device=device).contiguous()
        else:
            face_opacity = _empty_float32((0,), device)

    # Normalise lines: int -> cached topology, None -> no lines, Tensor -> regular
    if isinstance(lines, int):
        num_lines = lines
        lines = _empty_uint32((0, 2), device)
    elif lines is None:
        num_lines = 0
        lines = _empty_uint32((0, 2), device)
    else:
        num_lines = int(lines.shape[0])

    if line_opacity is None:
        if num_lines > 0:
            line_opacity = torch.ones((num_lines,), dtype=torch.float32, device=device).contiguous()
        else:
            line_opacity = _empty_float32((0,), device)

    # This entry point covers faces and lines; points go through rasterize_points().
    num_points = 0
    points = _empty_uint32((0,), device)
    point_opacity = _empty_float32((0,), device)

    # Pack the boolean render flags into a compact CPU tensor.
    render_options = make_render_options(
        white_bg=white_bg,
        bresen_lines=bresen_lines,
        cull_back_faces=cull_back_faces,
        aux_buffers=aux_buffers,
    )

    prim_id, bary_depth, rgba = _RasterizeFn.apply(
        vert_attrs,
        int(num_faces),
        faces,
        face_opacity,
        int(num_lines),
        lines,
        line_opacity,
        int(num_points),
        points,
        point_opacity,
        viewproj,
        campos,
        int(width),
        int(height),
        float(tau),
        int(seed),
        render_options,
    )

    if aux_buffers:
        return prim_id, bary_depth, rgba
    return rgba


def rasterize_points(
    vert_attrs: torch.Tensor,                             # float32 [num_verts,7]
    *,
    viewproj: torch.Tensor,                               # float32 [4,4] row-major (always CPU)
    campos: torch.Tensor | None = None,                   # float32 [3] (always CPU)
    points: torch.Tensor | int,                           # uint32 [num_points], or int for cached topology
    point_opacity: torch.Tensor | None = None,            # float32 [num_points] (default=ones)
    width: int,
    height: int,
    tau: float = 0.5,
    seed: int = 0,
    white_bg: bool = True,
    aux_buffers: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
    """Differentiable Vulkan point rasterizer (standalone, point-only path).

    Each point covers exactly one pixel; radius is ignored.

    Parameters
    ----------
    vert_attrs : float32 [num_verts,7]
        Packed vertex attributes ``(x,y,z,radius,r,g,b)``.  Radius is ignored.
    viewproj : float32 [4,4]
        Row-major view-projection matrix (always CPU).
    campos : float32 [3], optional
        Camera position (always CPU).  Defaults to ``[0,0,0]``.
    points : uint32 [num_points] | int
        Point index buffer mapping each point to a vertex.
        Pass an int to reuse previously cached topology.
    point_opacity : float32 [num_points], optional
        Per-point opacity (defaults to all ones).
    width, height : int
        Output image dimensions.
    tau : float
        Opacity threshold (``-1`` for stochastic, or ``[0,1]``).
    seed : int
        RNG seed for stochastic masking.
    white_bg : bool
        If True, background is white; otherwise black.
    aux_buffers : bool
        If True, return ``(prim_id, bary_depth, rgba)``.

    Returns
    -------
    Same as :func:`rasterize`.
    """
    device = vert_attrs.device

    viewproj = _require_contig(viewproj, dtype=torch.float32)
    if viewproj.shape != (4, 4):
        raise ValueError("viewproj must have shape [4,4]")
    if campos is None:
        campos = torch.zeros(3, dtype=torch.float32)
    else:
        campos = _require_contig(campos, dtype=torch.float32)
        if campos.shape != (3,):
            raise ValueError("campos must have shape [3]")

    # No faces or lines in the point-only path.
    num_faces = 0
    faces = _empty_uint32((0, 3), device)
    face_opacity = _empty_float32((0,), device)
    num_lines = 0
    lines = _empty_uint32((0, 2), device)
    line_opacity = _empty_float32((0,), device)

    # Normalise points: int -> cached topology, Tensor -> regular
    if isinstance(points, int):
        num_points = points
        points = _empty_uint32((0,), device)
    else:
        num_points = int(points.shape[0])

    if point_opacity is None:
        if num_points > 0:
            point_opacity = torch.ones((num_points,), dtype=torch.float32, device=device).contiguous()
        else:
            point_opacity = _empty_float32((0,), device)

    render_options = make_render_options(
        white_bg=white_bg,
        aux_buffers=aux_buffers,
    )

    prim_id, bary_depth, rgba = _RasterizeFn.apply(
        vert_attrs,
        int(num_faces),
        faces,
        face_opacity,
        int(num_lines),
        lines,
        line_opacity,
        int(num_points),
        points,
        point_opacity,
        viewproj,
        campos,
        int(width),
        int(height),
        float(tau),
        int(seed),
        render_options,
    )

    if aux_buffers:
        return prim_id, bary_depth, rgba
    return rgba


class _OpacityMaskAuxLossFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        face_opacity: torch.Tensor,    # float32 [num_faces]
        line_opacity: torch.Tensor,    # float32 [num_lines]
        point_opacity: torch.Tensor,   # float32 [num_points]
        error: torch.Tensor,           # float32 [height,width]
        eps: float,
    ) -> torch.Tensor:
        face_opacity   = _require_contig(face_opacity,   dtype=torch.float32)
        line_opacity   = _require_contig(line_opacity,   dtype=torch.float32)
        point_opacity  = _require_contig(point_opacity,  dtype=torch.float32)
        error          = _require_contig(error,          dtype=torch.float32)
        _ = _require_same_device(face_opacity, line_opacity, point_opacity, error)

        if face_opacity.ndim != 1:
            raise ValueError("face_opacity must have shape [num_faces]")
        if line_opacity.ndim != 1:
            raise ValueError("line_opacity must have shape [num_lines]")
        if point_opacity.ndim != 1:
            raise ValueError("point_opacity must have shape [num_points]")
        if error.ndim != 2:
            raise ValueError("error must have shape [height,width]")

        eps = float(eps)
        if not (eps > 0.0):
            raise ValueError("eps must be > 0")

        ctx.save_for_backward(
            face_opacity,
            line_opacity,
            point_opacity,
            error,
            torch.tensor(eps, dtype=torch.float32),
        )

        # The value is always zero; the opacity gradients come from backward().
        return error.new_zeros(())

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        face_opacity, line_opacity, point_opacity, error, eps_t = ctx.saved_tensors
        eps = float(eps_t.item())
        is_cuda = bool(error.is_cuda)

        # Scale the error by the upstream scalar gradient.
        gscale = float(grad_out.item()) if grad_out is not None else 1.0
        error_scaled = (error * gscale).contiguous() if gscale != 1.0 else error

        # outputs
        grad_face_opacity  = torch.empty_like(face_opacity).contiguous()
        grad_line_opacity  = torch.empty_like(line_opacity).contiguous()
        grad_point_opacity = torch.empty_like(point_opacity).contiguous()

        _core.opacity_grad(
            error_scaled,
            grad_face_opacity,
            grad_line_opacity,
            grad_point_opacity,
            eps,
            is_cuda,
        )

        # inputs: (face_opacity, line_opacity, point_opacity, error, eps)
        return (
            grad_face_opacity,
            grad_line_opacity,
            grad_point_opacity,
            None,
            None,
        )


def opacity_mask_aux_loss(
    *,
    face_opacity: torch.Tensor | None = None,      # float32 [num_faces]  (requires_grad=True)
    line_opacity: torch.Tensor | None = None,      # float32 [num_lines]  (requires_grad=True)
    point_opacity: torch.Tensor | None = None,     # float32 [num_points] (requires_grad=True)
    error: torch.Tensor,                           # float32 [height,width]
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Returns a scalar auxiliary loss whose backward calls fuzzydr._core.opacity_grad,
    accumulating gradients into:
      - face_opacity[:]      (if provided)
      - line_opacity[:]      (if provided)
      - point_opacity[:]     (if provided)
    """
    device = error.device

    if face_opacity is None:
        face_opacity = _empty_float32((0,), device)
    if line_opacity is None:
        line_opacity = _empty_float32((0,), device)
    if point_opacity is None:
        point_opacity = _empty_float32((0,), device)

    return _OpacityMaskAuxLossFn.apply(
        face_opacity, line_opacity, point_opacity, error, float(eps)
    )


# =============================================================================
# MSAA downsample: isotropic Gaussian pixel reconstruction
# =============================================================================

class _MsaaDownsampleRGBAFn(torch.autograd.Function):
    """Differentiable MSAA pixel reconstruction (isotropic Gaussian downsample).

    The input is the 4-channel rasterizer output; the filter reads RGB and
    ignores alpha.  The output is 3-channel RGB.
    """

    @staticmethod
    def forward(ctx, img: torch.Tensor, sigma: float) -> torch.Tensor:
        img = _require_contig(img, dtype=torch.float32)
        if img.ndim != 3 or img.shape[2] != 4:
            raise ValueError("img must have shape [height,width,4] (RGBA)")

        height, width = img.shape[0], img.shape[1]
        if width % 2 != 0 or height % 2 != 0:
            raise ValueError(
                f"msaa_downsample_rgba requires even dimensions, got {height}x{width}"
            )

        is_cuda = img.is_cuda
        out = torch.empty(
            (height // 2, width // 2, 3), dtype=torch.float32, device=img.device
        ).contiguous()

        _core.msaa_downsample_rgba(img, out, float(sigma), is_cuda)

        ctx.sigma = sigma
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        grad_out = grad_out.contiguous()
        if grad_out.ndim != 3 or grad_out.shape[2] != 3:
            raise ValueError("grad_out must have shape [height/2, width/2, 3]")

        width, height = grad_out.shape[1] * 2, grad_out.shape[0] * 2
        is_cuda = bool(grad_out.is_cuda)
        grad_in = torch.empty(
            (height, width, 4), dtype=torch.float32, device=grad_out.device
        ).contiguous()

        _core.msaa_downsample_rgba_backward(
            grad_out, grad_in, float(ctx.sigma), is_cuda
        )
        return grad_in, None  # None for sigma


def msaa_downsample_rgba(img: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
    """
    MSAA pixel reconstruction downsample for float32 RGBA [height,width,4].

    Isotropic Gaussian weighted downsample from 2x supersampled input,
    producing an output of shape [height/2, width/2, 3].  The input's
    alpha channel is ignored (the rasterizer pads it to 1); the filter
    kernel is a 6x6 hi-res footprint covering +/-1 output pixel.

    Parameters
    ----------
    img   : float32 [height, width, 4]  - 2x supersampled RGBA image.
    sigma : Gaussian sigma in *output* pixels (default 0.5).

    Width and height must be even.  Supports CPU and CUDA tensors with autograd.
    """
    return _MsaaDownsampleRGBAFn.apply(img, sigma)


def upsample2x2_scalar(img: torch.Tensor) -> torch.Tensor:
    """
    Naive 2x2 upsample for float32 [height, width] scalar image.
    """
    if img.ndim != 2:
        raise ValueError(f"Expected 2-D tensor [H, W], got ndim={img.ndim}")
    if img.dtype != torch.float32:
        raise ValueError(f"Expected float32, got {img.dtype}")
    return img.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1).contiguous()


# =============================================================================
# SH evaluation + vertex attribute packing
# =============================================================================

# Spherical harmonic constants (l=0..3, 16 basis functions).
# Signs match the 3DGS convention.
_C0 = 0.28209479177387814
_C1 = 0.4886025119029199
_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def _eval_sh16(d: torch.Tensor) -> torch.Tensor:
    """Evaluate 16 real SH basis functions (l=0..3) on unit directions.

    Uses the 3DGS sign convention: ``d`` is the direction from
    camera to point (``d = pos - campos``).

    Parameters
    ----------
    d : float32 [N, 3]
        Unit-length direction vectors.

    Returns
    -------
    Y : float32 [N, 16]
    """
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    xx, yy, zz = x * x, y * y, z * z

    return torch.stack([
        x.new_full(x.shape, _C0),                          # 0  (l=0)
        -_C1 * y,                                          # 1  (l=1)
        _C1 * z,                                           # 2
        -_C1 * x,                                          # 3
        _C2[0] * x * y,                                    # 4  (l=2)
        _C2[1] * y * z,                                    # 5
        _C2[2] * (2.0 * zz - xx - yy),                     # 6
        _C2[3] * x * z,                                    # 7
        _C2[4] * (xx - yy),                                # 8
        _C3[0] * y * (3.0 * xx - yy),                      # 9  (l=3)
        _C3[1] * x * y * z,                                # 10
        _C3[2] * y * (4.0 * zz - xx - yy),                 # 11
        _C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),     # 12
        _C3[4] * x * (4.0 * zz - xx - yy),                 # 13
        _C3[5] * z * (xx - yy),                            # 14
        _C3[6] * x * (xx - 3.0 * yy),                      # 15
    ], dim=-1)                                             # [N, 16]


def eval_sh_attrs(
    positions: torch.Tensor,       # float32 [N, 3]  (requires_grad)
    sh_coeffs: torch.Tensor,       # float32 [N, 3, K]  K in {1,4,9,16}  (requires_grad)
    radius: torch.Tensor,          # float32 [N]  (requires_grad, already decoded)
    *,
    campos: torch.Tensor,          # float32 [3]
) -> torch.Tensor:
    """Evaluate SH up to the degree implied by sh_coeffs, apply +0.5 DC offset,
    and pack vertex attributes.

    Parameters
    ----------
    positions : float32 [N, 3]
        World-space vertex positions.
    sh_coeffs : float32 [N, 3, K]
        Per-vertex, per-channel SH coefficients.  K must be one of
        1 (degree 0), 4 (degree 1), 9 (degree 2), or 16 (degree 3).
        Basis functions beyond the first K are discarded.
    radius : float32 [N]
        Per-vertex radius (already decoded by the caller).
    campos : float32 [3]
        Camera position in world space.

    Returns
    -------
    vert_attrs : float32 [N, 7]
        ``(x, y, z, radius, r, g, b)`` where
        ``rgb = clamp_min(0.5 + SH(view_dir), 0)``.
    """
    K = sh_coeffs.shape[2]
    if K not in (1, 4, 9, 16):
        raise ValueError(f"sh_coeffs.shape[2] must be 1, 4, 9, or 16; got {K}")

    # View direction: position - campos, normalized (3DGS convention).
    campos = campos.to(device=positions.device, dtype=positions.dtype)
    d = positions - campos.unsqueeze(0)                        # [N, 3]
    d = d / (d.norm(dim=-1, keepdim=True).clamp(min=1e-8))     # [N, 3]

    Y = _eval_sh16(d)[:, :K]                                   # [N, K]

    # rgb = clamp_min(0.5 + sum_k(sh[v,c,k] * Y[v,k]), 0), per the 3DGS
    # convention: colors are clamped per primitive before compositing.
    rgb = (0.5 + torch.einsum('nck,nk->nc', sh_coeffs, Y)).clamp_min(0.0)  # [N, 3]

    # Pack: (x, y, z, radius, r, g, b)
    return torch.cat([positions, radius.unsqueeze(-1), rgb], dim=-1).contiguous()


def pack_attrs(
    positions: torch.Tensor,   # float32 [N, 3]  (requires_grad)
    rgb: torch.Tensor,         # float32 [N, 3]  (requires_grad)
    radius: torch.Tensor,      # float32 [N]     (requires_grad)
) -> torch.Tensor:
    """Pack pre-computed per-vertex data into [N,7] vertex attributes.

    The non-SH counterpart of :func:`eval_sh_attrs`.

    Returns
    -------
    vert_attrs : float32 [N, 7]
        ``(x, y, z, radius, r, g, b)``.
    """
    return torch.cat([positions, radius.unsqueeze(-1), rgb], dim=-1).contiguous()
