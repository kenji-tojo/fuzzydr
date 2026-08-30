# examples/01_render.py
#
# Forward rendering: rasterize a triangle mesh and write out every buffer the
# renderer can hand back.
#
#   color.png       one sample per pixel
#   color_msaa.png  rendered at 2x and resolved with fuzzydr.msaa_downsample_rgba
#   prim_id.png     per-pixel triangle index (aux buffer)
#   depth.png       per-pixel window-space depth (aux buffer)
#
# Usage:
#   python examples/01_render.py
#   python examples/01_render.py --device cuda

from __future__ import annotations

import argparse

import torch

import common
import fuzzydr


def colorize_prim_id(prim_id: torch.Tensor) -> torch.Tensor:
    """Hash a ``uint32 [H, W]`` primitive-ID buffer to a random-looking RGB image.

    Background pixels carry ``0xFFFFFFFF`` and come out black.
    """
    # uint32 has no comparison or arithmetic support in torch, but the ID range
    # used here fits in int32, where the background sentinel reads as -1.
    ids = prim_id.view(torch.int32).to(torch.int64)
    background = ids < 0

    x = torch.where(background, torch.zeros_like(ids), ids)
    channels = [((x * m + a) >> 16) & 255 for m, a in
                ((1664525, 1013904223), (22695477, 1), (1103515245, 12345))]
    rgb = torch.stack(channels, dim=-1).to(torch.uint8)
    rgb[background] = 0
    return rgb


def colorize_depth(bary_depth: torch.Tensor, prim_id: torch.Tensor) -> torch.Tensor:
    """Grey-scale depth image, normalised over the covered pixels.

    Near surfaces are bright and far ones dim; the background stays black so it
    is distinguishable from the far end of the range.
    """
    depth = bary_depth[..., 2]
    covered = prim_id.view(torch.int32) >= 0
    if not bool(covered.any()):
        raise ValueError("nothing was rasterized; check the camera framing")

    near = depth[covered].min()
    far = depth[covered].max()
    depth01 = ((depth - near) / (far - near).clamp_min(1e-8)).clamp(0.0, 1.0)
    grey = 1.0 - 0.8 * depth01
    grey = torch.where(covered, grey, torch.zeros_like(grey))
    return grey.unsqueeze(-1).expand(-1, -1, 3).contiguous()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", type=str, default=str(common.DEFAULT_MESH))
    ap.add_argument("--out_dir", type=str, default="out/01_render")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--gpu_id", type=int, default=0)
    args = ap.parse_args()

    device = common.select_device(args.device, args.gpu_id)
    out_dir = common.make_out_dir(args.out_dir)
    width, height = args.width, args.height

    verts, faces = common.load_obj_mesh(args.mesh)
    colors = common.normals_to_rgb(common.vertex_normals(verts, faces))
    print(f"Loaded {args.mesh}: {verts.shape[0]} verts, {faces.shape[0]} faces")

    camera = common.SceneCamera.frame(verts, width, height)
    eye = camera.orbit_eye(azimuth_rad=0.0)
    viewproj = camera.view_proj(eye)

    verts = verts.to(device)
    faces = faces.to(device)
    # (x, y, z, radius, r, g, b); radius drives quad line width only, so a
    # face-only scene can leave it at zero.
    vert_attrs = fuzzydr.pack_attrs(
        verts, colors.to(device), torch.zeros(verts.shape[0], device=device)
    )

    fuzzydr.init()
    try:
        prim_id, bary_depth, rgba = fuzzydr.rasterize(
            vert_attrs,
            viewproj=viewproj,
            campos=eye,
            faces=faces,
            width=width,
            height=height,
            aux_buffers=True,
            white_bg=False)
        common.save_png(out_dir / "color.png", rgba[..., :3])
        common.save_png(out_dir / "prim_id.png", colorize_prim_id(prim_id))
        common.save_png(out_dir / "depth.png", colorize_depth(bary_depth, prim_id))

        # Anti-aliasing: rasterize at twice the output resolution and resolve
        # with the Gaussian downsample.  The filter takes RGBA in and returns
        # RGB at half the size, so this lands back at width x height.
        rgba_2x = fuzzydr.rasterize(
            vert_attrs,
            viewproj=viewproj,
            campos=eye,
            faces=faces,
            width=width * 2,
            height=height * 2,
            white_bg=False)
        common.save_png(out_dir / "color_msaa.png", fuzzydr.msaa_downsample_rgba(rgba_2x))
    finally:
        fuzzydr.shutdown()


if __name__ == "__main__":
    main()
