# examples/06_stochastic_opacity.py
#
# Stochastic opacity masking: with tau = -1 the rasterizer draws one random
# threshold per primitive per frame and discards the primitive when its opacity
# falls below it.  A primitive therefore survives with probability equal to its
# opacity, and the depth test keeps the nearest survivor - so across seeds the
# result is the alpha composite, reached without alpha blending and without the
# depth sort that would need.
#
# The threshold is drawn per primitive rather than per pixel, so a primitive is
# either wholly present or wholly absent in a frame.
#
#   opaque.png    the scene with tau = 0, for reference
#   sample.png    one stochastic draw: a random subset of the triangles
#   orbit.mp4     an orbit that redraws the mask every frame
#
# Usage:
#   python examples/06_stochastic_opacity.py
#   python examples/06_stochastic_opacity.py --opacity 0.25

from __future__ import annotations

import argparse
import math

import torch
from tqdm import tqdm

import common
import fuzzydr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", type=str, default=str(common.DEFAULT_MESH))
    ap.add_argument("--out_dir", type=str, default="out/06_stochastic_opacity")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--opacity", type=float, default=0.5,
                    help="per-face opacity, shared by every triangle")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--gpu_id", type=int, default=0)
    args = ap.parse_args()

    if not 0.0 <= args.opacity <= 1.0:
        ap.error("--opacity must be in [0, 1]")

    device = common.select_device(args.device, args.gpu_id)
    out_dir = common.make_out_dir(args.out_dir)
    width, height = args.width, args.height
    num_frames = max(1, int(round(args.fps * args.seconds)))

    verts, faces = common.load_obj_mesh(args.mesh)
    colors = common.normals_to_rgb(common.vertex_normals(verts, faces))
    camera = common.SceneCamera.frame(verts, width, height)

    verts = verts.to(device)
    faces = faces.to(device)
    vert_attrs = fuzzydr.pack_attrs(
        verts, colors.to(device), torch.zeros(verts.shape[0], device=device)
    )
    face_opacity = torch.full((faces.shape[0],), args.opacity,
                              dtype=torch.float32, device=device)

    def render(viewproj: torch.Tensor, eye: torch.Tensor, *,
               tau: float, seed: int) -> torch.Tensor:
        """Render at 2x and resolve, returning RGB in [0, 1]."""
        rgba = fuzzydr.rasterize(
            vert_attrs,
            viewproj=viewproj,
            campos=eye,
            faces=faces,
            face_opacity=face_opacity,
            width=width * 2,
            height=height * 2,
            tau=tau,
            seed=seed,
            white_bg=False)
        return fuzzydr.msaa_downsample_rgba(rgba)

    fuzzydr.init()
    try:
        eye = camera.orbit_eye(azimuth_rad=0.0)
        viewproj = camera.view_proj(eye)

        # An opacity is never below 0, so tau = 0 keeps every primitive: this
        # is the opaque render that the masked ones are drawn from.
        common.save_png(out_dir / "opaque.png", render(viewproj, eye, tau=0.0, seed=0))
        common.save_png(out_dir / "sample.png", render(viewproj, eye, tau=-1.0, seed=0))

        out_mp4 = out_dir / "orbit.mp4"
        writer = common.open_video(out_mp4, fps=args.fps)
        try:
            for frame in tqdm(range(num_frames), desc="Rendering orbit", ncols=80):
                eye = camera.orbit_eye(2.0 * math.pi * frame / num_frames)
                img = render(camera.view_proj(eye), eye, tau=-1.0, seed=frame)
                writer.append_data(common.to_u8(img).cpu().numpy())
        finally:
            writer.close()
        print(f"Wrote {out_mp4} ({num_frames} frames @ {args.fps} fps)")
    finally:
        fuzzydr.shutdown()


if __name__ == "__main__":
    main()
