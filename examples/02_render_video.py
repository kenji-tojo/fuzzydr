# examples/02_render_video.py
#
# Render a mesh from an orbiting camera and write the frames to an MP4.
#
# The topology is uploaded once and then reused: passing an int instead of the
# index tensor tells fuzzydr to keep the cached buffer, which is how a training
# loop avoids re-uploading static connectivity every iteration.
#
# Usage:
#   python examples/02_render_video.py
#   python examples/02_render_video.py --no-msaa --seconds 2

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
    ap.add_argument("--out_dir", type=str, default="out/02_render_video")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--msaa", action=argparse.BooleanOptionalAction, default=True,
                    help="rasterize at 2x and resolve with the Gaussian filter "
                         "instead of rendering straight to the output size")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--gpu_id", type=int, default=0)
    args = ap.parse_args()

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

    # Rasterize at 2x and resolve when antialiasing, otherwise straight to size.
    scale = 2 if args.msaa else 1
    out_mp4 = out_dir / "orbit.mp4"

    fuzzydr.init()
    writer = common.open_video(out_mp4, fps=args.fps)
    try:
        # First frame uploads the index buffer; the rest pass the face count so
        # the cached topology is reused.
        topology: torch.Tensor | int = faces

        for frame in tqdm(range(num_frames), desc="Rendering", ncols=80):
            eye = camera.orbit_eye(2.0 * math.pi * frame / num_frames)
            rgba = fuzzydr.rasterize(
                vert_attrs,
                viewproj=camera.view_proj(eye),
                campos=eye,
                faces=topology,
                width=width * scale,
                height=height * scale,
                white_bg=False)
            topology = int(faces.shape[0])

            img = (fuzzydr.msaa_downsample_rgba(rgba) if args.msaa
                   else rgba[..., :3].contiguous())
            writer.append_data(common.to_u8(img).cpu().numpy())
    finally:
        writer.close()
        fuzzydr.shutdown()

    print(f"Wrote {out_mp4} ({num_frames} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
