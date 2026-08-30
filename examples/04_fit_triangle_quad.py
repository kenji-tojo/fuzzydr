# examples/04_fit_triangle_quad.py
#
# The scene from 03 at full resolution, with the line drawn as a quad so that
# its per-vertex radius carries a gradient too.
#
# Everything the backward pass produces is exercised here at once - vertex
# positions and vertex colours for both primitive types, plus the per-vertex
# radius that sets the world-space width of a quad line.  Position and radius
# gradients come from the edge-gradient pass, which is what makes silhouettes
# differentiable; colour gradients come from the interior of the primitives.
#
# The image converges to the reference, but the recovered parameters need not:
# one view cannot separate how far away a line endpoint is from how thick it
# is, so the fit is free to trade depth against radius.  The printed summary
# shows both, which makes that degeneracy easy to see.
#
#   reference.png  the target render
#   initial.png    the perturbed starting point
#   final.png      the fit
#   optimize.mp4   side by side, current | reference (--video_fit_only
#                  drops the reference)
#   loss.png       log-scale loss curve
#   loss.txt       per-iteration loss
#
# Usage:
#   python examples/04_fit_triangle_quad.py
#   python examples/04_fit_triangle_quad.py --iters 2000 --no-msaa

from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

import common
import fuzzydr


# The scene sits in a plane facing the camera.  Vertices 0-2 are the triangle,
# 3-4 the endpoints of the line.
REFERENCE_POSITIONS = torch.tensor([
    [-0.5, -0.4, 0.4],
    [ 0.5, -0.4, 0.4],
    [ 0.0,  0.5, 0.4],
    [-0.6,  0.3, 0.6],
    [ 0.6, -0.5, 0.6],
], dtype=torch.float32)

REFERENCE_COLORS = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
], dtype=torch.float32)

# Per-vertex line radius: the segment is thin at one end and thick at the other.
REFERENCE_LINE_RADIUS = torch.tensor([0.02, 0.08], dtype=torch.float32)

FACES = torch.tensor([[0, 1, 2]], dtype=torch.uint32)
LINES = torch.tensor([[3, 4]], dtype=torch.uint32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=str, default="out/04_fit_triangle_quad")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--lr_pos", type=float, default=5e-4)
    ap.add_argument("--lr_color", type=float, default=1e-2)
    ap.add_argument("--lr_radius", type=float, default=5e-4)
    ap.add_argument("--perturb_pos", type=float, default=0.10)
    ap.add_argument("--perturb_color", type=float, default=0.10)
    ap.add_argument("--perturb_radius", type=float, default=0.02)
    ap.add_argument("--msaa", action=argparse.BooleanOptionalAction, default=True,
                    help="rasterize at 2x and resolve with the Gaussian filter "
                         "instead of rendering straight to the output size")
    ap.add_argument("--video_fit_only", action="store_true",
                    help="write the video without the reference beside it")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--gpu_id", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = common.select_device(args.device, args.gpu_id)
    out_dir = common.make_out_dir(args.out_dir)
    width, height = args.width, args.height

    ref_positions = REFERENCE_POSITIONS.to(device)
    ref_colors = REFERENCE_COLORS.to(device)
    ref_line_radius = REFERENCE_LINE_RADIUS.to(device)
    faces = FACES.to(device)
    lines = LINES.to(device)

    # A fixed camera looking down -Z at the plane the scene lives in.
    eye = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float32)
    viewproj = (
        common.perspective(60.0, width / height, z_near=0.1, z_far=10.0)
        @ common.look_at(eye, torch.zeros(3))
    ).contiguous()

    # Start from a random perturbation of the reference.
    positions = torch.nn.Parameter(
        ref_positions + torch.randn_like(ref_positions) * args.perturb_pos)
    colors = torch.nn.Parameter(
        (ref_colors + torch.randn_like(ref_colors) * args.perturb_color).clamp(0.0, 1.0))
    line_radius = torch.nn.Parameter(
        (ref_line_radius + torch.randn_like(ref_line_radius) * args.perturb_radius)
        .clamp_min(1e-4))

    scale = 2 if args.msaa else 1

    def render(pos: torch.Tensor, col: torch.Tensor, rad: torch.Tensor) -> torch.Tensor:
        """Render the scene, returning RGB in [0, 1]."""
        # Only the line endpoints carry a radius; the face pipeline ignores it.
        radius = torch.cat([pos.new_zeros(FACES.numel()), rad])
        rgba = fuzzydr.rasterize(
            fuzzydr.pack_attrs(pos, col, radius),
            viewproj=viewproj,
            campos=eye,
            faces=faces,
            lines=lines,
            width=width * scale,
            height=height * scale,
            white_bg=False)
        return fuzzydr.msaa_downsample_rgba(rgba) if args.msaa else rgba[..., :3].contiguous()

    fuzzydr.init()
    try:
        with torch.no_grad():
            reference = render(ref_positions, ref_colors, ref_line_radius)
            initial = render(positions, colors, line_radius)
        common.save_png(out_dir / "reference.png", reference)
        common.save_png(out_dir / "initial.png", initial)

        opt = torch.optim.Adam([
            {"params": [positions], "lr": args.lr_pos},
            {"params": [colors], "lr": args.lr_color},
            {"params": [line_radius], "lr": args.lr_radius},
        ])

        reference_u8 = common.to_u8(reference).cpu()
        out_mp4 = out_dir / "optimize.mp4"
        writer = common.open_video(out_mp4, fps=15)
        losses: list[float] = []

        def append_frame(img: torch.Tensor) -> None:
            frame = common.to_u8(img).cpu()
            if not args.video_fit_only:
                frame = torch.cat([frame, reference_u8], dim=1)
            writer.append_data(frame.numpy())

        try:
            append_frame(initial)

            pbar = tqdm(range(1, args.iters + 1), desc="Optimizing", ncols=80)
            for it in pbar:
                opt.zero_grad(set_to_none=True)
                img = render(positions, colors, line_radius)
                loss = (img - reference).abs().mean()
                loss.backward()
                opt.step()

                with torch.no_grad():
                    colors.clamp_(0.0, 1.0)
                    line_radius.clamp_(min=1e-4)

                losses.append(float(loss.detach()))
                pbar.set_postfix(loss=f"{losses[-1]:.3e}")

                if it % 10 == 0 or it == args.iters:
                    with torch.no_grad():
                        append_frame(render(positions, colors, line_radius))
        finally:
            writer.close()
        print(f"Wrote {out_mp4}")

        with torch.no_grad():
            common.save_png(out_dir / "final.png", render(positions, colors, line_radius))
        common.save_loss_plot(out_dir / "loss.png", losses)
        common.save_loss_curve(out_dir / "loss.txt", losses)

        pos_err = (positions.detach() - ref_positions).norm(dim=1)
        col_err = (colors.detach() - ref_colors).norm(dim=1)
        print(f"\nLoss: {losses[0]:.4e} -> {losses[-1]:.4e}")
        print(f"\n{'':>8s}  {'position error':>14s}  {'colour error':>14s}")
        for i, label in enumerate(["tri v0", "tri v1", "tri v2", "line v0", "line v1"]):
            print(f"{label:>8s}  {pos_err[i]:14.5f}  {col_err[i]:14.5f}")
        print(f"\n{'':>8s}  {'radius':>14s}  {'reference':>14s}")
        for i, label in enumerate(["line v0", "line v1"]):
            print(f"{label:>8s}  {line_radius.detach()[i]:14.5f}  "
                  f"{ref_line_radius[i]:14.5f}")
    finally:
        fuzzydr.shutdown()


if __name__ == "__main__":
    main()
