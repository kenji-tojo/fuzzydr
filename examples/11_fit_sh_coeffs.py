# examples/11_fit_sh_coeffs.py
#
# Fit view-dependent colour with spherical harmonics.
#
# The reference is a specular-shaded mesh, so its appearance changes with the
# camera.  The geometry is held fixed and only the per-vertex SH coefficients
# are trained: fuzzydr.eval_sh_attrs evaluates them against the current camera
# position and packs the result into vertex attributes, and autograd carries
# the image gradient back through that evaluation.
#
#   reference.png  the preview view, shaded
#   initial.png    the preview view at initialisation (all coefficients zero)
#   final.png      the preview view after fitting
#   progress.mp4   the preview view over training, fit | reference
#                  (--video_fit_only drops the reference)
#   orbit.mp4      an orbit of the finished fit, fit | reference
#   loss.png       log-scale loss curve
#   loss.txt       per-iteration loss
#
# Usage:
#   python examples/11_fit_sh_coeffs.py
#   python examples/11_fit_sh_coeffs.py --sh_degree 0   # view-independent
#   python examples/11_fit_sh_coeffs.py --shininess 8

from __future__ import annotations

import argparse
import math

import torch
from tqdm import tqdm

import common
import fuzzydr


# Coefficients per colour channel for each SH degree, as accepted by
# fuzzydr.eval_sh_attrs.
SH_COEFF_COUNT = {0: 1, 1: 4, 2: 9, 3: 16}


def shade(
    positions: torch.Tensor,   # float32 [N, 3]
    normals: torch.Tensor,     # float32 [N, 3]
    albedo: torch.Tensor,      # float32 [N, 3]
    eye: torch.Tensor,         # float32 [3]
    *,
    light_dir: torch.Tensor,   # float32 [3]  unit, world space
    shininess: float,
) -> torch.Tensor:
    """Blinn-Phong vertex colours in [0, 1].

    The specular lobe is what makes this view-dependent, and therefore what
    the SH coefficients have to learn.
    """
    view_dir = torch.nn.functional.normalize(eye - positions, dim=-1)
    half = torch.nn.functional.normalize(light_dir + view_dir, dim=-1)

    diffuse = (normals * light_dir).sum(-1, keepdim=True).clamp_min(0.0)
    specular = (normals * half).sum(-1, keepdim=True).clamp_min(0.0) ** shininess
    return (albedo * (0.25 + 0.75 * diffuse) + specular).clamp(0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", type=str, default=str(common.DEFAULT_MESH))
    ap.add_argument("--out_dir", type=str, default="out/11_fit_sh_coeffs")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--views", type=int, default=32,
                    help="cameras on a Fibonacci lattice over the sphere; "
                         "all of them are trained on")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--sh_degree", type=int, default=3, choices=sorted(SH_COEFF_COUNT))
    ap.add_argument("--lr_sh_dc", type=float, default=1e-2,
                    help="learning rate for the DC (l=0) coefficients")
    ap.add_argument("--lr_sh_rest", type=float, default=1e-3,
                    help="learning rate for the higher bands, which carry the "
                         "view-dependent part.  vkgrad trains these 20x slower "
                         "than the DC term over 30k iterations; this example "
                         "runs 1000, so the gap is narrowed to 10x")
    ap.add_argument("--fovy", type=float, default=40.0,
                    help="vertical field of view in degrees; lower is flatter, "
                         "and the camera backs off to keep the framing")
    ap.add_argument("--preview_azimuth", type=float, default=0.0,
                    help="azimuth in radians of the camera the stills and video use")
    ap.add_argument("--shininess", type=float, default=24.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--video_fit_only", action="store_true",
                    help="write the video without the reference beside it")
    ap.add_argument("--video_frames", type=int, default=120,
                    help="frames sampled from training into progress.mp4")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--gpu_id", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = common.select_device(args.device, args.gpu_id)
    out_dir = common.make_out_dir(args.out_dir)
    width, height = args.width, args.height
    num_frames = max(1, int(round(args.fps * args.seconds)))

    verts, faces = common.load_obj_mesh(args.mesh)
    normals = common.vertex_normals(verts, faces)
    albedo = common.normals_to_rgb(normals)
    camera = common.SceneCamera.frame(verts, width, height,
                                     fovy_deg=args.fovy)

    eyes = camera.sphere_eyes(args.views)
    # The camera the stills and the video use.  Training uses every lattice
    # view; this one only decides what you look at.
    preview_eye = camera.orbit_eye(args.preview_azimuth)

    verts = verts.to(device)
    faces = faces.to(device)
    normals = normals.to(device)
    albedo = albedo.to(device)
    light_dir = torch.nn.functional.normalize(
        torch.tensor([1.0, 1.0, 1.0], device=device), dim=0)
    radius = torch.zeros(verts.shape[0], device=device)

    # All coefficients start at zero: eval_sh_attrs adds a 0.5 offset, so the
    # initial render is flat grey and every bit of colour has to be learned.
    #
    # The DC term is the view-independent base colour and the higher bands are
    # the view-dependent correction on top of it, so they are separate
    # parameters and take separate learning rates - the same split 3DGS uses.
    num_coeffs = SH_COEFF_COUNT[args.sh_degree]
    sh_dc = torch.nn.Parameter(torch.zeros((verts.shape[0], 3, 1), device=device))
    sh_rest = torch.nn.Parameter(
        torch.zeros((verts.shape[0], 3, num_coeffs - 1), device=device))

    def sh_coeffs() -> torch.Tensor:
        return torch.cat([sh_dc, sh_rest], dim=2) if num_coeffs > 1 else sh_dc

    def render_reference(eye: torch.Tensor) -> torch.Tensor:
        colors = shade(verts, normals, albedo, eye.to(device),
                       light_dir=light_dir, shininess=args.shininess)
        return render(fuzzydr.pack_attrs(verts, colors, radius), eye)

    def render_sh(eye: torch.Tensor) -> torch.Tensor:
        attrs = fuzzydr.eval_sh_attrs(verts, sh_coeffs(), radius, campos=eye.to(device))
        return render(attrs, eye)

    def render(attrs: torch.Tensor, eye: torch.Tensor) -> torch.Tensor:
        rgba = fuzzydr.rasterize(
            attrs,
            viewproj=camera.view_proj(eye),
            campos=eye,
            faces=faces,
            width=width * 2,
            height=height * 2,
            white_bg=False)
        return fuzzydr.msaa_downsample_rgba(rgba)

    print(f"Fitting degree-{args.sh_degree} SH ({num_coeffs} coefficients per "
          f"channel) on {verts.shape[0]} vertices, {args.views} views")

    fuzzydr.init()
    try:
        with torch.no_grad():
            references = [render_reference(eyes[i]) for i in range(args.views)]
            preview_reference = render_reference(preview_eye)
            common.save_png(out_dir / "reference.png", preview_reference)
            common.save_png(out_dir / "initial.png", render_sh(preview_eye))

        opt = torch.optim.Adam([
            {"params": [sh_dc], "lr": args.lr_sh_dc},
            {"params": [sh_rest], "lr": args.lr_sh_rest},
        ])
        losses: list[float] = []

        video_iters = {round(k * args.iters / max(1, args.video_frames - 1))
                       for k in range(args.video_frames)}
        preview_reference_u8 = common.to_u8(preview_reference).cpu()
        out_progress = out_dir / "progress.mp4"
        writer = common.open_video(out_progress, fps=args.fps)

        def append_frame(img: torch.Tensor) -> None:
            frame = common.to_u8(img).cpu()
            if not args.video_fit_only:
                frame = torch.cat([frame, preview_reference_u8], dim=1)
            writer.append_data(frame.numpy())

        try:
            with torch.no_grad():
                append_frame(render_sh(preview_eye))

            view_order = torch.randperm(args.views).tolist()
            pbar = tqdm(range(1, args.iters + 1), desc="Fitting SH", ncols=90)
            for it in pbar:
                if (it - 1) % args.views == 0:
                    view_order = torch.randperm(args.views).tolist()
                view_idx = view_order[(it - 1) % args.views]

                opt.zero_grad(set_to_none=True)
                loss = (render_sh(eyes[view_idx]) - references[view_idx]).abs().mean()
                loss.backward()
                opt.step()

                losses.append(float(loss.detach()))
                pbar.set_postfix(loss=f"{losses[-1]:.3e}")

                if it in video_iters:
                    with torch.no_grad():
                        append_frame(render_sh(preview_eye))
        finally:
            writer.close()
        print(f"Wrote {out_progress}")

        with torch.no_grad():
            common.save_png(out_dir / "final.png", render_sh(preview_eye))
        common.save_loss_plot(out_dir / "loss.png", losses)
        common.save_loss_curve(out_dir / "loss.txt", losses)

        out_mp4 = out_dir / "orbit.mp4"
        writer = common.open_video(out_mp4, fps=args.fps)
        try:
            for frame in tqdm(range(num_frames), desc="Rendering orbit", ncols=90):
                eye = camera.orbit_eye(2.0 * math.pi * frame / num_frames)
                with torch.no_grad():
                    side_by_side = torch.cat(
                        [common.to_u8(render_sh(eye)).cpu(),
                         common.to_u8(render_reference(eye)).cpu()], dim=1)
                writer.append_data(side_by_side.numpy())
        finally:
            writer.close()
        print(f"Wrote {out_mp4} ({num_frames} frames @ {args.fps} fps)")

        print(f"\nLoss: {losses[0]:.4e} -> {losses[-1]:.4e}")
    finally:
        fuzzydr.shutdown()


if __name__ == "__main__":
    main()
