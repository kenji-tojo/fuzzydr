# examples/05_fit_mesh.py
#
# Deform a connected mesh into another one from images alone: a sphere is
# optimized until its renders match multi-view renders of the bunny.  No
# opacity is involved - every primitive stays opaque, and the only thing that
# moves the silhouette is the edge gradient.
#
# Descending directly on vertex positions makes the surface crumple, because
# the image gradient is concentrated on the silhouette and says nothing about
# the vertices behind it.  This uses the reparameterization from "Large Steps
# in Inverse Rendering of Geometry" (Nicolet et al. 2021) through the authors'
# `largesteps` package: optimize
#
#     u = (I + lambda L) v
#
# instead of v, where L is the Laplacian of the mesh.  A step in u is a smooth
# displacement field in v, so the surface moves as a sheet rather than per
# vertex.  The topology never changes, so the matrix is factorized once and
# every iteration is a solve.
#
#   reference.png  the preview view of the target mesh
#   initial.png    the starting sphere
#   final.png      the deformed mesh
#   progress.mp4   the preview view over training, current | reference
#                  (--video_fit_only drops the reference)
#   loss.png       log-scale loss curve
#   loss.txt       per-iteration loss
#   mesh.obj       the deformed mesh with its vertex colours
#
# Usage:
#   python examples/05_fit_mesh.py --device cuda
#   python examples/05_fit_mesh.py --device cuda --lambda_smooth 5 --iters 2000
#
# Note: largesteps.geometry allocates on 'cuda' unconditionally (0.2.x), so
# this example needs a CUDA device.  Every other example runs on CPU.

from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from largesteps.geometry import compute_matrix
from largesteps.optimize import AdamUniform
from largesteps.parameterize import from_differential, to_differential

import common
import fuzzydr


def fit_to_bbox(verts: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Recentre and rescale ``verts`` so it encloses ``target``'s bounding box."""
    def centre_radius(v):
        lo, hi = v.min(dim=0).values, v.max(dim=0).values
        return 0.5 * (lo + hi), max(0.5 * float(torch.linalg.norm(hi - lo)), 1e-6)

    src_centre, src_radius = centre_radius(verts)
    dst_centre, dst_radius = centre_radius(target)
    return ((verts - src_centre) * (dst_radius / src_radius) + dst_centre).contiguous()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", type=str, default=str(common.DEFAULT_MESH),
                    help="the mesh to fit to")
    ap.add_argument("--init_mesh", type=str, default=str(common.REPO_ROOT / "assets" / "sphere.obj"),
                    help="the mesh to start from")
    ap.add_argument("--out_dir", type=str, default="out/05_fit_mesh")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--views", type=int, default=32,
                    help="cameras on a Fibonacci lattice over the sphere; "
                         "all of them are trained on")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--lambda_smooth", type=float, default=5.0,
                    help="lambda in (I + lambda L); larger is stiffer")
    ap.add_argument("--lr_pos", type=float, default=1e-2,
                    help="position step size, as a fraction of the scene radius")
    ap.add_argument("--lr_color", type=float, default=2.5e-3)
    ap.add_argument("--fovy", type=float, default=40.0,
                    help="vertical field of view in degrees; lower is flatter, "
                         "and the camera backs off to keep the framing")
    ap.add_argument("--preview_azimuth", type=float, default=0.0)
    ap.add_argument("--msaa", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--video_fit_only", action="store_true",
                    help="write the video without the reference beside it")
    ap.add_argument("--video_frames", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--gpu_id", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = common.select_device(args.device, args.gpu_id)
    if device.type != "cuda":
        raise SystemExit(
            "05_fit_mesh needs --device cuda: largesteps.geometry builds the "
            "Laplacian with device='cuda' hardcoded, so the reparameterization "
            "cannot be constructed on CPU.")
    out_dir = common.make_out_dir(args.out_dir)
    width, height = args.width, args.height
    scale = 2 if args.msaa else 1

    ref_verts, ref_faces = common.load_obj_mesh(args.mesh)
    ref_colors = common.normals_to_rgb(common.vertex_normals(ref_verts, ref_faces))

    init_verts, faces = common.load_obj_mesh(args.init_mesh)
    init_verts = fit_to_bbox(init_verts, ref_verts)
    num_verts = init_verts.shape[0]

    camera = common.SceneCamera.frame(ref_verts, width, height,
                                     fovy_deg=args.fovy)
    eyes = camera.sphere_eyes(args.views)
    train_views = [(camera.view_proj(eyes[i]), eyes[i]) for i in range(args.views)]
    preview_eye = camera.orbit_eye(args.preview_azimuth)
    preview_view = (camera.view_proj(preview_eye), preview_eye)

    # ------------------------------------------------------------------
    # Large-steps reparameterization
    # ------------------------------------------------------------------
    # compute_matrix factorizes (I + lambda L) once; the topology is fixed for
    # the whole run, so this never has to be redone.
    faces = faces.to(device)
    init_verts = init_verts.to(device)
    matrix = compute_matrix(init_verts, faces.to(torch.int64), args.lambda_smooth)

    # u is the variable actually optimized; from_differential recovers the
    # positions that get rendered, and carries the gradient back to u.
    u = to_differential(matrix, init_verts).clone().requires_grad_(True)
    colors = torch.nn.Parameter(torch.full((num_verts, 3), 0.5, device=device))

    print(f"Fitting {args.init_mesh} ({num_verts} verts, {faces.shape[0]} faces) "
          f"to {args.mesh}, lambda={args.lambda_smooth}, {args.views} views, "
          f"device={device}")

    def positions() -> torch.Tensor:
        return from_differential(matrix, u, "Cholesky")

    def render(view, attrs: torch.Tensor, topology: torch.Tensor) -> torch.Tensor:
        viewproj, eye = view
        rgba = fuzzydr.rasterize(
            attrs, viewproj=viewproj, campos=eye, faces=topology,
            width=width * scale, height=height * scale,
            white_bg=False)
        return (fuzzydr.msaa_downsample_rgba(rgba) if args.msaa
                else rgba[..., :3].contiguous())

    def render_current(view) -> torch.Tensor:
        verts = positions()
        return render(view, fuzzydr.pack_attrs(
            verts, colors, verts.new_zeros(num_verts)), faces)

    fuzzydr.init()
    try:
        ref_attrs = fuzzydr.pack_attrs(
            ref_verts.to(device), ref_colors.to(device),
            torch.zeros(ref_verts.shape[0], device=device))
        ref_topology = ref_faces.to(device)

        references = [render(v, ref_attrs, ref_topology)
                      for v in tqdm(train_views, desc="Reference views", ncols=80)]
        preview_reference = render(preview_view, ref_attrs, ref_topology)

        with torch.no_grad():
            common.save_png(out_dir / "reference.png", preview_reference)
            common.save_png(out_dir / "initial.png", render_current(preview_view))

        # AdamUniform is the largesteps optimizer: it takes a uniformly sized
        # step in u regardless of gradient magnitude, which is what keeps the
        # reparameterized descent well conditioned.  Colours are unrelated to
        # the geometry and use plain Adam.
        opt_pos = AdamUniform([u], lr=args.lr_pos * camera.radius)
        opt_color = torch.optim.Adam([colors], lr=args.lr_color)

        losses: list[float] = []
        video_iters = {round(k * args.iters / max(1, args.video_frames - 1))
                       for k in range(args.video_frames)}
        preview_reference_u8 = common.to_u8(preview_reference).cpu()

        out_mp4 = out_dir / "progress.mp4"
        writer = common.open_video(out_mp4, fps=30)

        def append_frame(img: torch.Tensor) -> None:
            frame = common.to_u8(img).cpu()
            if not args.video_fit_only:
                frame = torch.cat([frame, preview_reference_u8], dim=1)
            writer.append_data(frame.numpy())

        try:
            with torch.no_grad():
                append_frame(render_current(preview_view))

            view_order = torch.randperm(args.views).tolist()
            pbar = tqdm(range(1, args.iters + 1), desc="Deforming", ncols=90)
            for it in pbar:
                if (it - 1) % args.views == 0:
                    view_order = torch.randperm(args.views).tolist()
                view_idx = view_order[(it - 1) % args.views]

                opt_pos.zero_grad(set_to_none=True)
                opt_color.zero_grad(set_to_none=True)
                loss = (render_current(train_views[view_idx])
                        - references[view_idx]).abs().mean()
                loss.backward()
                opt_pos.step()
                opt_color.step()

                with torch.no_grad():
                    colors.clamp_(0.0, 1.0)

                losses.append(float(loss.detach()))
                pbar.set_postfix(loss=f"{losses[-1]:.3e}")

                if it in video_iters:
                    with torch.no_grad():
                        append_frame(render_current(preview_view))
        finally:
            writer.close()
        print(f"Wrote {out_mp4}")

        with torch.no_grad():
            common.save_png(out_dir / "final.png", render_current(preview_view))
            common.save_obj_with_vertex_colors(
                out_dir / "mesh.obj", positions(), faces, colors)
        common.save_loss_plot(out_dir / "loss.png", losses)
        common.save_loss_curve(out_dir / "loss.txt", losses)

        print(f"\nLoss: {losses[0]:.4e} -> {losses[-1]:.4e}")
    finally:
        fuzzydr.shutdown()


if __name__ == "__main__":
    main()
