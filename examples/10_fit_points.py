# examples/10_fit_points.py
#
# Multi-view inverse rendering with stochastic opacity masking: fit a point
# cloud to renders of a reference mesh, and let the opacities decide how many
# points survive.
#
# The primitives start scattered at random through the reference bounding box,
# all nearly transparent.  Three things are trained together:
#
#   * vertex positions, through the edge-gradient pass;
#   * vertex colours, through the primitive interiors;
#   * per-primitive opacity, through fuzzydr.opacity_mask_aux_loss, which turns
#     the rendered error into a gradient on the random masking decision - a
#     primitive that helps the image gets pushed towards opaque, one that hurts
#     it towards transparent.
#
# Every so often the set is resampled: primitives that have gone transparent
# are deleted, and a random fraction of the survivors is duplicated - a clone starts on top of
# its original but draws its own stochastic threshold, so the pair covers the
# pixel more often and the two receive gradients on different frames.
#
#   progress.mp4     the preview view over training, current | reference
#                    (--video_fit_only drops the reference)
#   reference.png    the preview view's reference render
#   final.png        the fit, drawn with every surviving primitive opaque
#   loss.png         log-scale loss curve
#   loss.txt         per-iteration loss
#   points.npz       the fitted primitives (see common.save_checkpoint)
#
# Usage:
#   python examples/10_fit_points.py
#   python examples/10_fit_points.py --iters 3000

from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

import common
import fitting
import fuzzydr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", type=str, default=str(common.DEFAULT_MESH))
    ap.add_argument("--out_dir", type=str, default="out/10_fit_points")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--views", type=int, default=32,
                    help="cameras on a Fibonacci lattice over the sphere; "
                         "all of them are trained on")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--count", type=int, default=500_000,
                    help="primitives to start from")
    ap.add_argument("--opacity_init", type=float, default=0.1,
                    help="opacity every primitive starts at")
    ap.add_argument("--lr_pos", type=float, default=2.5e-5,
                    help="multiplied by the scene's spatial scale")
    ap.add_argument("--lr_color", type=float, default=2.5e-3)
    ap.add_argument("--lr_opacity", type=float, default=5e-2)
    ap.add_argument("--resample_every", type=int, default=200)
    ap.add_argument("--resample_start", type=int, default=200)
    ap.add_argument("--resample_end", type=float, default=0.9,
                    help="stop resampling after this fraction of the run, so "
                         "the last stretch only refines what is left")
    ap.add_argument("--max_growth", type=float, default=0.05,
                    help="most a single resample may add, as a fraction of the "
                         "primitives that survived pruning")
    ap.add_argument("--prune_threshold", type=float, default=0.05,
                    help="primitives below this opacity are deleted during training")
    ap.add_argument("--final_tau", type=float, default=0.5,
                    help="deterministic opacity threshold for the final render; "
                         "training itself always uses stochastic masking")
    ap.add_argument("--budget", type=int, default=None,
                    help="primitive count to hold, refilled at most "
                         "--max_growth per resample; defaults to --count, so "
                         "whatever pruning removes is put back by duplication")
    ap.add_argument("--fovy", type=float, default=40.0,
                    help="vertical field of view in degrees; lower is flatter, "
                         "and the camera backs off to keep the framing")
    ap.add_argument("--preview_azimuth", type=float, default=0.0,
                    help="azimuth in radians of the camera the video uses")
    ap.add_argument("--msaa", action=argparse.BooleanOptionalAction, default=True,
                    help="rasterize at 2x and resolve with the Gaussian filter")
    ap.add_argument("--video_fit_only", action="store_true",
                    help="write the video without the reference beside it")
    ap.add_argument("--video_frames", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--gpu_id", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = common.select_device(args.device, args.gpu_id)
    out_dir = common.make_out_dir(args.out_dir)
    width, height = args.width, args.height
    scale = 2 if args.msaa else 1

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------
    ref_verts, ref_faces = common.load_obj_mesh(args.mesh)
    ref_colors = common.normals_to_rgb(common.vertex_normals(ref_verts, ref_faces))
    camera = common.SceneCamera.frame(ref_verts, width, height,
                                     fovy_deg=args.fovy)

    # Training cameras are spread over the whole sphere, which puts several of
    # them overhead.  The video instead uses a separate three-quarter view from
    # the orbit, which is the angle the scene actually reads from.
    eyes = camera.sphere_eyes(args.views)
    train_views = [(camera.view_proj(eyes[i]), eyes[i]) for i in range(args.views)]
    lr_pos = args.lr_pos * common.spatial_lr_scale(eyes)
    preview_eye = camera.orbit_eye(args.preview_azimuth)
    preview_view = (camera.view_proj(preview_eye), preview_eye)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    bbox_min = ref_verts.min(dim=0).values
    bbox_max = ref_verts.max(dim=0).values
    init_verts, init_prims = common.make_random_points(
        args.count, bbox_min, bbox_max)

    primitives = fitting.Primitives(
        init_prims.to(device),
        init_verts.to(device),
        torch.rand_like(init_verts).to(device),
        opacity_init=args.opacity_init,
        lr_pos=lr_pos, lr_color=args.lr_color, lr_opacity=args.lr_opacity,
    )

    print(f"[points] {primitives.count} primitives at opacity {args.opacity_init}, "
          f"{args.views} views at {width}x{height}, msaa={args.msaa}, device={device}")

    def render(view, *, seed: int, tau: float = -1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Render the current primitives; returns (opacity, RGB).

        ``tau`` defaults to -1, i.e. stochastic opacity masking, which is what
        training uses.  Once training is over the fit is rendered with a fixed
        ``tau`` instead, so the image is deterministic: every primitive whose
        opacity is below the threshold is culled outright.
        """
        viewproj, eye = view
        opacity = primitives.opacity()
        attrs = fuzzydr.pack_attrs(
            primitives.verts, primitives.colors,
            primitives.verts.new_full((primitives.num_verts,), 0.0))
        rgba = fuzzydr.rasterize_points(
            attrs, viewproj=viewproj, campos=eye,
            points=primitives.prims.reshape(-1), point_opacity=opacity,
            width=width * scale, height=height * scale, tau=tau, seed=seed,
            white_bg=False)
        rgb = (fuzzydr.msaa_downsample_rgba(rgba) if args.msaa
               else rgba[..., :3].contiguous())
        return opacity, rgb

    def opacity_aux_loss(opacity: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
        # opacity_grad reads the primitive-ID buffer of the render that just
        # ran, so the error image has to match that render's resolution - which
        # is 2x the loss resolution whenever MSAA is on.
        if args.msaa:
            error = fuzzydr.upsample2x2_scalar(error)
        return fuzzydr.opacity_mask_aux_loss(point_opacity=opacity, error=error)

    fuzzydr.init()
    try:
        # ------------------------------------------------------------------
        # Reference renders.  The mesh has no per-face opacity, so it defaults
        # to 1 and the default threshold keeps all of it.
        # ------------------------------------------------------------------
        ref_attrs = fuzzydr.pack_attrs(
            ref_verts.to(device), ref_colors.to(device),
            torch.zeros(ref_verts.shape[0], device=device))
        ref_faces = ref_faces.to(device)

        def render_reference(view) -> torch.Tensor:
            viewproj, eye = view
            rgba = fuzzydr.rasterize(
                ref_attrs, viewproj=viewproj, campos=eye, faces=ref_faces,
                width=width * scale, height=height * scale,
                white_bg=False)
            return (fuzzydr.msaa_downsample_rgba(rgba) if args.msaa
                    else rgba[..., :3].contiguous())

        references = [render_reference(v)
                      for v in tqdm(train_views, desc="Reference views", ncols=80)]
        preview_reference = render_reference(preview_view)

        # ------------------------------------------------------------------
        # Training
        # ------------------------------------------------------------------
        losses: list[float] = []
        resample_end = int(args.iters * args.resample_end)
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
                append_frame(render(preview_view, seed=0)[1])

            view_order = torch.randperm(args.views).tolist()
            pbar = tqdm(range(1, args.iters + 1), desc="Optimizing", ncols=100)
            for it in pbar:
                # Cycle the views in a fresh random order each sweep.
                if (it - 1) % args.views == 0:
                    view_order = torch.randperm(args.views).tolist()
                view_idx = view_order[(it - 1) % args.views]

                primitives.zero_grad()
                opacity, img = render(train_views[view_idx], seed=it)
                residual = (img - references[view_idx]).abs()
                loss = residual.mean()
                aux = opacity_aux_loss(
                    opacity, residual.mean(dim=-1).detach().contiguous())
                (loss + aux).backward()
                primitives.step()

                losses.append(float(loss.detach()))

                if (args.resample_start <= it <= resample_end
                        and it % args.resample_every == 0):
                    summary = primitives.resample(
                        prune_threshold=args.prune_threshold,
                        target_count=args.budget or args.count,
                        max_growth=args.max_growth)
                    if summary:
                        pbar.write(f"[resample] it={it}  {summary}")

                pbar.set_postfix(loss=f"{losses[-1]:.3e}", n=primitives.count)

                if it in video_iters:
                    with torch.no_grad():
                        append_frame(render(preview_view, seed=it)[1])
        finally:
            writer.close()
        print(f"Wrote {out_mp4}")

        # ------------------------------------------------------------------
        # Results
        # ------------------------------------------------------------------
        with torch.no_grad():
            common.save_png(out_dir / "reference.png", preview_reference)
            common.save_png(out_dir / "final.png",
                            render(preview_view, seed=0, tau=args.final_tau)[1])
        common.save_loss_plot(out_dir / "loss.png", losses)
        common.save_loss_curve(out_dir / "loss.txt", losses)

        common.save_checkpoint(
            out_dir / "points.npz",
            primitive="points",
            verts=primitives.verts.detach(),
            colors=primitives.colors.detach(),
            prims=primitives.prims,
            opacity_logit=primitives.opacity_logit.detach(),
            radius=primitives.verts.detach().new_full(
                (primitives.num_verts,), 0.0),
            meta={
                "reference_mesh": str(args.mesh),
                "iters": args.iters,
                "opacity_init": args.opacity_init,
                "prune_threshold": args.prune_threshold,
                "msaa": args.msaa,
            },
        )

        kept = int((primitives.opacity().detach() >= args.final_tau).sum())
        print(f"\nLoss: {losses[0]:.4e} -> {losses[-1]:.4e}")
        print(f"{kept} of {primitives.count} at opacity >= {args.final_tau}")
        print(f"points: {args.count} -> {primitives.count}")
    finally:
        fuzzydr.shutdown()


if __name__ == "__main__":
    main()
