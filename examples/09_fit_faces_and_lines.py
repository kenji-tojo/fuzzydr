# examples/09_fit_faces_and_lines.py
#
# The same fit as 07 and 08, but with triangles and line segments optimized
# together in one render.  Both classes go into a single fuzzydr.rasterize
# call, so they share a depth buffer and compete for the same pixels: a region
# the triangles explain badly can be taken over by lines, and vice versa.
#
# This is the configuration the per-primitive RNG counter offset exists for.
# Primitive IDs are numbered per class, so face i and line i would otherwise
# draw the same stochastic threshold and be masked in lockstep wherever they
# overlap (see shaders/rasterize.frag).
#
#   progress.mp4     the preview view over training, current | reference
#                    (--video_fit_only drops the reference)
#   reference.png    the preview view's reference render
#   final.png        the fit, drawn with every surviving primitive opaque
#   loss.png         log-scale loss curve
#   loss.txt         per-iteration loss
#   faces.npz        the fitted triangles (see common.save_checkpoint)
#   lines.npz        the fitted line segments - a checkpoint holds one
#                    primitive class, so the two are written separately
#
# Usage:
#   python examples/09_fit_faces_and_lines.py
#   python examples/09_fit_faces_and_lines.py --iters 3000

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
    ap.add_argument("--out_dir", type=str, default="out/09_fit_faces_and_lines")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--views", type=int, default=32,
                    help="cameras on a Fibonacci lattice over the sphere; "
                         "all of them are trained on")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--face_count", type=int, default=100_000)
    ap.add_argument("--line_count", type=int, default=100_000)
    ap.add_argument("--prim_size", type=float, default=0.02,
                    help="initial triangle radius and segment length, as a "
                         "fraction of the scene radius")
    ap.add_argument("--opacity_init", type=float, default=0.1,
                    help="opacity every primitive starts at")
    ap.add_argument("--lr_pos", type=float, default=2.5e-5,
                    help="multiplied by the scene's spatial scale")
    ap.add_argument("--lr_color", type=float, default=2.5e-3)
    ap.add_argument("--lr_opacity", type=float, default=5e-2)
    ap.add_argument("--resample_every", type=int, default=200)
    ap.add_argument("--resample_start", type=int, default=200)
    ap.add_argument("--resample_end", type=float, default=0.9,
                    help="stop resampling after this fraction of the run")
    ap.add_argument("--max_growth", type=float, default=0.05,
                    help="most a single resample may add, as a fraction of the "
                         "primitives that survived pruning")
    ap.add_argument("--prune_threshold", type=float, default=0.05,
                    help="primitives below this opacity are deleted during training")
    ap.add_argument("--final_tau", type=float, default=0.5,
                    help="deterministic opacity threshold for the final render")

    ap.add_argument("--fovy", type=float, default=40.0,
                    help="vertical field of view in degrees; lower is flatter, "
                         "and the camera backs off to keep the framing")
    ap.add_argument("--preview_azimuth", type=float, default=0.0)
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

    eyes = camera.sphere_eyes(args.views)
    train_views = [(camera.view_proj(eyes[i]), eyes[i]) for i in range(args.views)]
    lr_pos = args.lr_pos * common.spatial_lr_scale(eyes)
    preview_eye = camera.orbit_eye(args.preview_azimuth)
    preview_view = (camera.view_proj(preview_eye), preview_eye)

    # ------------------------------------------------------------------
    # Initialisation - one trainable set per class
    # ------------------------------------------------------------------
    bbox_min = ref_verts.min(dim=0).values
    bbox_max = ref_verts.max(dim=0).values
    extent = camera.radius * args.prim_size

    def build(builder, count: int) -> fitting.Primitives:
        verts, prims = builder(count, extent, bbox_min, bbox_max)
        return fitting.Primitives(
            prims.to(device), verts.to(device), torch.rand_like(verts).to(device),
            opacity_init=args.opacity_init, lr_pos=lr_pos,
            lr_color=args.lr_color, lr_opacity=args.lr_opacity)

    faces = build(common.make_random_triangles, args.face_count)
    lines = build(common.make_random_lines, args.line_count)

    # One rasterize call takes a single vertex array, so the two sets are
    # concatenated - faces first, then lines - and the line index buffer is
    # shifted past the face vertices.  It only has to be rebuilt when a
    # resample changes how many face vertices there are.
    def shifted_line_prims() -> torch.Tensor:
        return ((lines.prims.to(torch.int64) + faces.num_verts)
                .to(torch.int32).view(torch.uint32).contiguous())

    line_prims = shifted_line_prims()

    print(f"[faces+lines] {faces.count} triangles + {lines.count} segments at "
          f"opacity {args.opacity_init}, {args.views} views at {width}x{height}, "
          f"msaa={args.msaa}, device={device}")

    def render(view, *, seed: int, tau: float = -1.0):
        """Render both classes together; returns (face opacity, line opacity, RGB).

        ``tau`` defaults to -1, i.e. stochastic opacity masking, which is what
        training uses.  The final render passes a fixed threshold instead, so
        the image is deterministic.
        """
        viewproj, eye = view
        face_opacity, line_opacity = faces.opacity(), lines.opacity()

        verts = torch.cat([faces.verts, lines.verts])
        colors = torch.cat([faces.colors, lines.colors])
        attrs = fuzzydr.pack_attrs(verts, colors, verts.new_zeros(verts.shape[0]))

        rgba = fuzzydr.rasterize(
            attrs, viewproj=viewproj, campos=eye,
            faces=faces.prims, face_opacity=face_opacity,
            lines=line_prims, line_opacity=line_opacity,
            width=width * scale, height=height * scale, tau=tau, seed=seed,
            bresen_lines=True, white_bg=False)
        rgb = (fuzzydr.msaa_downsample_rgba(rgba) if args.msaa
               else rgba[..., :3].contiguous())
        return face_opacity, line_opacity, rgb

    def opacity_aux_loss(face_opacity, line_opacity, error) -> torch.Tensor:
        # Both classes are handed to one aux loss, which routes the gradient to
        # each through the primitive-ID buffer of the render that just ran.
        if args.msaa:
            error = fuzzydr.upsample2x2_scalar(error)
        return fuzzydr.opacity_mask_aux_loss(
            face_opacity=face_opacity, line_opacity=line_opacity, error=error)

    fuzzydr.init()
    try:
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
                append_frame(render(preview_view, seed=0)[2])

            view_order = torch.randperm(args.views).tolist()
            pbar = tqdm(range(1, args.iters + 1), desc="Optimizing", ncols=110)
            for it in pbar:
                if (it - 1) % args.views == 0:
                    view_order = torch.randperm(args.views).tolist()
                view_idx = view_order[(it - 1) % args.views]

                faces.zero_grad()
                lines.zero_grad()
                face_opacity, line_opacity, img = render(train_views[view_idx], seed=it)
                residual = (img - references[view_idx]).abs()
                loss = residual.mean()
                aux = opacity_aux_loss(
                    face_opacity, line_opacity,
                    residual.mean(dim=-1).detach().contiguous())
                (loss + aux).backward()
                faces.step()
                lines.step()

                losses.append(float(loss.detach()))

                if (args.resample_start <= it <= resample_end
                        and it % args.resample_every == 0):
                    # Each class holds its own budget.
                    face_summary = faces.resample(
                        prune_threshold=args.prune_threshold,
                        target_count=args.face_count,
                        max_growth=args.max_growth)
                    line_summary = lines.resample(
                        prune_threshold=args.prune_threshold,
                        target_count=args.line_count,
                        max_growth=args.max_growth)
                    if face_summary or line_summary:
                        # Either resample can move the face vertex count, so
                        # the shifted line indices have to be rebuilt.
                        line_prims = shifted_line_prims()
                        pbar.write(f"[resample] it={it}  faces: {face_summary or 'unchanged'}")
                        pbar.write(f"[resample] it={it}  lines: {line_summary or 'unchanged'}")

                pbar.set_postfix(loss=f"{losses[-1]:.3e}",
                                 f=faces.count, l=lines.count)

                if it in video_iters:
                    with torch.no_grad():
                        append_frame(render(preview_view, seed=it)[2])
        finally:
            writer.close()
        print(f"Wrote {out_mp4}")

        # ------------------------------------------------------------------
        # Results
        # ------------------------------------------------------------------
        with torch.no_grad():
            common.save_png(out_dir / "reference.png", preview_reference)
            common.save_png(out_dir / "final.png",
                            render(preview_view, seed=0, tau=args.final_tau)[2])
        common.save_loss_plot(out_dir / "loss.png", losses)
        common.save_loss_curve(out_dir / "loss.txt", losses)

        for name, kind, primitives in (("faces", "faces", faces), ("lines", "lines", lines)):
            common.save_checkpoint(
                out_dir / f"{name}.npz",
                primitive=kind,
                verts=primitives.verts.detach(),
                colors=primitives.colors.detach(),
                prims=primitives.prims,
                opacity_logit=primitives.opacity_logit.detach(),
                meta={"reference_mesh": str(args.mesh), "iters": args.iters,
                      "opacity_init": args.opacity_init,
                      "prune_threshold": args.prune_threshold, "msaa": args.msaa},
            )

        kept_f = int((faces.opacity().detach() >= args.final_tau).sum())
        kept_l = int((lines.opacity().detach() >= args.final_tau).sum())
        print(f"\nLoss: {losses[0]:.4e} -> {losses[-1]:.4e}")
        print(f"faces: {args.face_count} -> {faces.count} "
              f"({kept_f} at opacity >= {args.final_tau})   "
              f"lines: {args.line_count} -> {lines.count} ({kept_l})")
    finally:
        fuzzydr.shutdown()


if __name__ == "__main__":
    main()
