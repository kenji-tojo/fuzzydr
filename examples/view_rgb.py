# examples/view_rgb.py
#
# Open one or more fitted primitive sets in the interactive viewer.
#
# A checkpoint holds a single primitive class, so 08 writes its triangles and
# its line segments separately.  Pass both to see what it actually fitted:
# they are merged into one vertex array here, which is what the viewer wants.
#
# The viewer is the separate, optional fuzzydr_viewer package:
#
#   pip3 install -v ./viewer/
#
# Usage:
#   python examples/view_rgb.py out/07_fit_faces/faces.npz
#   python examples/view_rgb.py out/08_fit_lines/lines.npz
#   python examples/view_rgb.py out/09_fit_faces_and_lines/{faces,lines}.npz
#
# The viewer's world up axis is +Z (fuzzydr_viewer 0.1.0 builds it as
# {0, 0, +-1}, and its "Flip up" only chooses the sign).  The example assets
# are Y-up, so --up y, the default, rotates the geometry a quarter turn about
# X on the way in; pass --up z for a scene that is already Z-up.

from __future__ import annotations

import argparse
import sys

import numpy as np

import common


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoints", type=str, nargs="+",
                    help="one or more .npz files written by 06-09")
    ap.add_argument("--opacity", type=float, default=0.5,
                    help="hide primitives whose opacity is below this; 0.5 is the "
                         "same deterministic threshold the examples' final.png uses")
    ap.add_argument("--up", type=str, default="y", choices=["y", "z"],
                    help="up axis of the checkpoint; the viewer's own is +Z")
    ap.add_argument("--flip_up", action="store_true",
                    help="point the camera's up reference at -Z instead of +Z")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    try:
        import fuzzydr_viewer
    except ImportError:
        sys.exit("fuzzydr_viewer is not installed; run: pip3 install -v ./viewer/")

    # Merge the checkpoints into one vertex array, shifting each set's indices
    # past the vertices already collected.
    verts, colors, radius = [], [], []
    prims: dict[str, list[np.ndarray]] = {"faces": [], "lines": []}
    num_verts = 0

    for path in args.checkpoints:
        ckpt = common.load_checkpoint(path)
        primitive = ckpt["meta"]["primitive"]
        if primitive == "points":
            sys.exit(f"{path}: the viewer draws faces and lines only, not points")

        # Opacities are stored as logits so any threshold can be applied here.
        keep = 1.0 / (1.0 + np.exp(-ckpt["opacity_logit"])) >= args.opacity
        kept = ckpt["prims"][keep].astype(np.uint32) + num_verts
        print(f"{path}: {kept.shape[0]} of {ckpt['prims'].shape[0]} {primitive} "
              f"at opacity >= {args.opacity}")

        prims[primitive].append(kept)
        verts.append(ckpt["verts"])
        colors.append(ckpt["colors"])
        radius.append(ckpt["radius"])
        num_verts += ckpt["verts"].shape[0]

    all_verts = np.concatenate(verts)
    if args.up == "y":
        # Quarter turn about X, taking +Y to +Z: (x, y, z) -> (x, -z, y).
        all_verts = np.stack(
            [all_verts[:, 0], -all_verts[:, 2], all_verts[:, 1]], axis=1)

    merged = {k: (np.concatenate(v) if v else None) for k, v in prims.items()}
    if all(m is None or m.shape[0] == 0 for m in merged.values()):
        sys.exit(f"nothing left above opacity {args.opacity}")

    all_radius = np.concatenate(radius)
    has_lines = merged["lines"] is not None and merged["lines"].shape[0] > 0

    fuzzydr_viewer.launch(
        np.ascontiguousarray(all_verts, dtype=np.float32),
        colors=np.concatenate(colors),
        # A non-zero radius means the lines were trained as quads, so hand it
        # to the viewer and it opens in the matching line style.
        radius=all_radius if np.any(all_radius > 0.0) else None,
        faces=merged["faces"],
        lines=merged["lines"],
        # launch() rebakes the segment list into strips by default, but that
        # branch rejects its own default when there are no lines to rebake
        # (fuzzydr_viewer 0.1.0).  With no lines the topology is moot anyway.
        line_topology="strip" if has_lines else "list",
        flip_up=args.flip_up,
        width=args.width,
        height=args.height,
        background=(0.0, 0.0, 0.0),   # the examples train against black
    )


if __name__ == "__main__":
    main()
