# examples/fitting.py
#
# Shared bookkeeping for the fitting examples (05-08).  The fuzzydr calls stay
# in the example scripts; what lives here is the primitive housekeeping that is
# the same whatever is being fitted:
#
#   * the trainable state - unshared vertices, an index buffer, one opacity
#     logit per primitive - and its optimizer;
#   * resampling: prune primitives that have gone transparent, then refill the
#     budget - the starting count is also the target, so however many pruning
#     removed, that many are put back by splitting the longest survivors
#     (triangles and lines) or duplicating random ones (points).
#
# Opacity is trained as a logit so it stays in (0, 1) with no clamping, and is
# stored that way in checkpoints so a reader can re-threshold without
# retraining.

from __future__ import annotations

import math

import torch

from fuzzydr.optimize import VectorAdam


def opacity_to_logit(opacity: float) -> float:
    """Inverse of ``sigmoid``, for turning an initial opacity into a logit."""
    if not 0.0 < opacity < 1.0:
        raise ValueError(f"opacity must be in (0, 1); got {opacity}")
    return math.log(opacity / (1.0 - opacity))


class Primitives:
    """A trainable set of primitives of one type.

    ``prims`` is ``uint32 [M, A]`` where the arity ``A`` is 3 for triangles, 2
    for lines and 1 for points.  Vertices are unshared: every primitive owns
    its own rows of ``verts`` and ``colors``, so one can be removed or split
    without disturbing the others.

    Positions, colours and opacity logits are all trained.  Positions use
    fuzzydr's VectorAdam, whose second moment is shared across a vertex's three
    components so the step does not depend on how the mesh is oriented; colour
    and opacity are per-channel scalars and use plain Adam.  Replacing any of
    them during resampling rebuilds the optimizers, since the moments no longer
    line up with the new indexing.
    """

    def __init__(
        self,
        prims: torch.Tensor,      # uint32  [M, A]
        verts: torch.Tensor,      # float32 [N, 3]
        colors: torch.Tensor,     # float32 [N, 3]
        *,
        opacity_init: float,
        lr_pos: float,
        lr_color: float,
        lr_opacity: float,
    ) -> None:
        self.prims = prims
        self.verts = torch.nn.Parameter(verts)
        self.colors = torch.nn.Parameter(colors)
        self.opacity_logit = torch.nn.Parameter(torch.full(
            (prims.shape[0],), opacity_to_logit(opacity_init),
            dtype=torch.float32, device=verts.device))
        self._lrs = (lr_pos, lr_color, lr_opacity)
        self._make_optimizer()

    # -- state ----------------------------------------------------------------

    @property
    def count(self) -> int:
        return int(self.prims.shape[0])

    @property
    def num_verts(self) -> int:
        return int(self.verts.shape[0])

    @property
    def arity(self) -> int:
        return int(self.prims.shape[1])

    def opacity(self) -> torch.Tensor:
        """Per-primitive opacity in (0, 1), with grad to the logits."""
        return torch.sigmoid(self.opacity_logit)

    def _make_optimizer(self) -> None:
        lr_pos, lr_color, lr_opacity = self._lrs
        self.opt_verts = VectorAdam([self.verts], lr=lr_pos)
        self.opt_attrs = torch.optim.Adam([
            {"params": [self.colors], "lr": lr_color},
            {"params": [self.opacity_logit], "lr": lr_opacity},
        ])

    def zero_grad(self) -> None:
        # VectorAdam has no zero_grad of its own; it inherits Optimizer's.
        self.opt_verts.zero_grad(set_to_none=True)
        self.opt_attrs.zero_grad(set_to_none=True)

    def step(self) -> None:
        self.opt_verts.step()
        self.opt_attrs.step()
        with torch.no_grad():
            self.colors.clamp_(0.0, 1.0)
            # Keep the logits where sigmoid still has usable gradient, so a
            # nearly transparent primitive can recover before the next prune.
            self.opacity_logit.clamp_(-12.0, 12.0)

    # -- resampling -----------------------------------------------------------

    def resample(self, *, prune_threshold: float, target_count: int,
                 max_growth: float = 0.05) -> str | None:
        """Prune transparent primitives, then refill the budget.

        Splitting a triangle or a segment replaces it with two halves, and
        duplicating a point adds one, so each operation is a net gain of one
        primitive.  The refill is whatever is missing from ``target_count``,
        capped at ``max_growth`` of the surviving count so no single resample
        can add more than a few percent - after a heavy prune the budget is
        recovered over several resamples rather than in one jump.  Pruning is
        therefore a redistribution: primitives leave where they are not
        earning their place and reappear where the geometry is coarsest.

        Returns a one-line summary, or ``None`` if nothing changed.
        """
        before = self.count
        # torch cannot index with uint32, so the surgery below works in int64
        # and the result is cast back once at the end.
        prims = self.prims.to(torch.int64)
        logit = self.opacity_logit.detach()
        verts = self.verts.detach()
        colors = self.colors.detach()

        keep = torch.sigmoid(logit) >= prune_threshold
        pruned = int((~keep).sum())
        if pruned:
            prims, logit = prims[keep], logit[keep]
            prims, verts, colors = _compact(prims, verts, colors)

        added = 0
        # Only ever refill, never trim: a budget larger than the current count
        # is a deficit to make up, a smaller one is left alone.
        survived = prims.shape[0]
        deficit = min(target_count - survived, int(max_growth * survived))
        if deficit > 0 and survived > 0:
            grow = _duplicate if self.arity == 1 else _split_longest
            prims, logit, verts, colors, added = grow(
                prims, logit, verts, colors, count=deficit)

        if not pruned and not added:
            return None

        self.prims = prims.to(torch.int32).view(torch.uint32).contiguous()
        self.verts = torch.nn.Parameter(verts.contiguous())
        self.colors = torch.nn.Parameter(colors.contiguous())
        self.opacity_logit = torch.nn.Parameter(logit.contiguous())
        self._make_optimizer()

        return (f"{before} -> {self.count} primitives "
                f"({pruned} pruned, {added} added), {self.num_verts} vertices")


# ---------------------------------------------------------------------------
# Index-buffer surgery.  All of these take and return int64 index buffers; the
# caller casts back to uint32 once at the end.
# ---------------------------------------------------------------------------

def _compact(
    prims: torch.Tensor,    # int64 [M, A]
    verts: torch.Tensor,
    colors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop vertices no primitive references, and renumber."""
    used = torch.unique(prims.reshape(-1))
    if used.numel() == verts.shape[0]:
        return prims, verts, colors
    remap = torch.full((verts.shape[0],), -1, dtype=torch.int64, device=verts.device)
    remap[used] = torch.arange(used.numel(), dtype=torch.int64, device=verts.device)
    return remap[prims], verts[used], colors[used]


def _split_longest(
    prims: torch.Tensor,    # int64 [M, A]   A = 3 (triangles) or 2 (lines)
    logit: torch.Tensor,    # float32 [M]
    verts: torch.Tensor,
    colors: torch.Tensor,
    *,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Replace the ``count`` longest primitives by two halves each.

    A triangle is cut from the midpoint of its longest edge to the opposite
    corner; a line segment is cut at its midpoint.  Both halves inherit the
    parent's opacity and get their own vertices, so they can drift apart.
    """
    total = prims.shape[0]
    # There is nothing to split beyond one child per existing primitive.
    chosen = min(count, total)
    if chosen <= 0:
        return prims, logit, verts, colors, 0

    arity = prims.shape[1]
    corner_pos = verts[prims]                                   # [M, A, 3]
    # Edge k runs from corner k to corner k+1.  A triangle has three such
    # edges; a segment has one (rolling a pair would just repeat it).
    num_edges = 3 if arity == 3 else 1
    edge_len = (corner_pos.roll(-1, dims=1) - corner_pos).norm(dim=-1)[:, :num_edges]
    longest = edge_len.argmax(dim=1)                            # [M]
    selected = torch.topk(edge_len.gather(1, longest[:, None]).squeeze(1), chosen).indices

    # Rotate each selected primitive so corner 0 and corner 1 span the longest
    # edge; for a triangle corner 2 is then the opposite one.
    order = (longest[selected][:, None]
             + torch.arange(arity, device=prims.device)[None, :]) % arity
    rotated = prims[selected].gather(1, order)                  # [chosen, A]

    a, b = rotated[:, 0], rotated[:, 1]
    mid_pos = 0.5 * (verts[a] + verts[b])
    mid_col = 0.5 * (colors[a] + colors[b])

    if arity == 3:
        c = rotated[:, 2]
        halves_pos = [torch.stack([verts[a], mid_pos, verts[c]], dim=1),
                      torch.stack([mid_pos, verts[b], verts[c]], dim=1)]
        halves_col = [torch.stack([colors[a], mid_col, colors[c]], dim=1),
                      torch.stack([mid_col, colors[b], colors[c]], dim=1)]
    else:
        halves_pos = [torch.stack([verts[a], mid_pos], dim=1),
                      torch.stack([mid_pos, verts[b]], dim=1)]
        halves_col = [torch.stack([colors[a], mid_col], dim=1),
                      torch.stack([mid_col, colors[b]], dim=1)]

    new_pos = torch.cat(halves_pos, dim=0).reshape(-1, 3)
    new_col = torch.cat(halves_col, dim=0).reshape(-1, 3)
    new_prims = (torch.arange(new_pos.shape[0], dtype=torch.int64, device=prims.device)
                 .reshape(-1, arity) + verts.shape[0])
    new_logit = logit[selected].repeat(2)

    survivors = torch.ones(total, dtype=torch.bool, device=prims.device)
    survivors[selected] = False

    prims = torch.cat([prims[survivors], new_prims], dim=0)
    logit = torch.cat([logit[survivors], new_logit], dim=0)
    verts = torch.cat([verts, new_pos], dim=0)
    colors = torch.cat([colors, new_col], dim=0)

    # Each parent became two halves, so the count grows by `chosen`; the
    # parents' own vertices are now unreferenced.
    prims, verts, colors = _compact(prims, verts, colors)
    return prims, logit, verts, colors, chosen


def _duplicate(
    prims: torch.Tensor,    # int64 [P, 1]
    logit: torch.Tensor,
    verts: torch.Tensor,
    colors: torch.Tensor,
    *,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Duplicate ``count`` randomly chosen points.

    A clone starts on top of its original and carries the same opacity, but
    draws its own stochastic threshold, so the pair covers the pixel more often
    than one point did and the two receive gradients on different frames.
    """
    total = prims.shape[0]
    chosen = min(count, total)
    if chosen <= 0:
        return prims, logit, verts, colors, 0

    selected = torch.randperm(total, device=prims.device)[:chosen]
    source = prims[selected].reshape(-1)

    new_prims = (torch.arange(chosen, dtype=torch.int64, device=prims.device)
                 .reshape(-1, 1) + verts.shape[0])
    prims = torch.cat([prims, new_prims], dim=0)
    logit = torch.cat([logit, logit[selected]], dim=0)
    verts = torch.cat([verts, verts[source]], dim=0)
    colors = torch.cat([colors, colors[source]], dim=0)
    return prims, logit, verts, colors, chosen
