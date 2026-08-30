# Examples

Example scripts demonstrating `fuzzydr`, ranging from forward rendering and stochastic opacity masking to differentiable primitive reconstruction.

## Setup

Install `fuzzydr` as described in the [project README](../README.md), then install the additional dependencies used by the examples for image processing, video output, and progress bars:

```bash
pip3 install -r examples/requirements.txt
```

The numbered examples can be run from the repository root:

```bash
python3 examples/01_render.py
```

Each script writes its output to its own subdirectory under `./out/`. Use `--out_dir` to change the output directory. All scripts also accept `--device cpu|cuda`; `fuzzydr` renders with Vulkan in either case, and this option only controls where the PyTorch tensors reside.

## The Examples

| Script | Content |
| --- | --- |
| [`01_render.py`](01_render.py) | Forward rendering, including intermediate rasterization buffers such as per-pixel primitive ID and depth. |
| [`02_render_video.py`](02_render_video.py) | Forward rendering of a video, with primitive topology cached across frames. |
| [`03_fit_triangle_line.py`](03_fit_triangle_line.py) | Minimal inverse rendering example at a 32x32 resolution using 1-pixel Bresenham line primitives. The output is magnified for pixel-level inspection. |
| [`04_fit_triangle_quad.py`](04_fit_triangle_quad.py) | Minimal inverse rendering example: fitting one triangle and one quad line primitive to a reference image. |
| [`05_fit_mesh.py`](05_fit_mesh.py) | Deforming a sphere into the Stanford bunny from reference images. Traditional differentiable rendering without stochastic opacity masking. **CUDA only** - see below. |
| [`06_stochastic_opacity.py`](06_stochastic_opacity.py) | Forward process of stochastic opacity masking without gradients, intended for gaining intuition about the method. |
| [`07_fit_faces.py`](07_fit_faces.py) | Demonstrating stochastic opacity gradients by fitting unconnected triangles from random initialization. |
| [`08_fit_lines.py`](08_fit_lines.py) | The same optimization using 1-pixel Bresenham line primitives. See [our paper](https://kenji-tojo.github.io/sa26-line-primitives/) for more details. |
| [`09_fit_faces_and_lines.py`](09_fit_faces_and_lines.py) | Fitting a mixture of triangle and line primitives, which `fuzzydr` supports natively. |
| [`10_fit_points.py`](10_fit_points.py) | The same optimization using 1-pixel point primitives. This is a beta feature originally used for an ablation study in our paper. |
| [`11_fit_sh_coeffs.py`](11_fit_sh_coeffs.py) | Fitting view-dependent color with spherical harmonics, natively supported in `fuzzydr` using a compute shader. |
| [`view_rgb.py`](view_rgb.py) | Opens a `faces.npz` or `lines.npz` produced by examples `07`-`09` in the interactive viewer. Requires the optional `fuzzydr_viewer` package. |

`05_fit_mesh.py` requires `--device cuda` because `largesteps.geometry` constructs its Laplacian with `device='cuda'` hardcoded (0.2.x), so its reparameterization cannot be constructed on CPU. See the [original paper](https://rgl.epfl.ch/publications/Nicolet2021Large) for details on the reparameterization.

## Anti-aliasing

When anti-aliasing is enabled, the examples rasterize at twice the target resolution and resolve to the target resolution using `fuzzydr.msaa_downsample_rgba`.

Opacity gradients require error maps at the rasterization resolution. We therefore use `fuzzydr.upsample2x2_scalar` to upsample the error map from the target resolution to the rasterization resolution. With `--no-msaa`, rasterization is performed directly at the target resolution, so this step is not necessary.

See [our paper](https://kenji-tojo.github.io/sa26-line-primitives/) for more details on this treatment.
