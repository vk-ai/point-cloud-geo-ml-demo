# point-cloud-geo-ml-demo

> **OSS / learning demo only.** This is a personal open-source teaching project by [vk-ai](https://github.com/vk-ai). It is **not** employer production software, is **not** affiliated with any employer (including Lowe's or any other company), and must **not** be described as production reality-capture, digital-twin, or warehouse-mapping infrastructure.

CPU-friendly **point-cloud / geometric ML fundamentals** on **fully synthetic** labeled shapes (planes, spheres, cubes). No large datasets, no GPU, no Open3D / PyTorch3D — **numpy + scikit-learn**.

Inspired by **NavVis-style** stretch learning goals around 3D spatial understanding: sampling, local geometry, and simple classifiers — as public fundamentals, not a product clone.

## Why

Geometric deep learning papers and reality-capture stacks are great, but for learning and CI you often want:

- **No multi-GB ScanNet / KITTI downloads**
- **Seconds on CPU**, not minutes on GPU
- Explicit **FPS / normals / voxels** you can read end-to-end

This repo is that slice.

## What’s inside

| Piece | Role |
|---|---|
| `src/point_cloud_geo/data.py` | Synthetic plane / sphere / cube clouds (labeled) |
| `src/point_cloud_geo/sampling.py` | Farthest Point Sampling (FPS) + random baseline + coverage proxies |
| `src/point_cloud_geo/voxelize.py` | Coarse voxel features (`occupancy` / `count` / `max_count_bin`) |
| `src/point_cloud_geo/features.py` | PCA normals on kNN + linearity/planarity/sphericity |
| `src/point_cloud_geo/normals_radius.py` | Radius-neighborhood PCA normals (BallTree; Open3D-free) |
| `src/point_cloud_geo/pointnet_lite.py` | PointNet-lite: shared MLP + max-pool (± normals ablation) |
| `src/point_cloud_geo/registration.py` | Numpy ICP: point-to-point (Kabsch) + point-to-plane (radius normals), `fitness` / `inlier_rmse` / `converged` (round 4) |
| `src/point_cloud_geo/segmentation.py` | Seeded RANSAC plane segmentation (`segment_plane` / `segment_planes`), bit-identical per seed (round 5) |
| `src/point_cloud_geo/model.py` | Tiny sklearn MLP or logistic on pooled features |
| `src/point_cloud_geo/train.py` | Train / test split + accuracy |
| `src/point_cloud_geo/eval.py` | JSON-friendly eval summary |
| `configs/default.yaml` | Cloud size, FPS, kNN, classifier knobs |
| `evals/runner.py` | CI-friendly eval CLI + soft accuracy gate |
| `tests/` | Unit + train smoke (pytest, seconds on CPU) |
| `ci/github-actions.yml` | GitHub Actions workflow (copy to `.github/workflows/ci.yml` to enable CI) |

## Quickstart

```bash
git clone https://github.com/vk-ai/point-cloud-geo-ml-demo.git
cd point-cloud-geo-ml-demo
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python examples/quickstart.py
python evals/runner.py
python -m point_cloud_geo
```

Example output:

```text
Point-cloud geometric ML demo (OSS / learning only)
  classes     ['plane', 'sphere', 'cube']
  features    …  classifier=mlp
  FPS / knn   64 pts, k=8
  train_acc   0.9x  (n=…)
  test_acc    0.8x  (n=…)
```

## Geometric ML concepts (brief)

**Point clouds** are unordered sets of 3D coordinates. Classic geometric ML builds **permutation-aware** or **pooled** representations:

1. **Farthest Point Sampling (FPS)** — greedily keep points that cover the surface well; a standard downsampling step before PointNet-style models.
2. **Local PCA normals** — for each point, PCA on its k-nearest neighbors; the smallest eigenvector is a surface normal. Eigenvalue ratios give **linearity / planarity / sphericity** descriptors (useful for walls vs pipes vs clutter in mapping-style scenes — here, toy shapes).
3. **Voxelization** — bin space into a coarse grid; occupancy histograms are a simple global descriptor.
4. **Tiny classifier** — concatenate global stats of coords / normals / shape ratios + voxel occupancy, then train a small **MLP** (or logistic regression). This is the “handcrafted geometric features + shallow net” cousin of PointNet’s learned per-point MLP + max-pool idea.

All of the above run on CPU in seconds with a few dozen synthetic clouds.


## Radius normals + PointNet-lite (round 3)

**Radius PCA normals:** for each point, neighbors within radius `r` → PCA → smallest eigenvector. Config: `normals.radius`, `normals.min_nn`. *r* must match your cloud’s metric spacing (too-small → noisy; too-large → oversmoothed) — same lesson Open3D users hit  
([PCL normal estimation](https://pcl.readthedocs.io/projects/tutorials/en/master/normal_estimation.html)).

**PointNet-lite:** shared per-point MLP → symmetric max-pool → class MLP on FPS-downsampled clouds (XYZ or XYZ+normals). Teachable stub — **not** PointNet++/LitePT  
([PointNetLite 2025](https://doi.org/10.1117/12.3063179) as motivation only).

```python
from point_cloud_geo import estimate_normals_radius, compare_normals_ablation, make_dataset
clouds, labels = make_dataset(n_per_class=20, n_points=64, seed=0)
n, e, c = estimate_normals_radius(clouds[0], radius=0.35)
print(compare_normals_ablation(clouds, labels, n_points=32, epochs=25))
```

## ICP registration (round 4)

`icp(source, target, method=...)` estimates the rigid 4×4 `T` with `target ≈ R·source + t`. It is written in numpy, with correspondences from sklearn `NearestNeighbors` (already a dependency).

- **`point_to_point`**: a closed-form Kabsch/SVD update on the inlier pairs.
- **`point_to_plane`** (optional): a linearized 6×6 least-squares step on `Σ((R p + t − q)·n)²`. It uses **target normals from the shipped radius-PCA estimator** (`normals_radius`, or pass `target_normals=`), and ω is mapped back to an exact rotation.

```python
from point_cloud_geo.registration import icp, make_registration_pair, transform_errors
src, tgt, T_true = make_registration_pair("cube", angle_deg=20.0, noise=0.01, seed=7)
res = icp(src, tgt, method="point_to_plane", max_corr_dist=0.25, max_iter=50)
res.fitness, res.inlier_rmse, res.converged, res.n_iter, res.stop_reason
transform_errors(res.T, T_true)   # {'rotation_error_deg': ..., 'translation_error': ...}
```

**Output definitions.** These are explicit because Open3D users found the upstream docs confusing: the fitness docs were wrong ([Open3D#7503](https://github.com/isl-org/Open3D/issues/7503)), `converged_` was always false ([#7296](https://github.com/isl-org/Open3D/issues/7296)), and the stop criteria were ignored ([#7367](https://github.com/isl-org/Open3D/issues/7367)).

| Output | Definition |
|---|---|
| `fitness` | `n_inliers / len(source)`: the fraction of **source** points whose nearest target point (after `T`) lies within `max_corr_dist`. Range 0…1. |
| `inlier_rmse` | `sqrt(mean(d²))` over **inlier** pairs only, with `d` the Euclidean point-to-point distance (for both methods). It is `nan` when there are no inliers. |
| `converged` | True only if a stopping test fired **before** `max_iter`: either a tiny update (`tol_rotation_deg` and `tol_translation`) or |Δfitness| < `tol_fitness` **and** |Δrmse| < `tol_rmse`. `stop_reason` says which. Hitting `max_iter` means `False`. |

Both metrics are recomputed at the **returned** `T` (`evaluate_registration` uses the same formula). **Neither one proves the transform is correct.**

`python evals/runner.py` now prints an `icp_recovery` block (cube, seed 7):

```text
small  20.0°  point_to_point  rot_err=  0.051°  t_err=0.0014  fitness=1.000  inlier_rmse=0.0169  iters=15  converged=True
small  20.0°  point_to_plane  rot_err=  0.235°  t_err=0.0031  fitness=1.000  inlier_rmse=0.0176  iters= 6  converged=True
large  90.0°  point_to_point  rot_err= 90.458°  t_err=0.0465  fitness=1.000  inlier_rmse=0.0826  iters=32  converged=True
large  90.0°  point_to_plane  rot_err= 89.636°  t_err=0.0453  fitness=1.000  inlier_rmse=0.0841  iters=50  converged=False
```

Lessons the tests pin down:

1. With small angles, both methods recover the known transform (< 1°). Point-to-plane needs far fewer iterations.
2. With large angles, ICP is **local**. At 90°, the cube's symmetry gives a wrong pose with fitness 1.0. Only the 5× higher `inlier_rmse` hints at the problem.
3. A **sphere**'s rotation is unobservable: fitness is 1.0 and the rotation is still wrong.
4. On a flat **plane**, point-to-plane cannot see in-plane sliding. Near-degenerate directions are truncated (`plane_rcond`) so it does not diverge, but it slides.
5. Junk source points lower `fitness` without inflating `inlier_rmse`.
6. Results are deterministic for a fixed seed.

**Honesty:** this is a teaching ICP. It has no global registration (no FPFH features or RANSAC correspondence matching; the round-5 RANSAC below segments planes and is not used to register), no robust kernels, no multi-scale schedule, and is not Open3D or PCL. The synthetic rigid transforms are not real LiDAR scans. Background: [LearnOpenCV ICP](https://learnopencv.com/iterative-closest-point-icp-explained/) · [SO: point-to-plane ICP](https://stackoverflow.com/questions/69490955/how-to-implement-icp-with-point-to-plane-distance) · [Open3D#6110 point-to-plane diverging](https://github.com/isl-org/Open3D/issues/6110).

## Seeded RANSAC plane segmentation (round 5)

`segment_plane(points, distance_threshold, ransac_n=3, num_iterations, seed)` returns the plane model `(a, b, c, d)` (`a·x + b·y + c·z + d = 0`, unit normal) plus the inlier indices. The names and shape follow Open3D's `segment_plane`, and the result unpacks the same way. It is **bit-identical for a given seed**. That property was the long-running upstream pain point: seeds stopped working in 0.16 ([Open3D#5647](https://github.com/isl-org/Open3D/issues/5647)), it took [PR #6308](https://github.com/isl-org/Open3D/pull/6308) and [PR #6580](https://github.com/isl-org/Open3D/pull/6580) to make it deterministic again, and results still looked random in 2025 ([Open3D#7270](https://github.com/isl-org/Open3D/issues/7270)).

```python
from point_cloud_geo.segmentation import make_plane_scene, segment_plane, segment_planes
pts, labels, truth = make_plane_scene([(0, 0, 1, -0.1)], n_per_plane=400, n_outliers=200, noise=0.005)
plane, inliers = segment_plane(pts, distance_threshold=0.02, num_iterations=1000, seed=0)
res = segment_plane(pts, 0.02, seed=0)       # PlaneResult: .plane_model .inliers .fitness .inlier_rmse .n_iter
planes = segment_planes(pts, max_planes=3, min_inliers=50, distance_threshold=0.02, seed=0)  # fit → remove → repeat
```

How determinism is kept:
- A private `np.random.default_rng(seed)` is used, never the global RNG.
- Minimal samples are drawn sequentially in a single thread.
- The best model is chosen by a total order: most inliers, then lowest inlier RMSE, then earliest iteration.
- Adaptive early stopping (`probability`, as in Open3D; 1.0 disables it) depends only on that sequence.
- The plane sign is canonical: the largest normal component is positive.
- A least-squares (SVD) refit on the inliers runs at the end, and it is only kept if it does not lose support.

The tests check the same seed in two processes with different `PYTHONHASHSEED`, and check that the global NumPy RNG is never touched.

`python evals/runner.py` adds a scene with a ground plane, a tilted wall, and 150 uniform outliers (seed 7):

```text
RANSAC planes (900 pts incl. 150 outliers, thresh=0.02, seed=7):
  plane 0  abcd=(-0.001,-0.003,+1.000,+0.004)  angle_err=0.181°  d_err=0.0044  inliers=511  precision=0.978  recall=1.000  iters=92
  plane 1  abcd=(+0.959,-0.001,+0.282,-0.767)  angle_err=0.335°  d_err=0.0004  inliers=249  precision=0.972  recall=0.968  iters=61
  same seed → identical: True  inliers by seed [511, 513, 512, 512, 513]
```

Different seeds may choose a slightly different inlier set (RANSAC is still random across seeds). The same seed never does. Outliers that happen to lie within the threshold of a plane count as inliers, which is why precision is not 1.0. Points near the floor–wall intersection go to whichever plane is extracted first. Config: `plane_segmentation` in `configs/default.yaml`.

**Honesty:** this is teaching RANSAC. It is O(iterations·N) in numpy, has no normal-consistency or connectivity checks, and is not Open3D or PCL. The scenes are synthetic, not LiDAR.

## Pipeline

```text
synthetic cloud (N×3)
    → FPS (M points)
    → kNN + PCA normals & eigenvalue ratios
    → global mean/std + voxel occupancy
    → StandardScaler + MLP / logistic
    → plane | sphere | cube
```

## Config

See `configs/default.yaml` for `n_per_class`, `n_points`, `fps_points`, `knn_k`, `voxel_size`, `voxel.reduction` (`occupancy` | `count` | `max_count_bin`), `sampling: fps | random`, `fps.start_index`, and `classifier: mlp | logistic`.

**Voxel reduction modes (teaching knob):** Open3D users often want richer than mean/occupancy reductions ([Open3D#6934](https://github.com/isl-org/Open3D/issues/6934)). This demo stays **numpy-only** (no Open3D): `occupancy` is binary density-normalized, `count` keeps per-bin mass, `max_count_bin` highlights the densest bin. Eval JSON includes `voxel_reduction`.

**FPS vs random (teaching ablation):** PointNet-family pipelines default to FPS for fixed-N tokens; learners need a cheap proof it beats uniform random on **coverage** (and when accuracy may not move). Set `sampling: fps | random` at the same `fps_points`; eval JSON includes a `sampling_compare` table with `test_acc`, `coverage_mean_nn_spacing`, and `coverage_bbox_fill_ratio`. `fps.start_index` (default `0`) mirrors Open3D’s start-index idea ([Open3D#7076](https://github.com/isl-org/Open3D/pull/7076)) — **numpy only**, not Open3D/PointNet/ScanNet numbers.

## CI

The workflow file lives at `ci/github-actions.yml` (a mirror). To enable GitHub Actions, copy it:

```bash
mkdir -p .github/workflows
cp ci/github-actions.yml .github/workflows/ci.yml
```

## License

MIT — see [LICENSE](LICENSE).

## Disclaimer

Personal **OSS / learning** project only. **Not** employer production. **Not** affiliated with Lowe's or any employer. Does not implement or claim any proprietary reality-capture / NavVis / digital-twin product.
