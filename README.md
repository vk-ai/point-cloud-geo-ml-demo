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
| `src/point_cloud_geo/sampling.py` | Farthest Point Sampling (FPS) |
| `src/point_cloud_geo/voxelize.py` | Coarse voxel occupancy features |
| `src/point_cloud_geo/features.py` | PCA normals on kNN + linearity/planarity/sphericity |
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

See `configs/default.yaml` for `n_per_class`, `n_points`, `fps_points`, `knn_k`, `voxel_size`, and `classifier: mlp | logistic`.

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
