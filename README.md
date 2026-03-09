# Fast Markov Random Field Optimisation for Topologically Noisy 3D Shape Matching

Official implementation of the 2026 CVPR paper "Fast Markov Random Field Optimisation for Topologically Noisy 3D Shape Matching" by Paul Roetzer, Johan Thunberg, Zorah Lähner, and Florian Bernard.


TLDR; Given a **source shape** (vertices `VX`, faces `FX`) and a **target shape** (vertices `VY`, faces `FY`), sm-gco finds a map from each source vertex to a target vertex by optimising a Markov Random Field energy over triangle-to-triangle assignments.

## Installation

```bash
pip install git+https://github.com/paul0noah/sm-gco-dev.git
```

**Requirements:** CMake >= 3.16, C++17 compiler (GCC 8+ or Clang). Dependencies (libigl, pybind11, robin-map) are fetched automatically.

## Usage

```python
import numpy as np
import igl
from sm_gco import GCOSM, COST_MODE, TriangleWiseOpts

# Load source and target shapes (numpy arrays)
vx, fx = igl.read_triangle_mesh("source.off")  # source shape
vy, fy = igl.read_triangle_mesh("target.off")  # target shape

# Per-vertex features (e.g. descriptors of dimension D)
feat_x = ...  # shape (|VX|, D) — features on source
feat_y = ...  # shape (|VY|, D) — features on target

# Compute per-vertex feature difference matrix (|VX| x |VY|)
feature_difference = np.zeros((len(vx), len(vy)))
for i in range(len(vx)):
    feature_difference[i, :] = np.linalg.norm(feat_y - feat_x[i, :], axis=1)

# Orient source faces outward
fx, _ = igl.orient_outward(vx, fx, np.ones_like(fx)[:, 0])

# Create solver: maps source -> target
# feature_difference is transposed to (|VY| x |VX|) for the constructor
smgco = GCOSM(vx, fx, vy, fy, feature_difference.T)

# Run triangle-wise matching (uses sensible defaults)
optime, point_map, tri_tri, raw_p2p, raw_tri_tri = smgco.triangle_wise()
print(f"Optimization took {optime:.2f} s")

# point_map: (N, 2) array — each row is [source_vertex, target_vertex]
# tri_tri:   (|FX|, 3) array — per source triangle, the 3 matched target vertex indices
```

## Configuration

`TriangleWiseOpts` controls the optimization. The defaults are tuned for typical shape matching:

```python
opts = TriangleWiseOpts()
# All defaults shown below — you only need to set what you want to change

# Algorithm: 0=swap, 1=alpha-expansion, 5=custom expansion with adaptive cycles (default)
opts.algorithm = 5

# Pairwise cost mode
opts.cost_mode = COST_MODE.MULTIPLE_LABLE_SPACE_GEODIST_MAX  # default

# Available cost modes:
#   MULTIPLE_LABLE_SPACE_GEODIST      - sum of geodesic distance between shared vertices
#   MULTIPLE_LABLE_SPACE_GEODIST_MAX  - max of geodesic distance between shared vertices (default)
#   MULTIPLE_LABLE_SPACE_L2DIST       - sum of L2 distance between shared vertices
#   MULTIPLE_LABLE_SPACE_L2DIST_MAX   - max of L2 distance between shared vertices
#   MULTIPLE_LABLE_SPACE_SO3          - rotation distance
#   MULTIPLE_LABLE_SPACE_SE3          - rigid transformation distance
#   MULTIPLE_LABLE_SPACE_GEODIST_SO3  - geodesic + rotation distance
#   SINGLE_LABLE_SPACE_L2             - single label space with L2 distance

# Energy weights
opts.smooth_weight = 1000.0              # pairwise term weight
opts.unary_weight = 1.0                  # data term weight
opts.smooth_scale_before_robust = 0.1    # scale pairwise cost before robust loss

# Robust cost: 0=none, 1=log, 2-7=various thresholds, 8=auto (default)
opts.robust_cost = 8

# Label space
opts.lable_space_cycle_size = 4          # max cycle length for triangle extraction on target
opts.lable_space_angle_thres = np.pi / 3 # angle threshold for orientation-preserving target triangles
opts.lable_space_degenerate = True       # allow matching source triangles to target edges/vertices

# Initialization: 0=none, 1=min-cost, 2=non-degenerate min-cost,
#   3=non-degenerate triangle neighbours, 4=triangle neighbours (default)
opts.set_initial_lables = 4

# Post-processing
opts.glue_solution = True                # glue per-source-triangle result into per-vertex map

# Optional settings
opts.label_order = 0                     # 0=default, 1=random, 2=degenerate-last, 3=min-cost, 4=alternating
opts.membrane_energy_weight = 0.0        # membrane energy weight (0 = disabled)
opts.bending_energy_weight = 0.0         # bending energy weight (0 = disabled)
opts.feature_weight = 1.0                # feature-based energy weight
opts.same_label_cost = 0.0              # > 0 makes the cost a metric instead of pseudometric
opts.bicolouring = False                 # enable bipartite graph colouring for optimization

# Iteration control
smgco.set_max_iter(-1)  # -1 = run until convergence (default)
```

## Return Values

`smgco.triangle_wise(opts)` returns a 5-tuple:

| Return          | Type           | Description                                                                    |
|-----------------|----------------|--------------------------------------------------------------------------------|
| `optime`        | `float`        | Optimization time in seconds                                                   |
| `point_map`     | `(N, 2) int`   | Per-vertex correspondences `[source_idx, target_idx]` (glued if `glue_solution=True`) |
| `tri_tri`       | `(numFX, 3) int` | Per source triangle: the 3 matched target vertex indices                     |
| `raw_p2p`       | `(M, 2) int`   | Raw per-vertex correspondences before gluing                                   |
| `raw_tri_tri`   | `(numFX, 3) int` | Raw triangle matching before gluing                                          |

## Utility Functions

```python
from sm_gco import get_cycle_triangles, tri_bicolouring

# Extract candidate triangles from the target mesh
cycle_tris = get_cycle_triangles(vy, fy, cycle_size=4, angle_threshold=np.pi/3)

# Compute bipartite graph colouring of source mesh faces
colours = tri_bicolouring(fx)  # returns (|FX|, 1) array of 0s and 1s
```

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## Attribution

If you use this code, please cite:

```bibtex
@inproceedings{roetzer2026markov,
    author    = {Paul Roetzer and Johan Thunberg and Zorah L\"{a}hner and Florian Bernard},
    title     = {Fast Markov Random Field Optimisation for Topologically Noisy 3D Shape Matching},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2026}
}
```
