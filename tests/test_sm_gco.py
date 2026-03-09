"""
Test suite for sm_gco shape matching library.

Uses the built-in test shapes (getTestShapeX, getTestShapeY, getFeatureDiffXY)
which are exposed via the Python bindings.

Run with: pytest tests/test_sm_gco.py -v
"""

import numpy as np
import pytest

from sm_gco import (
    GCOSM,
    COST_MODE,
    TriangleWiseOpts,
    get_test_shape_x,
    get_test_shape_y,
    get_feature_diff_xy,
    get_cycle_triangles,
    tri_bicolouring,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_data():
    """Load the built-in test shapes and feature difference matrix."""
    vx, fx = get_test_shape_x()
    vy, fy = get_test_shape_y()
    feat_diff = get_feature_diff_xy()
    return vx, fx, vy, fy, feat_diff


@pytest.fixture(scope="session")
def smgco_instance(test_data):
    """Create a GCOSM instance from the test data."""
    vx, fx, vy, fy, feat_diff = test_data
    return GCOSM(vx, fx, vy, fy, feat_diff)


# ---------------------------------------------------------------------------
# Test: Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_gcosm_construction(self, test_data):
        """GCOSM can be constructed with valid test data without crashing."""
        vx, fx, vy, fy, feat_diff = test_data
        smgco = GCOSM(vx, fx, vy, fy, feat_diff)
        assert smgco is not None

    def test_test_data_shapes(self, test_data):
        """Test data has expected shapes (3D vertices, triangular faces)."""
        vx, fx, vy, fy, feat_diff = test_data
        assert vx.shape[1] == 3, "VX should have 3 columns (x, y, z)"
        assert fx.shape[1] == 3, "FX should have 3 columns (triangle faces)"
        assert vy.shape[1] == 3, "VY should have 3 columns (x, y, z)"
        assert fy.shape[1] == 3, "FY should have 3 columns (triangle faces)"
        assert feat_diff.shape == (vx.shape[0], vy.shape[0]), \
            f"Feature diff should be |VX| x |VY|, got {feat_diff.shape}"


# ---------------------------------------------------------------------------
# Test: TriangleWiseOpts Defaults
# ---------------------------------------------------------------------------

class TestTriangleWiseOptsDefaults:
    def test_default_cost_mode(self):
        opts = TriangleWiseOpts()
        assert opts.cost_mode == COST_MODE.MULTIPLE_LABLE_SPACE_GEODIST_MAX

    def test_default_algorithm(self):
        opts = TriangleWiseOpts()
        assert opts.algorithm == 5

    def test_default_set_initial_lables(self):
        opts = TriangleWiseOpts()
        assert opts.set_initial_lables == 4

    def test_default_smooth_weight(self):
        opts = TriangleWiseOpts()
        assert opts.smooth_weight == pytest.approx(1000.0)

    def test_default_unary_weight(self):
        opts = TriangleWiseOpts()
        assert opts.unary_weight == pytest.approx(1.0)

    def test_default_smooth_scale_before_robust(self):
        opts = TriangleWiseOpts()
        assert opts.smooth_scale_before_robust == pytest.approx(0.1)

    def test_default_robust_cost(self):
        opts = TriangleWiseOpts()
        assert opts.robust_cost == 8

    def test_default_lable_space_cycle_size(self):
        opts = TriangleWiseOpts()
        assert opts.lable_space_cycle_size == 4

    def test_default_lable_space_angle_thres(self):
        opts = TriangleWiseOpts()
        assert opts.lable_space_angle_thres == pytest.approx(np.pi / 3)

    def test_default_lable_space_degenerate(self):
        opts = TriangleWiseOpts()
        assert opts.lable_space_degenerate is True

    def test_default_glue_solution(self):
        opts = TriangleWiseOpts()
        assert opts.glue_solution is True

    def test_default_label_order(self):
        opts = TriangleWiseOpts()
        assert opts.label_order == 0

    def test_default_membrane_energy_weight(self):
        opts = TriangleWiseOpts()
        assert opts.membrane_energy_weight == pytest.approx(0.0)

    def test_default_bending_energy_weight(self):
        opts = TriangleWiseOpts()
        assert opts.bending_energy_weight == pytest.approx(0.0)

    def test_default_feature_weight(self):
        opts = TriangleWiseOpts()
        assert opts.feature_weight == pytest.approx(1.0)

    def test_default_same_label_cost(self):
        opts = TriangleWiseOpts()
        assert opts.same_label_cost == pytest.approx(0.0)

    def test_default_bicolouring(self):
        opts = TriangleWiseOpts()
        assert opts.bicolouring is False


# ---------------------------------------------------------------------------
# Test: pointWise matching
# ---------------------------------------------------------------------------

class TestPointWise:
    @pytest.mark.skip(reason="pointWise has a pre-existing Bus error (segfault) — needs C++ debugging")
    def test_pointwise_l2(self, test_data):
        """pointWise with L2 smooth cost produces valid output."""
        vx, fx, vy, fy, feat_diff = test_data
        smgco = GCOSM(vx, fx, vy, fy, feat_diff)
        result = smgco.point_wise(False)
        assert result.shape[0] == vx.shape[0], "Output should have one row per vertex"
        assert result.shape[1] == 1, "Output should have 1 column"
        assert np.all(result >= 0), "All indices should be non-negative"
        assert np.all(result < vy.shape[0]), "All indices should be < |VY|"


# ---------------------------------------------------------------------------
# Test: triangleWise matching
# ---------------------------------------------------------------------------

class TestTriangleWise:
    def test_trianglewise_default_opts(self, test_data):
        """triangleWise with default opts returns the expected 5-tuple."""
        vx, fx, vy, fy, feat_diff = test_data
        smgco = GCOSM(vx, fx, vy, fy, feat_diff)
        result = smgco.triangle_wise()
        assert isinstance(result, tuple), "Should return a tuple"
        assert len(result) == 5, "Should return a 5-tuple"
        optime, p2p, tri_tri, raw_p2p, raw_tri_tri = result
        assert isinstance(optime, float), "optime should be a float"
        assert p2p.shape[1] == 2, "p2p should have 2 columns"
        assert np.all(p2p[:, 0] >= 0), "Source indices should be non-negative"

    def test_trianglewise_with_typical_opts(self, test_data):
        """triangleWise with the typical usage options runs successfully."""
        vx, fx, vy, fy, feat_diff = test_data
        smgco = GCOSM(vx, fx, vy, fy, feat_diff)
        opts = TriangleWiseOpts()
        opts.cost_mode = COST_MODE.MULTIPLE_LABLE_SPACE_GEODIST_MAX
        opts.algorithm = 5
        opts.set_initial_lables = 4
        opts.lable_space_cycle_size = 4
        opts.lable_space_angle_thres = np.pi / 3
        opts.smooth_weight = 1000.0
        opts.unary_weight = 1.0
        opts.robust_cost = 8
        opts.smooth_scale_before_robust = 0.1
        opts.lable_space_degenerate = True
        opts.glue_solution = True

        optime, p2p, tri_tri, raw_p2p, raw_tri_tri = smgco.triangle_wise(opts)
        assert optime >= 0, "Optimization time should be non-negative"
        assert p2p.shape[0] == vx.shape[0], "Glued p2p should have |VX| rows"
        assert p2p.shape[1] == 2, "p2p should have 2 columns"
        assert np.all(p2p[:, 1] >= 0), "Target indices should be non-negative"
        assert np.all(p2p[:, 1] < vy.shape[0]), "Target indices should be < |VY|"

    @pytest.mark.skip(reason="algorithm=0 (swap) segfaults on arm64 — pre-existing GCO issue")
    def test_trianglewise_algorithm_0_swap(self, test_data):
        """triangleWise with algorithm=0 (alpha-beta swap) runs."""
        vx, fx, vy, fy, feat_diff = test_data
        smgco = GCOSM(vx, fx, vy, fy, feat_diff)
        opts = TriangleWiseOpts()
        opts.algorithm = 0
        opts.cost_mode = COST_MODE.MULTIPLE_LABLE_SPACE_GEODIST_MAX
        opts.lable_space_degenerate = False  # swap doesn't work well with degenerate
        result = smgco.triangle_wise(opts)
        assert len(result) == 5

    @pytest.mark.skip(reason="algorithm=1 (expansion) segfaults on arm64 — pre-existing GCO issue")
    def test_trianglewise_algorithm_1_expansion(self, test_data):
        """triangleWise with algorithm=1 (alpha expansion) runs."""
        vx, fx, vy, fy, feat_diff = test_data
        smgco = GCOSM(vx, fx, vy, fy, feat_diff)
        opts = TriangleWiseOpts()
        opts.algorithm = 1
        opts.cost_mode = COST_MODE.MULTIPLE_LABLE_SPACE_GEODIST_MAX
        result = smgco.triangle_wise(opts)
        assert len(result) == 5

    def test_trianglewise_no_glue(self, test_data):
        """triangleWise with glue_solution=False returns raw correspondences."""
        vx, fx, vy, fy, feat_diff = test_data
        smgco = GCOSM(vx, fx, vy, fy, feat_diff)
        opts = TriangleWiseOpts()
        opts.glue_solution = False
        opts.cost_mode = COST_MODE.MULTIPLE_LABLE_SPACE_GEODIST_MAX
        optime, p2p, tri_tri, raw_p2p, raw_tri_tri = smgco.triangle_wise(opts)
        assert p2p.shape[1] == 2
        assert raw_p2p.shape[1] == 2

    def test_trianglewise_geodist_cost_mode(self, test_data):
        """triangleWise with GEODIST (sum) cost mode runs."""
        vx, fx, vy, fy, feat_diff = test_data
        smgco = GCOSM(vx, fx, vy, fy, feat_diff)
        opts = TriangleWiseOpts()
        opts.cost_mode = COST_MODE.MULTIPLE_LABLE_SPACE_GEODIST
        opts.algorithm = 5
        result = smgco.triangle_wise(opts)
        assert len(result) == 5

    def test_trianglewise_set_max_iter(self, test_data):
        """set_max_iter limits the number of iterations."""
        vx, fx, vy, fy, feat_diff = test_data
        smgco = GCOSM(vx, fx, vy, fy, feat_diff)
        smgco.set_max_iter(1)
        opts = TriangleWiseOpts()
        result = smgco.triangle_wise(opts)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Test: Utility functions
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_get_cycle_triangles(self, test_data):
        """get_cycle_triangles returns valid triangle indices."""
        _, _, vy, fy, _ = test_data
        cycle_tris = get_cycle_triangles(vy, fy, 4, np.pi / 3)
        assert cycle_tris.shape[1] == 3, "Should return Nx3 matrix of triangle vertices"
        assert cycle_tris.shape[0] > 0, "Should return at least some triangles"
        assert np.all(cycle_tris >= 0), "All vertex indices should be non-negative"
        assert np.all(cycle_tris < vy.shape[0]), "All vertex indices should be < |VY|"

    def test_tri_bicolouring(self, test_data):
        """tri_bicolouring returns valid 0/1 colouring."""
        _, fx, _, _, _ = test_data
        colours = tri_bicolouring(fx)
        assert colours.shape[0] == fx.shape[0], "One colour per face"
        assert np.all((colours == 0) | (colours == 1)), "Colours should be 0 or 1"


# ---------------------------------------------------------------------------
# Test: COST_MODE enum
# ---------------------------------------------------------------------------

class TestCostModeEnum:
    def test_all_cost_modes_exist(self):
        """All expected cost modes are available."""
        expected_modes = [
            "SINGLE_LABLE_SPACE_L2",
            "MULTIPLE_LABLE_SPACE_L2",
            "MULTIPLE_LABLE_SPACE_SO3",
            "MULTIPLE_LABLE_SPACE_SE3",
            "MULTIPLE_LABLE_SPACE_GEODIST",
            "MULTIPLE_LABLE_SPACE_GEODIST_MAX",
            "MULTIPLE_LABLE_SPACE_L2DIST",
            "MULTIPLE_LABLE_SPACE_L2DIST_MAX",
            "MULTIPLE_LABLE_SPACE_GEODIST_SO3",
        ]
        for mode_name in expected_modes:
            assert hasattr(COST_MODE, mode_name), f"COST_MODE.{mode_name} should exist"
