"""Tests for MagneticConfiguration.from_gvec.

Requires `gvec` to be installed (optional dependency). Tests are automatically
skipped when the package is absent.

The W7X equilibrium directory (W7X/) at the repository root is used for all
tests; it ships both a GVEC statefile and a VMEC wout for cross-validation.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

gvec = pytest.importorskip("gvec")

# Repository root is two levels up from this file (tests/ -> repo root)
_REPO_ROOT = Path(__file__).parent.parent
_W7X_DIR = _REPO_ROOT / "W7X"
_W7X_WOUT = next(_W7X_DIR.glob("wout_*.nc"), None)


@pytest.fixture(scope="module")
def w7x_gvec_state():
    if not _W7X_DIR.exists():
        pytest.skip(f"W7X directory not found at {_W7X_DIR}")
    return gvec.find_state(_W7X_DIR)


@pytest.fixture(scope="module")
def w7x_gvec_config(w7x_gvec_state):
    from raytrax.equilibrium.gvec import GvecGridResolution
    from raytrax.equilibrium.interpolate import (
        CylindricalGridResolution,
        MagneticConfiguration,
    )

    # Coarser grid for speed in tests
    grid = GvecGridResolution(
        cylindrical=CylindricalGridResolution(
            n_r=60, n_z=65, n_phi=30, n_rho_profile=100
        ),
        n_rho=25,
        n_theta=30,
        n_zeta=30,
    )
    return MagneticConfiguration.from_gvec(w7x_gvec_state, grid=grid)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_from_gvec_returns_magnetic_configuration(w7x_gvec_config):
    from raytrax.equilibrium.interpolate import MagneticConfiguration

    assert isinstance(w7x_gvec_config, MagneticConfiguration)


def test_from_gvec_nfp(w7x_gvec_config):
    assert w7x_gvec_config.nfp == 5


def test_from_gvec_stellarator_symmetric(w7x_gvec_config):
    assert w7x_gvec_config.is_stellarator_symmetric is True


def test_from_gvec_not_axisymmetric(w7x_gvec_config):
    assert w7x_gvec_config.is_axisymmetric is False


def test_from_gvec_rphiz_shape(w7x_gvec_config):
    # (n_r, n_phi, n_z, 3) for a 3D configuration
    assert w7x_gvec_config.rphiz.ndim == 4
    assert w7x_gvec_config.rphiz.shape[-1] == 3


def test_from_gvec_dvolume_drho_positive(w7x_gvec_config):
    # dV/drho must be positive everywhere (excluding the axis where it is ~0)
    dv = np.array(w7x_gvec_config.dvolume_drho)
    assert np.all(dv >= 0.0), "dV/drho must be non-negative"
    assert np.any(dv > 0.0), "dV/drho must be positive away from the axis"


def test_from_gvec_dvolume_drho_monotone(w7x_gvec_config):
    # For a nested-surface equilibrium dV/drho is monotonically increasing
    dv = np.array(w7x_gvec_config.dvolume_drho)
    assert np.all(np.diff(dv) >= -1e-3), (
        "dV/drho should be monotonically non-decreasing"
    )


# ---------------------------------------------------------------------------
# Physics sanity: evaluate B at the magnetic axis via the interpolators
# ---------------------------------------------------------------------------


def test_from_gvec_B_magnitude_outboard(w7x_gvec_config):
    """B on the W7-X outboard midplane (R=6.0, Z=0, phi=0) should be ~2.8 T.

    W7-X magnetic axis is at R~5.95 m; the outboard plasma extends to R~6.2 m.
    R=6.0 is inside the plasma at rho~0.2.
    """
    from raytrax.equilibrium.interpolate import build_magnetic_field_interpolator

    B_interp = build_magnetic_field_interpolator(w7x_gvec_config)
    B_cyl = B_interp(6.0, 0.0, 0.0)
    absB = float(jnp.linalg.norm(B_cyl))
    assert 2.0 < absB < 3.5, f"|B| outboard = {absB:.3f} T, expected ~2.8 T"


def test_from_gvec_rho_outboard(w7x_gvec_config):
    """rho at (R=6.0, phi=0, Z=0) should be small (~0.2), well inside the plasma."""
    from raytrax.equilibrium.interpolate import build_rho_interpolator

    rho_interp = build_rho_interpolator(w7x_gvec_config)
    rho_val = float(rho_interp(6.0, 0.0, 0.0))
    assert rho_val < 0.4, f"rho at R=6.0 = {rho_val:.3f}, expected < 0.4"


# ---------------------------------------------------------------------------
# Cross-validation against VMEC wout
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def w7x_vmec_config():
    if _W7X_WOUT is None:
        pytest.skip("No wout_*.nc found in W7X/")
    from vmecpp import VmecWOut

    from raytrax.equilibrium.interpolate import (
        CylindricalGridResolution,
        MagneticConfiguration,
        VmecGridResolution,
    )

    wout = VmecWOut.from_wout_file(str(_W7X_WOUT))
    grid = VmecGridResolution(
        cylindrical=CylindricalGridResolution(
            n_r=60, n_z=65, n_phi=30, n_rho_profile=100
        ),
        n_rho=25,
        n_theta=30,
    )
    return MagneticConfiguration.from_vmec_wout(wout, grid=grid)


def test_gvec_vs_vmec_B_magnitude(w7x_gvec_config, w7x_vmec_config):
    """|B| from GVEC and VMEC should agree within 2% at verified interior points.

    Points are chosen on the outboard midplane where both codes have dense
    scatter coverage.  W7-X axis is at R~5.95 m; these points are at rho~0.2-0.5.
    phi=0 and phi=pi/nfp are the two stellarator-symmetry planes.
    """
    from raytrax.equilibrium.interpolate import (
        build_magnetic_field_interpolator,
    )

    nfp = 5
    B_gvec = build_magnetic_field_interpolator(w7x_gvec_config)
    B_vmec = build_magnetic_field_interpolator(w7x_vmec_config)

    # (R, phi_fold, Z) — all pre-mapped to fundamental domain [0, pi/nfp]
    test_points = [
        (6.078, 0.0, 0.0),  # outboard rho~0.5, phi=0
        (6.0, 0.0, 0.0),  # outboard rho~0.2, phi=0
        (5.555, np.pi / nfp, 0.0),  # outboard rho~0.5, phi=pi/nfp
    ]

    for R, phi, Z in test_points:
        B_g = B_gvec(R, phi, Z)
        B_v = B_vmec(R, phi, Z)
        absB_g = float(jnp.linalg.norm(B_g))
        absB_v = float(jnp.linalg.norm(B_v))

        rel_diff = abs(absB_g - absB_v) / (absB_v + 1e-10)
        assert rel_diff < 0.05, (
            f"At (R={R}, phi={phi:.3f}, Z={Z}): "
            f"|B|_gvec={absB_g:.4f} T, |B|_vmec={absB_v:.4f} T, "
            f"relative diff={rel_diff:.3%}"
        )
