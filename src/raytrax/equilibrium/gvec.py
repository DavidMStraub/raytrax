"""GVEC equilibrium import onto the standard cylindrical grid.

GVEC's nfp-fold symmetry is, in general, a rotation by 2*pi/nfp around some
axis **a** in 3D space -- not necessarily the Z-axis (see
:func:`extract_gframe_axis`). The raytrax solver, however, hard-codes the
convention used by the VMEC pipeline: B and rho are stored on a plain
cylindrical (R, phi, Z) grid covering the fundamental domain
phi in [0, pi/nfp], with the symmetry axis assumed to be the Z-axis through
the coordinate origin.

Conventional toroidal devices (e.g. W7-X) satisfy this convention -- we
verified numerically that their extracted axis is the Z-axis through the
origin to ~1e-16. ``cylindrical_grid_for_gvec_equilibrium`` therefore checks
this assumption (:func:`_check_standard_axis`) and raises
``NotImplementedError`` if it doesn't hold, rather than silently producing a
wrong grid. Supporting a tilted/offset symmetry axis in general would require
carrying the symmetry frame through ``MagneticConfiguration``,
``Interpolators``, and the solver's field-evaluation hot path -- deferred
until a concrete equilibrium actually needs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from beartype import beartype as typechecker
from scipy.interpolate import griddata

from raytrax.equilibrium.interpolate import CylindricalGridResolution

if TYPE_CHECKING:
    import gvec


@dataclass
class GvecGridResolution:
    """Grid resolution for GVEC-based equilibrium imports.

    Attributes:
        cylindrical: Output cylindrical grid shared with all other importers.
        n_rho: Radial sample points on the GVEC curvilinear grid.
        n_theta: Poloidal sample points.
        n_zeta: Toroidal sample points per field period.
    """

    cylindrical: CylindricalGridResolution = dataclass_field(
        default_factory=CylindricalGridResolution
    )
    n_rho: int = 40
    n_theta: int = 45
    n_zeta: int = 50


def extract_gframe_axis(
    state: gvec.State,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract the G-frame rotation axis, centre, and reference directions.

    The nfp-fold symmetry of the equilibrium is a rotation by 2*pi/nfp around
    some axis **a** passing through centre **C**.  We find **a** from the
    period-start positions of the magnetic axis: sampling GVEC at
    zeta = k * 2*pi/nfp for k = 0...nfp-1 gives nfp points that lie on a circle;
    the normal to that circle is **a**.

    For nfp = 1 (no rotational reduction) we fall back to the Z-axis so that
    G-frame cylindrical = standard cylindrical.

    Returns:
        a:  Unit rotation axis (3,).
        C:  Reference point on the axis (3,), centroid of the nfp samples.
        e1: Unit vector from C toward the zeta=0 axis position, perp to a (3,).
        e2: a cross e1, completes the right-handed frame (3,).
    """
    nfp = state.nfp

    if nfp == 1:
        a = np.array([0.0, 0.0, 1.0])
        C = np.zeros(3)
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
        return a, C, e1, e2

    # Period-start axis positions: zeta = 0, 2*pi/nfp, ..., (nfp-1)*2*pi/nfp
    zeta_starts = np.array([k * 2.0 * np.pi / nfp for k in range(nfp)])
    ds = state.evaluate("pos", rho=0.001, theta=0.0, zeta=zeta_starts)
    pts = ds["pos"].values[:, 0, 0, :].T  # (nfp, 3)

    C = pts.mean(axis=0)

    # Rotation axis: normal to the plane of the nfp points.
    vecs = pts - C
    a_raw = np.sum(np.cross(vecs, np.roll(vecs, -1, axis=0)), axis=0)
    a = a_raw / np.linalg.norm(a_raw)

    # Normalise sign: ensure a points in the positive-Z half-space (or +X if horizontal).
    if a[2] < -1e-6:
        a = -a
    elif abs(a[2]) < 1e-6 and a[0] < -1e-6:
        a = -a

    # Reference direction e1: from C toward the zeta=0 axis position.
    v0 = pts[0] - C
    v0_perp = v0 - np.dot(v0, a) * a
    e1 = v0_perp / np.linalg.norm(v0_perp)
    e2 = np.cross(a, e1)

    return a, C, e1, e2


@jt.jaxtyped(typechecker=typechecker)
def dvolume_drho_gvec(
    state: gvec.State,
    rho_1d: jt.Float[np.ndarray, " n_rho"],
    n_theta_fine: int = 60,
    n_zeta_fine: int = 60,
) -> jt.Float[jax.Array, " n_rho"]:
    """Compute dV/drho on a 1D radial grid by integrating the GVEC Jacobian.

    dV/drho(rho) = nfp * integral_0^{2*pi/nfp} integral_0^{2*pi} J(rho, theta, zeta) dtheta dzeta

    Args:
        state: GVEC State object.
        rho_1d: Radial sample points in [0, 1].
        n_theta_fine: Poloidal resolution for the integration.
        n_zeta_fine: Toroidal resolution for the integration (one field period).

    Returns:
        dV/drho at each rho value.
    """
    theta_vals = np.linspace(0, 2 * np.pi, n_theta_fine, endpoint=False)
    zeta_vals = np.linspace(0, 2 * np.pi / state.nfp, n_zeta_fine, endpoint=False)

    ds = state.evaluate("Jac", rho=rho_1d, theta=theta_vals, zeta=zeta_vals)
    jac = ds["Jac"].values

    dtheta = 2 * np.pi / n_theta_fine
    dzeta = (2 * np.pi / state.nfp) / n_zeta_fine
    dv_drho = state.nfp * np.sum(jac, axis=(1, 2)) * dtheta * dzeta
    return jnp.array(dv_drho)


def _check_standard_axis(state: gvec.State, atol: float = 1e-6) -> None:
    """Raise unless the equilibrium's nfp-fold symmetry axis is the Z-axis through the origin.

    ``cylindrical_grid_for_gvec_equilibrium`` builds a plain $(R, \\phi, Z)$
    grid -- the same representation :func:`cylindrical_grid_for_equilibrium`
    builds for VMEC imports -- which implicitly assumes the discrete
    rotational symmetry is a rotation about the $Z$-axis through the
    coordinate origin (true for conventional toroidal devices like W7-X,
    verified numerically to ~1e-16 there). Supporting a tilted/offset axis
    in general would require carrying the symmetry frame through
    ``MagneticConfiguration``/``Interpolators``/the solver; that's deferred
    until a concrete equilibrium actually needs it.

    Raises:
        NotImplementedError: If the extracted axis, center, or reference
            direction deviates from the standard convention by more than
            ``atol``.
    """
    a, C, e1, _ = extract_gframe_axis(state)
    if not (
        np.allclose(a, [0.0, 0.0, 1.0], atol=atol)
        and np.allclose(C, 0.0, atol=atol)
        and np.allclose(e1, [1.0, 0.0, 0.0], atol=atol)
    ):
        raise NotImplementedError(
            "from_gvec only supports equilibria whose nfp-fold symmetry "
            "axis is the Z-axis through the coordinate origin (the "
            "conventional toroidal-device convention). Detected rotation "
            f"axis a={a}, center C={C}, reference direction e1={e1} -- "
            "general (tilted/offset) symmetry axes are not supported yet."
        )


@jt.jaxtyped(typechecker=typechecker)
def cylindrical_grid_for_gvec_equilibrium(
    state: gvec.State,
    n_rho: int,
    n_theta: int,
    n_zeta: int,
    n_r: int,
    n_z: int,
    n_phi: int,
) -> jt.Float[jax.Array, "n_r n_phi n_z rphizrhoBcyl=7"]:
    """Interpolate a GVEC equilibrium onto a standard cylindrical grid.

    Samples a full field period (zeta in [0, 2*pi/nfp)), converts every
    scatter point to standard cylindrical coordinates (R, phi, Z), and runs
    a single 3D scipy.griddata call to produce a regular output grid covering
    the fundamental domain phi in [0, pi/nfp].

    The output layout [R, phi, Z, rho, B_R, B_phi, B_Z] and domain match
    cylindrical_grid_for_equilibrium (the VMEC pipeline) exactly, so
    MagneticConfiguration.from_gvec produces a configuration the solver
    handles via its existing stellarator-fold machinery, unchanged.

    Args:
        state: GVEC State. Must be stellarator-symmetric with its nfp-fold
            symmetry axis the Z-axis through the coordinate origin -- see
            _check_standard_axis (raises NotImplementedError otherwise).
        n_rho: Radial sample points on the GVEC curvilinear grid.
        n_theta: Poloidal sample points.
        n_zeta: Toroidal sample points per field period.
        n_r: Output grid points along R.
        n_z: Output grid points along Z.
        n_phi: Output grid points along phi.

    Returns:
        Array of shape (n_r, n_phi, n_z, 7) on the cylindrical grid, with
        phi in [0, pi/nfp].  Points outside the plasma convex hull are NaN.
    """
    _check_standard_axis(state)

    nfp = state.nfp
    period = 2.0 * np.pi / nfp
    half_period = period / 2.0

    # --- 1. Sample GVEC scatter over a full field period ---
    rho_vals = np.linspace(0.0, 1.0, n_rho)
    theta_vals = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    zeta_vals = np.linspace(0.0, period, n_zeta, endpoint=False)

    ds = state.evaluate("pos", "B", rho=rho_vals, theta=theta_vals, zeta=zeta_vals)
    pos = ds["pos"].values  # (xyz=3, n_rho, n_theta, n_zeta)
    B_xyz = ds["B"].values  # (n_rho, n_theta, n_zeta, xyz=3)

    xyz_scatter = np.moveaxis(pos, 0, -1)  # (n_rho, n_theta, n_zeta, 3)

    # --- 2. Convert to standard cylindrical (R, phi, Z) ---
    R = np.hypot(xyz_scatter[..., 0], xyz_scatter[..., 1])
    phi = np.arctan2(xyz_scatter[..., 1], xyz_scatter[..., 0])
    Z = xyz_scatter[..., 2]

    cp, sp = np.cos(phi), np.sin(phi)
    B_cyl = np.stack(
        [
            B_xyz[..., 0] * cp + B_xyz[..., 1] * sp,
            -B_xyz[..., 0] * sp + B_xyz[..., 1] * cp,
            B_xyz[..., 2],
        ],
        axis=-1,
    )

    rho_grid = np.broadcast_to(rho_vals[:, None, None], R.shape)

    # --- 3. Add extrapolated rows at rho=1.1 and 1.2 ---
    # Linear expansion from axis (rho=0) through LCFS (rho=1), B held at LCFS.
    R_axis, Z_axis = R[0], Z[0]
    phi_lcfs = phi[-1]
    extra_R, extra_phi, extra_Z = [], [], []
    extra_rho, extra_BR, extra_Bphi, extra_BZ = [], [], [], []

    for rho_extra in [1.1, 1.2]:
        R_ex = R_axis + rho_extra * (R[-1] - R_axis)
        Z_ex = Z_axis + rho_extra * (Z[-1] - Z_axis)
        extra_R.append(R_ex.ravel())
        extra_phi.append(phi_lcfs.ravel())
        extra_Z.append(Z_ex.ravel())
        extra_rho.append(np.full(R_ex.size, rho_extra))
        extra_BR.append(B_cyl[-1, ..., 0].ravel())
        extra_Bphi.append(B_cyl[-1, ..., 1].ravel())
        extra_BZ.append(B_cyl[-1, ..., 2].ravel())

    # --- 4. Collect scatter ---
    # Wrap phi into [0, period) — scatter from full-period zeta sampling lands here
    phi_wrapped = phi % period

    R_sc = np.concatenate([R.ravel()] + extra_R)
    phi_sc = np.concatenate([phi_wrapped.ravel()] + extra_phi)
    Z_sc = np.concatenate([Z.ravel()] + extra_Z)
    rho_sc = np.concatenate([rho_grid.ravel()] + extra_rho)
    BR_sc = np.concatenate([B_cyl[..., 0].ravel()] + extra_BR)
    Bphi_sc = np.concatenate([B_cyl[..., 1].ravel()] + extra_Bphi)
    BZ_sc = np.concatenate([B_cyl[..., 2].ravel()] + extra_BZ)

    # --- 5. Output grid: fundamental domain phi in [0, pi/nfp] ---
    # Sampling the full period for the scatter (step 1) gives griddata good
    # coverage on both sides of the phi=pi/nfp boundary; only the half-period
    # sub-grid is queried/stored, matching cylindrical_grid_for_equilibrium's
    # convention so the solver's existing stellarator-fold logic applies here
    # unchanged.
    r_min, r_max = float(np.min(R_sc)), float(np.max(R_sc))
    z_min, z_max = float(np.min(Z_sc)), float(np.max(Z_sc))

    R_grid = np.linspace(r_min, r_max, n_r)
    phi_grid = np.linspace(0.0, half_period, n_phi)
    Z_grid = np.linspace(z_min, z_max, n_z)

    R_out, phi_out, Z_out = np.meshgrid(R_grid, phi_grid, Z_grid, indexing="ij")
    query_pts = np.stack([R_out.ravel(), phi_out.ravel(), Z_out.ravel()], axis=-1)

    # --- 6. 3D scatter -> grid ---
    scatter_pts = np.stack([R_sc, phi_sc, Z_sc], axis=-1)
    scatter_vals = np.stack([rho_sc, BR_sc, Bphi_sc, BZ_sc], axis=-1)

    interp = griddata(scatter_pts, scatter_vals, query_pts, method="linear").reshape(
        n_r, n_phi, n_z, 4
    )

    rphiz = np.stack([R_out, phi_out, Z_out], axis=-1)
    result = np.concatenate([rphiz, interp], axis=-1)
    return jnp.array(result)
