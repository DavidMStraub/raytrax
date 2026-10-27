"""Gradient-based ECRH beam steering using jax.grad through the ray tracer.

``trace(..., trim=False)`` returns a TraceResult whose BeamProfile contains the
full padded tensor output of the ODE solve, differentiable w.r.t. beam.position
and beam.direction. Any scalar loss defined from those tensors can be
differentiated with jax.grad.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from raytrax import Beam, MagneticConfiguration, RadialProfiles, trace  # noqa: E402

# ─── 1. Analytic tokamak equilibrium ──────────────────────────────────────────
#
# Circular cross-section, vacuum toroidal field B_phi = B0*R0/R.
# At R = R0 = 3 m, |B| = B0 = 2.5 T  →  f_ce ≈ 70 GHz  →  2nd harmonic 140 GHz.

R0, a, B0 = 3.0, 1.0, 2.5
n_R, n_Z = 50, 60

R_grid = jnp.linspace(R0 - 1.5 * a, R0 + 1.5 * a, n_R)
Z_grid = jnp.linspace(-1.5 * a, 1.5 * a, n_Z)
R_2d, Z_2d = jnp.meshgrid(R_grid, Z_grid, indexing="ij")

B_phi = B0 * R0 / R_2d
rho_2d = jnp.sqrt((R_2d - R0) ** 2 + Z_2d**2) / a

phi_grid = jnp.array([0.0])
rphiz = jnp.stack(jnp.meshgrid(R_grid, phi_grid, Z_grid, indexing="ij"), axis=-1)
mag_field_grid = jnp.stack(
    [jnp.zeros_like(R_2d), B_phi, jnp.zeros_like(R_2d)], axis=-1
)[:, jnp.newaxis, :, :]
rho_grid = rho_2d[:, jnp.newaxis, :]

rho_1d = jnp.linspace(0, 1, 200)
dvolume_drho = 4.0 * jnp.pi**2 * R0 * a**2 * rho_1d

eq_interp = MagneticConfiguration(
    rphiz=rphiz,
    magnetic_field=mag_field_grid,
    rho=rho_grid,
    nfp=1,
    is_stellarator_symmetric=False,
    rho_1d=rho_1d,
    dvolume_drho=dvolume_drho,
    is_axisymmetric=True,
)

# ─── 2. Parabolic plasma profiles ─────────────────────────────────────────────

rho_prof = jnp.linspace(0, 1, 200)
profiles = RadialProfiles(
    rho=rho_prof,
    electron_density=0.5 * (1.0 - rho_prof**2),
    electron_temperature=3.0 * (1.0 - rho_prof**2),
)

# ─── 3. Loss functions defined on BeamProfile ──────────────────────────────────
#
# trace(..., trim=False) returns a TraceResult whose beam_profile contains padded
# arrays (TracerSettings.max_steps + 1 slots: t0 + one per ODE step). All fields
# are differentiable w.r.t. beam.position and beam.direction. Padded entries have
# linear_power_density=0, so jnp.max and weighted jnp.sum are correct without
# any explicit trimming.
#
# Note: each call rebuilds interpolator Python objects (negligible for small
# grids; for W7-X-scale grids build them once and call trace_jitted directly).


def make_beam(position, direction):
    return Beam(
        position=position,
        direction=direction,
        frequency=jnp.array(140e9),
        mode="O",
        power=1e6,
    )


def absorbed_fraction(position: jax.Array, direction: jax.Array) -> jax.Array:
    """Total absorbed power fraction 1 − exp(−τ)."""
    result = trace(eq_interp, profiles, make_beam(position, direction), trim=False)
    # diffrax fills unused buffer slots with inf; ignore them when taking the max.
    tau_final = jnp.max(
        jnp.where(
            jnp.isfinite(result.beam_profile.optical_depth),
            result.beam_profile.optical_depth,
            0.0,
        )
    )
    return 1.0 - jnp.exp(-tau_final)


def deposition_centroid(
    position: jax.Array, direction: jax.Array, target_rho: float = 0.5
) -> jax.Array:
    """Gaussian-weighted linear power density centred at target_rho.

    Maximizing this steers the deposition toward the desired flux surface.
    """
    result = trace(eq_interp, profiles, make_beam(position, direction), trim=False)
    w = jnp.exp(
        -(((result.beam_profile.normalized_effective_radius - target_rho) / 0.1) ** 2)
    )
    return jnp.sum(result.beam_profile.linear_power_density * w)


# ─── 4. Forward pass ──────────────────────────────────────────────────────────

position = jnp.array([R0 + 1.5 * a, 0.0, 0.0])  # outer midplane, ρ > 1
direction = jnp.array([-1.0, 0.0, 0.0])  # inward

print("=" * 60)
print("Forward pass")
print("=" * 60)
print(f"  Absorbed fraction:      {float(absorbed_fraction(position, direction)):.6f}")
print(
    f"  Deposition @ ρ=0.5:     {float(deposition_centroid(position, direction)):.6f}"
)

# ─── 5. Gradients ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Gradients via jax.grad")
print("=" * 60)

grad_pos, grad_dir = jax.grad(absorbed_fraction, argnums=(0, 1))(position, direction)
print(f"  ∂(absorbed)/∂position  = {np.array(grad_pos)}")
print(f"  ∂(absorbed)/∂direction = {np.array(grad_dir)}")

grad_pos2, grad_dir2 = jax.grad(deposition_centroid, argnums=(0, 1))(
    position, direction
)
print(f"  ∂(centroid)/∂position  = {np.array(grad_pos2)}")
print(f"  ∂(centroid)/∂direction = {np.array(grad_dir2)}")

# ─── 6. Finite-difference check ───────────────────────────────────────────────
#
# Parametrise beam direction as (−cos θ, 0, sin θ) in the R–Z plane.
# Keeping N_y = 0 exactly avoids the adjoint instability tan(θ) = N_⊥/N_∥ → ∞
# that occurs when N_y drifts to floating-point noise after gradient steps.


@jax.jit
def absorbed_vs_angle(theta: jax.Array) -> jax.Array:
    d = jnp.array([-jnp.cos(theta), 0.0, jnp.sin(theta)])
    return absorbed_fraction(position, d)


theta_ref = jnp.array(0.3)
eps = 1e-5
fd = (
    float(absorbed_vs_angle(theta_ref + eps))
    - float(absorbed_vs_angle(theta_ref - eps))
) / (2 * eps)
ad = float(jax.grad(absorbed_vs_angle)(theta_ref))

print("\n" + "=" * 60)
print("Finite-difference check (angle θ, central, ε = 1e-5 rad)")
print("=" * 60)
print(f"  dP/dθ  (FD) : {fd:.4f}")
print(f"  dP/dθ  (AD) : {ad:.4f}")
print(f"  Relative error : {abs(fd - ad) / (abs(fd) + 1e-12):.4e}")

# ─── 7. Gradient-ascent optimization ──────────────────────────────────────────
#
# Maximize absorbed fraction by steering θ in the R–Z plane.
# N_y = 0 is mandatory: free N_y causes gradient explosion ~1/N_y² in adjoint.

print("\n" + "=" * 60)
print("Gradient ascent: maximize absorbed fraction (poloidal beam angle)")
print("=" * 60)

lr = 0.05
theta = jnp.array(0.3)
value_and_grad_angle = jax.jit(jax.value_and_grad(absorbed_vs_angle))

for step in range(20):
    p_val, g_theta = value_and_grad_angle(theta)
    if step % 5 == 0 or step == 19:
        d_theta = jnp.array([-jnp.cos(theta), 0.0, jnp.sin(theta)])
        print(
            f"  step {step:2d}: P_abs = {float(p_val):.6f},  θ = {float(theta):.4f} rad,"
            f"  dP/dθ = {float(g_theta):.4f},  dir = {np.array(d_theta)}"
        )
    theta = theta + lr * g_theta

print(f"\nFinal absorbed fraction: {float(absorbed_vs_angle(theta)):.6f}")
print(f"Final angle: θ = {float(theta):.4f} rad = {float(theta) * 180 / np.pi:.2f}°")
