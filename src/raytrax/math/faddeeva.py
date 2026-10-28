"""Plasma dispersion function Z(ζ) and its derivative, computed via the Faddeeva function.

The Faddeeva function w(z) = exp(-z²)·erfc(-iz) is implemented in pure JAX
using Weideman's (1994) rational approximation.

Reference:
    J. A. C. Weideman, "Computation of the complex error function", SIAM J.
    Numer. Anal. 31(5), 1497-1518 (1994).
"""

import jax
import jax.numpy as jnp
import numpy as np


def _weideman_coefficients(n: int) -> tuple[float, np.ndarray]:
    """Length scale L and polynomial coefficients (highest degree first), Weideman eq. (3.6)."""
    length = np.sqrt(n / np.sqrt(2))
    k = np.arange(-2 * n + 1, 2 * n)
    t = length * np.tan(k * np.pi / (4 * n))
    f = np.concatenate([[0.0], np.exp(-(t**2)) * (length**2 + t**2)])
    a = np.real(np.fft.fft(np.fft.fftshift(f))) / (4 * n)
    return length, a[1 : n + 1][::-1]


# 40 terms: ~1e-14 relative accuracy in the upper half-plane, where the Shkarofsky
# functions evaluate w, and ~2e-15 on the real axis, where their psi -> 0
# recurrence amplifies errors by ~1e4 (32 terms leave 3e-13 there).
_L, _A = _weideman_coefficients(40)


def _wofz_upper(z: jax.Array) -> jax.Array:
    """w(z) for Im(z) >= 0: one polynomial in (L + iz)/(L - iz), no loops or branches."""
    d = _L - 1j * z
    big_z = (_L + 1j * z) / d
    p = jnp.polyval(jnp.asarray(_A), big_z)
    return 2 * p / d**2 + 1 / (jnp.sqrt(jnp.pi) * d)


@jax.custom_jvp
def wofz_jax(z):
    """Compute the Faddeeva function w(z) = exp(-z²)·erfc(-iz) for complex z."""
    z = jnp.asarray(z, dtype=jnp.complex128)
    lower = jnp.imag(z) < 0.0
    # The approximation holds for Im(z) >= 0; reflect: w(z) = 2exp(-z²) - w(-z).
    w = _wofz_upper(jnp.where(lower, -z, z))
    w = jnp.where(lower, 2.0 * jnp.exp(-z * z) - w, w)
    # Im(z) = 0: Re(w(t)) = exp(-t²); the approximation is accurate there only in
    # absolute terms, so Re(w) of a large real argument would carry no digits.
    return jnp.where(
        jnp.imag(z) == 0.0, jnp.exp(-(jnp.real(z) ** 2)) + 1j * jnp.imag(w), w
    )


@wofz_jax.defjvp
def wofz_jax_jvp(primals, tangents):
    """Custom JVP rule: w'(z) = 2i/√π - 2z·w(z)."""
    (z,) = primals
    (dz,) = tangents
    w_z = wofz_jax(z)
    w_prime = 2j / jnp.sqrt(jnp.pi) - 2 * z * w_z
    return w_z, w_prime * dz


def plasma_dispersion_function(z):
    """Compute the plasma dispersion function Z for a complex argument z."""
    return 1j * jnp.sqrt(jnp.pi) * wofz_jax(z)


def plasma_dispersion_function_derivative(z):
    """Compute the derivative of the plasma dispersion function Z'."""
    return -2 * (1 + z * plasma_dispersion_function(z))
