import numpy as np
from physipy.utils import *
from physipy.numerics_data import Grid

__all__ = [
    'WKB_seed'
]

def WKB_seed(E, l, potential, grid = Grid(), outward = False, scattering = False, **kwargs):
    """
    Compute the seed value for the first Numerov integration step.

    For outward integration the behaviour depends on whether r_min lies in the
    classical region:

    - Classical at r_min (regular potential, e.g. HO, Coulomb, GP): the exact
      leading behaviour of the regular solution is u(r) ~ r^{l+1}. This
      avoids the phase-accumulation error of the WKB approximation, which
      cannot account for the integral of k from 0 to r_min.

    - Non-classical at r_min (hard-core potential, e.g. Lennard-Jones): the
      wavefunction is exponentially suppressed at the core wall and the
      hard-wall BC u(r_min) = 0 is physically correct. A WKB exponentially
      decaying seed is returned for the second point.

    For inward integration r_max is always in the classically forbidden region
    for a bound state, so the WKB exponentially decaying form is appropriate.

    Parameters
    ----------
    E : float
        Particle's energy.
    l : int
        Angular momentum quantum number.
    potential : callable
        Potential energy function V(r).
    grid : Grid
        Mesh parameters; r_min is used for outward, r_max for inward (default Grid()).
    outward : bool
        If True, seed is computed at r_min for outward integration.
        If False, seed is computed at r_max for inward integration (default False).
    scattering : bool
        If True, returns a complex seed for scattering states.
        If False, returns the real part for bound states (default False).
    kwargs : dict
        Additional arguments forwarded to k_squared.

    Returns
    -------
    seed : float or complex
        Value of psi at the second grid point from the boundary.
    """
    if outward:
        r = grid.r_min
        h = grid.h
        k2 = k_squared(r, l, E, potential, **kwargs)

        if np.asarray(k2).item() > 0:
            # Classical at r_min: regular potential (singularity weaker than
            # 1/r^2). The Frobenius indicial equation gives u(r) ~ r^{l+1}
            # as the regular solution regardless of the specific potential.
            return (r + h) ** (l + 1)
        else:
            # Non-classical at r_min: hard-core potential (e.g. LJ).
            # Return a WKB exponentially decaying seed for u(r_min + h);
            # the hard-wall BC u(r_min) = 0 is set by the caller.
            k2_2 = k_squared(r + h, l, E, potential, **kwargs)
            k_1 = np.sqrt(np.abs(k2))
            k_2 = np.sqrt(np.abs(k2_2))
            amplitude = np.where(k_2 > 0, np.sqrt(k_1 / k_2), 1.0)
            return amplitude * np.exp(-h / 2 * (k_1 + k_2))

    # Inward integration: r_max is in the classically forbidden region.
    # WKB gives the correct exponentially decaying form there.
    h = -grid.h
    r = grid.r_max

    k2_1 = k_squared(r, l, E, potential, **kwargs)
    k2_2 = k_squared(r + h, l, E, potential, **kwargs)

    k_1 = np.sqrt(np.abs(k2_1))
    k_2 = np.sqrt(np.abs(k2_2))

    amplitude = np.where(k_2 > 0, np.sqrt(k_1 / k_2), 1.0)

    # If k2_1 > 0 and k2_2 > 0 we are in the oscillatory region (scattering);
    # otherwise exponential decay (bound-state tail).
    if scattering:
        wkb_seed = np.where(
            (k2_1 > 0) & (k2_2 > 0),
            amplitude * np.exp( 1j * h / 2 * (k_1 + k_2)),
            amplitude * np.exp(-h / 2 * (k_1 + k_2))
        )
    else:
        wkb_seed = np.where(
            (k2_1 > 0) & (k2_2 > 0),
            np.real(amplitude * np.exp(1j * h / 2 * (k_1 + k_2))),
            amplitude * np.exp(-h / 2 * (k_1 + k_2))
        )

    return wkb_seed
