"""
Phase scrambling utilities for generating surrogate time series.

This module implements the phase randomization method for creating
surrogate time series that preserve the power spectrum while destroying
temporal correlations. Used for significance testing in time series analysis.
"""

import numpy as np
import xarray as xr
from scipy.stats import pearsonr
from statsmodels.distributions.empirical_distribution import ECDF
from .fft_utils import real_fft
from .timeseries import TimeSeries

import numpy as np
import xarray as xr
import xrft

def phase_scrambled_surrogates(
    ts: xr.DataArray,
    detrend: str | None = "constant",
    n_scrambled: int = 1,
    time_dim: str = "time",
):
    """
    Generate phase-scrambled surrogates of an xarray DataArray using xrft.

    Phase scrambling is applied independently along `time_dim` for every slice
    of the remaining dimensions.

    Parameters
    ----------
    ts : xarray.DataArray
        Input data containing `time_dim`.
    detrend : {"constant", "linear", None}, default "constant"
        Detrending option passed to xrft.fft.
    n_scrambled : int, default 1
        Number of surrogate series to generate.
    time_dim : str, default "time"
        Name of the dimension along which to phase scramble.

    Returns
    -------
    xarray.DataArray
        Phase-scrambled surrogate(s). A new dimension `"surrogate"` is added
        when `n_scrambled > 1`.
    """
    if not isinstance(ts, xr.DataArray):
        raise TypeError("ts must be an xarray.DataArray")
    if time_dim not in ts.dims:
        raise ValueError(f"{time_dim!r} must be a dimension of ts")
    if detrend not in {"constant", "linear", None}:
        raise ValueError("detrend must be 'constant', 'linear', or None")

    nt = ts.sizes[time_dim]
    freq_dim = f"freq_{time_dim}"

    F = xrft.fft(
        ts,
        dim=[time_dim], #apply fft along time dimension
        spacing_tol=1e-8,
        real_dim=time_dim, #make fft real along time dimension too! 
        shift=True,
        true_phase=True,
        true_amplitude=True,
        detrend=detrend,
    )

    freq = F[freq_dim]

    dc = freq == 0  # zero-frequency (mean) component

    # For even-length signals, the Nyquist bin is the largest-magnitude frequency.
    # That coefficient must remain purely real, so handle it separately.
    nyq = np.abs(freq) == np.abs(freq).max() if nt % 2 == 0 else xr.zeros_like(freq, dtype=bool)

    # All nonzero, non-Nyquist frequencies can be phase-randomized freely.
    interior = ~(dc | nyq)

    other_dims = [d for d in F.dims if d != freq_dim]
    phase_dims = ("surrogate", *other_dims, freq_dim)
    phase_coords = {"surrogate": np.arange(n_scrambled), **{d: F[d] for d in other_dims}, freq_dim: freq}
    phase_shape = (n_scrambled, *[F.sizes[d] for d in other_dims], F.sizes[freq_dim])

    phase = xr.DataArray(
        np.random.uniform(0.0, 2.0 * np.pi, size=phase_shape),
        dims=phase_dims,
        coords=phase_coords,
    )

    amp = np.abs(F).expand_dims(surrogate=phase["surrogate"])

    # Randomize phase only for frequencies that can take arbitrary complex phase.
    # Their amplitudes are preserved, so the power spectrum is unchanged.
    Fp = xr.where(interior, amp * np.exp(1j * phase), amp)

    # Remove the zero-frequency component so each surrogate has zero mean.
    Fp = xr.where(dc, 0.0 + 0.0j, Fp)

    # For even-length signals, the Nyquist coefficient must be purely real.
    # Use the Ebisuzaki-style real-valued randomization that preserves Nyquist
    # power in expectation across surrogates.
    Fp = xr.where(nyq, amp * np.cos(phase) * np.sqrt(2.0), Fp)

    out = xrft.ifft(
        Fp,
        dim=[freq_dim],
        true_phase=True,
        true_amplitude=True,
        real_dim=freq_dim,
    )
    out = np.real(out)

    out = out.assign_coords({"surrogate": np.arange(n_scrambled), time_dim: ts[time_dim]})
    out = out.rename("phase_scrambled_surrogates")

    out.attrs = {
        "description": f"{n_scrambled} phase-scrambled surrogate time series",
        "original_detrend": detrend,
        **({"dt": ts.attrs["dt"]} if "dt" in ts.attrs else {}),
    }

    if n_scrambled == 1:
        out = out.squeeze("surrogate", drop=True)

    return out
    
def phase_scrambled(ts: TimeSeries, detrend=True, n_scrambled=1, return_xarray=False):
    """
    Generate phase-scrambled surrogates of a time series.

    Creates surrogate time series by randomizing the phases of Fourier components
    while preserving the power spectrum. This destroys temporal correlations
    while maintaining spectral properties, enabling robust significance testing.

    Parameters
    ----------
    ts : TimeSeries
        Input time series to scramble.
    detrend : bool, default True
        If True, detrend the signal before phase scrambling.
    n_scrambled : int, default 1
        Number of surrogate series to generate.
    return_xarray : bool, default False
        If True, return as xarray.DataArray with proper coordinates.

    Returns
    -------
    np.ndarray or xarray.DataArray
        Phase-scrambled surrogate(s). Shape is (n_scrambled, len(time)) when
        n_scrambled > 1, otherwise (len(time),).

    Notes
    -----
    The phase scrambling algorithm:
    1. Computes FFT of the input signal
    2. Randomizes phases while preserving magnitudes
    3. Applies inverse FFT to get surrogate time series
    4. Handles even/odd length signals appropriately

    This method follows Ebisuzaki (1997) and is widely used in climate
    and geophysical time series analysis for creating null distributions.

    References
    ----------
    Ebisuzaki, W. (1997). A method to estimate the statistical significance of a
    correlation when the data are serially correlated. Journal of Climate, 10(9),
    2147-2153.
    """
    signal = np.asarray(ts.data)
    nt = signal.size
    F_da = real_fft(ts, detrend=detrend)
    freqs = F_da.coords["freq"].values
    
    nf = freqs.size

    F = F_da.values.copy()
    surrogates = np.empty((n_scrambled, nt), dtype=float)
    for i in range(n_scrambled):
        F_copy = F.copy()
        phases = np.random.uniform(0, 2*np.pi, size=nf)
        
        F_copy[0] = 0 * F_copy[0]  # Zero out the mean (DC component)
        F_copy[1:nf-1] = np.abs(F_copy[1:nf-1]) * np.exp(1j * phases[1:nf-1])
        if nt % 2 == 0 and nf > 1:
            F_copy[-1] = np.abs(F_copy[-1]) * np.cos(phases[-1]) * np.sqrt(2)
        surrogates[i, :] = np.real(np.fft.irfft(F_copy, nt))

    if n_scrambled == 1:
        result = surrogates[0]
    else:
        result = surrogates

    if return_xarray:
        da = xr.DataArray(
            result,
            coords={"iter": np.arange(n_scrambled), "time": ts.time},
            dims=["iter", "time"],
            name="phase_scrambled",
            attrs={
                "description": f"{n_scrambled} phase-scrambled surrogate time series",
                "dt": ts.dt,
                "original_detrended": detrend
            }
        )
        if n_scrambled == 1:
            da = da.drop_vars("iter")
        return da
    else:
        return result