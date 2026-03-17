"""
Correlation and phase scrambling utilities for time series analysis.
"""

import numpy as np
import xarray as xr
from scipy.stats import pearsonr, norm
from statsmodels.distributions.empirical_distribution import ECDF
from .fft_bootstrap import *
from .timeseries import TimeSeries
from .validation import validate_dataarray_inputs
from tqdm import trange



def cross_correlation(ts1, ts2, maxlags=None, method='pearson',
                     n_iter=5000, return_distributions=False, detrend=True,
                     significance_level=0.05):
    """
    Compute cross-correlation and p-values between two time series.

    Calculates correlation coefficients at different time lags using either
    Gaussian assumptions (Pearson) or phase scrambling bootstrap (Ebisuzaki).

    Parameters
    ----------
    ts1, ts2 : xarray.DataArray
        Input time series with matching time steps. If xarray.DataArray,
        must have 'time' coordinate.
    maxlags : int, optional
        Maximum lag to compute (in time steps). If None, uses full range (N-1).
    method : {'pearson', 'ebisuzaki'}, default 'pearson'
        Statistical method for significance testing:
        - 'pearson': Assumes Gaussian distributions for p-values
        - 'ebisuzaki': Uses phase scrambling bootstrap for robust testing
    n_iter : int, default 1000
        Number of bootstrap iterations (only used when method='ebisuzaki').
    return_distributions : bool, default False
        If True, return full bootstrap distributions (only used when method='ebisuzaki').
    detrend : bool, default True
        If True, detrend signals before scrambling (only used when method='ebisuzaki').
    significance_level : float, default 0.05
        Significance level for confidence intervals (only used for method='pearson').

    Returns
    -------
    xr.Dataset
        Dataset containing:
            - lag: lag values in time units
            - cross_correlation: correlation coefficients at each lag
            - cross_correlation_pvalue: p-values for each lag
            - pearson_ci_lower: lower bound of Pearson confidence interval
            - pearson_ci_upper: upper bound of Pearson confidence interval
            - cross_correlation_distribution: bootstrap distributions (if requested and method='ebisuzaki')

    Raises
    ------
    ValueError
        If time series have mismatched time steps, lengths, or invalid method specified.

    Notes
    -----
    Positive lags indicate ts2 leads ts1, negative lags indicate ts1 leads ts2.

    Method comparison:
    - 'pearson': Fast but assumes Gaussian distributions. May give incorrect
      p-values for autocorrelated time series.
    - 'ebisuzaki': Slower but robust for autocorrelated data. Uses phase
      scrambling to preserve power spectrum while destroying correlations.

    References
    ----------
    Ebisuzaki, W. (1997). A method to estimate the statistical significance of a
    correlation when the data are serially correlated. Journal of Climate, 10(9),
    2147-2153.
    """
    # Validate method parameter
    if method not in ['pearson', 'ebisuzaki']:
        raise ValueError(f"Method must be 'pearson' or 'ebisuzaki', got '{method}'")

    # Validate inputs (only accept xarray DataArrays)
    ts1_xr, ts2_xr = validate_dataarray_inputs(ts1, ts2, "cross_correlation")
    # Convert to TimeSeries for backend computation
    ts1_ts = TimeSeries.from_xarray(ts1_xr)
    ts2_ts = TimeSeries.from_xarray(ts2_xr)

    # Delegate to appropriate method
    if method == 'pearson':
        return _cross_correlation_pearson(ts1_ts, ts2_ts, maxlags, significance_level=significance_level)
    else:  # method == 'ebisuzaki'
        return _cross_correlation_ebisuzaki(ts1_ts, ts2_ts, maxlags, n_iter, return_distributions, detrend)
def _cross_correlation_pearson(ts1: TimeSeries, ts2: TimeSeries, maxlags=None, significance_level=0.05):
    """Helper function for Pearson cross-correlation."""
    x = np.asarray(ts1.data)
    y = np.asarray(ts2.data)
    dt = ts1.dt
    n = x.size
    if maxlags is None:
        maxlags = n - 1
    maxlags = min(maxlags, n - 1)
    lags = np.arange(-maxlags, maxlags+1) * dt
    ccf = np.zeros(len(lags))
    pvals = np.zeros(len(lags))
    ci_low = np.zeros(len(lags))
    ci_high = np.zeros(len(lags))
    
    z_crit = norm.ppf(1 - significance_level / 2)

    for i, k in enumerate(range(-maxlags, maxlags+1)):
        if k < 0:
            xi = x[:n+k]
            yi = y[-k:]
        else:
            xi = x[k:]
            yi = y[:n-k]

        # Skip correlation if insufficient data points
        if len(xi) < 2 or len(yi) < 2:
            ccf[i] = np.nan
            pvals[i] = np.nan
            ci_low[i] = np.nan
            ci_high[i] = np.nan
        else:
            ts_xi = TimeSeries(ts1.time[k:] if k >= 0 else ts1.time[:n+k], xi, ts1.dt)
            ts_yi = TimeSeries(ts2.time[-k:] if k < 0 else ts2.time[:n-k], yi, ts2.dt)
            r, pval = _correlation_pearson(ts_xi, ts_yi)
            ccf[i] = r
            pvals[i] = pval
            
            # Calculate confidence intervals
            n_eff = len(xi)
            if np.isnan(r) or n_eff < 4:
                ci_low[i] = np.nan
                ci_high[i] = np.nan
            else:
                r_clipped = np.clip(r, -1 + 1e-12, 1 - 1e-12)
                z = np.arctanh(r_clipped)
                se = 1 / np.sqrt(n_eff - 3)
                z_low = z - z_crit * se
                z_high = z + z_crit * se
                ci_low[i] = np.tanh(z_low)
                ci_high[i] = np.tanh(z_high)

    ccf_da = xr.DataArray(
        ccf,
        coords={"lag": lags},
        dims=["lag"],
        name="cross_correlation",
        attrs={"description": "Cross-correlation coefficient at each lag"}
    )
    pvals_da = xr.DataArray(
        pvals,
        coords={"lag": lags},
        dims=["lag"],
        name="cross_correlation_pvalue",
        attrs={"description": "Pearson p-value for cross-correlation at each lag"}
    )
    lag_da = xr.DataArray(
        lags,
        dims=["lag"],
        name="lag",
        attrs={"description": "Lag values"}
    )
    ci_low_da = xr.DataArray(
        ci_low,
        coords={"lag": lags},
        dims=["lag"],
        name="pearson_ci_lower",
        attrs={"description": f"Lower bound of {100*(1-significance_level):.0f}% confidence interval"}
    )
    ci_high_da = xr.DataArray(
        ci_high,
        coords={"lag": lags},
        dims=["lag"],
        name="pearson_ci_upper",
        attrs={"description": f"Upper bound of {100*(1-significance_level):.0f}% confidence interval"}
    )

    return xr.Dataset(
        {
            "lag": lag_da,
            "cross_correlation": ccf_da,
            "cross_correlation_pvalue": pvals_da,
            "pearson_ci_lower": ci_low_da,
            "pearson_ci_upper": ci_high_da,
        },
        attrs={
            "description": "Cross-correlation and p-values using Pearson method",
            "n_samples": n,
            "dt": dt,
        }
    )

def _cross_correlation_ebisuzaki(ts1: TimeSeries, ts2: TimeSeries, maxlags=None,
                                n_iter=1000, return_distributions=False, detrend=True):
    """Helper function for Ebisuzaki (phase scrambling) cross-correlation."""
    x = np.asarray(ts1.data)
    y = np.asarray(ts2.data)
    n = x.size
    dt = ts1.dt

    if maxlags is None:
        maxlags = n - 1
    maxlags = min(maxlags, n - 1)
    lags = np.arange(-maxlags, maxlags+1) * dt
    ccf = np.zeros(len(lags))
    ccf_pval = np.zeros(len(lags))
    if return_distributions:
        ccf_dist = np.zeros((n_iter, len(lags)))

    for i, k in enumerate(trange(-maxlags, maxlags+1, desc="Computing cross-correlation")):
        if k < 0:
            xi = x[:n+k]
            yi = y[-k:]
            ti = ts1.time[:n+k]
        else:
            xi = x[k:]
            yi = y[:n-k]
            ti = ts1.time[k:]
        ts_xi = TimeSeries(ti, xi, ts1.dt)
        ts_yi = TimeSeries(ti, yi, ts2.dt)
        ccf[i], ccf_pval[i] = _correlation_ebisuzaki(ts_xi, ts_yi, n_iter=n_iter, detrend=detrend)
        if return_distributions:
            # Need to get the full bootstrap distribution
            _, _, corrs = bootstrap_correlation(ts_xi, ts_yi, n_iter=n_iter, detrend=detrend)
        if return_distributions:
            ccf_dist[:, i] = corrs

    ccf_da = xr.DataArray(
        ccf,
        coords={"lag": lags},
        dims=["lag"],
        name="cross_correlation",
        attrs={"description": "Cross-correlation coefficient at each lag"}
    )
    ccf_pval_da = xr.DataArray(
        ccf_pval,
        coords={"lag": lags},
        dims=["lag"],
        name="cross_correlation_pvalue",
        attrs={"description": "Ebisuzaki bootstrap p-values for cross-correlation at each lag"}
    )
    lag_da = xr.DataArray(
        lags,
        dims=["lag"],
        name="lag",
        attrs={"description": "Lag values"}
    )

    data_vars = {
        "lag": lag_da,
        "cross_correlation": ccf_da,
        "cross_correlation_pvalue": ccf_pval_da
    }

    if return_distributions:
        ccf_dist_da = xr.DataArray(
            ccf_dist,
            coords={"bootstrap_iter": np.arange(n_iter), "lag": lags},
            dims=["bootstrap_iter", "lag"],
            name="cross_correlation_distribution",
            attrs={"description": "Bootstrap distributions for cross-correlation at each lag"}
        )
        data_vars["cross_correlation_distribution"] = ccf_dist_da

    return xr.Dataset(
        data_vars,
        attrs={
            "description": "Cross-correlation and p-values using Ebisuzaki method",
            "n_samples": n,
            "dt": dt,
        }
    )

def maximum_cross_correlation(ts1, ts2, maxlags=None, method='pearson'):
    """
    Find the lag and value of maximum cross-correlation.

    Parameters
    ----------
    ts1, ts2 : xarray.DataArray
        Input time series. If xarray DataArrays are provided,
        they will be converted to TimeSeries objects.
    maxlags : int, optional
        Maximum lag to compute.
    method : {'pearson', 'ebisuzaki'}, default 'pearson'
        Statistical method to use for cross-correlation computation.

    Returns
    -------
    tuple
        (lag_max, ccf_max):
            - lag_max: lag at which maximum correlation occurs
            - ccf_max: maximum correlation value

    Notes
    -----
    This function computes cross-correlation for all lags and returns
    the lag and value where correlation is maximized.
    """
    # Convert inputs to TimeSeries if needed (cross_correlation handles this)
    ds = cross_correlation(ts1, ts2, maxlags, method=method)
    idx = np.argmax(ds["cross_correlation"].values)
    lag_max = ds["lag"].values[idx]
    ccf_max = ds["cross_correlation"].values[idx]

    return lag_max, ccf_max

def bootstrap_correlation(ts1: TimeSeries, ts2: TimeSeries, n_iter=1000, detrend=True):
    """
    Compute bootstrapped correlation and p-value using phase scrambling.

    P-values are calculated using the empirical cumulative distribution function (ECDF)
    and a two-sided test comparing the observed correlation against the bootstrapped
    null distribution.

    Parameters
    ----------
    ts1, ts2 : TimeSeries
        Input time series to correlate.
    n_iter : int, default 1000
        Number of bootstrap iterations for building null distribution.
    detrend : bool, default True
        If True, detrend the signals before phase scrambling.

    Returns
    -------
    tuple
        (ref_corr, boot_p_value, bootstrapped_correlations):
            - ref_corr: observed correlation coefficient (float)
            - boot_p_value: bootstrapped p-value for two-sided test (float)
            - bootstrapped_correlations: null distribution values (np.ndarray)

    Notes
    -----
    The phase scrambling procedure preserves the power spectrum of each time series
    while randomizing phases, creating surrogates that maintain the same spectral
    properties but destroy any correlation structure. This follows the method
    described in Ebisuzaki (1997).
    """
    if ts1.dt != ts2.dt:
        raise ValueError(f"TimeSeries dt mismatch: {ts1.dt} vs {ts2.dt}")

    ref_corr = pearsonr(ts1.data, ts2.data)[0]
    xs_surrogates = phase_scrambled(ts1, detrend=detrend, n_scrambled=n_iter)
    ys_surrogates = phase_scrambled(ts2, detrend=detrend, n_scrambled=n_iter)
    # Compute correlations between surrogate pairs
    corrs = pearsonr(xs_surrogates, ys_surrogates, axis=1)[0]
    ecdf = ECDF(corrs)
    a = np.abs(ref_corr)
    boot_p_value = ecdf(-a) + (1.0 - ecdf(a)) #two-sided t-test
    return ref_corr, boot_p_value, corrs

def _correlation_pearson(ts1: TimeSeries, ts2: TimeSeries):
    """
    Backend function for Pearson correlation between two time series.

    Parameters
    ----------
    ts1, ts2 : TimeSeries
        Input time series to correlate.

    Returns
    -------
    tuple
        (correlation, p_value):
            - correlation: Pearson correlation coefficient (float)
            - p_value: two-sided p-value for the correlation (float)
    """
    if ts1.dt != ts2.dt:
        raise ValueError(f"TimeSeries dt mismatch: {ts1.dt} vs {ts2.dt}")

    if len(ts1.data) != len(ts2.data):
        raise ValueError(f"TimeSeries length mismatch: {len(ts1.data)} vs {len(ts2.data)}")

    if len(ts1.data) < 2:
        return np.nan, np.nan

    correlation, p_value = pearsonr(ts1.data, ts2.data)
    return correlation, p_value

def _correlation_ebisuzaki(ts1: TimeSeries, ts2: TimeSeries, n_iter=1000, detrend=True):
    """
    Backend function for Ebisuzaki correlation using phase scrambling.

    This is essentially a wrapper around bootstrap_correlation with consistent naming.

    Parameters
    ----------
    ts1, ts2 : TimeSeries
        Input time series to correlate.
    n_iter : int, default 1000
        Number of bootstrap iterations for building null distribution.
    detrend : bool, default True
        If True, detrend the signals before phase scrambling.

    Returns
    -------
    tuple
        (correlation, p_value):
            - correlation: observed correlation coefficient (float)
            - p_value: bootstrapped p-value for two-sided test (float)
    """
    correlation, p_value, _ = bootstrap_correlation(ts1, ts2, n_iter=n_iter, detrend=detrend)
    return correlation, p_value
