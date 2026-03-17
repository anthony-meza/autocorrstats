"""
Plotting utilities for cross-correlation analysis.

This module provides visualization functions for cross-correlation results,
including significance testing and confidence intervals.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import xarray as xr
from scipy.stats import norm


def plot_significant_correlations(
    ax,
    lags,
    ccf,
    pvals,
    significance_level: float = 0.05,
    sig_color: str = 'r',
    nonsig_color: str = 'k'
):
    """
    Scatter plot of cross-correlation coefficients, coloring points by significance.

    Points are colored red (or `sig_color`) if their p-value is below `significance_level`,
    and black (or `nonsig_color`) otherwise.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    lags : array-like
        Lag values.
    ccf : array-like
        Cross-correlation coefficients at each lag.
    pvals : array-like
        P-values for each lag.
    significance_level : float, default 0.05
        Significance threshold for p-values.
    sig_color : str, default 'r'
        Color for significant points.
    nonsig_color : str, default 'k'
        Color for non-significant points.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object with the plot.
    """
    sig_mask = pvals < significance_level
    nonsig_mask = ~sig_mask

    # Plot line connecting all points
    ax.plot(lags, ccf, c=nonsig_color, zorder=0)

    # Plot non-significant points
    ax.scatter(
        lags[nonsig_mask],
        ccf[nonsig_mask],
        c=nonsig_color,
        marker='o',
        label='Not Significant',
        zorder=2
    )

    # Plot significant points
    ax.scatter(
        lags[sig_mask],
        ccf[sig_mask],
        c=sig_color,
        marker='o',
        label='Significant',
        zorder=3
    )

    ax.set_xlabel('Lag')
    ax.set_ylabel('Correlation Coefficient')
    ax.legend()

    return ax


def plot_conf_intervals(
    ax,
    lags,
    ccf_ensemble,
    significance_level: float = 0.05,
    color: Optional[str] = None,
    label: Optional[str] = None
):
    """
    Plot confidence intervals from bootstrap ensemble of cross-correlations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    lags : array-like
        Lag values.
    ccf_ensemble : xr.DataArray
        Bootstrap ensemble of cross-correlation coefficients.
        Should have dimensions (bootstrap_iter, lag).
    significance_level : float, default 0.05
        Significance level for confidence intervals.
        Will plot (1-significance_level)*100% confidence intervals.
    color : str, optional
        Color for the confidence interval fill. Default is matplotlib default.
    label : str, optional
        Label for the confidence interval. If None, auto-generates from significance_level.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object with the plot.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import xarray as xr
    >>> from SpectralCorr import cross_correlation
    >>> from SpectralCorr.plotting import plot_conf_intervals
    >>>
    >>> # Create sample data
    >>> ts1 = xr.DataArray(np.random.randn(100), dims=['time'])
    >>> ts2 = xr.DataArray(np.random.randn(100), dims=['time'])
    >>>
    >>> # Compute cross-correlation with bootstrap distributions
    >>> result = cross_correlation(
    ...     ts1, ts2,
    ...     method='ebisuzaki',
    ...     return_distributions=True
    ... )
    >>>
    >>> # Plot confidence intervals
    >>> fig, ax = plt.subplots()
    >>> plot_conf_intervals(
    ...     ax,
    ...     result['lag'].values,
    ...     result['cross_correlation_distribution']
    ... )
    >>> plt.show()
    """
    if label is None:
        limits = str(round(100 * (1 - significance_level)))
        label = f"Bootstrapped {limits}% Confidence Limits"

    # Calculate confidence bounds
    lb = ccf_ensemble.quantile(dim="bootstrap_iter", q=significance_level/2)
    ub = ccf_ensemble.quantile(dim="bootstrap_iter", q=1-significance_level/2)

    # Plot fill
    fill_kwargs = {'alpha': 0.45, 'label': label}
    if color is not None:
        fill_kwargs['color'] = color

    ax.fill_between(lags, lb, ub, **fill_kwargs)
    ax.legend()

    return ax


def plot_pearson_conf_intervals(
    ax,
    lags,
    ci_lower,
    ci_upper,
    significance_level: float = 0.05,
    color: Optional[str] = None,
    label: Optional[str] = None
):
    """
    Plot Pearson confidence intervals for a cross-correlation function.
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    lags : array-like
        Lag values in time units.
    ci_lower : array-like
        Lower bound of the confidence interval.
    ci_upper : array-like
        Upper bound of the confidence interval.
    significance_level : float, default 0.05
        Significance level for labeling.
    color : str, optional
        Color for the confidence interval fill.
    label : str, optional
        Label for the confidence interval.
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object with the plot.
    """
    if label is None:
        limits = str(round(100 * (1 - significance_level)))
        label = f"Pearson {limits}% Confidence Limits"

    fill_kwargs = {'alpha': 0.45, 'label': label}
    if color is not None:
        fill_kwargs['color'] = color
    
    ax.fill_between(lags, ci_lower, ci_upper, **fill_kwargs)
    ax.legend()
    return ax


def plot_cross_correlation(
    result: xr.Dataset,
    significance_level: float = 0.05,
    show_significance: bool = True,
    show_confidence: bool = False,
    show_pearson_confidence: bool = False,
    figsize: tuple = (10, 6),
    ax = None,
    **kwargs
):
    """
    Convenience function to plot cross-correlation results.

    Parameters
    ----------
    result : xr.Dataset
        Output from `cross_correlation()` function containing:
        - 'lag': lag values
        - 'cross_correlation': correlation coefficients
        - 'cross_correlation_pvalue': p-values (if show_significance=True)
        - 'cross_correlation_distribution': bootstrap distributions (if show_confidence=True)
        - 'pearson_ci_lower', 'pearson_ci_upper': Pearson CI (if show_pearson_confidence=True)
    significance_level : float, default 0.05
        Significance threshold.
    show_significance : bool, default True
        If True, color points by significance using p-values.
    show_confidence : bool, default False
        If True, show confidence intervals from bootstrap distributions.
        Requires 'cross_correlation_distribution' in result.
    show_pearson_confidence : bool, default False
        If True, show confidence intervals from Pearson's formula.
        Requires 'pearson_ci_lower' and 'pearson_ci_upper' in result.
    figsize : tuple, default (10, 6)
        Figure size (width, height) in inches. Ignored if ax is provided.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates new figure and axes.
    **kwargs
        Additional keyword arguments passed to plotting functions.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure object. None if ax was provided.
    ax : matplotlib.axes.Axes
        The axes object.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> from SpectralCorr import cross_correlation
    >>> from SpectralCorr.plotting import plot_cross_correlation
    >>>
    >>> # Create sample data
    >>> ts1 = xr.DataArray(np.random.randn(100), dims=['time'])
    >>> ts2 = xr.DataArray(np.random.randn(100), dims=['time'])
    >>>
    >>> # Compute and plot
    >>> result = cross_correlation(ts1, ts2, method='ebisuzaki')
    >>> fig, ax = plot_cross_correlation(result)
    >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    lags = result['lag'].values
    ccf = result['cross_correlation'].values

    # Plot bootstrap confidence intervals
    if show_confidence:
        if 'cross_correlation_distribution' not in result:
            raise ValueError(
                "Cannot show bootstrap confidence intervals: 'cross_correlation_distribution' "
                "not found in result. Use return_distributions=True when calling "
                "cross_correlation()."
            )
        plot_conf_intervals(
            ax,
            lags,
            result['cross_correlation_distribution'],
            significance_level=significance_level,
            **kwargs.get('conf_kwargs', {})
        )

    # Plot Pearson confidence intervals
    if show_pearson_confidence:
        if 'pearson_ci_lower' not in result or 'pearson_ci_upper' not in result:
             raise ValueError(
                "Cannot show Pearson confidence intervals: 'pearson_ci_lower' or 'pearson_ci_upper' "
                "not found in result. Ensure method is 'pearson'."
            )
        
        plot_pearson_conf_intervals(
            ax,
            lags,
            result['pearson_ci_lower'].values,
            result['pearson_ci_upper'].values,
            significance_level=significance_level,
            **kwargs.get('pearson_conf_kwargs', {})
        )

    # Plot correlations with significance coloring
    if show_significance:
        if 'cross_correlation_pvalue' not in result:
            raise ValueError(
                "Cannot show significance: 'cross_correlation_pvalue' "
                "not found in result."
            )
        plot_significant_correlations(
            ax,
            lags,
            ccf,
            result['cross_correlation_pvalue'].values,
            significance_level=significance_level,
            **kwargs.get('sig_kwargs', {})
        )
    else:
        # Just plot the correlation line
        ax.plot(lags, ccf, c='k')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Correlation Coefficient')

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, zorder=0)
    ax.grid(True, alpha=0.3)

    return fig, ax
