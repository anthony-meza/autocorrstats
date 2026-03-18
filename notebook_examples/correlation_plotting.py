"""Small plotting helpers for the example notebooks."""

import matplotlib.pyplot as plt


def plot_cross_correlation(
    result,
    *,
    ax=None,
    significance_level=0.05,
    show_significance=True,
    show_confidence=False,
    show_pearson_confidence=False,
):
    """Plot a cross-correlation result dataset on one axis."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    lag = result["lag"]
    corr = result["cross_correlation"]
    ax.plot(lag, corr, color="0.2", linewidth=1, zorder=1)

    if show_confidence and "cross_correlation_distribution" in result:
        dist = result["cross_correlation_distribution"]
        lower = dist.quantile(significance_level / 2, dim="bootstrap_iter")
        upper = dist.quantile(1 - significance_level / 2, dim="bootstrap_iter")
        ax.fill_between(lag, lower, upper, alpha=0.2, color="C0", label="Bootstrap CI")

    if show_pearson_confidence and "pearson_ci_lower" in result and "pearson_ci_upper" in result:
        ax.fill_between(
            lag,
            result["pearson_ci_lower"],
            result["pearson_ci_upper"],
            alpha=0.2,
            color="C0",
            label="Pearson CI",
        )

    if show_significance and "cross_correlation_pvalue" in result:
        sig = result["cross_correlation_pvalue"] < significance_level
        ax.scatter(lag.where(~sig), corr.where(~sig), color="0.2", s=18, zorder=2)
        ax.scatter(lag.where(sig), corr.where(sig), color="C3", s=18, zorder=3, label= "Significant correlation")
    else:
        ax.scatter(lag, corr, color="0.2", s=18, zorder=2)

    ax.axhline(0, color="0.6", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Correlation")
    ax.grid(alpha=0.25)
    return ax
