# SpectralCorr

<img width="989" height="396" alt="output" src="https://github.com/user-attachments/assets/e27730ac-2e9d-4d30-b918-cc7c63e14d92" />

Power-spectrum based correlation significance testing for autocorrelated time series. This Python package implements a non-parametric correlation test that utilizes randomly generated time series with the appropriate power spectra (Ebisuzaki, 1997). The Ebisuzaki correlation test can be applied here to test both lagged and non-lagged correlations. 


## Installation

We recommend installing SpectralCorr in a virtual environment to avoid dependency conflicts:

```bash
# Create and activate a virtual environment
$ python -m venv spectralcorr_env
$ source spectralcorr_env/bin/activate  # On macOS/Linux
$ spectralcorr_env\Scripts\activate     # On Windows

# Install SpectralCorr
$ pip install git+https://github.com/anthony-meza/SpectralCorr.git@main
```

Alternatively, you can install directly without a virtual environment (though this may cause dependency conflicts with other packages):

```bash
$ pip install git+https://github.com/anthony-meza/SpectralCorr.git@main
```

## Usage

Here's a quick example to get you started:

```python
import numpy as np
from SpectralCorr import AR1_process, cross_correlation, phase_scrambled_surrogates

# Generate two AR(1) time series
ts1 = AR1_process(rho=0.9, sigma=1.0, y0=0.0, N=500, seed=42)
ts2 = AR1_process(rho=0.8, sigma=1.0, y0=0.0, N=500, seed=123)

# Compute cross-correlation with Pearson method (no autocorrelation in timeseries)
result_pearson = cross_correlation(ts1, ts2, maxlags=50, method='pearson')

# Or use the Ebisuzaki method for robust significance testing for autocorrelated timeseries 
result_ebisuzaki = cross_correlation(ts1, ts2, maxlags=50, method='ebisuzaki', n_surrogates=1000)

# Results are returned as xarray Datasets
print(result_ebisuzaki.cross_correlation)
print(result_ebisuzaki.cross_correlation_pvalue)

# The surrogate generator is also available directly
surrogates = phase_scrambled_surrogates(ts1, n_surrogates=100)
print(surrogates)
```

For an example notebook, see `notebook_examples/AR1_lagged_example.ipynb`.

## License

`SpectralCorr` was created by Anthony Meza. It is licensed under the terms of the MIT license.

## References 

[Ebisuzaki, W. (1997). A method to estimate the statistical significance of a correlation when the data are serially correlated. Journal of Climate, 10(9), 2147–2153. https://doi.org/10.1175/1520-0442(1997)010&#60;2147:amtets&#62;2.0.co;2](https://doi.org/10.1175/1520-0442(1997)010%3C2147:AMTETS%3E2.0.CO;2)
