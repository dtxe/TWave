# TWave: Wavelet convolutions on streamed EEG data for real-time phase estimation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15014410.svg)](https://doi.org/10.5281/zenodo.15014410)


## Usage
```python
from PhaseTracker import PhaseTracker
import numpy as np  # for sample data

# initialize phase tracker object
pt = PhaseTracker(fs=256)

# generate random sample data
newdata = np.random.rand(20)

# estimate oscillatory parameters
phase, freq, amp, quadrature = pt.estimate(newdata)
```
