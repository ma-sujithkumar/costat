"""CoStat: distribution-aware quantisation of weights and activations.

This package implements the software half of the CoStat flow: profile a trained
network, fit a parametric distribution to every layer, and use that distribution
to place quantisation levels and piecewise-linear activation breakpoints where
the data actually concentrates.
"""

__version__ = "0.1.0"
