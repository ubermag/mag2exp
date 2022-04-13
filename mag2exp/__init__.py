"""Simulate experimental measurements."""
import importlib

import pytest

from . import ltem, magnetometry, mfm, moke, quick_plots, sans, util, x_ray

__version__ = importlib.metadata.version(__name__)


def test():
    """Run all package tests.

    Examples
    --------
    1. Run all tests.

    >>> import mag2exp
    ...
    >>> # mag2exp.test()

    """
    return pytest.main(["-v", "--pyargs", "mag2exp", "-l"])  # pragma: no cover
