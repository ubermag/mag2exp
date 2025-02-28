"""Simulate experimental measurements."""

import importlib

import pytest

from . import fmr as fmr
from . import ltem as ltem
from . import magnetometry as magnetometry
from . import mfm as mfm
from . import moke as moke
from . import quick_plots as quick_plots
from . import sans as sans
from . import util as util
from . import x_ray as x_ray

__version__ = importlib.metadata.version(__package__)


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
