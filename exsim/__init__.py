import pytest
import importlib
from . import ltem
from . import quick_plots
from . import mfm
from . import util
from . import x_ray_holography



__version__ = importlib.metadata.version(__name__)


def test():
    """Run all package tests.

    Examples
    --------
    1. Run all tests.

    >>> import exsim
    ...
    >>> # exsim.test()

    """
    return pytest.main(['-v', '--pyargs',
                        'exsim', '-l'])  # pragma: no cover
