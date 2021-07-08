from . import ltem
import pytest
import importlib


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
