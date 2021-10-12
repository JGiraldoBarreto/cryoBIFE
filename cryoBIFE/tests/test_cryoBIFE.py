"""
Unit and regression test for the cryoBIFE package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import cryoBIFE


def test_cryoBIFE_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "cryoBIFE" in sys.modules
