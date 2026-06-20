import pytest

import cfdverify.utils as utils


def test_mesh_size():
    assert utils.mesh_size(100, 10, 3) == pytest.approx(10**(1/3))
    assert utils.mesh_size(100, 10, 2) == pytest.approx(10**(1/2))
    assert utils.mesh_size(1000, 10, 2) == pytest.approx(100**(1/2))