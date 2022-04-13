import discretisedfield as df
import pytest

import mag2exp


def test_util_gaussian_filter_dim():
    fwhm = (1e-9, 1e-9, 1e-9)
    mesh = df.Mesh(
        p1=(-5e-9, -4e-9, -1e-9), p2=(5e-9, 4e-9, 1e-9), cell=(2e-9, 1e-9, 0.5e-9)
    )
    field = df.Field(mesh, dim=3, value=(0, 0, 1))
    with pytest.raises(RuntimeError):
        mag2exp.util.gaussian_filter(field, fwhm)
    new_field = mag2exp.util.gaussian_filter(field.z, fwhm)
    assert new_field.dim == 1


def test_util_gaussian_filter_plane():
    fwhm = (1e-9, 1e-9, 1e-9)
    mesh = df.Mesh(
        p1=(-5e-9, -4e-9, -1e-9), p2=(5e-9, 4e-9, 1e-9), cell=(2e-9, 1e-9, 0.5e-9)
    )
    field = df.Field(mesh, dim=3, value=(0, 0, 1))
    new_field = mag2exp.util.gaussian_filter(field.z.plane("z"), fwhm)
    assert new_field.mesh.attributes["isplane"]


def test_util_gaussian_filter_fwhm():
    mesh = df.Mesh(
        p1=(-5e-9, -4e-9, -1e-9), p2=(5e-9, 4e-9, 1e-9), cell=(2e-9, 1e-9, 0.5e-9)
    )
    field = df.Field(mesh, dim=3, value=(0, 0, 1))
    fwhm = (1e-9, 1e-9)
    mag2exp.util.gaussian_filter(field.z.plane("z"), fwhm)
    fwhm = (1e-9, 1e-9, 1e-9)
    mag2exp.util.gaussian_filter(field.z, fwhm)
    fwhm = (1e-9, 1e-9, 1e-9, 1e-9)
    mag2exp.util.gaussian_filter(field.z, fwhm)
