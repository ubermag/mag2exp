import discretisedfield as df
import numpy as np
import pytest

import mag2exp


@pytest.mark.filterwarnings("ignore:This technique is currently under development")
def test_moke_e_field():
    mesh = df.Mesh(
        p1=(-5e-9, -4e-9, -2e-9), p2=(5e-9, 4e-9, 6e-9), cell=(2e-9, 1e-9, 2e-9)
    )

    field = df.Field(mesh, nvdim=3, value=(1, 1, 0), norm=384e3)
    E_i = [1, 1j]
    E_f = mag2exp.moke.e_field(field, 0, 2, 1, 600e-9, E_i, mode="reflection")
    assert E_f.nvdim == 2

    E_f = mag2exp.moke.e_field(field, 0, 2, 1, 600e-9, E_i, mode="transmission")
    assert E_f.nvdim == 2

    with pytest.raises(ValueError):
        mag2exp.moke.e_field(field, 0, 2, 1, 600e-9, E_i, mode="blah")


@pytest.mark.filterwarnings("ignore:This technique is currently under development")
def test_moke_intensity():
    mesh = df.Mesh(
        p1=(-5e-9, -4e-9, -2e-9), p2=(5e-9, 4e-9, 6e-9), cell=(2e-9, 1e-9, 2e-9)
    )

    field = df.Field(mesh, nvdim=3, value=(0, 0, 1), norm=384e3)
    E_i = [1, 1j]
    intensity = mag2exp.moke.intensity(field, 0, 1, 0, 600e-9, E_i, mode="reflection")

    assert np.allclose(intensity.array, 0)

    field = df.Field(mesh, nvdim=3, value=(1, 1, 0), norm=384e3)
    E_i = [1, 0]
    intensity = mag2exp.moke.intensity(field, 0, 1, 0, 600e-9, E_i, mode="reflection")

    assert np.allclose(intensity.array, 0)

    mag2exp.moke.intensity(field, 0, 1, 0, 600e-9, E_i, mode="transmission")


@pytest.mark.filterwarnings("ignore:This technique is currently under development")
def test_moke_angle():
    mesh = df.Mesh(
        p1=(-5e-9, -4e-9, -2e-9), p2=(5e-9, 4e-9, 6e-9), cell=(2e-9, 1e-9, 2e-9)
    )

    field = df.Field(mesh, nvdim=3, value=(0, 0, 1), norm=384e3)
    angle = mag2exp.moke.kerr_angle(field, 0, 2, 1, 600e-9)

    assert angle.array.all() != 0


@pytest.mark.filterwarnings("ignore:This technique is currently under development")
def test_moke_fwhm():
    mesh = df.Mesh(
        p1=(-5e-9, -4e-9, -2e-9), p2=(5e-9, 4e-9, 6e-9), cell=(2e-9, 1e-9, 2e-9)
    )

    field = df.Field(mesh, nvdim=3, value=(0, 0, 1), norm=384e3)
    E_i = [1, 1j]
    mag2exp.moke.intensity(
        field, 0, 1, 0, 600e-9, E_i, mode="reflection", fwhm=(1e-9, 1e-9)
    )
