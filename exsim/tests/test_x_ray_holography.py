# import pytest
import discretisedfield as df
import numpy as np
import exsim


def test_xray_holographic_image_inplane():
    mesh = df.Mesh(p1=(-5e-9, -4e-9, -2e-9), p2=(5e-9, 4e-9, 6e-9),
                   cell=(2e-9, 1e-9, 2e-9))

    field = df.Field(mesh, dim=3, value=(1, 1, 0), norm=384e3)
    xrh = exsim.x_ray_holography.holographic_image(field)
    assert (xrh.array == 0).all()


def test_xray_holographic_image_outofplane():
    mesh = df.Mesh(p1=(-5e-9, -4e-9, -2e-9), p2=(5e-9, 4e-9, 6e-9),
                   cell=(2e-9, 1e-9, 2e-9))

    field = df.Field(mesh, dim=3, value=(0, 0, 1), norm=384e3)
    xrh = exsim.x_ray_holography.holographic_image(field)
    assert (xrh.array != 0).all()


def test_xray_holographic_image_filter():
    mesh = df.Mesh(p1=(-5e-9, -4e-9, -2e-9), p2=(5e-9, 4e-9, 6e-9),
                   cell=(2e-9, 1e-9, 2e-9))

    field = df.Field(mesh, dim=3, value=(0, 0, 1), norm=384e3)
    xrh = exsim.x_ray_holography.holographic_image(field, 2e-9)
    assert (xrh.array != 0).all()


def test_xray_holographic_scattering():
    mesh = df.Mesh(p1=(-5e-9, -4e-9, -2e-9), p2=(5e-9, 4e-9, 6e-9),
                   cell=(2e-9, 1e-9, 2e-9))

    field = df.Field(mesh, dim=3, value=(0, 0, 1), norm=384e3)
    xrh = exsim.x_ray_holography.holographic_scattering(field)
    assert (xrh.array != 0).any()
    assert (np.isreal(xrh.array)).all()


def test_xray_holographic_scattering_inplane():
    mesh = df.Mesh(p1=(-5e-9, -4e-9, -2e-9), p2=(5e-9, 4e-9, 6e-9),
                   cell=(2e-9, 1e-9, 2e-9))

    field = df.Field(mesh, dim=3, value=(1, 1, 0), norm=384e3)
    xrh = exsim.x_ray_holography.holographic_scattering(field)
    assert (xrh.array == 0).all()
