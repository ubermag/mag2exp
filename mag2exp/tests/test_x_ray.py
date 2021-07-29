# import pytest
import discretisedfield as df
import numpy as np
import mag2exp


def test_xray_holography_inplane():
    mesh = df.Mesh(p1=(-5e-9, -4e-9, -2e-9), p2=(5e-9, 4e-9, 6e-9),
                   cell=(2e-9, 1e-9, 2e-9))

    field = df.Field(mesh, dim=3, value=(1, 1, 0), norm=384e3)
    xrh = mag2exp.x_ray.holography(field)
    assert (xrh.array == 0).all()


def test_xray_holography_outofplane():
    mesh = df.Mesh(p1=(-5e-9, -4e-9, -2e-9), p2=(5e-9, 4e-9, 6e-9),
                   cell=(2e-9, 1e-9, 2e-9))

    field = df.Field(mesh, dim=3, value=(0, 0, 1), norm=384e3)
    xrh = mag2exp.x_ray.holography(field)
    assert (xrh.array != 0).all()


def test_xray_holography_filter():
    mesh = df.Mesh(p1=(-5e-9, -4e-9, -2e-9), p2=(5e-9, 4e-9, 6e-9),
                   cell=(2e-9, 1e-9, 2e-9))

    field = df.Field(mesh, dim=3, value=(0, 0, 1), norm=384e3)
    xrh = mag2exp.x_ray.holography(field, [2e-9, 2e-9])
    assert (xrh.array != 0).all()


def test_xray_saxs():
    mesh = df.Mesh(p1=(-5e-9, -4e-9, -2e-9), p2=(5e-9, 4e-9, 6e-9),
                   cell=(2e-9, 1e-9, 2e-9))

    field = df.Field(mesh, dim=3, value=(0, 0, 1), norm=384e3)
    xrh = mag2exp.x_ray.saxs(field)
    assert (xrh.array != 0).any()
    assert (np.isreal(xrh.array)).all()


def test_xray_saxs_inplane():
    mesh = df.Mesh(p1=(-5e-9, -4e-9, -2e-9), p2=(5e-9, 4e-9, 6e-9),
                   cell=(2e-9, 1e-9, 2e-9))

    field = df.Field(mesh, dim=3, value=(1, 1, 0), norm=384e3)
    xrh = mag2exp.x_ray.saxs(field)
    assert (xrh.array == 0).all()


def test_xray_holography_analytical():
    region = df.Region(p1=(-50e-9, -100e-9, 0), p2=(50e-9, 100e-9, 30e-9))
    mesh = df.Mesh(region=region, cell=(5e-9, 5e-9, 0.3e-9))
    Ms = 1.1e6

    def m_fun(pos):
        x, y, z = pos
        qx = 30e-9
        qz = 20e-9
        return [0, 0, Ms*np.cos(2*np.pi*x/qx)*np.sin(2*np.pi*z/qz)]
    m = df.Field(mesh, dim=3, value=m_fun)
    holo = mag2exp.x_ray.holography(m)

    def a_fun(pos):
        x, y, z = pos
        qx = 30e-9
        qz = 20e-9
        analytical = (- qz/(2*np.pi) * Ms *
                      np.cos(2*np.pi*x/qx) * (np.cos(2*np.pi*30e-9/qz)-1))
        return analytical

    an_holo = df.Field(holo.mesh, dim=1, value=a_fun).plane('z')
    assert np.isclose(holo.array, an_holo.array, rtol=1e-3).all()