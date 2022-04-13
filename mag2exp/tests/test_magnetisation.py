# import pytest
import discretisedfield as df
import micromagneticmodel as mm
import numpy as np

import mag2exp


def test_magnetisation_analytical():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(6e-9, 4e-9, 1e-9), cell=(2e-9, 1e-9, 0.5e-9))
    field = df.Field(mesh, dim=3, value=(1, 1, 1))
    mag = mag2exp.magnetometry.magnetisation(field)
    assert np.isclose(mag, 1).all()


def test_torque_analytical_nodemag():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(6e-9, 4e-9, 1e-9), cell=(2e-9, 1e-9, 0.5e-9))
    field = df.Field(mesh, dim=3, value=(0, 0, 1))
    system = mm.System(name="Box2")
    system.energy = mm.Demag() + mm.Zeeman(H=(0, 1, 0))
    system.m = field
    torque = mag2exp.magnetometry.torque(system, use_demag=False)
    assert np.isclose(torque[0], -mm.consts.mu0)
    assert np.isclose(torque[1], 0)
    assert np.isclose(torque[2], 0)


def test_torque_analytical_parallel_nodemag():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(6e-9, 4e-9, 1e-9), cell=(2e-9, 1e-9, 0.5e-9))
    field = df.Field(mesh, dim=3, value=(0, 0, 1))
    system = mm.System(name="Box2")
    system.energy = mm.Demag() + mm.Zeeman(H=(0, 0, 1))
    system.m = field
    torque = mag2exp.magnetometry.torque(system, use_demag=False)
    assert np.isclose(torque, 0).all()


def test_torque_analytical_demag():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(6e-9, 4e-9, 1e-9), cell=(2e-9, 1e-9, 0.5e-9))
    field = df.Field(mesh, dim=3, value=(0, 1e5, 0))
    system = mm.System(name="Box2")
    system.energy = mm.Demag() + mm.Zeeman(H=(0, 0, 1e5))
    system.m = field
    torque = mag2exp.magnetometry.torque(system)
    assert np.isclose(torque[0], mm.consts.mu0 * 1e10)
    assert np.isclose(torque[1], 0)
    assert np.isclose(torque[2], 0)


def test_torque_analytical__parallel_demag():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(6e-9, 4e-9, 1e-9), cell=(2e-9, 1e-9, 0.5e-9))
    field = df.Field(mesh, dim=3, value=(0, 1e5, 0))
    system = mm.System(name="Box2")
    system.energy = mm.Demag() + mm.Zeeman(H=(0, 1e5, 0))
    system.m = field
    torque = mag2exp.magnetometry.torque(system)
    assert np.isclose(torque, 0).all()
