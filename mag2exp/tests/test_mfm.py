import discretisedfield as df
import micromagneticmodel as mm
import pytest

import mag2exp


def test_mfm_phase_tip_m():
    mesh = df.Mesh(
        p1=(-5e-9, -4e-9, -2e-9), p2=(5e-9, 4e-9, 6e-9), cell=(2e-9, 1e-9, 2e-9)
    )

    def f_val(position):
        x, y, z = position
        if x < 0:
            return (0, 0, 1)
        elif x < 1e-9:
            return (0, 1, 0)
        else:
            return (0, 0, -1)

    def Ms_fun(pos):
        x, y, z = pos
        if z < 0:
            return 384e3
        else:
            return 0

    system = mm.System(name="Box2")
    system.energy = mm.Demag()
    system.m = df.Field(mesh, dim=3, value=f_val, norm=Ms_fun)
    ps = mag2exp.mfm.phase_shift(system, tip_m=(0, 0, 1e-16), quality=650, k=3, tip_q=0)
    assert (ps.array != 0).any()
    ps = mag2exp.mfm.phase_shift(system, tip_m=(0, 1e-16, 0), quality=650, k=3, tip_q=0)
    assert (ps.array != 0).any()
    ps = mag2exp.mfm.phase_shift(system, tip_m=(1e-16, 0, 0), quality=650, k=3, tip_q=0)
    assert (ps.array != 0).any()


def test_mfm_phase_tip_q():
    mesh = df.Mesh(
        p1=(-5e-9, -4e-9, -2e-9), p2=(5e-9, 4e-9, 6e-9), cell=(2e-9, 1e-9, 2e-9)
    )

    def f_val(position):
        x, y, z = position
        if x < 0:
            return (0, 0, 1)
        elif x < 1e-9:
            return (0, 1, 0)
        else:
            return (0, 0, -1)

    def Ms_fun(pos):
        x, y, z = pos
        if z < 0:
            return 384e3
        else:
            return 0

    system = mm.System(name="Box2")
    system.energy = mm.Demag()
    system.m = df.Field(mesh, dim=3, value=f_val, norm=Ms_fun)
    ps = mag2exp.mfm.phase_shift(system, tip_m=(0, 0, 0), quality=650, k=3, tip_q=1e-6)
    assert (ps.array != 0).any()


def test_mfm_phase_no_tip():
    mesh = df.Mesh(
        p1=(-5e-9, -4e-9, -2e-9), p2=(5e-9, 4e-9, 6e-9), cell=(2e-9, 1e-9, 2e-9)
    )

    def f_val(position):
        x, y, z = position
        if x < 0:
            return (0, 0, 1)
        elif x < 1e-9:
            return (0, 1, 0)
        else:
            return (0, 0, -1)

    def Ms_fun(pos):
        x, y, z = pos
        if z < 0:
            return 384e3
        else:
            return 0

    system = mm.System(name="Box2")
    system.energy = mm.Demag()
    system.m = df.Field(mesh, dim=3, value=f_val, norm=Ms_fun)
    ps = mag2exp.mfm.phase_shift(system, tip_m=(0, 0, 0), quality=650, k=3, tip_q=0)
    assert (ps.array == 0).all()


def test_mfm_phase_quality():
    mesh = df.Mesh(
        p1=(-5e-9, -4e-9, -2e-9), p2=(5e-9, 4e-9, 6e-9), cell=(2e-9, 1e-9, 2e-9)
    )

    def f_val(position):
        x, y, z = position
        if x < 0:
            return (0, 0, 1)
        elif x < 1e-9:
            return (0, 1, 0)
        else:
            return (0, 0, -1)

    def Ms_fun(pos):
        x, y, z = pos
        if z < 0:
            return 384e3
        else:
            return 0

    system = mm.System(name="Box2")
    system.energy = mm.Demag()
    system.m = df.Field(mesh, dim=3, value=f_val, norm=Ms_fun)
    ps = mag2exp.mfm.phase_shift(system, tip_m=(0, 0, 1e-16), quality=0)
    assert (ps.array == 0).all()


def test_mfm_phase_k():
    mesh = df.Mesh(
        p1=(-5e-9, -4e-9, -2e-9), p2=(5e-9, 4e-9, 6e-9), cell=(2e-9, 1e-9, 2e-9)
    )

    def f_val(position):
        x, y, z = position
        if x < 0:
            return (0, 0, 1)
        elif x < 1e-9:
            return (0, 1, 0)
        else:
            return (0, 0, -1)

    def Ms_fun(pos):
        x, y, z = pos
        if z < 0:
            return 384e3
        else:
            return 0

    system = mm.System(name="Box2")
    system.energy = mm.Demag()
    system.m = df.Field(mesh, dim=3, value=f_val, norm=Ms_fun)
    with pytest.raises(RuntimeError):
        mag2exp.mfm.phase_shift(system, tip_m=(0, 0, 1e-16), k=0)
    with pytest.raises(RuntimeError):
        mag2exp.mfm.phase_shift(system, tip_m=(0, 0, 1e-16), k=-3)


def test_mfm_phase_demag():
    mesh = df.Mesh(
        p1=(-5e-9, -4e-9, -2e-9), p2=(5e-9, 4e-9, 6e-9), cell=(2e-9, 1e-9, 2e-9)
    )

    def f_val(position):
        x, y, z = position
        if x < 0:
            return (0, 0, 1)
        elif x < 1e-9:
            return (0, 1, 0)
        else:
            return (0, 0, -1)

    def Ms_fun(pos):
        x, y, z = pos
        if z < 0:
            return 384e3
        else:
            return 0

    system = mm.System(name="Box2")
    system.energy = mm.Exchange(A=8.78e-12)
    system.m = df.Field(mesh, dim=3, value=f_val, norm=Ms_fun)
    with pytest.raises(AttributeError):
        mag2exp.mfm.phase_shift(system, tip_m=(0, 0, 1e-16), k=3)


def test_mfm_phase_fwhm():
    mesh = df.Mesh(
        p1=(-5e-9, -4e-9, -2e-9), p2=(5e-9, 4e-9, 6e-9), cell=(2e-9, 1e-9, 2e-9)
    )

    def f_val(position):
        x, y, z = position
        if x < 0:
            return (0, 0, 1)
        elif x < 1e-9:
            return (0, 1, 0)
        else:
            return (0, 0, -1)

    def Ms_fun(pos):
        x, y, z = pos
        if z < 0:
            return 384e3
        else:
            return 0

    system = mm.System(name="Box2")
    system.energy = mm.Demag()
    system.m = df.Field(mesh, dim=3, value=f_val, norm=Ms_fun)
    mag2exp.mfm.phase_shift(
        system, tip_m=(0, 0, 1e-16), quality=650, k=3, tip_q=0, fwhm=(1e-9, 1e-9, 1e-9)
    )
