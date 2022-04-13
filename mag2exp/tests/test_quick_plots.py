# import pytest
import discretisedfield as df
import micromagneticmodel as mm

import mag2exp


def test_quick_plots_ltem_phase():
    mesh = df.Mesh(
        p1=(-5e-9, -4e-9, -1e-9), p2=(5e-9, 4e-9, 1e-9), cell=(2e-9, 1e-9, 0.5e-9)
    )
    field = df.Field(mesh, dim=3, value=(0, 0, 1))
    mag2exp.quick_plots.ltem_phase(field)


def test_quick_plots_ltem_ft_phase():
    mesh = df.Mesh(
        p1=(-5e-9, -4e-9, -1e-9), p2=(5e-9, 4e-9, 1e-9), cell=(2e-9, 1e-9, 0.5e-9)
    )
    field = df.Field(mesh, dim=3, value=(0, 0, 1))
    mag2exp.quick_plots.ltem_ft_phase(field)


def test_quick_plots_ltem_defocus():
    mesh = df.Mesh(
        p1=(-5e-9, -4e-9, -1e-9), p2=(5e-9, 4e-9, 1e-9), cell=(2e-9, 1e-9, 0.5e-9)
    )
    field = df.Field(mesh, dim=3, value=(0, 0, 1))
    mag2exp.quick_plots.ltem_defocus(field, voltage=300e3)


def test_quick_plots_ltem_integrated_mfd():
    mesh = df.Mesh(
        p1=(-5e-9, -4e-9, -1e-9), p2=(5e-9, 4e-9, 1e-9), cell=(2e-9, 1e-9, 0.5e-9)
    )

    def v_fun(point):
        x, y, z = point
        if x < -2e-9:
            return (0, 0, 1)
        elif x < 2e-9:
            return (0, 1, 0)
        else:
            return (0, 0, -1)

    field = df.Field(mesh, dim=3, value=v_fun)
    mag2exp.quick_plots.ltem_integrated_mfd(field)


def test_quick_plots_mfm_phase_shift():
    mesh = df.Mesh(
        p1=(-5e-9, -4e-9, -1e-9), p2=(5e-9, 4e-9, 1e-9), cell=(2e-9, 1e-9, 0.5e-9)
    )

    def v_fun(point):
        x, y, z = point
        if x < -2e-9:
            return (0, 0, 1)
        elif x < 2e-9:
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
    system.m = df.Field(mesh, dim=3, value=v_fun, norm=Ms_fun)
    mag2exp.quick_plots.mfm_phase_shift(system)


def test_quick_plots_x_ray_holography():
    mesh = df.Mesh(
        p1=(-5e-9, -4e-9, -1e-9), p2=(5e-9, 4e-9, 1e-9), cell=(2e-9, 1e-9, 0.5e-9)
    )
    field = df.Field(mesh, dim=3, value=(0, 0, 1))
    mag2exp.quick_plots.x_ray_holography(field)


def test_quick_plots_saxs():
    mesh = df.Mesh(
        p1=(-5e-9, -4e-9, -1e-9), p2=(5e-9, 4e-9, 1e-9), cell=(2e-9, 1e-9, 0.5e-9)
    )
    field = df.Field(mesh, dim=3, value=(0, 0, 1))
    mag2exp.quick_plots.saxs(field)


def test_quick_plots_sans_cross_section():
    mesh = df.Mesh(
        p1=(-5e-9, -4e-9, -1e-9), p2=(5e-9, 4e-9, 1e-9), cell=(2e-9, 1e-9, 0.5e-9)
    )
    field = df.Field(mesh, dim=3, value=(0, 0, 1))
    mag2exp.quick_plots.sans_cross_section(field, method="unpol")


def test_quick_plots_sans_chiral_function():
    mesh = df.Mesh(
        p1=(-5e-9, -4e-9, -1e-9), p2=(5e-9, 4e-9, 1e-9), cell=(2e-9, 1e-9, 0.5e-9)
    )
    field = df.Field(mesh, dim=3, value=(0, 0, 1))
    mag2exp.quick_plots.sans_chiral_function(field)
