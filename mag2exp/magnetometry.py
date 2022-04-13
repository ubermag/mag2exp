"""Magnetometry submodule.

Module for calculation of magnetometry based techniques.
"""
import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import oommfc as oc


def magnetisation(field):
    r"""Calculation of the magnetisation.

    Calculates the magnetisation in the regions where the norm is non-zero.

    Parameters
    ----------
    field : discretisedfield.Field
        Magnetisation field.

    Returns
    -------
    Magnetisation : tuple
        Magnetisation in :math:`\textrm{Am}^{-1}`.

    Examples
    --------

    1.Uniform magnetisation in :math:`z` direction.

    >>> import discretisedfield as df
    >>> import micromagneticmodel as mm
    >>> import mag2exp
    >>> mesh = df.Mesh(p1=(-25e-9, -25e-9, -2e-9),
    ...                p2=(25e-9, 25e-9, 50e-9),
    ...                cell=(1e-9, 1e-9, 2e-9))
    >>> field= df.Field(mesh, dim=3, value=(0,0,1), norm=1e6)
    >>> mag2exp.magnetometry.magnetisation(field)
    (0.0, 0.0, 1000000.0)

    2. Spatially dependent magnetisation.

    >>> import discretisedfield as df
    >>> import micromagneticmodel as mm
    >>> import mag2exp
    >>> mesh = df.Mesh(p1=(-25e-9, -25e-9, -2e-9),
    ...                p2=(25e-9, 25e-9, 50e-9),
    ...                cell=(1e-9, 1e-9, 2e-9))
    >>> def v_fun(point):
    ...     x, y, z = point
    ...     if x < -2e-9:
    ...         return (0, 0, 1)
    ...     elif x < 2e-9:
    ...         return (0, 1, 0)
    ...     else:
    ...         return (-1, 1, 0)
    >>> field= df.Field(mesh, dim=3, value=v_fun, norm=1e6)
    >>> mag2exp.magnetometry.magnetisation(field)
    (-325269.1193457478, 405269.1193455959, 460000.0)

    """
    return tuple(np.array(field.average) / field.orientation.norm.average)


def torque(system, /, use_demag=True):
    r"""Calculation of the torque.

    The torque is calculated using

    .. math::
        {\bf \tau} = {\bf m} \times {\bf B},

    where :math:`{\bf B}` is the magnetic flux density and :math:`{\bf m}` is
    the magnetic moment.
    The magnetisation :math:`{\bf M}` can be related to the magnetic moment
    using

    .. math::
        {\bf M} = \frac{d{\bf m}}{dV},

    where :math:`dV` is a volume element.

    These equations can be written in terms of its constituent components

    .. math::
        {\bf \tau} = {\mu_0 {\bf m} \times \left( {\bf H}_{app} +
                      {\bf H}_{demag} \right)},

    where :math:`{\bf H}_{app}` is the applied magnetic field, and
    :math:`{\bf H}_{demag}` is the demagnetisation field.

    Parameters
    ----------
    system : micromagneticmodel.System
        Micromagnetic system which must include the magnetisation
        configuration and an energy equation which includes demagnetisation.

    Returns
    -------
    tuple
        Torque in :math:`\textrm{Nm}^{-2}`.

    Examples
    --------

    1. Field along magnetisation direction with demagnetisation.

    >>> import discretisedfield as df
    >>> import micromagneticmodel as mm
    >>> import mag2exp
    >>> mesh = df.Mesh(p1=(-25e-9, -25e-9, -2e-9),
    ...                p2=(25e-9, 25e-9, 50e-9),
    ...                cell=(1e-9, 1e-9, 2e-9))
    >>> system = mm.System(name='Box2')
    >>> system.energy = mm.Zeeman(H=(0, 0, 1e6)) + mm.Demag()
    >>> system.m = df.Field(mesh, dim=3, value=(0, 0, 1), norm=1e6)
    >>> np.allclose(mag2exp.magnetometry.torque(system, use_demag=True), 0)
    Running OOMMF...
    True

    2. Field along magnetisation direction without demagnetisation.

    >>> import discretisedfield as df
    >>> import micromagneticmodel as mm
    >>> import mag2exp
    >>> mesh = df.Mesh(p1=(-25e-9, -25e-9, -2e-9),
    ...                p2=(25e-9, 25e-9, 50e-9),
    ...                cell=(1e-9, 1e-9, 2e-9))
    >>> system = mm.System(name='Box2')
    >>> system.energy = mm.Zeeman(H=(0, 0, 1e6))
    >>> system.m = df.Field(mesh, dim=3, value=(0, 0, 1), norm=1e6)
    >>> mag2exp.magnetometry.torque(system, use_demag=False)
    (0.0, 0.0, 0.0)

    3. Field perpendicular to magnetisation direction without demagnetisation.

    >>> import discretisedfield as df
    >>> import micromagneticmodel as mm
    >>> import mag2exp
    >>> mesh = df.Mesh(p1=(-25e-9, -25e-9, -2e-9),
    ...                p2=(25e-9, 25e-9, 50e-9),
    ...                cell=(1e-9, 1e-9, 2e-9))
    >>> system = mm.System(name='Box2')
    >>> system.energy = mm.Zeeman(H=(0, 1e6, 0))
    >>> system.m = df.Field(mesh, dim=3, value=(0, 0, 1), norm=1e6)
    >>> mag2exp.magnetometry.torque(system, use_demag=False)
    (-1256637.061435814, 0.0, 0.0)
    """
    if use_demag:
        total_field = mm.consts.mu0 * (
            oc.compute(system.energy.demag.effective_field, system)
            + system.energy.zeeman.H
        )
    else:
        total_field = mm.consts.mu0 * np.array(system.energy.zeeman.H)
    norm_field = df.Field(system.m.mesh, dim=1, value=(system.m.norm.array != 0))
    volume = df.integral(norm_field * df.dV, direction="xyz")
    moment = system.m * volume
    torque = moment & total_field
    return df.integral(torque * df.dV / volume**2, direction="xyz")
