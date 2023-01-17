"""Magnetometry submodule.

Module for calculation of magnetometry based techniques.
"""
import discretisedfield as df
import micromagneticmodel as mm
import numpy as np


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
    >>> field= df.Field(mesh, nvdim=3, value=(0,0,1), norm=1e6)
    >>> mag2exp.magnetometry.magnetisation(field)
    array([      0.,       0., 1000000.])

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
    >>> field= df.Field(mesh, nvdim=3, value=v_fun, norm=1e6)
    >>> mag2exp.magnetometry.magnetisation(field)
    array([-325269.11934575,  405269.1193456 ,  460000.        ])

    """
    # TODO: Valid volume
    return field.mean() / field.orientation.norm.mean()


def torque(field, H):
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

    These equations can be written as

    .. math::
        {\bf \tau} = {\mu_0 {\bf m} \times {\bf H}_{app}},

    where :math:`{\bf H}_{app}` is the applied magnetic field.

    Parameters
    ----------
    field : discretisedfield.Field
        Magnetisation field.

    H : tuple, discretisedfield.Field
        Applied magnetic flux density in :math:`\textrm{Am}^{-1}`.

    Returns
    -------
    tuple
        Torque in :math:`\textrm{Nm}^{-2}`.

    Examples
    --------

    1. Field along magnetisation direction.

    >>> import discretisedfield as df
    >>> import micromagneticmodel as mm
    >>> import mag2exp
    >>> mesh = df.Mesh(p1=(-25e-9, -25e-9, -2e-9),
    ...                p2=(25e-9, 25e-9, 50e-9),
    ...                cell=(1e-9, 1e-9, 2e-9))
    >>> system = mm.System(name='Box2')
    >>> system.energy = mm.Zeeman(H=(0, 0, 1e6)) + mm.Demag()
    >>> system.m = df.Field(mesh, nvdim=3, value=(0, 0, 1), norm=1e6)
    >>> np.allclose(mag2exp.magnetometry.torque(system.m, system.energy.zeeman.H), 0)
    True

    2. Field perpendicular to magnetisation direction.

    >>> import numpy as np
    >>> import discretisedfield as df
    >>> import micromagneticmodel as mm
    >>> import mag2exp
    >>> mesh = df.Mesh(p1=(-25e-9, -25e-9, -2e-9),
    ...                p2=(25e-9, 25e-9, 50e-9),
    ...                cell=(1e-9, 1e-9, 2e-9))
    >>> system = mm.System(name='Box2')
    >>> system.energy = mm.Zeeman(H=(1e6, 0, 0)) + mm.Demag()
    >>> system.m = df.Field(mesh, nvdim=3, value=(0, 0, 1), norm=1e6)
    >>> np.allclose(
    ...     mag2exp.magnetometry.torque(system.m, system.energy.zeeman.H),
    ...     (0, mm.consts.mu0*1e12, 0)
    ... )
    True
    """
    # TODO: Valid volume
    # TODO: Field of H
    total_field = mm.consts.mu0 * np.array(H)
    norm_field = df.Field(field.mesh, nvdim=1, value=(field.norm.array != 0))
    volume = norm_field.integrate()
    moment = field * volume[0]
    torque = moment & total_field
    return torque.integrate() / volume**2
