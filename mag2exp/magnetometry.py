"""Magnetometry submodule.

Module for calculation of magnetometry based techniques.
"""
import mag2exp
import discretisedfield as df
import micromagneticmodel as mm
import oommfc as oc


def magnetisation(field):
    r""" Calculation of the magnetisation.

    Parameters
    ----------
    field : discretisedfield.Field
        Magnetisation field.

    Returns
    -------
    tuple
        Magnetisation in :math:`\textrm{Am}^{-1}`.

    """
    norm_field = df.Field(field.mesh, dim=1, value=(field.norm.array != 0))
    volume = df.integral(norm_field * df.dV, direction='xyz')
    return (df.integral(field * df.dV / volume, direction='xyz'))


def torque(system):
    r""" Calculation of the torque.

    Parameters
    ----------
    system : micromagneticmodel.System
        Micromagnetic system which must include the magnetisation
        configuration and an energy equation which includes demagnetisation.

    Returns
    -------
    tuple
        Torque in :math:`\textrm{Nm}^{-2}`.

    """
    total_field = (mm.consts.mu0 *
                   (oc.compute(system.energy.demag.effective_field, system) +
                    system.energy.zeeman.H))
    norm_field = df.Field(system.m.mesh, dim=1,
                          value=(system.m.norm.array != 0))
    volume = df.integral(norm_field * df.dV, direction='xyz')
    moment = system.m * volume
    torque = (moment & total_field)
    return (df.integral(torque * df.dV / volume**2, direction='xyz'))
