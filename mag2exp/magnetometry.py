"""Magnetometry submodule.

Module for calculation of magnetometry based techniques.
"""
import mag2exp
import discretisedfield as df


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
