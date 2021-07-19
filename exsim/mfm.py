"""MFM submodule.

Module for calculation of Magnetic Force Microscopy related quantaties.
"""

import micromagneticmodel as mm
import oommfc as oc
# from exsim.util import gaussian_filter  # think about 3D gaussian


def phase_shift(system, /, tip_m=(0, 0, 0), Q=650, k=3, tip_q=0):
    r""" Calculation of the phase shift of an MFM tip.

    Parameters
    ----------
    system : micromagneticmodel.System
        ...

    Returns
    -------
    discretisedfield.Field
        Phase shift space.
    """

    if k <= 0:
        msg = '`k` has to be a positive non-zero number.'
        raise RuntimeError(msg)
    stray_field = oc.compute(system.energy.demag.effective_field, system)
    dh_dz = stray_field.derivative('z', n=1)
    d2h_dz2 = stray_field.derivative('z', n=2)
    phase_shift = (Q * mm.consts.mu0 / k) * (tip_q * dh_dz.z + d2h_dz2 @ tip_m)
    return phase_shift
