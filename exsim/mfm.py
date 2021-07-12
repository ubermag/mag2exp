import micromagneticmodel as mm
import oommfc as oc
from exsim.util import gaussian_filter


def phase_shift(system, /, tip_m=(0,0,0), Q=650, k=3, tip_q=0):
    """
    Parameters
    ----------
    system : micromagneticmodel.System
        ...
    """
    stray_field = oc.compute(system.energy.demag.effective_field, system)
    dh_dz = stray_field.derivative('z', n=1)
    d2h_dz2 = stray_field.derivative('z', n=2)
    phase_shift = (Q * mm.consts.mu0 / k) * (tip_q * dh_dz.z + d2h_dz2 @ tip_m)
    return phase_shift
