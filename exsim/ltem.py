import numpy as np
import discretisedfield as df
import micromagneticmodel as mm
from scipy import constants


def phase(field, /, kx=0.1, ky=0.1):
    """LTEM phase contrast.

    Parameters
    ----------
    field : df.Field
        Magnetisation field.
    kx : numbers.Real, optional
        Tikhonov filter radius in x in pixels.
    ky : numbers.Real, optional
        Tikhonov filter radius in y in pixels.

    Returns
    -------
    df.Field
        Phase in real space.
    df.Field
        Phase in Fourier space.

    Example
    -------
    ...
    """
    m_int = (field * df.dz).integral(direction='z')
    m_ft = m_int.fft2()

    k = df.Field(m_ft.mesh, dim=3, value=lambda x: x)
    denom = (k.x**2 + k.y**2) / (k.x**2 + k.y**2
                                 + k.mesh.dx**2*kx**2 + k.mesh.dy**2*ky**2)**2
    const = 1j * mm.consts.e * mm.consts.mu0 / mm.consts.h
    ft_phase = (m_ft & k).z * denom * const
    phase = ft_phase.ifft2()
    return phase, ft_phase


def defocus_image(phase, /, Cs=0, df_length=0.2e-3, U=None, wavelenght=None):
    """Defocused image.

    Either `wavelength` or `U` must be specified.

    Parameters
    ----------
    phase : discretisedfield.Field
        LTEM phase
    Cs : numbers.Real, optional
        Spherical aberration coefficient
    df_length : numbers.Real, optional
        Defocus length in m.
    U : numbers.Real, optional
        Accelerating voltage of electrons in V.
    wavelenght : numbers.Real, optional
        Relativistic wavelength of the electrons.

    Returns
    -------
    discretisedfield.Field
        ...
    """
    ft_wavefn = df.Field(phase.mesh, dim=phase.dim,
                         value=np.exp(phase.array * 1j)).fft2()
    k = df.Field(ft_wavefn.mesh, dim=3, value=lambda x: x)
    ksquare = (k.x**2 + k.y**2).array

    if wavelenght is None:
        if U is None:
            msg = ('Either `wavelength` or acceleration voltage `U` needs'
                   'to be specified.')
            raise RuntimeError(msg)
        wavelength = relativistic_wavelength(U)

    cts = -df_length + 0.5 * wavelength**2 * Cs * ksquare
    exp = np.exp(np.pi * cts * 1j * ksquare * wavelength)
    ft_def_wf_cts = ft_wavefn * exp
    def_wf_cts = ft_def_wf_cts.ifft2()
    intensity_cts = def_wf_cts.conjugate * def_wf_cts
    return intensity_cts.real


def integrated_magnetic_flux_density(phase):
    """Integrated magnetic flux density

    Parameters
    ----------
    phase : discretisedfield.Field
        Phase

    Returns
    -------
    discretisedfield.Field
        Integrated magnetic flux density

    Example
    -------
    ...
    """
    pref = mm.consts.hbar / mm.consts.e
    imfd = -1 * phase.real.derivative('y') << phase.real.derivative('x')
    return pref * imfd

def relativistic_wavelength(U):
    r"""Relativistic wavelength of an electron accelerated in a potential U.

    .. math:: \lambda = \frac{h}{\sqrt(2 m_e e U \cdot (1 + \frac{e U}{2 m_e c^2}))}

    Parameters
    ----------
    U : numbers.Real
        Accelerating voltage in V.

    Returns
    -------
    numbers.Real
        Wavelength in m.
    """
    return constants.h / np.sqrt(
        2 * constants.m_e * U * constants.e
        * (1 + constants.e * U / (2 * constants.m_e * constants.c**2)))
