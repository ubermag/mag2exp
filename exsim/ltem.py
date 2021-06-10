import numpy as np
import discretisedfield as df
import micromagneticmodel as mm
from scipy import constants


# TODO
# - remove computation of wavelength (not required)
# - recheck that all units are correct, everything should be in SI units
# - if possible: remove transposing before/after Fourier transform
# - Write a proper documentation.
# - Cs is not used. Can it be removed?
# - Write tests.
def ltem_phase(field, /, U, Cs, kx=0.1, ky=0.1):
    """LTEM phase contrast.

    Parameters
    ----------
    field : df.Field
        Magnetisation field.
    U : numbers.Real
        Accelerating voltage of electrons in V.
    Cs : numbers.Real
        Spherical aberration coefficient
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
    const = 1j * field.mesh.region.edges[2] / (
        2 * constants.codata.value('mag. flux quantum'))
    m_sat = np.max(field.norm.array) * mm.consts.mu0

    wavelength = _relativistic_wavelength(U)
    print(f'The electron beam has a wavelength of {wavelength * 1e9:.2e} nm.')

    mx_projection = np.transpose(field.orientation.x.project('z').array.squeeze())
    my_projection = np.transpose(field.orientation.y.project('z').array.squeeze())

    ft_mx = np.fft.fft2(mx_projection, axes=(-2, -1))
    ft_my = np.fft.fft2(my_projection, axes=(-2, -1))

    freq_comp_rows = np.fft.fftfreq(ft_mx.shape[0], d=field.mesh.dx)
    freq_comp_cols = np.fft.fftfreq(ft_mx.shape[1], d=field.mesh.dy)

    xs_ft, ys_ft = np.meshgrid(freq_comp_rows, freq_comp_cols, indexing='xy')
    dx_ft = abs(freq_comp_rows[0] - freq_comp_rows[1])
    dy_ft = abs(freq_comp_cols[0] - freq_comp_cols[1])

    nume = xs_ft**2 + ys_ft**2
    dnom = (xs_ft**2 + ys_ft**2 + dx_ft**2 * kx**2 + dy_ft**2 * ky**2)**2
    cross = - ft_my * xs_ft + ft_mx * ys_ft
    ft_phase = const * cross * nume / dnom * m_sat
    phase = np.fft.ifft2(ft_phase).real

    phase_field = df.Field(field.mesh.plane('z'), dim=1,
                           value=np.transpose(phase).reshape(
                               (*field.mesh.n[:2], 1, 1)))
    # TODO create df.Field from ft_phase and update return values
    # ft_phase_field = df.Field()
    return phase_field, ft_phase  # TODO replace ft_phase with ft_phase_field


def _relativistic_wavelength(U):
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
