import numpy as np
import discretisedfield as df
import micromagneticmodel as mm
from scipy import constants


# TODO
# - recheck that all units are correct, everything should be in SI units
# - if possible: remove transposing before/after Fourier transform
# - Write a proper documentation.
# - Write tests.
def ltem_phase(field, /, U, Cs, kx=0.1, ky=0.1):
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
    const = 1j * field.mesh.region.edges[2] / (
        2 * constants.codata.value('mag. flux quantum'))
    m_sat = np.max(field.norm.array) * mm.consts.mu0

    mx_projection = field.orientation.x.project('z').array.squeeze()
    my_projection = field.orientation.y.project('z').array.squeeze()

    ft_mx = np.fft.fft2(mx_projection, axes=(-2, -1))
    ft_my = np.fft.fft2(my_projection, axes=(-2, -1))

    freq_comp_rows = np.fft.fftfreq(ft_mx.shape[0], d=field.mesh.dx)
    freq_comp_cols = np.fft.fftfreq(ft_mx.shape[1], d=field.mesh.dy)

    xs_ft, ys_ft = np.meshgrid(freq_comp_rows, freq_comp_cols, indexing='ij')
    dx_ft = abs(freq_comp_rows[0] - freq_comp_rows[1])
    dy_ft = abs(freq_comp_cols[0] - freq_comp_cols[1])

    nume = xs_ft**2 + ys_ft**2
    dnom = (xs_ft**2 + ys_ft**2 + dx_ft**2 * kx**2 + dy_ft**2 * ky**2)**2
    cross = - ft_my * xs_ft + ft_mx * ys_ft
    ft_phase = const * cross * nume / dnom * m_sat
    phase = np.fft.ifft2(ft_phase).real

    phase_field = df.Field(field.mesh.plane('z'), dim=1,
                           value=phase.reshape((*field.mesh.n[:2], 1, 1)))
    # TODO create df.Field from ft_phase and update return values
    # ft_phase_field = df.Field()
    return phase_field, ft_phase  # TODO replace ft_phase with ft_phase_field


def ltem_defocus_image(phase, /, U, Cs, df=0.2):
    """Defocused image.

    Parameters
    ----------
    phase : discretisedfield.Field
        LTEM phase
    U : numbers.Real
        Accelerating voltage of electrons in V.
    Cs : numbers.Real
        Spherical aberration coefficient
    df : numbers.Real
        <explanation>

    Returns
    -------

    """
    wavefn = np.exp(phase.array * 1j)
    ft_wavefn = np.fft.fft2(wavefn)

    freq_comp_rows = np.fft.fftfreq(ft_wavefn.shape[0], d=phase.mesh.dx)
    freq_comp_cols = np.fft.fftfrex(ft_wavefn.shape[1], d=phase.mesh.dy)
    xs_ft, ys_ft = np.meshgrid(freq_comp_rows, freq_comp_cols, indexing='xy')

    ksquare_ft = xs_ft**2 + ys_ft**2

    intensity_cts = ctf(df, ksquare_ft, ft_wavefn, U, Cs)
    # TODO create df.field to return
    return intensity_cts


def ctf(df, ft_wf_k2, ft_wavefn, wavelength, Cs):
    cts = -0.5 * wavelength * df * ft_wf_k2 + 0.25 * wavelength**3 * Cs * ft_wf_k2**2
    ft_def_wf_cts = ft_wavefn * np.exp(2*np.pi * cts * 1j)
    def_wf_cts = np.fft.ift2(ft_def_wf_cts)
    intensity_cts = def_wf_cts.conjugate() * def_wf_cts
    return intensity_cts.real


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
