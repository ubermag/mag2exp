"""LTEM submodule.

Details ...
"""
import numpy as np
import discretisedfield as df
import micromagneticmodel as mm
from scipy import constants


def phase(field, /, kcx=0.1, kcy=0.1):
    """Calculation of the magnetic phase shift experienced by the electrons.

    The Fourier transform of the magnetic phase shift is calculated using

    .. math::

        \widetilde{\phi}_m (k_x,k_y) = \\frac {i e \mu_0 k_\perp^2}{h}
        \\frac{\left[ \widetilde{\\bf M}_I(k_x,k_y) \\times
        {\\bf k}_\perp \\right] _z}{\\left( k_\perp^2 + k_c^2 \\right)^2},

    where :math:`{\\bf M}_I` is the integrated magnetisation along the path of
    the electron beam. Here we define the electron beam to be
    propagating in the :math:`z` direction.
    :math:`\mu_0` is the vacuum permeability, and :math:`k` is the k-vector in
    Fourier space.
    :math:`k_c` is the radius of the filter and can be written in 2-Dimensions
    as to

    .. math::

        k_c^2 = k_{cx}^2 dk_x^2 +  k_{cy}^2 dk_y^2,

    where :math:`dk_x` and :math:`dk_y` are the fourier space resolution in the
    :math:`x` and :math:`y` directions respectively. :math:`k_{cx}` and
    :math:`k_{cy}` are the radii of the filter in each direction in units of
    cells.


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
    1. Uniform field

    >>> import discretisedfield as df
    >>> import exsim
    >>> mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 1), cell=(1, 1, 1))
    >>> field = df.Field(mesh, dim=3, value=(0, 0, 1))
    >>> phase, ft_phase = exsim.phase(field)
    >>> phase.array.mean()
    0

    2. Maybe second one
    """
    m_int = (field * df.dz).integral(direction='z')
    m_ft = m_int.fft2()

    k = df.Field(m_ft.mesh, dim=3, value=lambda x: x)
    denom = (k.x**2 + k.y**2) / (k.x**2 + k.y**2
                                 + k.mesh.dx**2*kcx**2 + k.mesh.dy**2*kcy**2)**2
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

    .. math:: \lambda = \frac{h}{\sqrt(2 m_e e U \cdot
    (1 + \frac{e U}{2 m_e c^2}))}

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
