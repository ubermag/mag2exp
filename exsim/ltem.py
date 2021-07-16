"""LTEM submodule.

Details ...
"""
import numpy as np
import discretisedfield as df
import micromagneticmodel as mm
from scipy import constants


def phase(field, /, kcx=0.1, kcy=0.1):
    r"""Calculation of the magnetic phase shift experienced by the electrons.

    The Fourier transform of the magnetic phase shift is calculated using

    .. math::

        \widetilde{\phi}_m (k_x,k_y) = \frac {i e \mu_0 k_\perp^2}{h}
        \frac{\left[ \widetilde{\bf M}_I(k_x,k_y) \times
        {\bf k}_\perp \right] _z}{\left( k_\perp^2 + k_c^2 \right)^2},

    where :math:`{\\bf M}_I` is the integrated magnetisation along the path of
    the electron beam. Here we define the electron beam to be
    propagating in the :math:`z` direction.
    :math:`\mu_0` is the vacuum permeability, and :math:`k` is the k-vector in
    Fourier space.
    :math:`k_c` is the radius of the filter and can be written in 2-Dimensions
    as to

    .. math::

        k_c^2 = \left(k_{cx} dk_x\right) ^2 +  \left(k_{cy} dk_y \right) ^2,

    where :math:`dk_x` and :math:`dk_y` are the resolution in fourier space for
    the :math:`x` and :math:`y` directions respectively. :math:`k_{cx}` and
    :math:`k_{cy}` are the radii of the filter in each direction in units of
    cells.


    Parameters
    ----------
    field : discretisedfield.Field
        Magnetisation field.
    kcx : numbers.Real, optional
        Tikhonov filter radius in :math:`x` in units of cells.
    kcy : numbers.Real, optional
        Tikhonov filter radius in :math:`y` in units of cells.

    Returns
    -------
    discretisedfield.Field
        Phase in real space.
    discretisedfield.Field
        Phase in Fourier space.

    Examples
    --------

    1. Uniform out-of-plane field.

    >>> import discretisedfield as df
    >>> import exsim
    >>> mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 1), cell=(1, 1, 1))
    >>> field = df.Field(mesh, dim=3, value=(0, 0, 1))
    >>> phase, ft_phase = exsim.ltem.phase(field)
    >>> phase.real.array.mean()
    0.0

    .. plot::
        :context: close-figs

        2. Visualising the phase using ``matplotlib``.

        >>> import discretisedfield as df
        >>> import exsim
        >>> mesh = df.Mesh(p1=(-5, -4, -1), p2=(5, 4, 1), cell=(2, 1, 0.5))
        ...
        >>> def value_fun(point):
        ...     x, y, z = point
        ...     if x > 0:
        ...         return (0, -1, 0)
        ...     else:
        ...         return (0, 1, 0)
        ...
        >>> field = df.Field(mesh, dim=3, value=value_fun)
        >>> phase, ft_phase = exsim.ltem.phase(field)
        >>> phase.real.mpl_scalar()

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
    r"""Calculating the defocused image.

    Either `wavelength` or `U` must be specified.

    The wavefunction of the electrons is created from the magnetic phase shift
    :math:`\phi_m`

    .. math::

        \psi_0 = e^{i\phi_m},

    and propagated through the electon microscope to the image plane by use of
    the Contrast Transfer Function :math:`T`.
    The wavfunction at a defocus length :math:`\Delta f` in Fourier space is
    given by

    .. math::

        \widetilde{\psi}_{\Delta f} = \widetilde{\psi}_0 e^{2 i \pi \lambda k^2
        (-\frac{1}{2} \Delta f + \frac{1}{4} C_s \lambda^2 k^2)},

    where :math:`\lambda` is the relativistic wavelength of the electrons,
    :math:`C_s` is the spherical aberration coefficient of the microscope, and
    :math:`k` is the wavevector in Fourier space.

    The intensity of an image at a specific defocus is given by

    .. math::

        I_{\Delta f} = \left\vert \psi_{\Delta f} \right\vert^2.

    In focus :math:`I=\left\vert\psi_0\right\vert^2=1`

    Parameters
    ----------
    phase : discretisedfield.Field
        Phase of the electrons from LTEM.
    Cs : numbers.Real, optional
        Spherical aberration coefficient.
    df_length : numbers.Real, optional
        Defocus length in m.
    U : numbers.Real, optional
        Accelerating voltage of electrons in V.
    wavelenght : numbers.Real, optional
        Relativistic wavelength of the electrons.

    Returns
    -------
    discretisedfield.Field
        Intensity at specified defocus.

    Examples
    --------

    1. Zero defocus.

    >>> import discretisedfield as df
    >>> import exsim
    >>> mesh = df.Mesh(p1=(-5e-9, -4e-9, -1e-9), p2=(5e-9, 4e-9, 1e-9),
    ...                    cell=(2e-9, 1e-9, 0.5e-9))
    ...
    >>> def value_fun(point):
    ...     x, y, z = point
    ...     if x > 0:
    ...         return (0,-1, 0)
    ...     else:
    ...         return (0, 1, 0)
    ...
    >>> field = df.Field(mesh, dim=3, value=value_fun)
    >>> phase, ft_phase = exsim.ltem.phase(field)
    >>> df_img = exsim.ltem.defocus_image(phase, Cs=0, df_length=0,
    ...                                   U=300e3)
    >>> df_img.array.mean()
    1.0

    .. plot::
        :context: close-figs

        2. Visualising the phase using ``matplotlib``.

        >>> import discretisedfield as df
        >>> import exsim
        >>> mesh = df.Mesh(p1=(-5e-9, -4e-9, -1e-9), p2=(5e-9, 4e-9, 1e-9),
        ...                    cell=(2e-9, 1e-9, 0.5e-9))
        ...
        >>> def value_fun(point):
        ...     x, y, z = point
        ...     if x > 0:
        ...         return (0, -1, 0)
        ...     else:
        ...         return (0, 1, 0)
        ...
        >>> field = df.Field(mesh, dim=3, value=value_fun)
        >>> phase, ft_phase = exsim.ltem.phase(field)
        >>> df_img = exsim.ltem.defocus_image(phase, Cs=8000, df_length=0.2e-3,
        ...                                   U=300e3)
        >>> df_img.mpl_scalar()

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
    r"""Calculate the integrated magnetic flux density.

    This calculates the magnetic flux density integrated along the beam
    direction given by

    .. math::
        \int_{0}^{t} {\bf B}_\perp dz = \frac{\Phi_0}{\pi}\left(
        \begin{array}{c}
         -\partial/\partial y \\
          \partial/\partial x
        \end{array} \right).

    Ask James!!!

    This quantity is most closely related to values obtained from electron
    holography.

    Parameters
    ----------
    phase : discretisedfield.Field
        Phase of the electrons from LTEM.

    Returns
    -------
    discretisedfield.Field
        Integrated magnetic flux density.

    Example
    -------
    .. plot::
        :context: close-figs

        1. Visualising the phase using ``matplotlib``.

        >>> import discretisedfield as df
        >>> import exsim
        >>> mesh = df.Mesh(p1=(-5e-9, -4e-9, -1e-9), p2=(5e-9, 4e-9, 1e-9),
        ...                    cell=(2e-9, 1e-9, 0.5e-9))
        ...
        >>> def value_fun(point):
        ...     x, y, z = point
        ...     if x > 0:
        ...         return (0, -1, 0)
        ...     else:
        ...         return (0, 1, 0)
        ...
        >>> field = df.Field(mesh, dim=3, value=value_fun)
        >>> phase, ft_phase = exsim.ltem.phase(field)
        >>> df_img = exsim.ltem.defocus_image(phase, Cs=8000, df_length=0.2e-3,
        ...                                   U=300e3)
        >>> imf = exsim.ltem.integrated_magnetic_flux_density(phase)
        >>> imf.mpl()

    """
    pref = mm.consts.hbar / mm.consts.e
    imfd = -1 * phase.real.derivative('y') << phase.real.derivative('x')
    return pref * imfd


def relativistic_wavelength(U):
    r"""Relativistic wavelength of an electron accelerated using a voltage U.

    The wavelength is calculated using

    .. math::
        \lambda = \frac{h}{2 m_e e U
        \sqrt{2 m_e e U \left(1 + \frac{e U}{2 m_e c^2}\right)}}

    where :math:`m_e` and :math:`e` are the mass and charge of an electron
    respectively, :math:`c` is the speed of light, and :math:`U` is the
    accelarating voltage in V.


    Parameters
    ----------
    U : numbers.Real
        Accelerating voltage in V.

    Returns
    -------
    numbers.Real
        Wavelength in m.

    Example
    -------
    1. Accelerating using 300 kV.

    >>> import exsim
    >>> exsim.ltem.relativistic_wavelength(300e3)
    1.9687489006848795e-12

    """
    return constants.h / np.sqrt(
        2 * constants.m_e * U * constants.e
        * (1 + constants.e * U / (2 * constants.m_e * constants.c**2)))
