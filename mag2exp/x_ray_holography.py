import mag2exp
import discretisedfield as df


def holographic_image(field, /, fwhm=None):
    r""" Calculation of the pattern obtained from X-ray holography.

    X-ray holography uses magnetic circular dichroism to measure the magnetic
    field parallel to the propagation direction of the light. Here, we define
    the experimental reference frame with the light propagating along :math:`z`
    direction. The results of X-ray holographic imaging is the real space
    integral of the magnetisation along the beam direction.

    Parameters
    ----------
    field : discretisedfield.field
        Magnetisation field.
    fwhm : array_like, optional
        If specified, convolutes the output image with a 2 Dimensional Gaussian
        with the full width half maximum (fwhm) specified.

    Returns
    -------
    discretisedfield.Field
        X-ray holographic image.

    Examples
    --------

    .. plot::
        :context: close-figs

        1. Visualising X-ray holography with ``matplotlib``.

        >>> import discretisedfield as df
        >>> import mag2exp
        >>> mesh = df.Mesh(p1=(-5e-9, -4e-9, 0), p2=(5e-9, 4e-9, 2e-9),
        ...                cell=(1e-9, 1e-9, 1e-9))
        >>> def value_fun(point):
        ...     x, y, z = point
        ...     if x < -2e-9:
        ...         return (0, 0, 1)
        ...     elif x < 2e-9:
        ...         return (0, 1, 0)
        ...     else:
        ...         return (0, 0, -1)
        >>> field = df.Field(mesh, dim=3, value=value_fun, norm=0.3e6)
        >>> xrh = mag2exp.x_ray_holography.holographic_image(field)
        >>> xrh.mpl.scalar()


    .. plot::
        :context: close-figs

        1. Visualising X-ray holography with ``matplotlib``.

        >>> import discretisedfield as df
        >>> import mag2exp
        >>> mesh = df.Mesh(p1=(-5e-9, -4e-9, 0), p2=(5e-9, 4e-9, 2e-9),
        ...                cell=(1e-9, 1e-9, 1e-9))
        >>> def value_fun(point):
        ...     x, y, z = point
        ...     if x < -2e-9:
        ...         return (0, 0, 1)
        ...     elif x < 2e-9:
        ...         return (0, 1, 0)
        ...     else:
        ...         return (0, 0, -1)
        >>> field = df.Field(mesh, dim=3, value=value_fun, norm=0.3e6)
        >>> xrh2 = mag2exp.x_ray_holography.holographic_image(field,
        ...                                                   fwhm=(2e-9,2e-9))
        >>> xrh2.mpl.scalar()
    """
    # Direction arg will be removed soon.
    magnetisation = df.integral(field.z * df.dz, direction='z')
    if fwhm is not None:
        magnetisation = mag2exp.util.gaussian_filter(magnetisation, fwhm=fwhm)
    return magnetisation


def holographic_scattering(field):
    r""" Calculation of the scattering pattern obtained from X-ray holography.

    X-ray holography uses magnetic circular dichroism to measure the magnetic
    field parallel to the propagation direction of the light. Here, we define
    the experimental reference frame with the light propagating along :math:`z`
    direction. The intensity of X-ray holographic scattering can be calculated
    from the Fourier transform of the real space integral of the magnetisation
    along the beam direction multiplied by its complex conjugate.

    .. math::
        M_I = \int M dz, \\
        I = \left\vert \widetilde{M_I} \right\vert ^2 .

    Parameters
    ----------
    field : discretisedfield.field
        Magnetisation field.

    Returns
    -------
    discretisedfield.Field
        X-ray holographic scattering.

    Examples
    --------
    .. plot::
        :context: close-figs

        1. Visualising the scattering with ``matplotlib``.

        >>> import discretisedfield as df
        >>> import numpy as np
        >>> import mag2exp
        >>> mesh = df.Mesh(p1=(0, 0, 0), p2=(50e-9, 50e-9, 1e-9),
        ...                cell=(1e-9, 1e-9, 1e-9))
        >>> def value_fun(point):
        ...     x, y, z = point
        ...     qx = 10e-9
        ...     return (np.sin(2 * np.pi * x/ qx),
        ...             0,
        ...             np.cos(2 * np.pi * x/ qx))
        >>> field = df.Field(mesh, dim=3, value=value_fun, norm=0.3e6)
        >>> xrs = mag2exp.x_ray_holography.holographic_scattering(field)
        >>> xrs.mpl.scalar()

    """
    # Direction arg will be removed soon.
    magnetisation = df.integral(field.z * df.dz, direction='z')
    m_fft = magnetisation.fft2()
    return (m_fft * m_fft.conjugate).real