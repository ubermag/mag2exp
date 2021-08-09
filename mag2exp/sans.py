import numpy as np
import discretisedfield as df


def cross_section(field, /, method, geometry):
    r""" Calculation of scattering cross sections.

    The spin-flip and non-spin-flip neutron scattering cross sections can be
    calculated for specific scattering geometries using

    .. math::
        \begin{align}
            \frac{d\sum^{\pm \pm}}{d\Omega} &\sim |{\bf Q}_z|^2 \\
            \frac{d\sum^{\pm \mp}}{d\Omega} &\sim |{\bf Q}_x|^2 +
                  |{\bf Q}_y|^2 \mp
                  i\left( {\bf Q}_x {\bf Q}^{\ast}_y -
                  {\bf Q}^{\ast}_x {\bf Q}_y \right) .
        \end{align}

    where :math:`{\bf Q}` is the magnetic interaction vector given by

    .. math::
        \begin{equation}
            {\bf Q} = \hat{\bf q} \times \left[ \hat{\bf q} \times
                       \widetilde{\bf M} \right],
        \end{equation}

    where :math:`\hat{\bf q}` is the unit scattering vector and
    :math:`\widetilde{\bf M}` is the Fourier transform of the magnetisation.
    The magnetic interaction vector is is dependent on the scattering geometry
    and the scattering vector is defined as

    .. math::

        \begin{equation}
            {\bf q} = {\bf k}_1 - {\bf k}_0.
        \end{equation}

    The spin based cross sections then be combined in order to get the half
    polarised cross sections

    .. math::

        \begin{align}
            \frac{d\sum^{+}}{d\Omega} &= \frac{d\sum^{++}}{d\Omega} +
            \frac{d\sum^{+-}}{d\Omega}, \\
            \frac{d\sum^{-}}{d\Omega} &= \frac{d\sum^{--}}{d\Omega} +
            \frac{d\sum^{-+}}{d\Omega}.
        \end{align}

    These can further be combined to get the unpolarised cross section

    .. math::
        \begin{align}
            \frac{d\sum}{d\Omega} &= \frac{1}{2} \left(
            \frac{d\sum^{+}}{d\Omega} + \frac{d\sum^{-}}{d\Omega} \right), \\
            \frac{d\sum}{d\Omega} &= \frac{1}{2}
            \left( \frac{d\sum^{++}}{d\Omega} + \frac{d\sum^{+-}}{d\Omega} +
            \frac{d\sum^{--}}{d\Omega} + \frac{d\sum^{-+}}{d\Omega} \right).
        \end{align}

    Parameters
    ----------
    field : discretisedfield.field
        Magnetisation field.
    method : str
        Used to select the relevant cross section and can take the value of

            * pp - positive positive non-spin-flip cross section,
            * nn - negative negative non-spin-flip cross section,
            * pn - positive negative spin-flip cross section,
            * np - negative positive spin-flip cross section,
            * p - positive half polarised cross section,
            * n - negitive half polarised cross section,
            * unpol - unpolarised cross section.
    geometry : str
        Define the experimental geometry with applied magnetic field parallel
        or perpendicular to the neutron propagation vector.

    Returns
    -------
    discretisedfield.Field
        Scattering cross section, arbitrary units.

    Examples
    --------

    .. plot::
        :context: close-figs

        1. Visualising unpolarised cross section with ``matplotlib``.

        >>> import discretisedfield as df
        >>> import micromagneticmodel as mm
        >>> import numpy as np
        >>> import mag2exp
        >>> mesh = df.Mesh(p1=(-25e-9, -25e-9, -2e-9),
        ...                p2=(25e-9, 25e-9, 50e-9),
        ...                cell=(1e-9, 1e-9, 2e-9))
        >>> def v_fun(point):
        ...     x, y, z = point
        ...     q = 10e-9
        ...     return (np.sin(2 * np.pi * x / q),
        ...             0,
        ...             np.cos(2 * np.pi * x / q))
        >>> field = df.Field(mesh, dim=3, value=v_fun, norm=1e5)
        >>> field.plane('z').mpl()
        >>> cs = mag2exp.sans.cross_section(field, method='unpol',
        ...                                 geometry='parallel')
        >>> cs.plane('z').real.mpl.scalar()

    .. plot::
        :context: close-figs

        2. Visualising spin-flip cross section with ``matplotlib``.

        >>> import discretisedfield as df
        >>> import micromagneticmodel as mm
        >>> import numpy as np
        >>> import mag2exp
        >>> mesh = df.Mesh(p1=(-25e-9, -25e-9, -2e-9),
        ...                p2=(25e-9, 25e-9, 50e-9),
        ...                cell=(1e-9, 1e-9, 2e-9))
        >>> def v_fun(point):
        ...     x, y, z = point
        ...     q = 10e-9
        ...     return (np.sin(2 * np.pi * x / q),
        ...             0,
        ...             np.cos(2 * np.pi * x / q))
        >>> field = df.Field(mesh, dim=3, value=v_fun, norm=1e5)
        >>> field.plane('z').mpl()
        >>> cs = mag2exp.sans.cross_section(field, method='pn',
        ...                                 geometry='parallel')
        >>> cs.plane('z').real.mpl.scalar()
    """
    magnetic_interaction = magnetic_interaction_vector(field,
                                                       geometry=geometry)
    if method in ('polarised_pp', 'pp'):
        return abs(magnetic_interaction.z)**2
    elif method in ('polarised_nn', 'nn'):
        return abs(magnetic_interaction.z)**2
    elif method in ('polarised_pn', 'pn'):
        return (abs(magnetic_interaction.x)**2 + abs(magnetic_interaction.y)**2
                - (magnetic_interaction.x * magnetic_interaction.y.conjugate
                   - magnetic_interaction.x.conjugate * magnetic_interaction.y)
                * 1j)
    elif method in ('polarised_np', 'np'):
        return (abs(magnetic_interaction.x)**2 + abs(magnetic_interaction.y)**2
                + (magnetic_interaction.x * magnetic_interaction.y.conjugate
                   - magnetic_interaction.x.conjugate * magnetic_interaction.y)
                * 1j)
    elif method in ('half_polarised_p', 'p'):
        pp = cross_section(field, method='polarised_pp',
                           geometry=geometry)
        pn = cross_section(field, method='polarised_pn',
                           geometry=geometry)
        return pp + pn
    elif method in ('half_polarised_n', 'n'):
        nn = cross_section(field, method='polarised_nn',
                           geometry=geometry)
        np = cross_section(field, method='polarised_np',
                           geometry=geometry)
        return nn + np
    elif method in ('unpolarised', 'unpol'):
        p = cross_section(field, method='half_polarised_p',
                          geometry=geometry)
        n = cross_section(field, method='half_polarised_n',
                          geometry=geometry)
        return 0.5 * (p + n)
    else:
        msg = f'Method {method} is unknown.'
        raise ValueError(msg)


def magnetic_interaction_vector(field, /, geometry):
    r"""Calculation of the magnetic interaction vector.

    The magnetic interaction vector :math:`{\bf Q}` given by

    .. math::
        \begin{equation}
            {\bf Q} = \hat{\bf q} \times \left[ \hat{\bf q} \times
                       \widetilde{\bf M} \right],
        \end{equation}

    where :math:`\hat{\bf q}` is the unit scattering vector and
    :math:`\widetilde{\bf M}` is the Fourier transform of the magnetisation.
    The magnetic interaction vector is is dependent on the scattering geometry
    and the scattering vector is defined as

    .. math::

        \begin{equation}
            {\bf q} = {\bf k}_1 - {\bf k}_0.
        \end{equation}

    There are common scattering geometries defined in SANS, namely with a
    magnetic field applied to the incoming neutron beam in either a
    perpendicular or and parallel geometry.

    **Perpendicular geometry**: the magnetic field is applied along the
    :math:`z` direction while the incoming neutron beam propagates along the
    :math:`x` direction.

    **Parallel geometry**: both the magnetic field and the incoming neutron
    beam are along the :math:`z` direction.

    Parameters
    ----------
    field : discretisedfield.field
        Magnetisation field.
    geometry : str
        Define the experimental geometry as field `parallel` or `perpendicular`
        to the neutron propagation vector.

    Returns
    -------
    discretisedfield.Field
        Magnetic interaction vector.
    """
    if geometry == 'parallel':
        magnetic_interaction = _magnetic_interaction_parallel(field)
    elif geometry == 'perpendicular':
        magnetic_interaction = _magnetic_interaction_perpendicular(field)
    else:
        msg = f'Geometry {geometry} is unknown.'
        raise ValueError(msg)
    return magnetic_interaction


def _magnetic_interaction_parallel(field):
    r"""Parallel magnetic interaction vector.

    Magnetic interaction vector for the geometry where both the applied
    magnetic field and the incoming neutron beam are parallel to the :math:`z`
    direction.

    The magnetic interaction vector :math:`{\bf Q}` is given by

    .. math::
        \begin{equation}
            {\bf Q} = \hat{\bf q} \times \left[ \hat{\bf q} \times
                       \widetilde{\bf M} \right],
        \end{equation}

    where :math:`\hat{\bf q}` is the unit scattering vector and
    :math:`\widetilde{\bf M}` is the Fourier transform of the magnetisation.
    The magnetic interaction vector is is dependent on the scattering geometry
    and the scattering vector is defined as

    .. math::

        \begin{equation}
            {\bf q} = {\bf k}_1 - {\bf k}_0.
        \end{equation}

    For this parallel geometry, the magnetic interaction vector can be written
    as

    .. math::

        {\bf Q}_{\parallel} = \begin{pmatrix}
                                 -\widetilde{\bf M}_x \sin^2\theta +
                                 \widetilde{\bf M}_y \sin\theta\cos\theta \\
                                 \widetilde{\bf M}_x \sin\theta\cos\theta -
                                 \widetilde{\bf M}_y \cos^2\theta \\
                                 -\widetilde{\bf M}_z
                                \end{pmatrix}

    where :math:`\theta` is the angle between the scattering vector and the
    :math:`x` axis.

    Parameters
    ----------
    field : discretisedfield.field
        Magnetisation field.

    Returns
    -------
    discretisedfield.Field
        Magnetic interaction vector.
    """
    m_p_ft = (field * df.dz).integral(direction='z').fftn
    theta = df.Field(m_p_ft.mesh, dim=1,
                     value=lambda x: np.arctan2(x[1], x[0]))
    magnetic_interaction_x = (-m_p_ft.x * np.sin(theta.array)**2 +
                              m_p_ft.y * np.cos(theta.array)
                              * np.sin(theta.array))
    magnetic_interaction_y = (-m_p_ft.y * np.cos(theta.array)**2 +
                              m_p_ft.x * np.cos(theta.array)
                              * np.sin(theta.array))
    magnetic_interaction_z = m_p_ft.z
    return (magnetic_interaction_x
            << magnetic_interaction_y
            << magnetic_interaction_z)


def _magnetic_interaction_perpendicular(field):
    r"""Perpendicular magnetic interaction vector.

    Magnetic interaction vector for the geometry where the applied
    magnetic field is along the :math:`z` direction and the incoming neutron
    beam is along the :math:`x` direction.

    The magnetic interaction vector :math:`{\bf Q}` is given by

    .. math::
        \begin{equation}
            {\bf Q} = \hat{\bf q} \times \left[ \hat{\bf q} \times
                       \widetilde{\bf M} \right],
        \end{equation}

    where :math:`\hat{\bf q}` is the unit scattering vector and
    :math:`\widetilde{\bf M}` is the Fourier transform of the magnetisation.
    The magnetic interaction vector is is dependent on the scattering geometry
    and the scattering vector is defined as

    .. math::

        \begin{equation}
            {\bf q} = {\bf k}_1 - {\bf k}_0.
        \end{equation}

    For this parallel geometry, the magnetic interaction vector can be written
    as

    .. math::

        {\bf Q}_{\perp} = \begin{pmatrix}
                            -\widetilde{\bf M}_x \\
                            -\widetilde{\bf M}_y \cos^2\theta +
                            \widetilde{\bf M}_z \sin\theta\cos\theta \\
                            \widetilde{\bf M}_y \sin\theta\cos\theta -
                            \widetilde{\bf M}_z \sin2\theta
                           \end{pmatrix},

    where :math:`\theta` is the angle between the scattering vector and the
    :math:`x` axis.

    Parameters
    ----------
    field : discretisedfield.field
        Magnetisation field.

    Returns
    -------
    discretisedfield.Field
        Magnetic interaction vector.
    """
    m_p_ft = (field * df.dx).integral(direction='x').fftn
    theta = df.Field(m_p_ft.mesh, dim=1,
                     value=lambda x: np.arctan2(x[2], x[1]))
    magnetic_interaction_x = -m_p_ft.x
    magnetic_interaction_y = (-m_p_ft.y * np.cos(theta.array)**2 +
                              m_p_ft.z * np.cos(theta.array)
                              * np.sin(theta.array))
    magnetic_interaction_z = (-m_p_ft.z * np.sin(theta.array)**2
                              + m_p_ft.y * np.cos(theta.array)
                              * np.sin(theta.array))
    return (magnetic_interaction_x
            << magnetic_interaction_y
            << magnetic_interaction_z)


def _magnetic_interaction_perpendicular_z(field):
    r""" Testing a perpendicular function.
    Parameters
    ----------
    field : discretisedfield.field
        Magneisation field.

    Returns
    -------
    discretisedfield.Field
        Magnetic interaction vector.
    """
    m_p_ft = (field * df.dz).integral(direction='z').fftn
    theta = df.Field(m_p_ft.mesh, dim=1,
                     value=lambda x: np.arctan2(x[1], x[0]))
    magnetic_interaction_x = (-m_p_ft.y * np.sin(theta.array)**2 +
                              m_p_ft.z * np.cos(theta.array) *
                              np.sin(theta.array))
    magnetic_interaction_y = (-m_p_ft.z*np.cos(theta.array)**2 +
                              m_p_ft.y*np.cos(theta.array) *
                              np.sin(theta.array))
    magnetic_interaction_z = m_p_ft.x
    return (magnetic_interaction_x
            << magnetic_interaction_y
            << magnetic_interaction_z)


def _magnetic_interaction_perpendicular_z_2(field):
    r""" Testing a perpendicular function.

    Parameters
    ----------
    field : discretisedfield.field
        Magnetisation field.

    Returns
    -------
    discretisedfield.Field
        Magnetic interaction vector.
    """
    m_p_ft = (field * df.dz).integral(direction='z').fftn
    theta = df.Field(m_p_ft.mesh, dim=1,
                     value=lambda x: np.arctan2(x[1], x[0]))
    magnetic_interaction_x = (m_p_ft.y * np.sin(theta.array)**2 -
                              m_p_ft.z * np.cos(theta.array) *
                              np.sin(theta.array))
    magnetic_interaction_y = (m_p_ft.z*np.cos(theta.array)**2 -
                              m_p_ft.y*np.cos(theta.array) *
                              np.sin(theta.array))
    magnetic_interaction_z = -m_p_ft.x
    return (magnetic_interaction_y
            << magnetic_interaction_z
            << magnetic_interaction_x)


def chiral_function(field, /, geometry):
    r"""Calculation of the chiral function.

    The chiral function can be calculated using

    .. math::
        \chi = Q_x Q_y^* - Q_x^* Qy

    where :math:`{\bf Q}` is the magnetic interaction vector given by

    .. math::
        \begin{equation}
            {\bf Q} = \hat{\bf q} \times \left[ \hat{\bf q} \times
                       \widetilde{\bf M} \right].
        \end{equation}

    :math:`\hat{\bf q}` is the unit scattering vector and
    :math:`\widetilde{\bf M}` is the Fourier transform of the magnetisation.
    The magnetic interaction vector is is dependent on the scattering geometry
    and the scattering vector is defined as

    .. math::

        \begin{equation}
            {\bf q} = {\bf k}_1 - {\bf k}_0.
        \end{equation}

    In general :math:`\chi=0` for the parallel scattering geometry.

    Parameters
    ----------
    field : discretisedfield.field
        Magnetisation field.
    geometry : str
        The experimental geometry defined with the external magnetic field
        `parallel` or `perpendicular` to the neutron propagation vector.

    Returns
    -------
    discretisedfield.Field
        Chiral function.

    Examples
    --------

    .. plot::
        :context: close-figs

        1. Visualising the chiral function with ``matplotlib``.

        >>> import discretisedfield as df
        >>> import micromagneticmodel as mm
        >>> import numpy as np
        >>> import mag2exp
        >>> mesh = df.Mesh(p1=(-25e-9, -25e-9, -25e-9),
        ...                p2=(25e-9, 25e-9, 25e-9),
        ...                cell=(1e-9, 1e-9, 1e-9))
        >>> def v_fun(point):
        ...     x, y, z = point
        ...     q = 10e-9
        ...     return (np.cos(2 * np.pi * y / q),
        ...             np.sin(2 * np.pi * y / q),
        ...             0)
        >>> field = df.Field(mesh, dim=3, value=v_fun, norm=1e5)
        >>> field.plane('x').mpl()
        >>> cf = mag2exp.sans.chiral_function(field,
        ...                                   geometry='perpendicular')
        >>> cf.plane('x').imag.mpl.scalar()
    """
    magnetic_interaction = magnetic_interaction_vector(field,
                                                       geometry=geometry)
    return (magnetic_interaction.x * magnetic_interaction.y.conjugate
            - magnetic_interaction.x.conjugate * magnetic_interaction.y)
