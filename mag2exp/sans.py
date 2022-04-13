"""SANS submodule.

Module for calculation of Small Angle Neutron Scattering related
quantities.
"""
import discretisedfield as df
import numpy as np
from scipy.spatial.transform import Rotation


def cross_section(field, /, method, polarisation=(0, 0, 1)):
    r""" Calculation of scattering cross sections.

    The scattering cross sections can be calculated using

    .. math::
        \frac{d\sum}{d\Omega} \sim |{\bf Q} \cdot {\bf \sigma}|^2,

    where :math:`{\bf \sigma}` is the Pauli vector

    .. math::
        {\bf \sigma} = \begin{bmatrix} \sigma_x \\
                                       \sigma_y \\
                                       \sigma_z \end{bmatrix},

    and

    .. math::
        \begin{align}
            \sigma_x &= \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \\
            \sigma_y &= \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \\
            \sigma_z &= \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}.
        \end{align}

    :math:`{\bf Q}` is the magnetic interaction vector given by

    .. math::
        \begin{equation}
            {\bf Q} = \hat{\bf q} \times \widetilde{\bf M} \times \hat{\bf q}
        \end{equation}

    :math:`\hat{\bf q}` is the unit scattering vector and
    :math:`\widetilde{\bf M}` is the Fourier transform of the magnetisation.
    The magnetic interaction vector is is dependent on the scattering geometry
    and the scattering vector is defined as

    .. math::

        \begin{equation}
            {\bf q} = {\bf k}_1 - {\bf k}_0.
        \end{equation}

    The spin-flip and non-spin-flip neutron scattering cross sections can be
    calculated for specific scattering geometries using

    .. math::
        \begin{equation}
         \frac{d\sum}{d\Omega} = \begin{pmatrix}
                                    \frac{d\sum^{++}}{d\Omega} &
                                    \frac{d\sum^{-+}}{d\Omega}\\
                                    \frac{d\sum^{+-}}{d\Omega} &
                                    \frac{d\sum^{--}}{d\Omega}
                                 \end{pmatrix}.
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
    polarisation : turple
        Defines the polarisation direction of the incoming reutron beam
        with respect to the sample reference frame.

    Returns
    -------
    discretisedfield.Field
        Scattering cross section in arbitary units.

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
        ...     return (0,
        ...             np.sin(2 * np.pi * x / q),
        ...             np.cos(2 * np.pi * x / q))
        >>> field = df.Field(mesh, dim=3, value=v_fun, norm=1e5)
        >>> field.plane('z').mpl()
        >>> cs = mag2exp.sans.cross_section(field, method='unpol',
        ...                                 polarisation=(0, 0, 1))
        >>> cs.plane(z=0).real.mpl.scalar()

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
        ...     return (0,
        ...             np.sin(2 * np.pi * x / q),
        ...             np.cos(2 * np.pi * x / q))
        >>> field = df.Field(mesh, dim=3, value=v_fun, norm=1e5)
        >>> field.plane('z').mpl()
        >>> cs = mag2exp.sans.cross_section(field, method='pn',
        ...                                 polarisation=(0, 0, 1))
        >>> cs.plane(z=0).real.mpl.scalar()
    """
    cross_s = _cross_section_matrix(field, polarisation=polarisation)

    if method in ("polarised_pp", "pp"):
        return cross_s.pp
    elif method in ("polarised_pn", "pn"):
        return cross_s.pn
    elif method in ("polarised_np", "np"):
        return cross_s.np
    elif method in ("polarised_nn", "nn"):
        return cross_s.nn
    elif method in ("half_polarised_p", "p"):
        return cross_s.pp + cross_s.pn
    elif method in ("half_polarised_n", "n"):
        return cross_s.nn + cross_s.np
    elif method in ("unpolarised", "unpol"):
        return 0.5 * (cross_s.pp + cross_s.pn + cross_s.np + cross_s.nn)
    else:
        msg = f"Method {method} is unknown."
        raise ValueError(msg)


def chiral_function(field, /, polarisation=(0, 0, 1)):
    r"""Calculation of the chiral function :math:`-2\pi i \chi`.

    The chiral function can be calculated using

    .. math::
        -2\pi i \chi = \frac{d\sum^{+-}}{d\Omega} - \frac{d\sum^{-+}}{d\Omega}

    where :math:`\frac{d\sum^{+-}}{d\Omega}` and
    :math:`\frac{d\sum^{-+}}{d\Omega}` are the spin-flip cross sections.

    Note that this function returns the quantity :math:`-2\pi i \chi`.

    Parameters
    ----------
    field : discretisedfield.field
        Magnetisation field.
    polarisation : turple
        Defines the polarisation direction of the incoming reutron beam
        with respect to the sample reference frame.

    Returns
    -------
    discretisedfield.Field
        Chiral function :math:`-2\pi i \chi`.

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
        ...     return (0,
        ...             np.sin(2 * np.pi * x / q),
        ...             np.cos(2 * np.pi * x / q))
        >>> field = df.Field(mesh, dim=3, value=v_fun, norm=1e5)
        >>> field.plane('z').mpl()
        >>> cf = mag2exp.sans.chiral_function(field,
        ...                                   polarisation=(1, 0, 0))
        >>> cf.plane(z=0).mpl.scalar()
    """
    cross_s = _cross_section_matrix(field, polarisation=polarisation)
    return cross_s.pn - cross_s.np


def _cross_section_matrix(field, /, polarisation):
    m_fft = field.fftn
    m_fft *= field.mesh.dV * 1e16  # TODO:  Normalisation
    q = df.Field(
        m_fft.mesh,
        dim=3,
        value=(
            lambda x: (0, 0, 0) if np.linalg.norm(x) == 0 else x / np.linalg.norm(x)
        ),
    )
    magnetic_interaction = q & m_fft & q

    # Rotation of Pauli matrices
    initial = (0, 0, 1)
    if initial == polarisation:
        r = Rotation.identity()
    else:
        fixed = np.cross(initial, polarisation)
        r = Rotation.align_vectors([polarisation, fixed], [initial, fixed])[0]
    p_x = [[0, 1], [1, 0]]
    p_y = [[0, -1j], [1j, 0]]
    p_z = [[1, 0], [0, -1]]
    p = np.array([p_x, p_y, p_z])
    p_new = np.einsum("ij,ibc->jbc", r.as_matrix(), p)  # Rotate Pauli matrices

    # Apply function to Pauli
    magnetic_interaction_new = np.einsum(
        "ijkl,lbc->ijkbc", magnetic_interaction.array, p_new
    )
    cs = np.power(np.abs(magnetic_interaction_new), 2)
    return df.Field(
        mesh=m_fft.mesh,
        dim=4,
        components=["pp", "np", "pn", "nn"],
        value=cs.reshape([*cs.shape[:3], 4]),
    )
