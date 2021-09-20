"""MOKE submodule.

Module for calculation of magneto-optical Kerr effect
quantities.
"""
import numpy as np


def calculate_A(theta_j, nj, Q, field):
    r"""Calculation of the boundary matrix.

    .. math::
        \begin{equation}
            A_{j} =
            \begin{pmatrix}
            1 & 0 & 1 & 0 \\
            \frac{iQ\alpha^2_{yi}}{2} \left(m_y \frac{1+\alpha^2_{zi}}
            {\alpha_{yi}\alpha_{zi}} - m_z\right) & \alpha_{zi} &
            -\frac{iQ\alpha^2_{yi}}{2} \left(m_y \frac{1+\alpha^2_{zi}}
            {\alpha_{yi}\alpha_{zi}} + m_z\right) & -\alpha_{zi} \\
            -\frac{in_jQ}{2} \left(m_y \alpha_{yi} + m_z \alpha_{zi} \right) &
            -n_j &
            -\frac{in_jQ}{2} \left(m_y \alpha_{yi} - m_z \alpha_{zi} \right) &
            -n_j \\
            n_j\alpha_{zj} & -\frac{in_jQ}{2} \left(m_y \frac{\alpha_{yi}}
            {\alpha_{zi}} - m_z \right) & -n_j \alpha_{zj} &
            \frac{in_jQ}{2} \left(m_y \frac{\alpha_{yi}}
            {\alpha_{zi}} + m_z \right)
            \end{pmatrix}
        \end{equation}

    where :math:`\alpha_{yi}=\sin\theta_j, \alpha_{zi}=\cos\theta_j`,
    `theta_j` is the complex refractive angle, :math:`Q` is the Voight
    parameter and :math:`n_j` is the refractive index of the :math:`j`th layer.
    The angle is measure with respect to the :math:`z` axis and is calculated
    using Snell's law.
    :math:`m_x`, :math:`m_y` and
    :math:`m_z` are the normalized magnetisation.
    """
    m_arr = field.orientation.array
    a_yj = np.sin(theta_j)
    a_zj = np.cos(theta_j)

    mx_arr = m_arr[..., 0].flatten()
    my_arr = m_arr[..., 1].flatten()
    mz_arr = m_arr[..., 2].flatten()
    s = np.shape(mx_arr)

    A = []

    for (mx, my, mz) in zip(mx_arr, my_arr, mz_arr):
        A.append([[1, 0, 1, 0],
                  [(1j*Q*a_yj**2)*(my*(1+a_zj**2)/(a_yj*a_zj) - mz)/2,
                   a_zj,
                   -(1j*Q*a_yj**2)/2 * (my*(1+a_zj**2)/(a_yj*a_zj) + mz),
                   -a_zj],
                  [-(1j*Q*nj)/2 * (my*a_yj + mz*a_zj),
                   -nj,
                   -(1j*Q*nj)/2 * (my*a_yj - mz*a_zj),
                   -nj],
                  [nj*a_zj,
                   -(1j*Q*nj)/2 * (my*a_yj/a_zj - mz),
                   -nj*a_zj,
                   (1j*Q*nj)/2 * (my*a_yj/a_zj + mz)]])
    return np.reshape(A, (*s, 4, 4))


def calculate_D(theta_j, nj, Q, dj, wavelength, field):
    r"""Calculation of the propagation matrix.

    .. math::
        \begin{equation}
            D_{j} =
            \begin{pmatrix}
            U\cos\delta^i & U\sin\delta^i & 0 & 0 \\
            -U\sin\delta^i & U\cos\delta^i & 0 & 0 \\
            0 & 0 & U^{-1}\cos\delta^r & U^{-1}\sin\delta^r \\
            0 & 0 & -U^{-1}\sin\delta^r & U^{-1}\cos\delta^r
            \end{pmatrix}
        \end{equation}

    where

    .. math::
        \begin{align}
            U &= \exp\left(\frac{-i2\pi n_j \alpha_{zj} d_j}{\lambda} \right)\\
            \delta^i &= -\frac{\pi n_j Q d_j g^i}{\lambda \alpha_{zj}} \\
            \delta^r &= -\frac{\pi n_j Q d_j g^r}{\lambda \alpha_{zj}} \\
            g^i &= m_z \alpha_{zj} + m_y \alpha_{yj} \\
            g^r &= m_z \alpha_{zj} - m_y \alpha_{yj}
        \end{align}

    where :math:`\alpha_{yi}=\sin\theta_j, \alpha_{zi}=\cos\theta_j`,
    `theta_j` is the complex refractive angle, :math:`Q` is the Voight
    parameter, :math:`n_j` is the refractive index, and :math:`d_j`
    is the thickness of the :math:`j`th layer.
    The angle is measure with respect to the :math:`z` axis and is calculated
    using Snell's law.
    :math:`m_x`, :math:`m_y` and
    :math:`m_z` are the normalized magnetisation.
    :math:`\lambda` is the wavelength of light.
    """
    m_arr = field.orientation.array
    a_yj = np.sin(theta_j)
    a_zj = np.cos(theta_j)

    mx_arr = m_arr[..., 0].flatten()
    my_arr = m_arr[..., 1].flatten()
    mz_arr = m_arr[..., 2].flatten()
    s = np.shape(mx_arr)

    D = []
    for (mx, my, mz) in zip(mx_arr, my_arr, mz_arr):
        gi = mz*a_zj + my*a_yj
        gr = mz*a_zj - my*a_yj
        di = -np.pi*nj*Q*dj*gi/(wavelength*a_zj)
        dr = -np.pi*nj*Q*dj*gr/(wavelength*a_zj)
        U = np.exp(-2j*np.pi*nj*a_zj*dj/wavelength)

        D.append([[U*np.cos(di), U*np.sin(di), 0, 0],
                  [-U*np.sin(di), U*np.cos(di), 0, 0],
                  [0, 0, np.cos(dr)/U, np.sin(dr)/U],
                  [0, 0, -np.sin(dr)/U, np.cos(dr)]])
    return np.reshape(D, (*s, 4, 4))


def angle_snell(theta0, n0, n1):
    r"""Calculation of the new angle from Snell's law.

    Snell's is

    .. math::
        n_0 \sin\theta_0 = n_1 \sin\theta_1

    where :math:`n_0` and :math:`n_1` are the refractive indexes of layer 0
    and 1 repectively. :math:`\theta_0` and :math:`\theta_1` are the angles
    of light in layer 0 and 1, with respect to the surface normal.
    """
    return np.arcsin((n0*np.sin(theta0))/n1)


def calculate_M(field, theta, n, Q, wavelength):
    r""" Mueller matrix.
    """
    # Free space
    A = calculate_A(theta, 1, 0, field.plane('z'))
    M = np.linalg.inv(A)

    # Sample
    z_min = field.mesh.index2point((0, 0, 0))[2]
    z_max = field.mesh.index2point(np.subtract(field.mesh.n, 1))[2]
    z_step = field.mesh.cell[2]
    values = np.arange(z_min, z_max+1e-20, z_step)

    theta = angle_snell(theta, 1, n)
    for z in values:  # Think about direction of loop
        A = calculate_A(theta, n, Q, field.plane(z=z))
        M = np.matmul(M, A)
        D = calculate_D(theta, n, Q, field.mesh.dz,
                        wavelength, field.plane(z=z))
        M = np.matmul(M, D)
        A = calculate_A(theta, n, Q, field.plane(z=z))
        M = np.matmul(M, np.linalg.inv(A))

    # Free space
    theta = angle_snell(theta, n, 1)
    A = calculate_A(theta, 1, 0, field.plane('z')))
    M = np.matmul(M, A)
    return M
