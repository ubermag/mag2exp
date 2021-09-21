"""MOKE submodule.

Module for calculation of magneto-optical Kerr effect
quantities.
"""
import numpy as np
import discretisedfield as df


def _calculate_A(theta_j, nj, Q, field):
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

    s = np.shape(m_arr)
    mx_arr = m_arr[..., 0].flatten()
    my_arr = m_arr[..., 1].flatten()
    mz_arr = m_arr[..., 2].flatten()

    A = []
    for (mx, my, mz) in zip(mx_arr, my_arr, mz_arr):
        A.append([[1, 0, 1, 0],
                  [(1j*Q)*(a_yj*my*(1+a_zj**2)/a_zj - mz*a_yj**2)/2,
                   a_zj,
                   -(1j*Q)/2 * (a_yj*my*(1+a_zj**2)/(a_zj) + mz*a_yj**2),
                   -a_zj],
                  [-(1j*Q*nj)/2 * (my*a_yj + mz*a_zj),
                   -nj,
                   -(1j*Q*nj)/2 * (my*a_yj - mz*a_zj),
                   -nj],
                  [nj*a_zj,
                   -(1j*Q*nj)/2 * (my*a_yj/a_zj - mz),
                   -nj*a_zj,
                   (1j*Q*nj)/2 * (my*a_yj/a_zj + mz)]])
    return np.reshape(A, (*s[0:2], 4, 4))


def _calculate_D(theta_j, nj, Q, dj, wavelength, field):
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

    s = np.shape(m_arr)
    mx_arr = m_arr[..., 0].flatten()
    my_arr = m_arr[..., 1].flatten()
    mz_arr = m_arr[..., 2].flatten()

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
    return np.reshape(D, (*s[0:2], 4, 4))


def _angle_snell(theta0, n0, n1):
    r"""Calculation of the new angle from Snell's law.

    Snell's is

    .. math::
        n_0 \sin\theta_0 = n_1 \sin\theta_1

    where :math:`n_0` and :math:`n_1` are the refractive indexes of layer 0
    and 1 repectively. :math:`\theta_0` and :math:`\theta_1` are the angles
    of light in layer 0 and 1, with respect to the surface normal.
    """
    return np.arcsin((n0*np.sin(theta0))/n1)


def _calculate_M(field, theta_0, n, Q, wavelength):
    r""" Product matrix.

    The product matrix is the matrix used to describe light propagation in
    a magnetic multilayer system. In this function, it has the form

    .. math::
        \begin{equation}
            M = A_f^{-1} \prod_{j} A_{j} D_j A_j^{-1} A_f,
        \end{equation}

    where :math:`A_j` is the boundary matrix for the :math:`j`th layer,
    :math:`D_j` is the propagation matrix for the :math:`j`th layer,
    and :math:`A_f` is the boundary layer matrix for free space.
    """
    # Free space
    A = _calculate_A(theta_0, 1, 0, field.plane('z'))
    M = np.linalg.inv(A)

    # Sample
    z_min = field.mesh.index2point((0, 0, 0))[2]
    z_max = field.mesh.index2point(np.subtract(field.mesh.n, 1))[2]
    z_step = field.mesh.cell[2]
    values = np.arange(z_min, z_max+1e-20, z_step)

    theta = _angle_snell(theta_0, 1, n)
    for z in values:  # Think about direction of loop
        A = _calculate_A(theta, n, Q, field.plane(z=z))
        D = _calculate_D(theta, n, Q, field.mesh.dz,
                         wavelength, field.plane(z=z))
        M = np.matmul(M, A)
        M = np.matmul(M, D)
        M = np.matmul(M, np.linalg.inv(A))

    # Free space
    theta = theta_0
    A = _calculate_A(theta, 1, 0, field.plane('z'))
    M = np.matmul(M, A)
    return M


def _M_to_r(M):
    r"""Product matrix to reflection matrix.
    """
    G_matrix = M[:, :, 0:2, 0:2]
    I_matrix = M[:, :, 2:4, 0:2]
    G_inv = np.linalg.inv(G_matrix)
    return np.matmul(I_matrix, G_inv)


def _M_to_t(M):
    r"""Product matrix to transmission matrix.
    """
    G_matrix = M[:, :, 0:2, 0:2]
    G_inv = np.linalg.inv(G_matrix)
    return G_inv


def intensity(field, theta, n, Q, wavelength, E_i, mode='reflection'):
    r"""MOKE image.
    """
    E_f = e_field(field, theta, n, Q, wavelength, E_i, mode=mode)
    return abs(E_f)**2


def kerr_angle(field, theta, n, Q, wavelength, E_i, mode='reflection'):
    r"""Kerr angle.
    """
    M = _calculate_M(field, theta, n, Q, wavelength)
    if mode in ('reflection', 'r'):
        m = _M_to_r(M)
    elif mode in ('transmission', 't'):
        m = _M_to_t(M)
    else:
        msg = f'Mode {mode} is unknown.'
        raise ValueError(msg)

    k_s = -m[..., 0, 0]/m[..., 0, 0]
    k_p = -m[..., 0, 0]/m[..., 1, 1]

    reshape((*m.shape[0:2], 1, 2))

    return df.Field(mesh=field.integral('z').mesh, dim=1,
                    value=E_f,
                    components=['s', 'p'])


def e_field(field, theta, n, Q, wavelength, E_i, mode='reflection'):
    r"""Kerr angle.
    """
    M = _calculate_M(field, theta, n, Q, wavelength)
    if mode in ('reflection', 'r'):
        m = _M_to_r(M)
    elif mode in ('transmission', 't'):
        m = _M_to_t(M)
    else:
        msg = f'Mode {mode} is unknown.'
        raise ValueError(msg)

    E_f = np.matmul(m, E_i).reshape((*m.shape[0:2], 1, 2))

    return df.Field(mesh=field.integral('z').mesh, dim=2,
                    value=E_f,
                    components=['s', 'p'])
