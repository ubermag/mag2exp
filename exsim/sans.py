import numpy as np


def cross_section(field, /, method, geometry, theta=0):
    """
    Parameters
    ----------
    ...
    """
    if geometry == 'parallel':
        Q = _Q_parallel(field=field, theta=theta)
    elif geometry == 'perpendicular':
        Q = _Q_perpendicular(field=field, theta=theta)
    else:
        msg = f'Geometry {geometry} is unknown.'
        raise ValueError(msg)
    if method == 'polarised_pp':
        return Q.z * Q.z.conjugate
    elif method == 'polarised_nn':
        return Q.z * Q.z.conjugate
    #...
    else:
        msg = f'Method {method} is unknown.'
        raise ValueError(msg)


def _Q_parallel(field, /, theta=0):
    m_p_ft = field.integral(direction='z').fft2()
    Qx = -m_p_ft.x * np.sin(theta)**2 + m_p_ft.y * np.cos(theta) * np.sin(theta)
    Qy = -m_p_ft.y*np.cos(theta)**2 + m_p_ft.x*np.cos(theta)*np.sin(theta)
    Qz = m_p_ft.z
    return Qx << Qy << Qz


def _Q_perpendicular(field, /, theta=0):
    m_p_ft = field.integral(direction='z').fft2()
    Qx = -m_p_ft.x
    Qy = -m_p_ft.y*np.cos(theta)**2 + m_p_ft.z*np.cos(theta)*np.sin(theta)
    Qz = m_p_ft.y*np.cos(theta)*np.sin(theta) - m_p_ft.z*np.sin(theta)**2
    return Qx << Qy << Qz
