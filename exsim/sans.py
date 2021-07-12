import numpy as np


def cross_section(field, /, method, geometry, theta=0):
    """
    Parameters
    ----------
    field : discretisedfield.field
        Magneisation field.
    method : str
        Used to select the relevant cross section.
    geometry : str
        Define the experimental geromtry as field parallel or perpendcular to
        the neutron propagation vector.

    theta : numbers.Real, optional
        The azimuthal angle on the detector of the mometum transfer vector.

    Returns
    -------
    discretisedfield.Field
        Scatering cross section.
    """
    Q = magnetic_interaction_vector(field, geometry=geometry, theta=theta)
    if method in ('polarised_pp', 'pp'):
        return Q.z * Q.z.conjugate
    elif method in ('polarised_nn', 'nn'):
        return Q.z * Q.z.conjugate
    elif method in ('polarised_pn', 'pn'):
        return (Q.x*Q.x.conjugate + Q.y*Q.y.conjugate
                - 1j * (Q.x*Q.y.conjugate - Q.x.conjugate*Q.y))
    elif method in ('polarised_np', 'np'):
        return (Q.x*Q.x.conjugate + Q.y*Q.y.conjugate
                + 1j * (Q.x*Q.y.conjugate - Q.x.conjugate*Q.y))
    elif method in ('half_polarised_p', 'p'):
        pp = cross_section(field, method='polarised_pp',
                           geometry=geometry, theta=theta)
        pn = cross_section(field, method='polarised_pn',
                           geometry=geometry, theta=theta)
        return pp + pn
    elif method in ('half_polarised_n', 'n'):
        nn = cross_section(field, method='polarised_nn',
                           geometry=geometry, theta=theta)
        np = cross_section(field, method='polarised_np',
                           geometry=geometry, theta=theta)
        return nn + np
    elif method in ('unpolarised', 'unpol'):
        p = cross_section(field, method='half_polarised_p',
                           geometry=geometry, theta=theta)
        n = cross_section(field, method='half_polarised_n',
                           geometry=geometry, theta=theta)
        return 0.5 * (p + n)
    else:
        msg = f'Method {method} is unknown.'
        raise ValueError(msg)


def magnetic_interaction_vector(field, /, geometry, theta=0):
    """
    Parameters
    ----------
    field : discretisedfield.field
        Magneisation field.
    geometry : str
        Define the experimental geromtry as field parallel or perpendcular to
        the neutron propagation vector.
    theta : numbers.Real, optional
        The azimuthal angle on the detector of the mometum transfer vector.

    Returns
    -------
    discretisedfield.Field
        Magnetic interaction vector.
    """
    if geometry == 'parallel':
        Q = _Q_parallel(field, theta=theta)
    elif geometry == 'perpendicular':
        Q = _Q_perpendicular(field, theta=theta)
    else:
        msg = f'Geometry {geometry} is unknown.'
        raise ValueError(msg)
    return Q


def _Q_parallel(field, /, theta=0):
    """
    Parameters
    ----------
    field : discretisedfield.field
        Magneisation field.
    theta : numbers.Real, optional
        The azimuthal angle on the detector of the mometum transfer vector.

    Returns
    -------
    discretisedfield.Field
        Magnetic interaction vector.
    """
    m_p_ft = field.integral(direction='z').fft2()
    Qx = -m_p_ft.x * np.sin(theta)**2 + m_p_ft.y * np.cos(theta) * np.sin(theta)
    Qy = -m_p_ft.y*np.cos(theta)**2 + m_p_ft.x*np.cos(theta)*np.sin(theta)
    Qz = m_p_ft.z
    return Qx << Qy << Qz


def _Q_perpendicular(field, /, theta=0):
    """
    Parameters
    ----------
    field : discretisedfield.field
        Magneisation field.
    theta : numbers.Real, optional
        The azimuthal angle on the detector of the mometum transfer vector.

    Returns
    -------
    discretisedfield.Field
        Magnetic interaction vector.
    """
    m_p_ft = field.integral(direction='z').fft2()
    Qx = -m_p_ft.x
    Qy = -m_p_ft.y*np.cos(theta)**2 + m_p_ft.z*np.cos(theta)*np.sin(theta)
    Qz = m_p_ft.y*np.cos(theta)*np.sin(theta) - m_p_ft.z*np.sin(theta)**2
    return Qx << Qy << Qz


def chiral_function(field, /, geometry, theta=0):
    """
    Parameters
    ----------
    field : discretisedfield.field
        Magneisation field.
    geometry : str
        Define the experimental geromtry as field parallel or perpendcular to
        the neutron propagation vector.
    theta : numbers.Real, optional
        The azimuthal angle on the detector of the mometum transfer vector.

    Returns
    -------
    discretisedfield.Field
        Chiral function.
    """
    Q = magnetic_interaction_vector(field, geometry=geometry, theta=theta)
    return Q.x*Q.y.conjugate - Q.x.conjugate*Q.y
