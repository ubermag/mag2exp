import numpy as np
import discretisedfield as df


def cross_section(field, /, method, geometry):
    """
    Parameters
    ----------
    field : discretisedfield.field
        Magnetisation field.
    method : str
        Used to select the relevant cross section.
    geometry : str
        Define the experimental geometry as field parallel or perpendicular to
        the neutron propagation vector.

    Returns
    -------
    discretisedfield.Field
        Scattering cross section.
    """
    Q = magnetic_interaction_vector(field, geometry=geometry)
    if method in ('polarised_pp', 'pp'):
        return Q.z * Q.z.conjugate  # is this abs(Q.z)**2?
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
    """
    Parameters
    ----------
    field : discretisedfield.field
        Magnetisation field.
    geometry : str
        Define the experimental geometry as field parallel or perpendicular to
        the neutron propagation vector.

    Returns
    -------
    discretisedfield.Field
        Magnetic interaction vector.
    """
    if geometry == 'parallel':
        Q = _Q_parallel(field)
    elif geometry == 'perpendicular':
        Q = _Q_perpendicular(field)
    elif geometry == 'perp_z':
        Q = _Q_perpendicular_z(field)
    elif geometry == 'perp_z_2':
        Q = _Q_perpendicular_z_2(field)
    else:
        msg = f'Geometry {geometry} is unknown.'
        raise ValueError(msg)
    return Q


def _Q_parallel(field):
    """
    Parameters
    ----------
    field : discretisedfield.field
        Magnetisation field.

    Returns
    -------
    discretisedfield.Field
        Magnetic interaction vector.
    """
    m_p_ft = (field * df.dz).integral(direction='z').fft2()
    theta = df.Field(m_p_ft.mesh, dim=1,
                     value=lambda x: np.arctan2(x[1], x[0]))
    Qx = (-m_p_ft.x * np.sin(theta.array)**2 +
          m_p_ft.y * np.cos(theta.array) * np.sin(theta.array))
    Qy = (-m_p_ft.y*np.cos(theta.array)**2 +
          m_p_ft.x*np.cos(theta.array)*np.sin(theta.array))
    Qz = m_p_ft.z
    return Qx << Qy << Qz


def _Q_perpendicular(field):
    """
    Parameters
    ----------
    field : discretisedfield.field
        Magneisation field.

    Returns
    -------
    discretisedfield.Field
        Magnetic interaction vector.
    """
    m_p_ft = (field * df.dx).integral(direction='x').fft2()
    theta = df.Field(m_p_ft.mesh, dim=1,
                     value=lambda x: np.arctan2(x[2], x[1]))
    Qx = -m_p_ft.x
    Qy = (-m_p_ft.y*np.cos(theta.array)**2 +
          m_p_ft.z*np.cos(theta.array)*np.sin(theta.array))
    Qz = (m_p_ft.y*np.cos(theta.array)*np.sin(theta.array) -
          m_p_ft.z*np.sin(theta.array)**2)
    return Qx << Qy << Qz


def _Q_perpendicular_z(field):
    """
    Parameters
    ----------
    field : discretisedfield.field
        Magneisation field.

    Returns
    -------
    discretisedfield.Field
        Magnetic interaction vector.
    """
    m_p_ft = (field * df.dz).integral(direction='z').fft2()
    theta = df.Field(m_p_ft.mesh, dim=1,
                     value=lambda x: np.arctan2(x[1], x[0]))
    Qx = (-m_p_ft.y * np.sin(theta.array)**2 +
          m_p_ft.z * np.cos(theta.array) * np.sin(theta.array))
    Qy = (-m_p_ft.z*np.cos(theta.array)**2 +
          m_p_ft.y*np.cos(theta.array)*np.sin(theta.array))
    Qz = m_p_ft.x
    return Qx << Qy << Qz


def _Q_perpendicular_z_2(field):
    """
    Parameters
    ----------
    field : discretisedfield.field
        Magnetisation field.

    Returns
    -------
    discretisedfield.Field
        Magnetic interaction vector.
    """
    m_p_ft = (field * df.dz).integral(direction='z').fft2()
    theta = df.Field(m_p_ft.mesh, dim=1,
                     value=lambda x: np.arctan2(x[1], x[0]))
    Qx = (m_p_ft.y * np.sin(theta.array)**2 -
          m_p_ft.z * np.cos(theta.array) * np.sin(theta.array))
    Qy = (m_p_ft.z*np.cos(theta.array)**2 -
          m_p_ft.y*np.cos(theta.array)*np.sin(theta.array))
    Qz = -m_p_ft.x
    return Qy << Qz << Qx


def chiral_function(field, /, geometry):
    """
    Parameters
    ----------
    field : discretisedfield.field
        Magnetisation field.
    geometry : str
        Define the experimental geometry as field parallel or perpendicular to
        the neutron propagation vector.

    Returns
    -------
    discretisedfield.Field
        Chiral function.
    """
    Q = magnetic_interaction_vector(field, geometry=geometry)
    return Q.x*Q.y.conjugate - Q.x.conjugate*Q.y
