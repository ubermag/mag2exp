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
        magnetic_interaction = _magnetic_interaction_parallel(field)
    elif geometry == 'perpendicular':
        magnetic_interaction = _magnetic_interaction_perpendicular(field)
    elif geometry == 'perp_z':
        magnetic_interaction = _magnetic_interaction_perpendicular_z(field)
    elif geometry == 'perp_z_2':
        magnetic_interaction = _magnetic_interaction_perpendicular_z_2(field)
    else:
        msg = f'Geometry {geometry} is unknown.'
        raise ValueError(msg)
    return magnetic_interaction


def _magnetic_interaction_parallel(field):
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
    magnetic_interaction = magnetic_interaction_vector(field,
                                                       geometry=geometry)
    return (magnetic_interaction.x * magnetic_interaction.y.conjugate
            - magnetic_interaction.x.conjugate * magnetic_interaction.y)
