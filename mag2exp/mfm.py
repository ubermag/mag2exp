"""MFM submodule.

Module for calculation of Magnetic Force Microscopy related quantities.
"""

import micromagneticmodel as mm
import oommfc as oc

import mag2exp


def phase_shift(system, /, tip_m=(0, 0, 0), quality=650, k=3, tip_q=0, fwhm=None):
    r"""Calculation of the phase shift of an MFM tip.

    The contrast in MFM images originates from the magnetic interaction between
    the magnetic tip of an oscillating cantilever and the samples stray field.
    As MFM is based on the stray field outside of a sample, an 'airbox' method
    should be used. This mean that the saturation magnetisation should be set
    to zero in the region outside the sample in which we wish to perform these
    MFM measurements.

    The magnetic cantilever is driven to oscillate near its resonant frequency
    when there is no stray field. In the presence of a stray magnetic field the
    phase shift of MFM cantilever is given by

    .. math::
        \Delta \phi = \frac{Q\mu_0}{k} \left( q \frac{\partial
        {\bf H}_{sz}}{\partial z} + {\bf M}_t \cdot
        \frac{\partial^2{\bf H}_{s}}{\partial z^2} \right),

    where
    :math:`Q` is the quality factor of the cantilever,
    :math:`k` is the spring constant of the cantilever in
    :math:`\textrm{Nm}^{-1}`, :math:`q` is the effective magnetic monopole
    moment of the tip in :math:`\textrm{Am}^{-2}`,
    :math:`{\bf M}_t` is the effective magnetic dipole moment of the tip in
    :math:`\textrm{Am}^{-1}` and,
    :math:`{\bf H}_{sz}` is the magnetic field due to the sample in
    :math:`\textrm{Am}^{-1}`.


    Parameters
    ----------
    system : micromagneticmodel.System
        Micromagnetic system which must include the magnetisation
        configuration and an energy equation which includes demagnetisation.

    tip_m : numbers.Real, array_like
        The effective magnetic dipole moment of the tip in the :math:`x`,
        :math:`y`, and :math:`z` directions, respectively. Values are given in
        units of :math:`\textrm{Am}^{-1}`.

    quality : numbers.Real
        The quality factor of the tip. Should be a real, positive number.

    k : numbers.Real
        The spring factor of the MFM cantilever. This should be a real,
        positive, non-zero number.

    tip_q : numbers.Real
        The effective magnetic monopole moment of the tip in units of
        :math:`\textrm{Am}^{-2}`.

    Returns
    -------
    discretisedfield.Field
        Phase shift of MFM cantilever.

    Examples
    --------

    .. plot::
        :context: close-figs

        1. Visualising MFM with ``matplotlib``.

        >>> import discretisedfield as df
        >>> import micromagneticmodel as mm
        >>> import mag2exp
        >>> mesh = df.Mesh(p1=(-25e-9, -25e-9, -2e-9),
        ...                p2=(25e-9, 25e-9, 50e-9),
        ...                cell=(1e-9, 1e-9, 2e-9))
        >>> def v_fun(point):
        ...     x, y, z = point
        ...     if x < -2e-9:
        ...         return (0, 0, 1)
        ...     elif x < 2e-9:
        ...         return (0, 1, 0)
        ...     else:
        ...         return (0, 0, -1)
        >>> def Ms_fun(pos):
        ...     x, y, z = pos
        ...     if (z < 0):
        ...         return 384e3
        ...     else:
        ...         return 0
        >>> system = mm.System(name='Box2')
        >>> system.energy = mm.Demag()
        >>> system.m = df.Field(mesh, dim=3, value=v_fun, norm=Ms_fun)
        >>> ps = mag2exp.mfm.phase_shift(system, tip_m=(0, 0, 1e-16))
        Running OOMMF...
        >>> ps.plane(z=10e-9).mpl.scalar()
        >>> ps.plane(z=40e-9).mpl.scalar()


    .. plot::
        :context: close-figs

        2. Visualising in-plane MFM with ``matplotlib``.

        >>> import discretisedfield as df
        >>> import micromagneticmodel as mm
        >>> import mag2exp
        >>> mesh = df.Mesh(p1=(-25e-9, -25e-9, -2e-9),
        ...                p2=(25e-9, 25e-9, 50e-9),
        ...                cell=(1e-9, 1e-9, 2e-9))
        >>> def v_fun(point):
        ...     x, y, z = point
        ...     if x < -2e-9:
        ...         return (0, 0, 1)
        ...     elif x < 2e-9:
        ...         return (0, 1, 0)
        ...     else:
        ...         return (0, 0, -1)
        >>> def Ms_fun(pos):
        ...     x, y, z = pos
        ...     if (z < 0):
        ...         return 384e3
        ...     else:
        ...         return 0
        >>> system = mm.System(name='Box2')
        >>> system.energy = mm.Demag()
        >>> system.m = df.Field(mesh, dim=3, value=v_fun, norm=Ms_fun)
        >>> ps = mag2exp.mfm.phase_shift(system, tip_m=(1e-16, 0, 0))
        Running OOMMF...
        >>> ps.plane(z=10e-9).mpl.scalar()
        >>> ps.plane(z=40e-9).mpl.scalar()


    .. plot::
        :context: close-figs

        3. Visualising monopole moment MFM with ``matplotlib``.

        >>> import discretisedfield as df
        >>> import micromagneticmodel as mm
        >>> import mag2exp
        >>> mesh = df.Mesh(p1=(-25e-9, -25e-9, -2e-9),
        ...                p2=(25e-9, 25e-9, 50e-9),
        ...                cell=(1e-9, 1e-9, 2e-9))
        >>> def v_fun(point):
        ...     x, y, z = point
        ...     if x < -2e-9:
        ...         return (0, 0, 1)
        ...     elif x < 2e-9:
        ...         return (0, 1, 0)
        ...     else:
        ...         return (0, 0, -1)
        >>> def Ms_fun(pos):
        ...     x, y, z = pos
        ...     if (z < 0):
        ...         return 384e3
        ...     else:
        ...         return 0
        >>> system = mm.System(name='Box2')
        >>> system.energy = mm.Demag()
        >>> system.m = df.Field(mesh, dim=3, value=v_fun, norm=Ms_fun)
        >>> ps = mag2exp.mfm.phase_shift(system, tip_q=1e-9)
        Running OOMMF...
        >>> ps.plane(z=10e-9).mpl.scalar()
        >>> ps.plane(z=40e-9).mpl.scalar()
    """

    if k <= 0:
        msg = "`k` has to be a positive non-zero number."
        raise RuntimeError(msg)

    stray_field = oc.compute(system.energy.demag.effective_field, system)
    dh_dz = stray_field.derivative("z", n=1)
    d2h_dz2 = stray_field.derivative("z", n=2)
    phase_shift = (quality * mm.consts.mu0 / k) * (tip_q * dh_dz.z + d2h_dz2 @ tip_m)
    if fwhm is not None:
        phase_shift = mag2exp.util.gaussian_filter(phase_shift, fwhm=fwhm)
    return phase_shift
