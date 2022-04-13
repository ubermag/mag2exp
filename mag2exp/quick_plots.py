import matplotlib
import matplotlib.pyplot as plt
import ubermagutil.units as uu

from . import ltem, mfm, sans, x_ray


def ltem_phase(field, /, kcx=0.1, kcy=0.1):
    r"""Quickplot of the magnetic phase shift.

    The phase is calculated using the :code:`ltem.phase`
    function and plotted using :code:`mpl.scalar`.

    .. seealso:: :py:func:`mag2exp.ltem.phase`
    """
    phase, _ = ltem.phase(field, kcx=kcx, kcy=kcy)
    phase.mpl.scalar(
        cmap="gray", interpolation="spline16", colorbar_label=r"$\phi$ (radians)"
    )


def ltem_ft_phase(field, /, kcx=0.1, kcy=0.1):
    r"""Quickplot of the magnetic phase shift in Fourier space.

    The Fourier transform of the phase is calculated using the
    :code:`ltem.phase` function and plotted using
    :code:`mpl.scalar`.

    .. seealso:: :py:func:`mag2exp.ltem.phase`
    """
    _, ft_phase = ltem.phase(field, kcx=kcx, kcy=kcy)
    fig, ax = plt.subplots()
    (ft_phase.conjugate * ft_phase).plane("z").real.mpl.scalar(
        ax=ax,
        cmap="gray",
        interpolation="spline16",
        colorbar_label=(
            r"$\left\vert\widetilde{\phi}\right\vert^2$" "(radians$^{-2}$)"
        ),
    )
    multiplier = uu.si_max_multiplier(ft_phase.mesh.region.edges)
    ax.add_patch(
        matplotlib.patches.Ellipse(
            xy=(0, 0),
            width=ft_phase.mesh.cell[0] * kcx * 2 / multiplier,
            height=ft_phase.mesh.cell[1] * kcy * 2 / multiplier,
            edgecolor="red",
            facecolor="none",
            linewidth=3,
            label="Tikhonov filter",
        )
    )
    ax.legend(frameon=True)


def ltem_defocus(
    field, /, kcx=0.1, kcy=0.1, cs=0, df_length=0.2e-3, voltage=None, wavelength=None
):
    r"""Quickplot of the LTEM defocus image.

    The phase is calculated using the :code:`ltem.phase`
    function and propagated to the image plane using
    :code:`ltem.defocus_image`.	This is then plotted using
    :code:`mpl.scalar`.

    .. seealso::
        :py:func:`mag2exp.ltem.phase`
        :py:func:`mag2exp.ltem.defocus_image`
    """
    phase, _ = ltem.phase(field, kcx=kcx, kcy=kcy)
    defocus = ltem.defocus_image(
        phase, cs=cs, df_length=df_length, voltage=voltage, wavelength=wavelength
    )
    defocus.mpl.scalar(
        cmap="gray", interpolation="spline16", colorbar_label="Intensity (counts)"
    )


def ltem_integrated_mfd(field, /, kcx=0.1, kcy=0.1):
    r"""Quickplot of the LTEM integrated magnetic flux density.

    The phase is calculated using the :code:`ltem.phase`
    function and the integrated magnetic flux density
    :code:`ltem.integrated_magnetic_flux_density`.
    This is then plotted using :code:`mpl.lightness` and :code:`mpl.vector`.

    .. seealso::
        :py:func:`mag2exp.ltem.phase`
        :py:func:`mag2exp.ltem.integrated_magnetic_flux_density`

    """
    phase, _ = ltem.phase(field, kcx=kcx, kcy=kcy)
    imf = ltem.integrated_magnetic_flux_density(phase)
    fig, ax = plt.subplots()
    imf.mpl.lightness(
        ax=ax,
        clim=[0, 0.5],
        interpolation="spline16",
        colorwheel_args=dict(width=0.75, height=0.75),
        colorwheel_xlabel=r"$m_x$",
        colorwheel_ylabel=r"$m_y$",
    )
    imf.mpl.vector(ax=ax, use_color=False, color="w")


def mfm_phase_shift(
    system, /, tip_m=(0, 0, 0), quality=650, k=3, tip_q=0, fwhm=None, z0=0
):
    r"""Quickplot of the magnetic phase shift.

    The phase is calculated using the :code:`ltem.phase`
    function and plotted using :code:`mpl.scalar`.

    .. seealso:: :py:func:`mag2exp.mfm.phase_shift`
    """
    phase_shift = mfm.phase_shift(
        system, tip_m=tip_m, quality=quality, k=k, tip_q=tip_q, fwhm=fwhm
    )
    phase_shift_p = phase_shift.plane(z=z0)
    phase_shift_p.mpl.scalar(
        interpolation="spline16", colorbar_label=r"Phase shift (radians.)"
    )


def x_ray_holography(field, /, fwhm=None):
    r"""Quickplot of the magnetic phase shift.

    The phase is calculated using the :code:`ltem.phase`
    function and plotted using :code:`mpl.scalar`.

    .. seealso:: :py:func:`mag2exp.x_ray.holography`
    """
    holo = x_ray.holography(field, fwhm=fwhm)
    holo.mpl.scalar(
        cmap="RdBu",
        interpolation="spline16",
        colorbar_label=r"Integrated Magnetisation (A)",
    )


def saxs(field):
    r"""Quickplot of the small angle x-ray scattering pattern.

    The small angle x-ray scattering pattern is calculated using the
    :code:`x_ray.saxs` function and plotted using :code:`mpl.scalar`.

    .. seealso:: :py:func:`mag2exp.x_ray.saxs`

    """
    cs = x_ray.saxs(field)
    cs.mpl.scalar(
        cmap="gray", interpolation="spline16", colorbar_label=r"Intensity (arb.)"
    )


def sans_cross_section(field, /, method, polarisation=(0, 0, 1)):
    r"""Quickplot of the small angle neutron scattering pattern.

    The small angle neutron scattering pattern is calculated using the
    :code:`sans.cross_section` function and plotted using :code:`mpl.scalar`.

    .. seealso:: :py:func:`mag2exp.sans.cross_section`

    """
    cs = sans.cross_section(field, method=method, polarisation=polarisation)
    cs.plane(z=0).mpl.scalar(
        cmap="gray", interpolation="spline16", colorbar_label=r"Intensity (arb.)"
    )


def sans_chiral_function(field, /, polarisation=(0, 0, 1)):
    r"""Quickplot of the small angle neutron scattering chiral function.

    The small angle neutron scattering chiral function is calculated using the
    :code:`sans.chiral_function` function and the imaginary component
    plotted using :code:`mpl.scalar`.

    .. seealso:: :py:func:`mag2exp.sans.chiral_function`
    """
    cf = sans.chiral_function(field, polarisation=polarisation)
    cf.plane(z=0).mpl.scalar(
        cmap="gray", interpolation="spline16", colorbar_label=r"Cross section (arb.)"
    )
