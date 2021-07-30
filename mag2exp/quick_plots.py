import matplotlib.pyplot as plt
import matplotlib
import ubermagutil.units as uu
from . import ltem
from . import mfm
from . import x_ray


def ltem_phase(field, /, kcx=0.1, kcy=0.1):
    r"""Quickplot of the magnetic phase shift.

    The phase is calculated using the :code:`ltem.phase`
    function and plotted using :code:`mpl.scalar`.

    .. seealso:: :py:func:`~ltem.phase`
    """
    phase, _ = ltem.phase(field, kcx=kcx, kcy=kcy)
    phase.real.mpl.scalar(cmap='gray',
                          interpolation='spline16',
                          colorbar_label=r'$\phi$ (radians)')


def ltem_ft_phase(field, /, kcx=0.1, kcy=0.1):
    r"""Quickplot of the magnetic phase shift in Fourier space.

    The Fourier transform of the phase is calculated using the
    :code:`ltem.phase` function and plotted using
    :code:`mpl.scalar`.

    .. seealso:: :py:func:`~ltem.phase`
    """
    _, ft_phase = ltem.phase(field, kcx=kcx, kcy=kcy)
    fig, ax = plt.subplots()
    (ft_phase.conjugate * ft_phase).plane('z').real.mpl.scalar(
        ax=ax, cmap='gray', interpolation='spline16',
        colorbar_label=r'$\widetilde{\phi}$ (radians$^{-1}$)')
    multiplier = uu.si_max_multiplier(ft_phase.mesh.region.edges)
    ax.add_patch(matplotlib.patches.Ellipse(
        xy=(0, 0),
        width=ft_phase.mesh.cell[0] * kcx * 2 / multiplier,
        height=ft_phase.mesh.cell[1] * kcy * 2 / multiplier,
        edgecolor='red',
        facecolor='none',
        linewidth=3,
        label='Tikhonov filter'))
    ax.legend(frameon=True)


def ltem_defocus(field, /, kcx=0.1, kcy=0.1,
                 cs=0, df_length=0.2e-3, voltage=None, wavelength=None):
    r"""Quickplot of the LTEM defocus image.

    The phase is calculated using the :code:`ltem.phase`
    function and propagated to the image plane using
    :code:`ltem.defocus_image`.	This is then plotted using
    :code:`mpl.scalar`.

    .. seealso:: :

    py:func:`~ltem.phase`
    py:func:`~ltem.defocus_image`
    """
    phase, _ = ltem.phase(field, kcx=kcx, kcy=kcy)
    defocus = ltem.defocus_image(phase, cs=cs, df_length=df_length,
                                 voltage=voltage, wavelength=wavelength)
    defocus.mpl.scalar(cmap='gray', interpolation='spline16',
                       colorbar_label='Intensity (counts)')


def ltem_integrated_mfd(field, /, kcx=0.1, kcy=0.1):
    r"""Quickplot of the LTEM integrated magnetic flux density.

    The phase is calculated using the :code:`ltem.phase`
    function and the integrated magnetic flux density
    :code:`ltem.integrated_magnetic_flux_density`.
    This is then plotted using :code:`mpl.lightness` and :code:`mpl.vector`.

    .. seealso:: :

    py:func:`~ltem.phase`
    py:func:`~ltem.integrated_magnetic_flux_density`
    py:func:`~mpl.lightness`
    py:func:`~mpl.vector`
    """
    phase, _ = ltem.phase(field, kcx=kcx, kcy=kcy)
    imf = ltem.integrated_magnetic_flux_density(phase)
    fig, ax = plt.subplots()
    imf.mpl.lightness(ax=ax, clim=[0, 0.5], interpolation='spline16',
                      colorwheel_args=dict(width=.75, height=.75),
                      colorwheel_xlabel=r'$m_x$', colorwheel_ylabel=r'$m_y$')
    imf.mpl.vector(ax=ax, use_color=False, color='w')


def mfm_phase_shift(system, /, tip_m=(0, 0, 0), quality=650, k=3, tip_q=0,
                    fwhm=None, z0=0):
    r"""Quickplot of the magnetic phase shift.

    The phase is calculated using the :code:`ltem.phase`
    function and plotted using :code:`mpl.scalar`.

    .. seealso:: :py:func:`~ltem.phase`
    """
    phase_shift = mfm.phase_shift(system, tip_m=tip_m, quality=quality,
                                  k=k, tip_q=tip_q, fwhm=fwhm)
    phase_shift_p = phase_shift.plane(z=z0)
    phase_shift_p.mpl.scalar(interpolation='spline16',
                             colorbar_label=r'Phase shift (radians.)')


def x_ray_holography(field, /, fwhm=None):
    r"""Quickplot of the magnetic phase shift.

    The phase is calculated using the :code:`ltem.phase`
    function and plotted using :code:`mpl.scalar`.

    .. seealso:: :py:func:`~ltem.phase`
    """
    holo = x_ray.holography(field, fwhm=fwhm)
    holo.mpl.scalar(cmap='RdBu',
                    interpolation='spline16',
                    colorbar_label=r'Integrated Magnetisation (A)')
