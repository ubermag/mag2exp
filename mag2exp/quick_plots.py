import matplotlib.pyplot as plt
import matplotlib
import ubermagutil.units as uu
from . import ltem


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
