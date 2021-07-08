from . import ltem


def phase(field, /, kx=0.1, ky=0.1):
    phase, _ = ltem.phase(field, kx=kx, ky=ky)
    phase.real.mpl_scalar(cmap='gray',
                          interpolation='spline16',
                          colorbar_label=r'$\phi$ (radians)')


def ft_phase(field, /, kx=0.1, ky=0.1):
    _, ft_phase = ltem.phase(field, kx=kx, ky=ky)
    fig, ax = plt.sublpots()
    (ft_phase.conjugate * ft_phase).plane('z').real.mpl_scalar(
        ax=ax, cmap='gray', interpolation='spline16',
        colorbar_label=r'$\widetilde{\phi}$ (radians$^{-1}$)')
    multiplier = uu.si_max_multiplier(ft_phase.mesh.region.edges)
    ax.add_patch(Ellipse(xy=(0, 0),
                        width=ft_phase.mesh.cell[0] * kx * 2 / multiplier,
                        height=ft_phase.mesh.cell[1] * ky * 2 / multiplier,
                        edgecolor='red',
                        facecolor='none',
                        linewidth=3,
                        label='Tikhonov filter'))
    ax.legend(frameon=True)
