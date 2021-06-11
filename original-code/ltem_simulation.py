import numpy as np
import colorsys
import matplotlib.pyplot as plt
from scipy import constants
from matplotlib.patches import Ellipse
import math


class LTEMSimulation(object):
    """
    m_x and m_y must be normalised
    m_x		::	2D array of size (nx, ny)
    m_y		::	2D array of size (nx, ny)
    dx		::	resolution in x direction in nm
    dy		::	resolution in y direction in nm
    thick	:: 	thickness of sample (z height) in nm
    m_sat   ::  saturation magnetic flux densitiy
    """

    def __init__(self, m_x, m_y, m_z, dx, dy, thick, m_sat=1):

        self.m_x = m_x
        self.m_y = m_y
        self.dx = dx
        self.dy = dy
        self.thick = thick
        self.m_sat = m_sat
        self.nx = self.m_x.shape[0]
        self.ny = self.m_y.shape[1]
        self.m_z = m_z
        self.x_begin = - self.nx * self.dx / 2
        self.x_end = + self.nx * self.dx / 2
        self.y_begin = - self.ny * self.dy / 2
        self.y_end = + self.ny * self.dy / 2
        self.plot_x_begin = self.x_begin - self.dx / 2
        self.plot_x_end = self.x_end + self.dx / 2
        self.plot_y_begin = self.y_begin - self.dy / 2
        self.plot_y_end = self.y_end + self.dy / 2
        self.x = np.linspace(self.x_begin, self.x_end, self.nx)
        self.y = np.linspace(self.y_begin, self.y_end, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y,indexing='xy')


    def plot_projected_magnetisation(self, skip_arrow=2, size_arrow=0.04, color='w', savefig=False):

        f, ax = plt.subplots(ncols=1, figsize=(8, 8))
        plt.title('Projected magnetisation', fontsize=25)
        rgb_map = generate_RGBs(np.column_stack((self.m_x[:, :].flatten(),
                                                 self.m_y[:, :].flatten(),
                                                 -self.m_z[:, :].flatten())))

        ax.imshow(rgb_map.reshape(self.nx, self.ny, 3), origin='lower',
                  extent=[self.plot_x_begin, self.plot_x_end,
                          self.plot_y_begin, self.plot_y_end],
                  interpolation='spline16')
        plt.xlabel("$x$ (nm)", fontsize=22)
        plt.ylabel("$y$ (nm)", fontsize=25)
        plt.tick_params(labelsize=20)
        plt.quiver(self.X[::skip_arrow, ::skip_arrow],
                   self.Y[::skip_arrow, ::skip_arrow],
                   self.m_x[::skip_arrow, ::skip_arrow],
                   self.m_y[::skip_arrow, ::skip_arrow],
                   scale=1/size_arrow, pivot='mid', units='width', color=color)
        plt.tight_layout()
        if savefig:
            plt.savefig(savefig)
        plt.show()

    def calculate_phase(self, kv, Cs, kx=0.1, ky=0.1):
        """

        kx		::	Tikhonov filter radius in x in pixels
        ky		::	Tikhonov filter radius in y in pixels
        kv		::	Acceleratig voltage of electrons in kV
        Cs 		::	Spherical aberration coefficient
        """
        self.kx = kx
        self.ky = ky
        self.Cs = Cs
        const = 1j * self.thick / (2 * constants.codata.value('mag. flux quantum') / (constants.nano**2))
        self.wavelength = wavelength_func(kv)  # Wavelength of electrons in nm
        print('The electron beam has a wavelength of %.2E nm.' % self.wavelength)

    # Fourier transform magnetisation
        ft_mx = np.fft.fft2(self.m_x, axes=(-2, -1))
        ft_my = np.fft.fft2(self.m_y, axes=(-2, -1))

        FreqCompRows = np.fft.fftfreq(ft_mx.shape[0], d=self.dx)
        FreqCompCols = np.fft.fftfreq(ft_mx.shape[1], d=self.dy)
        self.Xft, self.Yft = np.meshgrid(FreqCompRows, FreqCompCols, indexing='xy')  # Create a grid of coordinates
        self.sx = abs(FreqCompRows[0] - FreqCompRows[1])  # Resolution in reciprocal space
        self.sy = abs(FreqCompCols[0] - FreqCompCols[1])  # Resolution in reciprocal space

        nume = ((self.Xft**2) + (self.Yft**2))
        dnom = ((self.Xft**2) + (self.Yft**2) + (self.sx**2) * (self.kx**2) + (self.sy**2) * (self.ky**2))**2
        cross = - ft_my * self.Xft + ft_mx * self.Yft
        self.ft_phase = np.array(const * cross * nume / dnom) * self.m_sat
        self.phase = np.fft.ifft2(self.ft_phase).real



    def plot_ftphase(self, plot_limits=[None, None, None, None], savefig=False):
        ft_p_shift = np.fft.fftshift(abs(self.ft_phase))
        f, ax = plt.subplots(ncols=1, figsize=(8, 8))
        plt.title('Fourier Transform of the phase, $\widetilde{\Phi}$', fontsize=25)
        imgplot1 = plt.imshow(ft_p_shift, origin='lower',
                              extent=[np.min(self.Xft), np.max(self.Xft),
                                      np.min(self.Yft), np.max(self.Yft)],
                              cmap='gray', interpolation='spline16')
        # draw the ellipse
        ax = plt.gca()
        ax.add_patch(Ellipse((0,0), width=(self.sx*self.kx), height=(self.sy*self.ky),
                             edgecolor='red',
                             facecolor='none',
                             linewidth=3, label='Tikhonov filter'))
        cbar1 = plt.colorbar(imgplot1, fraction=0.046, pad=0.04)
        plt.xlabel("$k_x$ (nm$^{-1}$)", fontsize=25)
        plt.ylabel("$k_y$ (nm$^{-1}$)", fontsize=25)
        plt.xlim([plot_limits[0],plot_limits[1]])
        plt.ylim([plot_limits[2],plot_limits[3]])
        plt.tick_params(labelsize=20)
        cbar1.ax.set_ylabel(r'$\widetilde{\Phi}$ (radians)$^{-1}$', fontsize=25)
        cbar1.ax.tick_params(labelsize=20)
        plt.legend(loc=9, bbox_to_anchor=(1.4, 1.05))
        plt.tight_layout()
        if savefig:
            plt.savefig(savefig)
        plt.show()


    def plot_phase(self, savefig=False):
        f, ax = plt.subplots(ncols=1, figsize=(8, 8))
        plt.title(r'Phase, $\Phi$', fontsize=25)
        imgplot1 = plt.imshow(self.phase, origin='lower',
                              extent=[self.plot_x_begin, self.plot_x_end, self.plot_y_begin, self.plot_y_end], cmap='gray', interpolation='spline16')
        cbar1 = plt.colorbar(imgplot1, fraction=0.046, pad=0.04)
        plt.xlabel("$x$ (nm)", fontsize=25)
        plt.ylabel("$y$ (nm)", fontsize=25)
        plt.tick_params(labelsize=20)
        cbar1.ax.set_ylabel(r"Phase, $\Phi$ (radians)", fontsize=25)
        cbar1.ax.tick_params(labelsize=20)
        plt.tight_layout()
        if savefig:
            plt.savefig(savefig)
        plt.show()

    def plot_cosine_phase(self, pa=10, savefig=False):
        """
        pa      ::  phase amplification factor
        """
        self.cos_phase = np.cos(pa*self.phase)
        f, ax = plt.subplots(ncols=1, figsize=(8, 8))
        plt.title(r'$\cos \left( %g \cdot \phi \right)$' % pa, fontsize=25)
        imgplot3 = plt.imshow(self.cos_phase, origin='lower',
                              extent=[self.x_begin, self.plot_x_end,
                                      self.plot_y_begin, self.plot_y_end],
                              cmap='gray', interpolation='spline16')
        cbar1 = plt.colorbar(imgplot3, fraction=0.046, pad=0.04)
        plt.xlabel("$x$ (nm)", fontsize=25)
        plt.ylabel("$y$ (nm)", fontsize=25)
        plt.tick_params(labelsize=20)
        cbar1.ax.set_ylabel(r'$\cos \left( %g \cdot \phi  \right)$' % pa, fontsize=25)
        cbar1.ax.tick_params(labelsize=20)
        plt.tight_layout()
        if savefig:
            plt.savefig(savefig)
        plt.show()

    def calculate_magnetic_flux_density(self):
        d_phase = np.gradient(self.phase)
        b_const = (constants.codata.value('mag. flux quantum') / (constants.nano**2)) / (np.pi * self.thick)
        self.b_field_x = -b_const*d_phase[0]/self.dy
        self.b_field_y = b_const*d_phase[1]/self.dx

        self.mag_B = -(1 - np.sqrt(self.b_field_x**2 + self.b_field_y**2) / np.max(np.sqrt(self.b_field_x**2 + self.b_field_y**2)))

    def plot_magnetic_flux_density(self, skip_arrow=2, size_arrow=0.04, color='w', savefig=False, **kwargs):
        f, ax = plt.subplots(ncols=1, figsize=(8, 8))
        plt.title('Projected Magnetic Flux Density', fontsize=25)
        rgb_map = generate_RGBs(np.column_stack((self.b_field_x[:, :].flatten(),
                                                 self.b_field_y[:, :].flatten(),
                                                 self.mag_B[:, :].flatten())))

        ax.imshow(rgb_map.reshape(self.nx, self.ny, 3), origin='lower',
                  extent=[self.plot_x_begin, self.plot_x_end,
                          self.plot_y_begin, self.plot_y_end],
                  interpolation='spline16')
        plt.xlabel("$x$ (nm)", fontsize=25)
        plt.ylabel("$y$ (nm)", fontsize=25)
        plt.tick_params(labelsize=20)
        plt.quiver(self.X[::skip_arrow, ::skip_arrow],
                   self.Y[::skip_arrow, ::skip_arrow],
                   self.b_field_x[::skip_arrow, ::skip_arrow],
                   self.b_field_y[::skip_arrow, ::skip_arrow],
                   scale=1/size_arrow, pivot='mid', units='width', color=color, **kwargs)
        plt.tight_layout()
        if savefig:
            plt.savefig(savefig)
        plt.show()

    def calculate_defocus_image(self, df=0.2):
        """
        return a tuple of 7 elements
        """
        wavefn = np.exp(1j*self.phase)
        ft_wavefn = np.fft.fft2(wavefn)
        ft_wf_FreqCompRows = np.fft.fftfreq(ft_wavefn.shape[0], d=self.dx)
        ft_wf_FreqCompCols = np.fft.fftfreq(ft_wavefn.shape[1], d=self.dy)
        ft_wf_Xft, ft_wf_Yft = np.meshgrid(ft_wf_FreqCompRows, ft_wf_FreqCompCols,indexing='xy')  # Create a grid of coordinates
        ft_wf_k2 = ft_wf_Xft**2 + ft_wf_Yft**2
        intensity_cts = ctf(df, ft_wf_k2, ft_wavefn, self.wavelength, self.Cs)
        max_intensity = intensity_cts.max()
        min_intensity = intensity_cts.min()
        index_max_y, index_max_x = np.unravel_index(intensity_cts.argmax(), intensity_cts.shape)
        index_min_y, index_min_x = np.unravel_index(intensity_cts.argmin(), intensity_cts.shape)
        contrast = (max_intensity - min_intensity) / (max_intensity + min_intensity)
        return intensity_cts, max_intensity, min_intensity, index_max_x, index_max_y, index_min_x, index_min_y, contrast

    def plot_defocus_image(self, df=0.2, show_max_min=True, savefig=False):
        (intensity_cts, max_intensity,
         min_intensity, index_max_x,
         index_max_y, index_min_x,
         index_min_y, contrast) = self.calculate_defocus_image(df)

        f, ax = plt.subplots(ncols=1, figsize=(8, 8))
        plt.title('Image at %.2f mm defocus' % df, fontsize=25)
        imgplot3 = plt.imshow(intensity_cts, origin='lower',
                              extent=[self.plot_x_begin, self.plot_x_end,
                                      self.plot_y_begin, self.plot_y_end],
                              cmap='gray', interpolation='spline16')
        cbar1 = plt.colorbar(imgplot3, fraction=0.046, pad=0.04)
        plt.xlabel("$x$ (nm)", fontsize=25)
        plt.ylabel("$y$ (nm)", fontsize=25)
        cbar1.ax.set_ylabel("Scaled Intensity (counts/pixel)", fontsize=25)
        cbar1.ax.tick_params(labelsize=20)
        plt.tick_params(labelsize=20)
        plt.tight_layout()
        if savefig:
            plt.savefig(savefig,dpi=80, bbox_inches='tight')
        plt.show()

# Davids colour scheme
def convert_to_RGB(hls_color):
    return np.array(colorsys.hls_to_rgb(hls_color[0] / (2 * np.pi),
                                        hls_color[1],
                                        hls_color[2]))


def generate_RGBs(field_data):
    """
    field_data      ::  (n, 3) array
    """
    hls = np.ones_like(field_data)
    hls[:, 0] = np.arctan2(field_data[:, 1],
                           field_data[:, 0]
                           )
    hls[:, 0][hls[:, 0] < 0] = hls[:, 0][hls[:, 0] < 0] + 2 * np.pi
    hls[:, 1] = 0.5 * (field_data[:, 2] + 1)
    rgbs = np.apply_along_axis(convert_to_RGB, 1, hls)

    # Redefine colours less than zero
    # rgbs[rgbs < 0] += 2 * np.pi

    return rgbs


def wavelength_func(V):
    """
    Function for working out the relativistic wavelength of an electron
    Input in kV
    output in nm
    """
    V *= constants.kilo
    λ = constants.h/(constants.nano*np.sqrt(2*V*constants.m_e*constants.e))
    λ *= 1/(np.sqrt(1+(constants.e*V)/(2*constants.m_e*constants.c**2)))
    return λ


def ctf(df, ft_wf_k2, ft_wavefn, wavelength, Cs):
    """
    df          ::  defocus in mm
    ft_wf_k2    ::  k vector squared
    wavelength  ::  Wavelength of the electon beam
    Cs          ::  Spherical abberation
    """
    cts = -0.5 * wavelength * df * ft_wf_k2 * (10**6) + 0.25 * (wavelength**3) * Cs * (ft_wf_k2 * ft_wf_k2)
    ft_def_wf_cts = ft_wavefn * np.exp(1j*2*np.pi*cts)
    def_wf_cts = np.fft.ifft2(ft_def_wf_cts)
    intensity_cts = def_wf_cts.conjugate()*def_wf_cts
    return intensity_cts.real

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
