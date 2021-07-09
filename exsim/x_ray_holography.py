from exsim.util import gaussian_filter
import discretisedfield as df


def holographic_image(field, /, fwhm=None):
    """
    Parameters
    ----------
    """
    magnetisation = field.z.integral(direction='z')
    if fwhm is not None:
        magnetisation = gaussian_filter(magnetisation, fwhm=fwhm)
    return magnetisation
