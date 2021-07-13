from exsim.util import gaussian_filter
import discretisedfield as df


def holographic_image(field, /, fwhm=None):
    """
    Parameters
    ----------
    field : discretisedfield.field
        Magneisation field.
    fwhm : numbers.Real, optional
        If specified, convolutes with a 2 Dimentional Gaussian of full width
        half maxium (fwhm) specified.

    Returns
    -------
    discretisedfield.Field
        X-ray holographic image.
    """
    magnetisation = (field.z * df.dz).integral(direction='z')
    if fwhm is not None:
        magnetisation = gaussian_filter(magnetisation, fwhm=fwhm)
    return magnetisation
