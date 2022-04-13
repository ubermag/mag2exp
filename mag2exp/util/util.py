import discretisedfield as df
import numpy as np
import scipy.ndimage


def gaussian_filter(field, /, fwhm):
    """Gaussian filter.

    Convolution of a field with a 1 dimensional filter.

    Parameters
    ----------
    field : discretisedfield.field
        Magnetisation field.
    fwhm : array_like, optional
        If specified, convolutes the output image with a 2 Dimensional Gaussian
        with the full width half maximum (fwhm) specified.

    Returns
    -------
    discretisedfield.Field
        Convoluted field.

    Raises
    ------
    RuntimeError
        Gaussian filter only supports fields with field.dim=1.
    """
    if field.dim > 1:
        msg = "Gaussian filter only supports fields with field.dim=1."
        raise RuntimeError(msg)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    if field.mesh.attributes["isplane"]:
        sigma = (
            sigma[0] / field.mesh.cell[field.mesh.attributes["axis1"]],
            sigma[1] / field.mesh.cell[field.mesh.attributes["axis2"]],
        )
        value = scipy.ndimage.gaussian_filter(field.array.squeeze(), sigma=sigma)[
            ..., np.newaxis, np.newaxis
        ]
    else:
        sigma = [sigma[i] / field.mesh.cell[i] for i in range(3)]
        value = scipy.ndimage.gaussian_filter(field.array.squeeze(), sigma=sigma)[
            ..., np.newaxis
        ]

    return df.Field(field.mesh, dim=1, value=value)
