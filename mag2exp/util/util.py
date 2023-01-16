import discretisedfield as df
import micromagneticmodel as mm
import numpy as np
import oommfc as oc
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
        Gaussian filter only supports fields with field.nvdim=1.
    """
    if field.nvdim > 1:
        msg = "Gaussian filter only supports fields with field.nvdim=1."
        raise RuntimeError(msg)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    if field.mesh.region.ndim == 2:
        sigma = (
            sigma[0] / field.mesh.cell[0],
            sigma[1] / field.mesh.cell[1],
        )
        value = scipy.ndimage.gaussian_filter(field.array.squeeze(), sigma=sigma)[
            ..., np.newaxis
        ]
    else:
        sigma = [sigma[i] / field.mesh.cell[i] for i in range(3)]
        value = scipy.ndimage.gaussian_filter(field.array.squeeze(), sigma=sigma)[
            ..., np.newaxis
        ]

    return df.Field(field.mesh, nvdim=1, value=value)


def calculate_demag_field(field):
    """Calculate demagnetisation field.

    Calculate demagnetisation field using OOMMF.

    Parameters
    ----------
    field : discretisedfield.field
        Magnetisation field.

    Returns
    -------
    discretisedfield.Field
        Demagnetisation field.

    """
    system = mm.System(name="demag_calculation")
    system.energy = mm.Demag()
    system.m = field
    return oc.compute(system.energy.demag.effective_field, system, verbose=0)
