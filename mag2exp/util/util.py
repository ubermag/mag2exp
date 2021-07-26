import scipy.ndimage
import numpy as np
import discretisedfield as df


def gaussian_filter(field, /, fwhm):
    """
    Parameters
    ----------
    ...
    """
    if field.dim > 1:
        msg = f'Gaussian filter only supports fields with {field.dim=}'
        raise RuntimeError(msg)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    if field.mesh.attributes['isplane']:
        sigma = (sigma[0] / field.mesh.cell[field.mesh.attributes['axis1']],
                 sigma[1] / field.mesh.cell[field.mesh.attributes['axis2']])
        value = scipy.ndimage.gaussian_filter(field.array.squeeze(),
                                              sigma=sigma)[..., np.newaxis,
                                                           np.newaxis]
    else:
        sigma = [sigma[i] / field.mesh.cell[i] for i in range(3)]
        value = scipy.ndimage.gaussian_filter(field.array.squeeze(),
                                              sigma=sigma)[..., np.newaxis]

    return df.Field(field.mesh, dim=1, value=value)
