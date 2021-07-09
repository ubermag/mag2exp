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
    if not hasattr(field.mesh, 'info'):
        msg = 'Gaussian filter only supports a single plane'
        raise RuntimeError(msg)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    sigma = (sigma / field.mesh.cell[field.mesh.info['axis1']],
             sigma / field.mesh.cell[field.mesh.info['axis2']])
    value = scipy.ndimage.gaussian_filter(field.array.squeeze(), sigma=sigma)
    return df.Field(field.mesh, dim=1, value=value[..., np.newaxis, np.newaxis])
