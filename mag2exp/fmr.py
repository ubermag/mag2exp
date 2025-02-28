"""FMR submodule.

Module for calculation of ferromagnetic resonance related
quantities.
"""

from typing import Optional, Tuple

import discretisedfield as df
import micromagneticdata as mdata
import numpy as np
import scipy.fft as fft
import xarray as xr


def ringdown(
    drive: mdata.Drive, init_field: Optional[df.Field] = None
) -> Tuple[xr.DataArray, xr.DataArray]:
    r"""
    Compute the Ferromagnetic Resonance (FMR) power and phase spectra
    using the ringdown method.

    This function extracts the orientation of the magnetisation from the
    provided drive object, optionally subtracts an initial field orientation,
    and applies a Fourier transform to obtain the power
    and phase spectra.

    The Discrete Fourier Transform is computed at each spatial grid point for the
    :math:`k`-th component of the magnetisation.
    The corresponding power spectrum is given by:

    .. math::

        S_k(r_j, f) = \frac{1}{N} \left| F_k(r_j, f) \right|^2

    where :math:`F_k(r_j, f)` is the Fourier transform of the magnetisation component
    :math:`M_k(r_j, t)`.

    The phase spectrum, which provides spatially resolved phase information,
    is given by:

    .. math::

        \phi_k(r_j, f) = \arg F_k(r_j, f).

    where :math:`\arg` is the phase angle of the complex Fourier coefficient.

    For further details on numerical FMR analysis in micromagnetics see,
    A. Baker et al., J. Magn. Magn. Mater. 421, 428 (2017)

    Parameters
    ----------
    drive : micromagneticdata.Drive
        A micromagnetic drive object containing time-dependent
        magnetisation data.
    init_field : discretisedfield.Field, optional
        A reference field whose orientation is subtracted before processing.

    Returns
    -------
    power : xr.DataArray
        The computed power spectrum (squared magnitude of the FFT).
    phase : xr.DataArray
        The computed phase spectrum (FFT phase angle).
    """
    if not isinstance(drive, mdata.Drive):
        raise TypeError(
            "The 'drive' parameter must be an instance of "
            f"micromagneticdata.Drive, not {type(drive)=}"
        )
    if init_field is not None and not isinstance(init_field, df.Field):
        raise TypeError(
            "The 'init_field' parameter must be an instance of "
            f"micromagneticdata.Field if provided, not {type(init_field)=}."
        )

    if drive.x != "t":
        raise TypeError("The drive data must have a 't' (time) coordinate.")

    # Validate time step uniformity
    t_values = drive.table.data["t"]
    if t_values.size < 2:
        raise ValueError("Insufficient time points to compute a FFT.")
    dt_array = np.diff(t_values)
    if not np.allclose(dt_array, dt_array[0], atol=0.01 * dt_array[0]):
        raise ValueError("Time steps in the drive data are not uniform.")
    dt = dt_array[0]

    drive_orientation = drive.register_callback(lambda field: field.orientation)
    data_xarr = drive_orientation.to_xarray()

    if init_field is not None:
        data_xarr -= init_field.orientation.to_xarray()

    # Compute FFT frequencies and FFT along the time axis
    # (The first dimension of drive.to_xarray()).
    num_time_points = data_xarr.shape[0]
    freq = fft.rfftfreq(num_time_points, d=dt)
    fft_values = fft.rfft(data_xarr.data, axis=0, norm="ortho")

    # Keep all coordinates except the time coordinate ('t') to the FFT output.
    # Done like this as the names of dims can vary.
    fft_coords = {"freq_t": freq}
    fft_dims = ["freq_t"]
    for dim in data_xarr.dims:
        if dim != "t":
            fft_coords[dim] = data_xarr.coords[dim]
            fft_dims.append(dim)

    fft_xarr = xr.DataArray(fft_values, coords=fft_coords, dims=fft_dims, name="fft")

    power = np.abs(fft_xarr) ** 2
    power.name = "power"
    phase = np.arctan2(fft_xarr.imag, fft_xarr.real)
    phase.name = "phase"

    return power, phase
