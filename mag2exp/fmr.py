"""FMR submodule.

Module for calculation of Ferromagnetic Resonance related
quantities.
"""

from typing import Optional, Tuple

import discretisedfield as df
import micromagneticdata as micd
import numpy as np
import scipy.fft as fft
import xarray as xr


def fmr(
    drive: micd.Drive, init_field: Optional[df.Field] = None
) -> Tuple[xr.DataArray, xr.DataArray]:
    if not isinstance(drive, micd.Drive):
        raise TypeError(
            "The 'drive' parameter must be an instance of micromagneticdata.Drive."
        )
    if init_field is not None and not isinstance(init_field, micd.Field):
        raise TypeError(
            "The 'init_field' parameter must be an instance of "
            "micromagneticdata.Field if provided."
        )

    drive_orientation = drive.register_callback(lambda field: field.orientation)
    data_xarr = drive_orientation.to_xarray()

    if "t" not in data_xarr.coords:
        raise KeyError("The drive data must have a 't' (time) coordinate.")

    # Calculate the uniform time step (dt) from the 't' coordinate.
    t_values = data_xarr["t"].values
    if t_values.size < 2:
        raise ValueError("Insufficient time points to compute a time step.")
    dt_array = np.diff(t_values)
    if not np.allclose(dt_array, dt_array[0], atol=1e-8):
        raise ValueError("Time steps in the drive data are not uniform.")
    dt = dt_array[0]

    if init_field is not None:
        data_xarr = data_xarr - init_field.orientation.to_xarray()

    # Compute FFT frequencies and FFT along the time axis
    # (assumed to be the first dimension).
    num_time_points = data_xarr.shape[0]
    freq = fft.rfftfreq(num_time_points, d=dt)
    fft_values = fft.rfft(data_xarr.values, axis=0, norm="ortho")

    # Keep all coordinates except the time coordinate ('t') to the FFT output.
    fft_coords = {"freq_t": freq}
    fft_dims = ["freq_t"]
    for coord in data_xarr.coords:
        if coord != "t":
            fft_coords[coord] = data_xarr.coords[coord]
            fft_dims.append(coord)

    fft_xarr = xr.DataArray(fft_values, coords=fft_coords, dims=fft_dims, name="fft")

    power = np.abs(fft_xarr) ** 2
    power.name = "power"
    phase = np.arctan2(fft_xarr.imag, fft_xarr.real)
    phase.name = "phase"

    return power, phase
