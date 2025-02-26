"""FMR submodule.

Module for calculation of Ferromagnetic Resonance related
quantities.
"""

from typing import Optional, Tuple

import discretisedfield as df
import hvplot.xarray  # noqa
import micromagneticdata as micd
import numpy as np
import scipy.fft as fft
import xarray as xr


def fmr(
    drive: micd.Drive, init_field: Optional[df.Field] = None
) -> Tuple[xr.DataArray, xr.DataArray]:
    r"""
    Computes the Ferromagnetic Resonance (FMR) power and phase spectra.

    This function extracts the orientation of the magnetisation from the
    provided drive object, optionally subtracts an initial field orientation
    (if provided), and applies a Fourier transform to obtain the power
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
        A micromagnetic drive signal object containing time-dependent
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
    if not isinstance(drive, micd.Drive):
        raise TypeError(
            "The 'drive' parameter must be an instance of micromagneticdata.Drive."
        )
    if init_field is not None and not isinstance(init_field, df.Field):
        raise TypeError(
            "The 'init_field' parameter must be an instance of "
            "micromagneticdata.Field if provided."
        )

    drive_orientation = drive.register_callback(lambda field: field.orientation)
    data_xarr = drive_orientation.to_xarray()

    if "t" not in data_xarr.coords:
        raise KeyError("The drive data must have a 't' (time) coordinate.")

    # Validate time step uniformity
    t_values = data_xarr["t"].values
    if t_values.size < 2:
        raise ValueError("Insufficient time points to compute a time step.")
    dt_array = np.diff(t_values)
    if not np.allclose(dt_array, dt_array[0], atol=1e-12):
        raise ValueError("Time steps in the drive data are not uniform.")
    dt = dt_array[0]

    if init_field is not None:
        data_xarr -= init_field.orientation.to_xarray()

    # Compute FFT frequencies and FFT along the time axis
    # (assumed to be the first dimension).
    num_time_points = data_xarr.shape[0]
    freq = fft.rfftfreq(num_time_points, d=dt)
    fft_values = fft.rfft(data_xarr.values, axis=0, norm="ortho")

    # Keep all coordinates except the time coordinate ('t') to the FFT output.
    # Done like this as the names of dims can vary.
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
