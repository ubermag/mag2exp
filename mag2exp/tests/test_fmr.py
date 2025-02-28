import discretisedfield as df
import micromagneticdata as mdata
import micromagneticmodel as mm
import numpy as np
import oommfc as oc
import pytest
import xarray as xr

import mag2exp


@pytest.fixture(scope="module")
def simulation_data(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("sim")
    # Set up the mesh.
    mesh = df.Mesh(p1=(0, 0, 0), p2=(2e-9, 2e-9, 2e-9), n=(2, 2, 2))

    # Create the system and set up the energy, dynamics, and initial magnetisation.
    system = mm.System(name="test")
    system.energy = mm.Exchange(A=1.3e-11)
    system.dynamics = mm.Precession(gamma0=2.211e5) + mm.Damping(alpha=0.01)
    system.m = df.Field(mesh, nvdim=3, value=(0, 0, 1), norm=8e5)

    # Run the minimisation driver.
    md = oc.MinDriver()
    md.drive(system, dirname=tmp_dir)

    # Add sin wave
    system.energy += mm.Zeeman(H=(100, 0, 0), func="sin", f=10e9, t0=0)

    # Set simulation time parameters and run the time driver.
    T = 1e-9
    n = 100
    td = oc.TimeDriver()
    td.drive(system, t=T, n=n, dirname=tmp_dir)

    # Load the simulation data
    data = mdata.Data(system.name, dirname=str(tmp_dir))

    # Cleanup is automatic when the temporary directory is removed.
    return data


@pytest.fixture(scope="module")
def simulation_timedrive(simulation_data):
    return simulation_data[-1]


@pytest.fixture(scope="module")
def simulation_mindrive(simulation_data):
    return simulation_data[0]


@pytest.fixture(scope="module")
def simulation_field(simulation_data):
    return simulation_data[-1].m0


@pytest.mark.parametrize(
    "invalid_drive",
    [42, "invalid drive", None, [1, 2, 3], {}, simulation_data, simulation_field],
)
def test_invalid_drive_type(invalid_drive):
    with pytest.raises(TypeError):
        mag2exp.fmr.ringdown(invalid_drive)


def test_MinDrive(simulation_data):
    with pytest.raises(TypeError):
        mag2exp.fmr.ringdown(simulation_data[0])


@pytest.mark.parametrize(
    "invalid_field",
    [
        42,
        "invalid field",
        [1, 2, 3],
        {},
        simulation_timedrive,
        simulation_mindrive,
        simulation_data,
    ],
)
def test_invalid_init_field_type(simulation_data, invalid_field):
    with pytest.raises(TypeError):
        mag2exp.fmr.ringdown(simulation_data, init_field=invalid_field)


def test_fmr_returns_valid_arrays(simulation_timedrive):
    power, phase = mag2exp.fmr.ringdown(simulation_timedrive)

    assert isinstance(power, xr.DataArray), "power is not an xarray.DataArray."
    assert isinstance(phase, xr.DataArray), "phase is not an xarray.DataArray."

    expected_dims = {"freq_t", "x", "y", "z", "vdims"}
    assert set(power.dims) == expected_dims, "power dimensions mismatch."
    assert set(phase.dims) == expected_dims, "phase dimensions mismatch."

    expected_shape = (
        simulation_timedrive.n // 2 + 1,
        *simulation_timedrive.m0.array.shape,
    )
    assert expected_shape == power.shape, "shape of power not as expected"
    assert expected_shape == phase.shape, "shape of phase not as expected"

    assert np.all(np.isreal(power.values)), "Power array contains non-real values."
    assert np.all(np.isreal(phase.values)), "Phase array contains non-real values."

    assert phase.min() >= -np.pi, "Phase should be in the domain [-pi, pi]"
    assert phase.max() <= np.pi, "Phase should be in the domain [-pi, pi]"

    # Check largest peak is at zero frequency
    assert np.allclose(power.mean(dim=("x", "y", "z")).idxmax("freq_t"), 0)
    assert power.mean(dim=("x", "y", "z")).sel(vdims="z").max("freq_t") > 1e4

    # Check next largest peak is at excitation frequency
    assert np.allclose(
        power.mean(dim=("x", "y", "z")).isel(freq_t=slice(1, None)).idxmax("freq_t"),
        1e10,
    )


def test_fmr_with_field(simulation_timedrive, simulation_field):
    power, _ = mag2exp.fmr.ringdown(simulation_timedrive, simulation_field)

    # Check largest peak is at zero frequency
    assert np.allclose(power.mean(dim=("x", "y", "z")).idxmax("freq_t"), 0)
    assert power.mean(dim=("x", "y", "z")).sel(vdims="z").max("freq_t") < 1e4

    # Check next largest peak is at excitation frequency
    assert np.allclose(
        power.mean(dim=("x", "y", "z")).isel(freq_t=slice(1, None)).idxmax("freq_t"),
        1e10,
    )


def test_fmr_returns_valid_arrays_2d(simulation_timedrive):
    simulation_timedrive = simulation_timedrive.register_callback(
        lambda field: field.sel("z")
    )
    power, phase = mag2exp.fmr.ringdown(simulation_timedrive)

    assert isinstance(power, xr.DataArray), "power is not an xarray.DataArray."
    assert isinstance(phase, xr.DataArray), "phase is not an xarray.DataArray."

    expected_dims = {"freq_t", "x", "y", "vdims"}
    assert set(power.dims) == expected_dims, "power dimensions mismatch."
    assert set(phase.dims) == expected_dims, "phase dimensions mismatch."

    expected_shape = (
        simulation_timedrive.n // 2 + 1,
        *simulation_timedrive.m0.sel("z").array.shape,
    )
    assert expected_shape == power.shape, "shape of power not as expected"
    assert expected_shape == phase.shape, "shape of phase not as expected"

    assert np.all(np.isreal(power.values)), "Power array contains non-real values."
    assert np.all(np.isreal(phase.values)), "Phase array contains non-real values."

    assert phase.min() >= -np.pi, "Phase should be in the domain [-pi, pi]"
    assert phase.max() <= np.pi, "Phase should be in the domain [-pi, pi]"

    # Check largest peak is at zero frequency
    assert np.allclose(power.mean(dim=("x", "y")).idxmax("freq_t"), 0)
    assert power.mean(dim=("x", "y")).sel(vdims="z").max("freq_t") > 1e4

    # Check next largest peak is at excitation frequency
    assert np.allclose(
        power.mean(dim=("x", "y")).isel(freq_t=slice(1, None)).idxmax("freq_t"),
        1e10,
    )
