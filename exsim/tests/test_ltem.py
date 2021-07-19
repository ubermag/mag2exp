import pytest
import discretisedfield as df
import exsim


def test_relativistic_wavelength():
    assert exsim.ltem.relativistic_wavelength(0) == float('inf')


def test_ltem_phase():
    mesh = df.Mesh(p1=(-5e-9, -4e-9, -1e-9), p2=(5e-9, 4e-9, 1e-9),
                   cell=(2e-9, 1e-9, 0.5e-9))
    field = df.Field(mesh, dim=3, value=(0, 0, 1))
    phase, ft_phase = exsim.ltem.phase(field)
    assert (phase.real.array == 0).all()
    assert (phase.imag.array == 0).all()


def test_ltem_phase_neel():
    mesh = df.Mesh(p1=(-5e-9, -4e-9, -1e-9), p2=(5e-9, 4e-9, 1e-9),
                   cell=(2e-9, 1e-9, 0.5e-9))

    def f_val(position):
        x, y, z = position
        if x < 0:
            return (1, 0, 0)
        else:
            return (-1, 0, 0)
    field = df.Field(mesh, dim=3, value=f_val)
    phase, ft_phase = exsim.ltem.phase(field)
    assert (phase.real.array == 0).all()
    assert (phase.imag.array == 0).all()


def test_ltem_phase_bloch():
    mesh = df.Mesh(p1=(-5e-9, -4e-9, -1e-9), p2=(5e-9, 4e-9, 1e-9),
                   cell=(2e-9, 1e-9, 0.5e-9))

    def f_val(position):
        x, y, z = position
        if x < 0:
            return (0, 1, 0)
        else:
            return (0, -1, 0)
    field = df.Field(mesh, dim=3, value=f_val)
    phase, ft_phase = exsim.ltem.phase(field)
    assert (phase.real.array != 0).any()
    assert (phase.imag.array == 0).all()


def test_defocus_image():
    mesh = df.Mesh(p1=(-5e-9, -4e-9, -1e-9), p2=(5e-9, 4e-9, 1e-9),
                   cell=(2e-9, 1e-9, 0.5e-9))
    field = df.Field(mesh, dim=3, value=(0, 0, 1))
    phase, ft_phase = exsim.ltem.phase(field)
    with pytest.raises(RuntimeError):
        exsim.ltem.defocus_image(phase)


def test_defocus_image_zero_df():
    mesh = df.Mesh(p1=(-5e-9, -4e-9, -1e-9), p2=(5e-9, 4e-9, 1e-9),
                   cell=(2e-9, 1e-9, 0.5e-9))

    def f_val(position):
        x, y, z = position
        if x < 0:
            return (0, 1, 0)
        else:
            return (0, -1, 0)
    field = df.Field(mesh, dim=3, value=f_val)
    phase, ft_phase = exsim.ltem.phase(field)
    dfi = exsim.ltem.defocus_image(phase, Cs=0, df_length=0, U=300e3)
    assert (dfi.array == 1).all()


def test_defocus_image_df():
    mesh = df.Mesh(p1=(-5e-9, -4e-9, -1e-9), p2=(5e-9, 4e-9, 1e-9),
                   cell=(2e-9, 1e-9, 0.5e-9))

    def f_val(position):
        x, y, z = position
        if x < 0:
            return (0, 1, 0)
        else:
            return (0, -1, 0)
    field = df.Field(mesh, dim=3, value=f_val)
    phase, ft_phase = exsim.ltem.phase(field)
    dfi = exsim.ltem.defocus_image(phase, Cs=0, df_length=0.2e-3, U=300e3)
    assert (dfi.array != 1).any()


def test_integrated_magnetic_flux_density():
    mesh = df.Mesh(p1=(-5e-9, -4e-9, -1e-9), p2=(5e-9, 4e-9, 1e-9),
                   cell=(2e-9, 1e-9, 0.5e-9))

    def f_val(position):
        x, y, z = position
        if x < 0:
            return (0, 1, 0)
        else:
            return (0, -1, 0)
    field = df.Field(mesh, dim=3, value=f_val)
    phase, _ = exsim.ltem.phase(field)
    imf = exsim.ltem.integrated_magnetic_flux_density(phase)
    assert (imf.array != 0).any()
