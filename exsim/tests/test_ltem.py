import pytest
import discretisedfield as df
import exsim


def test_relativistic_wavelength():
    assert exsim.ltem.relativistic_wavelength(0) == float('inf')


def test_ltem_phase():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 1), cell=(1, 1, 1))
    field = df.Field(mesh, dim=3, value=(0, 0, 1))
    phase, ft_phase = exsim.ltem.phase(field)
    assert (phase.real.array == 0).all()
    assert (phase.imag.array == 0).all()

def test_ltem_phase_neel():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 1), cell=(1, 1, 1))
    def f_val(position):
        x, y, z = position
        if x < 5:
            return (1, 0, 0)
        else:
            return (-1, 0, 0)
    field = df.Field(mesh, dim=3, value=f_val)
    phase, ft_phase = exsim.ltem.phase(field)
    assert (phase.real.array == 0).all()
    assert (phase.imag.array == 0).all()

def test_ltem_phase_bloch():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 1), cell=(1, 1, 1))
    def f_val(position):
        x, y, z = position
        if x < 5:
            return (0, 1, 0)
        else:
            return (0, -1, 0)
    field = df.Field(mesh, dim=3, value=f_val)
    phase, ft_phase = exsim.ltem.phase(field)
    assert (phase.real.array != 0).all()
    assert (phase.imag.array != 0).all()


def test_defocus_image():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 1), cell=(1, 1, 1))
    field = df.Field(mesh, dim=3, value=(0, 0, 1))
    phase, ft_phase = exsim.ltem.phase(field)
    with pytest.raises(RuntimeError):
        exsim.ltem.defocus_image(phase)
