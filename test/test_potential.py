import torch

from boltzmann_generator_7870.potential import DoubleWellPotential, GaussianPotential, MCSampler

def test_doublewell_u_and_rc():
    pot = DoubleWellPotential()
    x = torch.tensor([[1.0, 2.0], [-1.0, 0.0]])

    u_vals = pot.u(x)
    rc_vals = pot.rc(x)

    assert u_vals.shape == (2,)
    assert rc_vals.shape == (2,)

    expected_u0 = 0.25 * 1 * 1**4 - 0.5 * 6 * 1**2 + 0.5 * 1 * 2**2
    expected_u1 = 0.25 * 1 * 1**4 - 0.5 * 6 * 1**2 + 0.5 * 1 * 0**2

    assert torch.allclose(u_vals, torch.tensor([expected_u0, expected_u1]), atol=1e-5)
    assert torch.allclose(rc_vals, torch.tensor([1.0, -1.0]))

def test_gaussian_u():
    pot = GaussianPotential(center=(0, 0), width=1.0)
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    
    u_vals = pot.u(x)
    
    assert torch.allclose(u_vals[0], torch.tensor(1.0), atol=1e-5)
    assert u_vals[1] < u_vals[0]

def test_mc_sampler_returns_correct_shape():
    pot = DoubleWellPotential()
    sampler = MCSampler(potential=pot, temperature=1.0, step_size=0.5, sample_freq=5)

    initial_position = torch.tensor([0.0, 0.0])
    samples = sampler.sample(initial_position, num_samples=100)

    assert samples.shape == (100, 2)

def test_mc_sampler_moves():
    pot = DoubleWellPotential()
    sampler = MCSampler(potential=pot, temperature=1.0, step_size=0.5, sample_freq=1)

    initial_position = torch.tensor([0.0, 0.0])
    samples = sampler.sample(initial_position, num_samples=10)

    assert not torch.allclose(samples, torch.zeros_like(samples))
    std = torch.std(samples, dim=0)
    assert (std > 0).all()
