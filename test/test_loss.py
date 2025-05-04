import pytest
import torch
import numpy as np

from boltzmann_generator_7870.loss import loss_by_example, loss_kl, loss_rc, u_latent

# Mock functions for models and KDE
def mock_inverse_model(x):
    z = x + 0.1
    log_r_xz = torch.sum(x, dim=1)
    return z, log_r_xz

def mock_forward_model(z):
    x = z - 0.1
    log_r_zx = torch.sum(z, dim=1)
    return x, log_r_zx

def mock_u_config(x):
    return torch.sum(x**2, dim=1)

class MockKDE:
    def evaluate(self, values):
        return torch.sum(values).detach().numpy()

def mock_reaction_coord(x):
    return x[:, 0]

def test_loss_by_example():
    x = torch.randn(10, 2)
    loss = loss_by_example(x, mock_inverse_model)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()

def test_loss_by_example_decreases_for_standard_normal():
    dim = 2
    batch_size = 10000

    def inverse_model_identity(x):
        z = x.clone()
        log_r_xz = -0.5 * torch.sum(x**2, dim=1)
        return z, log_r_xz

    x = torch.randn(batch_size, dim)
    loss = loss_by_example(x, inverse_model_identity)

    expected_entropy = 2.0
    assert torch.isclose(loss, torch.tensor(expected_entropy), rtol=0.1)

def test_loss_kl():
    z = torch.randn(10, 2)
    loss = loss_kl(z, mock_forward_model, mock_u_config)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()

def test_loss_rc():
    x = torch.randn(10, 2)
    kde = MockKDE()
    val = loss_rc(x, mock_reaction_coord, kde)
    assert isinstance(val, torch.Tensor)
    assert val.shape == ()

def test_u_latent_normal():
    z = torch.randn(300, 2)
    energy = u_latent(z)
    assert energy.shape == (300,)
    assert torch.all(energy >= 0)
    assert torch.mean(energy) - 1 < 1e-1

def test_u_latent_high_energy_clipping():
    z = torch.full((10, 5), 1e6)
    energy = u_latent(z)
    print(energy)
    assert energy.shape == (10,)
    assert torch.all(energy < 1e20)
    assert torch.all(torch.isfinite(energy))

if __name__ == "__main__":
    pytest.main()