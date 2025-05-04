import torch
from boltzmann_generator_7870.real_nvp import RealNVP

def built_test_realnvp():
    input_dim = 2
    hidden_dim = 4
    num_layers = 2
    mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    model = RealNVP(input_dim, hidden_dim, num_layers, mask)
    return model

def test_realnvp_invertibility():
    torch.manual_seed(0)
    model = built_test_realnvp()
    z = torch.randn(10, 2)

    x, log_det_fwd = model.generator(z)
    z_recon, log_det_inv = model.inverse_generator(x)

    assert torch.allclose(z, z_recon, atol=1e-5)
    assert torch.allclose(log_det_fwd, -log_det_inv, atol=1e-5)

def test_realnvp_shape_consistency():
    model = built_test_realnvp()
    z = torch.randn(10, 2)
    x, log_det = model.generator(z)

    assert x.shape == z.shape
    assert log_det.shape == (10,)

def test_realnvp_identity_behavior():
    model = built_test_realnvp()

    for layer in model.layers:
        for p in layer.parameters():
            torch.nn.init.constant_(p, 0.0)

    z = torch.randn(10, 2)
    x, log_det = model.generator(z)
    z_inv, log_det_inv = model.inverse_generator(x)

    assert torch.allclose(z, x, atol=1e-6)
    assert torch.allclose(z, z_inv, atol=1e-6)
    assert torch.allclose(log_det, torch.zeros_like(log_det))
    assert torch.allclose(log_det_inv, torch.zeros_like(log_det_inv))