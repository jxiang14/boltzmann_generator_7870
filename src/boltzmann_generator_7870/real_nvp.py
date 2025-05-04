import torch
import torch.nn as nn


class RealNVP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, mask=None):
        super(RealNVP, self).__init__()
        self.input_dim = input_dim
        self.mask = mask

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            order = i % 2
            layer = AffineCouplingLayer(input_dim, hidden_dim, order, mask)
            self.layers.append(layer)

    def generator(self, z):
        """
        Generate samples from the latent space to the data space.
        Args:
            z (torch.Tensor): Latent space samples of shape (N, input_dim).
        Returns:
            x (torch.Tensor): Generated samples in the data space of shape (N, input_dim).
            log_r_zx (torch.Tensor): Logarithm of the Jacobian determinant of the transformation.
        """
        # z += torch.randn_like(z) * 0.1
        log_r_zx = 0
        for layer in reversed(self.layers):
            z, log_r_zx_step = layer.generator(z)
            log_r_zx -= log_r_zx_step
        return z, log_r_zx

    def inverse_generator(self, x):
        """
        Generate samples from the data space to the latent space.
        Args:
            x (torch.Tensor): Data space samples of shape (N, input_dim).
        Returns:
            z (torch.Tensor): Generated samples in the latent space of shape (N, input_dim).
            log_r_xz (torch.Tensor): Logarithm of the Jacobian determinant of the transformation.
        """
        # x += torch.randn_like(x) * 0.1
        log_r_xz = 0
        for layer in self.layers:
            x, log_r_xz_step = layer.inverse_generator(x)
            log_r_xz += log_r_xz_step
        return x, log_r_xz


class AffineCouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, order, mask):
        super(AffineCouplingLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.order = order
        self.mask = mask[order]

        self.s = nn.Sequential(
            nn.Linear(input_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim//2)
        )

        self.t = nn.Sequential(
            nn.Linear(input_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim//2)
        )

    def generator(self, z):
        """
        One layer of the RealNVP model.
        Args:
            z (torch.Tensor): Latent space samples of shape (N, input_dim).
        Returns:
            x (torch.Tensor): Generated samples in the data space of shape (N, input_dim).
            log_r_zx (torch.Tensor): Logarithm of the Jacobian determinant of the transformation.
        """
        z1, z2 = z.chunk(2, dim=1)
        if self.order == 1:
            z1, z2 = z2, z1
        s = self.s(z1)
        t = self.t(z1)
        x2 = (z2 - t) * torch.exp(-s)
        if self.order == 1:
            z1, x2 = x2, z1
        return torch.cat([z1, x2], dim=1), torch.sum(s, dim=1)

    def inverse_generator(self, x):
        """
        One layer of the RealNVP model.
        Args:
            x (torch.Tensor): Data space samples of shape (N, input_dim).
        Returns:
            z (torch.Tensor): Generated samples in the latent space of shape (N, input_dim).
            log_r_xz (torch.Tensor): Logarithm of the Jacobian determinant of the transformation.
        """
        x1, x2 = x.chunk(2, dim=1)
        if self.order == 1:
            x1, x2 = x2, x1
        s = self.s(x1)
        t = self.t(x1)
        z2 = x2 * torch.exp(s) + t
        if self.order == 1:
            x1, z2 = z2, x1
        return torch.cat([x1, z2], dim=1), torch.sum(s, dim=1)