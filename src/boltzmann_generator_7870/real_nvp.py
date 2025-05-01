import torch
import torch.nn as nn


class RealNVP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, mask=None):
        super(RealNVP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.mask = mask

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            order = i % 2
            layer = AffineCouplingLayer(input_dim, hidden_dim, order, mask)
            self.layers.append(layer)
        # layer = TanhAffineCouplingLayer(input_dim, hidden_dim, order, mask)
        # self.layers.append(layer)

    def generator(self, z):
        # z += torch.randn_like(z) * 0.1
        log_r_zx = 0
        for layer in reversed(self.layers):
            z, log_r_zx_step = layer.generator(z)
            log_r_zx -= log_r_zx_step
        return z, log_r_zx

    def inverse_generator(self, x):
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

    # def inverse_generator(self, x):
    #     masked_z = x * self.mask
    #     s = self.s(masked_z) * (1 - self.mask)
    #     t = self.t(masked_z) * (1 - self.mask)
    #     z = masked_z + (masked_z - t) * torch.exp(-s) * (1 - self.mask)
    #     log_r = torch.sum(s, dim=-1)
    #     return z, log_r
    
    # def generator(self, z):
    #     masked_x = z * self.mask
    #     s = self.s(masked_x) * (1 - self.mask)
    #     t = self.t(masked_x) * (1 - self.mask)
    #     x = masked_x + (masked_x * torch.exp(s) + t) * (1 - self.mask)
    #     log_r = torch.sum(s, dim=-1)
    #     return x, log_r

    # def inverse_generator(self, x):
    #     s = self.s(x)
    #     t = self.t(x)
    #     z = x * torch.exp(s) + t
    #     log_r = torch.sum(s, dim=-1)
    #     return z, log_r

    # def inverse_generator(self, x):
    #     x1, x2 = x.chunk(2, dim=1)
    #     if self.order == 1:
    #         x1, x2 = x2, x1
    #     s = self.s(x1)
    #     t = self.t(x1)

    #     z1 = x1
    #     z2 = x2 * torch.exp(s) + t

    #     if self.order == 1:
    #         z1, z2 = z2, z1
    #         log_r = torch.cat([torch.zeros((z1.shape[0],)), torch.sum(s, dim=-1)], dim=0)
    #     else:
    #         log_r = torch.cat([torch.sum(s, dim=-1), torch.zeros((z1.shape[0],))], dim=0)
    #     return torch.cat([z1, z2], dim=1), torch.sum(s, dim=1)

    # def generator(self, z):
    #     s = self.s(z)
    #     t = self.t(z)
    #     x = (z - t) * torch.exp(-s)
    #     log_r = torch.sum(s, dim=-1)
    #     return x, log_r
    
    # def generator(self, z):
    #     z1, z2 = z.chunk(2, dim=1)
    #     if self.order == 1:
    #         z1, z2 = z2, z1

    #     s = self.s(z1)
    #     t = self.t(z1)

    #     x1 = z1
    #     x2 = (z2 - t) * torch.exp(-s)

    #     if self.order == 1:
    #         x1, x2 = x2, x1
    #         log_r = torch.cat([torch.zeros((z1.shape[0],)), torch.sum(s, dim=-1)], dim=0)
    #     else:
    #         log_r = torch.cat([torch.sum(s, dim=-1), torch.zeros((z1.shape[0],))], dim=0)

    #     return torch.cat([x1, x2], dim=1), torch.sum(s, dim=1)
    

    def generator(self, z):
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
        x1, x2 = x.chunk(2, dim=1)
        if self.order == 1:
            x1, x2 = x2, x1
        s = self.s(x1)
        t = self.t(x1)
        z2 = x2 * torch.exp(s) + t
        if self.order == 1:
            x1, z2 = z2, x1
        return torch.cat([x1, z2], dim=1), torch.sum(s, dim=1)
    
class TanhAffineCouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, order, mask):
        super(TanhAffineCouplingLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.order = order
        self.mask = mask[order]

        self.s = nn.Sequential(
            nn.Linear(input_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim//2),
            nn.Tanh()
        )

        self.t = nn.Sequential(
            nn.Linear(input_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim//2)
        )

    # def inverse_generator(self, x):
    #     masked_z = x * self.mask
    #     s = self.s(masked_z) * (1 - self.mask)
    #     t = self.t(masked_z) * (1 - self.mask)
    #     z = masked_z + (masked_z - t) * torch.exp(-s) * (1 - self.mask)
    #     log_r = torch.sum(s, dim=-1)
    #     return z, log_r
    
    # def generator(self, z):
    #     masked_x = z * self.mask
    #     s = self.s(masked_x) * (1 - self.mask)
    #     t = self.t(masked_x) * (1 - self.mask)
    #     x = masked_x + (masked_x * torch.exp(s) + t) * (1 - self.mask)
    #     log_r = torch.sum(s, dim=-1)
    #     return x, log_r

    # def inverse_generator(self, x):
    #     s = self.s(x)
    #     t = self.t(x)
    #     z = x * torch.exp(s) + t
    #     log_r = torch.sum(s, dim=-1)
    #     return z, log_r

    # def inverse_generator(self, x):
    #     x1, x2 = x.chunk(2, dim=1)
    #     if self.order == 1:
    #         x1, x2 = x2, x1
    #     s = self.s(x1)
    #     t = self.t(x1)

    #     z1 = x1
    #     z2 = x2 * torch.exp(s) + t

    #     if self.order == 1:
    #         z1, z2 = z2, z1
    #         log_r = torch.cat([torch.zeros((z1.shape[0],)), torch.sum(s, dim=-1)], dim=0)
    #     else:
    #         log_r = torch.cat([torch.sum(s, dim=-1), torch.zeros((z1.shape[0],))], dim=0)
    #     return torch.cat([z1, z2], dim=1), torch.sum(s, dim=1)

    # def generator(self, z):
    #     s = self.s(z)
    #     t = self.t(z)
    #     x = (z - t) * torch.exp(-s)
    #     log_r = torch.sum(s, dim=-1)
    #     return x, log_r
    
    # def generator(self, z):
    #     z1, z2 = z.chunk(2, dim=1)
    #     if self.order == 1:
    #         z1, z2 = z2, z1

    #     s = self.s(z1)
    #     t = self.t(z1)

    #     x1 = z1
    #     x2 = (z2 - t) * torch.exp(-s)

    #     if self.order == 1:
    #         x1, x2 = x2, x1
    #         log_r = torch.cat([torch.zeros((z1.shape[0],)), torch.sum(s, dim=-1)], dim=0)
    #     else:
    #         log_r = torch.cat([torch.sum(s, dim=-1), torch.zeros((z1.shape[0],))], dim=0)

    #     return torch.cat([x1, x2], dim=1), torch.sum(s, dim=1)
    

    def generator(self, z):
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
        x1, x2 = x.chunk(2, dim=1)
        if self.order == 1:
            x1, x2 = x2, x1
        s = self.s(x1)
        t = self.t(x1)
        z2 = x2 * torch.exp(s) + t
        if self.order == 1:
            x1, z2 = z2, x1
        return torch.cat([x1, z2], dim=1), torch.sum(s, dim=1)