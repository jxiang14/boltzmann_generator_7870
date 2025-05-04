import torch

def loss_by_example(x, inverse_model):
    """
    Compute the loss for a given example using the forward model.
    
    Args:
        x (torch.Tensor): Input tensor.
        inverse_model (callable): Inverse model function.
    
    Returns:
        torch.Tensor: Computed loss.
    """
    z, log_r_xz = inverse_model(x)

    ml = u_latent(z) - log_r_xz
    loss = torch.mean(ml)
    return loss

def loss_kl(z, forward_model, u_config):
    """
    Compute the KL divergence loss for a given example using the forward model.
    
    Args:
        x (torch.Tensor): Input tensor.
        forward_model (callable): Forward model function.
    
    Returns:
        torch.Tensor: Computed KL divergence loss.
    """
    x, log_r_zx = forward_model(z)

    kl = u_config(x) - log_r_zx
    loss = torch.mean(kl)
    return loss

def loss_rc(x, reaction_coord, kde):
    reaction_coord_vals = reaction_coord(x)
    log_p = kde.evaluate(reaction_coord_vals)
    log_p = torch.mean(torch.from_numpy(log_p))
    return log_p

def u_latent(z, e_high=1e10, e_max=1e20):
    energy = 0.5 * torch.sum(z**2, dim=1)
    energy = energy.clamp(max=e_max)
    energy = torch.where(energy < e_high, energy, e_high + torch.log(energy - e_high + 1))
    return energy