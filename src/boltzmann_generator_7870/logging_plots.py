import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde as KDE

def plot_resulting_generator(model, pot, ax, colors=None):
    """
    Plot the resulting generator.
    """
    z_samples = torch.randn(1000, 2)
    x_samples = model.generator(z_samples)[0].detach().numpy()

    pot.plot_points(x_samples, ax, colors)

def plot_config_samples(pot, x_samples, ax, colors=None):
    """
    Plot the configuration samples.
    """
    if colors is None:
        colors = ["red" if i < len(x_samples) // 2 else "blue" for i in range(len(x_samples))]
    pot.plot_points(x_samples, ax, colors)

def plot_latent_representation(model, pot, x_samples, ax, colors=None):
    """
    Plot the latent representation of the generator.
    """
    if colors is None:
        colors = ["red" if i < len(x_samples) // 2 else "blue" for i in range(len(x_samples))]
    z_samples = model.inverse_generator(x_samples)[0].detach().numpy()
    pot.plot_points(z_samples, ax, colors)

def plot_latent_samples(pot, ax, colors=None):
    """
    Plot the latent samples.
    """
    z_samples = torch.randn(1000, 2)
    pot.plot_points(z_samples, ax, colors)

def plot_results(model, pot, latent_pot, x_samples, colors=None):
    """
    Plot the results of the generator and the latent representation.
    """
    fig, axes = plt.subplots(1, 4, figsize=(35,3.5))
    plt.subplots_adjust(wspace=1)
    plot_latent_samples(latent_pot, axes[0], colors)
    plot_resulting_generator(model, pot, axes[1], colors)
    plot_config_samples(pot, x_samples, axes[2], colors)
    plot_latent_representation(model, latent_pot, x_samples, axes[3], colors)
    plt.show()


def plot_latent_interpolation(model, pot, latent_pot, x_samples, colors=None):
    """
    Plot the latent interpolation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(25, 5))
    x_samples = x_samples[::500]
    z_samples = model.inverse_generator(x_samples)[0].detach().numpy()
    colors = []
    z_interps = []
    x_interps = []

    for i in range(len(z_samples) - 1):
        z_start = z_samples[i]
        z_end = z_samples[i + 1]
        
        t_vals = np.linspace(0, 1, 10)
        z_interp = np.outer(1 - t_vals, z_start) + np.outer(t_vals, z_end)

        x_interp = model.generator(torch.tensor(z_interp, dtype=torch.float))[0].detach().numpy()

        color = f"C{i % 10}"
        color = [color for _ in range(len(z_interp))]
        colors.append(color)
        z_interps.append(z_interp)
        x_interps.append(x_interp)

    z_interps = np.concatenate(z_interps)
    x_interps = np.concatenate(x_interps)
    colors = np.concatenate(colors)
    print("z_interps shape:", z_interps.shape)

    latent_pot.plot_points(z_interps, axes[0], colors)

    pot.plot_points(x_interps, axes[1], colors)

    axes[0].set_title("Latent Space Interpolations")
    axes[1].set_title("Configuration Space Interpolations")
    axes[0].legend()
    axes[1].legend()
    plt.show()

def get_weights(x_samples, z_samples, log_r_zx, pot, latent_pot):
    """
    Compute the weights for the samples.
    """
    u_x = pot.u(x_samples)
    u_z = latent_pot.u(z_samples)
    weights = torch.exp(-u_x + u_z + log_r_zx)
    return weights.detach().numpy()

def plot_kde(model, pot, latent_pot):
    """
    Plot the KDE.
    """
    z = torch.randn(5000, 2)
    x_samples, log_r_zx = model.generator(z)
    kde = KDE(pot.rc(x_samples).detach().numpy(), weights=get_weights(x_samples, z, log_r_zx, pot, latent_pot))

    x = np.linspace(-3, 3, 100)
    y = kde.evaluate(x)
    y = -np.log(y + 1e-12)
    y = y - np.min(y) - 8.5
    x_torch = torch.tensor(x).reshape(-1, 1)
    u = pot.u(torch.cat((x_torch, torch.zeros_like(x_torch)), dim=1)).detach().numpy()
    plt.plot(x, y)
    plt.plot(x, u)
    plt.legend(["KDE", "Potential"])
    plt.xlabel("x1")
    plt.title("KDE")
    plt.show()

def plot_histogram(model, pot, latent_pot):
    """
    Plot the histogram of the samples.
    """
    z = torch.randn(1000, 2)
    x_samples, log_r_zx = model.generator(z)
    f = pot.rc(x_samples).detach().numpy()
    f = -f
    f = f - np.min(f)
    n, bins = np.histogram(f, bins=50, weights=get_weights(x_samples, z, log_r_zx, pot, latent_pot))
    plt.scatter(bins[:-1], n, s=1)
    x = np.linspace(-3, 3, 100)
    x_torch = torch.tensor(x).reshape(-1, 1)
    u = pot.u(torch.cat((x_torch, torch.zeros_like(x_torch)), dim=1)).detach().numpy()
    plt.plot(x, u)
    plt.title("Histogram of Samples")
    plt.show()