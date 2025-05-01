import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_resulting_generator(model, pot):
    """
    Plot the resulting generator.
    """
    z_samples = torch.randn(1000, 2)  # Example input data
    x_samples = model.generator(z_samples)[0].detach().numpy()

    pot.plot_points(x_samples)

def plot_latent_representation(model, pot, x_samples):
    """
    Plot the latent representation of the generator.
    """
    z_samples = model.inverse_generator(x_samples)[0].detach().numpy()
    pot.plot_points(z_samples)