import torch
import matplotlib.pyplot as plt
import numpy as np

class Potential:
    def __init__(self, output_dir="../output/"):
        self.output_dir = output_dir

    def u(self, x):
        """
        Compute the potential energy for a given input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, 2).
        
        Returns:
            torch.Tensor: Potential energy of shape (N,).
        """
        raise NotImplementedError("Base Potential energy function should not be used.")
    
    def rc(self, x):
        """
        Compute the reaction coordinate for a given input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, 2).
        
        Returns:
            torch.Tensor: Reaction coordinate of shape (N,).
        """
        raise NotImplementedError("Base reaction coordinate function should not be used.")
    
    def plot(self, title):
        """
        Plot the potential energy surface in 2D.
        """
        print("Plotting the potential energy surface...")
        x1 = np.linspace(-5, 5, 50)
        x2 = np.linspace(-4, 4, 50)
        X1, X2 = np.meshgrid(x1, x2)
        Z = self.u(torch.tensor(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)

        print("Potential energy surface computed.")

        plt.figure(figsize=(8, 6))
        plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
        plt.colorbar(label='Potential Energy')
        plt.title(title)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

    def plot_3d(self, title):
        """
        Plot the potential energy surface in 3D.
        """
        print("Plotting the potential energy surface in 3D...")
        x1 = np.linspace(-4, 4, 100)
        x2 = np.linspace(-2, 2, 100)
        X1, X2 = np.meshgrid(x1, x2)
        Z = self.u(torch.tensor(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)

        print("Potential energy surface computed.")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X1, X2, Z, cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('Potential Energy')
        plt.show()

    def plot_points(self, points, ax, colors, title):
        """
        Plot the potential energy surface with points.
        
        Args:
            points (np.ndarray): Points to plot on the potential energy surface.
            ax (matplotlib.axes.Axes): Axes to plot on.
            colors (list): List of colors for the points.
            title (str): Title for the plot.
        """
        print("Plotting the potential energy surface with points...")
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        x1 = np.linspace(-7, 7, 50)
        x2 = np.linspace(-7, 7, 50)
        X1, X2 = np.meshgrid(x1, x2)
        Z = self.u(torch.tensor(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)

        print("Potential energy surface computed.")
        print("Points to plot:", points)

        contour = ax.contourf(X1, X2, Z, levels=50, cmap='viridis')
        plt.colorbar(contour, ax=ax, label='Potential Energy')
        
        ax.scatter(points[:, 0], points[:, 1], color=colors if colors is not None else 'red', s=2)
        ax.set_title(title)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

class DoubleWellPotential(Potential):
    """
    Double well potential for a 2D system.
    """
    def __init__(self, a=1, b=6.0, c=0, d=1, output_dir="../output/"):
        super().__init__(output_dir)
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def u(self, x):
        """
        Compute the potential energy for a given input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, 2).
        
        Returns:
            torch.Tensor: Potential energy of shape (N,).
        """
        if x.ndim == 1:
            x = torch.reshape(x, (1, 2))
        x1, x2 = x[:, 0], x[:, 1]
        return 0.25 * self.a * x1**4 - 0.5 * self.b * x1**2 + self.c * x1 + 0.5 * self.d * x2**2
    
    def rc(self, x):
        """
        Compute the reaction coordinate for a given input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, 2).
        
        Returns:
            torch.Tensor: Reaction coordinate of shape (N,).
        """
        if x.ndim == 1:
            x = torch.reshape(x, (1, 2))
        return x[:, 0]
    
    def plot(self):
        """
        Plot the potential energy surface.
        """
        super().plot("Double Well Potential")

    def plot_3d(self):
        """
        Plot the potential energy surface in 3D.
        """
        super().plot_3d("Double Well Potential (3D)")

    def plot_points(self, points, ax, colors):
        """
        Plot the potential energy surface with points.
        
        Args:
            points (np.ndarray): Points to plot on the potential energy surface.
            ax (matplotlib.axes.Axes): Axes to plot on.
            colors (list): List of colors for the points.
        """
        super().plot_points(points, ax, colors, "Double Well Potential with Points")

class GaussianPotential(Potential):
    """
    Gaussian potential for a 2D system.
    """
    def __init__(self, center=(0, 0), width=1.0, output_dir="../output/"):
        super().__init__(output_dir)
        self.center = center
        self.width = width

    def u(self, x):
        """
        Compute the potential energy for a given input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, 2).
        
        Returns:
            torch.Tensor: Potential energy of shape (N,).
        """
        if x.ndim == 1:
            x = torch.reshape(x, (1, 2))
        return torch.exp(-((x[:, 0] - self.center[0])**2 + (x[:, 1] - self.center[1])**2) / (2 * self.width**2))
    
    def plot(self):
        """
        Plot the potential energy surface.
        """
        super().plot("Gaussian Potential")

    def plot_points(self, points, ax, colors):
        """
        Plot the potential energy surface with points.
        
        Args:
            points (np.ndarray): Points to plot on the potential energy surface.
        """
        super().plot_points(points, ax, colors, "Gaussian Potential with Points")

class MCSampler():
    """
    Monte Carlo sampler for a given potential.
    """
    def __init__(self, potential, temperature=1.0, num_samples=1000, step_size=0.1, sample_freq=10):
        self.potential = potential
        self.num_samples = num_samples
        self.step_size = step_size
        self.temperature = temperature
        self.sample_freq = sample_freq

    def sample(self, initial_position, num_samples=1000):
        """
        Sample points from the potential using a random walk.
        
        Args:
            initial_position (torch.Tensor): Initial position for sampling.
            num_samples (int): Number of samples to generate.
        
        Returns:
            torch.Tensor: Sampled points.
        """
        samples = torch.zeros((num_samples, 2))
        current_position = initial_position

        for i in range(num_samples * self.sample_freq):
            new_position = current_position + torch.randn(2) * self.step_size
            delta_u = self.potential.u(new_position) - self.potential.u(current_position)
            if delta_u < 0 or torch.rand(1) < torch.exp(-delta_u):
                current_position = new_position
            if i % self.sample_freq == 0:
                samples[i // self.sample_freq] = current_position
        return samples