import torch
import matplotlib.pyplot as plt
import numpy as np

class DoubleWellPotential:
    """
    Double well potential for a 2D system.
    """
    def __init__(self, a=1, b=6.0, c=0, d=1):
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
    
    def plot(self):
        """
        Plot the potential energy surface.
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
        plt.title('Double Well Potential')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

    def plot_3d(self):
        """
        Plot the potential energy surface in 3D.
        """
        print("Plotting the potential energy surface in 3D...")
        x1 = np.linspace(-5, 5, 100)
        x2 = np.linspace(-4, 4, 100)
        X1, X2 = np.meshgrid(x1, x2)
        Z = self.u(torch.tensor(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)

        print("Potential energy surface computed.")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X1, X2, Z, cmap='viridis')
        ax.set_title('Double Well Potential (3D)')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('Potential Energy')
        plt.show()

    def plot_points(self, points):
        """
        Plot the potential energy surface with points.
        
        Args:
            points (np.ndarray): Points to plot on the potential energy surface.
        """
        print("Plotting the potential energy surface with points...")
        x1 = np.linspace(-5, 5, 50)
        x2 = np.linspace(-4, 4, 50)
        X1, X2 = np.meshgrid(x1, x2)
        Z = self.u(torch.tensor(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)

        print("Potential energy surface computed.")
        print("Points to plot:", points)

        plt.figure(figsize=(8, 6))
        plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
        plt.colorbar(label='Potential Energy')
        plt.scatter(points[:, 0], points[:, 1], color='red', s=10)
        plt.title('Double Well Potential with Points')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

class GaussianPotential:
    """
    Gaussian potential for a 2D system.
    """
    def __init__(self, center=(0, 0), width=1.0):
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
        print("Plotting the Gaussian potential energy surface...")
        x1 = np.linspace(-5, 5, 50)
        x2 = np.linspace(-4, 4, 50)
        X1, X2 = np.meshgrid(x1, x2)
        Z = self.u(torch.tensor(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)

        print("Gaussian potential energy surface computed.")

        plt.figure(figsize=(8, 6))
        plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
        plt.colorbar(label='Potential Energy')
        plt.title('Gaussian Potential')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

    def plot_points(self, points):
        """
        Plot the potential energy surface with points.
        
        Args:
            points (np.ndarray): Points to plot on the potential energy surface.
        """
        print("Plotting the Gaussian potential energy surface with points...")
        x1 = np.linspace(-5, 5, 50)
        x2 = np.linspace(-4, 4, 50)
        X1, X2 = np.meshgrid(x1, x2)
        Z = self.u(torch.tensor(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)

        print("Gaussian potential energy surface computed.")
        print("Points to plot:", points)

        plt.figure(figsize=(8, 6))
        plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
        plt.colorbar(label='Potential Energy')
        plt.scatter(points[:, 0], points[:, 1], color='red', s=10)
        plt.title('Gaussian Potential with Points')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

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