import torch
import lightning as L
from boltzmann_generator_7870.real_nvp import RealNVP
from boltzmann_generator_7870.loss import loss_by_example, loss_kl, loss_rc
from torch.distributions import MultivariateNormal
from scipy.stats import gaussian_kde as KDE

class BoltzmannGenerator(L.LightningModule):
    def __init__(self, pot, mask, kde, output_dir, loss_weights=None, dim=2):
        super().__init__()
        self.model = RealNVP(input_dim=2, hidden_dim=128, num_layers=3, mask=mask)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        if loss_weights is not None:
            self.loss_weights = loss_weights
        else:
            self.loss_weights = [1.0, 1.0, 0.0]
        self.pot = pot
        self.output_dir = output_dir
        self.best_val_loss = float("inf")
        self.kde = kde

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        """
        if batch_idx == 0:
            self.train_loss = 0
        x, z = batch["x"], batch["z"]
        l_ex = 0
        l_kl = 0
        l_rc = 0
        if self.loss_weights[0] > 0:
            l_ex = loss_by_example(x, self.model.inverse_generator)
        if self.loss_weights[1] > 0:
            l_kl = loss_kl(z, self.model.generator, self.pot.u)
        if self.loss_weights[2] > 0:
            l_rc = loss_rc(x, self.pot.rc, self.kde)
        loss = (self.loss_weights[0] * l_ex +
                self.loss_weights[1] * l_kl +
                self.loss_weights[2] * l_rc)
        self.train_loss += loss.item()
        return loss
    
    def on_train_epoch_end(self):
        """
        Training epoch end for the model.
        """
        # Logging to TensorBoard (if installed) by default
        self.log("Total train_loss", self.train_loss)
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.
        """
        if batch_idx == 0:
            self.val_loss = 0
        x, z = batch["x"], batch["z"]
        l_ex = 0
        l_kl = 0
        l_rc = 0
        if self.loss_weights[0] > 0:
            l_ex = loss_by_example(x, self.model.inverse_generator)
        if self.loss_weights[1] > 0:
            l_kl = loss_kl(z, self.model.generator, self.pot.u)
        if self.loss_weights[2] > 0:
            l_rc = loss_rc(x, self.pot.rc, self.kde)
        loss = (self.loss_weights[0] * l_ex +
                self.loss_weights[1] * l_kl +
                self.loss_weights[2] * l_rc)
        self.val_loss += loss.item()
        return loss

    def on_validation_epoch_end(self):
        """
        Validation epoch end for the model.
        """
        # Logging to TensorBoard (if installed) by default
        self.log("Total val_loss", self.val_loss)
        if self.val_loss < self.best_val_loss:
            self.best_val_loss = self.val_loss
            self.log("Best val_loss", self.best_val_loss)
            # Save the model checkpoint
            torch.save(self.model.state_dict(), self.output_dir + "best_model.pt")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=0)
        return optimizer
    
