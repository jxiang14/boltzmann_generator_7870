import torch
import lightning as L
from lightning.pytorch.utilities import CombinedLoader
import argparse
from lightning.pytorch import loggers as pl_loggers
from boltzmann_generator_7870.potential import DoubleWellPotential, MCSampler, GaussianPotential
from boltzmann_generator_7870.generator import BoltzmannGenerator
from boltzmann_generator_7870.datasets import XSamplesDataModule, ZSamplesDataModule
from boltzmann_generator_7870.real_nvp import RealNVP
from boltzmann_generator_7870.logging_plots import plot_resulting_generator, plot_latent_representation

def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/",
        help="Output directory for the best model",
    )
    parser.add_argument(
        "--logs_output_dir",
        type=str,
        default="./output/logs/",
        help="Output directory for the logs",
    )

    return parser.parse_args()

def generate_samples(pot, num_samples=1000):
    mc = MCSampler(potential=pot)
    left_well = torch.tensor([-2.0, 0.0])
    right_well = torch.tensor([2.0, 0.0])
    x_samples_left = mc.sample(left_well, num_samples=num_samples)
    x_samples_right = mc.sample(right_well, num_samples=num_samples)
    x_samples = torch.cat([x_samples_left, x_samples_right], dim=0)
    return x_samples, x_samples_left, x_samples_right

def main(args):
    print("Initializing the Double Well Potential...")
    pot = DoubleWellPotential()
    weights = [1.0, 1.0, 0.0]
    mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    bg = BoltzmannGenerator(pot, mask, args.output_dir, loss_weights=weights)
    # x_samples, x_samples_left, x_samples_right = generate_samples(pot, num_samples=10000)
    # torch.save(x_samples, args.output_dir + "x_samples.pt")
    # torch.save(x_samples_left, args.output_dir + "x_samples_left.pt")
    # torch.save(x_samples_right, args.output_dir + "x_samples_right.pt")
    x_samples = torch.load(args.output_dir + "x_samples.pt")
    x_samples_data_module = XSamplesDataModule(x_samples, batch_size=1024)
    num_z_samples = 20000
    z_samples = torch.randn(num_z_samples, 2)
    z_samples_data_module = ZSamplesDataModule(z_samples, batch_size=1024)
    
    train_x_dataloader = x_samples_data_module.train_dataloader()
    train_z_dataloader = z_samples_data_module.train_dataloader()
    train_iterables = {'x': train_x_dataloader, 'z': train_z_dataloader}
    val_x_dataloader = x_samples_data_module.val_dataloader()
    val_z_dataloader = z_samples_data_module.val_dataloader()
    val_iterables = {'x': val_x_dataloader, 'z': val_z_dataloader}
    val_dataloader = CombinedLoader(val_iterables, mode="max_size_cycle")
    train_dataloader = CombinedLoader(train_iterables, mode="max_size_cycle")

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.logs_output_dir)

    trainer = L.Trainer(max_epochs=20, 
                        check_val_every_n_epoch=2,
                        num_sanity_val_steps=0, 
                        logger=tb_logger
                        )
    print("Starting training...")
    trainer.fit(bg, train_dataloader, val_dataloader)
    print("Training completed.")

    model_dict = torch.load(args.output_dir + "best_model.pt")
    model = RealNVP(input_dim=2, hidden_dim=128, num_layers=3, mask=mask)
    model.load_state_dict(model_dict)
    plot_resulting_generator(model, pot)
    pot_latent = GaussianPotential()
    x_samples_left = torch.load(args.output_dir + "x_samples_left.pt")
    x_samples_right = torch.load(args.output_dir + "x_samples_right.pt")
    plot_latent_representation(model, pot_latent, x_samples_left)
    plot_latent_representation(model, pot_latent, x_samples_right)

    # weights = [0.0, 1.0, 0.0]
    # bg2 = BoltzmannGenerator(loss_weights=weights)
    # trainer.fit(bg2, train_dataloader)

    # # pot.plot_3d()
    # plot_resulting_generator(bg, pot)

if __name__ == "__main__":
    args = _parse_args()
    main(args)