import torch.utils.data as data
import torch
import lightning as L

class ZSamplesDataset(data.Dataset):
    def __init__(self, z_samples: torch.tensor):
        self.z_samples = z_samples

    def __len__(self):
        return self.z_samples.shape[0]

    def __getitem__(self, idx):
        return (self.z_samples[idx])
    
class XSamplesDataset(data.Dataset):
    def __init__(self, x_samples: torch.tensor):
        self.x_samples = x_samples

    def __len__(self):
        return self.x_samples.shape[0]

    def __getitem__(self, idx):
        return (self.x_samples[idx])
    
class ZSamplesDataModule(L.LightningDataModule):
    def __init__(self, z_samples: torch.tensor, batch_size: int = 32):
        super().__init__()
        self.z_samples = z_samples
        self.batch_size = batch_size
        self.dataset = ZSamplesDataset(self.z_samples)
        train_end = int(0.8 * len(self.z_samples))
        self.train_dataset = data.Subset(self.dataset, range(0, train_end))
        self.valid_dataset = data.Subset(self.dataset, range(train_end, len(self.z_samples)))

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return data.DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)
    
class XSamplesDataModule(L.LightningDataModule):
    def __init__(self, x_samples: torch.tensor, batch_size: int = 32):
        super().__init__()
        self.x_samples = x_samples
        self.batch_size = batch_size
        self.dataset = XSamplesDataset(self.x_samples)
        train_end = int(0.8 * len(self.x_samples))
        self.train_dataset = data.Subset(self.dataset, range(0, train_end))
        self.valid_dataset = data.Subset(self.dataset, range(train_end, len(self.x_samples)))

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return data.DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)