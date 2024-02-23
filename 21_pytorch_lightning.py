from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler, TRAIN_DATALOADERS

# Hyper-Parameter
input_size = 784  # 28 * 28 pixel image size
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = .001


class NeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28 * 28)

        # Forward pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        tensorboard_logs = {'train_loss': loss}
        #return {'loss': loss, 'log': tensorboard_logs}
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(model.parameters(), lr=learning_rate)

    def train_dataloader(self):
        dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=3)
        return data_loader

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28 * 28)

        # Forward pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        return {'val_loss': loss}

    def val_dataloader(self):
        dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=3)
        return data_loader

    #def on_validation_epoch_end(self):
        # avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # tensorboard_logs = {'tavg_val_loss': avg_loss}
        # return {'val_loss': avg_loss, 'log': tensorboard_logs}


if __name__ == '__main__':
    #trainer = Trainer(fast_dev_run=True)
    trainer = Trainer(max_epochs=num_epochs, fast_dev_run=False)
    model = NeuralNet(input_size, hidden_size, num_classes)
    trainer.fit(model)
