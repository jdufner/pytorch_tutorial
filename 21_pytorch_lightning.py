from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# Hyper-Parameter
input_size = 784  # 28 * 28 pixel image size
hidden_size = 500
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = .001


class NeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        self.training_step_outputs = []

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

        preds = ''
        self.training_step_outputs.append(preds)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
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

    def on_train_epoch_end(self):
        # all_preds = torch.stack(self.training_step_outputs)
        self.training_step_outputs.clear()


if __name__ == '__main__':
    #trainer = Trainer(fast_dev_run=True)
    trainer = Trainer(max_epochs=num_epochs, fast_dev_run=False)
    model = NeuralNet(input_size, hidden_size, num_classes)
    trainer.fit(model)
