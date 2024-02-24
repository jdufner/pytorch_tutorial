from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# Hyper-Parameter
input_size: int = 784  # 28 * 28 pixel image size
hidden_size: int = 500
num_classes: int = 10
num_epochs: int = 2
batch_size: int = 100
learning_rate: float = .001


class NeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes) -> None:
        super(NeuralNet, self).__init__()
        self.input_size: int = input_size
        self.l1: Linear = Linear(input_size, hidden_size)
        self.relu: ReLU = ReLU()
        self.l2: Linear = Linear(hidden_size, num_classes)
        self.training_step_outputs: list = []

    def forward(self, x) -> Tensor:
        out: Tensor = self.l1(x)
        out: Tensor = self.relu(out)
        out: Tensor = self.l2(out)
        return out

    def training_step(self, batch, batch_idx) -> Tensor:
        images, labels = batch
        images: Tensor = images.reshape(-1, 28 * 28)

        # Forward pass
        outputs: Tensor = self(images)
        loss: Tensor = F.cross_entropy(outputs, labels)

        preds: str = ''
        self.training_step_outputs.append(preds)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Optimizer:
        return Adam(model.parameters(), lr=learning_rate)

    def train_dataloader(self) -> DataLoader:
        dataset: VisionDataset = MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        data_loader: DataLoader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=3)
        return data_loader

    def validation_step(self, batch, batch_idx) -> Tensor:
        images, labels = batch
        images: Tensor = images.reshape(-1, 28 * 28)

        # Forward pass
        outputs: Tensor = self(images)
        loss: Tensor = F.cross_entropy(outputs, labels)

        dictionary: dict = {'val_loss': loss}
        self.log_dict(dictionary, on_step=False, on_epoch=True)
        # return dict

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def val_dataloader(self) -> DataLoader:
        dataset: VisionDataset = MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
        data_loader: DataLoader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=3)
        return data_loader

    def on_train_epoch_end(self) -> None:
        # all_preds = torch.stack(self.training_step_outputs)
        self.training_step_outputs.clear()


if __name__ == '__main__':
    # trainer = Trainer(fast_dev_run=True)
    trainer = Trainer(max_epochs=num_epochs, fast_dev_run=False)
    model = NeuralNet(input_size, hidden_size, num_classes)
    trainer.fit(model)
