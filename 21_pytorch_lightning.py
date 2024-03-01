import logging
import os

from lightning import LightningDataModule
from lightning import LightningModule
from lightning import Trainer
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms

# Hyper-Parameter
input_size: int = 784  # 28 * 28 pixel image size
hidden_size: int = 500
num_classes: int = 10
num_epochs: int = 2
batch_size: int = 100
learning_rate: float = .001


class NeuralNet(LightningModule):
    def __init__(self, input_size, hidden_size, num_classes) -> None:
        super(NeuralNet, self).__init__()
        logger.debug('Enter NeuralNet constructor')
        self.input_size: int = input_size
        self.l1: Linear = Linear(input_size, hidden_size)
        self.relu: ReLU = ReLU()
        self.l2: Linear = Linear(hidden_size, num_classes)
        self.training_step_outputs: list = []
        logger.debug('Exit NeuralNet constructor')

    def forward(self, x) -> Tensor:
        logger.debug('Enter')
        out: Tensor = self.l1(x)
        out: Tensor = self.relu(out)
        out: Tensor = self.l2(out)
        logger.debug('Exit')
        return out

    def training_step(self, batch, batch_idx) -> Tensor:
        logger.debug('Enter')
        images, labels = batch
        images: Tensor = images.reshape(-1, 28 * 28)

        # Forward pass
        outputs: Tensor = self(images)
        loss: Tensor = F.cross_entropy(outputs, labels)

        preds: str = ''
        self.training_step_outputs.append(preds)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        logger.debug('Exit')
        return loss

    def configure_optimizers(self) -> Optimizer:
        logger.debug('Enter')
        logger.debug('Exit')
        return Adam(model.parameters(), lr=learning_rate)

    def validation_step(self, batch, batch_idx) -> Tensor:
        logger.debug('Enter')
        images, labels = batch
        images: Tensor = images.reshape(-1, 28 * 28)

        # Forward pass
        outputs: Tensor = self(images)
        loss: Tensor = F.cross_entropy(outputs, labels)

        # dictionary: dict = {'val_loss': loss}
        # self.log_dict(dictionary, on_step=False, on_epoch=True)
        # return dict

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        logger.debug('Exit')
        return loss

    def on_train_epoch_end(self) -> None:
        # all_preds = torch.stack(self.training_step_outputs)
        self.training_step_outputs.clear()
        logger.debug('Exit')


class DataModule(LightningDataModule):
    def __init__(self, data_dir="./data", batch_size_param=batch_size):
        logger.debug('Enter DataModule constructor')
        super(DataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size_param
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        logger.debug('Exit DataModule constructor')

    def prepare_data(self) -> None:
        logger.debug('Enter')
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)
        logger.debug('Exit')

    def setup(self, stage: str) -> None:
        logger.debug('Enter')
        entire_dataset: VisionDataset = MNIST(self.data_dir, train=True, transform=transforms.ToTensor(), download=False)
        train_size: int = int(0.8 * len(entire_dataset))
        val_size: int = len(entire_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(entire_dataset, [train_size, val_size])
        self.test_dataset: VisionDataset = MNIST(self.data_dir, train=False, transform=transforms.ToTensor(),
                                                 download=False)
        logger.debug('Exit')

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        logger.debug('Enter')
        logger.debug('Exit')
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=os.cpu_count() // 2, persistent_workers=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        logger.debug('Enter')
        logger.debug('Exit')
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=os.cpu_count() // 2, persistent_workers=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        logger.debug('Enter')
        logger.debug('Exit')
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=os.cpu_count() // 2, persistent_workers=True)


if __name__ == '__main__':
    logging.basicConfig(filename='./log/pytorch_tutorial.log', encoding='utf-8',
                        format='%(asctime)s,%(msecs)-3d - %(levelname)-8s - %(filename)s:%(lineno)d - %(module)s - %(funcName)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.debug('Start __main__')
    trainer = Trainer(fast_dev_run=True)
    # trainer = Trainer(max_epochs=num_epochs, fast_dev_run=False)
    model: LightningModule = NeuralNet(input_size, hidden_size, num_classes)
    data_module: LightningDataModule = DataModule()
    trainer.fit(model, datamodule=data_module)
    trainer.validate(model, datamodule=data_module)
    trainer.validate(model, datamodule=data_module)
    logger.debug('End __main__')
