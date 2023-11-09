import torch
from torch import nn, optim
import pytorch_lightning as pl
from model import Net
from data_loader import CIFAR10DataModule

class LitCIFAR10(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.net = Net()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)  # Log the loss
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss)  # Log the loss during validation
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('test_loss', loss)  # Log the loss during testing
        return loss

    def configure_optimizers(self):
        return self.optimizer

if __name__ == '__main__':
    # Initialize Lightning model
    lit_model = LitCIFAR10()

    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=2,
        log_every_n_steps=50,  # Log every 50 steps
        progress_bar_refresh_rate=20,  # Refresh progress bar every 20 steps
        logger=pl.loggers.WandbLogger(),  # Use WandbLogger for logging
    )

    # Train the model
    trainer.fit(lit_model)
