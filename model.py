import pickle
from pathlib import Path

import lightning as L
import torch.utils.data
from lightning import seed_everything
from torch import optim, nn

torch.set_default_dtype(torch.float64)
seed_everything(42, workers=True)



class IntegralModel(L.LightningModule):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(41, 2560)
        self.layer2 = nn.Linear(2560, 1)
        self.activation = nn.ReLU()  # TODO: pls discuss this
        self.ll_activation = nn.ReLU()  # TODO: pls discuss this
        self.loss = nn.MSELoss()  # TODO: discuss

    def forward(self, x):
        out = self.layer1(x)
        out = self.activation(out)
        out = self.layer2(out)
        out = self.ll_activation(out)
        return out

    def training_step(self, batch):
        u, x, Fx = batch
        x = torch.cat((u, x.unsqueeze(0).T), dim=1)
        out = self.forward(x)
        loss = self.loss(out, Fx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        u, x, Fx = batch
        x = torch.cat((u, x.unsqueeze(0).T), dim=1)
        out = self.forward(x)
        loss = self.loss(out, Fx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch):
        u, x, Fx = batch
        x = torch.cat((u, x.unsqueeze(0).T), dim=1)
        out = self.forward(x)
        loss = self.loss(out, Fx)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


model = IntegralModel()


def load_dataset(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


train_set = load_dataset(Path("datasets/first dataset_240223_2202_train.pickle"))
val_set = load_dataset(Path("datasets/first dataset_240223_2202_val.pickle"))
test_set = load_dataset(Path("datasets/first dataset_240223_2202_test.pickle"))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=8)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=8)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)

trainer = L.Trainer(deterministic=True)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# test model
trainer.test(ckpt_path="best", dataloaders=test_loader)
