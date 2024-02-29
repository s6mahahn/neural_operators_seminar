import pickle
from pathlib import Path

import lightning as L
import torch.utils.data
from lightning.pytorch.loggers import TensorBoardLogger
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader

from dataset_creation import create_datapoints, NUM_POINTS

torch.set_default_dtype(torch.float64)


# seed_everything(42, workers=True)


class IntegralModel(L.LightningModule):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(NUM_POINTS + 1, 2560)
        self.layer2 = nn.Linear(2560, 1)
        self.activation = nn.ReLU()  # TODO: pls discuss this
        self.loss = nn.MSELoss()  # TODO: discuss

        # initialize the weights of linear layers
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.activation(out)
        out = self.layer2(out)
        return out

    def training_step(self, batch):
        u, x, Fx = batch
        input_data = torch.cat((u, x.unsqueeze(0).T), dim=1)
        out = self.forward(input_data).squeeze(1)

        loss = self.loss(out, Fx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        u, x, Fx = batch
        x = torch.cat((u, x.unsqueeze(0).T), dim=1)
        out = self.forward(x).squeeze()
        loss = self.loss(out, Fx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch):
        u, x, Fx = batch
        x = torch.cat((u, x.unsqueeze(0).T), dim=1)
        out = self.forward(x).squeeze()
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


class MyDataset(Dataset):
    def __init__(self, datapoints):
        self.u = []
        self.x = []
        self.Gx = []
        for s in datapoints:
            u, x, Gx = s
            self.u.append(u)
            self.x.append(x)
            self.Gx.append(Gx)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return (torch.tensor(self.u[index]),
                torch.tensor(self.x[index]),
                torch.tensor(self.Gx[index]))

if __name__ == "__main__":
    train_set = MyDataset(create_datapoints())
    val_set = MyDataset(create_datapoints())
    test_set = MyDataset(create_datapoints())

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)
    test_loader = DataLoader(test_set, batch_size=1)

    logger = TensorBoardLogger("tb_logs", name="my_model")
    trainer = L.Trainer(logger=logger, max_epochs=10000)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                )

    # test model
    trainer.test(ckpt_path="best", dataloaders=test_loader)
