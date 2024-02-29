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
torch.manual_seed(42)

# depth
NUM_LAYERS = 2
# width
HIDDEN_SIZE = 10
# learing rate
LR = 0.01

class IntegralModel(L.LightningModule):

    def __init__(self, input_size = NUM_POINTS + 1, hidden_size = HIDDEN_SIZE, output_size = 1, num_layers = NUM_LAYERS):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Initialize layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, output_size)

        self.activation = nn.ReLU()
        self.loss = nn.MSELoss()

        # Initialize weights
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        for layer in self.layers:
            # linear layer
            linear_output = layer(x)
            # activation function
            out = self.activation(linear_output)
        out = self.output_layer(out)
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
        out = self.forward(x)
        loss = self.loss(out, Fx)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=LR)
        return optimizer

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



train_set = MyDataset(create_datapoints(1000))
val_set = MyDataset(create_datapoints(200))
test_set = MyDataset(create_datapoints(200))

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8)
test_loader = DataLoader(test_set, batch_size=1)

model = IntegralModel()
logger = TensorBoardLogger("tb_logs", name="my_model")
trainer = L.Trainer(logger=logger, max_epochs=10000)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


# test model
trainer.test(ckpt_path="best", dataloaders=test_loader)
