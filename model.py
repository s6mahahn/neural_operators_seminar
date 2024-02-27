import pickle
from pathlib import Path

import lightning as L
import torch.utils.data
from lightning import seed_everything

from lightning.pytorch.loggers import TensorBoardLogger
from torch import optim, nn

from dataset_creation import create_datapoints, NUM_TRAIN_SAMPLES, NUM_VAL_SAMPLES, NUM_TEST_SAMPLES, NUM_POINTS

torch.set_default_dtype(torch.float64)
# seed_everything(42, workers=True)



class IntegralModel(L.LightningModule):

    def __init__(self):
        super().__init__()
        
        
        # self.layer1 = nn.Linear(40, 1280)
        # self.layer12 = nn.Linear(1, 1280)
        
        self.layer1 = nn.Linear(NUM_POINTS + 1, 2560)
        self.layer2 = nn.Linear(2560, 1)
        self.activation = nn.Sigmoid()  # TODO: pls discuss this
        self.ll_activation = nn.Sigmoid()  # TODO: pls discuss this
        self.loss = nn.MSELoss()  # TODO: discuss
        
        # initialize the weights of linear layers
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)


    def forward(self, x):
        out = self.layer1(x)
        out = self.activation(out)
        
        out = self.layer2(out)
        out = self.ll_activation(out)
        return out


    def training_step(self, batch):
        u, x, Fx = batch
        input_data = torch.cat((u * 0, x.unsqueeze(0).T), dim=1)
        out = self.forward(input_data).squeeze(1)

        loss = self.loss(out, Fx)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss * 100

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


# train_set = load_dataset(Path("datasets/first dataset_240227_1948_train.pickle"))
# val_set = load_dataset(Path("datasets/first dataset_240227_1948_val.pickle"))
# test_set = load_dataset(Path("datasets/first dataset_240227_1948_test.pickle"))


train_set = create_datapoints(NUM_TRAIN_SAMPLES)
val_set = create_datapoints(NUM_VAL_SAMPLES)
test_set = create_datapoints(NUM_TEST_SAMPLES)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=8)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=8)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)


logger = TensorBoardLogger("tb_logs", name="my_model")
trainer = L.Trainer(logger=logger, max_epochs=100)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader,
            )

# test model
trainer.test(ckpt_path="best", dataloaders=test_loader)
