import lightning as L
import torch.utils.data
from lightning.pytorch.loggers import TensorBoardLogger
from torch import optim, nn
from torch.utils.data import DataLoader

from dataset_creation import create_datapoints, NUM_POINTS
from model import MyDataset


torch.set_default_dtype(torch.float64)


TRUNK_DEPTH = 3
TRUNK_WIDTH = 40
BRANCH_DEPTH = 2
BRANCH_WIDTH = 40


# learning rate
LR = 0.001


class DeepONet(L.LightningModule):

    def __init__(self,
                 input_size=NUM_POINTS,
                 output_size=1,
                 trunk_depth=TRUNK_DEPTH,
                    trunk_width=TRUNK_WIDTH,
                    branch_depth=BRANCH_DEPTH,
                    branch_width=BRANCH_WIDTH):
                
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        

        # Initialize trunk
        self.trunk = nn.Sequential()
        self.trunk.append(nn.Linear(1, trunk_width))
        for _ in range(trunk_depth - 1):
            self.trunk.append(nn.Linear(trunk_width, trunk_width))
            self.trunk.append(nn.ReLU())

        # Initialize branch
        self.branch = nn.Sequential()
        self.branch.append(nn.Linear(input_size, branch_width))
        for _ in range(branch_depth - 1):
            self.branch.append(nn.Linear(branch_width, branch_width))
            self.branch.append(nn.ReLU())

        #self.output_layer = nn.Linear(trunk_width + branch_width, output_size)
        # self.output_layer = nn.Linear(trunk_width, output_size)

        self.loss = nn.MSELoss()

        # Initialize weights
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        for layer in self.branch:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        #nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, u, x):
        
        branch_out = self.branch(u)
        trunk_out = self.trunk(x)

        #out = torch.cat((trunk_out, branch_out), dim=1)
        # out = trunk_out + branch_out

        #out = self.output_layer(out)
        out = torch.sum(branch_out * trunk_out)
        return out
    
    
    def training_step(self, batch):
        u, x, Fx = batch
        
        # u.shape torch.Size([8, 40])
        # x.shape torch.Size([8])
        # Fx.shape torch.Size([8])
    
        out = self.forward(u, x.unsqueeze(1)).squeeze()

        loss = self.loss(out, Fx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        u, x, Fx = batch
        
        out = self.forward(u, x.unsqueeze(1)).squeeze()

        loss = self.loss(out, Fx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch):
        u, x, Fx = batch
        
        out = self.forward(u, x.unsqueeze(1)).squeeze()

        loss = self.loss(out, Fx)
        self.log("test_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=LR)
        return optimizer


if __name__ == '__main__':
    train_set = MyDataset(create_datapoints(total_datapoints=1000,chebyshev_polymial=False))
    val_set = MyDataset(create_datapoints(total_datapoints=200,chebyshev_polymial=False))
    test_set = MyDataset(create_datapoints(total_datapoints=200,chebyshev_polymial=False))

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)
    test_loader = DataLoader(test_set, batch_size=1)
    
    # print(test_set[0])
    
    # exit(0)

    model = DeepONet()
    logger = TensorBoardLogger("tb_logs", name="my_model")
    trainer = L.Trainer(logger=logger, max_epochs=10000, check_val_every_n_epoch=1)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # test model
    trainer.test(ckpt_path="best", dataloaders=test_loader)