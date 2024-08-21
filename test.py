import torch
import spconv.pytorch as spconv
from torch import nn
    
tensor1 = torch.randint(0, 3, (1, 16, 120, 320, 320))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ExampleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = spconv.SparseSequential(
            nn.BatchNorm1d(1),
            spconv.SubMConv2d(1, 32, 3, 1),
            nn.ReLU(),
            spconv.SubMConv2d(32, 64, 3, 1),
            nn.ReLU(),
            # spconv.SparseMaxPool2d(2, 2),
            # spconv.ToDense(),
        )
        self.dense = spconv.ToDense()

    def forward(self, x):
        x = spconv.SparseConvTensor.from_dense(x)
        print(x.dense())
        x = self.net(x)
        x = self.dense(x)
        # print("x", x)
        return x
    
if __name__ == "__main__":
    x = torch.randint(0, 256, (1, 28, 28, 1), dtype=torch.float32).to(device)
    out = ExampleNet().to(device).forward(x)