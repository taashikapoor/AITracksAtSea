import torch
import torchvision
import torch.nn as nn

class LocDataNet(nn.Module):
    def __init__(self, input_features, out):
        super(LocDataNet, self).__init__()

        self.linear1 = nn.Linear(input_features, input_features*2)
        self.linear2 = nn.Linear(input_features*2, input_features*2)
        self.linear3 = nn.Linear(input_features*2, 4)
        self.linear4 = nn.Linear(4, out)

        self.bn1 = nn.BatchNorm1d(input_features*2)
        self.bn2 = nn.BatchNorm1d(input_features*2)
        self.bn3 = nn.BatchNorm1d(4)

        self.network = nn.Sequential(
            self.linear1,
            nn.LeakyReLU(),
            self.bn1,
            #nn.Dropout(),

            self.linear2,
            nn.LeakyReLU(),
            self.bn2,
            #nn.Dropout(),

            self.linear3,
            nn.LeakyReLU(),
            self.bn3,
            #nn.Dropout(),

            self.linear4
        )
    
    def forward(self, x):

        x = self.network(x)

        return x