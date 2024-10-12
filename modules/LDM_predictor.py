import torch
import torch.nn as nn

class LDM_predictor(nn.Module):
    def __init__(self):
        super(LDM_predictor, self).__init__()
        self.convldm = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1))
        self.fc1 = nn.Linear(in_features=6272, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=136)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.convldm(x)
        x = nn.ReLU(inplace=False)(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.ReLU(inplace=False)(x)
        x = self.fc2(x)
        return x.permute(0,2,1)
if __name__ == '__main__':
    x = torch.ones(32, 128, 16, 16)
    model = LDM_predictor()
    y = model(x)
    print(y.size())
