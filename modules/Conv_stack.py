import torch
import torch.nn as nn

class Conv_stack(nn.Module):
    def __init__(self):
        super(Conv_stack, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(4, 4), stride=(2, 2)),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1)),
        )
    
    def forward(self, x):
        x = self.conv_stack(x)
        return x 
    
if __name__ =="__main__":
    x = torch.ones(32, 1, 128, 128)
    model = Conv_stack()
    y = model(x)
    print(y.shape)