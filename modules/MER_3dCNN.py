import torch
import torch.nn as nn

class MER_3dCNN(nn.Module):
    def __init__(self,cls_num = 3):
        super(MER_3dCNN,self).__init__()
        self.cls = cls_num
        self.conv3d = nn.Conv3d(in_channels=128, out_channels=32, kernel_size=(2, 2, 5), stride=(2, 2, 1))
        self.fc1 = nn.Linear(in_features=2560, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=self.cls)
        self.maxpooling3d = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.dropout = nn.Dropout()
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = self.conv3d(x)
        x = self.maxpooling3d(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.ReLU(inplace=False)(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    mer_model = MER_3dCNN()
    x = torch.ones(32,128,36,36,8)
    y = mer_model(x)
    print(y.shape)