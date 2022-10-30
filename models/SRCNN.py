import torch.nn as nn
import torch.nn.functional as F
import torch

class ESPCN(nn.Module):
    def __init__(self, upscale_factor):
        super(ESPCN, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x
    
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        
        self.act = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(1, 64, (9, 9), (1, 1), (4, 4))
        self.conv2 = nn.Conv2d(64, 32, (1, 1), (1, 1), (0, 0))
        self.conv3 = nn.Conv2d(32, 1, (5, 5), (1, 1), (2, 2))
        

    def forward(self, x):
        x = F.interpolate(x, scale_factor= 2, mode='bicubic', align_corners=False)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x
    
if __name__ == '__main__':
    z = torch.randn((3, 1, 64, 64))
    model = SRCNN()
    z = model(z)
    print(z.shape)
    print(1)

    





