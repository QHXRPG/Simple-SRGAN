import torch
import torch.nn as nn
from torchvision.models import vgg16
import matplotlib.pyplot as plt
import torch.utils.data as Data
from data_utils import Traindata

vgg = vgg16().features[:29]
x = torch.rand(1,3,224,224)

"""生成器"""
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,1,1)
        self.conv2 = nn.Conv2d(64,64,3,1,1)
        self.conv3 = nn.Conv2d(64,256,3,1,1)
        self.conv4 = nn.Conv2d(64,3,3,1,1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_features=64)
        self.block = residual_blocks()
        self.up_samlpe = nn.PixelShuffle(2)
    def forward(self,x):
        x1 = self.relu(self.conv1(x))
        x2=x1
        for i in range(5):
            x2 = x2 + self.block(x2)
        x3 = x1 + self.bn(self.conv2(x2))
        x4 = self.relu(self.up_samlpe(self.conv3(self.relu(self.up_samlpe(self.conv3(x3))))))
        return self.conv4(x4)

"""残差块"""
class residual_blocks(nn.Module):
    def __init__(self):
        super(residual_blocks, self).__init__()
        self.l = nn.Sequential(nn.Conv2d(64,64,3,1,1),
                               nn.BatchNorm2d(num_features=64),
                               nn.ReLU(),
                               nn.Conv2d(64,64,3,1,1),
                               nn.BatchNorm2d(num_features=64))
    def forward(self,x):
        return self.l(x)

"""辨别器"""
class Discriminator(nn.Module):
    def __init__(self,output=64):
        super(Discriminator, self).__init__()
        self.cbl = []
        for i in range(7):
            if i % 2==0:
                self.cbl.append(nn.Conv2d(int(output*(2**(i/2))),int(output*(2**(i/2))),3,2,1))
                self.cbl.append(nn.BatchNorm2d(int(output*(2**(i/2)))))
                self.cbl.append(nn.LeakyReLU(0.2))
            else:
                self.cbl.append(nn.Conv2d(int(output*(2**(((i+1)/2)-1))),int(output*(2**((i+1)/2))),3,2,1))
                self.cbl.append(nn.BatchNorm2d(int(output*(2**((i+1)/2)))))
                self.cbl.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            *self.cbl,
            nn.AdaptiveAvgPool2d(1),#torch.Size([batch_size, y, h, w])----->torch.Size([batch_size, y, 1, 1])
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)#torch.Size([batch_size, 1, 1, 1])
        )
    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))

generator = Generator()
discriminator = Discriminator()

"""内容损失函数"""
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.features_get = vgg
        self.loss = nn.MSELoss()
    def forward(self,lr_img,target_img):
        fake_img = generator(lr_img)
        fake = self.features_get(fake_img)
        target = self.features_get(target_img)
        _, _, W, H =fake.shape
        Loss = self.loss(fake,target)/(W*H)
        return Loss

"""对抗损失函数"""
class GenLoss(nn.Module):
    def __init__(self):
        super(GenLoss, self).__init__()
        self.Loss = 0
    def forward(self,lr_img):
        batch,_,_,_ = lr_img.shape
        fake_img = generator(lr_img)
        for i in range(batch):
            self.Loss = -torch.log(discriminator(fake_img)) + self.Loss
        return self.Loss

"""数据预处理"""
data = Traindata(dataset_dir="/Users/qiuhaoxuan/Downloads/VOC2012/train",crop_size=100,upscale_factor=4)
data_loader =  Data.DataLoader(dataset=data,batch_size=64,num_workers=0)

"""损失函数与优化器实例化"""
vggloss = VGGLoss()
genloss = GenLoss()
opt1 = torch.optim.Adam(generator.parameters(),lr=0.005)
opt2 = torch.optim.Adam(discriminator.parameters(),lr=0.005)

"""训练"""
def train():
    for i in range(100):
        for i,(lr_img,target_img) in enumerate(data_loader):
            Loss = vggloss(lr_img,target_img) + 0.0001*genloss(lr_img)
            opt1.zero_grad()
            opt1.zero_grad()
            Loss.backward()
            opt1.step()
            opt1.step()