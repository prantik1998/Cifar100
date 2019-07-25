import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(3,64,3,padding=1),nn.BatchNorm2d(64),nn.LeakyReLU(0.1),nn.Dropout(0.5))
        self.conv1_2 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.LeakyReLU(0.1),nn.Dropout(0.5))
        

        self.conv2_1 = nn.Sequential(nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.LeakyReLU(0.1),nn.Dropout(0.5))
        self.conv2_2 = nn.Sequential(nn.Conv2d(128,128,3,padding=1),nn.BatchNorm2d(128),nn.LeakyReLU(0.1),nn.Dropout(0.5))
        

        self.conv3_1 = nn.Sequential(nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.LeakyReLU(0.1),nn.Dropout(0.5))
        self.conv3_2 = nn.Sequential(nn.Conv2d(256,256,3,padding=1),nn.BatchNorm2d(256),nn.LeakyReLU(0.1),nn.Dropout(0.5))
        self.conv3_3 = nn.Sequential(nn.Conv2d(256,256,3,padding=1),nn.BatchNorm2d(256),nn.LeakyReLU(0.1),nn.Dropout(0.5))

        self.conv4_1 = nn.Sequential(nn.Conv2d(256,512,3,padding=1),nn.BatchNorm2d(512),nn.LeakyReLU(0.1),nn.Dropout(0.5))
        self.conv4_2 = nn.Sequential(nn.Conv2d(512,512,3,padding=1),nn.BatchNorm2d(512),nn.LeakyReLU(0.1),nn.Dropout(0.5))
        self.conv4_3 = nn.Sequential(nn.Conv2d(512,512,3,padding=1),nn.BatchNorm2d(512),nn.LeakyReLU(0.1),nn.Dropout(0.5))


        self.conv5_1 = nn.Sequential(nn.Conv2d(512,512,3,padding=1),nn.BatchNorm2d(512),nn.LeakyReLU(0.1),nn.Dropout(0.5))
        self.conv5_2 = nn.Sequential(nn.Conv2d(512,512,3,padding=1),nn.BatchNorm2d(512),nn.LeakyReLU(0.1),nn.Dropout(0.5))
        self.conv5_3 = nn.Sequential(nn.Conv2d(512,512,3,padding=1),nn.BatchNorm2d(512),nn.LeakyReLU(0.1),nn.Dropout(0.5))

        

        self.pool = nn.MaxPool2d(2, 2)



        self.fc = nn.Linear(512* 1 * 1, 100)

        self.act=nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.pool(self.conv1_2(self.conv1_1(x)))
        x = self.pool(self.conv2_2(self.conv2_1(x)))
        x = self.pool(self.conv3_3(self.conv3_2(self.conv3_1(x))))
        x = self.pool(self.conv4_3(self.conv4_2(self.conv4_1(x))))
        x = self.pool(self.conv5_3(self.conv5_2(self.conv5_1(x))))

        x = x.view(-1, 512 * 1 * 1)
        
        x = self.act(self.fc(x))
        return x

if __name__=="__main__":
    net=Net()
    x=torch.rand(1,3,32,32)
    out=net(x)
    print(out.size())