import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
	def __init__(self,inc):
		super(Block,self).__init__()
		self.layer1=nn.Sequential(nn.Conv2d(inc,inc//2,1),nn.BatchNorm2d(inc//2),nn.LeakyReLU(0.1))
		self.layer2=nn.Sequential(nn.Conv2d(inc//2,inc,3,padding=1),nn.BatchNorm2d(inc),nn.LeakyReLU(0.1))
	def forward(self,x):
		return x + self.layer2(self.layer1(x))


class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.conv1=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.LeakyReLU(0.1))
		self.conv2=nn.Sequential(nn.Conv2d(32,64,3,stride=2,padding=1),nn.BatchNorm2d(64),nn.LeakyReLU(0.1))
		self.res1=Block(64)
		self.conv3=nn.Sequential(nn.Conv2d(64,128,3,stride=2,padding=1),nn.BatchNorm2d(128),nn.LeakyReLU(0.1))
		self.res2=nn.Sequential(Block(128),Block(128))
		self.conv4=nn.Sequential(nn.Conv2d(128,256,3,stride=2,padding=1),nn.BatchNorm2d(256),nn.LeakyReLU(0.1))
		self.res3=nn.Sequential(Block(256),Block(256),Block(256),Block(256),Block(256),Block(256),Block(256),Block(256))
		self.conv5=nn.Sequential(nn.Conv2d(256,512,3,stride=2,padding=1),nn.BatchNorm2d(512),nn.LeakyReLU(0.1))
		self.res4=nn.Sequential(Block(512),Block(512),Block(512),Block(512),Block(512),Block(512),Block(512),Block(512))
				

		self.fc=nn.Sequential(nn.Linear(512*2*2,100),nn.LeakyReLU(0.1))
	def forward(self,x):

		x=self.conv1(x)
		x=self.conv2(x)
		x=self.res1(x)
		x=self.conv3(x)

		x=self.conv4(x)

		x=self.res3(x)

		x=self.conv5(x)

		x=self.res4(x)		
		

		x=x.view(-1,2048)
		return self.fc(x)

if __name__=="__main__":
	x=torch.rand(1,3,32,32)
	net=Net()
	print(net(x).size())


		

