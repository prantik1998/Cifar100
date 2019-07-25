import torch
import torch.nn as nn

class Block(nn.Module):
	def __init__(self,in_ch,out_ch,stride):
		super(Block,self).__init__()
		self.lay1=nn.Sequential(nn.Conv2d(in_ch,in_ch,3,stride=stride,padding=1),nn.BatchNorm2d(in_ch),nn.ReLU())
		self.lay2=nn.Sequential(nn.Conv2d(in_ch,out_ch,3,stride=1,padding=1),nn.BatchNorm2d(out_ch),nn.ReLU())
		self.add=True if stride==1 else False
	def forward(self,x):
		if self.add:
			return x+ self.lay2(self.lay1(x))
		else:
			return self.lay2(self.lay1(x))


class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.layer1=nn.Sequential(nn.Conv2d(3,64,7,stride=2,padding=3),nn.BatchNorm2d(64),nn.ReLU())
		self.layer2=nn.Sequential(Block(64,64,1),Block(64,64,1),Block(64,64,1))
		self.layer3=nn.Sequential(Block(64,128,2),Block(128,128,1),Block(128,128,1),Block(128,128,1))
		self.layer4=nn.Sequential(Block(128,256,2),Block(256,256,1),Block(256,256,1),Block(256,256,1),Block(256,256,1),Block(256,256,1))
		self.layer5=nn.Sequential(Block(256,512,2),Block(512,512,1),Block(512,512,1),Block(512,512,1))
		self.pool=nn.AvgPool2d(2,2)
		self.fc=nn.Linear(512,100)
	def forward(self,x):
		x=self.layer1(x)
		x=self.layer2(x)
		x=self.layer3(x)
		x=self.layer4(x)
		x=self.layer5(x)
		x=self.pool(x)
		x=x.view(-1,512)
		return self.fc(x)

if __name__=="__main__":
	model=Net()
	x=torch.rand(1,3,32,32)
	print(model(x).size())




