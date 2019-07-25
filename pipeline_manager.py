import torch
from test import test,test_folder
from train import train


class pipeline_manager:
	def __init__(self,cuda,config):
		self.config=config
		self.cuda=cuda
	def train(self,trainloader):
		device=torch.device("cuda" if self.cuda else "cpu")
		if "pretrained" not in self.config.keys():
			train(device,self.config["lr"],self.config["epochs"],self.config["model_path"],trainloader)
		else:
			train(device,self.config["lr"],self.config["epochs"],self.config["model_path"],trainloader,pretrained=self.config["pretrained"])

	def test(self,testloader):
		device=torch.device("cuda:0" if self.cuda else "cpu")
		test(device,self.config["test_path"],testloader)

	def test_folder(self,testloader):
		device=torch.device("cuda:0" if self.cuda else "cpu")
		test_folder(device,self.config["test_folder"],testloader)
