import matplotlib.pyplot as plt 
import numpy as np
import click
import yaml

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from pipeline_manager import pipeline_manager

@click.group()
def main():
	pass

@main.command()
def train():
	manager.train(trainloader)

@main.command()
def test():
	manager.test(testloader)

@main.command()
def test_folder():
	manager.test_folder(testloader)



if __name__=="__main__":
	config=yaml.safe_load(open("config/config.yaml","r"))
	bs=config["batch_size"]
	print("batch_size:-"+str(bs))
	transform = transforms.Compose(
	    [transforms.ToTensor(),
	     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
	                                        download=False, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,
	                                          shuffle=True, num_workers=2)

	testset = torchvision.datasets.CIFAR100(root='./data', train=False,
	                                       download=False, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=bs,
	                                         shuffle=False, num_workers=2)	

	manager=pipeline_manager(torch.cuda.is_available(),config)
	main()
