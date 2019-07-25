import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms 

from Models.vgg_elu import Net


def train(device,lr,epochs,model_path,trainloader,pretrained=False):
	if pretrained:
		model=torch.load(pretrained)
	else :
		model=Net()
	model.to(device)
	criterion=nn.CrossEntropyLoss()
	print("learning rate:-"+str(lr))
	print("Epochs:-"+str(epochs))

	optimiser=optim.Adam(model.parameters(),lr=lr)
	for e in range(epochs):
			epoch_loss=0
			correct,total=0,0
			for i, data in enumerate(trainloader):
				inputs, labels = data

				output=model(inputs.to(device))
				loss=criterion(output,labels.to(device))
				optimiser.zero_grad()
				loss.backward()
				optimiser.step()
				epoch_loss += loss.item()
				_, predicted = torch.max(output.data, 1)
				total += labels.size(0)
				correct += (predicted == labels.to(device)).sum().item()
				print('Accuracy: %.4f %%' % (100 * correct / total))
			print("Epoch_Loss:- "+str(epoch_loss))
			print('Epoch: %e ' % (e+1))
			torch.save(model,model_path+"/model_"+str(e)+".path")

if __name__=="__main__":
	transform = transforms.Compose(
	    [transforms.ToTensor(),
	     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
	                                        download=False, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
	                                          shuffle=True, num_workers=2)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False,
	                                       download=False, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100,
	                                         shuffle=False, num_workers=2)
	device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	lr=1e-3
	epochs=200
	model_path="D:/Cifar10/src/Models/model_path"

	train(device,lr,epochs,model_path,trainloader)
