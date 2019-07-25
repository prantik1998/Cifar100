import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


def test(device,test_path,testloader):
	total=0
	i=0
	model = torch.load(test_path)
	max_acc=0
	min_acc=1
	with torch.no_grad():
		for batch_num,data in enumerate(testloader):
			inputs, target = data
			inputs = inputs.to(device)
			target = target.to(device)
			output = model(inputs)	
			prediction = torch.max(output, 1)

			test_correct = np.mean(prediction[1].cpu().numpy() == target.cpu().numpy())
			total +=test_correct
			max_acc=max(max_acc,test_correct)
			min_acc=min(min_acc,test_correct)
			i+=1
	print("Min Accuracy:-"+str(min_acc))
	print("Max Accuracy:-"+str(max_acc))	
	print("Mean Accuracy:-"+str(total/i))

def test_folder(device,test_path,testloader):
	max_acc_model=0
	model_name=""
	total=0
	i=0
	paths=[os.path.join(test_path,i) for i in os.listdir(test_path)]
	for path in paths:
		model = torch.load(path)
		max_acc=0
		min_acc=1
		with torch.no_grad():
			for batch_num,data in enumerate(testloader):
				inputs, target = data
				inputs = inputs.to(device)
				target = target.to(device)
				output = model(inputs)	
				prediction = torch.max(output, 1)

				test_correct = np.mean(prediction[1].cpu().numpy() == target.cpu().numpy())
				total +=test_correct
				max_acc=max(max_acc,test_correct)
				min_acc=min(min_acc,test_correct)
				i+=1
		print(path)
		print("Min Accuracy:-"+str(min_acc))
		print("Max Accuracy:-"+str(max_acc))	
		print("Mean Accuracy:-"+str(total/i))
		if max_acc_model<total/i:
			max_acc_model=total/i
			model_name=path
	print("Best Model:-"+model_name)
	print("Best Model Accuracy:-"+str(max_acc_model))


