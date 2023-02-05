from create_dataset import AdversarialDataset
import torch
from foolbox import PyTorchModel, attacks
import torchvision
from tqdm import tqdm
from foolbox.distances import l2
import matplotlib.pyplot as plt


img_dir = '/run/media/paul/EMTEC C410/images 1k'
csv_dir = '/run/media/paul/EMTEC C410/labels.csv'
nb_images = 10

#load the model (efficientnet b0, resnet18, resnet50)
model = torchvision.models.resnet18(pretrained=True)
model = model.eval()
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
bounds = (0, 1)
fmodel_resnet_18 = PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)

model = torchvision.models.resnet50(pretrained=True)
model = model.eval()
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
bounds = (0, 1)
fmodel_resnet_50 = PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)

model = torchvision.models.efficientnet_b0(pretrained=True)
model = model.eval()
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
bounds = (0, 1)
fmodel_efficientnet_b0 = PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)

#load the dataset
data = AdversarialDataset(img_dir,csv_dir)

# create a subset of the dataset where all the images are well classified
data = torch.utils.data.Subset(data,range(nb_images))
testloader = torch.utils.data.DataLoader(data,batch_size=1,num_workers=2)

#attack
print('Attack \n')
attack_bp_armijo = attacks.BPArmijo()
attack_bp = attacks.BP()

#list of results
result_bp_armijo_resnet_18 = []
is_adv_bp_armijo_resnet_18 = []
result_bp_armijo_resnet_50 = []
is_adv_bp_armijo_resnet_50 = []
result_bp_armijo_efficientnet_b0 = []
is_adv_bp_armijo_efficientnet_b0 = []

result_bp_resnet_18 = []
is_adv_bp_resnet_18 = []
result_bp_resnet_50 = []
is_adv_bp_resnet_50 = []
result_bp_efficientnet_b0 = []
is_adv_bp_efficientnet_b0 = []

is_adv_resnet_18 = []
is_adv_resnet_50 = []
is_adv_efficientnet_b0 = []

#test the attack on the model
for i, (images, labels) in tqdm(enumerate(testloader)):

	images = images.to('cuda')
	labels = labels.to('cuda')

	img = images[0].cpu()

	#print the image
	img = img.permute(1, 2, 0)
	plt.imshow(img)
	plt.show()


	# Starting classification
	is_adv_resnet_18.append(fmodel_resnet_18(images).argmax().item() != labels.item())
	is_adv_resnet_50.append(fmodel_resnet_50(images).argmax().item() != labels.item())
	is_adv_efficientnet_b0.append(fmodel_efficientnet_b0(images).argmax().item() != labels.item())

	# BP_armijo
	adversarial,_,_ = attack_bp_armijo(fmodel_resnet_18, images, labels, epsilons=0.3)
	result_bp_armijo_resnet_18.append(l2(adversarial,images))
	is_adv_bp_armijo_resnet_18.append(fmodel_resnet_18(adversarial).argmax().item() != labels.item())
	adversarial,_,_ = attack_bp_armijo(fmodel_resnet_50, images, labels, epsilons=0.3)
	result_bp_armijo_resnet_50.append(l2(adversarial,images))
	is_adv_bp_armijo_resnet_50.append(fmodel_resnet_50(adversarial).argmax().item() != labels.item())
	adversarial,_,_ = attack_bp_armijo(fmodel_efficientnet_b0, images, labels, epsilons=0.3)
	result_bp_armijo_efficientnet_b0.append(l2(adversarial,images))
	is_adv_bp_armijo_efficientnet_b0.append(fmodel_efficientnet_b0(adversarial).argmax().item() != labels.item())

	# BP
	'''adversarial,_,_ = attack_bp(fmodel_resnet_18, images, labels, epsilons=0.3)
	result_bp_resnet_18.append(l2(adversarial,images))
	is_adv_bp_resnet_18.append(fmodel_resnet_18(adversarial).argmax().item() != labels.item())
	adversarial,_,_ = attack_bp(fmodel_resnet_50, images, labels, epsilons=0.3)
	result_bp_resnet_50.append(l2(adversarial,images))
	is_adv_bp_resnet_50.append(fmodel_resnet_50(adversarial).argmax().item() != labels.item())
	adversarial,_,_ = attack_bp(fmodel_efficientnet_b0, images, labels, epsilons=0.3)
	result_bp_efficientnet_b0.append(l2(adversarial,images))
	is_adv_bp_efficientnet_b0.append(fmodel_efficientnet_b0(adversarial).argmax().item() != labels.item())'''

	if i == nb_images:
		break
	
#save the results on a csv file and print the results
import pandas as pd
import numpy as np
import csv

with open('results.csv', 'w', newline='') as csvfile:
	fieldNames = ["model", "is_adv", "result"]
	writer = csv.DictWriter(csvfile, fieldnames=fieldNames)
	writer.writeheader()

	data = [{"model": "resnet_18", "is_adv": str(is_adv_resnet_18), "result": str(result_bp_armijo_resnet_18)},
			{"model": "resnet_50", "is_adv": str(is_adv_resnet_50), "result": str(result_bp_armijo_resnet_50)},
			{"model": "efficientnet_b0", "is_adv": str(is_adv_efficientnet_b0), "result": str(result_bp_armijo_efficientnet_b0)},
			{"model": "BP_armijo_resnet_18", "is_adv": str(is_adv_bp_armijo_resnet_18), "result": str(result_bp_armijo_resnet_18)},
			{"model": "BP_armijo_resnet_50", "is_adv": str(is_adv_bp_armijo_resnet_50), "result": str(result_bp_armijo_resnet_50)},
			{"model": "BP_armijo_efficientnet_b0", "is_adv": str(is_adv_bp_armijo_efficientnet_b0), "result": str(result_bp_armijo_efficientnet_b0)},
			{"model": "BP_resnet_18", "is_adv": str(is_adv_bp_resnet_18), "result": str(result_bp_resnet_18)},
			{"model": "BP_resnet_50", "is_adv": str(is_adv_bp_resnet_50), "result": str(result_bp_resnet_50)},
			{"model": "BP_efficientnet_b0", "is_adv": str(is_adv_bp_efficientnet_b0), "result": str(result_bp_efficientnet_b0)}]
	for row in data:
		writer.writerow(row)

print(is_adv_resnet_18)
print(result_bp_armijo_resnet_18)
print(is_adv_bp_resnet_18)
print('resnet_18 : ', np.mean(is_adv_resnet_18) * 100, '%\n')
print('resnet_50 : ', np.mean(is_adv_resnet_50) * 100, '%\n')
print('efficientnet_b0 : ', np.mean(is_adv_efficientnet_b0) * 100, '%\n')
print('BP_armijo_resnet_18 : ', np.mean(result_bp_armijo_resnet_18), ' - ', np.mean(is_adv_bp_armijo_resnet_18) * 100, '%\n')
print('BP_armijo_resnet_50 : ', np.mean(result_bp_armijo_resnet_50), ' - ', np.mean(is_adv_bp_armijo_resnet_50) * 100, '%\n')
print('BP_armijo_efficientnet_b0 : ', np.mean(result_bp_armijo_efficientnet_b0), ' - ', np.mean(is_adv_bp_armijo_efficientnet_b0) * 100, '%\n')
'''print('BP_resnet_18 : ', np.mean(result_bp_resnet_18), ' - ', np.mean(is_adv_bp_resnet_18) * 100, '%\n')
print('BP_resnet_50 : ', np.mean(result_bp_resnet_50), ' - ', np.mean(is_adv_bp_resnet_50)	* 100, '%\n')
print('BP_efficientnet_b0 : ', np.mean(result_bp_efficientnet_b0), ' - ', np.mean(is_adv_bp_efficientnet_b0) * 100, '%\n')'''
