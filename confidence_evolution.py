import os
import random
from boundary_projection import BP
import foolbox as fb
import torch
import eagerpy as ep
import numpy as np

from torch.utils.data import DataLoader
from dataset import InferenceDataset
from load import get_model
from tqdm import tqdm

    
n_images = 50 # Nombre d'images
batch_size = 8
seed = 20
n_iterations = 20

model_name = "efficientnet_b0"
model_name = "resnet50"
# model_name = "levit_384"
output_dir = "/udd/tmaho/hlrf/{}-confidence_evolution".format(model_name)
os.makedirs(output_dir, exist_ok=True)
img_dir = "/udd/tmaho/transferable/correctly_classified/dataset_val-51_models.npy"
images = np.load(img_dir)
random.seed(seed)
images = random.choices(images, k=n_images)

# direction for the dataset


# load the modek=l
model = get_model(model_name).eval()
bounds = (0, 1)
fmodel = fb.PyTorchModel(model, bounds=bounds)

# load the dataset
data = InferenceDataset(n=n_images, transform=None, dataset="val", images=images, pre_transformed=True)
testloader = DataLoader(data, batch_size=batch_size, num_workers=2)



attack_list = []
for p in [0.1, 0.5, 1, 2, 4, 10]:
    attack_list.append((
        fb.attacks.iHL_RFAttack(
            steps=n_iterations, 
            abort_early=False, 
            tau=0.05, 
            confidence=0.01, 
            c_sigma_gain=1, 
            c_sigma_start=1.00001,
            omega=0.0001,
            d1_d2_ratio=p
            ),
        "confidence=={}".format(p)))

def total(attack_list):    
    
    size_im = (3, 224, 224)
    p = []
    for images, labels in testloader:
        p.append(fmodel(images.cuda(0)).argmax(1).cpu() == labels)
    acc = float(torch.cat(p).sum().cpu())
    print('Accuracy: {} '.format(acc / n_images))

    def att(attack, name):
        dist_adv = []
        queries = []
        for images, labels in tqdm(testloader):
            with torch.no_grad():
                images ,labels = images.cuda(),labels.cuda()
            if "bp" in name.lower():
                adv, _, _ = attack(fmodel, images * 255, labels, epsilons=4/255)
                adv /= 255
            else:
                adv, _, _ = attack(fmodel, images, labels, epsilons=None)

            if hasattr(attack, "queries"):
                queries += list(attack.queries)
            with torch.no_grad():
                for i in range(len(images)):
                    if fmodel(ep.reshape(images[i],(1,3,224,224))).argmax(axis=-1)==labels[i] and fmodel(ep.reshape(adv[i],(1,3,224,224))).argmax(axis=-1)!=labels[i]:
                        dist_adv.append((fb.distances.l2(ep.reshape(adv[i], (1,3,224,224))*255,ep.reshape(images[i], (1,3,224,224))*255)/np.sqrt(size_im[0]*size_im[1]*size_im[2])).item())
                    elif fmodel(ep.reshape(images[i],(1,3,224,224))).argmax(axis=-1)==labels[i]:
                        dist_adv.append(np.inf)
                    else:
                        dist_adv.append(0)
            del images,labels,adv
            torch.cuda.empty_cache()

        dist_adv = np.array(dist_adv)
        asr = (dist_adv < 100000).sum() / n_images
        mean_adv = np.mean([e for e in dist_adv if e < 10000])
        print("\t-ASR: {}".format(asr))
        print("\t-Mean distortion: {}".format(mean_adv))
        print("\t-Mean queries: {}".format(np.mean(queries)))
        return dist_adv, queries

    
    for attack, name in attack_list:
        # if os.path.exists(os.path.join(output_dir, "{}.npy".format(name))):
        #     continue

        print("Attack:  {}".format(name))
        dist_adv, queries = att(attack, name)
        np.save(os.path.join(output_dir, "{}.npy".format(name)), dist_adv)
        np.save(os.path.join(output_dir, "{}-queries.npy".format(name)), queries)
        torch.cuda.empty_cache()
        del dist_adv
        print("\n\n\n")
    

total(attack_list)
