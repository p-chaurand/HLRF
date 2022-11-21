from cgi import test
import os
import random
from shutil import which
import foolbox as fb
import torch
import eagerpy as ep
import numpy as np
import pandas as pd
import fnmatch

from torch.utils.data import DataLoader
from dataset import InferenceDataset
from load import get_model
from tqdm import tqdm
import matplotlib.pyplot as plt


n_images = 20 # Nombre d'images
batch_size = 32
seed = 10
model_name = "resnet50"
# model_name = "efficientnet_b0"
# model_name = "levit_384"
output_dir = "/home/paul/Documents/monitoring-{}".format(model_name)
os.makedirs(output_dir, exist_ok=True)
img_dir = '/run/media/paul/EMTEC C410/images 1k'
csv_dir = '/run/media/paul/EMTEC C410/labels.csv'
img_names = fnmatch.filter(os.listdir(img_dir), '*.JPEG')
images = []
labels = []
for i in range(n_images):
    image = np.array(plt.imread(os.path.join(img_dir, img_names[i])))
    images.append(image)
    labels.append(int(pd.read_csv(csv_dir,sep=',').iloc[i]['Label']))
images = np.array(images)
labels = np.array(labels)
data = InferenceDataset(img_dir,csv_dir)
testloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=4)
"""images = np.load(img_dir)
random.seed(seed)
images = random.choices(images, k=n_images)"""

# direction for the dataset


# load the modek=l
model = get_model(model_name).eval()
bounds = (0, 1)
fmodel = fb.PyTorchModel(model, bounds=bounds)

# load the dataset
"""data = InferenceDataset(n=n_images, transform=None, dataset="val", images=images, pre_transformed=True)
testloader = DataLoader(data, batch_size=batch_size, num_workers=2)
"""


attack = fb.attacks.iHL_RFAttack(
    steps=50, abort_early=False, 
    tau=0.5, 
    confidence=0.01, 
    c_sigma_gain=1, 
    c_sigma_start=1.00001,
    omega=0.1,
    smooth=1,
    d1_d2_ratio=2
    )


keys = [
    "armijo_respect",
    "history_norms",
    "armijo_step",
    "is_advs_history",
    "perturbation_added_dist",
    "query_history",
    "sigm_history",
    "perturbation_grad_ratio_history",
    "norm_grad_history",
    "descent_dir_norm"
]

def monitor_fails():
    failed_i = 0
    for images, labels in tqdm(testloader):
        with torch.no_grad():
            images ,labels = images.cuda(),labels.cuda()
        adv, _, _ = attack(fmodel, images, labels, epsilons=None)

        preds = fmodel(images).argmax(axis=1)
        preds_adv = fmodel(adv).argmax(axis=1)
        with torch.no_grad():
            for i in range(len(images)):
                if preds[i] == preds_adv[i]:
                    failed_i += 1
                    os.makedirs(os.path.join(output_dir, str(failed_i)), exist_ok=True)
                    
                    for k in keys:
                        plt.plot(getattr(attack, k)[i])
                        plt.savefig(os.path.join(output_dir, str(failed_i), "{}.pdf".format(k)))
                        plt.close()

                    plt.plot(attack.loss_gt_history[i])
                    plt.plot(attack.loss_p_history[i])
                    plt.savefig(os.path.join(output_dir, str(failed_i), "loss.pdf"))
                    plt.close()



def monitor_all():
    for images, labels in tqdm(testloader):
        with torch.no_grad():
            images ,labels = images.cuda(),labels.cuda()
        adv, _, _ = attack(fmodel, images, labels, epsilons=4/255)

        preds = fmodel(images).argmax(axis=1)
        preds_adv = fmodel(adv).argmax(axis=1)
        with torch.no_grad():
            for i in range(len(images)):
                if preds[i] == preds_adv[i]:
                    name = "failed" + str(i)
                else:
                    name = "success" + str(i)

                os.makedirs(os.path.join(output_dir, str(name)), exist_ok=True)
                
                for k in keys:
                    plt.plot(getattr(attack, k)[i])
                    plt.savefig(os.path.join(output_dir, str(name), "{}.pdf".format(k)))
                    plt.close()

                plt.plot(attack.loss_gt_history[i])
                plt.plot(attack.loss_p_history[i])
                plt.savefig(os.path.join(output_dir, str(name), "loss.pdf"))
                plt.close()


                plt.plot(attack.gradient_norm[i])
                plt.plot(attack.true_gradient_norm[i])
                plt.savefig(os.path.join(output_dir, str(name), "gradient_norm.pdf"))
                plt.close()

                plt.plot(attack.d1_norm[i], label="d1 norm")
                plt.plot(attack.d2_norm[i], label="d2 norm")
                m = max(max(attack.d1_norm[i]), max(attack.d2_norm[i]))
                m = min(m, 50)
                plt.ylim((-0.5, m))
                plt.legend()
                plt.savefig(os.path.join(output_dir, str(name), "direction_split_norm.pdf"))
                plt.close()

    
monitor_all()






