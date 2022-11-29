import random
import os
import json
import numpy as np

from torch.utils.data import Dataset
from torchvision.io import read_image

from torchvision import transforms as T

class InferenceDataset(Dataset):
    def __init__(self, n=None, transform=None, dataset="val", seed=None, images=None, pre_transformed=False):
        if seed is not None:
            random.seed(seed)
            
        self.pre_transformed = pre_transformed
        if dataset == "val":
            if pre_transformed:
                self.img_dir = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/data/imagenet/val_224/"
            else:
                self.img_dir = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/data/imagenet/val/"
            self.truths = json.load(open("/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/validation_labels.json", "r"))
            if images is not None:
                self.images = images
            elif n is not None:
                self.images = random.choices(os.listdir(self.img_dir), k=n)
            else:
                self.images = os.listdir(self.img_dir)
        elif dataset == "train":
            raise ValueError
            self.img_dir = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/data/imagenet/train/"
        elif dataset == "test":
            truth_path = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/data/imagenet/test_ensemble_truth.npy"
            self.truths = np.load(truth_path, allow_pickle=True).item()
            if pre_transformed:
                self.img_dir = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/data/imagenet/test_224/"
            else:
                self.img_dir = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/data/imagenet/test/"
            if images is not None:
                self.images = images
            elif n is not None:
                self.images = random.choices(list(self.truths.keys()), k=n)
            else:
                self.images = list(self.truths.keys())
            self.truths = {k[:-5]: v for k, v in self.truths.items()}
        else:
            raise ValueError

        #self.images = os.listdir(self.img_dir)[:n]
        self.transform = T.Compose([T.Resize(256), T.CenterCrop(224)]) if transform is None else transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = read_image(os.path.join(self.img_dir, self.images[idx]))

        y = self.truths[self.images[idx][:-5]]
        if not self.pre_transformed:
            if img.shape[0] == 1:
                img = img.repeat_interleave(3, 0)
            if img.shape[0] > 3:
                img = img[:3]
            img = self.transform(img).float()
        else:
            img = img.float()

        return img / 255, y