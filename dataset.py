import random
import os
import json
import numpy as np
import pandas as pd
import fnmatch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image

from torchvision import transforms

class InferenceDataset(Dataset):
    """FaceDataset: representation of the CPF dataset.
    
    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset."""
        
    

    def __init__(self, img_dir, lab_dir, shuffle=False):
        
        n_images = len(fnmatch.filter(os.listdir(img_dir), '*.JPEG'))
        df = pd.read_csv(lab_dir, sep=',')
        dict_labs = dict(zip(df.ImageID.values+'.JPEG', df.Label.values ))
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.JPEG')
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.labs_ = []
        for img_name in self.img_names:
            lab = dict_labs[img_name]
            self.labs_.append(lab)

    def __len__(self):
        return self.len
    
    def get_len(self):
        return self.len
    
    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labs_[idx]
        
        transform = transforms.ToTensor()
        image = transform(image)
        transform = transforms.Resize((224,224))
        image = transform(image)

        return image, label

'''class InferenceDataset(Dataset):
    def __init__(self, n=None, transform=None, dataset="val", seed=None, images=None, pre_transformed=False,labels=None):
        if seed is not None:
            random.seed(seed)
            
        self.pre_transformed = pre_transformed
        if dataset == "val":
            """if pre_transformed:
                self.img_dir = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/data/imagenet/val_224/"
            else:
                self.img_dir = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/data/imagenet/val/"""
            self.img_dir = "/run/media/paul/EMTEC C410/images 1k"
            lab_dir = "/run/media/paul/EMTEC C410/labels.csv"
            df = pd.read_csv(lab_dir, sep=',')
            # self.truths = json.load(open("/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/validation_labels.json", "r"))
            self.truths = dict(zip(df.ImageID.values+'.JPEG', df.Label.values ))
            if images is not None:
                self.images = images
            elif n is not None:
                images = []
                for i in range(n):
                    images.append(random.choice(os.listdir(self.img_dir)))
                print(images,"images1")
                self.images = images
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
                print(images,"images \n")
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
        print(self.images.shape,"images shape \n")
        img = self.images[idx]

        y = self.truths[idx]
        if not self.pre_transformed:
            if img.shape[0] == 1:
                img = img.repeat_interleave(3, 0)
            if img.shape[0] > 3:
                img = img[:3]
            img = self.transform(img).float()
        else:
            img = img.float()

        return img / 255, y'''