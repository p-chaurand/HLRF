import random
import os
import numpy as np
import fnmatch
import pandas as pd
import PIL.Image as pil


from torch.utils.data import Dataset
from torchvision.io import read_image

from torchvision import transforms as T

class AdversarialDataset(Dataset):
	def __init__(self,img_dir,lab_dir,shuffle=True):
		self.img_dir = img_dir
		self.lab_dir = lab_dir
		self.n_images = len(fnmatch.filter(os.listdir(self.img_dir),'*.JPEG'))
		df = pd.read_csv(self.lab_dir,sep = ',')
		self.images = os.listdir(self.img_dir)
		self.images_names = fnmatch.filter(os.listdir(self.img_dir),'*.JPEG')
		self.to_tensor = T.ToTensor()
		self.transform = T.Compose([T.Resize(256), T.CenterCrop(224)])
		self.shuffle = shuffle
		dict_labs = dict(zip(df.ImageID.values+'.JPEG', df.Label.values ))
		#self.images_path = [os.path.join(self.img_dir, self.images_names[i]) for i in range(self.n_images)]
		if self.shuffle:
			random.shuffle(self.images_names)
		self.labs_ = []
		for img_name in self.images_names:
			lab = dict_labs[img_name]
			self.labs_.append(lab)

	def __len__(self):
		return self.n_images

	def get_len(self):
		return self.n_images

	def __getitem__(self, idx):
		assert idx < len(self), 'index range error'
		img = pil.open(os.path.join(self.img_dir,self.images_names[idx]))
		y = self.labs_[idx]
		img = self.to_tensor(img)
		img = self.transform(img)
		return img, y