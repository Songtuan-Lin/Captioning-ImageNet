import os
import cv2
import torch
import json
import inflect
import numpy as np

from PIL import Image

from torch.utils.data import Dataset
from nltk.corpus import wordnet as wn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CustomizedDataset(Dataset):
    def __init__(self, root, transitions):
        '''
        Load an arbitrary dataset
        Args:
            root (str): root directory of dataset
            transitions (list[Tensor]): transition table of a finite automata 
            represented as a list of Pytorch tensor
        '''
        self.transitions = [transition.to(device) for _, transition in transitions.items()]
        image_files = os.listdir(root)
        self.image_paths = [os.path.join(root, image_file) for image_file in image_files]
    
    def _image_transform(self, image_path):
        '''
        Read an image and apply necessary transform

        Args:
            image_path (str): path to the image

        Returns:
            Tensor: image data
            int: scale used to resize image
        '''
        img = Image.open(image_path)
        im = np.array(img).astype(np.float32)
        if len(im) == 2:
            im = im[:,:,np.newaxis]
            im = np.concatenate((im,im,im), axis=2)
        im = im[:, :, ::-1]
        im -= np.array([102.9801, 115.9465, 122.7717])
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(800) / float(im_size_min)
        # Prevent the biggest axis from being more than max_size
        if np.round(im_scale * im_size_max) > 1333:
            im_scale = float(1333) / float(im_size_max)
        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=cv2.INTER_LINEAR
        )
        img = torch.from_numpy(im).permute(2, 0, 1)
        return img, im_scale 

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        image_raw = Image.open(image_path)
        image, image_scale = self._image_transform(image_path)
        data = {'image_raw': image_raw, 'image': image, 
                'image_scale': image_scale, 'transitions': self.transitions}
        return data

    def __len__(self):
        return len(self.image_paths)

