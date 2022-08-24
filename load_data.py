import os
import numpy as np
from torch.utils.data import Dataset
import cv2


class MyDatasets(Dataset):

    def __init__(self, data_dir):
        # obtaining the directory where data file is saved
        self.data_dir = data_dir
        # saving (image,label) data pairs
        self.image_target_list = []
        # reading addresses of all training samples
        with open(self.data_dir, 'r') as fp:
            content = fp.readlines()
            str_list = [s.rstrip().split() for s in content]
            self.image_target_list = [(x[0], int(x[1])) for x in str_list]

    # loading samples
    def __getitem__(self, index):
        image_label_pair = self.image_target_list[index]
        img = np.load(image_label_pair[0])
        img = cv2.resize(img, (25,25))
        img = np.swapaxes(img, 0, 2)
        img = img[50:250, :, :]
        img = img[None, :, :, :]
        return img, image_label_pair[1]

    def __len__(self):
        return len(self.image_target_list)

