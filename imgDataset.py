from torch.utils.data import Dataset
import cv2
import numpy as np


class ImgDataset(Dataset):
    def __init__(self, data_dir, labels, dim, shuffle=False):

        self.dim = dim
        self.data_dir = data_dir
        self.labels = labels

    def __getitem__(self, index):
        # print(self.data_dir[index])
        try:
            img = cv2.imread(self.data_dir[index], 0)
        except:
            print('cant read', self.data_dir[index])

        resized_img = cv2.resize(img, (self.dim[1], self.dim[0]))
        #normalize the image
        resized_img = (resized_img - np.min(resized_img))/(np.max(resized_img) - np.min(resized_img))
        resized_img = np.expand_dims(resized_img, axis=0).astype(np.float32)
        label = (self.labels[index]).astype(np.float32)
        return resized_img, label

    def __len__(self):
        return len(self.data_dir)

