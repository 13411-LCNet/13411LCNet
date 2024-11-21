import torch
import sys, os

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import json
import random
from tqdm import tqdm

class RopeDataset(data.Dataset):

    def __init__(self, root_dir, csv_file, transform=None):
        self.anno = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        self.img_path = os.path.join(self.root_dir, self.anno.iloc[index, 0])
        self.im = Image.open(self.img_path).convert("RGB")

        # y_label = torch.tensor((self.anno.iloc[index, 1:11]))
        self.y_label = self.anno.iloc[index, 1:12].to_numpy().astype(np.float32)

        # print("Index : ", index)
        # print("Image : ", self.im)
        # print("label: ", self.y_label)
        # print("Label Type: ", type(self.y_label))

        if self.transform:
            self.im = self.transform(self.im)

        # print("Index : ", index)
        # print("Image : ", self.im)
        # print("type Image : ", type(self.im))
        # print("label: ", self.y_label)
        # print("Label Type: ", type(self.y_label))
        # print("Label Type: ", type(self.y_label[0]))

        return(self.im, self.y_label)

    # def __init__(self, image_dir, anno_path, input_transform=None, 
    #                 labels_path=None,
    #                 used_category=-1):
    #     self.rope = dset.CocoDetection(root=image_dir, annFile=anno_path)
    #     # with open('./data/coco/category.json','r') as load_category:
    #     #     self.category_map = json.load(load_category)
    #     self.category_map = category_map
    #     self.input_transform = input_transform
    #     self.labels_path = labels_path
    #     self.used_category = used_category
	
    #     self.labels = []
    #     if os.path.exists(self.labels_path):
    #         self.labels = np.load(self.labels_path).astype(np.float64)
    #         self.labels = (self.labels > 0).astype(np.float64)
    #     else:
    #         print("No preprocessed label file found in {}.".format(self.labels_path))
    #         l = len(self.coco)
    #         for i in tqdm(range(l)):
    #             item = self.coco[i]
    #             # print(i)
    #             categories = self.getCategoryList(item[1])
    #             label = self.getLabelVector(categories)
    #             self.labels.append(label)
    #         self.save_datalabels(labels_path)
    #     # import ipdb; ipdb.set_trace()

    # def __getitem__(self, index):
    #     input = self.coco[index][0]
    #     if self.input_transform:
    #         input = self.input_transform(input)
    #     return input, self.labels[index]


    # def getCategoryList(self, item):
    #     categories = set()
    #     for t in item:
    #         categories.add(t['category_id'])
    #     return list(categories)

    # def getLabelVector(self, categories):
    #     label = np.zeros(80)
    #     # label_num = len(categories)
    #     for c in categories:
    #         index = self.category_map[str(c)]-1
    #         label[index] = 1.0 # / label_num
    #     return label

    # def __len__(self):
    #     return len(self.coco)

    # def save_datalabels(self, outpath):
    #     """
    #         Save datalabels to disk.
    #         For faster loading next time.
    #     """
    #     os.makedirs(os.path.dirname(outpath), exist_ok=True)
    #     labels = np.array(self.labels)
    #     np.save(outpath, labels)
