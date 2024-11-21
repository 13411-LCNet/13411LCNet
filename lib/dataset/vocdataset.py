import torch
import sys, os

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
import json
import random
from tqdm import tqdm

category_map = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4, "bottle": 5, "bus": 6, "car": 7, "cat": 8, "chair": 9, "cow": 10, "diningtable": 11, "dog": 12, "horse": 13, "motorbike": 14, "person": 15, "pottedplant": 16, "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}

class VOCdataset(data.Dataset):
    def __init__(self, rootDir, imgType, anno_path, input_transform=None, 
                    labels_path=None,
                    used_category=-1):
        
        # print("ALMOST FINISHED 1")
        self.voc = dset.VOCDetection(root=rootDir, year="2007", image_set=imgType, download=True)
        # self.voc = dset.VOCDetection(root=image_dir, annFile=anno_path)

        # print("ALMOST FINISHED 2")
        
        # with open('./data/coco/category.json','r') as load_category:
        #     self.category_map = json.load(load_category)
        self.category_map = category_map
        self.input_transform = input_transform
        self.labels_path = labels_path
        self.used_category = used_category
	
        self.labels = []
    
        print("No preprocessed label file found in {}.".format(self.labels_path))
        l = len(self.voc)
        for i in tqdm(range(l)):
            item = self.voc[i]
            # print("Item: ", item)
            categories = self.getCategoryList(item[1])
            # print(categories)
            label = self.getLabelVector(categories)
            #print("Label: ", label)
            #print("Label Type: ", type(label))
            self.labels.append(label)
        # self.save_datalabels(labels_path)
        # import ipdb; ipdb.set_trace()

    def __getitem__(self, index):
        input = self.voc[index][0]
        # print("Input: ", input)
        # print("Labels: ", self.labels[index])
        
        # print("Index : ", index)
        # print("Image : ", input)
        # print("Label: ", self.labels[index])
        # print("Label Type: ", type(self.labels[index]))

        if self.input_transform:
            input = self.input_transform(input)

        # print("Index : ", index)
        # print("Image : ", input)
        # print("type Image : ", type(input))
        # print("label: ", self.labels[index])
        # print("Label Type: ", type(self.labels[index]))
        # print("Label Type: ", type(self.labels[index][0]))

        return input, self.labels[index]


    def getCategoryList(self, item):
        categories = set()

        annotation = item.get('annotation', {})
    
        # Check if 'object' key exists and if it's a list of objects
        objects = annotation.get('object', [])

        # If there's only one object, it might not be a list, so wrap it in a list
        if isinstance(objects, dict):
            objects = [objects]

        for obj in objects:

            # print("obj name: ", obj['name'])
            categories.add(obj['name'])
        return list(categories)

    def getLabelVector(self, categories):
        label = np.zeros(20)
        # label_num = len(categories)
        for c in categories:
            index = self.category_map[str(c)]-1
            label[index] = 1.0 # / label_num

        # print("Label vector", label)
        return label

    def __len__(self):
        return len(self.voc)

    def save_datalabels(self, outpath):
        """
            Save datalabels to disk.
            For faster loading next time.
        """
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        labels = np.array(self.labels)
        np.save(outpath, labels)


