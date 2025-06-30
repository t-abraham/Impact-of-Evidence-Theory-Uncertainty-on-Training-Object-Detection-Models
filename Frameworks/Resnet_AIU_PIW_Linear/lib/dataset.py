# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:43:12 2023

@author: Shaik
"""
import numpy as np
import random
import pickle
import inspect
import os , sys
import torch
import torchvision

from PIL import Image
from torchvision.transforms import PILToTensor

import glob as glob
from xml.etree import ElementTree as et
from torch.utils.data import Dataset, DataLoader


from functools import partial





class Datasetloader_1(torch.utils.data.Dataset):
    
    def __init__(self, path, classes,  img_ext = 'jpg'):
        
        
        self.path = path
        self.transforms = torchvision.transforms.Compose(
                                                                    [
                                                                        
                                                                        torchvision.transforms.ToTensor()
                                                                        # torchvision.transforms.Resize((256,256))
                                                                    ]
                                                               )
        self.classes = classes
        self.img_ext =  img_ext
        self.image_idx = glob.glob(f"{self.path}\*.{self.img_ext}")
        self.all_images = [image_idx.split('\\')[-1] for image_idx in self.image_idx]
        self.all_images  = sorted(self.all_images)
        
       
        # self.to_tensor = PILToTensor()
        # self.cache = {}
        
    # def _lazy_load_image(self, image_path , lazy_load = False):
    #     if lazy_load:
    #         return Image.open(image_path)._lazy_load()
    #     else:
    #         return Image.open(image_path)
        
    def __getitem__(self, idx):
       
        # get the image name and the full image path
        image_name = self.all_images[idx]
        image_path = os.path.join(self.path, image_name)
        
        
        # # check if image is already in cache
        # if image_path in self.cache:
        #     image, target = self.cache[image_path]
            
        # else:
            
        if image_name.endswith('.jpg'):
            # loader = partial(self._lazy_load_image, lazy_load = True)
            loader = Image.open
        elif image_name.endswith('.png'):
            # loader = partial(self._lazy_load_image, lazy_load = True)
            loader = Image.open
        else:
            raise ValueError('Unsupported image format')
           
            
        image = loader(image_path)
        
       
        
        
        # get the labels from xml file
        annot_filename =  image_name[:-4] + '.xml'
        annot_path = os.path.join(self.path, annot_filename)
        
        
        boxes = []
        labels = []
        tree = et.parse(annot_path)
        root = tree.getroot()
        
                   
        
        # get bnb from xml file
        for data in root.findall('object'):
            class_name = data.find("name").text
            
            if class_name in self.classes:
                
                labels.append(self.classes.index(class_name))
                
                # # xmin = left corner x-coordinates
                # x_min = int(data.find('bndbox').find('xmin').text)
                # # xmax = right corner x-coordinates
                # x_max = int(data.find('bndbox').find('xmax').text)
                # # ymin = left corner y-coordinates
                # y_min = int(data.find('bndbox').find('ymin').text)
                # # ymax = right corner y-coordinates
                # y_max = int(data.find('bndbox').find('ymax').text)
                
                # xmin = left corner x-coordinates
                x_min = float(data.find('bndbox').find('xmin').text)
                # xmax = right corner x-coordinates
                x_max = float(data.find('bndbox').find('xmax').text)
                # ymin = left corner y-coordinates
                y_min = float(data.find('bndbox').find('ymin').text)
                # ymax = right corner y-coordinates
                y_max = float(data.find('bndbox').find('ymax').text)
                
                
                boxes.append([x_min, y_min, x_max, y_max])
            
        # converting to tensor
        boxes = torch.as_tensor(boxes, dtype= torch.float32)
        
        # area of bnb
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # iscrowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        
        labels = torch.as_tensor(labels, dtype= torch.int64)
        
        image_id = torch.tensor([idx])
        
               
        # final target
        target = {}
        
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = image_id
            
            # # add images and target to cache
            # self.cache[image_path] = (image, target)
           
                      
                  
        return self.transforms(image), target
        # return image, target
    
    
    
    def __len__(self):
        return len(self.all_images)



                
                
            
        

   
if __name__ == "__main__":
   
    
    pass

        


        
        
            
            
        
        
        
        