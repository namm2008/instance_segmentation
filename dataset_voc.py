import os,sys,re
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as data 
from PIL import Image as PILImage
from torchvision import transforms as transforms
from skimage.measure import label, regionprops
import json
import xml
import matplotlib.pyplot as plt

from xml.dom import minidom

# this class inherit Pytorch Dataset class
# loads 1 data point:
# 1 image and the vector of labels

class PascalVOC2012DatasetSegmentation(data.Dataset):

     def __init__(self, **kwargs): 
        # which problem
        self.problem = kwargs['problem'] 
        assert self.problem in ['semantic', 'instance']
        # Classes from Pascal VOC 2012 dataset, in the correct order without the bgr
        self.voc_classes = kwargs['classes']  
        self.dir = kwargs['dir']
        if self.problem == 'instance':
           self.dir_bbox = kwargs['dir_label_bbox']
        self.dir_masks = kwargs['dir_label_mask']
        #self.img_min_size = kwargs['img_min_size']
        self.img_max_size = kwargs['img_max_size']
        self.imgs = os.listdir(self.dir)
        self._classes = kwargs['classes']
        self.training = kwargs['training']
 
     # this method normalizes the image and converts it to Pytorch tensor
     # Here we use pytorch transforms functionality, and Compose them together,
     # Convert into Pytorch Tensor and transform back to large values by multiplying by 255
     def transform_img(self, img, img_max_size):
         if self.training == True:
            h,w,c = img.shape
            h_,w_ = img_max_size[0], img_max_size[1]
            img_size = tuple((h_,w_))
         # these mean values are for BGR!
         if self.training == False:
              t_ = transforms.Compose([
                              transforms.ToPILImage(),
                              #transforms.Resize(img_size),
                              transforms.ToTensor()])

         if self.training == True:
            t_ = transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.Resize(img_size),
                              transforms.ToTensor()])
         img = 255*t_(img)
         means = torch.tensor([255*0.485, 255*0.457, 255*0.407], dtype = torch.float)
         means = means[:,None,None]
         img = img - means
         # need this for the input in the model
         # returns image tensor (CxHxW)
         return img

     # load one image
     # idx: index in the list of images
     def load_img(self, idx):
         #im = cv2.imread(os.path.join(self.dir, self.imgs[idx]))
         #im = self.transform_img(im, self.img_max_size)
         
         im = PILImage.open(os.path.join(self.dir, self.imgs[idx]))
         #if self.training == True:
         im = np.array(im)
         im = self.transform_img(im, self.img_max_size)
         return im

     # this method returns the size of the object inside the bounding box:
     # input is a list in format xmin,ymin, xmax,ymax
     def get_size(bbox):
         _h, _w = bbox[3] - bbox[1], bbox[2]-bbox[0]
         size = _h *_w
         return size

     # semantic segmentation mask
     # size HxWx1, with pixels set to the values of correct class
     def extract_segmentation_mask_pascal(self, idx, list_of_classes):
         # all 'blobs' smaller than this value will be delted
         min_size = 500
         mask_name = self.imgs[idx].split('.')[0] + '.png'
         mask = PILImage.open(os.path.join(self.dir_masks,mask_name))
         if self.training == True:
            t_ = transforms.Compose([transforms.Resize(self.img_max_size)])
            mask = t_(mask)
            mask = np.array(mask)
         else:
            mask = np.array(mask)
         # clean the mask
         mask[mask==255] = 0
         lab = label(mask)
         regions = regionprops(lab)
         # loop through all isolated regions, get rid of small regions (convert to backgrund)
         for idx, r in enumerate(regions):
             if r.area<min_size:
                mask[lab == idx+1] = 0
         if np.max(mask)==0:
           mask = torch.as_tensor(mask, dtype=torch.uint8) 
         else:
           mask = mask/np.max(mask)
           mask = np.ceil(mask )
         mask = torch.as_tensor(mask, dtype=torch.uint8) 
         
         return mask


     #'magic' method: size of the dataset
     def __len__(self):
         return len(os.listdir(self.dir))        

 
     #'magic' method: iterates through the dataset directory to return the image and its gt
     def __getitem__(self, idx):
        # here you have to implement functionality using the methods in this class to return X (image) and y (its label)
        # X must be dimensionality (3,max_size[1], max_size[0]) if you use VGG16
        # y must be dimensioanlity (self.voc_classes)
        X = self.load_img(idx)
        if self.problem == 'semantic':
           y = self.extract_segmentation_mask_pascal(idx, self._classes)
        else:
           y = self.extract_bboxes_and_masks_pascal(idx, self._classes) 
        return idx, X,y
