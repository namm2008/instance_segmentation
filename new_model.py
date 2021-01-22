from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import os,sys,re
import os.path as osp
from DNModel import net
from img_process import preprocess_img, inp_to_image
import pandas as pd
import random
import fcn
import torch.nn as nn
import torchvision
from torchvision import transforms as transforms


#initiate some hyperparameters
batch_size = int(1.0)
confidence = float(0.5)
nms_thesh = float(0.4)
num_classes = 80

CUDA = torch.cuda.is_available()

if CUDA:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

classes = load_classes('data/coco.names') 
model = net("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")
model.DNInfo["height"] = '256'
in_dim = int(model.DNInfo["height"])

if CUDA:
    model.cuda()

model.eval()

pascal_object_categories = ['__bgr__', 'object']

# convert list to dict
pascal_voc_classes = {}
for id, name in enumerate(pascal_object_categories):
    pascal_voc_classes[name] = id

# also names from ids
pascal_voc_classes_name = {}
for id, name in enumerate(pascal_object_categories):
    pascal_voc_classes_name[id] = name

fcn8 = fcn.FCN8s()
weights_path = 'fcn8_pascal_500.pth'
sd = torch.load(weights_path,map_location=torch.device('cpu'))
fcn8.load_state_dict(sd)

if CUDA:
    fcn8.cuda()

fcn8.eval()


def preprocess_image(bbox):
    orig_im = bbox
    dim = orig_im.shape[1], orig_im.shape[0]
    means = np.array([255*0.485, 255*0.457, 255*0.407])
    means = means[None,None,:]
    img_ = orig_im - means
    img_ = img_.transpose((2,0,1)).copy()
    img_ = torch.tensor(img_, dtype = torch.float, device = device).unsqueeze(0)
    return img_, orig_im, dim



def process_dim(prediction,im_dim_list):
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
    if CUDA:
        im_dim_list = im_dim_list.cuda()

    im_dim_list = torch.index_select(im_dim_list, 0, prediction[:,0].long())
    scaling_factor = torch.min(in_dim/im_dim_list,1)[0].view(-1,1)

    prediction[:,[1,3]] -= (in_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
    prediction[:,[2,4]] -= (in_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
       
    prediction[:,1:5] /= scaling_factor

    for i in range(prediction.shape[0]):
        prediction[i, [1,3]] = torch.clamp(prediction[i, [1,3]], 0.0, im_dim_list[i,0])
        prediction[i, [2,4]] = torch.clamp(prediction[i, [2,4]], 0.0, im_dim_list[i,1])
    return  prediction

def pred_group(prediction):
    y1 = prediction[prediction[:, 7] == 0]
    y2 = prediction[prediction[:, 7] == 56]
    y3 = prediction[prediction[:, 7] == 57]
    y4 = prediction[prediction[:, 7] == 60]
    y5 = prediction[prediction[:, 7] == 61]
    t1 = torch.cat((y1, y2), 0)
    t2 = torch.cat((t1, y3), 0)
    t3 = torch.cat((t2, y4), 0)
    t4 = torch.cat((t3, y5), 0)
    return t4

def new_model(imlist):
    im_batches,orig_ims,im_dim_list = preprocess_img(imlist, in_dim)
    
    if CUDA:
        im_batches = im_batches.cuda()
    
    with torch.no_grad():
        prediction_ = model(im_batches, CUDA)
    
    #run the non-max suppression
    prediction_ = write_results(prediction_, confidence, num_classes, nms = True, nms_conf = nms_thesh)
    
    prediction = pred_group(prediction_)
    
    if len(prediction) == 0:
        return torch.tensor(()),torch.tensor(()) ,torch.tensor(()) , []
    
    #convert the bounding box scale
    prediction = process_dim(prediction,im_dim_list)  
    
    pred_bbox = prediction[:,1:5]
    pred_bbox = pred_bbox.int()
    
    #cropping bbox
    bbox_img = []
    for k in range(len(prediction)):
        b_box = orig_ims[pred_bbox[k,1]:pred_bbox[k,3],pred_bbox[k,0]:pred_bbox[k,2],:]
        b_box_batches, _,_ = preprocess_image(b_box)
        
        with torch.no_grad():
            bbox_mask_pred = fcn8(b_box_batches)
        
        bbox_mask_pred = bbox_mask_pred.argmax(1).squeeze_(0).detach().clone().cpu().numpy()         
        bbox_mask_pred = cv2.copyMakeBorder(bbox_mask_pred,  
                                            pred_bbox[k,1], 
                                            im_dim_list[1] - pred_bbox[k,3], 
                                            pred_bbox[k,0], 
                                            im_dim_list[0] - pred_bbox[k,2], 
                                            cv2.BORDER_CONSTANT)
        bbox_img.append(bbox_mask_pred)
    
    for p in range(len(prediction)):
        for h in range(len(prediction)):
            if p == h:
                continue
            intersection_ct = (bbox_img[p] & bbox_img[h]).sum()
            if intersection_ct == 0:
                continue
            else:
                intersection = (bbox_img[p] & bbox_img[h])
            if prediction[p,5] >= prediction[h,5]:
                bbox_img[h] = bbox_img[h] - intersection
            else: 
                bbox_img[p] = bbox_img[p] - intersection
    scores = prediction[:,5]
    labels = prediction[:,7].int()
    return scores, pred_bbox, labels, bbox_img
