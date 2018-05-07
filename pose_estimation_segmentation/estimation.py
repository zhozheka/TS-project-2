import torch
import os
import re
import sys
import cv2
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from torch import nn
from scipy.ndimage.filters import gaussian_filter
from PoseEstimator.PoseEstimation import model, process_image, process_output

def main():
    torch_device = torch.device('cuda:3')

    model_pose = model(pretrained=True)
    model_pose = model_pose.to(torch_device)
    model_pose.eval() ;

    image_path = './sample_image/b1.jpg'
    output_path = './output/'

    heatmap_avg, paf_avg, segm = process_image(model_pose, image_path, torch_device)

    #skeleton
    skeleton = process_output(heatmap_avg, paf_avg, image_path, skeleton=True)
    skeleton = skeleton.astype(np.uint8)
    b,g,r = cv2.split(skeleton)
    skeleton = cv2.merge((r,g,b))
    skeleton = np.dstack((skeleton, ((skeleton.sum(2)>0)*255)[:,:,None])).astype(np.uint8)
    plt.imsave(output_path + 'skeleton.png', skeleton)

    # segmentation
    prediction = nn.Upsample((skeleton.shape[0], skeleton.shape[1]), mode='bilinear', align_corners=True)(segm)      
    segmentation = nn.Softmax(dim=1)(prediction)[0,1].cpu().data.numpy()

    segmentation = ((segmentation[:,:,None]>0.55)*255).astype(np.uint8)[:,:,0]
    segmentation = gaussian_filter(segmentation, sigma=2)
    plt.imsave(output_path + 'segmentation.png', segmentation)
    
    # load image
    img = plt.imread(image_path)
    
    # image with segmentation
    img_segm = np.dstack([img, segmentation])
    plt.imsave(output_path + 'image_segmentation.png', img_segm)
    
    # skeleton with segmentation
    img_skeleton_segm = cv2.addWeighted(img_segm, 1, skeleton, 10, 0)
    plt.imsave(output_path + 'image_segmentation_skeleton.png', img_skeleton_segm)
    
    