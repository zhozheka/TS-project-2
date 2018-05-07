import numpy as np

import os
import re
import sys
import cv2
from torch import nn
import matplotlib.pyplot as plt
from torch import nn
from scipy.ndimage.filters import gaussian_filter
from PoseEstimator.PoseEstimation import model, process_image, process_output
import torch
import time

def main(torch_device, image_path):

    print('Loading model')
    model_pose = model(pretrained=True)
    model_pose = model_pose.to(torch_device)
    model_pose.eval() ;

    output_path = './output/'
    print('Processing network')
    heatmap_avg, paf_avg, segm = process_image(model_pose, image_path, torch_device)

    # load image
    img = plt.imread(image_path)
    
    
    # skeleton
    print('Processing skeleton')
    skeleton = process_output(heatmap_avg, paf_avg, image_path, skeleton=True)
    skeleton = skeleton.astype(np.uint8)
    b,g,r = cv2.split(skeleton)
    skeleton = cv2.merge((r,g,b))
    skeleton = np.dstack((skeleton, ((skeleton.sum(2)>0)*255)[:,:,None])).astype(np.uint8)
    plt.imsave(output_path + 'skeleton.png', skeleton)
    
    
    # image skeleton
    print('Processing image skeleton')
    img_skeleton = cv2.addWeighted(img, 1, skeleton[:,:,:3], 10, 0)
    plt.imsave(output_path + 'image_skeleton.png', img_skeleton)

    
    # segmentation
    print('Processing segmantation')
    prediction = nn.Upsample((skeleton.shape[0], skeleton.shape[1]), mode='bilinear', align_corners=True)(segm)      
    segmentation = nn.Softmax(dim=1)(prediction)[0,1].cpu().data.numpy()
    segmentation = ((segmentation[:,:,None]>0.55)*255).astype(np.uint8)[:,:,0]
    segmentation = gaussian_filter(segmentation, sigma=2)
    plt.imsave(output_path + 'segmentation.png', segmentation)
    
    
    # image segmentation
    print('Processing image segmentation')
    img_segm = np.dstack([img, segmentation])
    plt.imsave(output_path + 'image_segmentation.png', img_segm)
    
    
    # skeleton segmentation
    print('Processing image segmentation skeleton')
    img_skeleton_segm = cv2.addWeighted(img_segm, 1, skeleton, 10, 0)
    plt.imsave(output_path + 'image_segmentation_skeleton.png', img_skeleton_segm)
    
    print ('done!')
    
    
tic = time.time()
    
# parsing arguments
argv = sys.argv
arguments = {}
arguments['cuda'] = -1
arguments['image'] = ''
for i in range(1, len(argv),2):
    assert (argv[i][0] == '-')
    arguments[argv[i][1:]] = argv[i+1]
    
arguments['cuda']  = int(arguments['cuda'])
assert (arguments['cuda'] in range(-1,4))
assert not arguments['image'] == ''

if arguments['cuda'] == -1:
    torch_device = torch.device('cpu')
else:
    torch_device = torch.device('cuda:' + str(arguments['cuda']))
    #torch_device = torch.device('cuda:3')

if __name__ == '__main__':
    print ('\n', arguments['image'])
    print (torch_device,'\n')
    main(torch_device, arguments['image'] )
    
toc =time.time()
print ('time is %.5f'%(toc-tic))
