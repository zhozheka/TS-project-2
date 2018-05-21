import numpy as np

import os
import re
import sys
import cv2
from torch import nn
import matplotlib.pyplot as plt
from torch import nn
from scipy.ndimage.filters import gaussian_filter
from PoseEstimator.PoseEstimation import model, process_image2, process_output2
import torch
from skimage.transform import resize

from socketserver import start_server
from io import BytesIO

counter = 0

def cv2_from_stream(img_stream):
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def as_server(torch_device):
    def processor(input_data, model_pose, space_init):
        global counter
        processing_marker = input_data[0]
        imstream = BytesIO(input_data[1:])
        orig_img = cv2_from_stream(imstream) # B,G,R order
        heatmap_avg, paf_avg, segm = process_image2(model_pose, orig_img, torch_device)
        img = plt.imread(imstream, format='jpg')

#         print('Processing skeleton')
        skeleton = process_output2(heatmap_avg, paf_avg, orig_img, skeleton=True)
        skeleton = skeleton.astype(np.uint8)
        b,g,r = cv2.split(skeleton)
        skeleton = cv2.merge((r,g,b))
        skeleton = np.dstack((skeleton, ((skeleton.sum(2)>0)*255)[:,:,None])).astype(np.uint8)

        # segmentation
#         print('Processing segmantation')
        prediction = nn.Upsample((skeleton.shape[0], skeleton.shape[1]), mode='bilinear', align_corners=True)(segm)      
        segmentation = nn.Softmax(dim=1)(prediction)[0,1].cpu().data.numpy()
        segmentation = ((segmentation[:,:,None]>0.55)*255).astype(np.uint8)[:,:,0]
        segmentation = gaussian_filter(segmentation, sigma=2)

        # image segmentation
#         print('Processing image segmentation')
        img_segm = np.dstack([img, segmentation])

        # skeleton segmentation
#         print('Processing image segmentation skeleton')
        img_skeleton_segm = cv2.addWeighted(img_segm, 1, skeleton, 10, 0)
#         plt.imsave('output/image_for_output.png', img_skeleton_segm)

        
        space = (resize(space_init, img.shape[:2])*255).astype(np.uint8)
        img_space = np.zeros(img.shape,dtype=np.uint8)
        alpha_s = img_segm[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            img_space[:,:,c] = (alpha_s * img_segm[:,:,c] + alpha_l * space[:,:,c])
            

        outstream = BytesIO()
        if processing_marker == 1:
            plt.imsave(outstream, img_skeleton_segm, format='jpg')
        elif processing_marker == 2:
            plt.imsave(outstream, img_segm, format='jpg')
        elif processing_marker == 3:
            plt.imsave(outstream, skeleton, format='jpg')
        else:
            plt.imsave(outstream, img_space, format='jpg')
        
        counter += 1
        print ('images processed: {}'.format(counter))
        return outstream.getvalue()
    
    space_init = plt.imread('samples/space.jpg')

    model_pose = model(pretrained=True)
    model_pose = model_pose.to(torch_device)
    model_pose.eval() ;

    start_server(lambda data: processor(data, model_pose, space_init))
    
if __name__ == '__main__':
    argv = sys.argv
    torch_device = torch.device('cpu')
    if len(argv) > 1:
        if int(argv[1]) == -1:
            torch_device = torch.device('cpu')
        else:
            torch_device = torch.device('cuda:' + argv[1])

    print (torch_device,'\n')

    as_server(torch_device)
