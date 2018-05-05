import torch
from torch import nn
from config_reader import config_reader
import cv2
import util
import numpy as np

def make_blocks():
    blocks = {}

    block0  = [{'conv1_1':      [3,64,3,1,1]},
               {'conv1_2':      [64,64,3,1,1]},
               {'pool1_stage1': [2,2,0]},
               {'conv2_1':      [64,128,3,1,1]},
               {'conv2_2':      [128,128,3,1,1]},
               {'pool2_stage1': [2,2,0]},
               {'conv3_1':      [128,256,3,1,1]},
               {'conv3_2':      [256,256,3,1,1]},
               {'conv3_3':      [256,256,3,1,1]},
               {'conv3_4':      [256,256,3,1,1]},
               {'pool3_stage1': [2,2,0]},
               {'conv4_1':      [256,512,3,1,1]},
               {'conv4_2':      [512,512,3,1,1]},
               {'conv4_3_CPM':  [512,256,3,1,1]},
               {'conv4_4_CPM':  [256,128,3,1,1]}]

    blocks['block1_1']  = [{'conv5_1_CPM_L1':   [128,128,3,1,1]},
                           {'conv5_2_CPM_L1':   [128,128,3,1,1]},
                           {'conv5_3_CPM_L1':   [128,128,3,1,1]},
                           {'conv5_4_CPM_L1':   [128,512,1,1,0]},
                           {'conv5_5_CPM_L1':   [512,38,1,1,0]}]

    blocks['block1_2']  = [{'conv5_1_CPM_L2':   [128,128,3,1,1]},
                           {'conv5_2_CPM_L2':   [128,128,3,1,1]},
                           {'conv5_3_CPM_L2':   [128,128,3,1,1]},
                           {'conv5_4_CPM_L2':   [128,512,1,1,0]},
                           {'conv5_5_CPM_L2':   [512,19,1,1,0]}]

    for i in range(2,7):
        blocks['block%d_1'%i]  = [{'Mconv1_stage%d_L1'%i:   [185,128,7,1,3]},
                                  {'Mconv2_stage%d_L1'%i:   [128,128,7,1,3]},
                                  {'Mconv3_stage%d_L1'%i:   [128,128,7,1,3]},
                                  {'Mconv4_stage%d_L1'%i:   [128,128,7,1,3]},
                                  {'Mconv5_stage%d_L1'%i:   [128,128,7,1,3]},
                                  {'Mconv6_stage%d_L1'%i:   [128,128,1,1,0]},
                                  {'Mconv7_stage%d_L1'%i:   [128,38,1,1,0]}]
        
        blocks['block%d_2'%i]  = [{'Mconv1_stage%d_L2'%i:   [185,128,7,1,3]},
                                  {'Mconv2_stage%d_L2'%i:   [128,128,7,1,3]},
                                  {'Mconv3_stage%d_L2'%i:   [128,128,7,1,3]},
                                  {'Mconv4_stage%d_L2'%i:   [128,128,7,1,3]},
                                  {'Mconv5_stage%d_L2'%i:   [128,128,7,1,3]},
                                  {'Mconv6_stage%d_L2'%i:   [128,128,1,1,0]},
                                  {'Mconv7_stage%d_L2'%i:   [128,19,1,1,0]}]
    return block0, blocks
    
def make_layers(cfg_dict):
    layers = []
    for i in range(len(cfg_dict)-1):
        one_ = cfg_dict[i]
        for k,v in one_.items():      
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
                
    one_ = list(cfg_dict[-1].keys())
    k = one_[0]
    v = cfg_dict[-1][k]
    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
    layers += [conv2d]
    return nn.Sequential(*layers)


def make_model(block0, blocks):
    layers = []
    for i in range(len(block0)):
        one_ = block0[i]
        for k,v in one_.items():      
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]  

    models = {}           
    models['block0'] = nn.Sequential(*layers)        

    for k,v in blocks.items():
        models[k] = make_layers(v)

    return models
    
class model(nn.Module):
    def __init__(self, pretrained=True):
        super(model, self).__init__()
        
        #make blocks
        block0, blocks = make_blocks()
        
        #make model_dict
        model_dict= make_model(block0, blocks)
        
        #make model
        self.model0   = model_dict['block0']
        self.model1_1 = model_dict['block1_1']        
        self.model2_1 = model_dict['block2_1']  
        self.model3_1 = model_dict['block3_1']  
        self.model4_1 = model_dict['block4_1']  
        self.model5_1 = model_dict['block5_1']  
        self.model6_1 = model_dict['block6_1']  
        
        self.model1_2 = model_dict['block1_2']        
        self.model2_2 = model_dict['block2_2']  
        self.model3_2 = model_dict['block3_2']  
        self.model4_2 = model_dict['block4_2']  
        self.model5_2 = model_dict['block5_2']  
        self.model6_2 = model_dict['block6_2']
        
        if pretrained:
            self.weight_name = './model/pose_model.pth'
            self.load_state_dict(torch.load(self.weight_name))
        
    def forward(self, x):    
        out1 = self.model0(x)
        
        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2  = torch.cat([out1_1,out1_2,out1],1)
        
        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3   = torch.cat([out2_1,out2_2,out1],1)
        
        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4   = torch.cat([out3_1,out3_2,out1],1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5   = torch.cat([out4_1,out4_2,out1],1)  
        
        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6   = torch.cat([out5_1,out5_2,out1],1)         
              
        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)
        
        return out6_1, out6_2
    
def process_image(net, img_path, torch_device):
    param_, model_ = config_reader()
    
    oriImg = cv2.imread(img_path) # B,G,R order
    imageToTest = torch.FloatTensor(oriImg).permute((2,0,1))[None]
    multiplier = [x * model_['boxsize'] / oriImg.shape[0] for x in param_['scale_search']]

    heatmap_avg = torch.zeros((len(multiplier),19,oriImg.shape[0], oriImg.shape[1])).to(torch_device)
    paf_avg = torch.zeros((len(multiplier),38,oriImg.shape[0], oriImg.shape[1])).to(torch_device)

    for m in range(len(multiplier)):
        scale = multiplier[m]
        h = int(oriImg.shape[0]*scale)
        w = int(oriImg.shape[1]*scale)
        pad_h = 0 if (h%model_['stride']==0) else model_['stride'] - (h % model_['stride']) 
        pad_w = 0 if (w%model_['stride']==0) else model_['stride'] - (w % model_['stride'])
        new_h = h + pad_h
        new_w = w + pad_w

        imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_['stride'], model_['padValue'])
        imageToTest_padded = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5


        feed = torch.Tensor(imageToTest_padded).to(torch_device)

        output1, output2 = net(feed)
        print ('output1', output1.size())
        print ('output2', output2.size())

        heatmap = nn.Upsample((oriImg.shape[0], oriImg.shape[1]), mode='bilinear', align_corners=True)(output2)

        paf = nn.Upsample((oriImg.shape[0], oriImg.shape[1]), mode='bilinear', align_corners=True)(output1)       

        heatmap_avg[m] = heatmap[0].data
        paf_avg[m] = paf[0].data  

    heatmap_avg = heatmap_avg.mean(0).permute((1,2,0)).cpu().numpy()
    paf_avg = paf_avg.mean(0).permute((1,2,0)).cpu().numpy()

    return heatmap_avg, paf_avg