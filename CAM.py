
# https://github.com/KangBK0120/CAM/blob/master/create_cam.py
# https://github.com/pranavdheram/interpretable-DL/blob/d881ea3770a7b618fa85f979a3cd3fec5d18ef2b/CNN_visualizations/original_cam.py

import os
import argparse

import cv2
import torch
import numpy as np
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision import models
import utils
from model import ft_net, ft_net_dense, PCB, verif_net
from tripletfolder_oclu import TripletFolder
import random

def create_cam():
    # net = models.resnet50(pretrained=True)  
    # for name, _ in net.named_modules():
        # print(name)
    
    transform_train_list = [
        transforms.Resize((320, 120)),
        transforms.ToTensor(),
        #transforms.Normalize([0.486, 0.459, 0.408], [0.229, 0.224, 0.225])
        ]
    data_transforms = {
    'train': transforms.Compose( transform_train_list )}
    image_datasets = {}
    data_dir = '../Dataset/market1501/Market-1501-v15.09.15/pytorch'    
    image_datasets['train'] = TripletFolder(os.path.join(data_dir, 'train_all'), data_transforms['train'])
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=False, num_workers=1) for x in ['train']}
    
    model = ft_net(751, 'train_CNN')
    ckp_path = '../model_ckpts/oclu/pure_fc/epoch_7 loss_0.014358007150822052.pth'
    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    finalconv_name = 'layer4'
    # for name, _ in model.named_modules():
        # print(name)
    # hook
    feature_blobs = []
    def hook_feature(module, input, output):
        feature_blobs.append(output.cpu().data.numpy())

    model._modules["model"]._modules.get(finalconv_name).register_forward_hook(hook_feature)
    params = list(model.parameters())
    #print(model.parameters())
    for name, param in model.named_parameters():
        print(name)

    # get weight only from the last layer(linear)
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())
    #print(len(weight_softmax))
    
    def returnCAM(feature_conv, weight_softmax, class_idx):
        size_upsample = (320, 120)     
        _, nc, h, w = feature_conv.shape
        print(feature_conv.shape)
        output_cam = []
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam
    
    for data in dataloaders['train']:
        inputs, par_inputs, labels = data
        random_str = str(random.randint(0,1000))
        image_PIL = transforms.ToPILImage()(par_inputs[0])
        image_PIL.save('orig'+random_str+'.png')

        logit = model(par_inputs)
        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        print("True label : %d, Predicted label : %d, Probability : %.2f" % (labels.item(), idx[0].item(), probs[0].item()))
        CAMs = returnCAM(feature_blobs[0], weight_softmax, [idx[0].item()])
        
        img = cv2.imread('orig'+random_str+'.png')
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite('cam'+random_str+'.png', result)
        feature_blobs.clear()
        break
    
if __name__ == '__main__':
    create_cam()