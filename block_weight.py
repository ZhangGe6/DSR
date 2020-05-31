
# https://github.com/KangBK0120/CAM/blob/master/create_cam.py
# https://github.com/pranavdheram/interpretable-DL/blob/d881ea3770a7b618fa85f979a3cd3fec5d18ef2b/CNN_visualizations/original_cam.py

import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    #data_dir = '../Dataset/market1501/Market-1501-v15.09.15/pytorch'
    data_dir = '../Dataset/tmp_parREID'
    image_datasets['train'] = TripletFolder(data_dir, data_transforms['train'])
    #image_datasets['train'] = TripletFolder(os.path.join(data_dir, 'train_all'), data_transforms['train'])
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=True, num_workers=1) for x in ['train']}
    
    model = ft_net(751, 'train_CNN')
    ckp_path = '../model_ckpts/oclu/task1+task2/epoch_10 loss_0.006345328259383246.pth'
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
    # for name, param in model.named_parameters():
        # print(name)

    # get weight only from the last layer(linear)
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())
    #print(len(weight_softmax))
    
        
    
    def weightBlock(feature_conv, weight_softmax, class_idx):
   
        feature_conv = torch.from_numpy(feature_conv)        
        _, nc, h, w = feature_conv.size()
        print(feature_conv.size())   # (1, 2048, 20, 8)
       
        
        feature_conv1 = feature_conv.view((nc, h,w))
        print(feature_conv1.size()) 
        print(type(feature_conv1))
        feature_conv2 = feature_conv1.sum(dim=0)
        print(feature_conv2.size())
        print(feature_conv2)
        
        # cam = weight_softmax[class_idx].dot(feature_conv.view((nc, h*w)))
        # _, nc, h, w = feature_conv.shape
        # cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        # cam = cam.reshape(h, w)
        # feature_conv2 = torch.from_numpy(cam)
        
        Max = torch.max(feature_conv2)
        Min = torch.min(feature_conv2)
        print(Max,Min)
        feature_conv2 = (feature_conv2 - Min)/(Max-Min)
        print(feature_conv2)
        return feature_conv2              
        
        # output_cam = []
        # cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        # cam = cam.reshape(h, w)
        # cam = cam - np.min(cam)
        # cam_img = cam / np.max(cam)
        # cam_img = np.uint8(255 * cam_img)
        # output_cam.append(cv2.resize(cam_img, size_upsample))
        # return output_cam
    
    for data in dataloaders['train']:
        inputs, par_inputs, labels = data
        random_str = str(random.randint(0,1000))
        image_PIL = transforms.ToPILImage()(inputs[0])
        image_PIL.save('orig'+random_str+'.png')

        logit = model(inputs)
        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        print("True label : %d, Predicted label : %d, Probability : %.2f" % (labels.item(), idx[0].item(), probs[0].item()))
        summed_feat = weightBlock(feature_blobs[0], weight_softmax, [idx[0].item()])
        
        # img = cv2.imread('orig'+random_str+'.png')
        # height, width, _ = img.shape
        # heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
        # result = heatmap * 0.3 + img * 0.5
        # cv2.imwrite('cam'+random_str+'.png', result)
        feature_blobs.clear()
        break
    return summed_feat
    
class draw_block():
    def __init__(self):
        #背景颜色为黑色的画布
        #im = np.zeros((300, 300, 3), dtype="uint8") #3
        #背景颜色为白色的画布`
        self.im = np.ones((700,280,3),np.uint8)*255
        #画布使用默认字体
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        #起始作图坐标
        self.leftx = 5
        self.lefty = 5
        #矩形单元长高
        self.len = 30
        self.high = 30
     
        
        #第idx张画布（也许一张不够）
        #self.filename = filename
        
    #画一个矩形序列单元
    def draw_single_block(self, v, h, weight):
        self.leftx = 5
        self.lefty = 5
        for i in range(v):
            self.move_down()
        for i in range(h):
            self.move_right()
          
        #矩形边框颜色
        #cv2.rectangle(self.im, (self.leftx, self.lefty), (self.leftx+self.len, self.lefty+self.high), (int(listseq[0],16),int(listseq[1],16),int(listseq[2],16))) #12
        #填充矩形颜色
        #print(self.colorId%len(rgb_dict))
        #bseq, gseq, rseq = rgb_dict[self.colorId%len(rgb_dict)]
        # M = -self.colorId*30
        # print(M)
        # bseq, gseq, rseq = 178-M, 34-M, 34-M
        # M = np.ones((self.len,self.high), dtype = "uint8") * ( self.colorId)
        # cv2.subtract(image, M)
        M = -int((1-weight)*255)
        #print(M)
        bseq, gseq, rseq = 178-M, 34-M, 34-M
      
        cv2.rectangle(self.im, (self.leftx, self.lefty), (self.leftx+self.len, self.lefty+self.high), (rseq, gseq, bseq), thickness=-1)
        #填充文字
        seq = str(round(weight.item(),2))
       
        cv2.putText(self.im,seq,(self.leftx, self.lefty+10), self.font, 0.4, (255-rseq, 255-gseq,255-bseq), 1)
    
    def draw_all(self, mat):
        for v in range(len(mat)):
            for h in range(len(mat[0])):
                self.draw_single_block(v, h,mat[v][h])
    #保存序列图
    def write_jpg(self):
        cv2.imwrite("block.jpg", self.im)

    #往右移一个位置画序列单元
    def move_down(self):
        self.lefty = self.lefty + self.high+5
            
    #另起一行画序列单元
    def move_right(self):
        self.leftx = self.leftx+self.len+5

    
if __name__ == '__main__':
    summed_feat = create_cam()
    
    # mat = [[1,2,3,4],
           # [2,3,4,5]
            # ]
    
    im = draw_block()
    im.draw_all(summed_feat)
    im.write_jpg()