# -*- coding: utf-8 -*-
from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import copy
from PIL import Image
import time
import os
#from reid_sampler import StratifiedSampler
from model import ft_net, ft_net_dense, PCB, verif_net
from random_erasing import RandomErasing
#from tripletfolder import TripletFolder
import yaml
from shutil import copyfile

version =  torch.__version__

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='ft_net', type=str, help='output model name')
parser.add_argument('--data_dir',default='../Dataset/market1501/Market-1501-v15.09.15/pytorch',type=str, help='training dir path')
#parser.add_argument('--data_dir',default='../Dataset/DukeMTMC-reID/DukeMTMC-reID/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--alpha', default=1.0, type=float, help='alpha')      # V2: 2
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', default=False, type=bool, help='use PCB+ResNet50' )
parser.add_argument('--resume', default=False, type=bool, help='Whether to resume')
parser.add_argument('--fp16', default=True, type=bool, help='use float16 instead of float32, which will save about 50% memory' )
opt = parser.parse_args()

data_dir = opt.data_dir
print(data_dir)
name = opt.name

str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
print(gpu_ids[0])


######################################################################
# Load Data
# --------- 
#

transform_train_list = [
        # V2: 预处理与base保持一致
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((320, 120)),
        # transforms.Pad(10),
        # transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.486, 0.459, 0.408], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        transforms.Resize(size=(320, 120),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.486, 0.459, 0.408], [0.229, 0.224, 0.225])
        ]


if opt.erasing_p>0:
    print('Random Erasing 0.5')
    transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

#print(transform_train_list)
data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
}

 
train_all = ''
if opt.train_all:
     train_all = '_all'

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train_all'),
                                          data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])

batch = {}

class_names = image_datasets['train'].classes
print(len(class_names))
class_vector = [s[1] for s in image_datasets['train'].samples]
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize, 
                                             shuffle=True, num_workers=8)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

use_gpu = torch.cuda.is_available()

since = time.time()
#inputs, classes, pos, pos_classes = next(iter(dataloaders['train']))


######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    last_margin = 0.0
    
    start_epoch=0
    scheduler_step_OK = True
    
    if opt.resume:
        ckp_path = '../model_ckpts/aug/base/epoch_6 loss_0.005337492062590352.pth'
        checkpoint = torch.load(ckp_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])#优化参数
        start_epoch = checkpoint['epoch']#epoch，可以用于更新学习率等
        
        scheduler_step_OK = False
    
    for epoch in range(start_epoch+1, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                #scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_verif_loss = 0.0
            running_corrects = 0.0
            running_verif_corrects = 0.0
            # Iterate over data.
            loader_ID = 0
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                #print(len(labels))
                #print(labels)
                now_batch_size,c,h,w = inputs.shape
                
                if now_batch_size<opt.batchsize: # next epoch
                    continue

                if use_gpu:
                    #print("Use GPU")
                    inputs = Variable(inputs.cuda())
                    #pos = Variable(pos.cuda())
                    #neg = Variable(neg.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                #print(len(outputs))  #32x751 
                #print(len(f))        #32x512
               
                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)
               
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                # statistics
                if int(version[0])>0 or int(version[2]) > 3: # for the new version like 0.4.0 and 0.5.0
                    running_loss += loss.item() #* opt.batchsize
                    
                else :  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] 
                    
                running_corrects += float(torch.sum(preds == labels.data))
                
                loader_log ='\tloader[{}] of epoch[{}], loss: {}'.format(loader_ID, epoch, loss.item())
                print(loader_log)   
                mylog = open('log/loader_base.txt', mode = 'a',encoding='utf-8')
                print(loader_log, file=mylog)
                
                loader_ID += 1
                
            if not scheduler_step_OK:
                for i in range(start_epoch):
                    scheduler.step()
                scheduler_step_OK = True
            else:
                scheduler.step()
                
            datasize = dataset_sizes['train']//opt.batchsize * opt.batchsize
            epoch_loss = running_loss / datasize
           
            epoch_acc = running_corrects / datasize
           
            print('{} Loss: {:.4f}   Acc: {:.4f}  lr: {}'.format(
                phase, epoch_loss,  epoch_acc, optimizer.param_groups[0]['lr']))
            
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)
            # deep copy the model
            
            if (epoch%2 == 0):
                save_network(model,optimizer, epoch, epoch_loss)
            


######################################################################
# Save model
#---------------------------
def save_network(model,optimizer, epoch, loss):
    save_path = '../model_ckpts/aug/RF/epoch_{} loss_{}'.format(epoch, loss)+'.pth'
    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer':optimizer.state_dict(),'loss': loss}, save_path)


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#


model = ft_net(len(class_names), 'train_CNN')
model_verif = verif_net()

if use_gpu:
    model = model.cuda()
    model_verif = model_verif.cuda()

criterion = nn.CrossEntropyLoss()

if not opt.PCB:
    print('DD')
    print('save model per 2 epoch')
    
    ignored_params = list(map(id, model.model.fc.parameters() )) + list(map(id, model.classifier.parameters() ))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.1*opt.lr},
             {'params': model.model.fc.parameters(), 'lr': opt.lr},
             {'params': model.classifier.parameters(), 'lr': opt.lr},
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    
    #optimizer_ft = optim.Adam(model.parameters(),lr=1.5e-4, weight_decay=0.0005)   
    
else:
    print('note')
    ignored_params = list(map(id, model.model.fc.parameters() ))
    ignored_params += (list(map(id, model.classifier0.parameters() )) 
                     +list(map(id, model.classifier1.parameters() ))
                     +list(map(id, model.classifier2.parameters() ))
                     +list(map(id, model.classifier3.parameters() ))
                     +list(map(id, model.classifier4.parameters() ))
                     +list(map(id, model.classifier5.parameters() ))
                     #+list(map(id, model.classifier6.parameters() ))
                     #+list(map(id, model.classifier7.parameters() ))
                      )
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.001},
             {'params': model.model.fc.parameters(), 'lr': 0.01},
             {'params': model.classifier0.parameters(), 'lr': 0.01},
             {'params': model.classifier1.parameters(), 'lr': 0.01},
             {'params': model.classifier2.parameters(), 'lr': 0.01},
             {'params': model.classifier3.parameters(), 'lr': 0.01},
             {'params': model.classifier4.parameters(), 'lr': 0.01},
             {'params': model.classifier5.parameters(), 'lr': 0.01},
             #{'params': model.classifier6.parameters(), 'lr': 0.01},
             #{'params': model.classifier7.parameters(), 'lr': 0.01}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)

#exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[40,60], gamma=0.1)
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[10,60], gamma=0.1)
######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU. 
#

dir_name = '../model_ckpts/oclu/'
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
#record every run
copyfile('./DSR_oclu.py', dir_name+'/DSR_oclu.py')
copyfile('./model.py', dir_name+'/model.py')
copyfile('./tripletfolder_oclu.py', dir_name+'/tripletfolder_oclu.py')

# save opts
with open('%s/opts.yaml'%dir_name,'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)
    
model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=60)

