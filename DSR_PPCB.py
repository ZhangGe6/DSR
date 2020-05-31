# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from PIL import Image
import time
import os
from model import ft_net, PCB, PPCB
#from random_erasing import RandomErasing
import yaml
import math
from shutil import copyfile

version =  torch.__version__


######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='1', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='PCB', type=str, help='output model name')
parser.add_argument('--data_dir', default='../Dataset/market1501/Market-1501-v15.09.15/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', default=False, type=bool, help='use densenet121' )
parser.add_argument('--use_NAS', default=False, type=bool, help='use NAS' )
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--PCB', default=True, help='use PCB+ResNet50' )
parser.add_argument('--resume', default=False, help='use PCB+ResNet50' )
opt = parser.parse_args()


data_dir = opt.data_dir
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


if opt.PCB:
    transform_train_list = [
        #transforms.Resize((384,192), interpolation=3),
        transforms.Resize((320, 120)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.486, 0.459, 0.408], [0.229, 0.224, 0.225])
        ]
    transform_val_list = [
        transforms.Resize(size=(384,192),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.erasing_p>0:
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
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                          data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=8, pin_memory=True) # 8 workers may work faster
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()
since = time.time()
#inputs, classes = next(iter(dataloaders['train']))
#print(time.time()-since)
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

    #best_model_wts = model.state_dict()
    #best_acc = 0.0
    #warm_up = 0.1 # We start from the 0.1*lrRate
    #warm_iteration = round(dataset_sizes['train']/opt.batchsize)*opt.warm_epoch # first 5 epoch
    start_epoch=0
    scheduler_step_OK = True
    
    if opt.resume:
        ckp_path = '../model_ckpts/PCB/epoch_1 loss_20.482138268654186.pth'
        checkpoint = torch.load(ckp_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer']) #优化参数
        start_epoch = checkpoint['epoch'] #epoch，可以用于更新学习率等
        
        scheduler_step_OK = False
    
    for epoch in range(start_epoch+1, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            loader_ID = 0
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                now_batch_size,c,h,w = inputs.shape
                if now_batch_size<opt.batchsize: # skip the last batch
                    continue
                #print(inputs.shape)
                # wrap them in Variable
                
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                
                optimizer.zero_grad()

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                if not opt.PCB:
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                else:
                    part = {}
                    sm = nn.Softmax(dim=1)
                    num_part = 1+3+6
                    for i in range(num_part):
                        part[i] = outputs[i]

                    score = sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) +sm(part[4]) +sm(part[5])
                    _, preds = torch.max(score.data, 1)

                    loss = criterion(part[0], labels)
                    for i in range(num_part-1):
                        loss += criterion(part[i+1], labels)
                '''
                # backward + optimize only if in training phase
                if epoch<opt.warm_epoch and phase == 'train': 
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss *= warm_up
                '''
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                if int(version[0])>0 or int(version[2]) > 3: # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * now_batch_size
                else :  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))
                print('\tloader[{}] of epoch[{}], loss: {}'.format(loader_ID, epoch, loss.item()))   #32 
                loader_ID += 1
                
            if not scheduler_step_OK:
                for i in range(start_epoch):
                    scheduler.step()
                scheduler_step_OK = True
            else:
                scheduler.step()
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc) 
            
            if epoch%2 == 0:
                save_network(model,optimizer, epoch, epoch_loss)
            #draw_curve(epoch)


######################################################################
# Save model
#---------------------------
def save_network(model,optimizer, epoch, loss):
    save_path = '../model_ckpts/PPCB/epoch_{} loss_{}'.format(epoch, loss)+'.pth'
    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer':optimizer.state_dict(),'loss': loss}, save_path)

######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

# if opt.use_dense:
    # model = ft_net_dense(len(class_names), opt.droprate)
# elif opt.use_NAS:
    # model = ft_net_NAS(len(class_names), opt.droprate)
# else:
    # model = ft_net(len(class_names), opt.droprate, opt.stride)

if opt.PCB:
    print('PPCB')
    model = PPCB(len(class_names), mode='PPCB')
if use_gpu:
    model = model.cuda()
    
opt.nclasses = len(class_names)
#print(model)

if not opt.PCB:
    ignored_params = list(map(id, model.classifier.parameters() ))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.1*opt.lr},
             {'params': model.classifier.parameters(), 'lr': opt.lr}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)
else:
    ignored_params = list(map(id, model.model.fc.parameters() ))
    ignored_params += (list(map(id, model.classifier0.parameters() )) 
                     +list(map(id, model.classifier1.parameters() ))
                     +list(map(id, model.classifier2.parameters() ))
                     +list(map(id, model.classifier3.parameters() ))
                     +list(map(id, model.classifier4.parameters() ))
                     +list(map(id, model.classifier5.parameters() ))
                     +list(map(id, model.classifier6.parameters() ))
                     +list(map(id, model.classifier7.parameters() ))
                     +list(map(id, model.classifier8.parameters() ))
                     +list(map(id, model.classifier9.parameters() ))
                     # +list(map(id, model.classifier10.parameters() ))
                     # +list(map(id, model.classifier11.parameters() ))
                     # +list(map(id, model.classifier12.parameters() ))                     
                     #+list(map(id, model.classifier7.parameters() ))
                      )
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.1*opt.lr},
             {'params': model.model.fc.parameters(), 'lr': opt.lr},
             {'params': model.classifier0.parameters(), 'lr': opt.lr},
             {'params': model.classifier1.parameters(), 'lr': opt.lr},
             {'params': model.classifier2.parameters(), 'lr': opt.lr},
             {'params': model.classifier3.parameters(), 'lr': opt.lr},
             {'params': model.classifier4.parameters(), 'lr': opt.lr},
             {'params': model.classifier5.parameters(), 'lr': opt.lr},
             {'params': model.classifier6.parameters(), 'lr': opt.lr},
             {'params': model.classifier7.parameters(), 'lr': opt.lr},
             {'params': model.classifier8.parameters(), 'lr': opt.lr},
             {'params': model.classifier9.parameters(), 'lr': opt.lr},
             # {'params': model.classifier10.parameters(), 'lr': opt.lr},
             # {'params': model.classifier11.parameters(), 'lr': opt.lr},
             # {'params': model.classifier12.parameters(), 'lr': opt.lr},             
             #{'params': model.classifier6.parameters(), 'lr': 0.01},
             #{'params': model.classifier7.parameters(), 'lr': 0.01}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU. 
#
dir_name = '../model_ckpts/PPCB/'
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
#record every run
copyfile('./DSR_PPCB.py', dir_name+'/DSR_PPCB.py')
copyfile('./model.py', dir_name+'/model.py')

# save opts
with open('%s/opts.yaml'%dir_name,'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

criterion = nn.CrossEntropyLoss()

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=60)

