print('Hello, world!')

import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from config import Config
from utils import set_devices
from torch.optim import lr_scheduler
from dataset_init import create_dataset
from model import ft_net
from metrics import TripletLoss
from metrics import SFR_tri_loss
from utils import adjust_lr_exp
from utils import to_scalar
from utils import ExtractFeature
from utils import AverageMeter
from utils import load_ckpt
from utils import save_ckpt
import torch.nn as nn
from TripletFolder import TripletFolder
from torchvision import datasets, models, transforms
from sklearn import linear_model
import numpy as np


def update_dict(y, d, x, n_components):
    """
    使用KSVD更新字典的过程
    """
    for i in range(n_components):
        index = np.nonzero(x[i, :])[0]
        if len(index) == 0:
            continue
        d[:, i] = 0
        r = (y - np.dot(d, x))[:, index]
        u, s, v = np.linalg.svd(r, full_matrices=False)
        d[:, i] = u[:, 0].T
        x[i, index] = s[0] * v[0, :]
    return d, x

'''
def DSR_loss(X, Y, W, same_id):
    alpha = 1 if same_id else -1
    beta = 0.005
    DSR_loss = alpha * torch.norm(X-Y.mm(W)) + beta * torch.norm(W, p=1)
    return DSR_loss
''' 

'''
def DSR_loss(X, Y, W, same_id): #用Y来重构X, batch计算，返回loss平均值
    
    alpha = 0.3 if same_id else -0.3   # 当为1和-1时,neg pair的loss为负值.改小之后正负甚至有些随机
    beta = 0.4
    DSR_loss_sum = 0
    batch_size = len(X)
    for i in range(batch_size):
        DSR_loss = alpha * torch.norm(X[i]-Y[i].mm(W[i])) + beta * torch.norm(W[i], p=1)
        DSR_loss_sum += DSR_loss
    #if DSR_loss_sum < 0:
    #    DSR_loss_sum = -DSR_loss_sum
    return DSR_loss_sum/batch_size
'''       

def DSR_loss_L2(X, Y, same_id): #用Y来重构X, batch计算，返回loss平均值
    alpha = 0.3 if same_id else -0.3   # 当为1和-1时,neg pair的loss为负值.改小之后正负甚至有些随机
    beta = 0.001
    
    I = beta * torch.eye((torch.matmul(Y[0].t(), Y[0])).size(0))
    I = I.cuda()
    X= X.cuda()
    Y=Y.cuda()
    DSR_loss_sum = 0
    batch_size = len(X)
    for i in range(batch_size):
       Proj_M = torch.matmul(torch.inverse(torch.matmul(Y[i].t(), Y[i])+I), Y[i].t())
       Proj_M.cuda()
       Proj_M.detach()
       w = torch.matmul(Proj_M, X[i])
       w = w.cuda()
       w.detach()
       #a = torch.matmul(Y[i], w) - X[i]
       #DSR_loss = alpha * torch.pow(a,2).sum(0).sqrt() + beta * torch.norm(w, p=2)
       DSR_loss = alpha * torch.norm(X[i]-Y[i].mm(w)) + beta * torch.norm(w, p=2)   # 10 -> 0.4
       DSR_loss_sum += DSR_loss
    if DSR_loss_sum < 0:
        #DSR_loss_sum = DSR_loss_sum + 200*batch_size
        DSR_loss_sum = -DSR_loss_sum
    return DSR_loss_sum/batch_size
    
    
    
    
    
def main():
    cfg = Config()
    gpu_ids = '0'
    str_ids = gpu_ids.split(',')
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
    use_gpu = torch.cuda.is_available()
    
    
    
    
    '''
    train_set = create_dataset(**cfg.train_set_kwargs)
    '''  
    test_set = create_dataset(**cfg.test_set_kwargs)

    transform_train_list = [
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((320,120)),
        # transforms.Pad(10),
        # transforms.RandomCrop((256,128)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.486, 0.459, 0.408], [0.229, 0.224, 0.225])
        ]
    test_dataset = TripletFolder('../Dataset/market1501/Market-1501-v15.09.15/pytorch/train_all', transforms.Compose( transform_train_list ))
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64,shuffle=True, num_workers=8)

    
    model = ft_net(class_num=751, mode='train_SFR')
    if use_gpu:
         model = model.cuda()  
    ckp_path = '../model_ckpts/base/epoch_8 loss_0.0020782903844195574.pth'
    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint['state_dict'])     
    
    ignored_params = list(map(id, model.model.fc.parameters() )) + list(map(id, model.classifier.parameters() ))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    opt_lr = 0.0006
    optimizer = optim.SGD([
             {'params': base_params, 'lr': 0.1*opt_lr},
             {'params': model.model.fc.parameters(), 'lr': opt_lr},
             {'params': model.classifier.parameters(), 'lr': opt_lr},
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)
        
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[1,3], gamma=0.1)    
   
    count = 0
    for data in dataloader: # max = 2588
        inputs, labels, pos, neg = data
        if use_gpu:
           inputs = Variable(inputs.cuda())
           pos = Variable(pos.cuda())
           neg = Variable(neg.cuda())
           labels = Variable(labels.cuda())
        if count > 100:     # 约等于3000个pair
            break
        # 稀疏编码
        
        for epoch in range(5):
            in_global_feat, in_spatial_feat = model(inputs)
            pos_global_feat, pos_spatial_feat = model(pos)
            # neg_global_feat, neg_spatial_feat = model(neg)
            #print('pos pair feature extracted.')
            '''
            W_set = []
            for j in range(len(pos_spatial_feat)):   # batch_size
                # W = linear_model.orthogonal_mp(dictionary, Y)
                W = linear_model.orthogonal_mp(in_spatial_feat[j].detach().numpy(), pos_spatial_feat[j].detach().numpy())
                W = torch.from_numpy(W).float()
                
                W_set.append(W)
                #print(j, 'coded')
            
            #print(len(W_set[0]), len(W_set[0][0]))  # 128 128
            #print('---all pos pair in batch sparse coded.---')
            '''
            same_id = True
            optimizer.zero_grad()
            loss = DSR_loss_L2(pos_spatial_feat, in_spatial_feat, same_id)
            #DSR_loss(X, Y, W, same_id)
            #loss = DSR_loss(pos_spatial_feat, in_spatial_feat, W_set, same_id)
                      
            print('pos pair phase loss in epoch [', epoch, ']/count[',count,']: ',loss.item())
            loss.backward()
            optimizer.step()
            
        for epoch in range(5):
            in_global_feat, in_spatial_feat = model(inputs)
            #pos_global_feat, pos_spatial_feat = model(pos)
            neg_global_feat, neg_spatial_feat = model(neg)
            #print('neg pair feature extracted.')
            '''
            #print('--epoch:[', epoch, ']--')
            W_set = []
            for j in range(len(neg_spatial_feat)):   # batch_size
                # W = linear_model.orthogonal_mp(dictionary, Y)
                W = linear_model.orthogonal_mp(in_spatial_feat[j].detach().numpy(), neg_spatial_feat[j].detach().numpy())
                W = torch.from_numpy(W).float()
                W_set.append(W)
                #print(j, 'coded')
            #print(len(W_set[0]), len(W_set[0][0]))  # 128 128
            #print('---all neg pair in batch sparse coded.---')
            '''
            same_id = False
            optimizer.zero_grad()
            # DSR_loss(X, Y, W, same_id)
            loss = DSR_loss_L2(neg_spatial_feat, in_spatial_feat, same_id)
            print('neg pair phase loss in epoch [', epoch, ']/count[',count,']: ',loss.item())
            loss.backward()
            optimizer.step()
            
        if (count%40== 0):
            save_network(model,optimizer, count, loss)
        count += 1
   

def save_network(model,optimizer, epoch, loss):
    save_path = '../model_ckpts/DSR_finetune/epoch_{} loss_{}'.format(epoch, loss.item())+'.pth'
    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer':optimizer.state_dict(),'loss': loss}, save_path)

  
    
if __name__ == '__main__':
    main()