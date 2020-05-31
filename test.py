print('Hello, world!')

import time
import torch
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from config import Config
from utils import set_devices
from dataset_init import create_dataset
from metrics import TripletLoss
from metrics import SFR_tri_loss
from utils import adjust_lr_exp
from utils import to_scalar
from utils import ExtractFeature
from utils import AverageMeter
from utils import load_ckpt
from utils import save_ckpt
import torch.nn as nn
from metrics import SFR_tri_loss

from model import ft_net
from model import PCB, PPCB

def main():
    gpu_ids = '1'
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
    
    cfg = Config()
    test_set = create_dataset(**cfg.test_set_kwargs)
    
    
           
    alg = 'aug'
    path_pre = '../model_ckpts/aug/RF/'
    ckp_path_list  = [
                      
                       'epoch_6 loss_0.0059664649981083255',
                       'epoch_8 loss_0.002700634092083293',
                       'epoch_10 loss_0.0013723163490902892',
                       'epoch_12 loss_0.0008662273844538054',
                       'epoch_14 loss_0.0007970516701311906'
                       
                       
                       
                       ]
 
    if alg == 'IV':  
        for ckp_path in ckp_path_list:
            ckp_path = path_pre+ckp_path+'.pth'
            model = ft_net(class_num=751, mode='test'); 
            print('load model from: ', ckp_path)
            checkpoint = torch.load(ckp_path)
            model.load_state_dict(checkpoint['state_dict_iden'])
            
            test_set.set_feat_func(ExtractFeature(model, device='GPU'))
            test_set.eval(normalize_feat=False, mode='no_global')
        
    elif alg == 'aug':
        for ckp_path in ckp_path_list:
            ckp_path = path_pre+ckp_path+'.pth' 
            model = ft_net(class_num=751, mode='test')
            if use_gpu:
                model = model.cuda()
            
            print('load model from: ', ckp_path)
            checkpoint = torch.load(ckp_path)
            model.load_state_dict(checkpoint['state_dict'])
            
            test_set.set_feat_func(ExtractFeature(model, device='GPU'))
            test_set.eval(normalize_feat=False, mode='no_global')
        
    elif alg == 'PCB': 
        for ckp_path in ckp_path_list:
            ckp_path = path_pre+ckp_path+'.pth'
            model = PCB(class_num=751, mode='test')
            print('load model from: ', ckp_path)
            checkpoint = torch.load(ckp_path)
            model.load_state_dict(checkpoint['state_dict'])
            if use_gpu:
                model = model.cuda()
                
            test_set.set_feat_func(ExtractFeature(model, device='GPU'))
            test_set.eval(normalize_feat=False, mode='no_global')
        
    elif alg == 'PPCB': 
        for ckp_path in ckp_path_list:
            ckp_path = path_pre+ckp_path+'.pth'
            model = PPCB(class_num=751, mode='test')
            print('load model from: ', ckp_path)
            checkpoint = torch.load(ckp_path)
            model.load_state_dict(checkpoint['state_dict'])
            if use_gpu:
                model = model.cuda()
                
            test_set.set_feat_func(ExtractFeature(model, device='GPU'))
            test_set.eval(normalize_feat=False, mode='no_global')           
            
    elif alg == 'multi_DSR_dist':
        for ckp_path in ckp_path_list:
            ckp_path = path_pre+ckp_path+'.pth'
            model = ft_net(class_num=751, mode='multi_DSR_dist')
            print('load model from: ', ckp_path)
            checkpoint = torch.load(ckp_path)
            model.load_state_dict(checkpoint['state_dict'])
            
            test_set.set_feat_func(ExtractFeature(model, device='GPU'))
            test_set.eval(normalize_feat=False, mode='no_global')            
            
    elif alg == 'alpha':
        for ckp_path in ckp_path_list:
            ckp_path = path_pre+ckp_path+'.pth'
            model = ft_net(class_num=751, mode='test')
            if use_gpu:
                model = model.cuda()            
            print('load model from: ', ckp_path)
            checkpoint = torch.load(ckp_path)
            model.load_state_dict(checkpoint['state_dict'])
            
            test_set.set_feat_func(ExtractFeature(model, device='GPU'))
            test_set.eval(normalize_feat=False, mode='with_global')       
        
    print('done!')
    
if __name__ == '__main__':
    main()