import argparse
import numpy as np

class Config(object):
    def __init__(self):
       parser = argparse.ArgumentParser() 
       parser.add_argument('-d', '--sys_device_ids', type=eval, default=('1,'))
       parser.add_argument('--train_dataset', type=str, default='market1501',choices=['market1501', 'cuhk03', 'duke', 'combined'])
       parser.add_argument('--partial_dataset', type=str, default='Partial_iLIDS',choices=['Partial_REID', 'Partial_iLIDS', 'holistic'])
       parser.add_argument('--total_epochs', type=int, default=100)
       parser.add_argument('--base_lr', type=float, default=1.5e-4)
       parser.add_argument('--spatial_train', type=bool, default=False)
       parser.add_argument('--margin', type=float, default=0.45)
       
       parser.add_argument('--only_test', type=bool, default=False)
       parser.add_argument('--resume', type=bool, default=False)
       
       parser.add_argument('--code_version', type=str, default='crossEntrpyLoss_no_PP')
       
       args = parser.parse_args()
       self.sys_device_ids = args.sys_device_ids
       self.train_dataset = args.train_dataset
       self.partial_dataset = args.partial_dataset
       self.total_epochs = args.total_epochs
       self.base_lr = args.base_lr
       self.spatial_train = args.spatial_train
       self.margin = args.margin
       self.only_test = args.only_test
       self.resume = args.resume
       
       self.code_version = args.code_version
       
       self.train_batch_size = 32  
       self.test_batch_size = 32
       self.val_batch_size = 32
       self.ims_per_id = 4     
       
       
       dataset_kwargs = dict(
           resize_h_w=(320, 120),
           scale=True,     # Whether to scale by 1/255
           im_mean=[0.486, 0.459, 0.408],
           im_std=[0.229, 0.224, 0.225],
           batch_dims='NCHW',
           num_prefetch_threads=2)
        
       prng = np.random
       self.train_set_kwargs = dict(
           name=self.train_dataset,  #default='market1501'
           part='trainval',                     #使得返回的是Trainset类
           ids_per_batch=self.train_batch_size, 
           ims_per_id=self.ims_per_id,
           final_batch=True,
           shuffle=True,
           crop_prob=0,
           crop_ratio=1,
           mirror_type='random', 
           prng=prng)
       self.train_set_kwargs.update(dataset_kwargs)
       
       prng = np.random
       self.val_set_kwargs = dict(
           name='',
           part='val',                                              
           batch_size=self.test_batch_size,
           final_batch=True,
           shuffle=False,
           mirror_type=None, 
           prng=prng)
       self.val_set_kwargs.update(dataset_kwargs)
       
       prng = np.random
       self.test_set_kwargs = dict(
           name=self.partial_dataset,
           part='test',  
           batch_size=self.test_batch_size,
           final_batch=True,
           shuffle=False,
           mirror_type=None, 
           prng=prng)
       self.test_set_kwargs.update(dataset_kwargs)
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       