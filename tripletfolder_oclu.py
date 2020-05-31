from torchvision import datasets
import os
import numpy as np
import random
import torch

class TripletFolder(datasets.ImageFolder):

    def __init__(self, root, transform):
        super(TripletFolder, self).__init__(root, transform)
        targets = np.asarray([s[1] for s in self.samples])
        self.targets = targets
  
    def __getitem__(self, index):
        path, target = self.samples[index]

        sample = self.loader(path)
        #print(type(sample))
        width = sample.size[0]  # 图片大小
        height = sample.size[1]
        #print(width, height)
        
        crop_list = [(0, 0, width, height/2),   #上半身
                     (0, height/2, width, height),  #下半身
                     (0, 0, width/2, height),     #左半身
                     (width/2, 0, width, height) #右半身
                    ]
        par_sample = sample.crop(crop_list[random.randint(0,3)])
        # print(crop_list[2])
        #par_sample = sample.crop(crop_list[1])
        #print('Are you sane?')
        # sample.save("hol.jpg")
        # par_sample.save("par.jpg")
        
        if self.transform is not None:
            sample = self.transform(sample)
            par_sample = self.transform(par_sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
             
        return sample, par_sample, target
 