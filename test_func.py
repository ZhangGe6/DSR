
from tripletfolder_oclu import TripletFolder
from torchvision import transforms
import os
import torch

transform_train_list = [
        # V2: 预处理与base保持一致
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((320, 120)),
        #transforms.Pad(10),
        #transforms.RandomCrop((256,128)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.486, 0.459, 0.408], [0.229, 0.224, 0.225])
        ]

#print(transform_train_list)
data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    
}


data_dir = '../Dataset/market1501/Market-1501-v15.09.15/pytorch'
train_all = '_all'
image_datasets = {}
image_datasets['train'] = TripletFolder(os.path.join(data_dir, 'train_all'),
                                          data_transforms['train'])
                                                                                  
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=True, num_workers=1)  for x in ['train']}

i =0
for i in range(1):
    for data in dataloaders['train']:
        inputs,p,  labels = data
        break
    print()
    