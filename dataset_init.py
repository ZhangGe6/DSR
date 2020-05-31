import numpy as np
from utils import load_pickle
import numpy as np
import time
from dataset import TrainSet
from dataset import Partial_REID_dataset, Partial_iLIDS_dataset

def create_dataset(
    name=None,
    part=None,
    **kwargs):

  assert name in ['market1501', 'cuhk03', 'duke', 'combined', 'Partial_REID', 'Partial_iLIDS'], \
    "Unsupported Dataset {}".format(name)
  assert part in ['trainval', 'train', 'val', 'test'], \
    "Unsupported Dataset Part {}".format(part)
    
  if name == 'market1501':
    im_dir = '../Dataset/market1501/images'
    partition_file = '../Dataset/market1501/partitions.pkl'
    
    partitions = load_pickle(partition_file)
    im_names = partitions['{}_im_names'.format(part)]
    
  elif name == 'Partial_REID':
    im_dir = '../Dataset/PartialREID/'
  elif name == 'Partial_iLIDS':
    im_dir = '../Dataset/Partial_iLIDS/'
  
  # Use standard Market1501 CMC settings for all datasets here.
  cmc_kwargs = dict(separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
                    
  if part == 'trainval':
    ids2labels = partitions['trainval_ids2labels']
    
    ret_set = TrainSet(
      im_dir=im_dir,
      im_names=im_names,
      ids2labels=ids2labels,
      **kwargs)
      
  elif part == 'train':
    ids2labels = partitions['train_ids2labels']

    ret_set = TrainSet(
      im_dir=im_dir,
      im_names=im_names,
      ids2labels=ids2labels,
      **kwargs)
       
  elif part == 'val':
    marks = partitions['val_marks']
    kwargs.update(cmc_kwargs)

    ret_set = TestSet(
      im_dir=im_dir,
      im_names=im_names,
      marks=marks,
      **kwargs)
  
  elif part == 'test':
    kwargs.update(cmc_kwargs)
    if name == 'Partial_REID':
        ret_set = Partial_REID_dataset(im_dir=im_dir, **kwargs)
        #print(kwargs)
    elif name =='Partial_iLIDS':
        ret_set = Partial_iLIDS_dataset(im_dir=im_dir, **kwargs)
        
  if part in ['trainval', 'train']:
    print('-'*10)
    print('Train dataset created: [{}] [{}] part'.format(name, part))
    print('Number of ID: ', len(ids2labels))
    print('Number of images: ', len(im_names))
  elif part in ['test']:
    print('-'*10)
    print('Test dataset created:[{}] for [{}]'.format(name, 'test'))
  
  return ret_set
