import os
import pickle
import time
import torch
from contextlib import contextmanager
from torch.autograd import Variable


def may_transfer_modules_optims(modules_and_or_optims, device_id=-1):
  """Transfer optimizers/modules to cpu or specified gpu.
  Args:
    modules_and_or_optims: A list, which members are either torch.nn.optimizer 
      or torch.nn.Module or None.
    device_id: gpu id, or -1 which means transferring to cpu
  """
  for item in modules_and_or_optims:
    if isinstance(item, torch.optim.Optimizer):
      transfer_optim_state(item.state, device_id=device_id)
    elif isinstance(item, torch.nn.Module):
      if device_id == -1:
        item.cpu()
      else:
        item.cuda(device=device_id)
    elif item is not None:
      print('[Warning] Invalid type {}'.format(item.__class__.__name__))

class TransferVarTensor(object):
  """Return a copy of the input Variable or Tensor on specified device."""

  def __init__(self, device_id=-1):
    self.device_id = device_id

  def __call__(self, var_or_tensor):
    return var_or_tensor.cpu() if self.device_id == -1 \
      else var_or_tensor.cuda(self.device_id)

class TransferModulesOptims(object):
  """Transfer optimizers/modules to cpu or specified gpu."""

  def __init__(self, device_id=-1):
    self.device_id = device_id

  def __call__(self, modules_and_or_optims):
    may_transfer_modules_optims(modules_and_or_optims, self.device_id)

def set_devices(sys_device_ids):
  """
  It sets some GPUs to be visible and returns some wrappers to transferring 
  Variables/Tensors and Modules/Optimizers.
  Args:
    sys_device_ids: a tuple; which GPUs to use
      e.g.  sys_device_ids = (), only use cpu
            sys_device_ids = (3,), use the 4th gpu
            sys_device_ids = (0, 1, 2, 3,), use first 4 gpus
            sys_device_ids = (0, 2, 4,), use the 1st, 3rd and 5th gpus
  Returns:
    TVT: a `TransferVarTensor` callable
    TMO: a `TransferModulesOptims` callable
  """
  # Set the CUDA_VISIBLE_DEVICES environment variable
  import os
  if sys_device_ids==():
    print("Using CPU")
    
  visible_devices = ''
  for i in sys_device_ids:
    visible_devices += '{}, '.format(i)
  os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
  # Return wrappers.
  # Models and user defined Variables/Tensors would be transferred to the
  # first device.
  device_id = 0 if len(sys_device_ids) > 0 else -1
  TVT = TransferVarTensor(device_id)
  TMO = TransferModulesOptims(device_id)
  return TVT, TMO

def load_ckpt(modules_optims, ckpt_file, remove_fc = False, load_to_cpu=True, verbose=True):
  """Load state_dict's of modules/optimizers from file.
  Args:
    modules_optims: A list, which members are either torch.nn.optimizer 
      or torch.nn.Module.
    ckpt_file: The file path.
    load_to_cpu: Boolean. Whether to transform tensors in modules/optimizers 
      to cpu type.
  """
  map_location = (lambda storage, loc: storage) if load_to_cpu else None
  ckpt = torch.load(ckpt_file, map_location=map_location)
  
  model_dict = ckpt['state_dicts'][0]
  optimizer_dict = ckpt['state_dicts'][1]

  if remove_fc:
    model_dict = {k:v for k, v in model_dict.items() if not k.startswith('base.fc.')}
    modules_optims[0].load_state_dict(model_dict)
    model = modules_optims[0]
    if verbose:
        print('Resume/load from ckpt {} \nresume epoch {}, fc removed.'.format(ckpt_file, ckpt['ep']))
    return model, ckpt['ep']
  else:
    # print(model_dict.keys())
    modules_optims[0].load_state_dict(model_dict)
    modules_optims[1].load_state_dict(optimizer_dict)
           
    model = modules_optims[0]
    optimizer = modules_optims[1]
      
    if verbose:
        print('Resume/load from ckpt {} \nresume epoch {}.'.format(ckpt_file, ckpt['ep']))
    return model, optimizer, ckpt['ep']

def save_ckpt(modules_optims, ep, loss, save_dir):
  state_dicts = [m.state_dict() for m in modules_optims]
  ckpt = dict(state_dicts=state_dicts,
              ep=ep)
  
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)   
  ckpt_file = os.path.join(save_dir,  'epoch' + str(ep)+'loss'+str(loss)+'.pth')
  torch.save(ckpt, ckpt_file)
  print('ckpt saved. save_dir:', save_dir)

def load_state_dict(model, src_state_dict):
  """Copy parameters and buffers from `src_state_dict` into `model` and its 
  descendants. The `src_state_dict.keys()` NEED NOT exactly match 
  `model.state_dict().keys()`. For dict key mismatch, just
  skip it; for copying error, just output warnings and proceed.

  Arguments:
    model: A torch.nn.Module object. 
    src_state_dict (dict): A dict containing parameters and persistent buffers.
  Note:
    This is modified from torch.nn.modules.module.load_state_dict(), to make
    the warnings and errors more detailed.
  """
  from torch.nn import Parameter

  dest_state_dict = model.state_dict()
  for name, param in src_state_dict.items():
    if name not in dest_state_dict:
      continue
    if isinstance(param, Parameter):
      # backwards compatibility for serialized parameters
      param = param.data
    try:
      dest_state_dict[name].copy_(param)
    except Exception as msg:
      print("Warning: Error occurs when copying '{}': {}"
            .format(name, str(msg)))

  src_missing = set(dest_state_dict.keys()) - set(src_state_dict.keys())
  if len(src_missing) > 0:
    print("Keys not found in source state_dict: ")
    for n in src_missing:
      print('\t', n)

  dest_missing = set(src_state_dict.keys()) - set(dest_state_dict.keys())
  if len(dest_missing) > 0:
    print("Keys not found in destination state_dict: ")
    for n in dest_missing:
      print('\t', n)


def is_iterable(obj):
  return hasattr(obj, '__len__')

def may_set_mode(maybe_modules, mode):
  """maybe_modules: an object or a list of objects."""
  assert mode in ['train', 'eval']
  if not is_iterable(maybe_modules):
    maybe_modules = [maybe_modules]
  for m in maybe_modules:
    if isinstance(m, torch.nn.Module):
      if mode == 'train':
        m.train()
      else:
        m.eval()

def load_pickle(path):
  assert os.path.exists(path)
  with open(path, 'rb') as f:
    ret = pickle.load(f)
  return ret
  
@contextmanager
def measure_time(enter_msg, verbose=True):
  if verbose:
    st = time.time()
    print(enter_msg)
  yield
  if verbose:
    print('Done, {:.2f}s'.format(time.time() - st))
    
def adjust_lr_exp(optimizer, base_lr, ep, total_ep, start_decay_at_ep):
  if ep < start_decay_at_ep:
    return

  for g in optimizer.param_groups:
    g['lr'] = (base_lr * (0.001 ** (float(ep + 1 - start_decay_at_ep)
                                    / (total_ep + 1 - start_decay_at_ep))))
  print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))

def to_scalar(vt):
  """Transform a length-1 pytorch Variable or Tensor to scalar. 
  Suppose tx is a torch Tensor with shape tx.size() = torch.Size([1]), 
  then npx = tx.cpu().numpy() has shape (1,), not 1."""
  if isinstance(vt, Variable):
    return vt.data.cpu().numpy().flatten()[0]
  if torch.is_tensor(vt):
    return vt.cpu().numpy().flatten()[0]
  raise TypeError('Input should be a variable or tensor')
  
class ExtractFeature(object):
  """A function to be called in the val/test set, to extract features.
  Args:
    TVT: A callable to transfer images to specific device.
  """
  
  def __init__(self, model, device='GPU'):
    self.model = model
    self.device = device

  def __call__(self, ims):
    old_train_eval_model = self.model.training
    # Set eval mode.
    # Force all BN layers to use global mean and variance, also disable
    # dropout.
    self.model.eval()
    if self.device == 'GPU':
        ims = Variable(torch.from_numpy(ims).float().cuda())
    else:
        ims = Variable(self.TVT(torch.from_numpy(ims).float()))
    
    globalFeature, spatialFeature = self.model(ims)

    globalFeature = globalFeature.data.cpu().numpy()
    spatialFeature = spatialFeature.data.cpu().numpy()
    '''
    globalFeature, SpatialFeature1, SpatialFeature2 = self.model(ims)
    globalFeature = globalFeature.data.cpu().numpy()
    SpatialFeature1 = SpatialFeature1.data.cpu().numpy()
    SpatialFeature2 = SpatialFeature2.data.cpu().numpy()
    '''
    #print(len(spatialFeature))
    # Restore the model to its old train/eval mode.
    self.model.train(old_train_eval_model)
    return globalFeature, spatialFeature


class AverageMeter(object):
  """Modified from Tong Xiao's open-reid. 
  Computes and stores the average and current value"""

  def __init__(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = float(self.sum) / (self.count + 1e-20)
    
def PR_curve(distmat, gallery_ids, query_ids, name):
  import numpy as np

  # Ensure numpy array
  assert isinstance(distmat, np.ndarray)
  
  query_num, gallery_num = len(distmat), len(distmat[0])
  
  indices = np.argsort(distmat, axis=1)
  gallery_ids_pre = gallery_ids[indices]
  
  precision = np.zeros([query_num, gallery_num])
  recall = np.zeros([query_num, gallery_num])
  
  for i in range(query_num):
    match = 0; matched = False;
    for j in range(gallery_num):
        if gallery_ids_pre[i][j] == i+1:
            match += 1
            matched = True               
        precision[i][j] = match / (j+1)
        if matched:
            recall[i][j] = 1
            
  ave_recall = np.mean(recall, axis=0) 
  ave_precision = np.mean(precision, axis=0) 
  
  return ave_precision, ave_recall