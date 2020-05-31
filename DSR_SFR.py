print('Hello, world!')

import time
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from config import Config
from utils import set_devices
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
from metrics import SFR_tri_loss

def main():
    gpu_ids = '3'
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
    train_set = create_dataset(**cfg.train_set_kwargs)
    test_set = create_dataset(**cfg.test_set_kwargs)
    
    model = ft_net(class_num=751, mode='train_SFR')
    if use_gpu:
         model = model.cuda()
    optimizer = optim.Adam(model.parameters(),lr=cfg.base_lr, weight_decay=0.0005)
    # criterion = nn.CrossEntropyLoss()  
    Margin = 0.2
    print("margin: ", Margin)
    tri_loss = TripletLoss(margin=Margin)


    start_epoch=0
    scheduler_step_OK = True
    resume = False
    if resume:
        ckp_path = '../model_ckpts/aug/base/epoch_6 loss_0.005337492062590352.pth'
        checkpoint = torch.load(ckp_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])#优化参数
        start_epoch = checkpoint['epoch']#epoch，可以用于更新学习率等
        
        scheduler_step_OK = False

    print('-------')
    decay_begin_epoch = 151
    total_epochs = 30
    for ep in range(start_epoch, total_epochs):
        ep_st = time.time()

        prec_meter = AverageMeter(); sm_meter = AverageMeter();
        dist_ap_meter = AverageMeter(); dist_an_meter = AverageMeter();
        loss_meter = AverageMeter()
        
        cur_epoch_done = False
        step = 0
        while not cur_epoch_done:
          step += 1
          step_st = time.time()
          # 此处的cur_epoch_done指完整的一个train_set通过网络
          ims, im_names, labels, mirrored, cur_epoch_done = train_set.next_batch()
          #print(labels)  
          if use_gpu:
              ims_var = Variable(torch.from_numpy(ims).float().cuda())
              labels_t = Variable(torch.from_numpy(labels).long().cuda())
          #print(labels_t)
          globalFeature, spatialFeature = model(ims_var)
          loss, p_inds, n_inds, dist_ap, dist_an, dist_mat = SFR_tri_loss(tri_loss, globalFeature, spatialFeature, 
                                                                        labels_t, spatial_train=True, normalize_feature=False)
          #print('\tstep:',step, 'loss: ', loss)

          optimizer.zero_grad()  # 清空梯度缓存
          loss.backward()    # 反向传播
          optimizer.step()   # 更新权重
          
          ############
          # Step Log #
          ###########

          # precision
          prec = (dist_an > dist_ap).data.float().mean()
          # the proportion of triplets that satisfy margin
          sm = (dist_an > dist_ap + cfg.margin).data.float().mean()
          # average (anchor, positive) distance
          d_ap = dist_ap.data.mean()
          # average (anchor, negative) distance
          d_an = dist_an.data.mean()

          prec_meter.update(prec)
          sm_meter.update(sm)
          dist_ap_meter.update(d_ap)
          dist_an_meter.update(d_an)
          loss_meter.update(to_scalar(loss))
     
          # step log
          if step % 1 == 0:
            time_log = '\tStep {}/Ep {}, {:.2f}s'.format(step, ep + 1, time.time() - step_st, )
            tri_log = (', prec {:.2%}, sm {:.2%}, d_ap {:.4f}, d_an {:.4f}, loss {:.4f}'.format(prec_meter.val, sm_meter.val,dist_ap_meter.val, dist_an_meter.val,loss_meter.val))
            print(time_log+tri_log)
            
        # epoch log
        time_log = 'Ep {}, {:.2f}s'.format(ep + 1, time.time() - ep_st)
        tri_log = (', prec {:.2%}, sm {:.2%}, d_ap {:.4f}, d_an {:.4f}, loss {:.4f}'.format(prec_meter.avg, sm_meter.avg, dist_ap_meter.avg, dist_an_meter.avg, loss_meter.avg, ))
        
        # save models
        if ep % 2 == 0:
           save_network(model,optimizer, ep, loss)
 

    print('done!')
    
def save_network(model,optimizer, epoch, loss):
    save_path = '../model_ckpts/SFR/margin0.2/epoch_{} loss_{}'.format(epoch, loss)+'.pth'
    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer':optimizer.state_dict(),'loss': loss}, save_path)
    
    
if __name__ == '__main__':
    main()