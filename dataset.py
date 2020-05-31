import numpy as np
import os
from collections import defaultdict
from dataset_utils import PreProcessIm
from dataset_utils import PreProcessPartialIm
from dataset_utils import Prefetcher
from dataset_utils import parse_im_name
from distance import compute_dist
from distance import dsr_dist_L2, dsr_dist_L1, dsr_dist_L2_weight
from metrics import cmc, mean_ap


from utils import measure_time, PR_curve
from PIL import Image
from sklearn import linear_model
import time


class Dataset(object):
  """The core elements of a dataset.    
  Args:
    final_batch: bool. The last batch may not be complete, if to abandon this 
      batch, set 'final_batch' to False.
  """

  def __init__(
      self,
      dataset_size=None,
      batch_size=None,
      final_batch=True,
      shuffle=True,
      num_prefetch_threads=1,
      prng=np.random,
      **pre_process_im_kwargs):

    self.pre_process_im = PreProcessIm(
      prng=prng,
      **pre_process_im_kwargs)
    self.pre_process_im1 = PreProcessPartialIm(
      prng=prng,
      **pre_process_im_kwargs)
    self.prefetcher = Prefetcher(
      self.get_sample,
      dataset_size,
      batch_size,
      final_batch=final_batch,
      num_threads=num_prefetch_threads)

    self.shuffle = shuffle
    self.epoch_done = True
    self.prng = prng

  def set_mirror_type(self, mirror_type):
    self.pre_process_im.set_mirror_type(mirror_type)

  def get_sample(self, ptr):
    """Get one sample to put to queue."""
    raise NotImplementedError

  def next_batch(self):
    """Get a batch from the queue."""
    raise NotImplementedError

  def set_batch_size(self, batch_size):
    """You can change batch size, had better at the beginning of a new epoch.
    """
    self.prefetcher.set_batch_size(batch_size)
    self.epoch_done = True

  def stop_prefetching_threads(self):
    """This can be called to stop threads, e.g. after finishing using the 
    dataset, or when existing the python main program."""
    self.prefetcher.stop()

class TrainSet(Dataset):
  """Training set for triplet loss.
  Args:
    ids2labels: a dict mapping ids to labels
  """
  def __init__(
      self,
      im_dir=None,
      im_names=None,
      ids2labels=None,
      ids_per_batch=None,
      ims_per_id=None,
      **kwargs):

    # The im dir of all images
    self.im_dir = im_dir
    self.im_names = im_names
    self.ids2labels = ids2labels
    self.ids_per_batch = ids_per_batch
    self.ims_per_id = ims_per_id

    im_ids = [parse_im_name(name, 'id') for name in im_names]
    self.ids_to_im_inds = defaultdict(list)
    for ind, id in enumerate(im_ids):
      self.ids_to_im_inds[id].append(ind)
    self.ids = self.ids_to_im_inds.keys()

    super(TrainSet, self).__init__(dataset_size=len(self.ids), batch_size=ids_per_batch, **kwargs)

  def get_sample(self, ptr):
    """Here one sample means several images (and labels etc) of one id.
    Returns:
      ims: a list of images
    """
    inds = self.ids_to_im_inds[list(self.ids)[ptr]]
    if len(inds) < self.ims_per_id:
      inds = np.random.choice(inds, self.ims_per_id, replace=True)
    else:
      inds = np.random.choice(inds, self.ims_per_id, replace=False)
    im_names = [self.im_names[ind] for ind in inds]
    ims = [np.asarray(Image.open(os.path.join(self.im_dir, name)))
           for name in im_names]
    ims, mirrored = zip(*[self.pre_process_im(im) for im in ims])
    labels = [self.ids2labels[list(self.ids)[ptr]] for _ in range(self.ims_per_id)]

    return ims, im_names, labels, mirrored

  def next_batch(self):
    """Next batch of images and labels.
    Returns:
      ims: numpy array with shape [N, H, W, C] or [N, C, H, W], N >= 1
      img_names: a numpy array of image names, len(img_names) >= 1
      labels: a numpy array of image labels, len(labels) >= 1
      mirrored: a numpy array of booleans, whether the images are mirrored
      self.epoch_done: whether the epoch is over

    """
    # Start enqueuing and other preparation at the beginning of an epoch.
    if self.epoch_done and self.shuffle:
      np.random.shuffle(list(self.ids))
    samples, self.epoch_done = self.prefetcher.next_batch()
    im_list, im_names, labels, mirrored = zip(*samples)
    # t = time.time()
    # Transform the list into a numpy array with shape [N, ...]
    ims = np.stack(np.concatenate(im_list))
    # print '---stacking time {:.4f}s'.format(time.time() - t)
    im_names = np.concatenate(im_names)
    labels = np.concatenate(labels)
    mirrored = np.concatenate(mirrored)
    return ims, im_names, labels, mirrored, self.epoch_done

class Partial_REID_dataset(Dataset):
    def __init__(self, im_dir=None, im_names=None, marks=None, extract_feat_func=None, separate_camera_set=None, single_gallery_shot=None, first_match_break=None, **kwargs):
        #print('pre_process_im_kwargs: ',kwargs)
        super(Partial_REID_dataset, self).__init__(dataset_size=600, **kwargs) #??
        
        self.im_dir = im_dir
        
        self.extract_feat_func = extract_feat_func
        self.separate_camera_set = separate_camera_set
        self.single_gallery_shot = single_gallery_shot
        self.first_match_break = first_match_break        
        
    def set_feat_func(self, extract_feat_func):
        self.extract_feat_func = extract_feat_func
    def get_sample(self, ptr):
        print('get_sample func is called in [Partial_REID_dataset]!')
    def next_batch(self):
        if self.epoch_done and self.shuffle:
            self.prng.shuffle(self.im_names)
        samples, self.epoch_done = self.prefetcher.next_batch()       # prefetcher()从Dateset类继承而来
        im_list, ids, cams, im_names, marks = zip(*samples)
        
        # Transform the list into a numpy array with shape [N, ...]
        ims = np.stack(im_list, axis=0)
        ids = np.array(ids)
        cams = np.array(cams)
        im_names = np.array(im_names)
        marks = np.array(marks)
        return ims, ids, cams, im_names, marks, self.epoch_done
        
    def extract_feat(self, labels):      # used in eval():
        globalFeature, spatialFeature = [], []
        for i in range(0, len(labels)):
            im_path = os.path.join(self.im_dir, labels[i])
            im = np.asarray(Image.open(im_path))
            im, _ = self.pre_process_im(im)   # pre_process_im()是从Dataset类继承来的函数
            imgs = np.zeros((1, im.ndim, im.shape[1], im.shape[2]))
            imgs[0, :, :, :] = im
            globalFeature_, spatialFeature_ = self.extract_feat_func(imgs)
            
            #print(len(globalFeature_), len(globalFeature_[0]))    # 1 2048
            #print(len(spatialFeature_), len(spatialFeature_[0]), len(spatialFeature_[0][0]))    # 1 2048 52
            globalFeature.append(globalFeature_)
            spatialFeature.append(spatialFeature_)
            #print(labels[i], 'extracted.')
        return globalFeature, spatialFeature
     
     
    def eval(self, normalize_feat=False, mode='no_global'):
        print('-----Testing-----')
        
        with measure_time('Extracting Probe feature...', verbose=True):
            probe_label_path = os.path.join(self.im_dir, 'Probe.txt')
            fh = open(probe_label_path, 'r')
            labels = []

            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()           # ['partial_body_images/001_001.jpg']
                labels.append(words[0])       # 'partial_body_images/001_001.jpg'

            # print(len(labels))
            Probe_globalFeature, Probe_spatialFeature = self.extract_feat(labels)
        Probe_globalFeature = np.vstack(Probe_globalFeature)
        Probe_spatialFeature = np.vstack(Probe_spatialFeature)
        print(Probe_spatialFeature.shape)
        #np.save("./features/Probe_globalFeature"+str(time.time()),Probe_globalFeature)
        #np.save("./features/Probe_spatialFeature"+str(time.time()),Probe_spatialFeature)
        
        with measure_time('Extracting Gallery feature...', verbose=True):
            gallery_label_path = os.path.join(self.im_dir, 'Gallery.txt')
            fh = open(gallery_label_path, 'r')
            labels = []
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()           # ['partial_body_images/001_001.jpg']
                labels.append(words[0])       # 'partial_body_images/001_001.jpg'

            Gallery_globalFeature, Gallery_spatialFeature = self.extract_feat(labels)
        Gallery_globalFeature = np.vstack(Gallery_globalFeature)
        Gallery_spatialFeature = np.vstack(Gallery_spatialFeature)
        #np.save("./features/Gallery_globalFeature"+str(time.time()),Gallery_globalFeature)
        #np.save("./features/Gallery_spatialFeature"+str(time.time()),Gallery_spatialFeature)   
        
       
       # query, gallery, multi-query indices
        query_ids, gallery_ids, query_cams, gallery_cams = [], [], [], []
        for i in range(1, 61):        # 60 = 300/5，一共300张图，每人5张图，从1开始计数
        #for i in range(0, 3):        # 60 = 300/5，一共300张图，每人5张图，从1开始计数
            query_ids.append(i)
            gallery_ids.append(i)
            query_cams.append(0)
            gallery_cams.append(1)
        query_ids = np.hstack(query_ids)
        query_cams = np.hstack(query_cams)
        gallery_ids = np.hstack(gallery_ids)
        gallery_cams = np.hstack(gallery_cams)
        '''
        print('start')
        q_g_spatial_dist = dsr_dist_L1(Probe_spatialFeature, Gallery_spatialFeature)
        print('dis done')
        PR_curve(q_g_spatial_dist, gallery_ids, query_ids, 1)
        print('curve done')
        '''
        # A helper function just for avoiding code duplication.
        def compute_score(dist_mat):
            # Compute mean AP
            mAP = mean_ap(
                distmat=dist_mat,
                query_ids=query_ids, gallery_ids=gallery_ids,
                query_cams=query_cams, gallery_cams=gallery_cams)
            # Compute CMC scores
            cmc_scores = cmc(
                distmat=dist_mat,
                query_ids=query_ids, gallery_ids=gallery_ids,
                query_cams=query_cams, gallery_cams=gallery_cams,
                separate_camera_set=self.separate_camera_set,
                single_gallery_shot=self.single_gallery_shot,
                first_match_break=self.first_match_break,
                topk=10)
            return mAP, cmc_scores

        def print_cmc_scores(cmc):          
            print("Rank-1:", cmc[0]*100, '%')
            print("Rank-3:", cmc[2]*100, '%')
            print("Rank-5:", cmc[4]*100, '%')


        ################
        # Single shot #
        ################
        # query-gallery distance
        Q_G_global_dist = []
        Q_G_spatial_dist = []
        Q_Q_global_dist = []; G_G_global_dist = []
        Q_Q_spatial_dist = []; G_G_spatial_dist = []
        with measure_time('Computing distance...', verbose=True):
            # single shot: 对每个人的第j张照片
            for j in range(0, 5):
                # 采集每个人的第j张照片的Probe信息
                SpatialProbe = []  # 最终包含60个人的第j张照片的Probe集合
                for kk in range(j, 300, 5):
                    SpatialProbe.append(Probe_spatialFeature[kk])

                # global dis 用    
                marks = []
                for iii in range(0, 300):
                    if (iii % 5 == j):
                        marks.append(1)
                    else:
                        marks.append(0)
                marks = np.hstack(marks)
                p_inds = (marks == 1)  # 0/1 list -> bool list
                
                # 采集每个人的第i张照片的Gallery信息
                for i in range(0, 5):
                    print(j, i)
                    SpatialGallery = []
                    for k in range(i, 300, 5):
                        SpatialGallery.append(Gallery_spatialFeature[k])  # 最终包含60个人的第i张照片的Gallery集合
                        
                    # global dis 用    
                    marks = []
                    for iii in range(0, 300):
                        if (iii % 5 == i):
                            marks.append(1)
                        else:
                            marks.append(0)
                    marks = np.hstack(marks)
                    g_inds = marks == 1
                    
                    
                    # 一个q_g_global_dist/q_g_spatial_dist即对应一个“邻接矩阵”
                    q_g_global_dist = compute_dist(Probe_globalFeature[p_inds], Gallery_globalFeature[g_inds], type='euclidean')
                    q_g_spatial_dist = dsr_dist_L2(SpatialProbe, SpatialGallery)
                    
                    
                    #re-rank用
                    # q_q_global_dist = compute_dist(Probe_globalFeature[p_inds], Probe_globalFeature[p_inds], type='euclidean')
                    # g_g_global_dist = compute_dist(Gallery_globalFeature[g_inds], Gallery_globalFeature[g_inds], type='euclidean')
                    # q_q_spatial_dist = dsr_dist_L2(SpatialProbe, SpatialProbe)
                    # g_g_spatial_dist = dsr_dist_L2(SpatialProbe, SpatialGallery)
                    
                    Q_G_global_dist.append(q_g_global_dist)     # len(Q_G_global_dist) = 5*5 每一个元素是一个60*60的矩阵
                    Q_G_spatial_dist.append(q_g_spatial_dist)
                    
                    # Q_Q_global_dist.append(q_q_global_dist); G_G_global_dist.append(g_g_global_dist)
                    # Q_Q_spatial_dist.append(q_q_spatial_dist); G_G_spatial_dist.append(g_g_spatial_dist)
                    
					
            # np.save("./distances/Q_G_global_dist",Q_G_global_dist)
            # np.save("./distances/Q_G_spatial_dist",Q_G_spatial_dist)
            # np.save("./distances/Q_Q_global_dist",Q_Q_global_dist)
            # np.save("./distances/G_G_global_dist",G_G_global_dist)
            # np.save("./distances/Q_Q_spatial_dist",Q_Q_spatial_dist)
            # np.save("./distances/G_G_spatial_dist",G_G_spatial_dist)    
            
        with measure_time('Computing scores...', verbose=True):
            print("Test Dataset: Partial REID")
            
            alpha_rank1, alpha_mAP = [], []
            if mode == 'with_global':
                iter_num = 11 
            else:
                iter_num = 1
                
            for lam in range(0, iter_num):
                mAP1 = []
                cmc_scores1 = []
                precision1 = []
                recall1 = []
                weight = lam * 0.1
                print('----cur_weight:', weight, '----')
                for i in range(0, len(Q_G_global_dist)):         
                    q_g_dist = weight * Q_G_global_dist[i] + (1 - weight) * Q_G_spatial_dist[i]  #这个i便很好理解了
                    
                    mAP, cmc_scores = compute_score(q_g_dist)
                    precision, recall = PR_curve(q_g_dist, gallery_ids, query_ids, i)
                    
                    mAP1.append(mAP); cmc_scores1.append(cmc_scores)
                    precision1.append(precision); recall1.append(recall)
                       
                # sum(mAP1)/25 means 对这25个single shot做平均,返回一个值
                # sum(cmc_scores1) / 25同理，返回一个topk向量
                av_mAP = sum(mAP1) / 25
                av_cmc = sum(cmc_scores1) / 25
                ave_precision = sum(precision1) / 25
                ave_recall = sum(recall1) / 25
                
                print('mAP = ', av_mAP)
                print_cmc_scores(av_cmc)
                
                if mode == 'with_global':
                    # if lam == 0:
                        # np.savetxt("./PR/ave_recall"+str(time.time())+".txt",ave_recall)
                        # np.savetxt("./PR/ave_precision"+str(time.time())+".txt",ave_precision)
                        # mylog = open('./PR/cmc'+str(time.time())+'.txt', mode = 'a',encoding='utf-8')
                        # print(av_cmc, file=mylog)
                        
                    # np.savetxt("./PR/mutiLamda/ave_recall_lamda="+str(lam)+".txt",ave_recall)
                    # np.savetxt("./PR/mutiLamda/ave_precision_lamda="+str(lam)+".txt",ave_precision)
                    
                    # cmc_log = open('./PR/mutiLamda/cmc_lamda='+str(lam)+'.txt', mode = 'a',encoding='utf-8')
                    # print(av_cmc, file=cmc_log)   
                    #cmc_log.close()
                    alpha_rank1.append(av_cmc[0])
                    alpha_mAP.append(av_mAP)
                    if lam == 10:
                        np.savetxt("./PR/alpha_rank1.txt",alpha_rank1)
                        np.savetxt("./PR/alpha_mAP.txt",alpha_mAP)  
                    if lam == 0:
                        np.savetxt("./PR/ave_recall"+str(time.time())+".txt",ave_recall)
                        np.savetxt("./PR/ave_precision"+str(time.time())+".txt",ave_precision)
                        mylog = open('./PR/cmc'+str(time.time())+'.txt', mode = 'a',encoding='utf-8')
                        print(av_cmc, file=mylog)
                    

                if not (mode == 'with_global'):
                    np.savetxt("./PR/ave_recall"+str(time.time())+".txt",ave_recall)
                    np.savetxt("./PR/ave_precision"+str(time.time())+".txt",ave_precision)
                    mylog = open('./PR/cmc'+str(time.time())+'.txt', mode = 'a',encoding='utf-8')
                    print(av_cmc, file=mylog)
                    

class Partial_iLIDS_dataset(Dataset):
    def __init__(self, im_dir=None, im_names=None, marks=None, extract_feat_func=None, separate_camera_set=None, single_gallery_shot=None, first_match_break=None, **kwargs):
        #print('pre_process_im_kwargs: ',kwargs)
        super(Partial_iLIDS_dataset, self).__init__(dataset_size=119, **kwargs) #??
        
        self.im_dir = im_dir
        
        self.extract_feat_func = extract_feat_func
        self.separate_camera_set = separate_camera_set
        self.single_gallery_shot = single_gallery_shot
        self.first_match_break = first_match_break        
        
    def set_feat_func(self, extract_feat_func):
        self.extract_feat_func = extract_feat_func
    def get_sample(self, ptr):
        print('get_sample func is called in [Partial_iLDS_dataset]!')
    def next_batch(self):
        if self.epoch_done and self.shuffle:
            self.prng.shuffle(self.im_names)
        samples, self.epoch_done = self.prefetcher.next_batch()       # prefetcher()从Dateset类继承而来
        im_list, ids, cams, im_names, marks = zip(*samples)
        
        # Transform the list into a numpy array with shape [N, ...]
        ims = np.stack(im_list, axis=0)
        ids = np.array(ids)
        cams = np.array(cams)
        im_names = np.array(im_names)
        marks = np.array(marks)
        return ims, ids, cams, im_names, marks, self.epoch_done
        
    def extract_feat(self, labels):      # used in eval():
        globalFeature, spatialFeature = [], []
        for i in range(0, len(labels)):
            im_path = os.path.join(self.im_dir, labels[i])
            im = np.asarray(Image.open(im_path))
            im, _ = self.pre_process_im(im)   # pre_process_im()是从Dataset类继承来的函数
            imgs = np.zeros((1, im.ndim, im.shape[1], im.shape[2]))
            imgs[0, :, :, :] = im
            globalFeature_, spatialFeature_ = self.extract_feat_func(imgs)
            
            #print(len(globalFeature_), len(globalFeature_[0]))    # 1 2048
            #print(len(spatialFeature_), len(spatialFeature_[0]), len(spatialFeature_[0][0]))    # 1 2048 52
            globalFeature.append(globalFeature_)
            spatialFeature.append(spatialFeature_)
            #print(labels[i], 'extracted.')
        return globalFeature, spatialFeature
     
     
    def eval(self, normalize_feat=False, mode='no_global'):
        print('-----Testing, Dataset Partial iLIDS-----')
        
        with measure_time('Extracting Probe feature...', verbose=True):
            probe_label_path = os.path.join(self.im_dir, 'Probe.txt')
            fh = open(probe_label_path, 'r')
            labels = []

            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()           # ['partial_body_images/001_001.jpg']
                labels.append(words[0])       # 'partial_body_images/001_001.jpg'

            # print(len(labels))
            Probe_globalFeature, Probe_spatialFeature = self.extract_feat(labels)
        Probe_globalFeature = np.vstack(Probe_globalFeature)
        Probe_spatialFeature = np.vstack(Probe_spatialFeature)
        print(Probe_spatialFeature.shape)
        #np.save("./features/Probe_globalFeature"+str(time.time()),Probe_globalFeature)
        #np.save("./features/Probe_spatialFeature"+str(time.time()),Probe_spatialFeature)
        
        with measure_time('Extracting Gallery feature...', verbose=True):
            gallery_label_path = os.path.join(self.im_dir, 'Gallery.txt')
            fh = open(gallery_label_path, 'r')
            labels = []
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()           # ['partial_body_images/001_001.jpg']
                labels.append(words[0])       # 'partial_body_images/001_001.jpg'

            Gallery_globalFeature, Gallery_spatialFeature = self.extract_feat(labels)
        Gallery_globalFeature = np.vstack(Gallery_globalFeature)
        Gallery_spatialFeature = np.vstack(Gallery_spatialFeature)
        #np.save("./features/Gallery_globalFeature"+str(time.time()),Gallery_globalFeature)
        #np.save("./features/Gallery_spatialFeature"+str(time.time()),Gallery_spatialFeature)   
        
       
       # query, gallery, multi-query indices
        query_ids, gallery_ids, query_cams, gallery_cams = [], [], [], []
        for i in range(1, 120):        # 60 = 300/5，一共300张图，每人5张图，从1开始计数
        #for i in range(0, 3):        # 60 = 300/5，一共300张图，每人5张图，从1开始计数
            query_ids.append(i)
            gallery_ids.append(i)
            query_cams.append(0)
            gallery_cams.append(1)
        query_ids = np.hstack(query_ids)
        query_cams = np.hstack(query_cams)
        gallery_ids = np.hstack(gallery_ids)
        gallery_cams = np.hstack(gallery_cams)
        '''
        print('start')
        q_g_spatial_dist = dsr_dist_L1(Probe_spatialFeature, Gallery_spatialFeature)
        print('dis done')
        PR_curve(q_g_spatial_dist, gallery_ids, query_ids, 1)
        print('curve done')
        '''
        # A helper function just for avoiding code duplication.
        def compute_score(dist_mat):
            # Compute mean AP
            mAP = mean_ap(
                distmat=dist_mat,
                query_ids=query_ids, gallery_ids=gallery_ids,
                query_cams=query_cams, gallery_cams=gallery_cams)
            # Compute CMC scores
            cmc_scores = cmc(
                distmat=dist_mat,
                query_ids=query_ids, gallery_ids=gallery_ids,
                query_cams=query_cams, gallery_cams=gallery_cams,
                separate_camera_set=self.separate_camera_set,
                single_gallery_shot=self.single_gallery_shot,
                first_match_break=self.first_match_break,
                topk=10)
            return mAP, cmc_scores

        def print_cmc_scores(cmc):          
            print("Rank-1:", cmc[0]*100, '%')
            print("Rank-3:", cmc[2]*100, '%')
            print("Rank-5:", cmc[4]*100, '%')



        ################
        # Single shot #
        ################
        # query-gallery distance
        with measure_time('Computing distance...', verbose=True):
            q_g_global_dist = compute_dist(Probe_globalFeature, Gallery_globalFeature, type='euclidean')
            q_g_spatial_dist = dsr_dist_L2(Probe_spatialFeature, Gallery_spatialFeature)
            for lam in range(0, 1):
                mAP1 = []
                cmc_scores1 = []
                weight = lam * 0.1

                
                q_g_dist = (1 - weight) * q_g_spatial_dist + weight * q_g_global_dist
                mAP, cmc_scores = compute_score(q_g_dist)
                #cmc_scores1.append(cmc_scores)

                
                #print('mAP = ', av_mAP)
                print_cmc_scores(cmc_scores)
                
                # if mode == 'with_global':
                    # if lam == 0:
                        # np.savetxt("./PR/ave_recall"+str(time.time())+".txt",ave_recall)
                        # np.savetxt("./PR/ave_precision"+str(time.time())+".txt",ave_precision)
                        # mylog = open('./PR/cmc'+str(time.time())+'.txt', mode = 'a',encoding='utf-8')
                        # print(av_cmc, file=mylog)
                        
                    # np.savetxt("./PR/mutiLamda/ave_recall_lamda="+str(lam)+".txt",ave_recall)
                    # np.savetxt("./PR/mutiLamda/ave_precision_lamda="+str(lam)+".txt",ave_precision)
                    
                    # cmc_log = open('./PR/mutiLamda/cmc_lamda='+str(lam)+'.txt', mode = 'a',encoding='utf-8')
                    # print(av_cmc, file=cmc_log)   

                    # #cmc_log.close()

                # if not (mode == 'with_global'):
                    # np.savetxt("./PR/ave_recall"+str(time.time())+".txt",ave_recall)
                    # np.savetxt("./PR/ave_precision"+str(time.time())+".txt",ave_precision)
                    # mylog = open('./PR/cmc'+str(time.time())+'.txt', mode = 'a',encoding='utf-8')
                    # print(av_cmc, file=mylog)
                                   
    