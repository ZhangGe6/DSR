from __future__ import print_function
import torch
from torch.autograd import Variable
import time
from torch import nn
from collections import defaultdict
import numpy as np
from sklearn.metrics import average_precision_score


def normalize(x, axis=-1):
  """Normalizing to unit length along the specified dimension.
  Args:
    x: pytorch Variable
  Returns:
    x: pytorch Variable, same shape as input      
  """
  x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
  return x

def euclidean_dist(x, y):
  """
  Args:
    x: pytorch Variable, with shape [m, d]
    y: pytorch Variable, with shape [n, d]
  Returns:
    dist: pytorch Variable, with shape [m, n]
  """
  m, n = x.size(0), y.size(0)
  xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
  yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
  dist = xx + yy
  dist.addmm_(1, -2, x, y.t())
  dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
  return dist

def dsr_dist(x, y):
  """
  Args:
    x: pytorch Variable, with shape [m, d]
    y: pytorch Variable, with shape [n, d]
  Returns:
    dist: pytorch Variable, with shape [m, n]
  """
  #start = time.time()
  m, n = x.size(0), y.size(0)
  kappa = 0.001
  dist = Variable(torch.zeros(m,n))
  dist = dist.cuda()
  T = kappa * Variable(torch.eye(160))
  T = T.cuda()
  T.detach()

  for i in range(0, m):
    Proj_M = torch.matmul(torch.inverse(torch.matmul(x[i,::].t(), x[i,::])+T), x[i,::].t())
    Proj_M.detach()
    for j in range(0, n):
      w = torch.matmul(Proj_M, y[j,::])
      w.detach()
      a = torch.matmul(x[i,::], w) - y[j,::]
      dist[i,j] = torch.pow(a,2).sum(0).sqrt().mean()
  return dist
  
def hard_example_mining(dist_mat, labels, return_inds=False):
  """For each anchor, find the hardest positive and negative sample.
  Args:
    dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
    labels: pytorch LongTensor, with shape [N]
    return_inds: whether to return the indices. Save time if `False`(?)
  Returns:
    dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
    dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    p_inds: pytorch LongTensor, with shape [N]; 
      indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
    n_inds: pytorch LongTensor, with shape [N];
      indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
  NOTE: Only consider the case in which all labels have same num of samples, 
    thus we can cope with all anchors in parallel.
  """

  assert len(dist_mat.size()) == 2
  assert dist_mat.size(0) == dist_mat.size(1)
  dist_mat = dist_mat.cuda(); labels = labels.cuda()
  
  N = dist_mat.size(0);

  # shape [N, N]
  is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
  is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

  # `dist_ap` means distance(anchor, positive)
  # both `dist_ap` and `relative_p_inds` with shape [N, 1]
  dist_ap, relative_p_inds = torch.max(
    dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
  # `dist_an` means distance(anchor, negative)
  # both `dist_an` and `relative_n_inds` with shape [N, 1]
  dist_an, relative_n_inds = torch.min(
    dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
  # shape [N]
  dist_ap = dist_ap.squeeze(1)
  dist_an = dist_an.squeeze(1)

  if return_inds:
    # shape [N, N]
    ind = (labels.new().resize_as_(labels)
           .copy_(torch.arange(0, N).long())
           .unsqueeze( 0).expand(N, N))
    # shape [N, 1]
    p_inds = torch.gather(
      ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
    n_inds = torch.gather(
      ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
    # shape [N]
    p_inds = p_inds.squeeze(1)
    n_inds = n_inds.squeeze(1)
    return dist_ap, dist_an, p_inds, n_inds

  return dist_ap, dist_an

def DSR_L(x, y, p_inds, n_inds):
  """
  Args:
    x: pytorch Variable, with shape [m, d]
    y: pytorch Variable, with shape [n, d]
  Returns:
    dist: pytorch Variable, with shape [m, n]
  """
  #start = time.time()
  m = y.size(0)

  kappa = 0.001
  dist_p = Variable(torch.zeros(m, 1))
  dist_n = Variable(torch.zeros(m, 1))
  dist_p = dist_p.cuda()
  dist_n = dist_n.cuda()
  T = kappa * Variable(torch.eye(160))
  T = T.cuda()
  T.detach()

  for i in range(0, m):
    Proj_M1 = torch.matmul(torch.inverse(torch.matmul(x[p_inds[i],:,:].t(), x[p_inds[i],:,:])+T), x[p_inds[i],:,:].t())
    Proj_M1.detach()

    Proj_M2 = torch.matmul(torch.inverse(torch.matmul(x[n_inds[i],:,:].t(), x[n_inds[i],:,:])+T), x[n_inds[i],:,:].t())
    Proj_M2.detach()
    w1 = torch.matmul(Proj_M1, y[i,::])
    w1.detach()
    w2 = torch.matmul(Proj_M2, y[i,::])
    w2.detach()
    a1 = torch.matmul(x[p_inds[i],:,:], w1) - y[i,::]
    a2 = torch.matmul(x[n_inds[i], :, :], w2) - y[i, ::]
    dist_p[i, 0] = torch.pow(a1,2).sum(0).sqrt().mean()
    dist_n[i, 0] = torch.pow(a2, 2).sum(0).sqrt().mean()

  dist_n = dist_n.squeeze(1)
  dist_p = dist_p.squeeze(1)
  return dist_n, dist_p

def SFR_tri_loss(tri_loss, global_feat, spatial_feat, labels, spatial_train, normalize_feature=True):
  """
  Args:
    tri_loss: a `TripletLoss` object
    global_feat: pytorch Variable, shape [N, C]
    labels: pytorch LongTensor, with shape [N]
    normalize_feature: whether to normalize feature to unit length along the 
      Channel dimension
  Returns:
    loss: pytorch Variable, with shape [1]
    p_inds: pytorch LongTensor, with shape [N]; 
      indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
    n_inds: pytorch LongTensor, with shape [N];
      indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    ==================
    For Debugging, etc
    ==================
    dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
    dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    dist_mat: pytorch Variable, pairwise euclidean distance; shape [N, N]
  """
  if normalize_feature:
    global_feat = normalize(global_feat, axis=-1)
  ##spatial_feat = normalize1(spatial_feat, axis=-1)
  # shape [N, N]
  dist_mat = euclidean_dist(global_feat, global_feat)
  # dist_mat_p = dsr_dist(spatial_feat, spatial_feat)
  # dist_mat = dist_mat_g + dist_mat_p
  
  # 使用global distance 来寻找triplet对
  dist_ap, dist_an, p_inds, n_inds = hard_example_mining(
    dist_mat, labels, return_inds=True)
    
  # [if not spatial_train, then only global train]
  # 所以之前都是使用global loss embedded triloss来训练网络,训练没有使用spatial feature
  # 并以此训练好的网络来计算query的spatial feature，这是错位了的。
  # 如果是真的SFR，正确的做法是，spatial_train = True
  if spatial_train:
    dist_n, dist_p = DSR_L(spatial_feat, spatial_feat, p_inds, n_inds)
    dist_ap=dist_p+dist_ap 
    dist_an=dist_an+dist_n  
    
  loss = tri_loss(dist_ap, dist_an)
  return loss, p_inds, n_inds, dist_ap, dist_an, dist_mat
  
class TripletLoss(object):
  """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid). 
  Related Triplet Loss theory can be found in paper 'In Defense of the Triplet 
  Loss for Person Re-Identification'."""
  def __init__(self, margin=None):
    self.margin = margin
    if margin is not None:
      self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    else:
      self.ranking_loss = nn.SoftMarginLoss()

  def __call__(self, dist_ap, dist_an):
    """
    Args:
      dist_ap: pytorch Variable, distance between anchor and positive sample, 
        shape [N]
      dist_an: pytorch Variable, distance between anchor and negative sample, 
        shape [N]
    Returns:
      loss: pytorch Variable, with shape [1]
    """
    y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
    if self.margin is not None:
      loss = self.ranking_loss(dist_an, dist_ap, y)
    else:
      loss = self.ranking_loss(dist_an - dist_ap, y)
    return loss


def _unique_sample(ids_dict, num):
  mask = np.zeros(num, dtype=np.bool)
  for _, indices in ids_dict.items():
    i = np.random.choice(indices)
    mask[i] = True
  return mask

def cmc(
    distmat,
    query_ids=None,
    gallery_ids=None,
    query_cams=None,
    gallery_cams=None,
    topk=100,
    separate_camera_set=False,
    single_gallery_shot=False,
    first_match_break=False,
    average=True):
  """
  Args:
    distmat: numpy array with shape [num_query, num_gallery], the 
      pairwise distance between query and gallery samples
    query_ids: numpy array with shape [num_query]
    gallery_ids: numpy array with shape [num_gallery]
    query_cams: numpy array with shape [num_query]
    gallery_cams: numpy array with shape [num_gallery]
    average: whether to average the results across queries
  Returns:
    If `average` is `False`:
      ret: numpy array with shape [num_query, topk]
      is_valid_query: numpy array with shape [num_query], containing 0's and 
        1's, whether each query is valid or not
    If `average` is `True`:
      numpy array with shape [topk]
  """
  # Ensure numpy array
  assert isinstance(distmat, np.ndarray)
  assert isinstance(query_ids, np.ndarray)
  assert isinstance(gallery_ids, np.ndarray)
  assert isinstance(query_cams, np.ndarray)
  assert isinstance(gallery_cams, np.ndarray)

  m, n = distmat.shape
  # Sort and find correct matches
  indices = np.argsort(distmat, axis=1)
  matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
  # Compute CMC for each query
  ret = np.zeros([m, topk])
  is_valid_query = np.zeros(m)
  num_valid_queries = 0
  for i in range(m):
    # Filter out the same id and same camera
    valid = ((gallery_ids[indices[i]] != query_ids[i]) |
             (gallery_cams[indices[i]] != query_cams[i]))
    if separate_camera_set:
      # Filter out samples from same camera
      valid &= (gallery_cams[indices[i]] != query_cams[i])
    if not np.any(matches[i, valid]): continue
    is_valid_query[i] = 1
    if single_gallery_shot:
      repeat = 100
      gids = gallery_ids[indices[i][valid]]
      inds = np.where(valid)[0]
      ids_dict = defaultdict(list)
      for j, x in zip(inds, gids):
        ids_dict[x].append(j)
    else:
      repeat = 1
    for _ in range(repeat):
      if single_gallery_shot:
        # Randomly choose one instance for each id
        sampled = (valid & _unique_sample(ids_dict, len(valid)))
        index = np.nonzero(matches[i, sampled])[0]
      else:
        index = np.nonzero(matches[i, valid])[0]
      delta = 1. / (len(index) * repeat)
      for j, k in enumerate(index):
        if k - j >= topk: break
        if first_match_break:
          ret[i, k - j] += 1
          break
        ret[i, k - j] += delta
    num_valid_queries += 1
  if num_valid_queries == 0:
    raise RuntimeError("No valid query")
  ret = ret.cumsum(axis=1)
  if average:
    return np.sum(ret, axis=0) / num_valid_queries
  return ret, is_valid_query

def mean_ap(
    distmat,
    query_ids=None,
    gallery_ids=None,
    query_cams=None,
    gallery_cams=None,
    average=True):
  """
  Args:
    distmat: numpy array with shape [num_query, num_gallery], the 
      pairwise distance between query and gallery samples
    query_ids: numpy array with shape [num_query]
    gallery_ids: numpy array with shape [num_gallery]
    query_cams: numpy array with shape [num_query]
    gallery_cams: numpy array with shape [num_gallery]
    average: whether to average the results across queries
  Returns:
    If `average` is `False`:
      ret: numpy array with shape [num_query]
      is_valid_query: numpy array with shape [num_query], containing 0's and 
        1's, whether each query is valid or not
    If `average` is `True`:
      a scalar
  """

  # -------------------------------------------------------------------------
  # The behavior of method `sklearn.average_precision` has changed since version
  # 0.19.
  # Version 0.18.1 has same results as Matlab evaluation code by Zhun Zhong
  # (https://github.com/zhunzhong07/person-re-ranking/
  # blob/master/evaluation/utils/evaluation.m) and by Liang Zheng
  # (http://www.liangzheng.org/Project/project_reid.html).
  # My current awkward solution is sticking to this older version.
  import sklearn

  # Ensure numpy array
  assert isinstance(distmat, np.ndarray)
  assert isinstance(query_ids, np.ndarray)
  assert isinstance(gallery_ids, np.ndarray)
  assert isinstance(query_cams, np.ndarray)
  assert isinstance(gallery_cams, np.ndarray)

  m, n = distmat.shape

  # Sort and find correct matches
  # np.argsort:axis = 1 返回按行排序的索引值
  indices = np.argsort(distmat, axis=1)
  matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
  # Compute AP for each query
  aps = np.zeros(m)
  is_valid_query = np.zeros(m)
  for i in range(m):
    # Filter out the same id and same camera
    valid = ((gallery_ids[indices[i]] != query_ids[i]) |
             (gallery_cams[indices[i]] != query_cams[i]))
    y_true = matches[i, valid]
    y_score = -distmat[i][indices[i]][valid]
    if not np.any(y_true): continue
    is_valid_query[i] = 1
    aps[i] = average_precision_score(y_true, y_score)
  if len(aps) == 0:
    raise RuntimeError("No valid query")
  if average:
    return float(np.sum(aps)) / np.sum(is_valid_query)
  return aps, is_valid_query

