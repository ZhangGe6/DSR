import numpy as np
import torch
from torch.autograd import Variable
import time
from sklearn import linear_model


def normalize(nparray, order=2, axis=0):
  """Normalize a N-D numpy array along the specified axis."""
  norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
  return nparray / (norm + np.finfo(np.float32).eps)

def compute_dist(array1, array2, type='euclidean'):
  """Compute the euclidean or cosine distance of all pairs.
  Args:
    array1: numpy array with shape [m1, n]
    array2: numpy array with shape [m2, n]
    type: one of ['cosine', 'euclidean']
  Returns:
    numpy array with shape [m1, m2]
  """
  assert type in ['cosine', 'euclidean']
  if type == 'cosine':
    array1 = normalize(array1, axis=1)
    array2 = normalize(array2, axis=1)
    dist = np.matmul(array1, array2.T)
    return dist
  else:
    # shape [m1, 1]
    square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
    # shape [1, m2]
    square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
    squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
    squared_dist[squared_dist < 0] = 0
    dist = np.sqrt(squared_dist)
    return dist
    
def dsr_dist_L2(probeSpatialFeatures, gallerySpatialFeatures):

    beta = 0.001
    dist = torch.ones(len(probeSpatialFeatures), len(gallerySpatialFeatures))
    Y = gallerySpatialFeatures[0]
    Y = torch.FloatTensor(Y)
    I = beta * torch.eye((torch.matmul(Y.t(), Y)).size(0))
    dist = dist.cuda()
    I = I.cuda()

    tmp_for_Y_set = []
    for i in range(0, len(gallerySpatialFeatures)):
        Y = gallerySpatialFeatures[i]
        Y = torch.FloatTensor(Y)
        Y = Y.view(Y.size(0), Y.size(1))
        Y = Y.cuda()
        tmp_for_Y = torch.matmul(torch.inverse(torch.matmul(Y.t(), Y) + I), Y.t())  # W * X^(-1)
        tmp_for_Y = tmp_for_Y.cpu().numpy()
        tmp_for_Y_set.append(tmp_for_Y)

    #print(len(probeSpatialFeatures))
    for i in range(0, len(probeSpatialFeatures)):
        q = torch.FloatTensor(probeSpatialFeatures[i])
        q = q.view(q.size(0), q.size(1))
        q = q.cuda()

        for j in range(0, len(gallerySpatialFeatures)):
            Y = gallerySpatialFeatures[j]   
            Y = torch.FloatTensor(Y)
            Y = Y.view(Y.size(0), Y.size(1))
            Y = Y.cuda()
            Proj_M = torch.FloatTensor(tmp_for_Y_set[j])
            Proj_M = Proj_M.cuda()
            a = torch.matmul(Y, torch.matmul(Proj_M, q)) - q
            dist[i, j] = torch.pow(a, 2).sum(0).sqrt().mean()   # 第i个probe与和其(global dis)第j近的gallery之间的DSR距离

    dist = dist.cpu()
    dist = dist.numpy()
    return dist
    # 返回的是：60*60的矩阵。对应输入的60个probe与各自和输入的60个gallery之间的DSR/spatial距离

def dsr_dist_L2_weight(probeSpatialFeatures, gallerySpatialFeatures):
    print("weighted distance")
    beta = 0.001
    dist = torch.ones(len(probeSpatialFeatures), len(gallerySpatialFeatures))
    Y = gallerySpatialFeatures[0]
    Y = torch.FloatTensor(Y)
    I = beta * torch.eye((torch.matmul(Y.t(), Y)).size(0))
    dist = dist.cuda()
    I = I.cuda()

    tmp_for_Y_set = []
    for i in range(0, len(gallerySpatialFeatures)):
        Y = gallerySpatialFeatures[i]
        Y = torch.FloatTensor(Y)
        Y = Y.view(Y.size(0), Y.size(1))
        Y = Y.cuda()
        tmp_for_Y = torch.matmul(torch.inverse(torch.matmul(Y.t(), Y) + I), Y.t())  # W * X^(-1)
        tmp_for_Y = tmp_for_Y.cpu().numpy()
        tmp_for_Y_set.append(tmp_for_Y)

    #print(len(probeSpatialFeatures))
    for i in range(0, len(probeSpatialFeatures)):
        q = torch.FloatTensor(probeSpatialFeatures[i])
        q = q.cuda()
        #print(q.size())   #2048 160
        
        qt = q.sum(dim=0)
        #print(qt.size()) #160
        
        Max = torch.max(qt)
        Min = torch.min(qt)
        #print(Max, Min)
        q_weight = (qt - Min)/(Max-Min)
        # q_sum = q_weight.sum(dim=0)
        # print(q_sum)
        # q_weight = q_weight/q_sum
        
        q = q.view(q.size(0), q.size(1))
        #q = q.cuda()

        for j in range(0, len(gallerySpatialFeatures)):
            Y = gallerySpatialFeatures[j]   
            Y = torch.FloatTensor(Y)
            Y = Y.view(Y.size(0), Y.size(1))
            Y = Y.cuda()
            Proj_M = torch.FloatTensor(tmp_for_Y_set[j])
            Proj_M = Proj_M.cuda()
            a = torch.matmul(Y, torch.matmul(Proj_M, q)) - q
            a = torch.matmul(a, q_weight.t())
            dist[i, j] = torch.pow(a, 2).sum(0).sqrt().mean()   # 第i个probe与和其(global dis)第j近的gallery之间的DSR距离

    dist = dist.cpu()
    dist = dist.numpy()
    return dist
    # 返回的是：60*60的矩阵。对应输入的60个probe与各自和输入的60个gallery之间的DSR/spatial距离

    # 返回的是：60*60的矩阵。对应输入的60个probe与各自和输入的60个gallery之间的DSR/spatial距离
def dsr_dist_L1(probeSpatialFeatures, gallerySpatialFeatures):   
    dist = np.ones((len(probeSpatialFeatures), len(gallerySpatialFeatures)))
    for i in range(0, len(probeSpatialFeatures)):
        for j in range(0, len(gallerySpatialFeatures)):
            W = linear_model.orthogonal_mp(gallerySpatialFeatures[j], probeSpatialFeatures[i])
            dist[i][j] = np.linalg.norm(probeSpatialFeatures[i]-np.dot(gallerySpatialFeatures[j], W),  ord='fro')
            print(i, j, ' done')

    return dist
  