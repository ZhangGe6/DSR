#!/usr/bin/env python2/python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:46:56 2017
@author: luohao
Modified by Houjing Huang, 2017-12-22. 
- This version accepts distance matrix instead of raw features. 
- The difference of `/` division between python 2 and 3 is handled.
- numpy.float16 is replaced by numpy.float32 for numerical precision.

Modified by Zhedong Zheng, 2018-1-12.
- replace sort with topK, which save about 30s.
"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API
q_g_dist: query-gallery distance matrix, numpy array, shape [num_query, num_gallery]
q_q_dist: query-query distance matrix, numpy array, shape [num_query, num_query]
g_g_dist: gallery-gallery distance matrix, numpy array, shape [num_gallery, num_gallery]
k1, k2, lambda_value: parameters, the original paper is (k1=20, k2=6, lambda_value=0.3)
Returns:
  final_dist: re-ranked distance, numpy array, shape [num_query, num_gallery]
"""


import numpy as np
from metrics import cmc, mean_ap
from utils import measure_time, PR_curve
import time

def k_reciprocal_neigh( initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]

def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):

    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    # 下面两行对original_dist的处理对于性能影响很大
    #original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = original_dist.astype(np.float32)
    #original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)


    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist
    

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
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=True,
        topk=10)
    return mAP, cmc_scores

def print_cmc_scores(cmc):
    mylog = open('./PR/cmc'+str(time.time())+'.txt', mode = 'a',encoding='utf-8')
    print(cmc, file=mylog)
    mylog.close()
    
    print("Rank-1:", cmc[0]*100, '%')
    print("Rank-3:", cmc[2]*100, '%')
    print("Rank-5:", cmc[4]*100, '%')


# query, gallery, multi-query indices
query_ids, gallery_ids, query_cams, gallery_cams = [], [], [], []
for i in range(1, 61):        
    query_ids.append(i)
    gallery_ids.append(i)
    query_cams.append(0)
    gallery_cams.append(1)
query_ids = np.hstack(query_ids)
query_cams = np.hstack(query_cams)
gallery_ids = np.hstack(gallery_ids)
gallery_cams = np.hstack(gallery_cams)

# load type: LIST
Q_G_global_dist = np.load("./distances/Q_G_global_dist.npy")
Q_Q_global_dist = np.load("./distances/Q_Q_global_dist.npy")
G_G_global_dist = np.load("./distances/G_G_global_dist.npy")

Q_G_spatial_dist = np.load("./distances/Q_G_spatial_dist.npy")
Q_Q_spatial_dist = np.load("./distances/Q_Q_spatial_dist.npy")
G_G_spatial_dist = np.load("./distances/G_G_spatial_dist.npy") 

with measure_time('Computing scores...', verbose=True):
    for lam in range(0, 1):
        mAP1 = []
        cmc_scores1 = []
        precision1 = []
        recall1 = []
        weight = lam * 0.1
        print('----cur_weight:', weight, '----')
        for i in range(0, len(Q_G_spatial_dist)): 
            q_g_dist = weight * Q_G_global_dist[i] + (1 - weight) * Q_G_spatial_dist[i]  #这个i便很好理解了
            q_q_dist = weight * Q_Q_global_dist[i] + (1 - weight) * Q_Q_spatial_dist[i]
            g_g_dist = weight * G_G_global_dist[i] + (1 - weight) * G_G_spatial_dist[i]
            #print(q_g_dist[0])
            # -- base67下 --
            # while lambda == 1, rank1 = 39
            #re_ranked_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=10, k2=6, lambda_value=0.3)    #40.9
            #re_ranked_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=15, k2=8, lambda_value=0.3)    #41.3
            #re_ranked_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=15, k2=10, lambda_value=0.3)    #42
            #re_ranked_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=15, k2=13, lambda_value=0.3)    #42.86
            #re_ranked_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=15, k2=13, lambda_value=0.28)    #43.06
            #re_ranked_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=15, k2=13, lambda_value=0.25)    #43.33
            
            # -- base46下 --
            # while lambda == 1, rank1 = 39.2
            #re_ranked_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=15, k2=13, lambda_value=0.3)      #40.8
            #re_ranked_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=15, k2=13, lambda_value=0.25)      #41.39
            re_ranked_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=15, k2=13, lambda_value=0.1)      #44.4
            #re_ranked_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=15, k2=13, lambda_value=0)      #44.33
            #print(re_ranked_dist[0])
            mAP, cmc_scores = compute_score(re_ranked_dist)
            precision, recall = PR_curve(re_ranked_dist, gallery_ids, query_ids, i)
            print(precision[0])
   
            mAP1.append(mAP); cmc_scores1.append(cmc_scores)
            precision1.append(precision); recall1.append(recall)
            
            #print(precision1)
            #break   
        # sum(mAP1)/25 means 对这25个single shot做平均,返回一个值
        # sum(cmc_scores1) / 25同理，返回一个topk向量
        av_mAP = sum(mAP1) / 25
        av_cmc = sum(cmc_scores1) / 25
        ave_precision = sum(precision1) / 25
        ave_recall = sum(recall1) / 25
        
        print('mAP = ', av_mAP)
        print_cmc_scores(av_cmc)
        
        np.savetxt("./PR/ave_recall"+str(time.time())+".txt",ave_recall)
        np.savetxt("./PR/ave_precision"+str(time.time())+".txt",ave_precision)
