'''
Code adapted from https://github.com/WuJie1010/Temporally-language-grounding/blob/master/utils.py
'''
" Some useful functions "

import numpy as np
# from six.moves import xrange
import time
import pickle
import operator
import torch

def calculate_reward_batch_withstop(Previou_IoU, current_IoU, t):
    batch_size = len(Previou_IoU)
    reward = torch.zeros(batch_size)

    for i in range(batch_size):
        if current_IoU[i] > Previou_IoU[i] and Previou_IoU[i]>=0:
            reward[i] = 1 -0.001*t
        elif current_IoU[i] <= Previou_IoU[i] and current_IoU[i]>=0:
            reward[i] = -0.001*t
        else:
            reward[i] = -1 -0.001*t
    return reward


def calculate_reward(Previou_IoU, current_IoU, t):
    if current_IoU > Previou_IoU and Previou_IoU>=0:
        reward = 1-0.001*t
    elif current_IoU <= Previou_IoU and current_IoU>=0:
        reward = -0.001*t
    else:
        reward = -1-0.001*t

    return reward

def calculate_RL_IoU_batch(i0, i1):
    # calculate temporal intersection over union
    batch_size = len(i0)
    iou_batch = torch.zeros(batch_size)

    for i in range(len(i0)):
        union = (min(i0[i][0], i1[i][0]), max(i0[i][1], i1[i][1]))
        inter = (max(i0[i][0], i1[i][0]), min(i0[i][1], i1[i][1]))
        # if inter[1] < inter[0]:
        #     iou = 0
        # else:
        iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
        iou_batch[i] = iou
    return iou_batch

def calculate_IoU(i0, i1):
    # calculate temporal intersection over union
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
    return iou


def nms_temporal(x1,x2,s, overlap):
    pick = []
    assert len(x1)==len(s)
    assert len(x2)==len(s)
    if len(x1)==0:
        return pick

    union = map(operator.sub, x2, x1) # union = x2-x1
    union = list(union)

    I = [i[0] for i in sorted(enumerate(s), key=lambda x:x[1])] # sort and get index

    while len(I)>0:
        i = I[-1]
        pick.append(i)

        xx1 = [max(x1[i],x1[j]) for j in I[:-1]]
        xx2 = [min(x2[i],x2[j]) for j in I[:-1]]
        inter = [max(0.0, k2-k1) for k1, k2 in zip(xx1, xx2)]
        o = [inter[u]/(union[i] + union[I[u]] - inter[u]) for u in range(len(I)-1)]
        I_new = []
        for j in range(len(o)):
            if o[j] <=overlap:
                I_new.append(I[j])
        I = I_new
    return pick


def compute_IoU_recall_top_n_forreg_rl(top_n, iou_thresh, sentence_image_reg_mat, sclips):
    correct_num = 0.0
    for k in range(sentence_image_reg_mat.shape[0]):
        gt = sclips[k]
        # print(gt)
        gt_start = float(gt.split("_")[1])
        gt_end = float(gt.split("_")[2])

        pred_start = sentence_image_reg_mat[k, 0]
        pred_end = sentence_image_reg_mat[k, 1]
        iou = calculate_IoU((gt_start, gt_end),(pred_start, pred_end))
        if iou>=iou_thresh:
            correct_num+=1

    return correct_num

def compute_IoU_recall_top_n_forreg(top_n, iou_thresh, sentence_image_mat, sentence_image_reg_mat, sclips, iclips):
    correct_num = 0.0
    for k in range(sentence_image_mat.shape[0]):
        gt = sclips[k]
        gt_start = float(gt.split("_")[1])
        gt_end = float(gt.split("_")[2])
        #print gt +" "+str(gt_start)+" "+str(gt_end)
        sim_v = [v for v in sentence_image_mat[k]]
        starts = [s for s in sentence_image_reg_mat[k,:,0]]
        ends = [e for e in sentence_image_reg_mat[k,:,1]]
        picks = nms_temporal(starts,ends, sim_v, iou_thresh-0.05)
        #sim_argsort=np.argsort(sim_v)[::-1][0:top_n]
        if top_n<len(picks): picks=picks[0:top_n]
        for index in picks:
            pred_start = sentence_image_reg_mat[k, index, 0]
            pred_end = sentence_image_reg_mat[k, index, 1]
            iou = calculate_IoU((gt_start, gt_end),(pred_start, pred_end))
            if iou>=iou_thresh:
                correct_num+=1
                break
    return correct_num