"""
Code adapted from https://github.com/WuJie1010/Temporally-language-grounding/blob/master/main_charades_SL.py
"""
# from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import numpy as np
import os
import argparse

from charades_test_dataset import Charades_Test_dataset
from charades_train_dataset import Charades_Train_dataset
from model import MML
from file_config import FileConfig
from utils import *
import random
from torch.autograd import Variable


path = "./checkpoints"

# Configs
use_bert_sentence = True
use_object_features = True
use_softmax_features = True
use_caption_features = True
scale_ratio = 0.005
activity_dropout_ratio = 0
dropout_ratio = 0.5
caption_scale_ratio = 0.005


file_config = FileConfig()
test_dataset = Charades_Test_dataset(file_config, use_bert_sentence=use_bert_sentence, use_object_features = use_object_features, use_caption_features=use_caption_features)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


net = MML(use_bert_sentence=use_bert_sentence, use_object_features = use_object_features, object_scale_ratio=scale_ratio, object_dropout_ratio = dropout_ratio, use_softmax=use_softmax_features, activity_dropout_ratio=activity_dropout_ratio, use_caption_features=use_caption_features, caption_scale_ratio = caption_scale_ratio)
stats = torch.load("./checkpoints/best_R1_IOU5_model_drobj0.5dr_act0_sr0.005_csr0.005_best_with_captioning.t7")
net.load_state_dict(stats["net"])
net = net.cuda()

def test(dropout_ratio, scale_ratio, activity_dropout_ratio, caption_scale_ratio):
    global best_R1_IOU5
    global best_R5_IOU5
    # net.train()
    net.eval()

    IoU_thresh = [0.1, 0.3, 0.5, 0.7]
    all_correct_num_10 = [0.0] * 5
    all_correct_num_5 = [0.0] * 5
    all_correct_num_1 = [0.0] * 5
    all_retrievd = 0.0
    all_number = len(test_dataset.movie_names)
    idx = 0
    for movie_name in test_dataset.movie_names:
        idx += 1
        print("[dr-obj %.2f][dr-act %.2f][sr %.3f][csr %.3f] %d/%d" % (dropout_ratio, activity_dropout_ratio, scale_ratio, caption_scale_ratio, idx, all_number))

        movie_clip_featmaps, movie_clip_sentences = test_dataset.load_movie_slidingclip(movie_name, 16)
        print("sentences: " + str(len(movie_clip_sentences)))
        print("clips: " + str(len(movie_clip_featmaps)))  # candidate clips)

        sentence_image_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps)])
        sentence_image_reg_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps), 2])
        for k in range(len(movie_clip_sentences)):

            sent_vec = movie_clip_sentences[k][1]
            sent_vec = torch.reshape(sent_vec, [1, sent_vec.shape[0]])  # 1,4800
            sent_vec = sent_vec.cuda()

            VP_spacy_vec = movie_clip_sentences[k][2]
            VP_spacy_vec = torch.reshape(VP_spacy_vec, [1, VP_spacy_vec.shape[0]])
            VP_spacy_vec = VP_spacy_vec.float().cuda()

            for t in range(len(movie_clip_featmaps)):
                featmap = movie_clip_featmaps[t][1]
                visual_clip_name = movie_clip_featmaps[t][0]
                softmax_ = movie_clip_featmaps[t][2]

                start = float(visual_clip_name.split("_")[1])
                end = float(visual_clip_name.split("_")[2].split("_")[0])
                conf_score = float(visual_clip_name.split("_")[7])

                featmap = np.reshape(featmap, [1, featmap.shape[0]])
                featmap = torch.from_numpy(featmap).cuda()

                softmax_ = np.reshape(softmax_, [1, softmax_.shape[0]])
                softmax_ = torch.from_numpy(softmax_).cuda()

                # network forward
                outputs = net(featmap, sent_vec, softmax_, VP_spacy_vec)

                outputs = outputs.squeeze(1).squeeze(1)

                sigmoid_output0 = 1 / float(1 + torch.exp(-outputs[0]))
                sentence_image_mat[k, t] = sigmoid_output0 * conf_score

                # sentence_image_mat[k, t] = expit(outputs[0]) * conf_score
                reg_end = end + outputs[2]
                reg_start = start + outputs[1]

                sentence_image_reg_mat[k, t, 0] = reg_start
                sentence_image_reg_mat[k, t, 1] = reg_end

        iclips = [b[0] for b in movie_clip_featmaps]
        sclips = [b[0] for b in movie_clip_sentences]

        # calculate Recall@m, IoU=n
        for k in range(len(IoU_thresh)):
            IoU = IoU_thresh[k]
            correct_num_10 = compute_IoU_recall_top_n_forreg(10, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
            correct_num_5 = compute_IoU_recall_top_n_forreg(5, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
            correct_num_1 = compute_IoU_recall_top_n_forreg(1, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
            print(movie_name + " IoU=" + str(IoU) + ", R@10: " + str(correct_num_10 / len(sclips)) + "; IoU=" + str(
                IoU) + ", R@5: " + str(correct_num_5 / len(sclips)) + "; IoU=" + str(IoU) + ", R@1: " + str(
                correct_num_1 / len(sclips)))

            all_correct_num_10[k] += correct_num_10
            all_correct_num_5[k] += correct_num_5
            all_correct_num_1[k] += correct_num_1
        all_retrievd += len(sclips)
    for k in range(len(IoU_thresh)):
        print(" IoU=" + str(IoU_thresh[k]) + ", R@10: " + str(all_correct_num_10[k] / all_retrievd) + "; IoU=" + str(
            IoU_thresh[k]) + ", R@5: " + str(all_correct_num_5[k] / all_retrievd) + "; IoU=" + str(
            IoU_thresh[k]) + ", R@1: " + str(all_correct_num_1[k] / all_retrievd))

        test_result_output.write("IoU=" + str(IoU_thresh[k]) + ", R@10: " + str(
            all_correct_num_10[k] / all_retrievd) + "; IoU=" + str(IoU_thresh[k]) + ", R@5: " + str(
            all_correct_num_5[k] / all_retrievd) + "; IoU=" + str(IoU_thresh[k]) + ", R@1: " + str(
            all_correct_num_1[k] / all_retrievd) + "\n")

    R1_IOU5 = all_correct_num_1[2] / all_retrievd
    R5_IOU5 = all_correct_num_5[2] / all_retrievd

    if R1_IOU5 > best_R1_IOU5:
        print("best_R1_IOU5: %0.3f" % R1_IOU5)
        best_R1_IOU5 = R1_IOU5

    if R5_IOU5 > best_R5_IOU5:
        print("best_R5_IOU5: %0.3f" % R5_IOU5)
        best_R5_IOU5 = R5_IOU5

setup_seed(0)
best_R1_IOU5 = 0
best_R5_IOU5 = 0

if not os.path.isdir(path):
    os.mkdir(path)
test_result_output = open(os.path.join(path, "test_results.txt"), "w")

while True: # An issue with RTX 2080Ti causes a CuBLAS error occasionally. Using try-catch to avoid it.
    # Complete Error: https://github.com/pytorch/pytorch/issues/13038
    try:
        with torch.no_grad():
            test(dropout_ratio, scale_ratio, activity_dropout_ratio, caption_scale_ratio)
            break
    except:
        pass

print("[dr-obj %.2f][dr-act %.2f][sr %.3f][csr: %.3f] best_R1_IOU5: %0.3f" % (
dropout_ratio, activity_dropout_ratio, scale_ratio, caption_scale_ratio, best_R1_IOU5))
print("[dr-obj %.2f][dr-act %.2f][sr %.3f][csr: %.3f] best_R5_IOU5: %0.3f" % (
dropout_ratio, activity_dropout_ratio, scale_ratio, caption_scale_ratio, best_R5_IOU5))

