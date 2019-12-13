'''
Code adapted from https://github.com/WuJie1010/Temporally-language-grounding/blob/master/model_MAC.py
The MAC model has been improved to the Multi-Faceted Moment Localizing model we propose.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, mean=0, std=0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data)
        m.bias.data.fill_(0)


class MML(nn.Module):
    def __init__(self, use_bert_sentence=True, use_object_features=True, object_dropout_ratio=0.5,
                 activity_dropout_ratio=0, object_scale_ratio=0.005, use_softmax=True, use_caption_features=True,
                 caption_scale_ratio=0.005):
        super(MML, self).__init__()
        self.semantic_size = 1024  # the size of visual and semantic comparison size

        if not use_bert_sentence:
            self.sentence_embedding_size = 4800
        else:
            self.sentence_embedding_size = 768

        self.spacy_vec_dim = 300
        self.visual_feature_dim = 4096 * 3
        self.use_object_features = use_object_features
        self.clip_softmax_dim = 0
        self.use_softmax = use_softmax
        self.acitivity_dropout_ratio = activity_dropout_ratio
        if use_softmax:
            self.clip_softmax_dim += 400

        if use_object_features:
            self.object_features_size = 150
            self.clip_softmax_dim += 150
            self.object_dropout_ratio = object_dropout_ratio
            self.object_scale_ratio = object_scale_ratio

        self.use_caption_features = use_caption_features
        if use_caption_features:
            self.visual_feature_dim = (4096 + 2048) * 3

        self.caption_scale_ratio = caption_scale_ratio

        self.action_semantic_size = 300

        self.activity_dropout = nn.Dropout(activity_dropout_ratio)
        self.object_dropout = nn.Dropout(object_dropout_ratio)
        self.v2s_lt = nn.Linear(self.visual_feature_dim, self.semantic_size)
        self.s2s_lt = nn.Linear(self.sentence_embedding_size, self.semantic_size)

        self.soft2s_lt = nn.Linear(self.clip_softmax_dim, self.action_semantic_size)
        self.VP2s_lt = nn.Linear(self.spacy_vec_dim * 2, self.action_semantic_size)

        self.fc1 = torch.nn.Conv2d(5296, 1000, kernel_size=1, stride=1)
        self.fc2 = torch.nn.Conv2d(1000, 3, kernel_size=1, stride=1)
        # Initializing weights
        self.apply(weights_init)

    def cross_modal_comb(self, visual_feat, sentence_embed, size):
        batch_size = visual_feat.size(0)

        vv_feature = visual_feat.expand([batch_size, batch_size, size])
        ss_feature = sentence_embed.repeat(1, 1, batch_size).view(batch_size, batch_size, size)

        concat_feature = torch.cat([vv_feature, ss_feature], 2)

        mul_feature = vv_feature * ss_feature  # 56,56,1024
        add_feature = vv_feature + ss_feature  # 56,56,1024

        comb_feature = torch.cat([mul_feature, add_feature, concat_feature], 2)

        return comb_feature

    def forward(self, visual_feature_train, sentence_embed_train, softmax_train, VP_embed_train):
        transformed_clip_train = self.v2s_lt(visual_feature_train)
        transformed_clip_train_norm = F.normalize(transformed_clip_train, p=2, dim=1)
        transformed_sentence_train = self.s2s_lt(sentence_embed_train)
        transformed_sentence_train_norm = F.normalize(transformed_sentence_train, p=2, dim=1)
        cross_modal_vec_train = self.cross_modal_comb(transformed_clip_train_norm, transformed_sentence_train_norm,
                                                      self.semantic_size)

        #         prev_softmax_train = softmax_train
        #         softmax_train = self.activity_dropout(softmax_train)
        if self.use_softmax:
            if self.use_object_features:
                object_features = softmax_train[:, :self.object_features_size]
                object_features = F.normalize(object_features, p=2, dim=1)
                object_features = self.object_dropout(object_features)
                object_features = object_features * (self.object_scale_ratio)
                #                 object_features = torch.zeros(object_features.shape).cuda() ## RODO: REMOVE THIS LINE
                fc8_features = softmax_train[:, self.object_features_size:]
                fc8_features = self.activity_dropout(fc8_features)
                softmax_train = torch.cat([object_features, fc8_features], dim=1)
            else:
                softmax_train = self.activity_dropout(softmax_train)
        else:
            object_features = softmax_train[:, :self.object_features_size]
            object_features = self.object_dropout(object_features)
            #             object_features = torch.zeros(object_features.shape)
            #             fc8_features = softmax_train[:,self.object_features_size:]
            softmax_train = object_features  # torch.cat([object_features, fc8_features], dim=1)

        if self.use_caption_features:
            new_visual_features = []
            for i in range(3):
                visual_features_block = visual_feature_train[:, i * (4096 + 2048):i * (4096 + 2048) + (4096)]
                caption_features_block = visual_feature_train[:,
                                         i * (4096 + 2048) + 4096:i * (4096 + 2048) + (4096 + 2048)]
                caption_features_block = caption_features_block * (self.caption_scale_ratio)
                new_visual_features.append(visual_features_block)
                new_visual_features.append(caption_features_block)
            #                 visual_feature_train[:,i*(4096+2048) + 4096:i*(4096+2048) + (4096+2048)] = caption_features_block
            #             captioning_features = softmax_train[:,:self.object_features_size]
            visual_feature_train = torch.cat(new_visual_features, dim=1)

        transformed_softmax_train = self.soft2s_lt(softmax_train)
        transformed_softmax_train_norm = F.normalize(transformed_softmax_train, p=2, dim=1)

        transformed_VP_train = self.VP2s_lt(VP_embed_train)

        transformed_VP_train_norm = F.normalize(transformed_VP_train, p=2, dim=1)
        cross_modal_action_train = self.cross_modal_comb(transformed_softmax_train_norm, transformed_VP_train_norm,
                                                         self.action_semantic_size)

        # may not need normalization
        cross_modal_vis_sent_train = F.normalize(cross_modal_vec_train, p=2, dim=2)
        cross_modal_action_train = F.normalize(cross_modal_action_train, p=2, dim=2)

        # concatenate two
        cross_modal_train = torch.cat([cross_modal_vis_sent_train, cross_modal_action_train], 2)
        cross_modal_train = cross_modal_train.unsqueeze(0).permute(0, 3, 1, 2)

        mid_output = self.fc1(cross_modal_train)
        mid_output = F.relu(mid_output)
        sim_score_mat_train = self.fc2(mid_output).squeeze(0)

        return sim_score_mat_train
