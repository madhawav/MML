'''
Code adapted from https://github.com/WuJie1010/Temporally-language-grounding/blob/master/dataloader_charades_SL.py
The code has been improved to support the Multi-Faceted Moment Localizing model we propose.
'''
import torch
import torch.utils.data
import os
import pickle
import numpy as np
import math
from utils import *
import random
import spacy
from nltk.stem import WordNetLemmatizer
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

class Charades_Test_dataset(torch.utils.data.Dataset):
    def __init__(self, file_config, use_bert_sentence=True, use_object_features=True, use_caption_features=True):
        self.use_object_features = use_object_features
        # il_path: image_label_file path
        self.context_num = 1
        self.use_caption_features = use_caption_features
        self.context_size = 128
        self.visual_feature_dim = 4096 * 3
        self.feats_dimen = 4096
        self.unit_size = 16
        self.context_size = 128
        self.semantic_size = 4800
        self.epochs_completed = 0
        self.index_in_epoch = 0
        self.spacy_vec_dim = 300
        self.sent_vec_dim = 4800
        self.clip_softmax_dim = 400
        self.softmax_unit_size = 32
        self.sliding_clip_path = file_config.sliding_clip_path
        self.test_caption_features_dir = file_config.test_caption_features_dir
        self.test_softmax_dir = file_config.test_softmax_dir
        self.test_object_features_dir = file_config.test_object_features_dir
        self.test_swin_txt_path = file_config.test_swin_txt_path
        self.clip_sentence_pairs = pickle.load(open(file_config.clip_sentence_pairs))
        print str(len(self.clip_sentence_pairs)) + " test videos are readed"  # 1334

        self.cached_test_softmax = {}
        self.cached_sliding_clip = {}
        self.cached_caption_features = {}

        self.use_bert_sentence = use_bert_sentence

        if use_caption_features:
            self.feats_dimen += 2048
            self.visual_feature_dim = self.feats_dimen * 3

        if self.use_object_features:
            self.clip_softmax_dim += 150
            self.loaded_object_features = {}

        if use_bert_sentence:
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # Load pre-trained model (weights)
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')

            # Put the model in "evaluation" mode, meaning feed-forward operation.
            self.bert_model.eval()

        movie_names_set = set()
        for ii in self.clip_sentence_pairs:
            for iii in self.clip_sentence_pairs[ii]:
                clip_name = iii
                movie_name = ii
                if not movie_name in movie_names_set:
                    movie_names_set.add(movie_name)
        self.movie_names = list(movie_names_set)

        self.sliding_clip_names = []
        with open(self.test_swin_txt_path) as f:
            for l in f:
                self.sliding_clip_names.append(l.rstrip().replace(" ", "_"))
        print "sliding clips number for test: " + str(len(self.sliding_clip_names))  # 36364

        self.movie_length_dict = {}
        with open(file_config.movie_length_info)  as f:
            for l in f:
                self.movie_length_dict[l.rstrip().split(" ")[0]] = float(l.rstrip().split(" ")[1])

    def read_unit_level_feats(self, clip_name):
        # read unit level feats by just passing the start and end number
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2])
        num_units = (end - start) / self.unit_size
        curr_start = start

        start_end_list = []
        while (curr_start + self.unit_size <= end):
            start_end_list.append((curr_start, curr_start + self.unit_size))
            curr_start += self.unit_size

        original_feats = np.zeros([num_units, self.feats_dimen], dtype=np.float32)
        for k, (curr_s, curr_e) in enumerate(start_end_list):
            np_path = self.sliding_clip_path + movie_name + "_" + str(curr_s) + ".0_" + str(curr_e) + ".0.npy"
            if np_path not in self.cached_sliding_clip:
                self.cached_sliding_clip[np_path] = np.load(np_path)
            one_feat = self.cached_sliding_clip[np_path]

            if self.use_caption_features:
                np_path_caption = self.test_caption_features_dir + movie_name + "_" + str(curr_s) + ".0_" + str(
                    curr_e) + ".0.npy"
                if np_path_caption not in self.cached_caption_features:
                    self.cached_caption_features[np_path_caption] = np.load(np_path_caption)
                one_feat_captions = self.cached_caption_features[np_path_caption]
                one_feat_captions = one_feat_captions / np.linalg.norm(one_feat_captions)
                #                 print(one_feat.shape)
                #                 print(one_feat_captions.shape)
                one_feat = np.concatenate([one_feat, one_feat_captions])

            original_feats[k] = one_feat

        return np.mean(original_feats, axis=0)

    def get_bert_sentence_tokens(self, sentences):
        # Code adapted from: https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
        marked_texts = ["[CLS] " + text + " [SEP]" for text in sentences]
        tokenized_texts = [self.bert_tokenizer.tokenize(marked_text) for marked_text in marked_texts]
        indexed_tokens_of_texts = [self.bert_tokenizer.convert_tokens_to_ids(tokenized_text) for tokenized_text in
                                   tokenized_texts]
        segments_ids_of_texts = [[1] * len(tokenized_text) for tokenized_text in tokenized_texts]

        # Convert inputs to PyTorch tensors
        sentence_embeddings = []
        for i in range(len(sentences)):
            tokens_tensor = torch.tensor([indexed_tokens_of_texts[i]])
            segments_tensors = torch.tensor([segments_ids_of_texts[i]])

            # Predict hidden states features for each layer
            with torch.no_grad():
                encoded_layers, _ = self.bert_model(tokens_tensor, segments_tensors)

            token_embeddings = []
            sentences_word_embeddings = []

            # For each token in the sentence...
            for token_i in range(len(tokenized_texts[i])):

                # Holds 12 layers of hidden states for each token
                hidden_layers = []

                # For each of the 12 layers...
                for layer_i in range(len(encoded_layers)):
                    # Lookup the vector for `token_i` in `layer_i`
                    vec = encoded_layers[layer_i][0][token_i]

                    hidden_layers.append(vec)

                token_embeddings.append(hidden_layers)
            sentence_embedding = torch.mean(encoded_layers[11], 1)
            sentence_embeddings.append(sentence_embedding)

            summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings]
            sentences_word_embeddings.append(summed_last_4_layers)
        return torch.cat(sentence_embeddings), sentences_word_embeddings, tokenized_texts

    def read_unit_level_softmax(self, clip_name):
        # read unit level softmax by just passing the start and end number
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2])
        num_units = (end - start) / self.unit_size - (self.softmax_unit_size / self.unit_size) + 1
        _is_clip_shorter_than_unit_size = False
        if num_units <= 0:
            num_units = 1
            _is_clip_shorter_than_unit_size = True

        softmax_feats = np.zeros([num_units, self.clip_softmax_dim], dtype=np.float32)
        if _is_clip_shorter_than_unit_size:
            _start_here = start
            _end_here = end
            _npy_file_path_this = self.test_softmax_dir + movie_name + ".mp4_" + str(curr_s) + "_" + str(
                curr_e) + ".npy"
            if not os.path.exists(_npy_file_path_this):
                _npy_file_path_this = self.test_softmax_dir + movie_name + ".mp4_" + str(curr_s) + "_" + str(
                    curr_e) + ".npy"

            if _npy_file_path_this not in self.cached_test_softmax:
                self.cached_test_softmax[_npy_file_path_this] = np.load(_npy_file_path_this)
            one_feat = self.cached_test_softmax[_npy_file_path_this]

            if self.use_object_features:
                object_features_file = self.test_object_features_dir + movie_name + ".mp4_" + str(curr_s) + "_" + str(
                    curr_e) + ".pt"
                if object_features_file not in self.loaded_object_features:
                    self.loaded_object_features[object_features_file] = torch.load(object_features_file).numpy()
                object_features = self.loaded_object_features[object_features_file]
                softmax_feats[0] = np.concatenate([object_features, one_feat])
            else:
                softmax_feats[0] = one_feat

        else:
            curr_start = start
            start_end_list = []
            while (curr_start + self.softmax_unit_size <= end):
                start_end_list.append((curr_start, curr_start + self.softmax_unit_size))
                curr_start += self.unit_size
            for k, (curr_s, curr_e) in enumerate(start_end_list):
                one_feat_path = self.test_softmax_dir + movie_name + ".mp4_" + str(curr_s) + "_" + str(curr_e) + ".npy"
                if one_feat_path not in self.cached_test_softmax:
                    self.cached_test_softmax[one_feat_path] = np.load(one_feat_path)
                one_feat = self.cached_test_softmax[one_feat_path]

                if self.use_object_features:
                    object_features_file = self.test_object_features_dir + movie_name + ".mp4_" + str(
                        curr_s) + "_" + str(curr_e) + ".pt"
                    if object_features_file not in self.loaded_object_features:
                        self.loaded_object_features[object_features_file] = torch.load(object_features_file).numpy()
                    object_features = self.loaded_object_features[object_features_file]
                    softmax_feats[k] = np.concatenate([object_features, one_feat])
                else:
                    softmax_feats[k] = one_feat

        return np.mean(softmax_feats, axis=0)

    def feat_exists(self, clip_name):
        # judge the feats is existed or not
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2])

        return os.path.exists(
            self.sliding_clip_path + movie_name + "_" + str(end - 16) + ".0_" + str(end) + ".0.npy") and \
               os.path.exists(
                   self.sliding_clip_path + movie_name + "_" + str(start) + ".0_" + str(start + 16) + ".0.npy")

    def get_context_window(self, clip_name, win_length):
        # compute left (pre) and right (post) context features based on read_unit_level_feats().
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2])
        clip_length = self.context_size
        left_context_feats = np.zeros([win_length, self.feats_dimen], dtype=np.float32)
        right_context_feats = np.zeros([win_length, self.feats_dimen], dtype=np.float32)
        last_left_feat = self.read_unit_level_feats(clip_name)
        last_right_feat = self.read_unit_level_feats(clip_name)
        for k in range(win_length):
            left_context_start = start - clip_length * (k + 1)
            left_context_end = start - clip_length * k
            right_context_start = end + clip_length * k
            right_context_end = end + clip_length * (k + 1)
            left_context_name = movie_name + "_" + str(left_context_start) + "_" + str(left_context_end)
            right_context_name = movie_name + "_" + str(right_context_start) + "_" + str(right_context_end)
            if self.feat_exists(left_context_name):
                left_context_feat = self.read_unit_level_feats(left_context_name)
                last_left_feat = left_context_feat
            else:
                left_context_feat = last_left_feat
            if self.feat_exists(right_context_name):
                right_context_feat = self.read_unit_level_feats(right_context_name)
                last_right_feat = right_context_feat
            else:
                right_context_feat = last_right_feat
            left_context_feats[k] = left_context_feat
            right_context_feats[k] = right_context_feat
        return np.mean(left_context_feats, axis=0), np.mean(right_context_feats, axis=0)

    def load_movie_slidingclip(self, movie_name, sample_num):
        # load unit level feats and sentence vector
        movie_clip_sentences = []
        movie_clip_featmap = []

        for dict_2nd in self.clip_sentence_pairs[movie_name]:
            for dict_3rd in self.clip_sentence_pairs[movie_name][dict_2nd]:
                VP_spacy_vec_ = np.zeros(self.spacy_vec_dim * 2)
                subj_spacy_vec_ = np.zeros(self.spacy_vec_dim)
                obj_spacy_vec_ = np.zeros(self.spacy_vec_dim)

                if len(dict_3rd['dobj_or_VP']) != 0:
                    VP_spacy_one_by_one_this_ = dict_3rd['VP_spacy_vec_one_by_one_word'][
                        random.choice(xrange(len(dict_3rd['dobj_or_VP'])))]
                    if len(VP_spacy_one_by_one_this_) == 1:
                        VP_spacy_vec_[:self.spacy_vec_dim] = VP_spacy_one_by_one_this_[0]
                    else:
                        VP_spacy_vec_ = np.concatenate((VP_spacy_one_by_one_this_[0], VP_spacy_one_by_one_this_[1]))
                if len(dict_3rd['subj']) != 0:
                    subj_spacy_vec_ = dict_3rd['subj_spacy_vec'][random.choice(xrange(len(dict_3rd['subj'])))]
                if len(dict_3rd['obj']) != 0:
                    obj_spacy_vec_ = dict_3rd['obj_spacy_vec'][random.choice(xrange(len(dict_3rd['obj'])))]
                VP_spacy_vec_ = torch.Tensor(VP_spacy_vec_)
                subj_spacy_vec_ = torch.Tensor(subj_spacy_vec_)
                obj_spacy_vec_ = torch.Tensor(obj_spacy_vec_)

                if self.use_bert_sentence:
                    if "sent_bert" not in dict_3rd:
                        dict_3rd["sent_bert"], _, _ = self.get_bert_sentence_tokens([dict_3rd['sentence']])
                        dict_3rd["sent_bert"] = dict_3rd["sent_bert"][0]
                    sentence_vec_ = dict_3rd["sent_bert"]
                else:
                    sentence_vec_ = torch.Tensor(dict_3rd['sent_skip_thought_vec'][0][0, :self.sent_vec_dim])

                movie_clip_sentences.append((dict_2nd, sentence_vec_, VP_spacy_vec_, subj_spacy_vec_, obj_spacy_vec_))

        for k in xrange(len(self.sliding_clip_names)):
            if movie_name in self.sliding_clip_names[k]:
                left_context_feat, right_context_feat = self.get_context_window(self.sliding_clip_names[k],
                                                                                self.context_num)
                feature_data = self.read_unit_level_feats(self.sliding_clip_names[k])

                # read softmax batch
                softmax_center_clip = self.read_unit_level_softmax(self.sliding_clip_names[k])

                comb_feat = np.hstack((left_context_feat, feature_data, right_context_feat))
                movie_clip_featmap.append((self.sliding_clip_names[k], comb_feat, softmax_center_clip))
                # movie_clip_featmap.append((self.sliding_clip_na
        return movie_clip_featmap, movie_clip_sentences
