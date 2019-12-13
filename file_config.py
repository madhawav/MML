import os

class FileConfig:
    def __init__(self):
        self.train_caption_features_dir = "./data/video_understanding_features_train/"
        self.train_object_features_dir = "./data/clip_object_features_train/"
        self.train_softmax_dir = './data/train_softmax/'
        self.sliding_clip_path = "./data/all_fc6_unit16_overlap0.5/"
        self.clip_sentence_pairs_iou = "./data/ref_info/charades_sta_train_semantic_sentence_VP_sub_obj.pkl"
        self.movie_length_info = "./data/ref_info/charades_movie_length_info.txt"
        self.sliding_clip_path = "./data/all_fc6_unit16_overlap0.5/"
        self.test_caption_features_dir = "./data/video_understanding_features_test/"
        self.test_softmax_dir = './data/test_softmax/'
        self.test_object_features_dir = "./data/clip_object_features_test/"
        self.test_swin_txt_path = "./data/ref_info/charades_sta_test_swin_props_num_36364.txt"
        self.clip_sentence_pairs = "./data/ref_info/charades_sta_test_semantic_sentence_VP_sub_obj.pkl"
        self.train_cache_path = "train_cache.pt"