import pickle

import string
import numpy
import math
import pickle
import gensim
from gensim import corpora
import lda
import logging
import random

class DataLoader:
    def __init__(self, feat_path):
        self.feat_path = feat_path
        self.input_vector = None
        self.target_vector = None


    def load_labeled_features(self, segment="train"):
        cnt_words = 0
        cnt_oov = 0
        oov = []
        self.input_vector = []
        feats_file_str = "%s/feats_%s.txt"%(self.feat_path,segment)
        targets_file_str = "%s/targets_%s.txt"%(self.feat_path,segment)

        with open(feats_file_str) as feats_file:
            for line in feats_file:
                line = line.strip().split()
                line = [str(word) for word in line]
                self.input_vector.append(line)


        self.target_vector = numpy.loadtxt(targets_file_str)


