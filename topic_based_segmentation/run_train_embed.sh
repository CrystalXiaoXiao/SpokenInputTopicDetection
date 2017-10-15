#!/usr/bin/env bash
EMBED_PATH1="/media/mohamed/24E8F168E8F1389E/googlenews.txt"
EMBED_PATH3="/home/mohamed/Desktop/scripts/tdt_reader/glove.6B.50d.txt"

MODEL_TYPE="lda"
STAGE="train"
EMBED_PATH="/home/mohamed/Desktop/scripts/word2vec_trainer/models/w2v_train_50d_10context_cbow.txt"
WINDOW=4
MAX_WORDS=10
DIM=150
VOCAB="vocab_lda_latest.pckl"
FEAT_PATH="/media/mohamed/24E8F168E8F1389E/UbuntuFiles/topic_segmentation/feat"
MODELS_PATH="/media/mohamed/24E8F168E8F1389E/UbuntuFiles/topic_segmentation/models"
LOAD_MODEL=""
SUFFIX="manual_test"

python train_keras_model.py $MODEL_TYPE $STAGE $EMBED_PATH $WINDOW $MAX_WORDS $DIM $VOCAB $FEAT_PATH $MODELS_PATH
