#!/usr/bin/env bash

LDA_DIM=150
VOCAB_PATH="vocab_train.pckl"
PATH_TRAIN="/home/mohamed/Desktop/scripts/tdt_reader/corpus_train_annotated_data_nltk_boundaries.pckl"

python FeatureConstructor.py $VOCAB_PATH $PATH_TRAIN $LDA_DIM