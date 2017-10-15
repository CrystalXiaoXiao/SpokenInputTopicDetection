#!/usr/bin/env bash
WINDOW=$1
MAX_WORDS=$2
MODEL_TYPE="lda"

VOCAB_PATH="vocab_lda_latest.pckl"
PATH_TRAIN="/home/mohamed/Desktop/scripts/tdt_reader/corpus_train_annotated_data_nltk_boundaries.pckl"
PATH_TEST="/home/mohamed/Desktop/scripts/tdt_reader/corpus_test_annotated_data_nltk_boundaries.pckl"
PATH_DEV="/home/mohamed/Desktop/scripts/tdt_reader/corpus_dev_annotated_data_nltk_boundaries.pckl"

PATH_MANUAL_TEST="/home/mohamed/Desktop/scripts/tdt_reader/corpus_annotated_manual_test_data_nltk_boundaries.pckl"
PATH_AUTOMATIC_TEST="/home/mohamed/Desktop/scripts/spoken_data_reader/spoken_data_test_annotated.pckl"

python FeatureConstructor.py dev_${MODEL_TYPE} $VOCAB_PATH $WINDOW $MAX_WORDS $PATH_DEV
python FeatureConstructor.py train_${MODEL_TYPE} $VOCAB_PATH $WINDOW $MAX_WORDS $PATH_TRAIN
python FeatureConstructor.py test_${MODEL_TYPE} $VOCAB_PATH $WINDOW $MAX_WORDS $PATH_TEST



#python FeatureConstructor.py manual_test vocab_lda_latest.pckl $WINDOW $MAX_WORDS $PATH_MANUAL_TEST
#python FeatureConstructor.py spoken_test vocab_lda_latest.pckl $WINDOW $MAX_WORDS $PATH_AUTOMATIC_TEST
