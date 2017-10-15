SEN_SIZE=$1
SHIFT=$2
PATH_TRAIN="/home/mohamed/Desktop/scripts/tdt_reader/corpus_train_annotated_data_nltk_boundaries.pckl"
PATH_TEST="/home/mohamed/Desktop/scripts/tdt_reader/corpus_test_annotated_data_nltk_boundaries.pckl"
PATH_DEV="/home/mohamed/Desktop/scripts/tdt_reader/corpus_dev_annotated_data_nltk_boundaries.pckl"
VOCAB_PATH="vocab_train.pckl"

PATH_MANUAL_TEST="/home/mohamed/Desktop/scripts/tdt_reader/corpus_annotated_manual_test_data_nltk_boundaries.pckl"
PATH_AUTOMATIC_TEST="/home/mohamed/Desktop/scripts/spoken_data_reader/spoken_data_test_annotated.pckl"

#python FeatureConstructor.py dev $VOCAB_PATH $SEN_SIZE $SHIFT $PATH_TRAIN
#python FeatureConstructor.py train $VOCAB_PATH $SEN_SIZE $SHIFT $PATH_TEST
#python FeatureConstructor.py test $VOCAB_PATH $SEN_SIZE $SHIFT $PATH_DEV

python FeatureConstructor.py manual_test $VOCAB_PATH $SEN_SIZE $SHIFT $PATH_MANUAL_TEST
#python FeatureConstructor.py spoken_test $VOCAB_PATH $SEN_SIZE $SHIFT $PATH_AUTOMATIC_TEST
