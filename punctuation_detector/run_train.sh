SEN_SIZE=$1
SHIFT=$2
EMBED_PATH=$3
STAGE=$4
VOCAB=$5
FEAT_PATH=$6
MODELS_PATH=$7

EMBED_PATH1="/media/mohamed/24E8F168E8F1389E/googlenews.txt"
EMBED_PATH3="/home/mohamed/Desktop/scripts/tdt_reader/glove.6B.50d.txt"

EMBED_PATH="/home/mohamed/Desktop/scripts/word2vec_trainer/models/w2v_train_50d_10context_cbow.txt"
SEN_SIZE=90
SHIFT=30
VOCAB="vocab_train.pckl"
STAGE="dev"
FEAT_PATH="/media/mohamed/24E8F168E8F1389E/UbuntuFiles/sen_segmentation/feat"
MODELS_PATH="/media/mohamed/24E8F168E8F1389E/UbuntuFiles/sen_segmentation/models"
LOAD_MODEL="model-seq-fulltrain-finalver-contepoch201-embed50-c90c30-300x2-epoch04"
SUFFIX="manual_test"

python run_keras_model.py $STAGE $EMBED_PATH $SEN_SIZE $SHIFT $VOCAB $FEAT_PATH $MODELS_PATH $LOAD_MODEL $SUFFIX
