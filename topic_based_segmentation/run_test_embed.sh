#!/usr/bin/env bash
EMBED_PATH1="/media/mohamed/24E8F168E8F1389E/googlenews.txt"
EMBED_PATH3="/home/mohamed/Desktop/scripts/tdt_reader/glove.6B.50d.txt"

MODEL_TYPE="embed"
STAGE="test"
EMBED_PATH="/home/mohamed/Desktop/scripts/word2vec_trainer/models/w2v_train_50d_10context_cbow.txt"
WINDOW=4
MAX_WORDS=10
DIM=50
VOCAB="vocab_train.pckl"
FEAT_PATH="/media/mohamed/24E8F168E8F1389E/UbuntuFiles/topic_segmentation/feat"
MODELS_PATH="/media/mohamed/24E8F168E8F1389E/UbuntuFiles/topic_segmentation/models"
LOAD_MODEL="model-topicseg-embed-fulltrain-finalver-dim50-window4max10-300x2-epoch13"
SUFFIX="manual_test"

python train_keras_model.py $MODEL_TYPE $STAGE $EMBED_PATH $WINDOW $MAX_WORDS $DIM $VOCAB $FEAT_PATH $MODELS_PATH $LOAD_MODEL $SUFFIX

echo "-------------------";
echo "evaluating embedding model...";
mv test_gold_${MODEL_TYPE}.txt ./results/test_gold_${MODEL_TYPE}_${DIM}_${SUFFIX}.txt
python get_bounds.py test_hyp_${MODEL_TYPE}.txt > ./results/test_hyp_${MODEL_TYPE}_${DIM}_${SUFFIX}.txt
rm -rf test_hyp_${MODEL_TYPE}.txt;
python ./results/evaluate.py ./results/test_gold_${MODEL_TYPE}_${DIM}_${SUFFIX}.txt ./results/test_hyp_${MODEL_TYPE}_${DIM}_${SUFFIX}.txt $FEAT_PATH $WINDOW $MAX_WORDS $MODEL_TYPE $SUFFIX





VOCAB="vocab_lda_latest.pckl"
MODEL_TYPE="lda"
LDA_DIM=150
LOAD_MODEL="model-topicseg-lda-fulltrain-finalver-dim150-window4max10-300x2-epoch07"

python train_keras_model.py $MODEL_TYPE $STAGE $EMBED_PATH $WINDOW $MAX_WORDS $LDA_DIM $VOCAB $FEAT_PATH $MODELS_PATH $LOAD_MODEL $SUFFIX

echo "-------------------";
echo "evaluating lda model...";
mv test_gold_${MODEL_TYPE}.txt ./results/test_gold_${MODEL_TYPE}_${LDA_DIM}_${SUFFIX}.txt
python get_bounds.py test_hyp_${MODEL_TYPE}.txt > ./results/test_hyp_${MODEL_TYPE}_${LDA_DIM}_${SUFFIX}.txt
rm -rf test_hyp_${MODEL_TYPE}.txt;
python ./results/evaluate.py ./results/test_gold_${MODEL_TYPE}_${LDA_DIM}_${SUFFIX}.txt ./results/test_hyp_${MODEL_TYPE}_${LDA_DIM}_${SUFFIX}.txt $FEAT_PATH $WINDOW $MAX_WORDS $MODEL_TYPE $SUFFIX

python ./results/get_bounds.py ./results/test_hyp_lda_${LDA_DIM}_${SUFFIX}.txt ./results/test_hyp_embed_${DIM}_${SUFFIX}.txt
echo "-------------------";
echo "evaluating merged model...";
for alpha in 0.1 0.3 0.5 0.7 0.9;
do
beta=0`bc <<< "1-$alpha"`;

echo "alpha: " $alpha "beta: " $beta;

python ./results/evaluate.py ./results/test_gold_${MODEL_TYPE}_${LDA_DIM}_${SUFFIX}.txt ./results/merged_${alpha}_${beta}.txt $FEAT_PATH $WINDOW $MAX_WORDS $MODEL_TYPE $SUFFIX;
done
echo "-------------------";