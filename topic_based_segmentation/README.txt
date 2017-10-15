Topic-based segmentation is composed of two trained models, one using LDA vectors as features
and the other uses Word Embeddings as features. The output of both models are combined
to provide final decision. Segmentation is performed on the sentence level.

Four main scripts are available:

1.run_train_lda.sh
Constructs LDA feature table p(w|t) and corresponding LDA vocab

-Parameters:
    -LDA_DIM number of topics (feature dim)
    -VOCAB_PATH path to vocab pickled object
    -PATH_TRAIN path to training pickled TDTCorpus object
-Output:
    -lda_matrix_DIM_train.pckl a pickle object containing the lda matrix
    -vocab_lda_latest.pckl a pickle object containing the vocab for the LDA training data used

2.run_feature_preparation.sh
Prepares feature files for network training.
Features are extracted as follows, for each possible boundary (end of sentence), a window of
words around the boundary is used as the input.

This script should be run twice, once using MODEL_TYPE="lda" and another using MODEL_TYPE="embed"

-Parameters:
    -PATHS for train/test/dev TDTCorpus pickle objects
    -WINDOW number of sentences to consider around the boundary
    -MAX_WORDS maximum number of words per sentence to consider before clipping the vector
    -MODEL_TYPE type of model to generate features for ("lda" or "embed")
-Output:
    -feats_*.txt feature vectors file
    -targets_*.txt targets file
    -doc_length_*.txt lengths of each document in sentences

3.run_train_embed.sh
Trains a single LSTM to perform binary classification on sentence boundaries for topic change detection
Should be run twice:
once with MODEL_TYPE="lda" and another time using MODEL_TYPE="embed". Make sure to set DIM and VOCAB params each
time accordingly

-Parameters:
    -MODEL_TYPE type of model to train ("lda" or "embed")
    -STAGE set to "train" or "test"
    -EMBED_PATH path to word embeddings (GloVe format)
    -WINDOW number of sentences to consider around the boundary
    -MAX_WORDS maximum number of words per sentence to consider before clipping the vector
    -DIM feature vector dimension
    -VOCAB path to vocab pickled object
    -FEAT_PATH path to feature files folder
    -MODELS_PATH path to save the model trained
    -LOAD_MODEL file name of a previously trained model (will continue training if STAGE="train")
    -SUFFIX do not use in training mode
-Output:
    -Trained model saved in MODELS_PATH

4.run_test_embed.sh
Performs inference using two trained models (lda+embed), combines the predictions and outputs a final prediction.
Also performs Pk evaluation on the result and prints the error rates to the console.

-Parameters:
    -Same parameter names as before, make sure to set everything accordingly
-Output:
    -Pk values printed to console