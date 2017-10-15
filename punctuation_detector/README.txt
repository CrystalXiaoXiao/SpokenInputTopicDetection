Punctuation detection is a sequence tagger using BiLSTM.
A sequence of words and their tags (SPACE, PERIOD) is constructed using a sliding window.

Three main scripts are available

1. run_feature_preparation.sh
Constructs train/dev/test feature and target vectors
-Parameters: 
	-PATHS for train/dev/test TDTCorpus pickled objects
	-SEN_SIZE maximum sequence length (LSTM time steps)
	-SHIFT	sliding window shift value
	-VOCAB_PATH path for pickled vocab object
-Output:
	-feats_*.txt -> feature vector file
	-targets_*.txt -> target vector file
	-marker_*.txt -> a list aligned to the feature vector file marking the end of documents
	-doc_length_*.txt -> document lengths in sentences

2. run_train.sh
Trains a BiLSTM for punctuation detection
-Parameters:
	-SEN_SIZE maximum sequence length (LSTM time steps)
	-SHIFT sliding window shift value
	-EMBED_PATH path for word embeddings file (GloVe format)
	-STAGE set to "train" to train a network, "dev" to perform inference
	-VOCAB path to vocab pickled object
	-FEAT_PATH path to previously generated features folder
	-MODELS_PATH path to saved models
	-LOAD_MODEL model file name (set only if performing inference)
	-SUFFIX identifies which feature files to perform inference on (set to "dev" or "test")
-Output:
	-If STAGE="train", we perform training, output is a model saved in $MODELS_PATH
	-If STAGE="dev", we perform inference, predictions and reference are saved to
	 "./dev_hyp" and "./dev_ref" respectively.

3. run_evaluate.sh
Evaluates Precision, Recall, F1 for the last set we performed inference on
-Parameters:
	-THRESHOLD network confidence value
	-SEN_SIZE maximum sequence length (LSTM time steps)
	-SHIFT sliding window shift value
	-FEAT_PATH path to previously generated features folder
	-SUFFIX identifies which feature files to perform inference on (set to "dev" or "test")
-Output:
	-Precision, Recall and F1 are printed to console
