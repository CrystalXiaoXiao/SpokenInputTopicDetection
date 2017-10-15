from DataLoader import DataLoader

from keras.models import Sequential
from keras.layers import Dense, Activation, \
    LSTM, Dropout, Embedding, Lambda, Bidirectional, Merge
from keras.layers import Conv1D, MaxPooling1D
import keras
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras import backend as K

import pickle
import numpy
import math
from sklearn.metrics import f1_score
import sys
from WordEmbeddings import WordEmbeddings

window = int(sys.argv[4])
max_words = int(sys.argv[5])
max_sen_length = window*max_words
feature_dim = int(sys.argv[6])

embedding_vector_length = 0
lda_feature_length = 150

layers = [300, 300]


load_epoch = ""

def init_lstm_model(embeddings=None, stage="train", model_type="embed"):
    model_options = "topicseg-%s-fulltrain-finalver"%model_type
    out_dim = 2


    #loader.input_vector = numpy.reshape(loader.input_vector,
    #                                    (loader.input_vector.shape[0],
    #                                     1,
    #                                     loader.input_vector.shape[1]))
    model = Sequential()
    model.add(Embedding(len(embeddings),
                        feature_dim,
                        input_length=max_sen_length,
                        weights=[embeddings]))
    model.add(Dropout(0.5))
    model.layers[0].trainable = False

    for i, lstm_layer in enumerate(layers):
        if i == len(layers) - 1:
            model.add(LSTM(lstm_layer))
        else:
            model.add(LSTM(lstm_layer, return_sequences=True))
        model.add(Dropout(0.3))

    model.add(Dense(out_dim,
                    activation='softmax'))
    adam = keras.optimizers.Adam(lr=0.001,
                                 beta_1=0.9,
                                 beta_2=0.999,
                                 epsilon=1e-08,
                                 decay=0.0)

    if stage == "test" or stage == "dev":
        load_model = sys.argv[10]
        model.load_weights('%s/%s.hdf5'%(sys.argv[9],load_model))
    else:
        if len(sys.argv) >= 11:
            load_model = sys.argv[10]
            model.load_weights('%s/%s.hdf5'%(sys.argv[9],load_model))
            model_options += "-contepoch%s"%load_model[-2:]
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=[fbetascore])

    if stage == "train":
        filepath='%s/model-%s-dim%s-window%smax%s-%sx%s-epoch{epoch:02d}.hdf5'%(sys.argv[9],
                                                                 model_options,
                                                                 feature_dim,
                                                                 window,
                                                                 max_words,
                                                                 str(layers[0]),
                                                                 str(len(layers)))
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        return model, callbacks_list

    elif stage == "test" or stage == "dev":
        return model, []


def fbetascore(y_true, y_pred, beta=1):
    '''Compute F score, the weighted harmonic mean of precision and recall.

    This is useful for multi-label classification where input samples can be
    tagged with a set of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.

    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.

    '''
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Weight precision and recall together as a single scalar.
    beta2 = beta ** 2
    f_score = (1 + beta2) * (precision * recall) / (beta2 * precision + recall)
    return f_score


def main():

    train = None
    dev = None
    test = None

    model_type = sys.argv[1]
    stage = sys.argv[2]
    embeddings_path = sys.argv[3]
    window = int(sys.argv[4])
    max_words = int(sys.argv[5])
    vocab_path = sys.argv[7]
    feat_path = sys.argv[8]
    if len(sys.argv) >= 12:
        feat_suffix = sys.argv[11]

    vocab = pickle.load(open(vocab_path, 'r'))

    if model_type == "embed":
        embeddings = WordEmbeddings(embeddings_path)
        embeddings.construct_embedding_matrix(vocab)
    elif model_type == "lda":
        embeddings = WordEmbeddings(embeddings_path)
        embeddings.construct_embedding_matrix(vocab, True, topics=int(sys.argv[6]))

    loader_train = DataLoader(feat_path)
    loader_dev = DataLoader(feat_path)
    loader_test = DataLoader(feat_path)

    if stage == "train":
        loader_train.load_labeled_features("window%i_max%i_train_%s" % (window, max_words, model_type))
        loader_dev.load_labeled_features("window%i_max%i_dev_%s" % (window, max_words, model_type))

        train = (loader_train.input_vector, loader_train.target_vector)
        dev = (loader_dev.input_vector, loader_dev.target_vector)

        train = (train[0], numpy.array(keras.utils.np_utils.to_categorical(train[1], 2)))
        dev = (dev[0], numpy.array(keras.utils.np_utils.to_categorical(dev[1], 2)))

        train = (sequence.pad_sequences(train[0],
                                        maxlen=max_sen_length), train[1])

        dev = (sequence.pad_sequences(dev[0],
                                      maxlen=max_sen_length), dev[1])
    elif stage == "test":
        if feat_suffix == "dev":
            loader_test.load_labeled_features("window%i_max%i_%s_%s" % (window, max_words, feat_suffix, model_type))
        else:
            loader_test.load_labeled_features("window%i_max%i_%s_%s" % (window, max_words, model_type, feat_suffix))

        test = (loader_test.input_vector, loader_test.target_vector)

        test = (test[0], numpy.array(keras.utils.np_utils.to_categorical(test[1], 2)))

        test = (sequence.pad_sequences(test[0],
                                        maxlen=max_sen_length), test[1])




    model, callbacks_list = init_lstm_model(embeddings.embedding_matrix, stage, model_type)


    if stage == "train":
        print model.summary()
        model.fit(train[0],
                  train[1],
                  validation_data=(dev[0], dev[1]),
                  nb_epoch=15,
                  batch_size=150,
                  #class_weight = class_weight,
                  #shuffle=False,
                  callbacks=callbacks_list
                  )

    elif stage == "test":
        y_hyp = model.predict(test[0],batch_size=150)
        #y_hyp = numpy.argmax(y_hyp, axis=1)
        #y_hyp = adjust_content_vector(y_hyp)
        #y_ref = calculate_euclidean_dist(dev[1])
        #y_hyp = calculate_euclidean_dist(y_hyp)
        y_ref = numpy.argmax(test[1],axis=1)
        f = open("test_gold_%s.txt"%model_type,'w')
        for num in y_ref:
            f.write(str(num))
            f.write("\n")
        f.close()
        f = open("test_hyp_%s.txt"%model_type,'w')
        for num in y_hyp:
            f.write(str(num))
            f.write("\n")

        f.close()

if __name__ == "__main__":
    main()
