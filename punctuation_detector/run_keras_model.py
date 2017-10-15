from DataLoader import DataLoader

from keras.models import Sequential
from keras.layers import Dense, Activation, \
    LSTM, Dropout, Embedding, Lambda, Bidirectional, Merge, TimeDistributed
from keras.layers import Conv1D, MaxPooling1D
import keras
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import pickle

from WordEmbeddings import WordEmbeddings

import sys

import numpy
import math
from sklearn.metrics import f1_score

prev_context = int(sys.argv[3])
next_context = int(sys.argv[4])
max_sen_length = int(sys.argv[3])

embedding_vector_length = 50

layers = [300, 300]


load_epoch = ""

def init_lstm_model(train, dev,test, embeddings=None, stage="train"):
    model_options = "seq-fulltrain-finalver"
    if train:
        in_dim = len(train[0][0])
    elif test:
        in_dim = len(test[0][0])
    out_dim = 2
    

    model = Sequential()
    model.add(Embedding(len(embeddings),
                        embedding_vector_length,
                        input_length=max_sen_length,
                        weights=[embeddings]))
    model.add(Dropout(0.5))
    model.layers[0].trainable = False

    
    for lstm_layer in layers:
        model.add(Bidirectional(LSTM(lstm_layer, return_sequences=True)))
        model.add(Dropout(0.3))

    model.add(TimeDistributed(Dense(out_dim,
                    activation='softmax')))
    adam = keras.optimizers.Adam(lr=0.001,
                                 beta_1=0.9,
                                 beta_2=0.999,
                                 epsilon=1e-08,
                                 decay=0.0)

    if stage == "test" or stage == "dev":
        load_model = sys.argv[8]
        model.load_weights('%s/%s.hdf5'%(sys.argv[7],load_model))
    else:
        if len(sys.argv) >= 9:
            load_model = sys.argv[8]
            model.load_weights('%s/%s.hdf5'%(sys.argv[7],load_model))
            model_options += "-contepoch2%s"%load_model[-2:]
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=[fbetascore])

    if stage == "train":
        filepath='%s/model-%s-embed%s-c%sc%s-%sx%s-epoch{epoch:02d}.hdf5'%(sys.argv[7],
                                                                 model_options,
                                                                 embedding_vector_length,
                                                                 prev_context,
                                                                 next_context,
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
 
    stage = sys.argv[1]
    embeddings_path = sys.argv[2]
    sen_size = int(sys.argv[3])
    shift = int(sys.argv[4])
    vocab_path = sys.argv[5]
    feat_path = sys.argv[6]
    feat_suffix = sys.argv[9]

    vocab = pickle.load(open(vocab_path,'r'))

    embeddings = WordEmbeddings(embeddings_path)
    embeddings.construct_embedding_matrix(vocab)

    loader_train = DataLoader(feat_path)
    loader_dev = DataLoader(feat_path)
    loader_test = DataLoader(feat_path)

    if stage == "test":
        loader_dev.load_labeled_features("sen%i_shift%i_test"%(sen_size, shift))
        X_test = loader_test.input_vector
        Y_test = loader_test.target_vector
        test = (X_test, Y_test)
        test = (test[0],keras.utils.np_utils.to_categorical(test[1], 2))
        test = (sequence.pad_sequences(test[0],
                                        maxlen=max_sen_length), test[1])
    elif stage == "train":
        loader_train.load_labeled_features("sen%i_shift%i_train"%(sen_size, shift))
        loader_dev.load_labeled_features("sen%i_shift%i_dev"%(sen_size, shift))

        train = (loader_train.input_vector, loader_train.target_vector)
        dev = (loader_dev.input_vector, loader_dev.target_vector)


        train = (train[0],numpy.array([keras.utils.np_utils.to_categorical(tag_seq, 2) for tag_seq in train[1]]))
        dev = (dev[0],numpy.array([keras.utils.np_utils.to_categorical(tag_seq, 2) for tag_seq in dev[1]]))

        train = (sequence.pad_sequences(train[0],
                                        maxlen=max_sen_length), train[1])
        
        dev = (sequence.pad_sequences(dev[0],
                                        maxlen=max_sen_length), dev[1])
    elif stage == "dev":
        loader_dev.load_labeled_features("sen%i_shift%i_%s"%(sen_size, shift, feat_suffix))


        dev = (loader_dev.input_vector, loader_dev.target_vector)

        dev = (dev[0],numpy.array([keras.utils.np_utils.to_categorical(tag_seq, 2) for tag_seq in dev[1]]))

        dev = (sequence.pad_sequences(dev[0],
                                        maxlen=max_sen_length), dev[1])

        test = dev


	


    model, callbacks_list = init_lstm_model(train, dev,test, embeddings.embedding_matrix, stage)

    print model.summary()

    #class_weight = {0 : 1, 1: 1}

    if stage == "train":
        model.fit(train[0],
                  train[1],
                  validation_data=(dev[0], dev[1]),
                  nb_epoch=10,
                  batch_size=150,
                  #class_weight = class_weight,
                  #shuffle=False,
                  callbacks=callbacks_list
                  )

    elif stage == "test":
        y_hyp = model.predict(test[0],batch_size=16)

        y_ref = numpy.argmax(test[1],axis=1)
        f = open("test_target.txt",'w')
        for num in y_ref:
            f.write(str(num))
            f.write("\n")
        f.close()
        f = open("test_hyp.txt",'w')
        for num in y_hyp:
            f.write(str(num))
            f.write("\n")
    elif stage == "dev":
        y_hyp = model.predict(test[0],batch_size=500)

        y_ref = [numpy.argmax(seq,axis=1) for seq in test[1]]
        f = open("dev_target.txt",'w')
        for num in y_ref:
            for tag in num:
                f.write(str(tag) + "\t")
            f.write("\n")
        f.close()
        f = open("dev_hyp.txt",'w')
        for num in y_hyp:
            for tag in num:
            	f.write(str(tag[1]) + "\t")
            f.write("\n")

        f.close()

if __name__ == "__main__":
    main()
