import numpy as np
import math
import os.path

import pickle

class TFIDFConstructor():
    def __init__(self, documents):
        self.tf = dict()
        self.idf = dict()
        self.N = 0

        self.tfidf_features = dict()

        self.vocab = dict()
        self.idx_vocab = dict()

        if os.path.exists("./vocab.pckl"):
            self.vocab = pickle.load( open( "./vocab.pckl", "rb" ) )
            for key, val in self.vocab.iteritems():
                self.idx_vocab[val] = key
        else:
            self.construct_vocab(documents)
            pickle.dump( self.vocab, open( "./vocab.pckl", "wb" ) )

        if os.path.exists("./idf.pckl"):
            self.idf = pickle.load( open("./idf.pckl", "rb" ) )
        else:
            self.construct_idf_values(documents)
            pickle.dump( self.idf, open("./idf.pckl", "wb" ) )



    def construct_vocab(self, documents):
        idx = 0
        for key, doc in documents.iteritems():
            for word in doc:
                if word not in self.vocab:
                    self.vocab[word] = idx
                    self.idx_vocab[idx] = word
                    idx += 1

    def construct_tf_features(self, documents):
        for key, doc in documents.iteritems():
            doc_feat = np.zeros( len(self.vocab.keys()) )
            for word in doc:
                idx = self.vocab[word]
                doc_feat[idx] += 1

            #normalize BOW TF vector
            doc_feat = doc_feat / len(doc)

            self.tf[key] = doc_feat

    def construct_document_tf_features(self, doc):
        doc_feat = np.zeros( len(self.vocab.keys()) + 1)
        for word in doc:
            if word in self.vocab:
                idx = self.vocab[word]
                doc_feat[idx] += 1
            else:
                doc_feat[len(self.vocab.keys())] += 1

        #normalize BOW TF vector
        doc_feat = doc_feat / len(doc)

        return doc_feat

    def construct_tfidf_features(self, documents):
        self.construct_tf_features(documents)

        for key, bow in self.tf.iteritems():
            bow_tfidf = np.zeros(self.vocab.keys())

            for idx, tf in enumerate(bow):
                word = self.idx_vocab[idx]
                idf_val = self.idf[word]

                tfidf_val = idf_val * tf
                bow_tfidf[idx] = tfidf_val
            self.tfidf_features[key] = bow_tfidf

    def construct_document_tfidf_features(self, doc):
        tf_bow = self.construct_document_tf_features(doc)

        bow_tfidf = np.zeros(len(self.vocab.keys()) + 1)

        for idx, tf in enumerate(tf_bow):
            if idx in self.idx_vocab:
                word = self.idx_vocab[idx]
                idf_val = self.idf[word]
            else:
                idf_val = math.log( (self.N + 1) / 1 )

            tfidf_val = idf_val * tf
            bow_tfidf[idx] = tfidf_val

        return bow_tfidf

    def construct_idf_values(self, documents):
        word_occ_in_docs = np.zeros(len(self.vocab.keys()))
        i = 1
        for key, doc in documents.iteritems():
            if i%50000 == 0:
                print "idf: processed %f..."%(i*100/(1.0*len(documents.keys())))

            word_in_doc = np.zeros(len(self.vocab.keys()))
            for word in doc:
                idx = self.vocab[word]
                word_in_doc[idx] = 1

            word_occ_in_docs += word_in_doc
            i+=1

        N = len(documents.keys())
        self.N = N
        self.idf['<UNK>'] = math.log( (self.N + 1) / 1 )
        for i, val in enumerate(word_occ_in_docs):
            word_occ_in_docs[i] = math.log( (N + 1) / (val + 1) )
            word = self.idx_vocab[i]

            self.idf[word] = word_occ_in_docs[i]

