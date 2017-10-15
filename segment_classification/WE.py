#This module is used to load GloVe word vectors#
#and calculate sentence vector representation#

import numpy
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
import os
import sys


class WordVectorConstructor:
    def __init__(self, embed_path):
        #The system's vocabulary#
        self.vocab = None

        #Mapping between words and their GloVe vector: i.e 'football' -> [0.1,...,-0.5]#
        self.embeddings = dict()

        #Path for GloVe embedding file#
        self.path = embed_path

    #extracts the unique words from a corpus of sentences#
    def _set_corpus_vocab(self, corpus):
        self.vocab = dict()
        for key, doc in corpus.iteritems():
            for sent in doc:
                for word in sent:
                    self.vocab[word] = ""

    #loads the mapping between the words and their word vectors#
    #words not in vocab are kept in the mappings#
    def _load_embedding_vectors(self, path):
        with open(path) as f:
            for line in f.readlines():
                line = line.split()
                if len(line) < 3:
                    continue
                word = line.pop(0)
                line = [float(x) for x in line]

                self.embeddings[word] = numpy.array(line)

        print('loaded embeddings with dim: '+str(len(self.embeddings[self.embeddings.keys()[0]])))
        print('vocab size: '+ str(len(self.embeddings)))


    #loads the mapping between the words and their word vectors#
    #words not in vocab are discarded to save space#
    def _load_embedding_vectors_with_vocab(self, path,vocab):
        if self.vocab is None:
            print("run 'set_corpus_vocab' before loading embeddings ")
            return

        with open(path) as f:
            for line in f:
                line = line.split()
                word = line.pop(0)
                if word not in vocab:
                    continue

                line = [float(x) for x in line]

                self.embeddings[word] = numpy.array(line)

        print('loaded embeddings with dim: '+str(len(embeddings[embeddings.keys()[0]])))
        print('vocab size: '+ str(len(embeddings)))

    #returns the vector representation of a sentence#
    #sentence words centroid: mean of words in the sentence#
    def get_doc_vec_centroid(self, document):
        dim = self.embeddings[self.embeddings.keys()[0]].shape[0]
        doc_centroid = numpy.zeros(dim)
        word_count = 0
        for word in document:
            if word in self.embeddings:
                word_vec = self.embeddings[word]
                doc_centroid = numpy.add(doc_centroid,word_vec)
                word_count += 1

        #if word_count > 0:
        #    doc_centroid /= word_count

        return doc_centroid

    def get_doc_vec_centroid_weighted(self, document, idf_weights):

        dim = self.embeddings[self.embeddings.keys()[0]].shape[0]
        doc_centroid = numpy.zeros(dim)
        word_count = 0
        for word in document:
            if word in self.embeddings:
                weight = idf_weights['<UNK>']
                if word in idf_weights:
                    weight = idf_weights[word]


                word_vec = weight * numpy.array(self.embeddings[word])
                doc_centroid = numpy.add(doc_centroid,word_vec)
                word_count += 1

        #if word_count > 0:
        #    doc_centroid /= word_count

        return doc_centroid

    #Initialize the word embedding module#
    def init_word_embeddings(self, vocab=None, corpus=None):
        if vocab is not None:
            self._set_corpus_vocab(corpus)
            self._load_embedding_vectors_with_vocab(self.path, vocab)
            return

        self._load_embedding_vectors(self.path)
