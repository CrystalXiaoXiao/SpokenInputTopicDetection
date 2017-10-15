import pickle
from TDTReader import TDTCorpus
from nltk.corpus import stopwords
import string
import numpy
import math
import pickle
import gensim
from gensim import corpora
import lda
import logging
import random

class DataLoader:
    def __init__(self, path):


        self.lda_matrix = None



        self.input_vector = None
        self.target_vector = None
        self.feat_path = path




    def _get_corpus_vocab(self, corpus):
        vocab = dict()
        word_id = 1
        for key, doc in corpus.iteritems():
            for sent in doc:
                for word in sent:
                    if word not in vocab:
                        vocab[word] = word_id
                        word_id += 1
        vocab['<bnd>'] = word_id
        print "unique words: %i"%len(vocab.keys())

        return vocab

    def _load_embedding_vectors_with_vocab(self, vocab):
        self.embeddings = dict()
        with open(self.embeddings_path) as f:
            for line in f:
                line = line.split()
                word = line.pop(0)
                if word not in vocab:
                    continue

                line = [float(x) for x in line]

                self.embeddings[word] = (line)

        #self.embeddings["<bnd>"] = numpy.zeros(len(self.embeddings[self.embeddings.keys()[0]]))

        print('loaded embeddings with dim: '+str(len(self.embeddings[self.embeddings.keys()[0]])))
        print('embedding vocab size: '+ str(len(self.embeddings)))
        return len(self.embeddings[self.embeddings.keys()[0]])



    def load_labeled_features(self, segment="train"):
        cnt_words = 0
        cnt_oov = 0
        oov = []
        self.input_vector = []
        feats_file_str = "%s/feats_%s.txt"%(self.feat_path,segment)
        targets_file_str = "%s/targets_%s.txt"%(self.feat_path,segment)

        with open(feats_file_str) as feats_file:
            for line in feats_file:
                line = line.strip().split()
                line = [str(word) for word in line]
                self.input_vector.append(line)


        self.target_vector = numpy.loadtxt(targets_file_str)

    def prepare_lda(self, vocab):

        documents, sent_bnds = self._get_corpus_sentences(vocab)

        vocab = self._prepare_lda(documents, sent_bnds, vocab)
        print "done saving lda matrix"

        pickle.dump(vocab,open('vocab_lda.pckl','wb'))

    def save_documents_as_txt(self):
        documents, sent_bnds = self._get_corpus_sentences(remove_stop=False)
        file = open('./documents.txt','w')
        for key, doc in documents.iteritems():
            for sent in doc:
                sent = ' '.join(sent)
                file.write(sent)
                file.write("\n")



    def _get_words(self, vocab, x):
        words = []
        for idx in x:
            for key in vocab:
                if vocab[key] == idx:
                    words.append(key)
                    break
        return words

    def _load_vocab_from_text(self, path):
        vocab = dict()
        ids = 1
        with open(path) as file:
            for line in file.readlines():
                line = line.split()
                vocab[line[1]] = int(line[0]) + 1
                ids += 1

        vocab['<bnd>'] = ids
        return vocab

    def _prepare_lda(self, documents, sent_bnds, vocab):
        num_topics = 150
        logging.basicConfig(format='%(message)s', level=logging.INFO)
        self.lda_matrix = numpy.zeros((len(vocab.keys()) + 1, num_topics))

        self._set_list_stories_fast(documents, sent_bnds)

        dict = corpora.Dictionary(self.stories)
        dict.save_as_text('./models/dict.txt')
        vocab = self._load_vocab_from_text('./models/dict.txt')

        bow_feats = [dict.doc2bow(story) for story in self.stories]

        print len(bow_feats)
        lda_model = gensim.models.LdaMulticore(corpus=bow_feats,
                                                num_topics=num_topics,
                                              id2word=dict,
                                               chunksize=150,
                                              passes=2,
                                               workers=4)

        lda_model.print_topics(20)
        for i in range(num_topics):
            p_w_t = lda_model.get_topic_terms(i,len(vocab)-1)
            for tuple in p_w_t:
                self.lda_matrix[tuple[0] + 1][i] = tuple[1]

        #for word, word_id in vocab.iteritems():
        #    print word, word_id
        #    if word_id -1 < len(vocab) - 1:
        #        print lda_model.get_term_topics(word_id -1)


        pickle.dump(self.lda_matrix, open('lda_matrix_%i_train.pckl'%num_topics,'wb'))

        return vocab