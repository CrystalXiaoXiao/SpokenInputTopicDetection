import sys
import pickle
from TDTReader import TDTCorpus
from nltk.corpus import stopwords
import string

import logging
import numpy
from gensim import corpora
import gensim

class FeatureConstructor:
    def __init__(self, path=None):
        self.stories = []
        if path:
            self.corpus = pickle.load(open(path, 'r'))

    def _preprocess_text(self, text, remove_stop=True):
        stopwordlist = stopwords.words('english')

        text = text.replace('\'',' ')
        text = text.replace('-',' ')
        text = text.translate(None,string.punctuation)
        text = text.lower()
        text = text.split()

        if remove_stop:
            filtered_text = [word for word in text if word not in stopwordlist]
        else:
            filtered_text = text
        #stemmer = PorterStemmer()
        #stemmed_text = [stemmer.stem(word) for word in filtered_text]


        return filtered_text

    # extracts words and boundaries from corpus
    # preprocesses input words
    def _get_corpus_sentences(self, vocab_train, remove_stop=True):
        documents = dict()
        words = 0
        words_cont = 0
        oov_words = 0
        oov_words_cont = 0
        sentences = 0
        print('loading sentences...')
        for key in self.corpus.text_corpus_bnds.keys():
            key = key.strip()
            val = self.corpus.text_corpus_bnds[key]
            document = []


            text = ' '.join(val)
            text = text.split('<bnd>')

            for sent in text:


                sent_all = self._preprocess_text(sent,False)
                words += len(sent_all)
                for w in sent_all:
                    if w not in vocab_train:
                        oov_words += 1
                sent = self._preprocess_text(sent,remove_stop)
                for w in sent:
                    if w not in vocab_train:
                        oov_words_cont += 1
                words_cont += len(sent)

                document.append(sent)
                sentences += 1

            del document[-1]
            documents[key] = document
        print words
        print words_cont
        print sentences
        print oov_words
        print oov_words_cont
        print('finished loading sentences')
        return documents, self.corpus.sent_boundaries

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

    def _set_list_stories_fast(self, documents, boundaries):
        self.stories = []

        for file_name in documents.keys():
            document = documents[file_name]
            boundary = boundaries[file_name]

            story = []
            for sent, is_bnd in zip(document, boundary):
                story.extend(sent)

                if is_bnd:
                    self.stories.append(story)
                    story = []


            del documents[file_name]
            del boundaries[file_name]

    def prepare_lda(self, vocab, dim):

        documents, sent_bnds = self._get_corpus_sentences(vocab)
        num_topics = dim
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
            p_w_t = lda_model.get_topic_terms(i, len(vocab) - 1)
            for tuple in p_w_t:
                self.lda_matrix[tuple[0] + 1][i] = tuple[1]

        # for word, word_id in vocab.iteritems():
        #    print word, word_id
        #    if word_id -1 < len(vocab) - 1:
        #        print lda_model.get_term_topics(word_id -1)


        pickle.dump(self.lda_matrix, open('lda_matrix_%i_train.pckl' % num_topics, 'wb'))

        return vocab

    #constructs feature/target vector pairs for topic segmentation
    def build_labeled_features_segment_detector(self,vocab, window=5, max_words=100, suffix=""):

        documents, sent_bnds = self._get_corpus_sentences(vocab)

        left_context = ( window/2 ) - 1
        right_context = window/2

        #vocab = self._get_corpus_vocab(documents)

        #print "preparing lda..."
        #vocab = self._prepare_lda(documents, sent_bnds, vocab)
        #print "done saving lda matrix"
        #pickle.dump(vocab,open('vocab_lda.pckl','wb'))
        #return

        #return


        del self.corpus


        print "building features..."
        features_file = open("feats_window%i_max%i_%s.txt"%(window, max_words, suffix),'w')
        targets_file = open("targets_window%i_max%i_%s.txt"%(window, max_words, suffix),'w')
        documents_lengths = open("doc_length_window%i_max%i_%s.txt"%(window, max_words, suffix),'w')
        self.input_vector = []
        self.target_vector = []

        for key, doc in documents.iteritems():
            for id, sen in enumerate(doc):
                feat = []
                label = sent_bnds[key][id]
                prev_win = id - (left_context)#id - (window/2)
                next_win = id + right_context

                if prev_win < 0:
                    prev_win = 0

                if next_win >= len(doc):
                    next_win = len(doc) - 1

                lc = doc[prev_win:id+1]
                lc = [vocab[word] if word in vocab else vocab['<bnd>'] for sen in lc for word in sen]
                if len(lc) >= (right_context*max_words):
                    clip = len(lc) - (right_context*max_words)
                    lc = lc[clip:]
                else:
                    pad = (right_context*max_words) - len(lc)
                    pad = [0]*pad
                    lc = pad + lc

                rc = doc[id+1:next_win+1]
                rc = [vocab[word] if word in vocab else vocab['<bnd>'] for sen in rc for word in sen]

                if len(rc) >= (right_context*max_words):
                    rc = rc[:right_context*max_words]
                else:
                    pad = (right_context*max_words) - len(rc)
                    pad = [0]*pad
                    rc = rc + pad

                feat = lc + rc


                feat = ' '.join(map(str,feat))
                target = str(label)

                features_file.write(feat+"\n")
                targets_file.write(target+"\n")
            documents_lengths.write(str(len(doc))+"\n")

        print "done building features"



def main():
    if len(sys.argv) > 5:
        segment = sys.argv[1]
        vocab_file = sys.argv[2]
        num_sentences = int(sys.argv[3])
        max_words = int(sys.argv[4])
        path = sys.argv[5]

        vocab = pickle.load(open(vocab_file))

        featCons = FeatureConstructor(path)
        featCons.build_labeled_features_segment_detector(vocab, num_sentences, max_words, segment)
    else:
        vocab_file = sys.argv[1]
        path = sys.argv[2]
        dim = int(sys.argv[3])

        vocab = pickle.load(open(vocab_file))

        featCons = FeatureConstructor(path)
        lda_vocab = featCons.prepare_lda(vocab, dim)

        pickle.dump(lda_vocab, open("./vocab_lda_latest.pckl",'w'))



if "__main__" == __name__:
    main()
