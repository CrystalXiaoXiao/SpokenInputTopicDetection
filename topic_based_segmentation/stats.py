import pickle

from TDTReader import TDTCorpus
from TDTReader import Segment
from nltk.corpus import stopwords
import string

def _preprocess_text(text, remove_stop=True):
    stopwordlist = stopwords.words('english')

    text = text.replace('\'', ' ')
    text = text.replace('-', ' ')
    text = text.translate(None, string.punctuation)
    text = text.lower()
    text = text.split()

    if remove_stop:
        filtered_text = [word for word in text if word not in stopwordlist]
    else:
        filtered_text = text
    # stemmer = PorterStemmer()
    # stemmed_text = [stemmer.stem(word) for word in filtered_text]


    return filtered_text

def get_corpus_sentences(corpus):
    documents = dict()
    words = 0
    words_cont = 0
    oov_words = 0
    oov_words_cont = 0
    sentences = 0

    for key in corpus.text_corpus_bnds.keys():
        val = corpus.text_corpus_bnds[key]
        document = []

        text = ' '.join(val)
        text = text.split('<bnd>')

        for sent in text:
            sent_all = _preprocess_text(sent, False)
            words += len(sent_all)

            sent = _preprocess_text(sent, True)
            words_cont += len(sent)

            document.append(sent)
            sentences += 1

        del document[-1]
        documents[key] = document
    print "number of words: %d "%words
    print "number of content words: %d"%words_cont
    print "number of sentences: %d"%sentences

    return documents, corpus.sent_boundaries, words, words_cont

PATH_TEST="/home/mohamed/Desktop/scripts/tdt_reader/corpus_test_annotated_data_nltk_boundaries.pckl"
PATH_DEV="/home/mohamed/Desktop/scripts/tdt_reader/corpus_dev_annotated_data_nltk_boundaries.pckl"

vocab_dev = pickle.load(open('../punctuation_detector/vocab_dev.pckl','r'))
vocab_train = pickle.load(open('./vocab_lda_latest.pckl','r'))
vocab_test = pickle.load(open('../punctuation_detector/vocab_test.pckl','r'))

corpus_test =  pickle.load(open(PATH_TEST, 'r'))
corpus_dev  =  pickle.load(open(PATH_DEV , 'r'))

print "---------------------"
print "getting dev stats..."
docs, bnds, words, words_cont = get_corpus_sentences(corpus_dev)
oov = 0
for key, val in docs.iteritems():
    for sent in val:
       for word in sent:
           if word not in vocab_train:
               oov += 1
print "running oov words: %d"%oov
print "oov running %.2f"%((oov*100)/(1.0*words_cont))
print "---------------------"

print "----------------------"
print "getting test stats..."
docs, bnds, words, words_cont = get_corpus_sentences(corpus_test)
oov = 0
for key, val in docs.iteritems():
    for sent in val:
       for word in sent:
           if word not in vocab_train:
               oov += 1
print "running oov words: %d"%oov
print "oov running %.2f"%((oov*100)/(1.0*words_cont))
print "----------------------"