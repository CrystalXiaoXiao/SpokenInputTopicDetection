from TrainableEmbeddings import TrainableEmbeddings
from TDTReader import TDTCorpus
import pickle
import string
import sys

def _preprocess_text(text):

        text = text.replace('\'',' ')
        text = text.replace('-',' ')
        text = text.translate(None,string.punctuation)
        text = text.lower()
        text = text.split()


        filtered_text = text
        #stemmer = PorterStemmer()
        #stemmed_text = [stemmer.stem(word) for word in filtered_text]


        return filtered_text

def _get_corpus_sentences(path):
        documents = dict()
        corpus = pickle.load(open(path,'r'))

        print('loading sentences...')
        for key in corpus.text_corpus_bnds.keys():
            key = key.strip()
            val = corpus.text_corpus_bnds[key]
            document = []


            text = ' '.join(val)
            text = text.split('<bnd>')

            for sent in text:
                sent = _preprocess_text(sent)
                document.append(sent)
            del document[-1]
            documents[key] = document
        print('finished loading sentences')
        return documents, corpus.sent_boundaries

#path = "/home/mohamed/Desktop/scripts/tdt_reader/corpus_%s_annotated_data_nltk_boundaries.pckl"%sys.argv[1]

#documents, bnds = _get_corpus_sentences(path)

#file = open('traintxt','w')
#for key, doc in documents.iteritems():
#    for sent in doc:
#        file.write(' '.join(sent))
#        file.write('\n')
#print "finished creating data file"
word2vec_trainer = TrainableEmbeddings()
word2vec_trainer.train_word2vec_model("/media/mohamed/24E8F168E8F1389E/UbuntuFiles/text_punct_train/train_text")