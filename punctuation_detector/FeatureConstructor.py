import sys
import pickle
from TDTReader import TDTCorpus
from nltk.corpus import stopwords
import string

class FeatureConstructor:
    def __init__(self, path=None):
        if path:
            self.corpus = pickle.load(open(path,'r'))

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

    #extracts words and boundaries from corpus
    #preprocesses input words 
    def _get_corpus_sentences(self, remove_stop=True):
        documents = dict()

        print('loading sentences...')
        for key in self.corpus.text_corpus_bnds.keys():
            key = key.strip()
            val = self.corpus.text_corpus_bnds[key]
            document = []


            text = ' '.join(val)
            text = text.split('<bnd>')

            for sent in text:
                sent = self._preprocess_text(sent,remove_stop)
                document.append(sent)
            del document[-1]
            documents[key] = document
        print('finished loading sentences')
        return documents, self.corpus.sent_boundaries
    
    #convert corpus consisting of list of sentences to a corpus of list of pseudo-sentences
    #from: [["This", "is", "sentence", "one"],["This", "is", "sentence", "two"]]
    #to: [["This", "is"], ["sentence", "one"],["This", "is"],["sentence", "two"]]
    #window = 2
    def _convert_sentences_2_word_windows(self, documents, sent_boundaries, window):
        word_bnds = dict()
        documents_word = dict()
        for key, doc in documents.iteritems():
            bnds = sent_boundaries[key]
            documents_word[key] = []
            word_bnds[key] = []
            word_window = []
            for sent, bnd in zip(doc, bnds):
                for id, word in enumerate(sent):
                    if len(word_window) < window:
                        word_window.append(word)
                    else:
                        documents_word[key].append(word_window[:])
                        word_window = [word]
                    word_bnds[key].append(0)

                word_bnds[key][-1] = 1
                if len(word_window) > 0:
                    while len(word_window) < window:
                        word_window.append("<bnd>")
                    documents_word[key].append(word_window[:])
                    word_window = []
            if len(word_window) > 0:
                while len(word_window) < window:
                        word_window.append("<bnd>")
                documents_word[key].append(word_window[:])

        return documents_word, word_bnds
    
    #create feature/target vector pairs for punctuation detection
    def build_labeled_features_punct_detector_tagged_seq(self,vocab, sen_size=18, shift_value=6, suffix=""):
        documents, sent_bnds = self._get_corpus_sentences(remove_stop=False)
        documents, sent_bnds = self._convert_sentences_2_word_windows(documents, sent_bnds, 1)


        del self.corpus


        print "building features..."
        features_file = open("feats_sen%i_shift%i_%s.txt"%(sen_size,shift_value,suffix),'w')
        targets_file = open("targets_sen%i_shift%i_%s.txt"%(sen_size,shift_value,suffix),'w')
        file_markers = open("marker_sen%i_shift%i_%s.txt"%(sen_size,shift_value,suffix), 'w')
        documents_lengths = open("doc_length_sen%i_shift%i_%s.txt"%(sen_size,shift_value,suffix),'w')

        total_bnds = 0
        in_vocab = 0
        segments = 0

        for key, doc in documents.iteritems():
            doc_bnds = sent_bnds[key]

            id = 0
            while id < len(doc):
                if id+sen_size <= len(doc):
                    feat = doc[id:id+sen_size]
                    feat = [vocab[word] if word in vocab else vocab["<bnd>"] for sen in feat for word in sen]
                    target = doc_bnds[id:id+sen_size]
                else:
                    feat = doc[id:]
                    target = doc_bnds[id:]
                    feat = [vocab[word] if word in vocab else vocab["<bnd>"] for sen in feat for word in sen]
                    while len(feat) < sen_size:
                        feat.append(0)
                        target.append(0)

                total_bnds += 1

                feat = ' '.join(map(str,feat))
                target = [str(label) for label in target]
                target = ' '.join(target)
                if id == 0:
                    file_markers.write("1\n")
                else:
                    file_markers.write("0\n")
                features_file.write(feat+"\n")
                targets_file.write(target+"\n")

                id += shift_value
            documents_lengths.write(str(len(doc))+"\n")

        print "done building features"



def main():
    segment =       sys.argv[1]
    vocab_file =    sys.argv[2]
    sen_size =      int(sys.argv[3])
    shift =         int(sys.argv[4])
    path = 	    sys.argv[5]

    vocab = pickle.load(open(vocab_file))

    featCons = FeatureConstructor(path)
    featCons.build_labeled_features_punct_detector_tagged_seq(vocab, sen_size, shift, segment)

if "__main__" == __name__:
    main()
