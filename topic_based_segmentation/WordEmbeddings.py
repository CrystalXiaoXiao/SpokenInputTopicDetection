import numpy
import pickle

class WordEmbeddings:
    def __init__(self, embeddings_path):
        self.embeddings_path = embeddings_path
        self.embeddings = None
        self.embedding_matrix = None

        self.lda_matrix = None

    def _load_embedding_vectors_with_vocab(self, vocab):
        self.embeddings = dict()
        first = False
        with open(self.embeddings_path) as f:
            for line in f:
                if first == True:
                    first = False
                    continue
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

    def construct_embedding_matrix(self, vocab, lda=False, topics=150):
        if not lda:
            dim = self._load_embedding_vectors_with_vocab(vocab)
            self.embedding_matrix = numpy.zeros((len(vocab.keys()) + 1, dim))
        else:
            lda_dim = topics
            self.lda_matrix = pickle.load(open('./lda_matrix_%i_train.pckl' % lda_dim, 'rb'))
            self.embedding_matrix = numpy.zeros((len(vocab.keys()) + 1, lda_dim))
        oov = []
        if lda:
            for word, idx in vocab.iteritems():
                if word in vocab:
                    zer = numpy.array([0]*lda_dim)
                    top_topics = self.lda_matrix[idx].argsort()[-7:][::-1]
                    zer[top_topics] = 1
                    self.embedding_matrix[idx] = zer
                else:
                    embed = [0] * lda_dim
                    oov.append(word)
                    self.embedding_matrix[idx] = embed
        else:
            for word, idx in vocab.iteritems():

                if word in self.embeddings:
                    self.embedding_matrix[idx] = self.embeddings[word]
                else:
                    embed = [0]*dim
                    oov.append(word)
                    self.embedding_matrix[idx] = embed


        with open('oov_words.txt','w') as file:
            file.write('\n'.join(oov))

        del self.embeddings
