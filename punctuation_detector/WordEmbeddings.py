import numpy

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

    def construct_embedding_matrix(self, vocab):
        dim = self._load_embedding_vectors_with_vocab(vocab)
        lda_dim = 150
        oov = []
        self.embedding_matrix = numpy.zeros((len(vocab.keys()) + 1, dim))
        #self.lda_matrix = pickle.load(open('./lda_matrix_%i.pckl'%lda_dim,'rb'))
        for word, idx in vocab.iteritems():

            #num = random.randint(0,lda_dim - 1)
            if word in self.embeddings:
                #zer = numpy.array([0]*lda_dim)
                #top_topics = self.lda_matrix[idx].argsort()[-7:][::-1]
                #zer[top_topics] = 1
                #self.embeddings[word].extend(zer)
                self.embedding_matrix[idx] = self.embeddings[word]
            else:
                #zer = numpy.array([0]*lda_dim)
                #top_topics = self.lda_matrix[idx].argsort()[-7:][::-1]
                #zer[top_topics] = 1
                embed = [0]*dim

                oov.append(word)
                #embed.extend(zer)
                self.embedding_matrix[idx] = embed


        with open('oov_words.txt','w') as file:
            file.write('\n'.join(oov))

        del self.embeddings
