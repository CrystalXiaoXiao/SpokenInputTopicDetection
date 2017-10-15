from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
import logging
import sys

class TrainableEmbeddings:
    def __init__(self):
        #Mapping between words and custom trained word2vec#
        self.word2vec_model = None



    def train_word2vec_model(self, data_path):
        print data_path
        #model parameters#
        dim = 50
        context = 10
        min_count = 1
        workers = 4
        sg = 1

        #setting up logger#
        logger = logging.getLogger("Training")

        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
        logging.root.setLevel(level=logging.INFO)
        logger.info("Running %s" % ' '.join(sys.argv))
        max_length = 0

        with open(data_path, 'r') as f:
            for line in f.readlines():
                max_length = max(max_length, len(line))
                if len(line) == 0:
                    print line

        model_params = {
        #vector dimension#
        'size': dim,
        #context window#
        'window': context,
        #minimum word frequency to consider#
        'min_count': min_count,
        #number of cores#
        'workers': workers, #max(1, multiprocessing.cpu_count() - 1),
        #skip-gram model#
        'sg': sg
        }

        print model_params

        #using LineSentence to stream article directly from disk#
        word2vec_model = Word2Vec(LineSentence(data_path, max_sentence_length=max_length),
                        **model_params)

        #save model to disk#
        model_str = ''
        if sg:
            model_str = "skip-gram"
        else:
            model_str = "cbow"

        #train a word2vec model#
        word2vec_model.save_word2vec_format('./models/w2v_mincount1_%id_%icontext_%s.txt'%(dim,context,model_str),binary=False)
        self.word2vec_model = word2vec_model

    def load_word2vec_model(self, model_path):
        self.word2vec_model = Word2Vec.load_word2vec_format(model_path, binary=False)


embed = TrainableEmbeddings()
embed.train_word2vec_model('./documents.txt')