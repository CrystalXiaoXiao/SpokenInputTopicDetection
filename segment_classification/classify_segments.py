from TDTReader import TDTCorpus
import pickle
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import string
import random

from sim_model import SimilarityModel
from WE import WordVectorConstructor
from TFIDF import TFIDFConstructor

def _preprocess_text(text, remove_stop=True):
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

def preprocess_segments(story_segments):
    pp_story_segments = dict()
    for key, segment in story_segments.iteritems():
        segment = _preprocess_text(segment)
        pp_story_segments[key] = segment

    return pp_story_segments

def load_topics(path):
    topics = dict()
    with open(path) as file:
        for line in file:
            line = line.split()
            t_id = int(line[0])
            topic_name = ' '.join(line[1:])

            topics[t_id] = topic_name

    return topics

def load_explications(path):
    topic_explication = dict()
    with open(path) as file:
        for line in file:
            line = line.split()
            t_id = line[0]
            topic_expl = ' '.join(line[1:])

            sentences = sent_tokenize(topic_expl)
            sentences = [sen.strip() for sen in sentences]
            sentences = ' '.join(sentences)

            sentences = _preprocess_text(sentences)

            topic_explication[t_id] = sentences
    return topic_explication


print "loading assets..."
embed_path = "../word2vec_trainer/models/w2v_train_300d_10context_cbow.txt"

#segments = pickle.load(open("story_segments.pckl",'r'))

#segments = preprocess_segments(segments)

#pickle.dump(segments,open("preprocessed_story_segments.pckl",'w'))

segments = pickle.load(open("preprocessed_story_segments.pckl"))

topics_annot = pickle.load(open("story_topics.pckl",'r'))

topics = [0]*100

for key, val in topics_annot.iteritems():
    for annot in val:

        topics[annot[0] - 1] += 1


topics_list = load_topics('./topics.txt')
topics_expl = load_explications('./topic_explication.txt')

print "preparing word embeddings..."
word_embeddings = WordVectorConstructor(embed_path)
word_embeddings.init_word_embeddings()

print "preparing idf features..."
tfidf = TFIDFConstructor(segments)

models = [word_embeddings]

for tfidf_weight in [0.1, 0.3, 0.5, 0.7, 0.9]:
    embedding_weight = 1 - tfidf_weight

    print "preparing similarity model..."
    print "tfidf: %f embedding: %f"%(tfidf_weight, embedding_weight)

    sim_model = SimilarityModel()
    sim_model.init_response_bank(topics_expl, models, tfidf_constructor=tfidf, features='tfidf', weights = [tfidf_weight, embedding_weight])

    total = 0
    correct_top1 = 0
    correct_top3 = 0
    correctness = dict()

    max_len = 0
    for key, segment in segments.iteritems():
        if len(segment) > max_len:
            max_len = len(segment)
        predicted = False
        predicted_top1 = False
        top_k = sim_model.get_best_k_matches(segment, models, tfidf, 3)
        top_k = [int(candidate) for candidate in top_k]

        segment_tags = topics_annot[key]

        for tag in segment_tags:
            topic_id = tag[0]
            for index, id in enumerate(top_k):
                if topic_id == id:
                    if index == 0:
                        predicted_top1 = True
                    predicted = True

        correctness[key] = predicted

        if predicted:
            correct_top3 += 1

        if predicted_top1:
            correct_top1 += 1

        total += 1

    length_hist = [0]*(int(max_len/50) + 1)
    error_length = [0]*(int(max_len/50) + 1)

    for key, segment in segments.iteritems():
        length_hist[len(segment)/50] += 1
        if correctness[key] != True:
            error_length[len(segment)/50] += 1



    print "classification accuracy top 1: %.2f"%((100*correct_top1)/(total*1.0))
    print "classification accuracy top 3: %.2f"%((100*correct_top3)/(total*1.0))

