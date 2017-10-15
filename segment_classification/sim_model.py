#Similarity model module#
#This module calculates sentence similarity and runs a kNN to get top k matches#

import numpy
import math
from scipy import spatial

class SimilarityModel:
    def __init__(self):
        self.response_bank_rep = []
        self.model_weights = []
        self.features = '$'

    #calculate the vector representations of the data bank and stores them#
    def init_response_bank(self, response_bank, embedding_models, tfidf_constructor=None, features='tfidf', weights=[]):
        self.features = features
        self.model_weights = weights

        if self.features == 'tfidf':

            for weight in weights:
                self.response_bank_rep.append(dict())

            for key, response in response_bank.iteritems():
                explication_rep    = tfidf_constructor.construct_document_tfidf_features(response)

                self.response_bank_rep[0][key] = explication_rep

            for key, response in response_bank.iteritems():
                explication_rep    = embedding_models[0].get_doc_vec_centroid(response)
                self.response_bank_rep[1][key] = explication_rep


        elif self.features == 'mean-we':
            for model, weight in zip(embedding_models, weights):
                self.response_bank_rep.append(dict())

            for key, response in response_bank.iteritems():
                for model, model_rep in zip(embedding_models, self.response_bank_rep):
                    explication_rep    = model.get_doc_vec_centroid(response)

                    model_rep[key] = explication_rep

        elif self.features == 'idf-weighted-mean-we':

            for model, weight in zip(embedding_models, weights):
                self.response_bank_rep.append(dict())


            for key, response in response_bank.iteritems():
                for model, model_rep in zip(embedding_models, self.response_bank_rep):
                    explication_rep    = model.get_doc_vec_centroid_weighted(response,tfidf_constructor.idf)

                    model_rep[key] = explication_rep

    def _get_sim_scores(self, hyp_segment, word_embeddings,tfidf_constructor, ref):
        max_sim = float("-inf")
        best_response_key = -1
        response_scores = dict()

        if self.features == 'tfidf' and tfidf_constructor != None:
            hyp_segment_rep = tfidf_constructor.construct_document_tfidf_features(hyp_segment)

        elif self.features == 'mean-we' or tfidf_constructor == None:
            #get the vector representation of the user entered sentence#
            hyp_segment_rep = word_embeddings.get_doc_vec_centroid(hyp_segment)

        elif self.features == 'idf-weighted-mean-we':
            #get the vector representation of the user entered sentence#
            hyp_segment_rep = word_embeddings.get_doc_vec_centroid_weighted(hyp_segment, tfidf_constructor.idf)

        #loop over all the responses and calculates similarity with each one#
        for key, response in ref.iteritems():


            #calculate the semantic similarity #
            #between the user question and one data bank candidate#
            sim = self._calculate_cosine_similarity(hyp_segment_rep,
                                                      response)

            if sim > max_sim:
                max_sim = sim
                best_response_key = key
            response_scores[key] = sim
        return response_scores

    def _merge_models_score(self, sim_scores):
        response = []
        for key in sim_scores[0].keys():
            final_score = 0
            for model_scores, weight in zip(sim_scores, self.model_weights):
                key_model_score = model_scores[key]
                final_score += weight * key_model_score
            response.append((key,final_score))
        return response

    #get the top k matches for a user question#
    def get_best_k_matches(self, hyp_segment, embedding_models, tfidf_constructor, k):

        if self.features == 'tfidf':
            models_score = []
            model_sim_scores = self._get_sim_scores(hyp_segment, None,tfidf_constructor, self.response_bank_rep[0])
            models_score.append(model_sim_scores)

            model_sim_scores = self._get_sim_scores(hyp_segment, embedding_models[0],None, self.response_bank_rep[1])
            models_score.append(model_sim_scores)

            response_scores = self._merge_models_score(models_score)

        elif self.features == 'mean-we':
            models_score = []

            for model, ref in zip(embedding_models,self.response_bank_rep):
                model_sim_scores = self._get_sim_scores(hyp_segment, model, tfidf_constructor, ref)
                models_score.append(model_sim_scores)

            response_scores = self._merge_models_score(models_score)

        elif self.features == 'idf-weighted-mean-we':
            models_score = []

            for model, ref in zip(embedding_models,self.response_bank_rep):
                model_sim_scores = self._get_sim_scores(hyp_segment, model, tfidf_constructor, ref)
                models_score.append(model_sim_scores)

            response_scores = self._merge_models_score(models_score)

        #sort the list of candidates according to their similarity score#
        response_scores = sorted(response_scores, key=lambda pair: pair[1])

        #get the top k candidates#
        top_k = [pair[0] for pair in response_scores[-k:]]

        #reverse the list of candidates (higher score first)#
        top_k = top_k[::-1]

        return top_k

    #1 - euclidean distance between two vectors#
    def _calculate_euclidean_similarity(self, vec1, vec2):
        return 1 - numpy.linalg.norm(vec1 - vec2)

    #1 - cosine distance between two vectors#
    def _calculate_cosine_similarity(self, vec1, vec2):

        if numpy.count_nonzero(vec1) == 0 or numpy.count_nonzero(vec2) == 0:
            return 0

        return 1 - spatial.distance.cosine(vec1, vec2)

