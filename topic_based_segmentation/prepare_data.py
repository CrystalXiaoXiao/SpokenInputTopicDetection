from DataLoader import DataLoader

import numpy
import math


def main():

    path = None
    path = "/home/mohamed/Desktop/scripts/tdt_reader/corpus_test_annotated_data_nltk_boundaries.pckl"

    loader = DataLoader(path)
    #print "saving documents as text..."
    #loader.save_documents_as_txt()
    #print "done saving documents"
    #return
    loader.build_labeled_features_segment_detector(6,10)


if "__main__" == __name__:
    main()