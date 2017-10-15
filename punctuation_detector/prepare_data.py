from DataLoader import DataLoader

import numpy
import math
import sys

def main():

    path = None
    stage = sys.argv[1]

    path = "/home/mohamed/Desktop/scripts/tdt_reader/corpus_%s_annotated_data_nltk_boundaries.pckl"%stage

    loader = DataLoader(path)
    #print "saving documents as text..."
    #loader.save_documents_as_txt()
    #print "done saving documents"
    #return
    #loader.build_labeled_features_punct_detector(5,4, stage)
    loader.build_labeled_features_punct_detector_tagged_seq(18*5, 6*5, stage)

if "__main__" == __name__:
    main()
