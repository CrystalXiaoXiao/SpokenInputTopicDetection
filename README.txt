--------------------------------------------------
|    INCLUDED PROJECTS				 |             
--------------------------------------------------

*punctuation_detector

A keras implementation of a period punctuation detector. This system inserts a period to a stream of unstructured text.


*topic_based_segmentation

A keras implementation of a system that divides a stream of sentences into segments where 
adjacent segments are different in topics.


*segment_classification

A kNN implementation (based on Word Embeddings and TFIDF features) that classifies a given segment into one of the available topics given their summaries.

--------------------------------------------------
|    DATASET USED				 |             
--------------------------------------------------
TDT-2 Corpus

--------------------------------------------------
|    DEPENDANCIES				 |             
--------------------------------------------------
keras 1.2.0
gensim
word2vec
numpy
nltk


--------------------------------------------------
|    CORPUS                                      |              
--------------------------------------------------
For the experiments, a TDTCorpus pickled object (*.pckl) is loaded that has two attributes: "text_corpus_bnds" and "sent_boundaries". Any object having these two attributes can work as input

Explaining "text_corpus_bnds":
A dict() object having as keys the ids of the documents, and values the list of tokens (words) of each document.
Tokens (words) that mark an end of sentence contain a marker "<bnd>"

EXAMPLE:

text_corpus_bnds = 	{	"document_id_1":["This","is","a", "sentence.<bnd>", "this", "is", "another", "sentence<bnd>"]
				,
				"document_id_2":["This","is","A", "non-pre-procces", "sentence.<bnd>", "this", "is", "another", "sentence<bnd>"]
				,
				"document_id_3":["This","is","a", "sentence.<bnd>", "this", "is", "another", "sentence<bnd>"]
			}




Explaining "sent_boundaries":

A dict() object having as keys the ids of the documents, and values the list of sentence-based TOPIC boundaries. So each "i"th index in the list
represents whether there is a topic change after "i"th sentence in the document or not.

EXAMPLE:

sent_boundaries = 	{	"document_id_1":[0,0,0,1,0,0,0,0,0,1] 
				#document id 1 has two story segments: segment from sentence 1 > 4, segment from sentence 5 > 10
				,
				"document_id_2":[0,1,0,0,1] 
				#document id 2 has two story segments: segment from sentence 1 > 2, segment from sentence 3 > 5
				,
				"document_id_3":[0,0,1,0,0,1,1] 
				#document id 1 has three story segments: segment from sentence 1 > 3, segment from sentence 4 > 6 
				#and segment containing sentence 7
			}


--------------------------------------------------
|    VOCAB                                       |              
--------------------------------------------------
Another input used in experiments is the vocabulary of the training set.
A vocab is a dict stored in a pickle object (*.pckl)
The dict() has as keys the words and the value is a unique id to that word.
Word ids should start from 1.
A special word "<bnd>" is added to the dict and has the id: len(vocab) + 1. 
This <bnd> word is reserved for unkown words.
Words should be lowercased.

EXAMPLE:


vocab = 	{	"word"	: 1,
			"usa"	: 2,
			"<bnd>" : 3
		}





