import math
import re
import sys
import numpy as np

"""
TFIDF Class for CS224N Winter 2019
Author: Matt Linker

Reference: All code shown here was written by the author. Much of it is modified from assignment work in
CS124, Winter 2018. Credit for concepts used in get_unique_words and prepare_data to the CS124 teaching
team. index, compute_tfidf, and get_tfidf were modified directly from functions I wrote in CS124.

"""

class TFIDF:

	def __init__(self, docs):
		""" 
		Initialize TF-IDF class.
		@param docs: List of list of strings of all documents in training set
		Might need to massage data a bit to fit in here
		"""
		self.vocab = []
		self.docs = docs
		self.alphanum = re.compile('[^a-zA-Z0-9]')
		self.tfidf = {}


	def index(self):
        """
        Build an index of the documents. This should be called after docs are loaded and in the correct format.
        This also needs to be called after get_unique_words is called for get_unique_words to work correctly.
        """
        inv_index = {}
        for word in self.vocab:
            inv_index[word] = []

        id_to_bag_of_words = {}
        for d, doc in enumerate(self.docs):
            index = 0
            bag_of_words = {}
            for word in doc:
                inv_index[word].append((d, index))
                index += 1
                if word not in bag_of_words:
                    bag_of_words[word] = 1
                else:
                    bag_of_words[word] += 1
            id_to_bag_of_words[d] = bag_of_words
        self.docs = id_to_bag_of_words
        self.inv_index = inv_index

	def get_unique_words(self):
		""" 
		Get the set of unique words in corpus.
		@return words: The set of unique words found
		"""
		words = set()
		for d in self.docs:
			for w in d:
				words.add(w)
		return words

	def prepare_data(self):
		""" 
		Populate vocab using get_unique_words
		"""
		self.vocab = [w for w in self.get_unique_words()]

	def compute_tfidf(self):
		""" 
		Compute TF-IDF for every word/document pair in set. Populates self.tfidf
		"""
        for word in self.vocab:
            count = 0
            docCounts = {}
            for d in range(len(self.docs)):
                arr = self.docs[d]
                if word in arr:
                    count+=arr[word]
                    docCounts[d] = arr[word]
            for d in docCounts.keys():
                if word not in self.tfidf:
                    self.tfidf[word] = {}
                tf_log = float(docCounts[d])
                tf = 1+math.log10(tf_log)
                idf_log = float(len(self.docs))/float(len(docCounts))
                idf = math.log10(idf_log)
                self.tfidf[word][d] = (tf*idf)

    #TODO: Add a get_tfidf variant for an arbitrary new doc???
    #TODO: use stemmed queries?

    def get_tfidf(self, word):
    	""" 
    	Get the tf-idf score for a given doc in the vocab
    	@param word: the word from a query
    	@return the tf-idf score between that word and document
    	"""
        if word not in self.tfidf:
            return 0.0
        if document not in self.tfidf[word]:
            return 0.0
        return self.tfidf[word][document]

