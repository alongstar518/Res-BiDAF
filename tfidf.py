import math
import re
import sys
import pickle
import numpy as np

"""
TFIDF Class for CS224N Winter 2019
Author: Matt Linker

Reference: All code shown here was written by the author. A few lines are modified from assignment work in
CS124, Winter 2018. Credit for concepts used in get_unique_words to the CS124 teaching team. The rest of 
the functions I wrote from scratch. While called tf-idf, this implementation ultimately just uses idf as a
proxy for word rarity and term-matching.

Usage:
1. Create TFIDF class: scorer = TFIDF(docs)
2. Prepare data: scorer.prepare_data() (pre-process all docs in training set)
3. get scores to feed as a feature: 
	-normalized_additive_idf returns the mean idf value
	-normalized_additive_idf_ignore_common_words returns the mean idf value, ignoring words that are common across docs
	-min_idf returns the minimum idf value over the query
	-max_idf returns the maximum idf value over the query
4. To cut down on pre-process (lengthy) redundancy, save pre-processed values using save_to_pickle. They can
later be populated into a new (empty) TFIDF object using get_from_pickle.


Note: Running step 2 is a time-consuming process. I recommend either a. outputting data to a pickle file and
recovering it, or b. code architecture such that you only run it once for train and test combined (on train
data both times).
"""

class TFIDF:

	def __init__(self, docs=[]):
		""" 
		Initialize TF-IDF class.
		@param docs: List of strings. Each string is a doc in training set
		Might need to massage data a bit to fit in here. Use empty doc if 
		reloading from pickle.
		"""
		self.vocab = []
		self.docs = docs
		self.word2Ind = {}
		#self.count = 0.0

	def get_unique_words(self):
		""" 
		Get the set of unique words in corpus.
		@return words: The set of unique words found
		"""
		words = set()
		for d in self.docs:
			for w in d:
				words.add(w.lower())
		return words

	def prepare_data(self):
		""" 
		Populate vocab using get_unique_words
		Feed data into compute_idf_scores
		"""
		self.docs = [d.split() for d in self.docs]
		self.vocab = [w for w in self.get_unique_words()]
		self.word2Ind = dict((y,x) for x,y in enumerate(self.vocab))
		vocab_len = len(self.vocab)
		self.idf_count = np.zeros(vocab_len)
		self.compute_idf_scores()


	def compute_idf_scores(self):
		"""
		Compute idf scores for train dataset. You will need this data loaded
		from train at test time, but should not load test data into it.
		"""
		count = 0.0
		for d in self.docs:
			count += 1.0
			words_found = set(d)
			for w in words_found:
				self.idf_count[self.word2Ind[w.lower()]] += 1.0


		self.idf_score = [math.log10(1+count/x) for x in self.idf_count]


	def get_word_score(self, word):
		"""
		Get the idf score of an arbitrary word. OOV words have max score
		@param word: word to check score for
		@return score: float of the idf score of input word
		"""
		if word not in self.word2Ind:
			return 1.0
		score = self.idf_score[self.word2Ind[word]]
		return score

	def normalized_additive_idf(self, query_string):
		"""
		Get the additive idf score of a given punctuation-stripped, case-insensitive string, 
		normalized to the length of this string. Note that this can be the query text itself,
		or could also be a sentence from a given document, depending on the BiDAF feature we
		are adding.
		@param query_string: the string to get an additive score for
		@return score: float of normalized additive idf score - average idf score over query
		"""
		score = 0
		splitted = query_string.lower().split()
		norm_len = len(splitted)
		for w in splitted:
			score+= (self.get_word_score(w) / norm_len)
		return score

	def normalized_additive_idf_ignore_common_words(self, query_string, threshold_frequency=0.5):
		"""
		Get the additive idf score of a given punctuation-stripped, case-insensitive string, 
		normalized to the length of this string, completely ignoring common words, as determined by
		the given threshold. Note that this can be the query text itself, or could also be a sentence 
		from a given document, depending on the BiDAF feature we are adding.
		@param query_string: the string to get an additive score for
		@param threshold_frequency: frequency of word at which we ignore it when averaging
		@return score: float of normalized additive idf score - average idf score over query
		"""
		threshold_score = math.log10(1.0 + 1.0/threshold_frequency)
		score = 0
		splitted = query_string.lower().split()
		norm_len = 0
		for w in splitted:
			tmp_score = self.get_word_score(w)
			if tmp_score > threshold_score:
				norm_len += 1
				score += tmp_score
		return score/norm_len

	def max_idf(self, query_string):
		"""
		Get the maximum idf score of a given punctuation-stripped, case-insensitive string. 
		Note that this can be the query text itself, or could also be a sentence from a given 
		document, depending on the BiDAF feature we are adding.
		@param query_string: the string to get a max score for
		@return score: float of max idf score
		"""
		score = -1
		splitted = query_string.lower().split()
		norm_len = len(splitted)
		for w in splitted:
			test_score = self.get_word_score(w)
			if test_score > score:
				score = test_score
		return score

	def min_idf(self, query_string):
		"""
		Get the minimum idf score of a given punctuation-stripped, case-insensitive string. 
		Note that this can be the query text itself, or could also be a sentence from a given 
		document, depending on the BiDAF feature we are adding.
		@param query_string: the string to get a min score for
		@return score: float of min idf score
		"""
		score = 2
		splitted = query_string.lower().split()
		norm_len = len(splitted)
		for w in splitted:
			test_score = self.get_word_score(w)
			if test_score < score:
				score = test_score
		return score

	def save_to_pickle(self):
		"""
		Saves the given IDF score table and word index lookup map to a pickle file. Useful so
		we don't need to recompute every time. NOTE: This function cannot be called until AFTER
		you have called prepare_data on the dataset. You should not need to call this more than
		once overall, since the training dataset only needs to be computed once, then we can
		save_to_pickle and access whenever we need to.
		"""
		with open('./data/word2Ind.pickle', 'wb') as handle:
			pickle.dump(self.word2Ind, handle, protocol=pickle.HIGHEST_PROTOCOL)
		with open('./data/idf_score.pickle', 'wb') as handle:
			pickle.dump(self.idf_score, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def get_from_pickle(self):
		"""
		Retrieves an IDF score table and word index lookup map from a call to save_to_pickle.
		To use, generate a new "empty" TFIDF object, then call on it with no data. NOTE: This
		function will need a previous call to save_to_pickle to work, and additionally will
		NOT populate any fields other than the two shown below. This means that once you load, 
		the only safe functions to use are those where you get scores (step 3 in the instructions).
		"""
		with open('./data/word2Ind.pickle', 'rb') as handle:
			self.word2Ind = pickle.load(handle)
		with open('./data/idf_score.pickle', 'rb') as handle:
			self.idf_score = pickle.load(handle)


