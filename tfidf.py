import math
import re
import sys
import pickle
import numpy as np
import string

"""
TFIDF Class for CS224N Winter 2019
Author: Matt Linker

Reference: All code shown here was written by the author. A few lines are modified from assignment work in
CS124, Winter 2018. Credit for concepts used in get_unique_words to the CS124 teaching team. The rest of 
the functions I wrote from scratch. While called tf-idf, this implementation ultimately just uses idf as a
proxy for word rarity and term-matching.

Usage:
1. Create TFIDF class: scorer = TFIDF(docs)
2. Prepare data: scorer.prepare_data() (pre-process all docs in training set). Use include_tf to pre-compute
tf values for different contexts in training (not sure if necessary)
3. get scores to feed as a feature: 
	-normalized_additive_idf returns the mean idf value
	-normalized_additive_idf_ignore_common_words returns the mean idf value, ignoring words that are common across docs
	-min_idf returns the minimum idf value over the query
	-max_idf returns the maximum idf value over the query
4. To cut down on pre-process (lengthy) redundancy, save pre-processed values using save_to_pickle. They can
later be populated into a new (empty) TFIDF object using get_from_pickle.

For "Full" TF-IDF, instead of step 3, run get_tfidf_normalized_additive or get_tfidf_normalized_additive_ignore_common
on your desired query string and context string. Query string can either be a "true" query string, or a
candidate answer string.


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
		self.tf_lookup_table = {}
		self.punct = str.maketrans({key: None for key in string.punctuation})
		#self.count = 0.0

	def get_unique_words(self, include_tf=False):
		""" 
		Get the set of unique words in corpus.
		@param include_tf: Whether to build with tf tables for training data. Will allow faster tf calculations
		but causes pre-processing to run slower and take more memory.
		@return words: The set of unique words found
		"""
		words = set()

		for d in self.docs:
			if include_tf:
				word_vec = {}
				for w in d:
					words.add(w.lower().translate(self.punct))
					if w not in word_vec:
						word_vec[w.lower().translate(self.punct)] = 1
					else:
						word_vec[w.lower().translate(self.punct)] = word_vec[w.lower().translate(self.punct)] + 1
				self.tf_lookup_table[" ".join(d)] = word_vec
			else:
				for w in d:
					words.add(w.lower().translate(self.punct))
		return words

	def prepare_data(self, include_tf=False):
		""" 
		Populate vocab using get_unique_words
		Feed data into compute_idf_scores
		@param include_tf: Whether to pre-compute raw tf on training data
		"""
		self.docs = [d.lower().translate(self.punct).split() for d in self.docs]
		self.vocab = [w for w in self.get_unique_words(include_tf=include_tf)]
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

	def get_tf_score(self, word, context_string):
		"""
		Get the term-frequency score of an arbitrary word and context string, not found
		in the contexts dataset.
		@param word: The word to compute the score for
		@param context_string: The context paragraph
		@return ret: The tf score for word with context context_string
		"""
		word = word.lower()
		context_list = context_string.lower().translate(self.punct).split()
		ret = context_list.count(word)
		return ret


	def get_word_score(self, word):
		"""
		Get the idf score of an arbitrary word. OOV words have high score
		@param word: word to check score for
		@return score: float of the idf score of input word
		"""
		if word not in self.word2Ind:
			return 5.0 #arbitrary high score consistent with appearance in 1/100000 contexts
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
		splitted = query_string.lower().translate(self.punct).split()
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
		splitted = query_string.lower().translate(self.punct).split()
		norm_len = 0
		for w in splitted:
			tmp_score = self.get_word_score(w)
			if tmp_score > threshold_score:
				norm_len += 1
				score += tmp_score
		if norm_len == 0:
			return -1
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
		splitted = query_string.lower().translate(self.punct).split()
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
		splitted = query_string.lower().translate(self.punct).split()
		norm_len = len(splitted)
		for w in splitted:
			test_score = self.get_word_score(w)
			if test_score < score:
				score = test_score
		return score

	def get_tfidf_normalized_additive(self, query_string, context_string):
		"""
		Produces a tf-idf score for a given query string and context string, using a normalized additive
		score. At training time, if a lookup table is available it will attempt to look up existing tf 
		scores when iterating over the training set. Otherwise, it will simply compute tf for each context 
		string. All scores will need to be individually computed at test time. 
		@param query_string: The query string to compute a score for
		@param context_string: The context string used to answer the question
		@return: the tf-idf score of the query string against the context string
		"""
		count = 0
		tot = 0
		context_string = " ".join(context_string.lower().translate(self.punct).split())
		if context_string in self.tf_lookup_table:
			table = self.tf_lookup_table[context_string]
			for w in query_string.lower().translate(self.punct).split():
				if w in table:
					tf = table[w]
				else:
					tf = 0
				idf = self.normalized_additive_idf(w)
				count += 1
				tot = tot + tf*idf
		else:
			for w in query_string.lower().translate(self.punct).split():
				tf = self.get_tf_score(w, context_string)
				idf = self.normalized_additive_idf(w)
				if idf>0:
					count += 1
					tot = tot + tf*idf
		return float(tot)/float(count)


	def get_tfidf_normalized_additive_ignore_common(self, query_string, context_string, threshold_frequency=0.5):
		"""
		Produces a tf-idf score for a given query string and context string, using a normalized additive
		score. This version ignores common words as set by the threshold frequency. At training time, if
		a lookup table is available it will attempt to look up existing tf scores when iterating over the 
		training set. Otherwise, it will simply compute tf for each context string. All scores will need to
		be individually computed at test time. Only "uncommon" words are included for the normalization
		factor.
		@param query_string: The query string to compute a score for
		@param context_string: The context string used to answer the question
		@param threshold_frequency: The frequency of occurence at which a word will be ignored
		@return: the tf-idf score of the query string against the context string
		"""
		count = 0
		tot = 0
		context_string = " ".join(context_string.lower().translate(self.punct).split())
		if context_string in self.tf_lookup_table:
			table = self.tf_lookup_table[context_string]
			for w in query_string.lower().translate(self.punct).split():
				if w in table:
					tf = table[w]
				else:
					tf = 0
				idf = self.normalized_additive_idf_ignore_common_words(w, threshold_frequency=threshold_frequency)
				if idf>0:
					count += 1
					tot = tot + tf*idf
		else:
			for w in query_string.lower().translate(self.punct).split():
				tf = self.get_tf_score(w, context_string)
				idf = self.normalized_additive_idf_ignore_common_words(w, threshold_frequency=threshold_frequency)
				if idf>0:
					count += 1
					tot = tot + tf*idf
		return float(tot)/float(count)


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
		with open('./data/tflookup.pickle', 'wb') as handle:
			pickle.dump(self.tf_lookup_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
		with open('./data/tflookup.pickle', 'rb') as handle:
			self.tf_lookup_table = pickle.load(handle)

"""
#sanity check code - feel free to uncomment for your own checks
str1 = "Hello how are you"
str2 = "I'm doing well how are you"
str3 = "Fine thank you"
str4 = "When in the course of human events"
dicts = []
dicts.append(str1)
dicts.append(str2)
dicts.append(str3)
dicts.append(str4)
scorer = TFIDF(dicts)
scorer.prepare_data(include_tf=True)
print(scorer.get_tfidf_normalized_additive_ignore_common("Hello how are human?", "Fred is a human fool!"))
"""


