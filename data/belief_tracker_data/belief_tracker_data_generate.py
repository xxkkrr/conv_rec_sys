from nltk.util import ngrams
import json
import re

def generate_ngrams(s, n):
	'''
	s: document, str type
	n: n-gram

	output: list of n-grams 
	'''
	tokens = [token for token in s.split(" ") if token != ""]
	output = []
	for n_ in range(1,n):
		output.extend(list(ngrams(tokens, n_)))
	return output

ngrams_dict = {}
ngrams_idx = 0

with open('../dialogue_data.json', 'r') as f:
	dialogue_data_dict = json.load(f)

for key in dialogue_data_dict:
	content_list = dialogue_data_dict[key]["content"]
	for utterance_dict in content_list:
		utterance = utterance_dict["nl"]
		ngrams_list = generate_ngrams(utterance, 3)
		for ngram_ in ngrams_list:
			ngram_ = ' '.join(ngram_)
			if ngram_ not in ngrams_dict:
				ngrams_dict[ngram_] = ngrams_idx
				ngrams_idx += 1

with open('vocab_dict.json', 'w') as f:
    json.dump(ngrams_dict, f, indent=4)


