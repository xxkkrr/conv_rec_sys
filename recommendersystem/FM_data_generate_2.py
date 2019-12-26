from nltk.util import ngrams
import json
import re
from collections import defaultdict

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

with open('vocab_dict.json', 'r') as f:
	ngrams_dict = json.load(f)

with open('user_dict.json', 'r') as f:
	user_dict = json.load(f)

with open('business_dict.json', 'r') as f:
	business_dict = json.load(f)

with open('../dialogue_data.json', 'r') as f:
	dialogue_data_dict = json.load(f)

FM_data_list = []

for key in dialogue_data_dict:
	content_list = dialogue_data_dict[key]["content"]
	current_utterance_list = []
	user_id = user_dict[dialogue_data_dict[key]["user_id"]]
	business_id = business_dict[dialogue_data_dict[key]["business_id"]]
	score = dialogue_data_dict[key]["stars"]
	for utterance_dict in content_list:
		utterance = utterance_dict["nl"]
		utterance_idx = []
		ngrams_list = generate_ngrams(utterance, 3)
		for ngram_ in ngrams_list:	
			ngram_str = ' '.join(ngram_)
			ngrams_idx = ngrams_dict[ngram_str]
			utterance_idx.append(ngrams_idx)
		current_utterance_list.append(utterance_idx)
		FM_dict = {}
		FM_dict['user_id'] = user_id
		FM_dict['business_id'] = business_id
		FM_dict['score'] = score
		FM_dict['data'] = current_utterance_list.copy()
		FM_data_list.append(FM_dict.copy())

with open('FM_data_list.json', 'w') as f:
	json.dump(FM_data_list, f)