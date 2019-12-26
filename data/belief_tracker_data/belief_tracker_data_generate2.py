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

with open('restaurants_slot_value_filter.json', 'r') as f:
    slot_value_dict = json.load(f)

with open('../dialogue_data.json', 'r') as f:
	dialogue_data_dict = json.load(f)

aspect_all_data = defaultdict(list)

for key in dialogue_data_dict:
	content_list = dialogue_data_dict[key]["content"]
	current_utterance_list = []
	current_slot_value_dict = {}
	for utterance_dict in content_list:
		utterance = utterance_dict["nl"]
		current_slot_value_dict.update(utterance_dict["slots"])
		utterance_idx = []
		ngrams_list = generate_ngrams(utterance, 3)
		for ngram_ in ngrams_list:	
			ngram_str = ' '.join(ngram_)
			ngrams_idx = ngrams_dict[ngram_str]
			utterance_idx.append(ngrams_idx)
		current_utterance_list.append(utterance_idx)
		for slot in current_slot_value_dict:
			value = current_slot_value_dict[slot]
			data_dict = {}
			data_dict["data"] = current_utterance_list.copy()
			data_dict["label"] = slot_value_dict[slot][value]
			aspect_all_data[slot].append(data_dict)

with open('aspect_all_data.json', 'w') as f:
	json.dump(aspect_all_data, f)