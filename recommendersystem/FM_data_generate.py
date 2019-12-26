import json

with open('../dialogue_data.json', 'r') as f:
	dialogue_data_dict = json.load(f)

user_dict = {}
user_count = 0
business_dict = {}
business_count = 0

for key in dialogue_data_dict:
	user_id = dialogue_data_dict[key]["user_id"]
	business_id = dialogue_data_dict[key]["business_id"]
	if user_id not in user_dict:
		user_dict[user_id] = user_count
		user_count += 1
	if business_id not in business_dict:
		business_dict[business_id] = business_count
		business_count += 1

with open('user_dict.json', 'w') as f:
	json.dump(user_dict, f)

with open('business_dict.json', 'w') as f:
	json.dump(business_dict, f)