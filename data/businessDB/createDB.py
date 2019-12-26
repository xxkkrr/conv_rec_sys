import json

with open('business_dict.json', 'r') as f:
    business_dict = json.load(f)
with open('restaurants_info_filter.json', 'r') as f:
    restaurants_info_filter = json.load(f)
with open('restaurants_slot_value_filter.json', 'r') as f:
    restaurants_slot_value_filter = json.load(f)

businessDB_dict = {}
for business_name in business_dict.keys():
    if business_name not in restaurants_info_filter.keys():
        continue
    business_id = business_dict[business_name]
    current_business_dict = {}
    for slot_name in restaurants_info_filter[business_name].keys():
        value_name = str(restaurants_info_filter[business_name][slot_name])
        value_id = restaurants_slot_value_filter[slot_name][value_name]
        current_business_dict[slot_name] = value_id
    businessDB_dict[business_id] = current_business_dict

with open('businessDB_dict.json', 'w') as f:
    json.dump(businessDB_dict, f)