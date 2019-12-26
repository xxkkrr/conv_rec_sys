import json
import csv

with open("dialogue_data.json", 'r') as f:
    dialogue_data_dict = json.load(f)

count_all = 0
count_high = 0
high_ratings_list = []
for key in dialogue_data_dict:
    user_name = dialogue_data_dict[key]["user_id"]
    business_name = dialogue_data_dict[key]["business_id"]
    score = float(dialogue_data_dict[key]["stars"])
    count_all += 1
    if score > 4.9:
        count_high += 1
        high_ratings_list.append(tuple([user_name, business_name, score]))

print("count_high:", count_high)
print("count_all:", count_all)

with open('high_ratings.csv','w') as f:
    f_csv = csv.writer(f)
    headers = ["user_id", "business_id", "stars"]
    f_csv.writerow(headers)
    f_csv.writerows(high_ratings_list)