import random
import json
import csv

dialogue_pattern = \
{
	"categories":[
				{ 
				"slots":["categories", "state"],
				"nl":"Find me some $categories$ restaurants in $state$",
				},
				{ 
				"slots":["categories", "state"],
				"nl":"I'm looking for $categories$ restaurants in $state$",
				},
				{ 
				"slots":["categories", "state"],
				"nl":"I want $categories$ restaurants in $state$",
				},
				{ 
				"slots":["categories", "state"],
				"nl":"I need $categories$ food in $state$",
				},
				{ 
				"slots":["categories", "state"],
				"nl":"Find me some $categories$ food in $state$",
				},
				{ 
				"slots":["categories", "state"],
				"nl":"I'm looking for $categories$ food in $state$",
				},
				{ 
				"slots":["categories", "state"],
				"nl":"I want $categories$ food in $state$",
				},
				{ 
				"slots":["categories", "state"],
				"nl":"I need $categories$ food in $state$",
				},
				{ 
				"slots":["categories", "city"],
				"nl":"Find me some $categories$ restaurants in $city$",
				},
				{ 
				"slots":["categories", "city"],
				"nl":"I'm looking for $categories$ restaurants in $city$",
				},
				{ 
				"slots":["categories", "city"],
				"nl":"I want $categories$ restaurants in $city$",
				},
				{ 
				"slots":["categories", "city"],
				"nl":"I need $categories$ food in $city$",
				},
				{ 
				"slots":["categories", "city"],
				"nl":"Find me some $categories$ food in $city$",
				},
				{ 
				"slots":["categories", "city"],
				"nl":"I'm looking for $categories$ food in $city$",
				},
				{ 
				"slots":["categories", "city"],
				"nl":"I want $categories$ food in $city$",
				},
				{ 
				"slots":["categories", "city"],
				"nl":"I need $categories$ food in $city$",
				},
				{ 
				"slots":["categories"],
				"nl":"I need $categories$ food",
				},
				{ 
				"slots":["categories"],
				"nl":"I want $categories$ food",
				},
				{ 
				"slots":["categories"],
				"nl":"Find me $categories$ food",
				},	
				{ 
				"slots":["categories"],
				"nl":"I'm looking for $categories$ food",
				},
				{ 
				"slots":["categories"],
				"nl":"I need $categories$ restaurants",
				},
				{ 
				"slots":["categories"],
				"nl":"I want $categories$ restaurants",
				},
				{ 
				"slots":["categories"],
				"nl":"Find me $categories$ restaurants",
				},	
				{ 
				"slots":["categories"],
				"nl":"I'm looking for $categories$ restaurants",
				},		
				{ 
				"slots":["categories"],
				"nl":"$categories$",
				},
				{ 
				"slots":["categories"],
				"nl":"$categories$ food",
				},
				{ 
				"slots":["categories"],
				"nl":"$categories$ restaurants",
				},
				],
	"state":[
				{ 
				"slots":["state"],
				"nl":"$state$",
				},	
				{ 
				"slots":["state"],
				"nl":"In $state$",
				},	
			],
	"city":[
				{ 
				"slots":["city"],
				"nl":"$city$",
				},	
				{ 
				"slots":["city"],
				"nl":"In $city$",
				},		
			],
	"price":[
				{ 
				"slots":["price"],
				"nl":"$price$ price",
				},	
				{ 
				"slots":["price"],
				"nl":"$price$ pricing",
				},		
				{ 
				"slots":["price"],
				"nl":"$price$",
				},				
			],
	"stars":[
				{ 
				"slots":["stars"],
				"nl":"$stars$",
				},	
				{ 
				"slots":["stars"],
				"nl":"$stars$ stars",
				},		
			]
}

with open('restaurants_info_filter.json', 'r') as f:
	restaurants_info_dict = json.load(f)

dialogue_list = {}
dialogue_id = 0
with open('ratings_filter.csv') as f:
	f_csv = csv.reader(f)
	headers = next(f_csv)
	for row in f_csv:
		user_id = row[0]
		business_id = row[1]
		stars = row[2]

		new_dialogue = {}
		new_dialogue["user_id"] = user_id
		new_dialogue["business_id"] = business_id
		new_dialogue["stars"] = stars
		new_dialogue["content"] = []

		slot_list = ["state", "city", "price", "stars"]
		random.shuffle(slot_list)
		slot_list.insert(0, "categories")
		said_slot = set()

		for slot in slot_list:
			if slot in said_slot:
				continue
			pattern_dict = random.choice(dialogue_pattern[slot])
			utterance_content = pattern_dict["nl"]
			utterance_dict = {}
			utterance_dict["slots"] = {}
			for current_slot in pattern_dict["slots"]:
				slot_val = str(restaurants_info_dict[business_id][current_slot])
				utterance_content = utterance_content.replace('$'+current_slot+'$', slot_val, 1)
				utterance_dict["slots"][current_slot] = slot_val
				said_slot.add(current_slot)
			utterance_dict["nl"] = utterance_content
			new_dialogue["content"].append(utterance_dict)
		dialogue_list[dialogue_id] = new_dialogue
		dialogue_id += 1

with open('dialogue_data.json', 'w') as f:
    json.dump(dialogue_list, f, indent=4)