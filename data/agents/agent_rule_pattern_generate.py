import json

utt_gen_dict = \
{
	"categories":[
				{ 
				"slots":["categories"],
				"nl":"What kind of food do you like?",
				},
				],
	"state":[
				{ 
				"slots":["state"],
				"nl":"Which state are you in?",
				},		
			],
	"city":[
				{ 
				"slots":["city"],
				"nl":"Which city are you in?",
				},			
			],
	"price":[
				{ 
				"slots":["price"],
				"nl":"Which price range do you want?",
				},					
			],
	"stars":[
				{ 
				"slots":["stars"],
				"nl":"Which rating range do you like?",
				},		
			]
}

with open('utt_gen_dict.json', 'w') as f:
    json.dump(utt_gen_dict, f, indent=4)