echo "start..."
echo "generate dialogue..."
cd ./data
python3 yelp_generate_dialogue.py
echo "generate belief tracker data"
cd ./belief_tracker_data
python3 belief_tracker_data_generate.py
python3 belief_tracker_data_generate2.py
echo "generate FM data"
cd ../FM_data
python3 FM_data_generate_2.py
echo "generate high score data"
cd ..
python3 generate_high_rating.py
echo "generate agent data"
cd ./RL_data
python3 create_RL_data.py
python3 create_RL_pretrain_data.py