import random
import pickle
import csv

data_list = []
with open('../high_ratings.csv','r') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    for row in f_csv:
        data_list.append(row)

random.shuffle(data_list)

train_size = 35000
dev_size = 2500
test_size = 2500

train_data = data_list[: train_size]
dev_data = data_list[train_size: train_size+dev_size]
test_data = data_list[train_size+dev_size: train_size+dev_size+test_size]

assert len(train_data) == train_size
assert len(dev_data) == dev_size
assert len(test_data) == test_size

print(train_data[:3])
print(dev_data[:3])
print(test_data[:3])

with open('RL_data.pkl','wb') as f:
	pickle.dump([train_data, dev_data, test_data], f)
