import json
import random

with open('./data/CLIO SRL dataset_ver2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

random.shuffle(data)

n = len(data)
n_train = int(n * 0.8)
n_valid = int(n * 0.1)
n_test = n - n_train - n_valid 

train_data = data[:n_train]
valid_data = data[n_train:n_train + n_valid]
test_data = data[n_train + n_valid:]

with open('./data/clio_train.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open('./data/clio_valid.json', 'w', encoding='utf-8') as f:
    json.dump(valid_data, f, ensure_ascii=False, indent=2)

with open('./data/clio_test.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)