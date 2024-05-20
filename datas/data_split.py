import json
import os 
import random

src_path = "/data/LateOrchestration/Event_Domain_Adaptation/"
dataset_size = 3000

# collect event data
event_list = ["denoised_event/" + e for e in os.listdir(src_path + "denoised_event")]

random.seed(0)
random.shuffle(event_list)
event_data = event_list[0:dataset_size]

# load image and annotations
with open(src_path + 'masked_ann_data.json', 'r') as json_file:
    data = json.load(json_file)

# oversample
random.shuffle(data)
duplicate_data = data[0:dataset_size-len(data)]
image_data = data + duplicate_data
print(len(image_data))

# concat
for i in range(len(image_data)):
    image_data[i]["event path"] = event_data[i]

print(image_data[0])

# train_test_split
random.shuffle(image_data)
train_size = int(len(image_data) * 0.7)
val_size = int(len(image_data) * 0.2)

train_data = image_data[0:train_size]
val_data = image_data[train_size:train_size+val_size]
test_data = image_data[train_size+val_size:]

print(len(train_data), len(val_data), len(test_data))

for data_name, sub_data in zip(["train", "val", "test"], [train_data, val_data, test_data]):
    with open(src_path + 'masked_{}.json'.format(data_name), 'w') as json_file:
        json.dump(sub_data, json_file, indent=4)
