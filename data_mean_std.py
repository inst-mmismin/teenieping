import os 
import numpy as np 
from PIL import Image

train_dir = 'dataset/train'
count = 0 
for ping_name in os.listdir(train_dir): 
    ping_dir = os.path.join(train_dir, ping_name)
    for file_name in os.listdir(ping_dir):
        image_path = os.path.join(ping_dir, file_name)
        data = Image.open(image_path).convert('RGB')
        data = data.resize((128, 128))
        data = np.array(data)[None, :]

        all_data = data if count == 0 else np.append(all_data, data, axis=0)
        count += 1 

mean = np.mean(all_data, axis=(0, 1, 2)) / 255
std = np.std(all_data, axis=(0, 1, 2)) / 255

print(mean) # [0.6151984  0.51760532 0.46836003]
print(std) # [0.26411435 0.24187316 0.26402279]