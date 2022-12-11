import json
import cv2
import numpy as np

path = '../data/'

f = open(path + 'train.json')
data = json.load(f)

# data preprocessing for yolo
for item in data:
    img = cv2.imread(path + item['file'])
    for i in item['nums']:
        box = np.array(i['box'])
        w = box[1, 0] - box[0, 0]
        h = box[3, 1] - box[0, 1]
        x = box[0, 0] + w / 2
        y = box[0, 1] + h / 2
        with open(path + item['file'].replace('jpg', '').replace('bmp', '') + 'txt', 'a+') as l:
            l.write(f'0 {x / img.shape[1]} {y / img.shape[0]} {w / img.shape[1]} {h / img.shape[0]}\n')
    with open(path + 'train.txt', 'a+') as i:
        i.write(f'data/obj/{item["file"].replace("train/", "")}\n')
print("Done")

