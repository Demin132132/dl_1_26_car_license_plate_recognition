from os import listdir
from os.path import isfile, join

import torch
import cv2
import pytesseract


input_dir = 'inputs/'
output_dir = 'outputs/'
model_path = 'models/best.pt'


files = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]

model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.eval()

for f in files:

    img = cv2.imread(input_dir + f)

    results = model(img, size=640)

    table = results.pandas().xyxy[0]

    print(table)

    ymin = int(table.ymin.loc[0]) - 15
    xmin = int(table.xmin.loc[0]) - 20
    ymax = int(table.ymax.loc[0]) + 15
    xmax = int(table.xmax.loc[0]) + 20

    img = img[ymin:ymax, xmin:xmax]

    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯ'  # только заглавные

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(img, lang='rus', config=custom_config)

    print(text)

    with open('result.txt', 'a') as file:
        file.write(text)
        file.write("\n")
