import torch
import cv2

from os import listdir
from os.path import isfile, join

onlyfiles = [f for f in listdir('../utils/test') if isfile(join('../utils/test', f))]

model = torch.hub.load('ultralytics/yolov5', 'custom', path='../models/best.pt')
model.eval()

i = 0
for p in onlyfiles:
    img = cv2.imread('../utils/test' + p)

    results = model(img, size=640)

    table = results.pandas().xyxy[0]

    ymin = int(table.ymin.loc[0]) - 15
    xmin = int(table.xmin.loc[0]) - 20
    ymax = int(table.ymax.loc[0]) + 15
    xmax = int(table.xmax.loc[0]) + 20

    roi = img[ymin:ymax, xmin:xmax]
    cv2.imwrite('../utils/plates' + f'{i}.jpg', roi)
    i += 1
