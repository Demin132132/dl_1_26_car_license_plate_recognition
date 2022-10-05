import torch
import cv2
from os import listdir
from os.path import isfile, join

onlyfiles = [f for f in listdir('C:\\data\\test') if isfile(join('C:\\data\\test', f))]

model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/yolo_license_plates2/weights/best.pt')
model.eval()

i = 0
for p in onlyfiles:
    img = cv2.imread('C:\\data\\test\\' + p)

    results = model(img, size=640)

    table = results.pandas().xyxy[0]

    ymin = int(table.ymin.loc[0]) - 15
    xmin = int(table.xmin.loc[0]) - 20
    ymax = int(table.ymax.loc[0]) + 15
    xmax = int(table.xmax.loc[0]) + 20

    roi = img[ymin:ymax, xmin:xmax]
    cv2.imwrite('C:\\data\\plates\\' + f'{i}.jpg', roi)
    i += 1
