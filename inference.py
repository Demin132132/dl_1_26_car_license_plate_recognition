from os import listdir
from os.path import isfile, join

import torch
import cv2
import pytesseract
import fastwer


input_dir = 'test/'
model_path = 'models/best.pt'


files = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
files = sorted(files)

model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.eval()

ref = ["K009ЕЕ97", "Х215ЕР125", "В286ММ77"]

for index, f in enumerate(files):

    img = cv2.imread(input_dir + f)

    results = model(img, size=640)

    table = results.pandas().xyxy[0]

    print(table)

    ymin = int(table.ymin.loc[0]) - 15
    xmin = int(table.xmin.loc[0]) - 20
    ymax = int(table.ymax.loc[0]) + 15
    xmax = int(table.xmax.loc[0]) + 20

    img = img[ymin:ymax, xmin:xmax]

    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789АВЕКМНОРСТУ'

    # new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(new_img, lang='rus', config=custom_config).upper()

    print(f)
    print(text)
    #print(ref[index])

    cer = fastwer.score_sent(text, ref[index], char_level=True)
    wer = fastwer.score_sent(text, ref[index], char_level=False)

    print(cer, wer, "\n")
