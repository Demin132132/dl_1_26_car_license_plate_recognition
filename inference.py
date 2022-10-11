from os import listdir
from os.path import isfile, join

import torch
import cv2
import pytesseract
import fastwer
#import easyocr


input_dir = 'test/'
output_dir = 'outputs/'
model_path = 'models/best.pt'

# reader = easyocr.Reader(['ru'])

files = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
files = sorted(files)

model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.eval()

ref = ["КО09ЕЕ97", "Е007ВР71", "Х215ЕР125", "В151РМ35", "В286ММ77", "Р104АВ142", "В606OO64", "С070НУ42"]

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

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(img, lang='rus', config=custom_config).upper()

    # text = reader.readtext(img_rgb)

    print(f)
    print(text)
    #print(ref[index])

    cer = fastwer.score_sent(text, ref[index], char_level=True)
    wer = fastwer.score_sent(text, ref[index], char_level=False)

    print(cer, wer, "\n")