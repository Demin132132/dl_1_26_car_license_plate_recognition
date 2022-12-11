import torch
import cv2
import pytesseract


INPUT_FILE = 'test/1.jpg'
MODEL_PATH = 'models/best.pt'


def read_img(file_path):
    img = cv2.imread(file_path)
    return img


def get_model(model_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    model.eval()
    return model


def inference(model, img):
    results = model(img, size=640)
    table = results.pandas().xyxy[0]

    ymin = int(table.ymin.loc[0]) - 15
    xmin = int(table.xmin.loc[0]) - 20
    ymax = int(table.ymax.loc[0]) + 15
    xmax = int(table.xmax.loc[0]) + 20

    img = img[ymin:ymax, xmin:xmax]

    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789АВЕКМНОРСТУ'

    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(new_img, lang='rus', config=custom_config).upper()

    print(INPUT_FILE)
    print(text)


def main():
    model = get_model(MODEL_PATH)
    img = read_img(INPUT_FILE)
    inference(model, img)


if __name__ == '__main__':
    main()
