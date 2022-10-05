import cv2
import os, glob
from os import listdir
from os.path import isfile, join

path = '/data/plates'
new_path = '/data/plates_gray'

files = list(filter(lambda f: isfile(join(path, f)), listdir(path)))
for image in files:
    try:
        img = cv2.imread(os.path.join(path, image))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dstPath = join(new_path, image)
        cv2.imwrite(dstPath, gray)
    except:
        print("{} is not converted".format(image))
for fil in glob.glob("*.jpg"):
    try:
        image = cv2.imread(fil)
        gray_image = cv2.cvtColor(os.path.join(path, image), cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(new_path, fil), gray_image)
    except:
        print('{} is not converted')
