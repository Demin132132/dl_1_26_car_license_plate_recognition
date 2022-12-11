import os
from os import listdir
from os.path import isfile, join

import glob
import cv2


PATH = '../data/utils/plates'
NEW_PATH = '../data/utils/plates_gray'


def main():
    files = list(filter(lambda f: isfile(join(PATH, f)), listdir(PATH)))
    for image in files:
        try:
            img = cv2.imread(os.path.join(PATH, image))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dstPath = join(NEW_PATH, image)
            cv2.imwrite(dstPath, gray)
        except:
            print("{} is not converted".format(image))
    for fil in glob.glob("*.jpg"):
        try:
            image = cv2.imread(fil)
            gray_image = cv2.cvtColor(os.path.join(PATH, image), cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(NEW_PATH, fil), gray_image)
        except:
            print('{} is not converted')


if __name__ == '__main__':
    main()
