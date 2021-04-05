import cv2
import os
import shutil
import pytesseract
import difflib


def catchFaceAndClassify(dataset, name_lst, frame, num_frame):

    classfier = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # change to greystyle
    faceRects = classfier.detectMultiScale(frame, 1.1, 5, minSize=(8, 8))   # objects are returned as a list of rectangles.
    if len(faceRects) > 0:
        for faceRect in faceRects:
            x, y, w, h = faceRect

            for i in range(5):
                for j in range(5):
                    if 256*j <= x <= 256*(j + 1) and 144 * i <= y <= 144 * (i + 1):
                        cropped = grey[144 * (i + 1) - 32:144 * (i + 1), 256 * j:256 * j + 120]  # decrease length
                        cropped = cv2.resize(cropped, None, fx=1.2, fy=1.2)
                        # if pixel greyscale>185, set this pixel=255, preprocess the character image to get good quality for OCR
                        ret, thresh1 = cv2.threshold(cropped, 185, 255, cv2.THRESH_TOZERO)
                        text = pytesseract.image_to_string(thresh1)   # OCR
                        text = ''.join([char for char in text if char.isalpha()])  # remove the character like '\n',' ','\0Xc'
                        # print('before text is:', text)
                        text = string_comparison(text, name_lst)

                        try:
                            if text not in os.listdir(dataset) and text != '':
                                os.makedirs("./" + dataset + "/" + text)
                                # print('creat folder of ' + text)
                            elif text == '':
                                continue
                        except Exception as e:
                            print("frame number:", num_frame, e)
                            pass
                        else:
                            # print('after text is:', text)
                            clip_img = grey[y - 10:y + h + 10, x - 10:x + w + 10]
                            if clip_img.size != 0 and text.isalpha():
                                cv2.imwrite(dataset + '/' + text + '/{0}.jpg'.format(num_frame), clip_img)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return frame



def string_comparison(text, name_lst):  # get rid of small difference of OCR for the same character
    simlar_str = text
    lst = difflib.get_close_matches(text, name_lst, n=1, cutoff=0.75)
    if lst:
        return lst[0]
    else:
        return simlar_str



