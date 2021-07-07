import cv2
import os
import pytesseract
import difflib
import platform


def catchFaceAndClassify(dataset, name_lst, frame, num_frame, viewInfo, tmp_dict):

    classfier = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")
    if platform.system() == 'Windows':
        pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    elif platform.system() == 'Darwin':
        pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

    fps = viewInfo.get('fps')
    ocr_period = viewInfo.get('ocr_period')
    if num_frame % (fps * ocr_period) == 0:  # every period reset tmp_dict
        tmp_dict.clear()
        print('clear tmp dict')

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # change to greystyle
    row = viewInfo.get('Row')
    column = viewInfo.get('Column')
    clip_width = int(viewInfo.get('Width') / row)
    clip_height = int(viewInfo.get('Height') / column)
    faceRects = classfier.detectMultiScale(frame, 1.1, 5, minSize=(8, 8))   # objects are returned as a list of rectangles.
    if len(faceRects) > 0:
        for faceRect in faceRects:
            x, y, w, h = faceRect

            face_row = int(x / clip_width)
            face_col = int(y / clip_height)
            if w>clip_width or h>clip_height:  # avoid capture error
                continue
            tmp_row = int((x+w) / clip_width)
            tmp_col = int((y+h) / clip_height)
            if tmp_row!=face_row or tmp_col!=face_col:  # avoid capture error
                continue
            if (str(face_row), str(face_col)) in tmp_dict.keys():
                historical_name = tmp_dict[str(face_row), str(face_col)]
                clip_img = grey[y - 10:y + h + 10, x - 10:x + w + 10]
                if clip_img.size != 0:
                    cv2.imwrite(dataset + '/' + historical_name + '/{0}.jpg'.format(num_frame), clip_img)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            else:
                cropped = grey[clip_height * (face_col + 1) - 32:clip_height * (face_col + 1), clip_width * face_row:clip_width * face_row + 120]  # decrease length
                cropped = cv2.resize(cropped, None, fx=1.2, fy=1.2)
                # if pixel greyscale>185, set this pixel=255, preprocess the character image to get good quality for OCR
                ret, thresh1 = cv2.threshold(cropped, 185, 255, cv2.THRESH_TOZERO)
                text = pytesseract.image_to_string(thresh1)  # OCR
                text = ''.join([char for char in text if char.isalpha()])  # remove the character like '\n',' ','\0Xc'
                if text == '':
                    continue
                # print('before text is:', text)
                text = string_comparison(text, name_lst)
                tmp_dict[(str(face_row), str(face_col))] = text  # add ocr result in dict and every one second refresh
                try:
                    if text not in os.listdir(dataset) and text.isalpha():
                        os.makedirs("./" + dataset + "/" + text)
                        # print('creat folder of ' + text)
                except Exception as e:
                    print("frame number:", num_frame, e)
                    pass

                # print('after text is:', text)
                clip_img = grey[y - 10:y + h + 10, x - 10:x + w + 10]
                if clip_img.size != 0:
                    cv2.imwrite(dataset + '/' + text + '/{0}.jpg'.format(num_frame), clip_img)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return frame, tmp_dict



def string_comparison(text, name_lst):  # get rid of small difference of OCR for the same character
    simlar_str = text
    lst = difflib.get_close_matches(text, name_lst, n=1, cutoff=0.75)
    if lst:
        return lst[0]
    else:
        return simlar_str



# 140min for classify 5min video if one frame ocr successful then the other 24 frame use same text in one second