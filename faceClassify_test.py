import cv2
import os
import pytesseract
import difflib
import platform
import pandas as pd
import matplotlib.pyplot as plt


def catchFaceAndClassify(dataset, name_lst, frame, num_frame, viewInfo):

    # classfier = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")
    detector = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
    if platform.system() == 'Windows':
        pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    elif platform.system() == 'Darwin':
        pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # change to greystyle
    row = viewInfo.get('Row')
    column = viewInfo.get('Column')
    width = viewInfo.get('Width')
    height = viewInfo.get('Height')
    clip_width = int(width / row)
    clip_height = int(height / column)

    base_img = frame.copy()
    original_size = frame.shape
    target_size = (300, 300)  # (width, height)
    image = cv2.resize(frame, target_size)
    aspect_ratio_x = (original_size[1] / target_size[1])
    aspect_ratio_y = (original_size[0] / target_size[0])
    # print([aspect_ratio_x, aspect_ratio_y])
    imageBlob = cv2.dnn.blobFromImage(image=image)
    detector.setInput(imageBlob)
    detections = detector.forward()

    detections_df = pd.DataFrame(detections[0][0], columns=["img_id", "is_face", "confidence", "left", "top", "right", "bottom"])
    detections_df = detections_df[detections_df['is_face'] == 1]  # 0: background, 1: face
    # print(detections_df.head())
    detections_df = detections_df[detections_df['confidence'] >= 0.15]

    for i, instance in detections_df.iterrows():
        #print(instance)

        confidence_score = str(round(100*instance["confidence"], 2))+" %"

        left = int(instance["left"] * 300)
        right = int(instance["right"] * 300)
        top = int(instance["top"] * 300)
        bottom = int(instance["bottom"] * 300)
        # print(instance)
        # print([left, right, top, bottom])
        face_col = int(left/target_size[0]*width/clip_width)
        face_row = int(top/target_size[1]*height/clip_height)+1
        # print('face_row: ', face_row)
        # print('face_col: ', face_col)
        #low resolution
        # detected_face = image[top:bottom, left:right]

        #high resolution
        detected_face = base_img[int(top*aspect_ratio_y):int(bottom*aspect_ratio_y), int(left*aspect_ratio_x):int(right*aspect_ratio_x)]

        if detected_face.shape[0] > 0 and detected_face.shape[1] > 0:

            #plt.figure(figsize = (3, 3))

            #low resolution
            #cv2.putText(image, confidence_score, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            #cv2.rectangle(image, (left, top), (right, bottom), (255, 255, 255), 1) #draw rectangle to main image

            #high resolution
            cv2.putText(base_img, confidence_score, (int(left*aspect_ratio_x), int(top*aspect_ratio_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(base_img, (int(left*aspect_ratio_x), int(top*aspect_ratio_y)), (int(right*aspect_ratio_x), int(bottom*aspect_ratio_y)), (255, 255, 255), 1) #draw rectangle to main image


            # cropped = grey[clip_height * face_row - 32:clip_height * face_row, clip_width * (face_col-1):clip_width * (face_col-1) + 120]  # decrease length
            cropped = grey[clip_height * face_row - 32:clip_height * face_row, clip_width * face_col:clip_width * face_col + 120]  # decrease length
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
                    # print('create folder of ' + text)
                elif text == '':
                    continue
            except Exception as e:
                print("frame number:", num_frame, e)
                pass

            # print('after text is:', text)

            if text.isalpha():
                detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)  # change to greystyle
                cv2.imwrite(dataset + '/' + text + '/{0}.jpg'.format(num_frame), detected_face)

    return base_img



def string_comparison(text, name_lst):  # get rid of small difference of OCR for the same character
    simlar_str = text
    lst = difflib.get_close_matches(text, name_lst, n=1, cutoff=0.75)
    if lst:
        return lst[0]
    else:
        return simlar_str



