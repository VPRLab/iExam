import cv2
import os
import pytesseract
import difflib
import platform
import pandas as pd


def catchFaceAndClassify(dataset, name_lst, frame, num_frame, viewInfo, tmp_dict):
    # base_img, tmp_dict = opencv_haar_cascade(dataset, name_lst, frame, num_frame, viewInfo, tmp_dict)
    base_img, tmp_dict = opencv_dnn_classify(dataset, name_lst, frame, num_frame, viewInfo, tmp_dict)

    return base_img, tmp_dict

def opencv_haar_cascade(dataset, name_lst, frame, num_frame, viewInfo, tmp_dict):
    # opencv haar cascade classify
    classfier = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")
    if platform.system() == 'Windows':
        pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    elif platform.system() == 'Darwin':
        pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

    base_img = frame.copy()
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
                historical_name = tmp_dict[(str(face_row), str(face_col))]
                clip_img = grey[y - 10:y + h + 10, x - 10:x + w + 10]
                if clip_img.size != 0:
                    cv2.imwrite(dataset + '/' + historical_name + '/{0}.jpg'.format(num_frame), clip_img)
                cv2.rectangle(base_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            else:
                cropped = grey[clip_height * (face_col + 1) - 32:clip_height * (face_col + 1), clip_width * face_row:clip_width * face_row + 120]  # decrease length
                text = ''  # initialize text
                for k in range(2, 8, 1):
                    resized_text = cv2.resize(cropped, None, fx=k, fy=k)
                    ret, thresh1 = cv2.threshold(resized_text, 185, 255, cv2.THRESH_TOZERO)
                    text = pytesseract.image_to_string(thresh1)  # OCR
                    text = ''.join([char for char in text if char.isalpha()])  # remove the character like '\n',' ','\0Xc'
                    if text == '':
                        print('cannot recognize text using OCR frame num is:', num_frame, [face_row, face_col])
                        # cv2.imwrite('cropped text.jpg', cropped_text)
                        thresh1 = cv2.Canny(image=thresh1, threshold1=80, threshold2=150)
                        text = pytesseract.image_to_string(thresh1)  # OCR
                        text = ''.join([char for char in text if char.isalpha()])  # remove the character like '\n',' ','\0Xc'
                        if text == '':
                            print('is empty ', [face_row, face_col], text, 'k=', k)
                            continue
                        else:
                            text = string_comparison(text, name_lst)
                            if text not in name_lst:
                                print('not match ', [face_row, face_col], text, 'k=', k)
                            else:
                                # print('text:', text)
                                break
                    else:
                        text = string_comparison(text, name_lst)
                        if text not in name_lst:
                            thresh1 = cv2.Canny(image=thresh1, threshold1=80, threshold2=150)
                            text = pytesseract.image_to_string(thresh1)  # OCR
                            text = ''.join([char for char in text if char.isalpha()])  # remove the character like '\n',' ','\0Xc'
                            if text == '':
                                print('is empty ', [face_row, face_col], text, 'k=', k)
                                continue
                            else:
                                text = string_comparison(text, name_lst)
                                if text not in name_lst:
                                    print('not match ', [face_row, face_col], text, 'k=', k)
                                else:
                                    # print('text:', text)
                                    break
                        else:
                            # print('text:', text)
                            break

                if text == '':
                    continue
                else:
                    tmp_dict[(str(face_row), str(face_col))] = text  # add ocr result in dict and every one second refresh
                try:
                    if text not in os.listdir(dataset) and text.isalpha() and text in name_lst:
                        os.makedirs("./" + dataset + "/" + text)

                except Exception as e:
                    print("frame number:", num_frame, e)
                    pass

                # print('after text is:', text)
                clip_img = grey[y - 10:y + h + 10, x - 10:x + w + 10]
                if clip_img.size != 0:
                    cv2.imwrite(dataset + '/' + text + '/{0}.jpg'.format(num_frame), clip_img)
                cv2.rectangle(base_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return base_img, tmp_dict


def opencv_dnn_classify(dataset, name_lst, frame, num_frame, viewInfo, tmp_dict):
    # opencv dnn classify (0,0) at left upper corner, so right>left and bottom>top
    detector = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
    base_img = frame.copy()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # change to greystyle
    row = viewInfo.get('Row')
    column = viewInfo.get('Column')
    clip_width = int(viewInfo.get('Width') / row)
    clip_height = int(viewInfo.get('Height') / column)
    original_size = frame.shape
    target_size = (300, 300)
    image = cv2.resize(frame, target_size)
    aspect_ratio_x = (original_size[1] / target_size[1])
    aspect_ratio_y = (original_size[0] / target_size[0])
    imageBlob = cv2.dnn.blobFromImage(image=image)
    detector.setInput(imageBlob)
    detections = detector.forward()

    detections_df = pd.DataFrame(detections[0][0], columns=["img_id", "is_face", "confidence", "left", "top", "right", "bottom"])
    detections_df = detections_df[detections_df['is_face'] == 1]  # 0: background, 1: face
    # print(detections_df.head())
    detections_df = detections_df[detections_df['confidence'] >= 0.15]
    for i, instance in detections_df.iterrows():
        # print(instance)
        confidence_score = str(round(100 * instance["confidence"], 2)) + " %"

        left = int(instance["left"] * 300)
        bottom = int(instance["bottom"] * 300)
        right = int(instance["right"] * 300)
        top = int(instance["top"] * 300)
        # print('instance: ', instance)
        # low resolution
        detected_face = image[top:bottom, left:right]
        saved_face = base_img[(int(top * aspect_ratio_y)):(int(bottom * aspect_ratio_y)),(int(left * aspect_ratio_x)):(int(right * aspect_ratio_x))]

        # high resolution
        # detected_face = base_img[int(top*aspect_ratio_y):int(bottom*aspect_ratio_y), int(left*aspect_ratio_x):int(right*aspect_ratio_x)]

        if detected_face.shape[0] > 0 and detected_face.shape[1] > 0:
            # low resolution
            # cv2.putText(image, confidence_score, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # cv2.rectangle(image, (left, top), (right, bottom), (255, 255, 255), 1) #draw rectangle to main image

            # high resolution
            # cv2.putText(base_img, confidence_score, (int(left * aspect_ratio_x), int(top * aspect_ratio_y - 10)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # cv2.rectangle(base_img, (int(left * aspect_ratio_x), int(top * aspect_ratio_y)),(int(right * aspect_ratio_x), int(bottom * aspect_ratio_y)), (255, 255, 255),1)  # draw rectangle to main image

            # -------------------
            face_row = int(int(left * aspect_ratio_x) / clip_width)
            face_col = int(int(top * aspect_ratio_y) / clip_height)
            if ((bottom - top) * aspect_ratio_y) > clip_width or ((right - left) * aspect_ratio_x) > clip_height:  # avoid capture error
                continue
            tmp_row = int(int(right * aspect_ratio_x) / clip_width)
            tmp_col = int(int(bottom * aspect_ratio_y) / clip_height)
            if tmp_row != face_row or tmp_col != face_col:  # avoid capture error
                continue
            if (str(face_row), str(face_col)) in tmp_dict.keys():
                historical_name = tmp_dict[str(face_row), str(face_col)]
                if saved_face.size != 0:
                    written_img = cv2.cvtColor(saved_face, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(dataset + '/' + historical_name + '/{0}.jpg'.format(num_frame), written_img)
                cv2.putText(base_img, confidence_score, (int(left * aspect_ratio_x), int(top * aspect_ratio_y - 10)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.rectangle(base_img, (int(left * aspect_ratio_x), int(top * aspect_ratio_y)),(int(right * aspect_ratio_x), int(bottom * aspect_ratio_y)), (255, 255, 255),1)  # draw rectangle to main image

            else:
                cropped = grey[clip_height * (face_col + 1) - 32:clip_height * (face_col + 1),
                          clip_width * face_row:clip_width * face_row + 120]  # decrease text length
                text = ''
                for k in range(2, 8, 1):
                    resized_text = cv2.resize(cropped, None, fx=k, fy=k)
                    ret, thresh1 = cv2.threshold(resized_text, 185, 255, cv2.THRESH_TOZERO)
                    text = pytesseract.image_to_string(thresh1)  # OCR
                    text = ''.join(
                        [char for char in text if char.isalpha()])  # remove the character like '\n',' ','\0Xc'
                    if text == '':
                        print('cannot recognize text using OCR frame num is:', num_frame, [face_row, face_col])
                        # cv2.imwrite('cropped text.jpg', cropped_text)
                        thresh1 = cv2.Canny(image=thresh1, threshold1=80, threshold2=150)
                        text = pytesseract.image_to_string(thresh1)  # OCR
                        text = ''.join(
                            [char for char in text if char.isalpha()])  # remove the character like '\n',' ','\0Xc'
                        if text == '':
                            print('is empty ', [face_row, face_col], text, 'k=', k)
                            continue
                        else:
                            text = string_comparison(text, name_lst)
                            if text not in name_lst:
                                print('not match ', [face_row, face_col], text, 'k=', k)
                            else:
                                # print('text:', text)
                                break
                    else:
                        text = string_comparison(text, name_lst)
                        if text not in name_lst:
                            thresh1 = cv2.Canny(image=thresh1, threshold1=80, threshold2=150)
                            text = pytesseract.image_to_string(thresh1)  # OCR
                            text = ''.join(
                                [char for char in text if char.isalpha()])  # remove the character like '\n',' ','\0Xc'
                            if text == '':
                                print('is empty ', [face_row, face_col], text, 'k=', k)
                                continue
                            else:
                                text = string_comparison(text, name_lst)
                                if text not in name_lst:
                                    print('not match ', [face_row, face_col], text, 'k=', k)
                                else:
                                    # print('text:', text)
                                    break
                        else:
                            # print('text:', text)
                            break

                # cropped = cv2.resize(cropped, None, fx=7, fy=7)
                # # if pixel greyscale>185, set this pixel=255, preprocess the character image to get good quality for OCR
                # ret, thresh1 = cv2.threshold(cropped, 185, 255, cv2.THRESH_TOZERO)
                # text = pytesseract.image_to_string(thresh1)  # OCR
                # text = ''.join([char for char in text if char.isalpha()])  # remove the character like '\n',' ','\0Xc'
                if text == '':
                    continue
                # # print('before text is:', text)
                # text = string_comparison(text, name_lst)
                tmp_dict[(str(face_row), str(face_col))] = text  # add ocr result in dict and every one second refresh
                try:
                    if text not in os.listdir(dataset) and text.isalpha():
                        os.makedirs("./" + dataset + "/" + text)
                        # print('creat folder of ' + text)
                except Exception as e:
                    print("frame number:", num_frame, e)
                    pass

                # print('after text is:', text)
                if saved_face.size != 0:
                    written_img = cv2.cvtColor(saved_face, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(dataset + '/' + text + '/{0}.jpg'.format(num_frame), written_img)
                cv2.putText(base_img, confidence_score, (int(left * aspect_ratio_x), int(top * aspect_ratio_y - 10)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.rectangle(base_img, (int(left * aspect_ratio_x), int(top * aspect_ratio_y)),(int(right * aspect_ratio_x), int(bottom * aspect_ratio_y)), (255, 255, 255),1)  # draw rectangle to main image

    return base_img, tmp_dict

def string_comparison(text, name_lst):  # get rid of small difference of OCR for the same character
    simlar_str = text
    lst = difflib.get_close_matches(text, name_lst, n=1, cutoff=0.75)
    if lst:
        return lst[0]
    else:
        return simlar_str



# 140min for classify 5min video if one frame ocr successful then the other 24 frame use same text in one second

# opencv dnn 19min29s ocr_period=1
# opencv dnn 17min01s ocr_period=2
# opencv dnn 15min43s ocr_period=3
# opencv haar 48min03s ocr_period=1
# opencv haar 41min40s ocr_period=2
# opencv haar 37min18s ocr_period=3