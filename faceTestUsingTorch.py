import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchTrain import Net
import pandas as pd

# use single cell (256*144) to predict not use captured face
# def recognize(classes, frame, namedict, frameCounter, net_path, studyCollection, time_slot, viewInfo):
#     # print('view format: ', viewInfo)
#     row = viewInfo.get('Row')
#     column = viewInfo.get('Column')
#     clip_width = int(viewInfo.get('Width') / row)
#     clip_height = int(viewInfo.get('Height') / column)
#
#     grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     for j in range(row):
#         for i in range(column):
#             image = frame[clip_height * i:clip_height * (i + 1), clip_width * j:clip_width * (j + 1)]
#             # opencv to PIL: BGR2RGB
#             PIL_image = cv2pil(image)
#             if PIL_image is None:
#                 continue
#             # using model to recognize
#             label = predict_model(PIL_image, net_path, len(classes))
#
#             # cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 0, 255), 1)
#             cv2.putText(frame, classes[label], (clip_width * j+20, clip_height * i+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)  # label name
#             print('i: ', i, 'j: ', j, classes[label])
#             if frameCounter % time_slot == 1:  # every one time slot reset
#                 for k in studyCollection.keys():
#                     studyCollection[k] = 0
#
#             if not namedict[classes[label]]:
#                 namedict[classes[label]].append(frameCounter)
#                 namedict[classes[label]].append(1)
#             else:
#                 namedict[classes[label]][1] += 1
#
#             # get the time of this student appear in a time slot
#             studyCollection[classes[label]] += 1
#
#
#     return frame, namedict, studyCollection

# use every captured face to predict in one cell, if one frame detect and predict successfully, then the left 24 frames (1s has 25 frames) not detect and predict
# use time 14min for 5min test video in windows, 12min in mac
def recognize(classes, frame, namedict, frameCounter, net_path, studyCollection, time_slot, viewInfo, tmp_dict):
    classfier = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    row = viewInfo.get('Row')
    column = viewInfo.get('Column')
    clip_width = int(viewInfo.get('Width') / row)  # 256
    clip_height = int(viewInfo.get('Height') / column)  # 144
    if frameCounter % int(time_slot / 20) == 0:  # every second reset tmp_dict
        tmp_dict.clear()
        print('clear tmp dict')
    if frameCounter % time_slot == 1:  # every one time slot reset
        for k in studyCollection.keys():
            studyCollection[k] = 0
    label = -1

    for j in range(row):
        for i in range(column):
            if (str(j), str(i)) in tmp_dict.keys():
                label = tmp_dict[str(j), str(i)]
                cv2.putText(frame, classes[label], (clip_width * j +30, clip_height * i+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)  # label name
            else:
                cropped = grey[clip_height * i:clip_height * (i + 1), clip_width * j:clip_width * (j +1)]  # single cell
                # cv2.imshow("cropped", cropped)
                face_rects = classfier.detectMultiScale(cropped, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
                # print('num of detected face: ', len(face_rects))
                # print([j, i])
                # cv2.waitKey(200)
                if len(face_rects) > 0:
                    for face_rect in face_rects:
                        x, y, w, h = face_rect
                        image = cropped[y - 10:y + h + 10, x - 10:x + w + 10]
                        # opencv to PIL: BGR2RGB
                        PIL_image = cv2pil(image)
                        if PIL_image is None:
                            continue
                        # using model to recognize
                        label = predict_model(PIL_image, net_path, len(classes))
                        # cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 0, 255), 1)
                        cv2.putText(frame, classes[label], (clip_width * j +30, clip_height * i +30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)  # label name
                        tmp_dict[(str(j), str(i))] = label

            if label != -1:
                if namedict[classes[label]]==[]:
                    namedict[classes[label]].append(frameCounter)
                    namedict[classes[label]].append(1)
                else:
                    namedict[classes[label]][1] += 1
                # get the time of this student appear in a time slot
                studyCollection[classes[label]] += 1

    return frame, namedict, studyCollection, tmp_dict

# use every captured face to predict in one frame not one cell, if one frame detect and predict successfully, then the left 24 frames (1s has 25 frames) ect and predict
# use time 27min for 5min test video
# def recognize(classes, frame, namedict, frameCounter, net_path, studyCollection, time_slot, viewInfo, tmp_dict):
#     classfier = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
#
#     grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     row = viewInfo.get('Row')
#     column = viewInfo.get('Column')
#     clip_width = int(viewInfo.get('Width') / row)  # 256
#     clip_height = int(viewInfo.get('Height') / column)  # 144
#
#     if frameCounter % int(time_slot / 20) == 0:  # every second reset tmp_dict
#         tmp_dict.clear()
#         print('clear tmp dict')
#     for j in range(row):
#         for i in range(column):
#             if (str(j), str(i)) in tmp_dict.keys():
#                 historical_name = tmp_dict[str(j), str(i)]
#                 cv2.putText(frame, historical_name, (clip_width * j +30, clip_height * i+30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)  # label name
#
#     face_rects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
#     if len(face_rects) > 0:
#         for face_rect in face_rects:
#             x, y, w, h = face_rect
#             for j in range(row):
#                 for i in range(column):
#                     if clip_width*j <= x <= clip_width*(j + 1) and clip_height * i <= y <= clip_height * (i + 1) and (str(j), str(i)) not in tmp_dict.keys():
#                         image = grey[y - 10:y + h + 10, x - 10:x + w + 10]
#                         # opencv to PIL: BGR2RGB
#                         PIL_image = cv2pil(image)
#                         if PIL_image is None:
#                             continue
#                         # using model to recognize
#                         label = predict_model(PIL_image, net_path, len(classes))
#                         # cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 0, 255), 1)
#                         cv2.putText(frame, classes[label], (clip_width * j +30, clip_height * i +30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)  # label name
#                         tmp_dict[(str(j), str(i))] = classes[label]
#
#                         if frameCounter % time_slot == 1:  # every one time slot reset
#                             for k in studyCollection.keys():
#                                 studyCollection[k] = 0
#
#                         if not namedict[classes[label]]:
#                             namedict[classes[label]].append(frameCounter)
#                             namedict[classes[label]].append(1)
#                         else:
#                             namedict[classes[label]][1] += 1
#
#                         # get the time of this student appear in a time slot
#                         studyCollection[classes[label]] += 1
#
#
#     return frame, namedict, studyCollection, tmp_dict


# use captured face to predict in one frame not one cell, if one frame detect and predict successfully, then the left 24 frames (1s has 25 frames) detect but not predict
# use timein  27mfor 5min test video
# def recognize(classes, frame, namedict, frameCounter, net_path, studyCollection, time_slot, viewInfo, tmp_dict):
#     classfier = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
#
#     grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     row = viewInfo.get('Row')
#     column = viewInfo.get('Column')
#     clip_width = int(viewInfo.get('Width') / row)  # 256
#     clip_height = int(viewInfo.get('Height') / column)  # 144
#     switch = 1
#     if frameCounter % int(time_slot / 20) == 0:  # every second reset tmp_dict
#         tmp_dict.clear()
#         print('clear tmp dict')
#     face_rects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
#     if len(face_rects) > 0:
#         for face_rect in face_rects:
#             x, y, w, h = face_rect
#             for i in range(column):
#                 for j in range(row):
#                     if clip_width*j <= x <= clip_width*(j + 1) and clip_height * i <= y <= clip_height * (i + 1):
#                         if (str(j), str(i)) in tmp_dict.keys():
#                             historical_name = tmp_dict[str(j), str(i)]
#                             cv2.putText(frame, historical_name, (clip_width * j + 30, clip_height * i + 30),
#                                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)  # label name
#                             switch = 0
#                             break
#                         else:
#                             image = frame[y - 10:y + h + 10, x - 10:x + w + 10]
#                             # opencv to PIL: BGR2RGB
#                             PIL_image = cv2pil(image)
#                             if PIL_image is None:
#                                 continue
#                             # using model to recognize
#                             label = predict_model(PIL_image, net_path, len(classes))
#                             cv2.putText(frame, classes[label], (clip_width * j +30, clip_height * i +30), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                                         (255, 0, 0), 2, cv2.LINE_AA)  # label name
#
#                             tmp_dict[(str(j), str(i))] = classes[label]
#
#                             if frameCounter % time_slot == 1:  # every one time slot reset
#                                 for k in studyCollection.keys():
#                                     studyCollection[k] = 0
#
#                             if not namedict[classes[label]]:
#                                 namedict[classes[label]].append(frameCounter)
#                                 namedict[classes[label]].append(1)
#                             else:
#                                 namedict[classes[label]][1] += 1
#
#                             # get the time of this student appear in a time slot
#                             studyCollection[classes[label]] += 1
#                 if switch == 0:
#                     break
#             if switch == 0:
#                 switch = 1
#                 continue
#
#     return frame, namedict, studyCollection, tmp_dict

# use captured face to predict, and detect and predict every frame
# use time 42min for 5min test video
# def recognize(classes, frame, namedict, frameCounter, net_path, studyCollection, time_slot):
#     classfier = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
#
#     grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     face_rects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
#     if len(face_rects) > 0:
#
#         for face_rect in face_rects:
#             x, y, w, h = face_rect
#             image = frame[y - 10:y + h + 10, x - 10:x + w + 10]
#             # opencv to PIL: BGR2RGB
#             PIL_image = cv2pil(image)
#             if PIL_image is None:
#                 continue
#             # using model to recognize
#             label = predict_model(PIL_image, net_path, len(classes))
#             cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 0, 255), 1)
#             cv2.putText(frame, classes[label], (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)  # label name
#
#             if frameCounter % time_slot == 1:  # every one time slot reset
#                 for k in studyCollection.keys():
#                     studyCollection[k] = 0
#
#             if not namedict[classes[label]]:
#                 namedict[classes[label]].append(frameCounter)
#                 namedict[classes[label]].append(1)
#             else:
#                 namedict[classes[label]][1] += 1
#
#             # get the time of this student appear in a time slot
#             studyCollection[classes[label]] += 1
#
#
#     return frame, namedict, studyCollection


def get_transform():
    return transforms.Compose([
            transforms.Resize(32),  # reszie image to 32*32
            transforms.CenterCrop(32),  # center crop 32*32
            transforms.ToTensor()  # each pixel to tensor
        ])

def cv2pil(image):
    if image.size != 0:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        return None

def predict_model(image, net_path, class_num):

    data_transform = get_transform()
    image = data_transform(image)  # change PIL image to tensor
    image = image.view(-1, 3, 32, 32)
    net = Net(class_num)
    # net = torch.load(net_path)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(DEVICE)
    # load net
    net.load_state_dict(torch.load(net_path))
    output = net(image.to(DEVICE))
    # get the maximum
    pred = output.max(1, keepdim=True)[1]
    return pred.item()


