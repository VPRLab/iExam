import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchTrain import Net


def recognize(classes, frame, namedict, frameCounter, net_path, studyCollection, time_slot):
    classfier = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(face_rects) > 0:

        for face_rect in face_rects:
            x, y, w, h = face_rect
            image = frame[y - 10:y + h + 10, x - 10:x + w + 10]
            # opencv to PIL: BGR2RGB
            PIL_image = cv2pil(image)
            if PIL_image is None:
                continue
            # using model to recognize
            label = predict_model(PIL_image, net_path, len(classes))
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 0, 255), 1)
            cv2.putText(frame, classes[label], (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)  # label name

            if frameCounter % time_slot == 1:  # every one time slot reset
                for k in studyCollection.keys():
                    studyCollection[k] = 0

            if not namedict[classes[label]]:
                namedict[classes[label]].append(frameCounter)
                namedict[classes[label]].append(1)
            else:
                namedict[classes[label]][1] += 1

            # get the time of this student appear in a time slot
            studyCollection[classes[label]] += 1


    return frame, namedict, studyCollection


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


