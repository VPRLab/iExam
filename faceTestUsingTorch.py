import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import time
import tkinter.filedialog
from torchTrain import Net

def chooseVideo(window_name='face recognize'):
    root = tkinter.Tk()  # create a Tkinter.Tk() instance
    root.withdraw()  # hide Tkinter.Tk() instance
    video = tkinter.filedialog.askopenfilename(title=u'choose file')
    if video == '':
        exit()
    else:
        recognize_video(window_name, video)


def recognize_video(window_name, video):
    classfier = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')
    classes_3min = ('BailinHE', 'BingyuanHUANG', 'DonghaoLI', 'DuoHAN', 'FanyuanZENG',
               'HaoWANG', 'HaoweiLIU', 'JiahuangCHEN', 'KaihangLIU', 'LeiLIU',
               'MengYIN', 'MingshengMA', 'NgokTingYIP', 'QidongZHAI', 'RuikaiCAI',
               'RunzeWANG', 'ShengtongZHU', 'YalingZHANG', 'YirunCHEN', 'YuqinCHENG',
               'ZhijingBAO', 'ZiyaoZHANG', 'ZiyiLI')

    classes_10min = ('BailinHE', 'BingHU', 'BowenFAN', 'ChenghaoLYUk', 'HanweiCHEN',
                     'JiahuangCHEN', 'LiZHANG', 'LiujiaDU', 'PakKwanCHAN', 'QijieCHEN',
                     'RouwenGE', 'RuiGUO', 'RunzeWANG', 'RuochenXie', 'SiqinLI',
                     'SiruiLI', 'TszKuiCHOW', 'YanWU', 'YimingZOU', 'YuMingCHAN',
                     'YuanTIAN', 'YuchuanWANG', 'ZiwenLU', 'ZiyaoZHANG')

    classes = classes_10min

    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(video)
    frameCounter = 0
    namedict = defaultdict(list)
    tic = time.time()
    while cap.isOpened():
        print('num of frame:', frameCounter)
        ok, frame = cap.read()
        if not ok:
            break
        # catch_frame = catch_face(frame)
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
                label = predict_model(PIL_image)
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 0, 255), 1)
                cv2.putText(frame, classes[label], (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                if not namedict[classes[label]]:
                    namedict[classes[label]].append(frameCounter)
                    namedict[classes[label]].append(1)
                else:
                    namedict[classes[label]][1] += 1
        cv2.imshow(window_name, frame)
        # cv2.imshow(window_name, catch_frame)
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break
        frameCounter += 1
    cap.release()
    cv2.destroyAllWindows()
    toc = time.time()
    print('time: ', toc - tic)
    for k, v in namedict.items():
        print(str(k)+' first detected at '+str(namedict[k][0])+' frames,'+' total detect times: '+str(namedict[k][1]))

# def catch_face(frame):
#
#
#     grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     face_rects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
#     if len(face_rects) > 0:
#
#         for face_rects in face_rects:
#             x, y, w, h = face_rects
#             image = frame[y - 10:y + h + 10, x - 10:x + w + 10]
#             # opencv 2 PIL格式图片
#             PIL_image = cv2pil(image)
#             if PIL_image is None:
#                 continue
#             # 使用模型进行人脸识别
#             label = predict_model(PIL_image)
#             cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 1)
#             cv2.putText(frame, classes[label], (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
#
#     return frame

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

def predict_model(image):

    data_transform = get_transform()
    image = data_transform(image)  # change PIL image to tensor
    image = image.view(-1, 3, 32, 32)
    net = Net()
    # net = torch.load(net_path)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(DEVICE)
    # load net
    net.load_state_dict(torch.load(net_path))
    output = net(image.to(DEVICE))
    # get the maximum
    pred = output.max(1, keepdim=True)[1]
    return pred.item()

# class Net(nn.Module):  # define net, which extends torch.nn.Module
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)  # convolution layer
#         self.pool = nn.MaxPool2d(2, 2)  # pooling layer
#         self.conv2 = nn.Conv2d(6, 16, 5)  # convolution layer
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)  # fully connected layer
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 23)  # output is 24, 24 is the number of class in dataset
#
#     def forward(self, x):  # feed forward
#
#         x = self.pool(F.relu(self.conv1(x)))  # F is torch.nn.functional
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)  # .view( ) is a method tensor, which automatically change tensor size but elements number not change
#
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

if __name__ == '__main__':
    net_path = 'net_params_10minCopy.pkl'
    chooseVideo()
    # recognize_video('test', )

    # 3min result:
    # DonghaoLI first detected at 0 frames, total detect times: 2131
    # NgokTingYIP first detected at 0 frames, total detect times: 3344
    # DuoHAN first detected at 0 frames, total detect times: 4330
    # ZhijingBAO first detected at 0 frames, total detect times: 4359
    # YuqinCHENG first detected at 0 frames, total detect times: 825
    # YalingZHANG first detected at 2 frames, total detect times: 2350
    # QidongZHAI first detected at 12 frames, total detect times: 2128
    # BingyuanHUANG first detected at 21 frames, total detect times: 2827
    # HaoweiLIU first detected at 28 frames, total detect times: 1695
    # RuikaiCAI first detected at 38 frames, total detect times: 231
    # JiahuangCHEN first detected at 42 frames, total detect times: 3190
    # MingshengMA first detected at 76 frames, total detect times: 1072
    # MengYIN first detected at 78 frames, total detect times: 4418
    # HaoWANG first detected at 146 frames, total detect times: 4621
    # YirunCHEN first detected at 206 frames, total detect times: 1262
    # ZiyaoZHANG first detected at 208 frames, total detect times: 3057
    # ShengtongZHU first detected at 211 frames, total detect times: 1040
    # LeiLIU first detected at 312 frames, total detect times: 972
    # RunzeWANG first detected at 331 frames, total detect times: 4170
    # ZiyiLI first detected at 465 frames, total detect times: 3511
    # KaihangLIU first detected at 586 frames, total detect times: 259
    # BailinHE first detected at 788 frames, total detect times: 3387
    # FanyuanZENG first detected at 2372 frames, total detect times: 417

    # 10min result:
    # time:  4299.933976173401
    # SiruiLI first detected at 0 frames, total detect times: 15990
    # YanWU first detected at 0 frames, total detect times: 7295
    # BailinHE first detected at 0 frames, total detect times: 12745
    # YuanTIAN first detected at 0 frames, total detect times: 28373
    # RuiGUO first detected at 0 frames, total detect times: 17148
    # ZiyaoZHANG first detected at 0 frames, total detect times: 12779
    # ChenghaoLYUk first detected at 0 frames, total detect times: 13028
    # LiujiaDU first detected at 0 frames, total detect times: 22317
    # JiahuangCHEN first detected at 0 frames, total detect times: 772
    # QijieCHEN first detected at 0 frames, total detect times: 7595
    # YuMingCHAN first detected at 0 frames, total detect times: 14549
    # RouwenGE first detected at 0 frames, total detect times: 9147
    # HanweiCHEN first detected at 0 frames, total detect times: 13456
    # RunzeWANG first detected at 0 frames, total detect times: 16500
    # YuchuanWANG first detected at 6 frames, total detect times: 10224
    # ZiwenLU first detected at 16 frames, total detect times: 5397
    # RuochenXie first detected at 27 frames, total detect times: 6009
    # BowenFAN first detected at 79 frames, total detect times: 5759
    # TszKuiCHOW first detected at 81 frames, total detect times: 11198
    # SiqinLI first detected at 230 frames, total detect times: 5317
    # PakKwanCHAN first detected at 242 frames, total detect times: 7305
    # LiZHANG first detected at 388 frames, total detect times: 5240
    # BingHU first detected at 591 frames, total detect times: 10751
    # YimingZOU first detected at 874 frames, total detect times: 4335