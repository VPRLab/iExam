import cv2
import os
import shutil
import tkinter.filedialog
import pytesseract
import difflib

def chooseVideo(name_lst, window_name='test for detecting face by video'):
    root = tkinter.Tk()  # create a Tkinter.Tk() instance
    root.withdraw()  # hide Tkinter.Tk() instance
    video = tkinter.filedialog.askopenfilename(title=u'choose file')
    dataset = 'marked_image_' + video.split('/')[-1].split('_')[0]


    if video != '':
        CatchVideo(window_name, video, dataset, name_lst)

def CatchVideo(window_name, video, dataset, name_lst):
    # cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(video)
    classfier = cv2.CascadeClassifier("../haarcascades/haarcascade_frontalface_default.xml")
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

    num_frame = 1

    # remove historical folder
    if os.path.exists(dataset):
        shutil.rmtree(dataset)
    # establish newly personal folder
    if not os.path.exists(dataset):
        os.makedirs(dataset)

    while cap.isOpened():
        ok, frame = cap.read()  # read one frame
        # width, height = cap.get(3), cap.get(4)
        # print('width height:',width,height)
        # property: width 1280, height 720, one frame:40ms  1280/5 = 256 720/5 = 144
        if not ok:
            break
        print('the number of captured frame：' + str(num_frame))
        # cv2.imwrite('./capture_image_3min/' + str(num_frame) + '.jpg', frame)  # save captured frames

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # change to greystyle

        faceRects = classfier.detectMultiScale(frame, 1.1, 5, minSize=(8, 8))   # objects are returned as a list of rectangles.
        # s = s + len(faceRects)
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect

                for i in range(5):
                    for j in range(5):
                        if 256*j <= x <= 256*(j + 1) and 144 * i <= y <= 144 * (i + 1):
                            cropped = grey[144 * (i + 1) - 32:144 * (i + 1), 256 * j:256 * j + 120]  # decrease length
                            cropped = cv2.resize(cropped, None, fx=1.2, fy=1.2)
                            # if pixel greyscale>90, set this pixel=255, preprocess the character image to get good quality for OCR
                            ret, thresh1 = cv2.threshold(cropped, 185, 255, cv2.THRESH_TOZERO)
                            text = pytesseract.image_to_string(thresh1)   # OCR
                            text = ''.join([char for char in text if char.isalpha()])  # remove the character like '\n',' ','\0Xc'
                            # print('before text is:', text)
                            text = string_comparison(text, name_lst)

                            try:
                                if text not in os.listdir(dataset) and text != '':
                                    os.makedirs("./" + dataset + "/" + text)
                                    # print('creat folder of ' + text)
                            except Exception as e:
                                print("frame number:", num_frame, e)
                                pass
                            else:
                                # print('after text is:', text)
                                clip_img = grey[y - 10:y + h + 10, x - 10:x + w + 10]
                                if clip_img.size != 0 and text.isalpha():
                                    cv2.imwrite("./" + dataset + "/" + text + "/{0}.jpg".format(num_frame), clip_img)
                                    # print("./" + dataset + "/" + text + "/{0}.jpg".format(num_frame))


                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # image show
        # cv2.imshow(window_name, frame)
        num_frame += 1
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):  # exit the video detect
            break

    cap.release()
    cv2.destroyAllWindows()

def string_comparison(text, name_lst):  # get rid of small difference of OCR for the same character
    simlar_str = text
    lst = difflib.get_close_matches(text, name_lst, n=1, cutoff=0.75)
    if lst:
        return lst[0]
    else:
        return simlar_str

if __name__ == '__main__':
    # name_lst = ("AnboWANG", "BailinHE", "BingHU", "BingyuanHUANG", "BowenFAN",
    #             "ChenghaoLYUk", "JiayiHOU", "DonghaoLI", "DuoHAN", "YuqinCHENG",
    #             "FanyuanZENG", "MengYIN", "HiuYanTONG", "GuozhengCHEN", "HanweiCHEN",
    #             "HaoWANG", "HaoweiLIU", "JiahuangCHEN", "KaihangLIU", "YuanTIAN",
    #             "LeiLIU", "LiujiaDU", "LiZHANG", "MingshengMA", "NgokTingYIP",
    #             "PakKwanCHAN", "QianqianCAO", "QidongZHAI", "QijieCHEN", "QingboLI",
    #             "RouwenGE", "RuiGUO", "RuikaiCAI", "RunzeWANG", "RuochenXie",
    #             "ShengtongZHU", "SiqinLI", "SiruiLI", "SuweiSUN", "TszKuiCHOW",
    #             "YalingZHANG", "YanWU", "YimingZOU", "YirunCHEN", "YuchuanWANG",
    #             "YuMingCHAN", "ZhijingBAO", "ZicongZHENG", "ZiwenLU", "ZiyaoZHANG",
    #             "ZiyiLI")
    name_lst_full = ('ZhijingBAO', 'RuikaiCAI', 'KexinCAO', 'QianqianCAO', 'PakKwanCHAN',
                'YuMingCHAN', 'GuozhengCHEN', 'HanweiCHEN', 'JiahuangCHEN', 'JiaxianCHEN',
                'QijieCHEN', 'YirunCHEN', 'YuqinCHENG', 'TszKuiCHOW', 'LiujiaDU',
                'BowenFAN', 'RouwenGE', 'RuiGUO', 'DuoHAN', 'YouyangHAN',
                'BailinHE', 'JiayiHOU', 'BingHU', 'BingyuanHUANG', 'HoNamLAI',
                'DonghaoLI', 'QingboLI', 'SiqinLI', 'SiruiLI', 'ZiyiLI',
                'HaoweiLIU', 'JinzhangLIU', 'KaihangLIU', 'LeiLIU', 'ZiwenLU',
                'KuanLV', 'ChenghaoLYU', 'MingshengMA', 'SuweiSUN', 'YuanTIAN',
                'HiuYanTONG', 'AnboWANG', 'HaoWANG', 'RunzeWANG', 'YuchuanWANG',
                'YanWU', 'RuochenXIE', 'MengYIN', 'ZijingYIN', 'NgokTingYIP',
                'FanyuanZENG', 'QidongZHAI', 'LiZHANG', 'YalingZHANG', 'ZiyaoZHANG',
                'ZicongZHENG', 'ShengtongZHU', 'YifanZHU', 'YimingZOU')
    chooseVideo(name_lst=name_lst_full)

    # file_num2.pop(13)
    # file_num2.append(0)
    #
    # video = tkinter.filedialog.askopenfilename(title=u'choose file', initialdir=(os.path.expanduser(default_dir)))
    # if video == '':
    #     exit()
    # else:
    #     file_num2 = CatchVideo(window_name, video, file_num2)



