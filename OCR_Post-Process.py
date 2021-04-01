import os
import shutil
import numpy as np
import difflib
import cv2

dataset = 'marked_image_30minCopy'
dirs = os.listdir(dataset)

for dir in dirs:
    if dir[-4:] != '.jpg':
        ch_dirs = os.listdir(dataset + "/" + dir)
        if len(ch_dirs) < 300:
            shutil.rmtree(dataset + "/" + dir)
    else:
        os.remove(dataset + "/" + dir)
print(np.asarray(dirs))
print(len(dirs))
name_lst_10min = ["YuanTIAN", "BailinHE", "BingHU", "BowenFAN", "ChenghaoLYUk",
            "HanweiCHEN", "JiahuangCHEN", "LiujiaDU", "LiZHANG", "PakKwanCHAN",
            "QijieCHEN", "QingboLl", "RouwenGE", "RuiGUO", "RunzeWANG",
            "RuochenXie", "SiqinLI", "SiruiLI", "TszKuiCHOW", "YanWU",
            "YimingZOU", "YuchuanWANG", "YuMingCHAN", "ZiwenLU", "ZiyaoZHANG"]

name_lst_30min = ["AnboWANG", "BailinHE", "BingHU", "BingyuanHUANG", "BowenFAN",
                "ChenghaoLYUk", "JiayiHOU", "DonghaoLI", "DuoHAN", "YuqinCHENG",
                "FanyuanZENG", "MengYIN", "HiuYanTONG", "GuozhengCHEN", "HanweiCHEN",
                "HaoWANG", "HaoweiLIU", "JiahuangCHEN", "KaihangLIU", "YuanTIAN",
                "LeiLIU", "LiujiaDU", "LiZHANG", "MingshengMA", "NgokTingYIP",
                "PakKwanCHAN", "QianqianCAO", "QidongZHAI", "QijieCHEN", "QingboLI",
                "RouwenGE", "RuiGUO", "RuikaiCAI", "RunzeWANG", "RuochenXie",
                "ShengtongZHU", "SiqinLI", "SiruiLI", "SuweiSUN", "TszKuiCHOW",
                "YalingZHANG", "YanWU", "YimingZOU", "YirunCHEN", "YuchuanWANG",
                "YuMingCHAN", "ZhijingBAO", "ZicongZHENG", "ZiwenLU", "ZiyaoZHANG",
                "ZiyiLI"]

name_lst_full = ['ZhijingBAO', 'RuikaiCAI', 'KexinCAO', 'QianqianCAO', 'PakKwanCHAN',
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
                 'ZicongZHENG', 'ShengtongZHU', 'YifanZHU', 'YimingZOU']

name_lst = name_lst_full
# combine folder using the whole substr locate in str head, tail or body
dirs = os.listdir(dataset)
for name in name_lst:
    for dir in dirs:
        if name in dir:
            print(name + "*******" + dir)
            source_path = dataset+"/"+dir
            target_path = dataset+"/"+name
            if dir == name:
                break
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            try:
                for file in os.listdir(source_path):
                    if source_path +"/"+file not in os.listdir(target_path):
                        shutil.move(source_path +"/"+file, target_path)
            except Exception:
                pass

dirs = os.listdir(dataset)
for dir in dirs:
    lst = difflib.get_close_matches(dir, name_lst, n=1, cutoff=0.5)  # need to adjust cutoff
    if lst:
        simlar_str = lst[0]
        print('similar: ', simlar_str+"********"+dir)
        source_path = dataset + "/" + dir
        target_path = dataset+"/"+lst[0]
        if dir == lst[0]:
            continue
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        try:
            for file in os.listdir(source_path):
                shutil.move(source_path +"/"+file, target_path)
        except Exception:
            pass
# delete empty folder and not in name_lst
try:
    dirs = os.listdir(dataset)
    for dir in dirs:
        if len(os.listdir(dataset+"/"+dir)) == 0 or dir not in name_lst:
            os.removedirs(dataset+"/"+dir)
except Exception:
    pass

try:
    dirs = os.listdir(dataset)
    for dir in dirs:
        if dir not in name_lst:
            print('dir: ', dir)
            shutil.rmtree(dataset + "/" + dir)
except Exception:
    pass

# dirs = os.listdir(dataset)
# for dir in dirs:
#     print(dir)
#     ch_dirs = os.listdir(dataset + "/" + dir)
#     for image in ch_dirs:
#         # print(dataset+'/'+dir+'/'+image)
#         # cv2.namedWindow(window_name)
#         img = cv2.imread(dataset+'/'+dir+'/'+image)
#         classfier = cv2.CascadeClassifier("../haarcascades/haarcascade_frontalface_default.xml")
#         faceRects = classfier.detectMultiScale(img, 1.1, 5, minSize=(8, 8))
#         if len(faceRects) == 0:
#             print('remove '+ dataset+'/'+dir+'/'+image)
#             os.remove(dataset+'/'+dir+'/'+image)