import os
import shutil
import difflib
import cv2

def OCRprocessing(dataset, name_lst):
    # dataset = 'marked_image_30minCopy'
    dirs = os.listdir(dataset)
    print('classes number in dataset:', len(dirs))
    print('classes:', dirs)
    tmp = {'file':[], 'subfolder':[]}
    for dir in dirs:
        if dir[-4:] != '.jpg':
            ch_dirs = os.listdir(dataset + "/" + dir)
            if len(ch_dirs) < 100:
                tmp['subfolder'].append(dir)
                shutil.rmtree(dataset + "/" + dir)
        else:
            tmp['file'].append(dir)
            os.remove(dataset + "/" + dir)
    if len(tmp['subfolder'])>0:
        print('remove subfolder: ', " ".join(tmp['subfolder']))
    if len(tmp['file'])>0:
        print('remove file: ', " ".join(tmp['file']))


    # combine folder using the whole substr locate in str head, tail or body
    dirs = os.listdir(dataset)
    for name in name_lst:
        for dir in dirs:
            if name in dir:
                print('combine: ',name + "*******" + dir)
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

    # adjust OCR threshold to get a similar folder name
    dirs = os.listdir(dataset)
    for dir in dirs:
        if dir not in name_lst:
            lst = difflib.get_close_matches(dir, name_lst, n=1, cutoff=0.5)  # need to adjust cutoff
            if lst:
                simlar_str = lst[0]
                print('similar: ', simlar_str+"#######"+dir)
                source_path = dataset + "/" + dir
                target_path = dataset+"/"+lst[0]
                if not os.path.exists(target_path):
                    os.makedirs(target_path)
                try:
                    for file in os.listdir(source_path):
                        if source_path +"/"+file not in os.listdir(target_path):
                            shutil.move(source_path +"/"+file, target_path)
                except Exception:
                    pass

    # delete empty folder
    try:
        dirs = os.listdir(dataset)
        for dir in dirs:
            if len(os.listdir(dataset+"/"+dir)) == 0:
                os.removedirs(dataset+"/"+dir)
    except Exception:
        pass

    # remove subfolder whose name not in name_lst
    try:
        dirs = os.listdir(dataset)
        for dir in dirs:
            if dir not in name_lst:
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