from datetime import datetime
from datetime import date
from analysis import analyse
from analysis import longestConsecutive
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def dnn_analyse():
    lines = []
    for line in open('dnn_result.log', 'r', encoding='UTF-8'):
        lines.append(line)

    print('length of log file', len(lines))
    count_face_num_is_zero = 0
    count_test_num = 0
    start_time = 0
    end_time = 0
    for line in lines:
        # if 'start' in line:
        #     print(line)
        #     print(lines.index(line))
        # if 'end' in line:
        #     print(line)
        #     print(lines.index(line))
        if 'face size' in line:
            count_test_num += 1
        if 'face size: 0' in line:
            # print(line)
            # print(lines.index(line))
            count_face_num_is_zero += 1
        if 'start exam' in line:
            start_time = line.split('start exam ')[1].split('\n')[0]
        if 'end up exam' in line:
            end_time = line.split('end up exam ')[1].split('\n')[0]

    print('test num is: ', count_test_num)
    print('zero face num is: ', count_face_num_is_zero)

    face_dict = {}
    i = 0
    for line in lines:
        if 'test.js:113' in line and 'face size: 0' not in line:
            index = lines.index(line)
            time = line.split('time:  ')[1].split('  face size')[0]
            face_dict[i] = []

            while(1):
                if 'test.js:115' in lines[index+1]:
                    x = lines[index+1].split('x: ')[1].split(' y:')[0]
                    y = lines[index+1].split('y: ')[1].split(' width:')[0]
                    w = lines[index+1].split('width: ')[1].split(' height:')[0]
                    h = lines[index+1].split('height: ')[1].split('\n')[0]
                    if float(w) < 50:  # remove tiny rectangle
                        index += 1
                        continue
                    face_dict[i].append([(float(x)), (float(y)), (float(w)), (float(h)), time])
                    index += 1
                else:
                    i += 1
                    break
    counter_more_than_three = 0
    more_than_three_face_dict = {}
    for k, v in face_dict.items():
        if len(face_dict[k]) > 3:
            # print(face_dict[k])
            more_than_three_face_dict[k] = v
            counter_more_than_three += 1

    normal_result_face_dict = {}
    normal_result_face_dict = {k: v for k, v in face_dict.items() if k not in more_than_three_face_dict}

    print('normal detect face num:', len(normal_result_face_dict))

    only_ta_face_dict = {}
    for k in list(face_dict.keys()):
        tmp_list = []
        for i in range(len(face_dict[k])):
            x, y, w, h, time = face_dict[k][i]
            if 400 <= int(x) <= 650:
                tmp_list.append(face_dict[k][i])

        if tmp_list != []:
            only_ta_face_dict[k] = tmp_list
        for element in tmp_list:
            # print(type(element), element)
            face_dict[k].remove(element)
        if len(face_dict[k]) == 0:
            face_dict.pop(k)

    print('only_ta_face_dict num:', len(only_ta_face_dict))

    counter_stu1 = 0
    counter_stu2 = 0
    remain_face_dict = {}
    stu1_face_dict = {}
    stu2_face_dict = {}
    for k in list(face_dict.keys()):
        tmp1, tmp2 = [], []
        remain_face_dict[k] = []
        for element in face_dict[k]:
            x, y, w, h, time = element
            if int(x) < 400:
                counter_stu1 += 1
                tmp1.append(element)
            elif 650 < int(x):
                counter_stu2 += 1
                tmp2.append(element)
            else:
                remain_face_dict[k].append(element)
        if len(remain_face_dict[k]) == 0:
            remain_face_dict.pop(k)
        if tmp1 != []:
            stu1_face_dict[k] = tmp1
        if tmp2 != []:
            stu2_face_dict[k] = tmp2

    one_second_detect_stu1_more_than_one = {}
    one_second_detect_stu2_more_than_one = {}
    for k in stu1_face_dict:
        if len(stu1_face_dict[k]) > 1:
            one_second_detect_stu1_more_than_one[k] = stu1_face_dict[k]
    for k in stu2_face_dict:
        if len(stu2_face_dict[k]) > 1:
            one_second_detect_stu2_more_than_one[k] = stu2_face_dict[k]



    print('stu1_face_dict num:', len(stu1_face_dict))
    print('stu2_face_dict num:', len(stu2_face_dict))
    print('remain_face_dict num:', len(remain_face_dict))
    print('one_second_detect_stu1_more_than_one:', len(one_second_detect_stu1_more_than_one))
    print('one_second_detect_stu2_more_than_one:', len(one_second_detect_stu2_more_than_one))

    stu1_appear_time = []
    stu2_appear_time = []
    for k in stu1_face_dict:
        stu1_appear_time.append(stu1_face_dict[k][0][4])
    for k in stu2_face_dict:
        stu2_appear_time.append(stu2_face_dict[k][0][4])


    date_time_obj1 = datetime.strptime(start_time, '%H:%M:%S').time()
    datetime1 = datetime.combine(date.today(), date_time_obj1)
    stu1_appear_time_second = []  # the time(second) from start time student face detected
    stu2_appear_time_second = []
    for time in stu1_appear_time:
        date_time_obj2 = datetime.strptime(time, '%H:%M:%S').time()
        datetime2 = datetime.combine(date.today(), date_time_obj2)
        time_elapsed = int((datetime2 - datetime1).total_seconds())
        if time_elapsed not in stu1_appear_time_second:
            stu1_appear_time_second.append(time_elapsed)
    for time in stu2_appear_time:
        date_time_obj2 = datetime.strptime(time, '%H:%M:%S').time()
        datetime2 = datetime.combine(date.today(), date_time_obj2)
        time_elapsed = int((datetime2 - datetime1).total_seconds())
        if time_elapsed not in stu2_appear_time_second:
            stu2_appear_time_second.append(time_elapsed)

    datetime2 = datetime.combine(date.today(), datetime.strptime(end_time, '%H:%M:%S').time())
    end_second = int((datetime2 - datetime1).total_seconds())  # timestamp from start time to end time (7203s)

    stu1_disappear_time_second = []
    stu2_disappear_time_second = []
    for i in range(1, end_second+1):
        if i not in stu1_appear_time_second:
            stu1_disappear_time_second.append(i)
        if i not in stu2_appear_time_second:
            stu2_disappear_time_second.append(i)

    return [stu1_disappear_time_second, stu2_disappear_time_second]

def dnn_longestConsecutive(nums):
    hash_dict = dict()

    max_length = 0
    for num in nums:
        if num not in hash_dict:
            left = hash_dict.get(num-1, 0)
            right = hash_dict.get(num+1, 0)

            cur_length = 1 + left + right
            if cur_length > max_length:
                max_length = cur_length

            hash_dict[num] = cur_length
            hash_dict[num-left] = cur_length
            hash_dict[num+right] = cur_length

    return max_length, hash_dict

def drawGraph(stu1_max_length, stu2_max_length):

    ind = np.arange(2)
    width = 0.4
    dnn_max_consecutived_disappear = [119, 37]
    max_consecutived_disappear = [86, 57]

    dnn_face_detect_time = [2179, 7085]
    face_detect_time = [2972, 7717]
    category = ('dnn', 'haar-cascade')
    color = ['#CED6E7', '#DEBCA1']

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    fig.subplots_adjust(top=0.9)
    plt.suptitle('two methods for real-time test')
    axes[0].bar(ind, dnn_max_consecutived_disappear, width, label='stu1 consecutive disappear period', color=color[0])
    axes[0].bar(ind+width, max_consecutived_disappear, width, label='stu2 consecutive disappear period', color=color[1])
    axes[0].set_xticks(ind+0.5*width)
    axes[0].set_xticklabels(tuple(category))
    axes[0].legend()

    axes[1].bar(ind, dnn_face_detect_time, width, label='stu1 face detect times', color=color[0])
    axes[1].bar(ind+width, face_detect_time, width, label='stu2 face detect times', color=color[1])
    axes[1].set_xticks(ind+0.5*width)
    axes[1].set_xticklabels(tuple(category))
    axes[1].legend()

    plt.savefig('real-time-result.pdf', dpi=600, format='pdf')
    plt.show()



if __name__ == '__main__':
    stu1_disappear_time_second, stu2_disappear_time_second = dnn_analyse()
    stu1_max_length, stu1_hash_dict = dnn_longestConsecutive(stu1_disappear_time_second)
    stu2_max_length, stu2_hash_dict = dnn_longestConsecutive(stu2_disappear_time_second)
    print(stu1_max_length, stu2_max_length)

    stu3_disappear_time_second, stu4_disappear_time_second = analyse()
    stu3_max_length, stu3_hash_dict = longestConsecutive(stu3_disappear_time_second)
    stu4_max_length, stu4_hash_dict = longestConsecutive(stu4_disappear_time_second)
    print(stu3_max_length, stu4_max_length)

    drawGraph(stu1_max_length, stu2_max_length)






