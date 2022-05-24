# encoding: utf-8
"""
@author: YANG Xu
@contact: blitheyang99@gmail.com
@file: paperGraph.py
@time: 12/4/2022 4:51 PM
@desc: paper graph script
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def detection_comparison():
    d1 = {'det_num': [127834, 116626, 129359, 101339], 'index': ['mtcnn', 'haar', 'dnn', 'dlib']}  # detection num
    d2 = {'time_cost': [3167, 960, 196, 1683], 'index': ['mtcnn', 'haar', 'dnn', 'dlib']}  # time cost
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    fig.suptitle('Mainstream Detection Approach Comparison', fontsize=20)
    axes[0] = sns.barplot(data=d1, ax=axes[0], x='index', y='det_num')
    axes[0].set_title('detection amount', fontsize=10)
    axes[1] = sns.barplot(data=d2, ax=axes[1], x='index', y='time_cost')
    axes[1].set_title('time cost', fontsize=10)
    axes[1].set_ylabel('seconds', fontsize=13)
    for i in axes[0].containers:  # mark specific number in each bar
        axes[0].bar_label(i,)
    for i in axes[1].containers:
        axes[1].bar_label(i, )
    # plt.savefig('Detection Approach Comparison.pdf', dpi=600, format='pdf')


def findLongest(nums: list):    # find the longest consecutive sequence
    nums = set(nums)
    longest = 0
    end_time = 0
    start_time = 0
    for num in nums:
        if num - 1 not in nums:
            end = num + 1
            while end in nums:
                end += 1
            if (end - num) > longest:
                longest = end -num
                end_time = end-1
                start_time = num


    return longest, [start_time, end_time]

def draw_detection_times_using_feedback(feedback_name):
    # draw a bar chart to show all reg times
    file = open(feedback_name, 'r')
    lines = file.readlines()
    all_reg_times = {}
    for idx, line in enumerate(lines):
        line = line[:-1]  # remove '\n'
        if 'first detected at' in line:
            tmp1 = line.split(' first')
            tmp2 = line.split('times: ')
            all_reg_times[tmp1[0]] = int(tmp2[1])
        elif 'has not recognized' in line:
            tmp1 = line.split(' has')
            all_reg_times[tmp1[0]] = 0

    data = pd.DataFrame.from_dict([all_reg_times]).melt().rename(columns={'variable': 'stu_name', 'value': 'det_num'})
    plt.figure(figsize=(13, 6))
    plot = sns.barplot(data=data, x='det_num', y='stu_name', dodge=False)
    plot.set_title('conclusion of recognition times', fontsize=20)
    plot.set_ylabel('student name', fontsize=15)
    plot.set_xlabel('Recognition Times', fontsize=15)
    # plt.savefig('feedback_conclusion_90min.pdf', dpi=600, format='pdf')
    # plt.show()
    lst_stu_name = data['stu_name'].values.tolist()
    return lst_stu_name


def draw_recognition_distribution(log_name, lst_stu_name: list):
    # draw a graph according present distribution
    file = open(log_name, 'r')
    lines = file.readlines()
    all_log_time = {}
    all_log_name = {}
    tmp = 0
    for idx, line in enumerate(lines):
        line = line[:-1]
        if 'following people have not be recognized' in line:
            tmp1 = line.split('from ')[1]
            tmp2 = tmp1.split(' to ')
            tmp2[0] = tmp2[0][:-1]
            tmp2[1] = tmp2[1][:-2]
            all_log_time[idx] = tmp2      # idx is line number
            all_log_name[idx] = []
            tmp = idx  # tmp store idx
        elif 'all students can be recognized' in line:
            tmp1 = line.split('from ')[1]
            tmp2 = tmp1.split(' to ')
            tmp2[0] = tmp2[0][:-1]
            tmp2[1] = tmp2[1][:-2]
            all_log_time[idx] = tmp2
            all_log_name[idx] = []
        else:
            all_log_name[tmp].append(line)

    # print(all_log_time)
    # print(len(all_log_time))
    # print(all_log_name)

    time_slot = None
    color_lst = []

    for key in all_log_time.keys():
        tmp_lst = []
        for i in range(len(lst_stu_name)):
            tmp_lst.append('#b5b4ff')
        time_slot = int(all_log_time[key][1]) - int(all_log_time[key][0])
        for name in all_log_name[key]:
            idx = lst_stu_name.index(name)
            tmp_lst[idx] = '#f4b7b5'
        color_lst.append(tmp_lst)

    x = []
    for i in range(len(lst_stu_name)):
        x.append(time_slot)

    bars = []
    for i in range(len(lst_stu_name)):
        bars.append(x)

    ind = np.arange(len(bars))
    bar_categories = lst_stu_name
    bar_width = 0.5

    plt.figure(figsize=(12, 6))  # set figure size
    plt.tick_params(axis='x', labelsize=7)  # set xlabel size
    plt.xticks(rotation=35)
    plt.ylabel('Time(seconds)')

    plt.bar(ind, x, width=bar_width, color=color_lst[0])
    plt.bar(ind, x, bottom=x, width=bar_width, color=color_lst[1])
    bottom = np.add(x, x)
    for i in range(len(all_log_time) - 2):
        plt.bar(ind, x, bottom=bottom.tolist(), width=bar_width, color=color_lst[i + 2])
        bottom = np.add(x, bottom)

    plt.xticks(ind, bar_categories)
    plt.title('recognition distribution')
    # plt.savefig('systemLog_conclusion_90min.pdf', dpi=600, format='pdf')
    # plt.show()
    return [all_log_time, all_log_name]

def draw_consecutive_disappear(lst_stu_name: list, all_log_info: list):
    # draw a diagram with priority (max disappear in continuous time)
    priority_dict = {}
    result = {}
    start_end_time = {}
    for name in lst_stu_name:
        priority_dict[name] = []
        result[name] = 0

    [all_log_time, all_log_name] = all_log_info
    for key in all_log_name.keys():
        for item in all_log_name[key]:
            priority_dict[item] += [v for v in range(int(all_log_time[key][0]), int(all_log_time[key][1])+1)]

    for key in priority_dict.keys():
        result[key], start_end_time[key] = findLongest(priority_dict[key])
    sorted_result = sorted(result.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(sorted_result)):
        sorted_result[i] = list(sorted_result[i])

    tmp_name = []
    tmp_data = []
    for i in range(len(sorted_result)):
        tmp_name.append(sorted_result[i][0])
        tmp_data.append(sorted_result[i][1])
    df = pd.DataFrame(list(zip(tmp_name, tmp_data)), columns=['name', 'num'])

    plt.figure(figsize=(15, 8))  # set figure size
    plot = sns.barplot(data=df, x='name', y='num', dodge=False)

    plt.xlim(-1)  # x label start from -1
    plt.xticks(rotation=35)
    plt.ylabel('Time(seconds)')
    plt.title('Consecutive disappear time for testing video')
    period_lst = []
    for name in tmp_name:
        for key in start_end_time.keys():
            if name == key:
                text = str(start_end_time[key][0]) + 's-' + str(start_end_time[key][1]) + 's'
                period_lst.append(text)
    # add specific time period above each bar
    for idx, rect in enumerate(plot.containers[0]):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 2, str(period_lst[idx]), ha='center', fontsize=6)
    # change bar width
    new_width_value = 0.8
    for patch in plot.patches:
        current_width = patch.get_width()
        diff = current_width - new_width_value
        # change the bar width
        patch.set_width(new_width_value)
        # recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

    # plt.savefig('consecutive_disappear_90min.pdf', dpi=600, format='pdf')
    plt.show()

def drawGraph(feedback_name, log_name):
    lst_stu_name = draw_detection_times_using_feedback(feedback_name)
    all_log_info = draw_recognition_distribution(log_name, lst_stu_name)
    draw_consecutive_disappear(lst_stu_name, all_log_info)


if __name__ == '__main__':
    # detection_comparison()
    lst_stu_name = draw_detection_times_using_feedback('feedback.txt')
    all_log_info = draw_recognition_distribution('systemLog.txt', lst_stu_name)
    draw_consecutive_disappear(lst_stu_name, all_log_info)
