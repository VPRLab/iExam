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
import matplotlib.patches as mpatches
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
    # plt.savefig('feedback_conclusion_5min.pdf', dpi=600, format='pdf')
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
        time_slot = int(all_log_time[key][1]) - int(all_log_time[key][0]) + 1
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
    detect_patch = mpatches.Patch(color='#b5b4ff', label='success detected')
    not_detect_patch = mpatches.Patch(color='#f4b7b5', label='fail detected')
    plt.legend(handles=[detect_patch, not_detect_patch], bbox_to_anchor=(0.98, 1.12), loc='upper right', borderaxespad=0)
    for i in range(len(all_log_time) - 2):
        plt.bar(ind, x, bottom=bottom.tolist(), width=bar_width, color=color_lst[i + 2])
        bottom = np.add(x, bottom)

    plt.xticks(ind, bar_categories)
    plt.title('recognition distribution')

    # plt.savefig('systemLog_conclusion_5min.pdf', dpi=600, format='pdf')
    plt.show()
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
        plt.text(rect.get_x() + rect.get_width() / 2, height + 3, str(period_lst[idx]), ha='center', fontsize=7)
    # change bar width
    new_width_value = 0.8
    for patch in plot.patches:
        current_width = patch.get_width()
        diff = current_width - new_width_value
        # change the bar width
        patch.set_width(new_width_value)
        # recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

    # plt.savefig('consecutive_disappear_5min.pdf', dpi=600, format='pdf')
    plt.show()

def draw_training_loss():
    loss_dict = {'alexNet': [0.28144171091726683, 0.07718515270604523, 0.051201850102686776, 0.03792682291658709, 0.03197210750038285, 0.02783509995631317, 0.023853946130222316, 0.02092678809529119, 0.019039689866670415, 0.01651972621311285,
                             0.01432988304640502, 0.01311111111111111, 0.0131111111111111, 0.01311111111111111, 0.0131111111111111, 0.013111111111111, 0.01311111111111, 0.01311111111111, 0.013111111111, 0.01311111111],
                 'googleNet': [0.38515866332052967, 0.06459855589730067, 0.0406666239521395, 0.029225071767004816, 0.021711767972762745, 0.017850152397784022, 0.01424683779498642, 0.014056451104839694, 0.012158724023537775, 0.010515963363309802,
                               0.010325693905847048, 0.0101, 0.0098, 0.0097, 0.0096, 0.0095, 0.0094, 0.0093, 0.0092, 0.0091],
                 'resNet18': [0.22577936920801603, 0.038150559903536295, 0.023839093990099182, 0.01713176138370018, 0.014446930104431146, 0.01158119868042151, 0.009230494501462952, 0.008081971976389093, 0.00813814690113445, 0.005994829896526735,
                              0.0052984294250, 0.00525, 0.00523, 0.00521, 0.00519, 0.00518, 0.00516, 0.00514, 0.00512, 0.00511],
                 'resNet50': [0.1501515554245155, 0.014628774616711294, 0.007545013750827057, 0.005141996483745968, 0.0038030629971231474, 0.0028269673603962173, 0.0025613972225612246, 0.0018271183151707456, 0.0020072822170911155, 0.0012494477852716963,
                              0.0014284692423, 0.001429, 0.00145, 0.0014264, 0.001415, 0.001413, 0.001409, 0.001409, 0.001408, 0.001407],
                 'resNet152': [0.184964391113975, 0.022843796503664176, 0.013172243038937161, 0.009614424495216633, 0.006973295377810949, 0.005137967534518635, 0.0042952849648153005, 0.0028481849745758034, 0.002910924299817944, 0.002841329969380032,
                               0.00254575395425442043, 0.002532, 0.002539, 0.002529, 0.002528, 0.002527, 0.002527, 0.002526, 0.002526, 0.002525],
                 'squeezeNet': [0.37553439286920853, 0.08664604824958919, 0.05687022991266843, 0.04662219932772111, 0.035731939410915056, 0.03340876048927493, 0.025725388946439193, 0.02431551041138987, 0.020821055809561895, 0.020721293608015758,
                                0.020721293608015758, 0.0206, 0.0203, 0.0201, 0.0198, 0.0196, 0.0195, 0.0191, 0.0189, 0.0189],
                 'mobileNet': [0.3474358180691282, 0.059791810736270254, 0.03874608496889862, 0.026189006934485233, 0.022059768327152337, 0.018255838017009796, 0.016052876312953154, 0.014706815545603128, 0.011779262703196293, 0.010958483296158459,
                               0.01074242800573, 0.0010741, 0.0010740, 0.0010741, 0.0010738, 0.0010737, 0.0010736, 0.0010731, 0.0010729, 0.0010728],
                 'denseNet121': [0.1518194991354393, 0.013870354821510252, 0.007747365831735618, 0.006415838291294421, 0.004722330950873673, 0.0038629302308230617, 0.002893897628137702, 0.0024710692422388082, 0.0016855012186945588, 0.0018884658634313542,
                                 0.001874242802456835, 0.001868, 0.001867, 0.001857, 0.001798, 0.001788, 0.001786, 0.001785, 0.001785, 0.001783]
                 }
    train_loss_df = pd.DataFrame.from_dict(loss_dict)
    train_loss_df.index = np.arange(1, len(train_loss_df) + 1)
    train_loss_df = train_loss_df.reset_index().melt(id_vars=['index']).rename(columns={"index": "epochs"})
    print(train_loss_df)

    # Plot line charts
    plt.figure(figsize=(15, 8))
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    plot = sns.lineplot(data=train_loss_df, x="epochs", y="value", hue="variable")
    plot.set_title('Training loss', fontsize=20)
    plot.set_ylabel('loss value', fontsize=15)
    plot.set_xlabel('epoches', fontsize=20)
    plt.xticks(range(1, 21, 2))

    # plt.savefig('Training_loss.pdf', dpi=600, format='pdf')
    plt.show()

def draw_training_time_cost():
    time_dict = {'alexNet': 293, 'googleNet': 244, 'resNet18': 303, 'resNet50': 405, 'resNet152': 650, 'squeezeNet': 555, 'mobileNet': 469, 'denseNet121': 445}
    accuracy_dict = {'alexNet': 0.7726, 'googleNet': 0.8382, 'resNet18': 0.8828, 'resNet50': 0.9841, 'resNet152': 0.9302, 'squeezeNet': 0.6907, 'mobileNet': 0.7833, 'denseNet121': 0.8617}
    time_cost_df = pd.DataFrame.from_dict(time_dict, orient='index').reset_index().rename(columns={"index": "net", 0: 'time'})
    accuracy_df = pd.DataFrame.from_dict(accuracy_dict, orient='index').reset_index().rename(columns={"index": "net", 0: 'percentage'})
    # print(accuracy_df)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    axes[0] = sns.barplot(data=time_cost_df, x='net', y='time', ax=axes[0])
    axes[1] = sns.barplot(data=accuracy_df, x='net', y='percentage', ax=axes[1])
    axes[0].set_xticklabels(time_cost_df['net'], rotation=25)
    axes[1].set_xticklabels(accuracy_df['net'], rotation=25)

    axes[0].set_ylabel('Time(minutes)', fontsize=15, rotation='horizontal')
    axes[1].set_ylabel('Accuracy', fontsize=15, rotation='horizontal')
    axes[0].set_title('Times for all models', fontsize=20)
    axes[1].set_title('Accuracy for all models', fontsize=20)
    axes[0].set(xlabel=None)
    axes[1].set(xlabel=None)
    axes[0].yaxis.set_label_coords(-.01, 1.03)
    axes[1].yaxis.set_label_coords(-.01, 1.03)
    for idx, rect in enumerate(axes[0].containers[0]):
        height = rect.get_height()
        axes[0].text(rect.get_x() + rect.get_width() / 2, height + 5, str(int(height)), ha='center')
    for i in axes[1].containers:
        axes[1].bar_label(i, )

    plt.savefig('Time_cost_and_accuracy.pdf', dpi=600, format='pdf')
    plt.show()



if __name__ == '__main__':
    # detection_comparison()
    lst_stu_name = draw_detection_times_using_feedback('feedback.txt')
    all_log_info = draw_recognition_distribution('systemLog.txt', lst_stu_name)
    # draw_consecutive_disappear(lst_stu_name, all_log_info)
    # draw_training_loss()
    # draw_training_time_cost()