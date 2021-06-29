import matplotlib.pyplot as plt
import numpy as np



# find the longest consecutive sequence
def findLongest(nums):
    if nums == []:
        return 0
    nums.sort()
    maxlen = 1
    curlen = 1
    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1] + 1:
            curlen += 1
            maxlen = max(maxlen, curlen)
        else:
            maxlen = max(maxlen, curlen)
            curlen = 1
    return maxlen

def drawGraph(feedback_name, log_name):
    # draw a bar chart to show all reg times
    file = open(feedback_name, 'r')
    lines = file.readlines()
    all_reg_times = {}
    for idx, line in enumerate(lines):
        line = line[:-1]
        if 'first detected at' in line:
            tmp1 = line.split(' first')
            tmp2 = line.split('times: ')
            all_reg_times[tmp1[0]] = int(tmp2[1])
        elif 'has not recognized' in line:
            tmp1 = line.split(' has')
            all_reg_times[tmp1[0]] = 0

    lst_data = sorted(all_reg_times.items(), key=lambda items: items[1], reverse=True)  # descending data of reg time
    print(len(lst_data))
    print(lst_data)
    lst_data_name = []
    lst_data_num = []
    for i in range(len(lst_data)):
        lst_data_num.append(lst_data[i][1])
        lst_data_name.append((lst_data[i][0]))

    print('name: ', lst_data_name)
    print('num: ', lst_data_num)

    plt.figure(figsize=(10, 6))  # set figure size
    plt.tick_params(axis='y', labelsize=7)  # set ylabel size
    plt.xlabel('Recognition Times')
    plt.barh(lst_data_name, lst_data_num)
    plt.title('conclusion of recognition times')
    # plt.savefig('feedback_conclusion_90min.png')
    # plt.show()

    # draw a graph according present distribution
    file = open(log_name, 'r')
    lines = file.readlines()
    all_log_time = {}
    all_log_name = {}
    tmp = 0
    for idx, line in enumerate(lines):
        line = line[:-1]
        # print(idx, ' line: ', line)
        if 'following people have not be recognized' in line:
            tmp1 = line.split('from ')[1]
            tmp2 = tmp1.split(' to ')
            tmp2[0] = tmp2[0][:-1]
            tmp2[1] = tmp2[1][:-2]
            all_log_time[idx] = tmp2
            all_log_name[idx] = []
            tmp = idx  # tmp store idx
        elif 'all students can be recognized' in line:
            tmp1 = line.split('from ')[1]
            tmp2 = tmp1.split(' to ')
            tmp2[0] = tmp2[0][:-1]
            tmp2[1] = tmp2[1][:-1]
            all_log_time[idx] = tmp2
            all_log_name[idx] = []
        else:
            all_log_name[tmp].append(line)

    print(all_log_time)
    print(len(all_log_time))
    print(all_log_name)

    time_slot = None
    color_lst = []

    for key in all_log_time.keys():
        tmp_lst = []
        for i in range(len(lst_data_name)):
            tmp_lst.append('b')
        time_slot = int(all_log_time[key][1]) - int(all_log_time[key][0])
        for name in all_log_name[key]:
            idx = lst_data_name.index(name)
            tmp_lst[idx] = 'r'
        color_lst.append(tmp_lst)


    print('color: ', color_lst)

    x = []
    for i in range(len(lst_data_name)):
        x.append(time_slot)

    bars = []
    for i in range(len(lst_data_name)):
        bars.append(x)

    ind = np.arange(len(bars))
    bar_categories = lst_data_name
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
    # plt.savefig('systemLog_conclusion_90min.png')
    # plt.show()


    # draw a diagram with priority (max disappear in continuous time)
    priority_dict = {}
    result = {}
    for name in lst_data_name:
        priority_dict[name] = []
        result[name] = 0
    tmp_idx = 1

    for key in all_log_name.keys():
        for item in all_log_name[key]:
            priority_dict[item].append(tmp_idx)
        tmp_idx += 1
    print('priority_dict: ', priority_dict)

    for key in priority_dict.keys():
        result[key] = findLongest(priority_dict[key])
    sorted_result = sorted(result.items(), key=lambda x: x[1], reverse=True)
    print(sorted_result)

    for i in range(len(sorted_result)):
        sorted_result[i] = list(sorted_result[i])
        sorted_result[i][1] = time_slot * sorted_result[i][1]
    print(sorted_result)

    plt.figure(figsize=(12, 6))  # set figure size
    plt.tick_params(axis='x', labelsize=7)  # set xlabel size
    plt.xticks(rotation=35)
    plt.ylabel('Time(seconds)')
    plt.title('Consecutive disappear time for 90min test video')

    tmp_name = []
    tmp_data = []
    for i in range(len(sorted_result)):
        tmp_name.append(sorted_result[i][0])
        tmp_data.append(sorted_result[i][1])
    plt.bar(tmp_name, tmp_data)

    # plt.savefig('consecutive_disappear_90min.png')
    plt.show()



