import sys
import os
import re
import json

list_rec = []
list_cause = []
src_course_list = []
src_pref_list = []

with open("./info/user_info_small_CHN.json", "r", encoding='utf-8') as f:
    user_dataset = json.load(f)
with open("./info/course_info_small_CHN.json", "r", encoding='utf-8') as f:
    with open("./course.txt", "w", encoding='utf-8') as g:
        course_dataset = json.load(f)
        for index, course in enumerate(course_dataset):
            if course["course_name"] not in src_course_list:
                src_course_list.append(course["course_name"])
                print(course["course_name"], file=g)
with open("pref.txt", "r", encoding='utf-8') as f:
    for line in f:
        src_pref_list.append(line.strip())


def gen_rec_prompt():
    global g, index, course
    user_format = "同学{index}历史记录: {course_list}"
    count = 0
    with open("./prompts/prompt_info_MOOCCube.txt", "w", encoding="utf-8") as g:
        g.write("")
    for _, user in enumerate(user_dataset):
        with open("./prompts/prompt_info_MOOCCube.txt", "a", encoding="utf-8") as g:
            course_list_raw = user['courses']
            timestamp_list_raw = user['timestamps']
            index = user['user_cnt']

            # 检查 timestamp_list_raw 的长度是否与 course_list_raw 一致
            if len(course_list_raw) != len(timestamp_list_raw):
                raise ValueError(
                    f"Error: Mismatched lengths for user {index}. Courses: {len(course_list_raw)}, Timestamps: {len(timestamp_list_raw)}")
            # 创建唯一课程列表 course_list 和对应的 timestamp_list
            course_list = []
            timestamp_list = []
            for course, timestamp in zip(course_list_raw, timestamp_list_raw):
                if course not in course_list:
                    course_list.append(course)
                    timestamp_list.append(timestamp)
            # 格式化用户信息，加入 course_list 和 index
            user_info = user_format.format_map({**user, "course_list": course_list, "index": index})
            # 将信息添加到 list_io
            list_rec.append({"input": course_list, "timestamp": timestamp_list, "output": ""})
            # 将格式化的用户信息写入文件
            print("%s" % user_info, file=g)
            count += 1
    print(len(list_rec))
    with open("./prompts/prompt_info.json", "w", encoding="utf-8") as h:
        json.dump(list_rec, h, ensure_ascii=False, indent=4)


def gen_cause_prompt():
    list_pair = []
    pref_cnt = len(src_pref_list)
    for i in range(pref_cnt - 1):
        for j in range(i + 1, pref_cnt):
            list_pair.append((src_pref_list[i], src_pref_list[j]))

    with open("./prompts/prompt_cause_MOOCCube.txt", "w", encoding="utf-8") as g:
        g.write("")
    pref_format = "A: {}\'{}\'类; B: {}\'{}\'类; C: {}\'{}\'类"
    choice = ["曾选过", "未选过", "将被推荐选择"]
    for rec in src_pref_list:
        with open("./prompts/prompt_cause_MOOCCube.txt", "a", encoding="utf-8") as g:
            def append_cause(c1, c2, rec):
                for item in list_pair:
                    pref_info = pref_format.format(c1, item[0],
                                                   c2, item[1],
                                                   choice[2], rec)
                    list_cause.append(pref_info)
                    print("%s" % pref_info, file=g)

            append_cause(choice[0], choice[0], rec)
            append_cause(choice[0], choice[1], rec)
            append_cause(choice[1], choice[1], rec)
            append_cause(choice[1], choice[0], rec)


# gen_rec_prompt()
gen_cause_prompt()
