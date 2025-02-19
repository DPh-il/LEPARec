import json

list_rec = []
list_cause = []
src_course_list = []
src_pref_list = []


# 读取用户数据和课程数据
with open("students_courses_text.json", "r", encoding='utf-8') as f:
    user_dataset = json.load(f)
with open("course.txt", "r", encoding='utf-8') as f:
    for line in f:
        course_id, course_name = line.strip().split(',')
        if course_name not in src_course_list:
            src_course_list.append(course_name)
with open("pref.txt", "r", encoding='utf-8') as f:
    for line in f:
        src_pref_list.append(line.strip())


def gen_rec_prompt():
    global index
    user_format = "同学{index}历史记录: {course_list}"
    count = 0
    with open("./prompts/prompt_info_MOOCData.txt", "w", encoding="utf-8") as g:
        g.write("")
    for user_id, user_data in user_dataset.items():
        with open("./prompts/prompt_info_MOOCData.txt", "a", encoding="utf-8") as g:
            course_list = user_data['input']
            index = user_id

            user_info = user_format.format_map({"course_list": course_list, "index": index})
            list_rec.append({"input": course_list,
                             "course_count": user_data['course_count'],
                             "timestamp": user_data['timestamp'],
                             "output": ""})
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

    with open("./prompts/prompt_cause_MOOCData.txt", "w", encoding="utf-8") as g:
        g.write("")
    pref_format = "A: {}\'{}\'类; B: {}\'{}\'类; C: {}\'{}\'类"
    choice = ["曾选过", "未选过", "将被推荐选择"]
    for rec in src_pref_list:
        with open("./prompts/prompt_cause_MOOCData.txt", "a", encoding="utf-8") as g:
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


gen_rec_prompt()
#gen_cause_prompt()
