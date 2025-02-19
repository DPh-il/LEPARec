import csv
import json
from datetime import datetime
from collections import defaultdict, Counter

def convert_timestamp(timestamp: str) -> int:
    # 定义输入的时间格式
    time_format = "%Y/%m/%d %H:%M"
    # 将字符串时间戳转化为datetime对象
    dt_object = datetime.strptime(timestamp, time_format)
    # 转化为可排序的数字时间戳（以秒为单位的时间戳）
    numeric_timestamp = int(dt_object.timestamp())
    return numeric_timestamp


def extract_types_with_highest_frequency(csv_file_path, types_txt_file_path):
    type_dict = defaultdict(list)

    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            type_id = row['type_id'].strip()
            type_name = row['type'].strip().split()[0] if row['type'].strip() else ''  # 只保留第一个空格前的词汇
            if type_id.isdigit():  # 确保 type_id 是有效的整数
                type_dict[type_id].append(type_name)

    # 保留每个type_id下出现频率最高的type名称
    highest_frequency_types = {}
    for type_id, types in type_dict.items():
        type_counter = Counter(types)
        most_common_type = type_counter.most_common(1)[0][0]
        highest_frequency_types[type_id] = most_common_type

    # 按照type_id排序
    sorted_types = sorted(highest_frequency_types.items(), key=lambda x: int(x[0]))

    # 写入TXT文件
    with open(types_txt_file_path, mode='w', encoding='utf-8') as txt_file:
        for type_id, type_name in sorted_types:
            txt_file.write(f"{type_name}\n")


def extract_course_names(csv_file_path, names_txt_file_path, valid_course_ids):
    names = set()

    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if row['course_index'] in valid_course_ids:
                names.add((row['course_index'], row['name']))

    # 排序
    sorted_names = sorted(names, key=lambda x: int(x[0]))

    # 写入TXT文件
    with open(names_txt_file_path, mode='w', encoding='utf-8') as txt_file:
        for course_id, name in sorted_names:
            txt_file.write(f"{course_id},{name}\n")


def read_courses(csv_file_path):
    courses = {}
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            courses[row['course_index']] = row['name']
    return courses

def read_students(csv_file_path):
    students = defaultdict(list)
    timestamps = defaultdict(list)
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            students[row['stu_id']].append(row['course_index'])
            timestamps[row['stu_id']].append(convert_timestamp(row['time']))
    return students, timestamps

def generate_course_lists(students, courses):
    students_courses_text = {}
    students_courses_index = {}
    valid_user_ids = []
    valid_course_ids = set()

    for stu_id, course_indices in students.items():
        if len(course_indices) >= 5:  # 过滤掉选课数小于5的用户
            course_names = [courses[course_index] for course_index in course_indices]
            students_courses_index[stu_id] = {
                "input": course_indices,
                "course_count": len(course_indices),
                "timestamp": timestamps[stu_id],
                "output": ""
            }
            students_courses_text[stu_id] = {
                "input": course_names,
                "course_count": len(course_names),
                "timestamp": timestamps[stu_id],
                "output": ""
            }
            valid_user_ids.append(int(stu_id))
            valid_course_ids.update(course_indices)  # 记录有效课程ID

    return students_courses_text, students_courses_index, sorted(valid_user_ids), valid_course_ids

def save_to_json(data, json_file_path):
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

def save_user_ids(user_ids, user_txt_file_path):
    with open(user_txt_file_path, mode='w', encoding='utf-8') as txt_file:
        for user_id in user_ids:
            txt_file.write(f"{user_id}\n")

courses_csv = '../../Dataset/mooc_data/course.csv'  # 替换为您的课程CSV文件名
relation_csv = '../../Dataset/mooc_data/learn_relation.csv'  # 替换为您的学生CSV文件名

# 读取课程和学生数据
courses = read_courses(courses_csv)
students, timestamps = read_students(relation_csv)

# 生成课程列表和用户ID列表
students_courses_text, students_courses_index, valid_user_ids, valid_course_ids = generate_course_lists(students, courses)

# 保存到JSON文件
save_to_json(students_courses_text, 'students_courses_text.json')
save_to_json(students_courses_index, 'students_courses_index.json')

# 保存用户ID到TXT文件
user_txt = 'user.txt'  # 保存用户ID的TXT文件名
save_user_ids(valid_user_ids, user_txt)

# 提取类型到TXT文件
pref_txt = 'pref.txt'  # 保存类型的TXT文件名
extract_types_with_highest_frequency(courses_csv, pref_txt)

# 提取课程名称到TXT文件
course_txt = 'course.txt'  # 保存名称的TXT文件名
extract_course_names(courses_csv, course_txt, valid_course_ids)


# 提取课程-偏好关系
def extract_relations(pref_txt, relation_txt):
    course_rank = {}
    courses = {}
    prefs = {}
    with open(relation_csv, mode="r", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if row['course_index'] not in course_rank.keys():
                course_rank[row['course_index']] = 0
            course_rank[row['course_index']] += 1

    def get_key(dict, value):
        return [k for k, v in dict.items() if v > value]
    chosen_courses = get_key(course_rank, 200)
    print(len(chosen_courses))

    with open(pref_txt, mode="r", encoding='utf-8') as txt_file:
        id = 0
        for row in txt_file:
            prefs[row.strip()] = id + 1
            id += 1
        #print(prefs)
    def prefer(item):
        if item in prefs.keys():
            return str(prefs[item])
        else:
            return ""
    with open(courses_csv, mode='r', encoding='utf-8') as csv_file:
        with open(relation_txt, mode="w", encoding='utf-8') as txt_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row['course_index'] in courses.keys():
                    continue
                courses[row['course_index']] = row['type'].split(' ')
                if row['course_index'] in chosen_courses:
                    line = " ".join(map(prefer, row['type'].split(' ')))
                    print(row['course_index'], line, file=txt_file)

            #print(courses)


relation_txt = 'relation.txt'
extract_relations(pref_txt, relation_txt)




print("done")
