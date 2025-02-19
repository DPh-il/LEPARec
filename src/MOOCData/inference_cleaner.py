import json
import sys
import os
import re
import random
import math

list_pref = []
list_user = []
list_course = []
list_id = []


def match_string(external_string):
    for s in list_pref:
        if external_string in s:
            return s
        if s in external_string:
            return s
    return None


with open("./pref.txt", "r", encoding='utf-8') as f_ref:
    for line in f_ref:
        list_pref.append(line.strip())
with open("./user.txt", "r", encoding='utf-8') as f_ref:
    for line in f_ref:
        list_user.append(line.strip())
with open("./course.txt", "r", encoding='utf-8') as f_ref:  #strip name from id
    for line in f_ref:
        list_course.append(line.strip().split(",", maxsplit=2)[1])

# print(list_pref)

def cleanse_output(name):
    with open("./prompts/prompt_info.json", "r", encoding='utf-8') as f_src:
        list_output = json.load(f_src)
    with open("./output/output_" + name + ".txt", "r", encoding='utf-8') as f_in:
        count = 0
        while True:
            read_in = f_in.readline().strip()  # id
            if read_in == "":
                break
            seq = int(read_in)
            read_in = f_in.readline().strip()
            if seq != count:
                print(seq, count)
                print(read_in)
            list_output[count]["output"] = []
            list_output[count]["id"] = list_user[count]
            while "-------------------" not in read_in:
                #print(read_in)
                prefs = re.findall(r'\'(.*?)\'', read_in)
                appendix = re.findall(r'\"(.*?)\"', read_in)
                for pref in appendix:
                    prefs.append(pref)
                #print(prefs)
                for i in range(len(prefs) - 1, -1, -1):
                    pref = match_string(prefs[i])
                    if pref is None:
                        continue
                    if pref in list_output[count]["output"]:
                        continue
                    list_output[count]["output"].append(pref)
                read_in = f_in.readline().strip()
            count += 1
            if count % 10000 == 0:
                print(count)
    #print(list_output)
    list_output[0]["output"] = '创业', '计算机', '建筑', '艺术·设计', '教育', '经管·会计', '社科·法律', '其他'
    # list_output = [e for e in list_output if len(e["output"]) > 0]
    list_output = [e for e in list_output if e["id"] in list_user]

    with open("output/output_" + name + ".json", "w", encoding='utf-8') as f_out:
        json.dump(list_output, f_out, ensure_ascii=False, indent=4)

    count = 0
    for user in list_output:
        user_dict = {"id": count + 1, "input": [], "output": [], "timestamp": user["timestamp"]}
        for course in user["input"]:
            user_dict["input"].append(list_course.index(course) + 1)
        for pref in user["output"]:
            user_dict["output"].append(list_pref.index(pref) + 1)
        list_id.append(user_dict)
        count += 1
    """    
    with open("output/output_" + name + "_id.json", "w", encoding='utf-8') as f_out:
        json.dump(list_id, f_out, ensure_ascii=False, indent=4)
    """
    list_count = [0 for i in range(100)]

    # for TiSASRec
    with open("output/4GNN/" + "MOOC_data.txt", "w", encoding='utf-8') as f_out:
        for user in list_id:
            items = user["input"]
            timestamps = user["timestamp"]
            for i in range(len(items)):
                f_out.write("%d\t%d\t%d\t%d\n" % (user["id"], items[i], 1, timestamps[i]))

    # for DCRec/recbole models
    with open("output/4GNN/" + name + "/MOOC_data.train.inter", "w", encoding='utf-8') as f_train:
        with open("output/4GNN/" + name + "/MOOC_data.test.inter", "w", encoding='utf-8') as f_test:
            with open("output/4GNN/" + name + "/MOOC_data.valid.inter", "w", encoding='utf-8') as f_valid:
                f_train.write("session_id:token\titem_id_list:token_seq\titem_id:token\n")
                f_test.write("session_id:token\titem_id_list:token_seq\titem_id:token\n")
                f_valid.write("session_id:token\titem_id_list:token_seq\titem_id:token\n")
                for user in list_id:
                    train_rate = min(math.floor(len(user["input"]) * 0.8), len(user) - 1)
                    valid_rate = min(math.floor(len(user["input"]) * 0.9), len(user))
                    train_items = user["input"][:train_rate]
                    valid_items = user["input"][:valid_rate]
                    test_items = [item for item in user["input"]]
                    def format_items(items):
                        if len(items) > 1:
                            return " ".join(str(item) for item in items[:-1]) + "\t" + str(items[-1])
                        else:
                            return str(items[0]) if items else ""

                    f_test.write("%d\t%s\n" % (user["id"], format_items(test_items)))
                    f_train.write("%d\t%s\n" % (user["id"], format_items(train_items)))
                    f_valid.write("%d\t%s\n" % (user["id"], format_items(valid_items)))

    # for our model    
    with open("output/4GNN/" + name + "/train.txt", "w", encoding='utf-8') as f_train:
        with open("output/4GNN/" + name + "/test.txt", "w", encoding='utf-8') as f_test:
            with open("output/4GNN/" + name + "/pref.txt", "w", encoding='utf-8') as f_pref:
                for user in list_id:
                    test_rate = math.ceil(len(user["input"]) * 0.2)
                    #if test_rate < 0 or test_rate > 10:
                        #print(user["output"])
                    test_items = random.sample(user["input"], test_rate)
                    train_items = [item for item in user["input"] if item not in test_items]
                    list_count[len(test_items)] += 1
                    f_test.write("%d %s\n" % (user["id"], " ".join(str(item) for item in test_items)))
                    f_train.write("%d %s\n" % (user["id"], " ".join(str(item) for item in train_items)))
                    f_pref.write("%d %s\n" % (user["id"], " ".join(str(item) for item in user["output"])))



cleanse_output("base")
#cleanse_output("sft")
#cleanse_output("llama2")

def check_file(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        cnt = 1
        for line in file:
            text = line.split("\t")
            if len(text) < 3:
                print("fault @ line", cnt, "in file", file_name)
        #print("all good")

check_file("./output/4GNN/sft/MOOC_data.train.inter")
check_file("./output/4GNN/sft/MOOC_data.test.inter")
check_file("./output/4GNN/sft/MOOC_data.valid.inter")