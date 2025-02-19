import pandas as pd
import os
import sys
import csv

dir_src = "../used/"
dir_inter = "./inter/"
dir_dst = "./dst/"
file_ml = "ml-100k"
file_mooc = "mooc"
file_cube = "mooccube"
file_music = "Musical_Instruments"
file_office = "Office_Products"
file_patio = "Patio_Lawn_and_Garden"
file_custom = "ml-1m"
per_train = 0.8
per_test = 0.1


def path_dst(file, model, suffix, type=""):
    if type == "":
        return dir_dst + model + "/" + file + suffix
    if type == "sep":
        return dir_dst + model + "/" + file + "/" + file + suffix
    return dir_dst + model + "/" + file + "/" + file + "." + type + suffix


def path_inter(file, type=""):
    if type == "":
        return dir_inter + file + ".tsv"
    return dir_inter + file + "_" + type + ".tsv"


dtype_spec = {
    'user_id': str,
    'item_id': str,
    'rating': float,
    'timestamp': float
}

df_ml = pd.read_csv(path_inter(file_ml), sep='\t', dtype=dtype_spec)
df_mooc = pd.read_csv(path_inter(file_mooc), sep='\t', dtype=dtype_spec)
df_cube = pd.read_csv(path_inter(file_cube), sep='\t', dtype=dtype_spec)
df_music = pd.read_csv(path_inter(file_music), sep='\t', dtype=dtype_spec)
df_office = pd.read_csv(path_inter(file_office), sep='\t', dtype=dtype_spec)
df_patio = pd.read_csv(path_inter(file_patio), sep='\t', dtype=dtype_spec)
df_custom = pd.read_csv(path_inter(file_custom), sep='\t', dtype=dtype_spec)

def dtype_to_custom_type(dtype):
    if dtype == 'object':
        return 'token'
    elif dtype in ['float64', 'float32']:
        return 'float'
    elif dtype in ['int64', 'int32']:
        return 'int'
    else:
        return str(dtype)


# for TiSASRec
def dump_default(df, file):  # DuoRec, TiSASRec
    df = df[["user_id", "item_id", "rating", "timestamp"]]
    header = [f"{col}:{dtype_to_custom_type(str(df[col].dtype))}" for col in df.columns]
    df.to_csv(path_dst(file, "recbole", ".inter", type="sep"), sep='\t', index=False, header=header)
    df.to_csv(path_dst(file, "TiSASRec", ".txt"), sep='\t', index=False, header=False)


def grouping(df):
    if df.empty:
        print("警告：输入的 DataFrame 为空！")
        return pd.DataFrame(columns=['user_id', 'item_id'])
    grouped = df.groupby('user_id')['item_id'].agg(lambda x: ' '.join(x.astype(str))).reset_index()
    if grouped.empty:
        print("警告：分组后的 DataFrame 为空！")
    return grouped


def printing_data(df, path, type=""):
    with open(path, 'w', newline='') as file:
        for _, row in df.iterrows():
            line = f"{row['user_id']}\t{row['item_id']}\n"
            file.write(line)
    return


def dump_splited(df, file):  # DCRec
    bound_train = int(per_train * len(df))
    bound_test = int((1 - per_test) * len(df))
    train_set = df.iloc[:bound_train]
    #valid_set = df.iloc[bound_train:bound_test]
    #test_set = df.iloc[bound_test:]
    valid_set = pd.concat([train_set, df.iloc[bound_train:bound_test]], ignore_index=True)
    test_set = pd.concat([train_set, df.iloc[bound_test:]], ignore_index=True)
    train_grouped = grouping(train_set)
    valid_grouped = grouping(valid_set)
    test_grouped = grouping(test_set)
    printing_data(train_grouped, path_dst(file, "ICLRec", ".inter", "train"))
    printing_data(valid_grouped, path_dst(file, "ICLRec", ".inter", "valid"))
    printing_data(test_grouped, path_dst(file, "ICLRec", ".inter", "test"))


def dump_mess(df, file):
    bound_train = int(per_train * len(df))
    bound_test = int((1 - per_test) * len(df))
    train_set = df.iloc[:bound_train]
    valid_set = df.iloc[bound_train:bound_test]
    test_set = df.iloc[bound_test:]
    train_grouped = grouping(train_set)
    valid_grouped = grouping(valid_set)
    test_grouped = grouping(test_set)

    if bound_test <= bound_train:
        raise ValueError("数据划分错误：测试集边界应大于训练集边界")
    if len(test_set) == 0:
        raise ValueError("测试集为空，请检查划分参数或数据大小")
    print(f"总数据集大小: {len(df)}")
    print(f"训练集大小: {len(train_set)} ({len(train_set) / len(df):.2%})")
    print(f"验证集大小: {len(valid_set)} ({len(valid_set) / len(df):.2%})")
    print(f"测试集大小: {len(test_set)} ({len(test_set) / len(df):.2%})")
    print(len(train_grouped))
    print(len(valid_grouped))
    print(len(test_grouped))

    train_path = path_dst(file, "DCRec", ".inter", "train")
    test_path = path_dst(file, "DCRec", ".inter", "test")
    valid_path = path_dst(file, "DCRec", ".inter", "valid")
    with open(train_path, 'w', newline='') as f_train, \
            open(test_path, 'w', newline='') as f_test, \
            open(valid_path, 'w', newline='') as f_valid:
        headers = "session_id:token\titem_id_list:token_seq\titem_id:token\n"
        f_train.write(headers)
        f_test.write(headers)
        f_valid.write(headers)

        valid_session = 0
        test_session = 0

        for _, row in train_grouped.iterrows():
            user_id = row['user_id']
            item_ids = list(map(int, row['item_id'].split(' ')))

            # 写入 train_line，仅当 item_ids 长度 >= 2
            if len(item_ids) >= 2:
                train_line = f"{user_id}\t{' '.join(map(str, item_ids[:-1]))}\t{item_ids[-1]}\n"
                f_train.write(train_line)

            # 处理 valid_grouped
            if user_id in valid_grouped['user_id'].values:
                valid_items = valid_grouped.loc[valid_grouped['user_id'] == user_id, 'item_id'].iloc[0] # TODO
                print(valid_items)
                if not isinstance(valid_items, str) or not valid_items.strip():
                    print(f"警告：用户 {user_id} 的验证集 item_id 数据为空，跳过写入")
                    continue
                valid_items = list(map(int, valid_items.split(' ')))
                for idx, item in enumerate(valid_items):
                    valid_line = f"{valid_session}\t{' '.join(map(str, item_ids))}{' ' if idx > 0 else ''}{' '.join(['1'] * idx)}\t{item}\n"
                    f_valid.write(valid_line)
                    valid_session += 1

            # 处理 test_grouped
            if user_id in test_grouped['user_id'].values:
                test_items = test_grouped.loc[test_grouped['user_id'] == user_id, 'item_id'].iloc[0]    # TODO
                print(test_items)
                if not isinstance(test_items, str) or not test_items.strip():
                    print(f"警告：用户 {user_id} 的测试集 item_id 数据为空，跳过写入")
                    continue
                test_items = list(map(int, test_items.split(' ')))
                for idx, item in enumerate(test_items):
                    valid_items = valid_items if valid_items else []
                    place_holders = ' '.join(['1'] * (len(valid_items) + idx))
                    test_line = f"{test_session}\t{' '.join(map(str, item_ids))}{(' ' + place_holders) if place_holders else ''}\t{item}\n"
                    f_test.write(test_line)
                    test_session += 1
    train_path = path_dst(file, "BERT4Rec", ".inter", "train")
    test_path = path_dst(file, "BERT4Rec", ".inter", "test")
    valid_path = path_dst(file, "BERT4Rec", ".inter", "valid")
    with open(train_path, 'w', newline='') as f_train, \
            open(test_path, 'w', newline='') as f_test, \
            open(valid_path, 'w', newline='') as f_valid:
        headers = "session_id:token\titem_id_list:token_seq\titem_id:token\n"
        f_train.write(headers)
        f_test.write(headers)
        f_valid.write(headers)

        valid_session = 0
        test_session = 0

        for _, row in train_grouped.iterrows():
            user_id = row['user_id']
            item_ids = list(map(int, row['item_id'].split(' ')))
            if len(item_ids) < 2:
                continue
            valid_items = []

            base_line = f"{' '.join(map(str, item_ids))}"
            train_line = f"{user_id}\t{' '.join(map(str, item_ids[:-1]))}\t{item_ids[-1]}\n"
            f_train.write(train_line)

            if user_id in valid_grouped['user_id'].values:
                valid_items = list(
                    map(int, valid_grouped[valid_grouped['user_id'] == user_id]['item_id'].iloc[0].split(' ')))
                for idx, item in enumerate(valid_items):
                    valid_line = f"{valid_session}\t{base_line}{' ' if idx > 0 else ''}{' '.join(map(str, valid_items[:idx]))}\t{item}\n"
                    f_valid.write(valid_line)
                    valid_session += 1

            if user_id in test_grouped['user_id'].values:
                test_items = list(
                    map(int, test_grouped[test_grouped['user_id'] == user_id]['item_id'].iloc[0].split(' ')))
                for idx, item in enumerate(test_items):
                    # 构建占位符字符串，考虑valid_items的长度和当前idx
                    place_holders = ' '.join(map(str, valid_items + test_items[:idx]))
                    # 仅在place_holders非空时添加前导空格
                    test_line = f"{test_session}\t{base_line}{(' ' + place_holders) if place_holders else ''}\t{item}\n"
                    f_test.write(test_line)
                    test_session += 1

def dump_all(df, file):
    # dump_default(df, file)
    # dump_splited(df, file)
    dump_mess(df, file)



#dump_all(df_ml, file_ml)
"""
dump_all(df_mooc, file_mooc)
dump_all(df_cube, file_cube)
dump_all(df_music, file_music)
dump_all(df_office, file_office)
dump_all(df_patio, file_patio)
"""
dump_all(df_custom, file_custom)
