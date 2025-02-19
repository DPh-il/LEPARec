import pandas as pd
import os
import sys

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


def path_src(file):
    return dir_src + file + ".csv"


def path_inter(file, type=""):
    if type == "":
        return dir_inter + file + ".tsv"
    return dir_inter + file + "_" + type + ".tsv"


df_ml = pd.read_csv(path_src(file_ml), usecols=['u', 'i', 'ts'])
df_ml = df_ml.rename(columns={'u': 'user_id', 'i': 'item_id', 'ts': 'timestamp'})
df_mooc = pd.read_csv(path_src(file_mooc), usecols=['user_id', 'item_id', 'timestamp'])
df_cube = pd.read_csv(path_src(file_cube), usecols=['user_id', 'item_id', 'timestamp'])
df_music = pd.read_csv(path_src(file_music), usecols=['user_id', 'item_id', 'timestamp'])
df_office = pd.read_csv(path_src(file_office), usecols=['user_id', 'item_id', 'timestamp'])
df_patio = pd.read_csv(path_src(file_patio), usecols=['user_id', 'item_id', 'timestamp'])
df_custom = pd.read_csv(path_src(file_custom), usecols=['u', 'i', 'ts'])
df_custom = df_custom.rename(columns={'u': 'user_id', 'i': 'item_id', 'ts': 'timestamp'})

df_ml["rating"] = 1
df_mooc["rating"] = 1
df_cube["rating"] = 1
df_music["rating"] = 1
df_office["rating"] = 1
df_patio["rating"] = 1
df_custom["rating"] = 1

"""
tuple: (user, item, timestamp)
8-1-1 partition, timestamp-ordered
"""
per_train = 0.8
per_test = 0.1


def dump_train(df, file):
    bound_train = int(per_train * len(df))
    df.iloc[:bound_train].to_csv(path_inter(file, "train"), sep='\t', index=False)


def dump_valid(df, file):
    bound_train = int(per_train * len(df))
    bound_test = int((1 - per_test) * len(df))
    df.iloc[bound_train:bound_test].to_csv(path_inter(file, "valid"), sep='\t', index=False)


def dump_test(df, file):
    bound_test = int((1 - per_test) * len(df))
    df.iloc[bound_test:].to_csv(path_inter(file, "test"), sep='\t', index=False)


def dump_total(df, file):
    df = df[["user_id", "item_id", "rating", "timestamp"]]
    df.to_csv(path_inter(file), sep='\t', index=False)

"""
dump_total(df_ml, file_ml)
dump_total(df_mooc, file_mooc)
dump_total(df_cube, file_cube)
dump_total(df_music, file_music)
dump_total(df_office, file_office)
dump_total(df_patio, file_patio)
"""
dump_total(df_custom, file_custom)