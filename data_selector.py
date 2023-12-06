import pandas as pd
import torch


def select_data(data, phase):
    df = pd.read_csv("./datalists/" + data + ".csv")
    df = df[df["fold"] == phase]
    return df


def get_labels(data_name):
    return torch.tensor(sorted(pd.read_csv(f"./datalists/{data_name}.csv")["age"].unique().tolist()))


def get_data_specs(data_name):
    datas = {"agedb": {"age_num": 100}, "afad": {"age_num": 57}, "cacd": {"age_num": 49}, "utkface": {"age_num": 40}}

    age_num = datas[data_name]["age_num"]
    labels = get_labels(data_name)

    return age_num, labels
