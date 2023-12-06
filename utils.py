import numpy as np
import torch
import pandas as pd
import os
import random


def set_seed(seed: int = 1) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def select_refs(train_results, device):
    print(f"References selection is started.")

    train_feats = torch.load(train_results[1])
    train_results = pd.read_csv(train_results[0])

    refs = torch.empty((0, 256)).to(device)
    ages = sorted(train_results["age"].unique().tolist())

    for i in ages:
        # SELECT ROWS BASED ON CURRENT AGE
        current_age_refs = train_results.loc[(train_results["age"] == i)]

        # SELECT INDEX FOR BEST SAMPLE WITH MIN AGE ERROR
        indices = current_age_refs.index[
            current_age_refs["age_error"] == current_age_refs["age_error"].min()].tolist()

        # SELECT FEATURES OF A BEST SAMPLE
        refs = torch.cat((refs, train_feats[random.choice(indices)][None, :]), dim=0)  # [age_num, 256]

    print(f"{refs.size()[0]} references are selected.")

    return refs


def save_train_results(filenames, gt_ages, pred_ages, results):
    # SAVE FEATURES, TRUE AGES AND AGE PREDICTION ERRORS
    for file, gt_age, error in zip(filenames, gt_ages, torch.abs(torch.sub(pred_ages, gt_ages))):
        result = pd.DataFrame([{"filename": file, "age": gt_age.item(), "age_error": error.item()}])
        results = pd.concat([results, result], ignore_index=True)
    return results


def get_acc_gender(pred_genders, gt_genders):
    _, predicted_gender = torch.max(pred_genders, 1)
    return (predicted_gender == gt_genders).sum().item() / list(gt_genders.size())[0]


def get_age_pos(age_labels, ages):
    ranks = torch.empty(0)
    for age in ages:
        rank = (age_labels == age).nonzero().item()
        ranks = torch.cat((ranks, torch.tensor(rank).reshape(1)), dim=0)
    return ranks


def get_cs_age(pred_ages, gt_ages):
    age_difference = torch.sub(pred_ages, gt_ages)
    return (torch.abs(age_difference) <= 5).sum().item() / list(gt_ages.size())[0]