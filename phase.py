from tqdm import tqdm
from utils import *


def train(loader, model, refs, device, criterion_age, criterion_gender, criterion_pos, labels, optimizer, results,
          input_feats, re_weighting, is_gender):
    print("TRAINING:")
    model.train()
    age_mae = age_cs = gender_acc = 0
    for i, data in tqdm(enumerate(loader), total=len(loader)):
        if is_gender:
            files, inputs, gt_ages, gt_genders = data
            gt_genders = gt_genders.to(device)
        else:
            files, inputs, gt_ages = data
        gt_ages = gt_ages.to(device)
        gt_pos = get_age_pos(labels, gt_ages)
        gt_pos = gt_pos.type(torch.LongTensor).to(device)

        if re_weighting:
            if is_gender:
                features, pred_ages, pred_genders, pred_pos = model(inputs.to(device), refs, re_weighting)

                gender_loss = criterion_gender(pred_genders, gt_genders)
                gender_acc += get_acc_gender(pred_genders, gt_genders)

            else:
                features, pred_ages, pred_pos = model(inputs.to(device), refs, re_weighting)

            pos_loss = criterion_pos(pred_pos, gt_pos)

        else:
            if is_gender:
                features, pred_ages, pred_genders = model(inputs.to(device), refs, re_weighting)
                gender_loss = criterion_gender(pred_genders, gt_genders)
                gender_acc += get_acc_gender(pred_genders, gt_genders)
            else:
                features, pred_ages = model(inputs.to(device), refs, re_weighting)

        pred_ages = torch.sum(pred_ages * labels, dim=1)
        age_loss = criterion_age(pred_ages, gt_ages)
        age_mae += age_loss.item()

        age_cs += get_cs_age(pred_ages, gt_ages)

        if re_weighting:
            if is_gender:
                loss = age_loss + gender_loss + pos_loss
            else:
                loss = age_loss + pos_loss
        else:
            if is_gender:
                loss = age_loss + gender_loss
            else:
                loss = age_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # SAVE FEATURES, TRUE AGES AND AGE PREDICTIONS
        results = save_train_results(files, gt_ages, pred_ages, results)
        input_feats = torch.cat((input_feats, features.detach()), dim=0)

    return age_mae, age_cs, gender_acc, results, input_feats


def val(loader, model, device, criterion_age, labels, is_gender):
    model.eval()
    age_mae = age_cs = gender_acc = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            if is_gender:
                files, inputs, gt_ages, gt_genders = data
                gt_genders = gt_genders.to(device)
            else:
                files, inputs, gt_ages = data
            gt_ages = gt_ages.to(device)
            if is_gender:
                features, pred_ages, pred_genders = model(x=inputs.to(device))
                gender_acc += get_acc_gender(pred_genders, gt_genders)
                _, predicted_gender = torch.max(pred_genders, 1)
            else:
                features, pred_ages = model(x=inputs.to(device))

            pred_ages = torch.sum(pred_ages * labels, dim=1)
            age_loss = criterion_age(pred_ages, gt_ages)
            age_mae += age_loss.item()

            age_cs += get_cs_age(pred_ages, gt_ages)

    return age_mae, age_cs, gender_acc


def test(loader, model, device, criterion_age, age_labels, is_gender):
    print("TESTING:")

    age_mae, age_cs, gender_acc = val(loader, model, device, criterion_age, age_labels, is_gender)
    print(f"AGE --- MAE: {round(age_mae / len(loader), 4)}")
    print(f"AGE --- CS: {round(age_cs / len(loader) * 100, 4)}")

    if is_gender:
        print(f"GENDER --- ACC: {round(gender_acc / len(loader) * 100, 4)}")


def epoch_train(epoch_num, model, train_loader, val_loader, device, criterion_age, criterion_gender, criterion_pos,
                optimizer, scheduler, model_path, age_labels, train_results, is_gender):
    min_loss = np.inf
    for epoch in range(epoch_num):
        print(f"\nEpoch #{epoch + 1}")
        if epoch == 0:
            re_weighting = False
            refs = None
        else:
            re_weighting = True
            refs = select_refs(train_results=train_results, device=device)

        print(f"Re-weighting module is set to {re_weighting}")

        input_feats = torch.empty((0, 256)).to(device)
        results = pd.DataFrame(columns=["filename", "age", "age_error"])

        age_mae, age_cs, gender_acc, results, input_feats = (
            train( train_loader, model, refs, device, criterion_age, criterion_gender, criterion_pos, age_labels,
                   optimizer, results, input_feats, re_weighting, is_gender))

        print(f"AGE --- MAE: {round(age_mae / len(train_loader), 4)}")
        print(f"AGE --- CS: {round(age_cs / len(train_loader) * 100, 4)}")
        if is_gender:
            print(f"GENDER --- ACC: {round(gender_acc / len(train_loader) * 100, 4)}")

        print("VALIDATION:")
        age_mae, age_cs, gender_acc = val(val_loader, model, device, criterion_age, age_labels, is_gender)
        print(f"AGE --- MAE: {round(age_mae / len(val_loader), 4)}")
        print(f"AGE --- CS: {round(age_cs / len(val_loader) * 100, 4)}")
        if is_gender:
            print(f"GENDER --- ACC: {round(gender_acc / len(val_loader) * 100, 4)}")

        # SAVE TRAIN RESULTS
        if age_mae < min_loss:
            min_loss = age_mae
            results.to_csv(train_results[0], index=False)
            torch.save(input_feats, train_results[1])
            torch.save(model.state_dict(), model_path)
            print("Training results are saved.")

        scheduler.step()
