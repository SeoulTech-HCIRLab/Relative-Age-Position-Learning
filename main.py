from model.networks import IR50_EVR_AgeRM_GP, IR50_EVR_AgeRM
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from data_loader import get_loader
from data_selector import get_data_specs
from opt import parse_args
from phase import *


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    age_num, labels = get_data_specs(args.data_name)

    data = {"agedb": {"gender": True, "model": IR50_EVR_AgeRM_GP},
            "afad": {"gender": True,  "model": IR50_EVR_AgeRM_GP},
            "cacd": {"gender": False, "model": IR50_EVR_AgeRM}}

    data_specs = data[args.data_name]
    is_gender = data_specs["gender"]
    model = data_specs["model"](age_num)

    print(f"Device: {device}, random seed: {args.seed}")
    print(f"Dataset: {args.data_name}, number of age labels: {age_num}, gender labels: {is_gender}")
    print(f"Model: {model.__class__.__name__}, batch size: {args.batch}, phase: {args.phase}")

    model = model.to(device)
    criterion_gender = torch.nn.CrossEntropyLoss()
    criterion_pos = torch.nn.CrossEntropyLoss()
    criterion_age = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=1e-1)

    train_results = "./train_results.csv"
    train_features = "./train_features.pt"

    train_results = [train_results, train_features]

    if args.phase == "train":
        train_loader = get_loader("train", args.data_name, args.batch, args.data_path, is_gender)
        val_loader = get_loader("val", args.data_name, args.batch, args.data_path, is_gender)

        epoch_train(args.epoch, model, train_loader, val_loader, device, criterion_age, criterion_gender,
                    criterion_pos, optimizer, scheduler, args.model_path, labels.to(device), train_results,
                    is_gender)

    elif args.phase == "test":
        test_loader = get_loader("test", args.data_name, args.batch, args.data_path, is_gender)
        model.load_state_dict(torch.load(args.model_path))

        test(test_loader, model, device, criterion_age, labels.to(device), is_gender)
    else:
        raise ValueError("Wrong phase argument.")


if __name__ == '__main__':
    main()
