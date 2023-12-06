import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
from data_selector import select_data

IM_WIDTH = IM_HEIGHT = 112
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class Loader(Dataset):

    def __init__(self, df, data_path, is_gender, transform):
        self.filenames = df['filename'].values
        self.ages = df['age'].values
        self.is_gender = is_gender
        if self.is_gender:
            self.genders = df["gender"].values
        self.normalize = transforms.Normalize(MEAN, STD)
        self.data_path = data_path
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.data_path + self.filenames[index], mode='r').convert("RGB")
        img = self.transform(img)
        img = F.to_tensor(img)
        img = self.normalize(img)
        age = torch.tensor(self.ages[index], dtype=torch.float32)
        filename = self.filenames[index]
        if self.is_gender:
            gender = torch.tensor(self.genders[index])
            return filename, img, age, gender
        else:
            return filename, img, age

    def __len__(self):
        return self.filenames.shape[0]


def get_transforms(phase):
    if phase == "train":
        train_transform = transforms.Compose([transforms.Resize((IM_HEIGHT, IM_WIDTH)),
                                              transforms.RandomGrayscale(0.1),
                                              transforms.RandomHorizontalFlip(0.2),
                                              transforms.RandomRotation(degrees=30),
                                              transforms.RandomAdjustSharpness(0.2),
                                              transforms.RandomVerticalFlip(0.2)])
        return train_transform
    else:
        test_transforms = transforms.Resize((IM_HEIGHT, IM_WIDTH))
        return test_transforms


def get_loader(phase, data, batch, data_path, is_gender):
    df = select_data(data=data, phase=phase)
    transform = get_transforms(phase=phase)
    if phase == "train":
        shuffle = True
    else:
        shuffle = False
    dataset = Loader(df=df, data_path=data_path, is_gender=is_gender, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch, shuffle=shuffle, num_workers=4)

    return loader
