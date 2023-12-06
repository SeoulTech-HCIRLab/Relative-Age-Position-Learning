import torch
import torch.nn as nn
from model.IR50 import Backbone as IR50
from torch.nn.parameter import Parameter
from einops import repeat


def load_state_dict(self, state_dict):
    model_state = self.state_dict()
    for name, param in state_dict.items():
        if name not in model_state:
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        model_state[name].copy_(param)


class IR50_EVR_AgeRM_GP(nn.Module):
    def __init__(self, age_num):
        super().__init__()
        self.age_num = age_num
        self.backbone = IR50()
        # load_state_dict(self.backbone, torch.load('pretrained_model/ir50_imdb.pt'))
        self.fc_gender = nn.Linear(256, 2)
        self.fc_age = nn.Linear(256, self.age_num)
        self.fc_feature = nn.Linear(self.age_num, 1)
        self.fc_pos = nn.Linear(256, self.age_num)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = torch.nn.Softmax(dim=1)
        self.parameter = Parameter(torch.Tensor(self.age_num))
        self.sigmoid = torch.nn.Sigmoid()

    def re_weight(self, x, refs):
        # FEATURE REPRESENTATION
        refs = refs.permute(1, 0)  # [256, n]
        refs = refs * self.parameter[None, :]
        # AGE WEIGHTS
        weights = self.sigmoid(refs)  # [256, n]
        # RE-WEIGHT
        x = repeat(x[:, :, None], 'b c n -> b c (repeat n)', repeat=self.age_num)  # [b, 256, n]
        x = x * weights[None, :, :]  # [b, 256, n]

        return x

    def forward(self, x, refs_feats=None, re_weighting=False):
        x = self.backbone(x)  # [b, 256, 14, 14]
        x = torch.flatten(self.avgpool(x), 1)  # [b, 256]

        features = x

        age_out = self.softmax(self.fc_age(x))  # [b, 256] --> [b, n]
        gender_out = self.fc_gender(x)  # [b, 256] --> [b, 2]

        if re_weighting:
            out = self.re_weight(x, refs_feats)  # [b, 256, n]
            pos_out = self.fc_pos(torch.flatten(self.fc_feature(out), 1))  # [b, n]
            return features, age_out, gender_out, pos_out

        else:
            return features, age_out, gender_out


class IR50_EVR_AgeRM(nn.Module):
    def __init__(self, age_num):
        super().__init__()
        self.age_num = age_num
        self.backbone = IR50()
        # load_state_dict(self.backbone, torch.load('pretrained_model/ir50_imdb.pt'))
        self.fc_age = nn.Linear(256, self.age_num)
        self.fc_feature = nn.Linear(self.age_num, 1)
        self.fc_pos = nn.Linear(256, self.age_num)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = torch.nn.Softmax(dim=1)
        self.parameter = Parameter(torch.Tensor(self.age_num))
        self.sigmoid = torch.nn.Sigmoid()

    def re_weight(self, x, refs):
        # FEATURE REPRESENTATION
        refs = refs.permute(1, 0)  # [256, n]
        refs = refs * self.parameter[None, :]
        # AGE WEIGHTS
        weights = self.sigmoid(refs)  # [256, n]
        # RE-WEIGHT
        x = repeat(x[:, :, None], 'b c n -> b c (repeat n)', repeat=self.age_num)  # [b, 256, n]
        x = x * weights[None, :, :]  # [b, 256, n]

        return x

    def forward(self, x, refs_feats=None, re_weighting=False):
        x = self.backbone(x)  # [b, 256, 14, 14]
        x = torch.flatten(self.avgpool(x), 1)  # [b, 256]

        features = x

        age_out = self.softmax(self.fc_age(x))  # [b, 256] --> [b, n]

        if re_weighting:
            out = self.re_weight(x, refs_feats)  # [b, 256, n]
            pos_out = self.fc_pos(torch.flatten(self.fc_feature(out), 1))  # [b, n]
            return features, age_out, pos_out

        else:
            return features, age_out

