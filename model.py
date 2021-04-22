import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet


class BaseModel(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


class Vgg19BasedModel(nn.Module):
    def __init__(self, num_classes: int = 18, grad_point: int = 12, test:bool = False):
        """
            get pretrained vgg19 model. can set criterion where require_grad begin.
            :param num_classes: output classes numbers
            :param criterion: (0 <= criterion <=16) decide start point where require_grad begin in cnn module.
            :return: torch.nn.Module
            """
        if not (0 <= grad_point <= 16):
            raise RuntimeError("criterion out of range. it must be between 0 and 16")

        super().__init__()

        if test:
            vgg19 = models.vgg19(pretrained=False)
        else:
            vgg19 = models.vgg19(pretrained=True)

        self.my_model = nn.Sequential()

        features = nn.Sequential()
        i = 0  # count up every time meet conv layer in order to give them number
        for layer in vgg19.features.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f"conv_{i}"
            elif isinstance(layer, nn.ReLU):
                name = f"relu_{i}"
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                # must inplace=False
                # if not, runtimeerror occur when opimize backward
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f"maxpool_{i}"
            elif isinstance(layer, nn.BatchNorm2d):
                name = f"bn_{i}"
            else:
                raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

            if i < grad_point:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True

            features.add_module(name, layer)
        self.my_model.add_module("features", features)

        self.my_model.add_module("avgpool", vgg19.avgpool)

        self.my_model.add_module("flatten", nn.Flatten())

        classifier = nn.Sequential()
        i = 0
        for layer in vgg19.classifier.children():
            if isinstance(layer, nn.Linear):
                i += 1
                name = f"linear_{i}"
            elif isinstance(layer, nn.ReLU):
                name = f"relu_{i}"
            elif isinstance(layer, nn.Dropout):
                name = f"dropout_{i}"
            else:
                raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

            classifier.add_module(name, layer)

        classifier.add_module(f"relu_{i}", nn.ReLU(inplace=True))
        classifier.add_module(f"dropout_{i}", nn.Dropout())
        classifier.add_module("linear_last", nn.Linear(1000, num_classes))

        self.my_model.add_module("classifier", classifier)

    def forward(self,x):
        return self.my_model(x)


class EfficientNet_b4(nn.Module):
    def __init__(self, num_classes: int = 18, test:bool = False, **kwargs):
        super().__init__()

        if test:
            self.my_model = EfficientNet.from_pretrained('efficientnet-b4', advprop=False, num_classes=num_classes)
        else:
            self.my_model = EfficientNet.from_pretrained('efficientnet-b4', advprop=True, num_classes=num_classes)

    def forward(self, x):
        return self.my_model(x)


class EffieicientBasedModel(nn.Module):
    def __init__(self, num_classes: int = 18, test:bool = False, **kwargs):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4', advprop=True)
        # self.my_model = nn.Sequential(*model.children())
        # for idx, param in enumerate(model.children()):
        #     self.my_model.add_module(str(idx), param)
        #     if idx == 7: break
        self.mask = nn.Linear(1000, 3)
        self.gender = nn.Linear(1000, 2)
        self.age = nn.Linear(1000, 3)

    def forward(self, x):
        x = self.model(x)
        mask = self.mask(x)
        gender = self.gender(x)
        age = self.age(x)
        return mask, gender, age

class EffieicientBasedModel2(nn.Module):
    def __init__(self, num_classes: int = 18, test:bool = False, **kwargs):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4', advprop=True)
        # self.my_model = nn.Sequential(*model.children())
        # for idx, param in enumerate(model.children()):
        #     self.my_model.add_module(str(idx), param)
        #     if idx == 7: break
        self.mask1 = nn.Linear(1000, 512)
        self.mask2 = nn.Linear(512, 256)
        self.mask3 = nn.Linear(256, 128)
        self.mask4 = nn.Linear(128, 56)
        self.mask5 = nn.Linear(56, 3)
        self.gender1 = nn.Linear(1000, 512)
        self.gender2 = nn.Linear(512, 256)
        self.gender3 = nn.Linear(256, 128)
        self.gender4 = nn.Linear(128, 56)
        self.gender5 = nn.Linear(56, 2)
        self.age1 = nn.Linear(1000, 512)
        self.age2 = nn.Linear(512, 256)
        self.age3 = nn.Linear(256, 128)
        self.age4 = nn.Linear(128, 56)
        self.age5 = nn.Linear(56, 3)

    def forward(self, x):
        x = self.model(x)
        mask = self.mask1(x)
        mask = self.mask2(mask)
        mask = self.mask3(mask)
        mask = self.mask4(mask)
        mask = self.mask5(mask)
        gender = self.gender1(x)
        gender = self.gender2(gender)
        gender = self.gender3(gender)
        gender = self.gender4(gender)
        gender = self.gender5(gender)
        age = self.age1(x)
        age = self.age2(age)
        age = self.age3(age)
        age = self.age4(age)
        age = self.age5(age)
        return mask, gender, age


class EffieicientBasedModel3(nn.Module):
    def __init__(self, num_classes: int = 18, test:bool = False, **kwargs):
        super(EffieicientBasedModel3,self).__init__()
        efmodel = EfficientNet.from_pretrained('efficientnet-b0', advprop=True, include_top=False, in_channels=3)
        self.conv_stem = efmodel._conv_stem
        self.bn0 = efmodel._bn0
        self.blocks = nn.Sequential(*efmodel._blocks)
        self.splitlayer1 = nn.Sequential(
            nn.Conv2d(320, 640, kernel_size=3, stride=1),
            nn.Conv2d(640, 320, kernel_size=3, stride=1),
            efmodel._conv_head,
            efmodel._avg_pooling,
            efmodel._dropout,
            nn.Flatten(),
            efmodel._fc,
            efmodel._swish
        )
        self.splitlayer2 = nn.Sequential(
            nn.Conv2d(320, 640, kernel_size=3, stride=1),
            nn.Conv2d(640, 320, kernel_size=3, stride=1),
            efmodel._conv_head,
            efmodel._avg_pooling,
            efmodel._dropout,
            nn.Flatten(),
            efmodel._fc,
            efmodel._swish
        )
        self.splitlayer3 = nn.Sequential(
            nn.Conv2d(320, 640, kernel_size=3, stride=1),
            nn.Conv2d(640, 320, kernel_size=3, stride=1),
            efmodel._conv_head,
            efmodel._avg_pooling,
            efmodel._dropout,
            nn.Flatten(),
            efmodel._fc,
            efmodel._swish
        )
        self.mask1 = nn.Linear(1000, 512)
        self.mask2 = nn.Linear(512, 128)
        self.mask3 = nn.Linear(128, 3)
        self.gender1 = nn.Linear(1000, 512)
        self.gender2 = nn.Linear(512, 128)
        self.gender3 = nn.Linear(128, 2)
        self.age1 = nn.Linear(1000, 512)
        self.age2 = nn.Linear(512, 128)
        self.age3 = nn.Linear(128, 3)


    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn0(x)
        x = self.blocks(x)
        mask = self.splitlayer1(x)
        gender = self.splitlayer2(x)
        age = self.splitlayer3(x)
        mask = self.mask1(mask)
        mask = self.mask2(mask)
        mask = self.mask3(mask)
        gender = self.gender1(gender)
        gender = self.gender2(gender)
        gender = self.gender3(gender)
        age = self.age1(age)
        age = self.age2(age)
        age = self.age3(age)
        return mask, gender, age


class EffieicientBasedModel4(nn.Module):
    def __init__(self, num_classes: int = 18, test:bool = False, **kwargs):
        super(EffieicientBasedModel4,self).__init__()
        efmodel = EfficientNet.from_pretrained('efficientnet-b0', advprop=True, include_top=False, in_channels=3)
        self.conv_stem = efmodel._conv_stem
        self.bn0 = efmodel._bn0
        self.blocks = nn.Sequential(*efmodel._blocks)
        self.splitlayer1 = nn.Sequential(
            nn.Conv2d(320, 1152, kernel_size=3, stride=1),
            nn.Conv2d(1152, 58, kernel_size=3, stride=1),
            nn.Conv2d(58, 320, kernel_size=3, stride=1),
            efmodel._conv_head,
            efmodel._avg_pooling,
            efmodel._dropout,
            nn.Flatten(),
            efmodel._fc,
            efmodel._swish
        )
        self.splitlayer2 = nn.Sequential(
            nn.Conv2d(320, 1152, kernel_size=3, stride=1),
            nn.Conv2d(1152, 58, kernel_size=3, stride=1),
            nn.Conv2d(58, 320, kernel_size=3, stride=1),
            efmodel._conv_head,
            efmodel._avg_pooling,
            efmodel._dropout,
            nn.Flatten(),
            efmodel._fc,
            efmodel._swish
        )
        self.splitlayer3 = nn.Sequential(
            nn.Conv2d(320, 1152, kernel_size=3, stride=1),
            nn.Conv2d(1152, 58, kernel_size=3, stride=1),
            nn.Conv2d(58, 320, kernel_size=3, stride=1),
            efmodel._conv_head,
            efmodel._avg_pooling,
            efmodel._dropout,
            nn.Flatten(),
            efmodel._fc,
            efmodel._swish
        )
        self.mask1 = nn.Linear(1000, 512)
        self.mask2 = nn.Linear(512, 256)
        self.mask3 = nn.Linear(256, 128)
        self.mask4 = nn.Linear(128, 56)
        self.mask5 = nn.Linear(56, 3)
        self.gender1 = nn.Linear(1000, 512)
        self.gender2 = nn.Linear(512, 256)
        self.gender3 = nn.Linear(256, 128)
        self.gender4 = nn.Linear(128, 56)
        self.gender5 = nn.Linear(56, 2)
        self.age1 = nn.Linear(1000, 512)
        self.age2 = nn.Linear(512, 256)
        self.age3 = nn.Linear(256, 128)
        self.age4 = nn.Linear(128, 56)
        self.age5 = nn.Linear(56, 3)


    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn0(x)
        x = self.blocks(x)
        mask = self.splitlayer1(x)
        gender = self.splitlayer2(x)
        age = self.splitlayer3(x)
        mask = self.mask1(mask)
        mask = self.mask2(mask)
        mask = self.mask3(mask)
        mask = self.mask4(mask)
        mask = self.mask5(mask)
        gender = self.gender1(gender)
        gender = self.gender2(gender)
        gender = self.gender3(gender)
        gender = self.gender4(gender)
        gender = self.gender5(gender)
        age = self.age1(age)
        age = self.age2(age)
        age = self.age3(age)
        age = self.age4(age)
        age = self.age5(age)
        return mask, gender, age


model_dict = {
    # "basemodel": BaseModel,
    # "vgg19basedmodel": Vgg19BasedModel,
    # "efficientnetb4": EfficientNet_b4,
    "efficientnetbasedmodel2": EffieicientBasedModel2,
    "efficientnetbasedmodel4": EffieicientBasedModel4,
}


def build_model(config, name="model"):
    assert config.model_name.lower() in model_dict.keys(), f"Please, check pretrained model list"

