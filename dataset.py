import os
from pathlib import Path
from collections import defaultdict
import random

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.transforms import *
import albumentations as A
# from albumentations.pytorch import ToTensor

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class BaseAugmentation:
    def __init__(self, resize, mean, std, test: bool = False, **args):
        if test:
            self.transform = transforms.Compose([
                Resize(resize, Image.BILINEAR),
                ToTensor(),
                Normalize(mean=mean, std=std)
            ])
        else:
            self.transform = transforms.Compose([
                Resize(resize, Image.BILINEAR),
                ToTensor(),
                Normalize(mean=mean, std=std)
            ])

    def __call__(self, image):
        return self.transform(image)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            CenterCrop((320, 256)),
            Resize(resize, Image.BILINEAR),
            ColorJitter(0.1, 0.1, 0.1, 0.1),
            ToTensor(),
            Normalize(mean=mean, std=std),
            AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)

class CustomAugmentation2:
    def __init__(self, resize, mean, std, test: bool = False, **args):
        if test:
            self.transform = A.Compose([
                A.CenterCrop(355,355),
                A.Resize(260,260),
                A.RandomCrop(resize[0],resize[1]),
                A.HorizontalFlip(),
                A.ColorJitter(),
                A.Rotate((-10,10)),
                A.Normalize(mean=mean, std=std),
                ToTensor()
            ])
        else:
            self.transform = A.Compose([
                A.CenterCrop(355, 355),
                A.Resize(260, 260),
                A.CenterCrop(resize[0], resize[1]),
                A.Normalize(mean=mean, std=std),
                ToTensor()
            ])

    def __call__(self, image):
        return self.transform(image=np.array(image))['image']


class MaskBaseDataset(data.Dataset):
    num_classes = 3 * 2 * 3

    class MaskLabels:
        mask = 0
        incorrect = 1
        normal = 2

    class GenderLabels:
        male = 0
        female = 1

    class AgeGroup:
        map_label = lambda x: 0 if int(x) < 30 else 1 if int(x) < 60 else 2

    _file_names = {
        "mask1": MaskLabels.mask,
        "mask2": MaskLabels.mask,
        "mask3": MaskLabels.mask,
        "mask4": MaskLabels.mask,
        "mask5": MaskLabels.mask,
        "incorrect_mask": MaskLabels.incorrect,
        "normal": MaskLabels.normal
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith('.'):  # "." 로 시작하는 파일 무시
                continue

            img_folder = Path(self.data_dir) / profile
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:
                    continue

                img_path = Path(self.data_dir) / profile / file_name
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = getattr(self.GenderLabels, gender)
                age_label = self.AgeGroup.map_label(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print(
                "[Warning] Calculating statistics... It can takes huge amounts of time depending on your CPU machine :(")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def read_image(self, index):
        return Image.open(self.image_paths[index])

    def get_mask_label(self, index):
        return self.mask_labels[index]

    def get_gender_label(self, index):
        return self.gender_labels[index]

    def get_age_label(self, index):
        return self.age_labels[index]

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label):
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label):
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self):
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = torch.utils.data.random_split(self, [n_train, n_val])
        return train_set, val_set


class MaskBaseDataset_test(data.Dataset):
    num_classes = 3 * 2 * 3

    image_paths = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std

        self.transform = None
        self.setup()

    def setup(self):
        image_files = os.listdir(self.data_dir)
        for image_file in image_files:
            if image_file.startswith('.'):  # "." 로 시작하는 파일 무시
                continue

            self.image_paths.append(os.path.join(self.data_dir, image_file))

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        image = self.read_image(index)

        image_transform = self.transform(image)
        return image_transform, self.image_paths[index]

    def __len__(self):
        return len(self.image_paths)

    def read_image(self, index):
        return Image.open(self.image_paths[index])

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label):
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label):
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val 나누는 기준을 이미지에 대해서 random 이 아닌
        사람(profile)을 기준으로 나눕니다.
        구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
        이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """
    class AgeGroup:
        map_label = lambda x: 0 if int(x) < 31 else 1 if int(x) < 59 else 2

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.choices(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile,
                                            file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = getattr(self.GenderLabels, gender)
                    age_label = self.AgeGroup.map_label(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self):
        return [Subset(self, indices) for phase, indices in self.indices.items()]

class MaskSplitByProfileDataset_test(data.Dataset):
    num_classes = 3 * 2 * 3

    image_paths = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std

        self.transform = None
        self.setup()

    def setup(self):
        image_files = os.listdir(self.data_dir)
        for image_file in image_files:
            if image_file.startswith('.'):  # "." 로 시작하는 파일 무시
                continue

            self.image_paths.append(os.path.join(self.data_dir, image_file))

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        image = self.read_image(index)

        image_transform = self.transform(image)
        return image_transform, self.image_paths[index]

    def __len__(self):
        return len(self.image_paths)

    def read_image(self, index):
        return Image.open(self.image_paths[index])


class MaskMultiLabelDataset(MaskSplitByProfileDataset):
    """
    기존에 label을 0~17로 classes 18로 하지 말고 mask, gender, age 3개의 label로 나누기.

    """

    class AgeGroup:
        map_label = lambda x: 0 if int(x) < 31 else 1 if int(x) < 59 else 2


    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        super().__init__(data_dir,mean,std,val_ratio)

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile,
                                            file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = getattr(self.GenderLabels, gender)
                    age_label = self.AgeGroup.map_label(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def __getitem__(self, index):
        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)

        image_transform = self.transform(image=image)
        return image_transform, mask_label, gender_label, age_label

class MaskMultiLabelDataset_test(MaskSplitByProfileDataset):
    """
    기존에 label을 0~17로 classes 18로 하지 말고 mask, gender, age 3개의 label로 나누기.

    age는 3개의 label이 아니라 그냥 60개의 one_hot vector classes로.
    """

    class AgeGroup:
        map_label = lambda x: 0 if int(x) < 30 else 1 if int(x) < 58 else 2


    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        super().__init__(data_dir,mean,std,val_ratio)

    def setup(self):
        image_files = os.listdir(self.data_dir)
        for image_file in image_files:
            if image_file.startswith('.'):  # "." 로 시작하는 파일 무시
                continue

            self.image_paths.append(os.path.join(self.data_dir, image_file))

    def __getitem__(self, index):
        image = self.read_image(index)

        image_transform = self.transform(image)
        return image_transform, self.image_paths[index]