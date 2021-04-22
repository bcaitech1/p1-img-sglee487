import os
import argparse
import re
from pathlib import Path
import glob
from importlib import import_module


from tqdm import tqdm
import torch
import pandas as pd
from torch.utils.data import DataLoader

def find_dir_max_try(path):
    """
    Automatically find max try

    :param path (str or pathlib.Path): f"{model_dir}/{args.name}".
    :return: f"{path}{n}", max number of jtry
    """
    path = Path(path)
    dirs = glob.glob(f"{path}*")
    matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
    i = [int(m.groups()[0]) for m in matches if m]
    n = max(i) if i else 1
    return n


def find_dir_try(try_num, model_dir, name):
    try_num_max = find_dir_max_try(Path(model_dir) / name)
    if try_num == -1:
        try_dir = Path(model_dir) / f"{name}{int(try_num_max)}"
        return try_dir, try_num_max
    else:
        assert 1 <= try_num <= try_num_max
        try_dir = Path(model_dir) / f"{name}{int(try_num)}"
        return try_dir, try_num


def find_max_epoch(path):
    """
    Automatically find max epoch in {name} folder

    :param path (str or pathlib.Path): f"{model_dir}/{name}{}"
    :return: max number of epoch
    """
    path = Path(path)
    dirs = glob.glob(f"{path}/*")
    matches = [re.search(rf"[a-zA-Z0-9]*_(\d+).csv" ,d) for d in dirs]
    i = [int(m.groups()[0]) for m in matches if m]
    n = max(i) if i else 1
    return n


def find_dir_epoch(epoch, try_dir):
    """
    find path with model/{name}{}/[s]_{epoch}.pth

    :param path (str or pathlib.Path) : f"{model_dir}/{name}{}"
    :return: f"{path}{n}"
    """
    epoch_max = find_max_epoch(try_dir)
    if epoch == -1:
        epoch = epoch_max
    else:
        assert 0 <= epoch <= epoch_max
    path = Path(try_dir)
    dirs = glob.glob(f"{path}/*")
    matches = [(d, re.search(rf"[a-zA-Z0-9]*_({epoch}).csv", d)) for d in dirs]
    for dir, m in matches:
        if not m: continue
        return dir, epoch

def encode_multi_class(mask_label, gender_label, age_label):
    return mask_label * 6 + gender_label * 3 + age_label

def merge(save_dir, try_epochs, paths, args):

    # -- load pandas dataframe
    eval_df_mask = pd.read_csv(paths['epoch_dir_mask'])
    eval_df_gender = pd.read_csv(paths['epoch_dir_gender'])
    eval_df_age = pd.read_csv(paths['epoch_dir_age'])
    eval_df = pd.DataFrame.merge(eval_df_mask,eval_df_gender)
    eval_df = pd.DataFrame.merge(eval_df,eval_df_age)

    # age_cri1 = 31
    # age_cri2 = 59
    # map_label = lambda x: 0 if int(x) < age_cri1 else 1 if int(x) < age_cri2 else 2
    # eval_df['age'] = eval_df['age'].apply(map_label)
    eval_df['ans'] = eval_df['mask']*6 + eval_df['gender']*3 + eval_df['age']
    eval_df = eval_df.drop(['mask','gender','age'], axis='columns')
    print(eval_df)
    save_name = f"mask-{try_epochs['try_num_mask']}-{try_epochs['epoch_mask']}_" \
                f"gender-{try_epochs['try_num_gender']}-{try_epochs['epoch_gender']}_" \
                f"age-{try_epochs['try_num_age']}-{try_epochs['epoch_age']}"

    save_csv(eval_df, save_dir, save_name)

    return

def save_csv(df, save_dir,csv_name):
    save_file_name = os.path.join(save_dir,f"sub_{csv_name}.csv")
    print(f'Saved csv {save_file_name}')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir()

    df.to_csv(save_file_name, index=True)


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--name', type=str, default='try', help='model save at {SM_MODEL_DIR}/{name} (default: "exp")')

    parser.add_argument('--try_num_mask', type=int, default=-1,
                        help='load in model/{name}{try_num}/mask folder (default: -1 (lastest folder))')
    parser.add_argument('--epoch_mask', type=int, default=-1,
                        help='load {epoch} trained model in model/{name}{try_num}/mask/{epoch_mask}.pth. (default: -1 (lastest epoch))')
    parser.add_argument('--try_num_gender', type=int, default=-1,
                        help='load in model/{name}{try_num}/mask folder (default: -1 (lastest folder))')
    parser.add_argument('--epoch_gender', type=int, default=-1,
                        help='load {epoch} trained model in model/{name}{try_num}/mask/{epoch_gender}.pth. (default: -1 (lastest epoch))')
    parser.add_argument('--try_num_age', type=int, default=-1,
                        help='load in model/{name}{try_num}/mask folder (default: -1 (lastest folder))')
    parser.add_argument('--epoch_age', type=int, default=-1,
                        help='load {epoch} trained model in model/{name}{try_num}/mask/{epoch_age}.pth. (default: -1 (lastest epoch))')

    # Container environment
    parser.add_argument('--test_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/eval/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model_multi/custom'))
    parser.add_argument('--save_dir', type=str, default='./model_multi/custom', help="test result will be saved in {save_dir} folder (default: ./model_multi/custom)")

    args = parser.parse_args()
    print(args)

    name = args.name
    data_dir = args.test_dir
    model_dir = args.model_dir

    try_dir_mask, try_num_mask = find_dir_try(args.try_num_mask, model_dir, name)
    try_dir_mask = os.path.join(try_dir_mask,'mask')
    epoch_dir_mask, epoch_mask = find_dir_epoch(args.epoch_mask, try_dir_mask)

    try_dir_gender, try_num_gender = find_dir_try(args.try_num_gender, model_dir, name)
    try_dir_gender = os.path.join(try_dir_gender, 'gender')
    epoch_dir_gender, epoch_gender = find_dir_epoch(args.epoch_gender, try_dir_gender)

    try_dir_age, try_num_age = find_dir_try(args.try_num_age, model_dir, name)
    try_dir_age = os.path.join(try_dir_age, 'age')
    epoch_dir_age, epoch_age = find_dir_epoch(args.epoch_age, try_dir_age)

    save_dir = os.path.join(model_dir, 'results')

    try_epochs = {
        'try_num_mask': try_num_mask,
        'epoch_mask': epoch_mask,
        'try_num_gender': try_num_gender,
        'epoch_gender': epoch_gender,
        'try_num_age': try_num_age,
        'epoch_age': epoch_age,
    }

    paths = {
        'try_dir_mask': try_dir_mask,
        'epoch_dir_mask': epoch_dir_mask,
        'try_dir_gender': try_dir_gender,
        'epoch_dir_gender': epoch_dir_gender,
        'try_dir_age': try_dir_age,
        'epoch_dir_age': epoch_dir_age,
    }
    merge(save_dir, try_epochs, paths, args)