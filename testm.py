import os
import argparse
import re
from pathlib import Path
import glob
import json
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
    else:
        assert 1 <= try_num <= try_num_max
        try_dir = Path(model_dir) / f"{name}{int(try_num)}"

    return try_dir


def find_max_epoch(path):
    """
    Automatically find max epoch in {name} folder

    :param path (str or pathlib.Path): f"{model_dir}/{name}{}"
    :return: max number of epoch
    """
    path = Path(path)
    dirs = glob.glob(f"{path}/*")
    matches = [re.search(rf"[a-zA-Z0-9]*_(\d+).pth" ,d) for d in dirs]
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
    matches = [(d, re.search(rf"[a-zA-Z0-9]*_({epoch}).pth", d)) for d in dirs]
    for dir, m in matches:
        if not m: continue
        return dir

def test(test_dir, try_dir, epoch_dir, args):
    with open(Path(try_dir) / 'config.json', 'r') as f:
        json_data = json.load(f)

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_name = json_data['dataset'] + "_test"
    dataset_module = getattr(import_module("dataset"), dataset_name)
    dataset = dataset_module(
        data_dir=test_dir
    )
    num_classes = dataset.num_classes # 18

    # -- augmentation
    resize = json_data['resize']
    augmentation = json_data['augmentation']
    transform_module = getattr(import_module("dataset"), augmentation)
    transform = transform_module(
        resize=resize,
        mean=dataset.mean,
        std=dataset.std,
        test=True
    )
    dataset.set_transform(transform)

    # -- data_loader
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=use_cuda
    )

    # -- model
    model_name = json_data['model']
    model_module = getattr(import_module('model'), model_name)
    model = model_module(
        num_classes=num_classes,
        test=True
    ).to(device)
    model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(epoch_dir))
    model.eval()

    # -- load pandas dataframe
    eval_dir = args.eval_dir
    eval_df = pd.read_csv(Path(eval_dir) / 'info.csv')

    for idx, test_batch in enumerate(tqdm(test_loader)):
        inputs, file_paths = test_batch
        inputs = inputs.to(device)
        preds_mask, preds_gender, preds_age = model(inputs)
        _, preds_mask = torch.max(preds_mask, 1)
        _, preds_gender = torch.max(preds_gender, 1)
        _, preds_age = torch.max(preds_age, 1)
        for pred_mask,pred_gender,pred_age, file_path in zip(preds_mask,preds_gender,preds_age, file_paths):
            pred_mask = int(pred_mask.cpu().clone())
            pred_gender = int(pred_gender.cpu().clone())
            pred_age = int(pred_age.cpu().clone())
            file_name = file_path.split('/')[-1]
            person_idx = eval_df.loc[eval_df['ImageID'] == file_name].index
            pred = pred_mask * 6 + pred_gender * 3 + pred_age
            eval_df.loc[person_idx[0], 'ans'] = pred

    save_csv(eval_df,args.save_dir,epoch_dir.split('/')[-1].split('.pth')[0])

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
    parser.add_argument('--name', type=str, default='exp', help='model save at {SM_MODEL_DIR}/{name} (default: "exp")')
    parser.add_argument('--try_num', type=int, default=-1,
                        help='load in model/{name}{try_num} folder (default: -1 (lastest folder))')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size for testing (default: 32)')
    parser.add_argument('--epoch', type=int, default=-1,
                        help='load {epoch} trained model in model/exp{expint} pth. (default: -1 (lastest epoch))')

    # Container environment
    parser.add_argument('--test_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/eval/images'))
    parser.add_argument('--eval_dir', type=str, default='../input/data/eval', help='eval data folder (default : ../input/data/eval)')
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--save_dir', type=str, default='results', help="test result will be saved in {save_dir} folder (default: result)")

    args = parser.parse_args()
    print(args)

    name = args.name
    data_dir = args.test_dir
    model_dir = args.model_dir
    try_num = args.try_num
    epoch = args.epoch

    try_dir = find_dir_try(try_num, model_dir, name)

    epoch_dir = find_dir_epoch(epoch, try_dir)

    print(try_dir)
    print(epoch_dir)

    test(data_dir, try_dir, epoch_dir, args)