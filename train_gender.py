import argparse
import json
import glob
import os
import random
import re
import time
from pathlib import Path
from importlib import import_module

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from utils.images import plot_row_images, plot_mask_images
from loss import create_criterion
from test import find_dir_try, find_dir_epoch
from adamp import AdamP


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2)) # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8) # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

        plt.show() # for IDLE

    return figure


def increment_path(path, exist_ok=False):
    """
    Automatically increment path, i.e. model/exp1 --> model/exp2, model/exp3 etc.

    :param path (str or pathlib.Path): f"{model_dir}/{args.name}".
    :param exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        # return str(path)
        return f"{path}"
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 1
        return f"{path}{n}"


def save_model(model, epoch, loss, acc, save_path, name='MM'):
    print('saved model {}'.format(save_path))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_dir = Path(save_path)
    if not save_dir.exists():
        save_dir.mkdir()

    tm = time.gmtime()
    time_string = time.strftime('%yy%mm%dd%H_%M_%S', tm)
    file_name = f'{name}_{time_string}_{loss:4.4f}_{acc:4.4}_{epoch}.pth'
    file_path = save_dir / file_name
    torch.save(model.state_dict(), file_path)


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset) # MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
        val_ratio=args.val_ratio
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True
    )

    # -- model
    models = []
    model_module_gender = getattr(import_module("model"), args.model_gender)  # default: BaseModel
    model_gender = model_module_gender(
        num_classes=args.num_classes_gender,
        grad_point=args.grad_point
    ).to(device)
    model_gender = torch.nn.DataParallel(model_gender)

    # -- loss & metric

    criterion_gender = create_criterion(args.criterion_gender, classes=args.num_classes_gender)  # default: f1
    if args.optimizer == "AdamP":
        optimizer_gender = AdamP(filter(lambda p: p.requires_grad, model_gender.parameters()),
                          lr=args.lr,
                          weight_decay=5e-4)
    else:
        opt_module = getattr(import_module('torch.optim'), args.optimizer)  # default: Adam
        optimizer_gender = opt_module(
            filter(lambda p: p.requires_grad, model_gender.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        )
    scheduler_gender = StepLR(optimizer_gender, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger_gender = SummaryWriter(log_dir=os.path.join(save_dir,'gender'))
    with open(Path(save_dir) / 'gender' / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc_gender = 0
    best_val_loss_gender = np.inf
    for epoch in range(args.epochs):
        # train loop
        model_gender.train()
        loss_value_gender = 0
        matches_gender = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels_mask, labels_gender, labels_age = train_batch
            inputs = inputs.to(device)
            labels_gender = labels_gender.to(device)

            optimizer_gender.zero_grad()

            outs_gender = model_gender(inputs)
            preds_gender = torch.argmax(outs_gender, dim=-1)
            loss_gender = criterion_gender(outs_gender, labels_gender)

            loss_gender.backward()
            optimizer_gender.step()

            loss_value_gender += loss_gender.item()
            matches_gender += (preds_gender == labels_gender).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss_gender = loss_value_gender / args.log_interval
                train_acc_gender = matches_gender / args.batch_size / args.log_interval
                current_lr_gender = get_lr(optimizer_gender)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss_gender:4.4} || training accuracy {train_acc_gender:4.2%} || lr {current_lr_gender}"
                )
                logger_gender.add_scalar("Train/loss", train_loss_gender, epoch * len(train_loader) + idx)
                logger_gender.add_scalar("Train/accuracy", train_acc_gender, epoch * len(train_loader) + idx)

                loss_value_gender = 0
                matches_gender = 0

        scheduler_gender.step()

        #val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model_gender.eval()
            val_loss_items_gender = []
            val_acc_items_gender = []
            figure = None
            for val_batch in val_loader:
                inputs, labels_mask, labels_gender, labels_age = val_batch
                inputs = inputs.to(device)
                labels_gender = labels_gender.to(device)

                outs_gender = model_gender(inputs)
                preds_gender = torch.argmax(outs_gender, dim=-1)

                loss_item_gender = criterion_gender(outs_gender, labels_gender).item()
                acc_item_gender = (labels_gender == preds_gender).sum().item()
                val_loss_items_gender.append(loss_item_gender)
                val_acc_items_gender.append(acc_item_gender)

                if figure is None:
                    # inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = torch.clone(inputs).detach().cpu()
                    inputs_np = inputs_np.permute(0,2,3,1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(inputs_np, labels_mask, preds_gender, args.dataset != "MaskSplitByProfileDataset")
                    plt.show()

            val_loss_gender = np.sum(val_loss_items_gender) / len(val_loader)
            val_acc_gender = np.sum(val_acc_items_gender) / len(val_set)
            if val_loss_gender < best_val_loss_gender or val_acc_gender > best_val_acc_gender:
                save_model(model_gender, epoch, val_loss_gender, val_acc_gender, os.path.join(save_dir, "gender"),
                           args.model_gender)
                if val_loss_gender < best_val_loss_gender and val_acc_gender > best_val_acc_gender:
                    print(
                        f"New best model_gender for val acc and val loss : {val_acc_gender:4.2%} {val_loss_gender:4.2}! saving the best model_gender..")
                    best_val_loss_gender = val_loss_gender
                    best_val_acc_gender = val_acc_gender
                elif val_loss_gender < best_val_loss_gender:
                    print(f"New best model_gender for val loss : {val_loss_gender:4.2}! saving the best model_gender..")
                    best_val_loss_gender = val_loss_gender
                elif val_acc_gender > best_val_acc_gender:
                    print(f"New best model_gender for val accuracy : {val_acc_gender:4.2%}! saving the best model_gender..")
                    best_val_acc_gender = val_acc_gender

            print(
                f"[Val] acc: {val_acc_gender:4.2%}, loss: {val_loss_gender:4.2} || "
                f"best acc: {best_val_acc_gender:4.2%}, best loss: {best_val_loss_gender:4.2}"
            )
            logger_gender.add_scalar("Val/loss", val_loss_gender, epoch)
            logger_gender.add_scalar("Val/accuracy", val_acc_gender, epoch)
            logger_gender.add_figure("results", figure, epoch)
            print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--name', type=str, default='exp', help='model save at {SM_MODEL_DIR}/{name} (default: "exp")')
    parser.add_argument('--dataset', type=str, default='MaskMultiLabelDataset',
                        help='which dataset to use (default: MaskMultiLabelDataset')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='ratio for validation (default: 0.1)')
    parser.add_argument('--num_classes_gender', type=int, default=2, help='model_gender output classes (default: 2)')
    parser.add_argument('--model_gender', type=str, default='BaseModel',
                        help='gender model type (default: BaseModel) (BaseModel, Vgg19BasedModel, EfficientNet_b4)')
    parser.add_argument('--criterion_gender', type=str, default='f1', help='gender loss function for training (default f1) (cross_entropy, focal, label_smoothing, f1)')
    parser.add_argument('--grad_point', type=int, default=12, help='start require_grad=True layer')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation',
                        help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument('--resize', nargs="+", type=list, default=[224,224], help='resize size for image when training (default: [224, 224]')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training (default: Adam) (SGD, Adam)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for training (default: 1e-3)')
    parser.add_argument('--lr_decay_step', type=int, default=18, help='learning rate scheduler decay step (default: 18)')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train (default: 2)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')

    # if continue model train in load saved model
    parser.add_argument('--continue_train', type=bool, default=False, help="you can train continue by load saved checkpoint")
    parser.add_argument('--continue_try_num', type=int, default=-1, help='load in model/{name}{try_num} folder (default: -1 (lastest folder))')
    parser.add_argument('--continue_epoch', type=int, default=-1, help='load {epoch} trained model in model/exp{expint} pth. (default: -1 (lastest epoch))')
    parser.add_argument('--continue_name', type=str, default='exp', help='model save at {SM_MODEL_DIR}/{name} (default: "exp")')

    # Container environment
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model_multi'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
