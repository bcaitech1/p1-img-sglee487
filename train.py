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
import wandb

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
    return
    # print('saved model {}'.format(save_path))
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    #
    # save_dir = Path(save_path)
    # if not save_dir.exists():
    #     save_dir.mkdir()
    #
    # tm = time.gmtime()
    # time_string = time.strftime('%yy%mm%dd%H_%M_%S', tm)
    # file_name = f'{name}_{time_string}_{loss:4.4f}_{acc:4.4}_{epoch}.pth'
    # file_path = save_dir / file_name
    # torch.save(model.state_dict(), file_path)


def train(data_dir, model_dir, args):
    seed_everything(args.seed)
    # args.__dict__ == vars(args)
    wandb.init(project="train_01", config=vars(args))

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
    model_module = getattr(import_module("model"), args.model) # default: BaseModel
    model = model_module(
        num_classes=num_classes,
        grad_point=args.grad_point
    ).to(device)
    model = torch.nn.DataParallel(model)
    # if want model train begin from args.continue_epoch checkpoint.
    if args.continue_train:
        try_dir = find_dir_try(args.continue_try_num, model_dir, args.continue_name)
        epoch_dir = find_dir_epoch(args.continue_epoch, try_dir)
        model.load_state_dict(torch.load(epoch_dir))


    # -- loss & metric
    if args.criterion == "cross_entropy":
        criterion = create_criterion(args.criterion) # default: cross_entropy
    else:
        criterion = create_criterion(args.criterion, classes = num_classes) # default: cross_entropy
    if args.optimizer == "AdamP":
        optimizer = AdamP(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.lr,
                          weight_decay=5e-4)
    else:
        opt_module = getattr(import_module('torch.optim'), args.optimizer) # default: Adam
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- logging
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(Path(save_dir) / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                wandb.log({"Train/loss":train_loss, "Train/accuracy": train_acc})

                loss_value = 0
                matches = 0

        scheduler.step()

        #val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                if figure is None:
                    # inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = torch.clone(inputs).detach().cpu()
                    inputs_np = inputs_np.permute(0,2,3,1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(inputs_np, labels, preds, args.dataset != "MaskSplitByProfileDataset")
                    plt.show()

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            if val_loss < best_val_loss or val_acc > best_val_acc:
                save_model(model,epoch,val_loss,val_acc,save_dir,args.model)
                if val_loss < best_val_loss and val_acc > best_val_acc:
                    print(f"New best model for val acc and val loss : {val_acc:4.2%} {val_loss:4.2}! saving the best model..")
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                elif val_loss < best_val_loss:
                    print(f"New best model for val loss : {val_loss:4.2}! saving the best model..")
                    save_model(model, epoch, val_loss, val_acc, save_dir, args.model)
                    best_val_loss = val_loss
                elif val_acc > best_val_acc:
                    print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                    save_model(model,epoch,val_loss,val_acc,save_dir,args.model)
                    best_val_acc = val_acc

            print(
                f"[Val] acc: {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc: {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            wandb.log({"Val/loss":val_loss, "Val/accuracy": val_acc})
            print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--name', type=str, default='exp', help='model save at {SM_MODEL_DIR}/{name} (default: "exp")')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset',
                        help='which dataset to use (default: MaskBaseDataset')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validation (default: 0.2)')
    parser.add_argument('--model', type=str, default='BaseModel',
                        help='model type (default: BaseModel) (BaseModel, Vgg19BasedModel, EfficientNet_b4)')
    parser.add_argument('--grad_point', type=int, default=12, help='start require_grad=True layer')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation',
                        help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument('--resize', nargs="+", type=list, default=[224,224], help='resize size for image when training (default: [224, 224]')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='loss function for training (default cross_entropy) (cross_entropy, focal, label_smoothing, f1)')
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
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
