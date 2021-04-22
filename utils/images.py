from .file_namer2 import get_ext
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

num2class = ['incorrect_mask', 'mask1', 'mask2', 'mask3',
             'mask4', 'mask5', 'normal']
class2num = {k: v for v, k in enumerate(num2class)}

def plot_row_images(img_dir, img_id):
    """
    마스크 미착용 이미지를 시각화하는 함수입니다.

    Args:
        img_dir: image dataset folder path
        img_id: dataset folder name
    """
    ext = get_ext(img_dir, img_id)
    img = np.array(Image.open(os.path.join(img_dir, img_id, 'normal' + ext)))

    plt.figure(figsize=(6, 6))
    plt.imshow(img)


def plot_mask_images(img_dir, img_id):
    """
    마스크 정상착용 5장과 이상하게 착용한 1장을 2x3의 격자에 시각화하는 함수입니다.

    Args:
        img_dir: 학습 데이터셋 이미지 폴더 경로
        img_id: 학습 데이터셋 하위폴더 이름
    """
    ext = get_ext(img_dir, img_id)
    imgs = [np.array(Image.open(os.path.join(img_dir, img_id, class_name + ext))) for class_name in num2class[:-1]]

    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(15, 12))
    for i in range(n_rows * n_cols):
        axes[i // (n_rows + 1)][i % n_cols].imshow(imgs[i])
        axes[i // (n_rows + 1)][i % n_cols].set_title(f'{num2class[i]}', color='r')
    plt.tight_layout()