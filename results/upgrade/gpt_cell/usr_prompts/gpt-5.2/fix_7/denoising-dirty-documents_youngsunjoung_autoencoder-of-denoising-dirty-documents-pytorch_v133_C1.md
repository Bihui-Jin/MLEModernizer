# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.13

# 2. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        input/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        working/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
```

-> data/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> data/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> working/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

# 4. Code solution

## === cell 0
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import v2
import math
import torchvision.transforms.functional as TF
import torch.nn.functional as F



## === cell 1
from torchinfo import summary



## === cell 2
import zipfile


def unzip_if_needed(zip_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(out_dir) and any(
        root.endswith(("train", "test", "train_cleaned"))
        for root, _, _ in os.walk(out_dir)
    ):
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


unzip_root = "/kaggle/working/denoising_data"
unzip_if_needed("/kaggle/input/denoising-dirty-documents/train.zip", unzip_root)
unzip_if_needed("/kaggle/input/denoising-dirty-documents/test.zip", unzip_root)
unzip_if_needed("/kaggle/input/denoising-dirty-documents/train_cleaned.zip", unzip_root)



## === cell 3
train_dir = "/kaggle/working/denoising_data/train"
train_cleaned_dir = "/kaggle/working/denoising_data/train_cleaned"
test_dir = "/kaggle/working/denoising_data/test"




## === cell 4
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):  # PNG파일 로드
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)  # OpenCV로 이미지 읽기(BGR형식)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
            images.append(img)
    return images




## === cell 5
def load_images_from_folder(folder):
    if not os.path.isdir(folder):
        candidates = []

        name = os.path.basename(folder)

        base = os.path.dirname(folder)
        candidates.append(os.path.join(base, "denoising-dirty-documents", name))
        candidates.append(os.path.join(folder, name))

        try:
            for entry in os.listdir(base):
                candidates.append(os.path.join(base, entry, name))
        except FileNotFoundError:
            pass

        candidates.append(
            os.path.join("/kaggle/working/denoising-dirty-documents", name)
        )
        candidates.append(os.path.join("/kaggle/input/denoising-dirty-documents", name))

        for walk_base in (base, "/kaggle/working", "/kaggle/input"):
            try:
                for root, dirs, _ in os.walk(walk_base):
                    if name in dirs:
                        candidates.append(os.path.join(root, name))
            except FileNotFoundError:
                pass

        resolved = next((p for p in candidates if os.path.isdir(p)), None)
        if resolved is None:
            raise FileNotFoundError(
                f"Could not find image directory. Tried '{folder}' and candidates: {candidates}"
            )
        folder = resolved

    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):  # PNG파일 로드
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)  # OpenCV로 이미지 읽기(BGR형식)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
            images.append(img)
    return images


train_images = load_images_from_folder(train_dir)
train_cleaned_images = load_images_from_folder(train_cleaned_dir)
test_images = load_images_from_folder(test_dir)


## === cell 6
print(f"train_images: {len(train_images)}")
print(f"train_cleaned_images: {len(train_cleaned_images)}")
print(f"test_images: {len(test_images)}")



## === cell 7
train_images[0]



## === cell 8
train_cleaned_images[0]



## === cell 9
test_images[0]



## === cell 10
train_sizes = [img.shape[:2] for img in train_images]
train_cleaned_sizes = [img.shape[:2] for img in train_cleaned_images]
test_sizes = [img.shape[:2] for img in test_images]



## === cell 11
unique_train_sizes = np.unique(train_sizes, axis=0)
unique_train_cleaned_sizes = np.unique(train_cleaned_sizes, axis=0)
unique_test_sizes = np.unique(test_sizes, axis=0)



## === cell 12
print(f"train_images:\n {unique_train_sizes}")
print(f"train_cleaned_images:\n {unique_train_cleaned_sizes}")
print(f"test_images:\n {unique_test_sizes}")




## === cell 13
class PadToSize:
    def __init__(self, target_size):
        self.target_size = target_size  # (높이, 너비)

    def __call__(self, img):
        _, height, width = img.shape
        target_height, target_width = self.target_size

        pad_top = (target_height - height) // 2
        pad_bottom = target_height - height - pad_top
        pad_left = (target_width - width) // 2
        pad_right = target_width - width - pad_left

        return TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=0)




## === cell 14
class Grayscale:
    def __call__(self, img):
        pil_img = TF.to_pil_image(img) if isinstance(img, torch.Tensor) else img
        grayscale_img = pil_img.convert("L")  # RGB -> Grayscale
        return TF.to_tensor(grayscale_img)




## === cell 15
train_transforms = v2.Compose(
    [
        v2.ToImage(),
        PadToSize((420, 540)),  # 크기 통일
        v2.RandomApply(
            [v2.GaussianBlur(kernel_size=3)], p=0.4
        ),  # 가우시안 블러 (40% 확률 적용)
        v2.RandomApply(
            [v2.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5
        ),  # 밝기 & 대비 조절 (50% 확률 적용)
        Grayscale(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

val_transforms = v2.Compose(
    [
        v2.ToImage(),
        PadToSize((420, 540)),  # 크기 통일
        Grayscale(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

test_transforms = v2.Compose(
    [
        v2.ToImage(),
        PadToSize((420, 540)),  # 크기 통일
        Grayscale(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)




## === cell 16
class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = sorted(
            [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".png")
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")  # RGB로 열기
        if self.transform:
            image = self.transform(image)
        return image




## === cell 17
def _resolve_dir(expected_dir):
    if os.path.isdir(expected_dir):
        return expected_dir

    name = os.path.basename(expected_dir)
    base = os.path.dirname(expected_dir)

    candidates = [
        os.path.join(base, "denoising-dirty-documents", name),
        os.path.join(expected_dir, name),
        os.path.join("/kaggle/working/denoising-dirty-documents", name),
        os.path.join("/kaggle/input/denoising-dirty-documents", name),
    ]

    try:
        for entry in os.listdir(base):
            candidates.append(os.path.join(base, entry, name))
    except FileNotFoundError:
        pass

    for walk_base in (base, "/kaggle/working", "/kaggle/input"):
        try:
            for root, dirs, _ in os.walk(walk_base):
                if name in dirs:
                    candidates.append(os.path.join(root, name))
        except FileNotFoundError:
            pass

    resolved = next((p for p in candidates if os.path.isdir(p)), None)
    if resolved is None:
        raise FileNotFoundError(
            f"Could not find image directory. Tried '{expected_dir}' and candidates: {candidates}"
        )
    return resolved


def _id_from_path(p):
    return int(os.path.splitext(os.path.basename(p))[0])


train_dir = _resolve_dir(train_dir)
train_cleaned_dir = _resolve_dir(train_cleaned_dir)

train_files_all = [
    os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".png")
]
cleaned_files_all = [
    os.path.join(train_cleaned_dir, f)
    for f in os.listdir(train_cleaned_dir)
    if f.endswith(".png")
]

train_map = {_id_from_path(p): p for p in train_files_all}
cleaned_map = {_id_from_path(p): p for p in cleaned_files_all}

common_ids = sorted(set(train_map.keys()) & set(cleaned_map.keys()))
paired_train_files = [train_map[i] for i in common_ids]
paired_cleaned_files = [cleaned_map[i] for i in common_ids]

train_files, val_files, cleaned_train, cleaned_val = train_test_split(
    paired_train_files, paired_cleaned_files, test_size=2 / 9, random_state=42
)


## === cell 18
class PairedImageDataset(Dataset):
    def __init__(self, train_files, cleaned_files, transform=None):
        self.train_files = train_files
        self.cleaned_files = cleaned_files
        self.transform = transform

    def __len__(self):
        return len(self.train_files)

    def __getitem__(self, idx):
        train_img = Image.open(self.train_files[idx]).convert("RGB")
        cleaned_img = Image.open(self.cleaned_files[idx]).convert("RGB")

        if self.transform:
            train_img = self.transform(train_img)
            cleaned_img = self.transform(cleaned_img)

        return train_img, cleaned_img




## === cell 19
train_dataset = PairedImageDataset(train_files, cleaned_train, train_transforms)
val_dataset = PairedImageDataset(val_files, cleaned_val, val_transforms)
test_dataset = ImageDataset(test_dir, test_transforms)

train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True
)  # 훈련 데이터 셔플 O
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)  # 검증 데이터 셔플 X
test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False
)  # 테스트 데이터 셔플 X




## --- ERROR in cell 19, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/366607308.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0mtrain_dataset[0m [0;34m=[0m [0mPairedImageDataset[0m[0;34m([0m[0mtrain_files[0m[0;34m,[0m [0mcleaned_train[0m[0;34m,[0m [0mtrain_transforms[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0mval_dataset[0m [0;34m=[0m [0mPairedImageDataset[0m[0;34m([0m[0mval_files[0m[0;34m,[0m [0mcleaned_val[0m[0;34m,[0m [0mval_transforms[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 3[0;31m [0mtest_dataset[0m [0;34m=[0m [0mImageDataset[0m[0;34m([0m[0mtest_dir[0m[0;34m,[0m [0mtest_transforms[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0;34m[0m[0m
[1;32m      5[0m train_loader = DataLoader(

[0;32m/tmp/ipykernel_11/3077604300.py[0m in [0;36m__init__[0;34m(self, data_dir, transform)[0m
[1;32m      6[0m             [
[1;32m      7[0m                 [0mos[0m[0;34m.[0m[0mpath[0m[0;34m.[0m[0mjoin[0m[0;34m([0m[0mdata_dir[0m[0;34m,[0m [0mf[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 8[0;31m                 [0;32mfor[0m [0mf[0m [0;32min[0m [0mos[0m[0;34m.[0m[0mlistdir[0m[0;34m([0m[0mdata_dir[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m                 [0;32mif[0m [0mf[0m[0;34m.[0m[0mendswith[0m[0;34m([0m[0;34m".png"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m             ]

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: '/kaggle/working/denoising_data/test'

## === cell 20
def visualize_paired_dataset(paired_loader, num_images=5):
    for train_images, cleaned_images in paired_loader:
        fig, axes = plt.subplots(
            num_images, 2, figsize=(8, num_images * 3)
        )  # num_images 행, 2열 (위: train, 아래: cleaned)
        for i in range(num_images):
            axes[i, 0].imshow(
                train_images[i].permute(1, 2, 0).cpu().numpy(), cmap="gray"
            )  # [C, H, W] -> [H, W, C]
            axes[i, 0].set_title(f"Original {i+1}")
            axes[i, 0].axis("off")  # 축 비활성화

            axes[i, 1].imshow(
                cleaned_images[i].permute(1, 2, 0).cpu().numpy(), cmap="gray"
            )  # [C, H, W] -> [H, W, C]
            axes[i, 1].set_title(f"Cleaned {i+1}")
            axes[i, 1].axis("off")  # 축 비활성화

        plt.tight_layout()
        plt.show()
        break  # 한 번만 실행하도록 break


visualize_paired_dataset(train_loader, num_images=3)
visualize_paired_dataset(val_loader, num_images=2)
