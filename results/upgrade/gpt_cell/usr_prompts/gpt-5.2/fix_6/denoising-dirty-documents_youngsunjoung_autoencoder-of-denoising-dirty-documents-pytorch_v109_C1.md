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
try:
    from torchinfo import summary
except Exception as e:
    summary = None
    print("torchinfo not available; model summary will be skipped:", repr(e))



## === cell 2
import zipfile


def _safe_unzip(zip_path, out_dir):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(zip_path)
    os.makedirs(out_dir, exist_ok=True)
    if any(str(p).endswith(".png") for p in os.listdir(out_dir)):
        return
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


_safe_unzip(
    "/kaggle/input/denoising-dirty-documents/train.zip", "/content/denoising_data/train"
)
_safe_unzip(
    "/kaggle/input/denoising-dirty-documents/test.zip", "/content/denoising_data/test"
)
_safe_unzip(
    "/kaggle/input/denoising-dirty-documents/train_cleaned.zip",
    "/content/denoising_data/train_cleaned",
)



## === cell 3
train_dir = "/content/denoising_data/train"
train_cleaned_dir = "/content/denoising_data/train_cleaned"
test_dir = "/content/denoising_data/test"




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
train_images = load_images_from_folder(train_dir)
train_cleaned_images = load_images_from_folder(train_cleaned_dir)
test_images = load_images_from_folder(test_dir)



## === cell 6
print(f"train_images: {len(train_images)}")
print(f"train_cleaned_images: {len(train_cleaned_images)}")
print(f"test_images: {len(test_images)}")



## === cell 7
def load_images_from_folder(folder):
    images = []
    for root, _, files in os.walk(folder):
        for filename in files:
            if filename.lower().endswith(".png"):
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path)  # OpenCV로 이미지 읽기(BGR형식)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
                images.append(img)
    return images


## === cell 8
import zipfile


def _safe_unzip(zip_path, out_dir):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(zip_path)
    os.makedirs(out_dir, exist_ok=True)

    for root, _, files in os.walk(out_dir):
        if any(f.lower().endswith(".png") for f in files):
            return

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


_safe_unzip(
    "/kaggle/input/denoising-dirty-documents/train.zip", "/content/denoising_data/train"
)
_safe_unzip(
    "/kaggle/input/denoising-dirty-documents/test.zip", "/content/denoising_data/test"
)
_safe_unzip(
    "/kaggle/input/denoising-dirty-documents/train_cleaned.zip",
    "/content/denoising_data/train_cleaned",
)


## === cell 9
if not isinstance(test_images, list) or len(test_images) == 0:
    test_images = load_images_from_folder(test_dir)

if len(test_images) == 0:
    raise RuntimeError(
        f"No test images found under {test_dir}. Check unzip/extraction and directory structure."
    )

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
noisy_train_transforms = v2.Compose(
    [
        v2.ToImage(),
        PadToSize((420, 540)),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.4),
        v2.RandomApply([v2.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
        Grayscale(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

clean_target_transforms = v2.Compose(
    [
        v2.ToImage(),
        PadToSize((420, 540)),
        Grayscale(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

val_noisy_transforms = clean_target_transforms
val_clean_transforms = clean_target_transforms
test_transforms = clean_target_transforms




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
def _build_paired_filelists(noisy_dir, clean_dir):
    noisy = {
        os.path.splitext(f)[0]: os.path.join(noisy_dir, f)
        for f in os.listdir(noisy_dir)
        if f.endswith(".png")
    }
    clean = {
        os.path.splitext(f)[0]: os.path.join(clean_dir, f)
        for f in os.listdir(clean_dir)
        if f.endswith(".png")
    }
    keys = sorted(set(noisy).intersection(clean), key=lambda x: int(x))
    noisy_files = [noisy[k] for k in keys]
    clean_files = [clean[k] for k in keys]
    return noisy_files, clean_files, keys


train_files_all, cleaned_files_all, train_keys = _build_paired_filelists(
    train_dir, train_cleaned_dir
)
print("paired train count:", len(train_files_all), len(cleaned_files_all))

train_files, val_files, cleaned_train, cleaned_val = train_test_split(
    train_files_all, cleaned_files_all, test_size=2 / 9, random_state=42
)




## --- ERROR in cell 17, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/359019340.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     23[0m [0mprint[0m[0;34m([0m[0;34m"paired train count:"[0m[0;34m,[0m [0mlen[0m[0;34m([0m[0mtrain_files_all[0m[0;34m)[0m[0;34m,[0m [0mlen[0m[0;34m([0m[0mcleaned_files_all[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     24[0m [0;34m[0m[0m
[0;32m---> 25[0;31m train_files, val_files, cleaned_train, cleaned_val = train_test_split(
[0m[1;32m     26[0m     [0mtrain_files_all[0m[0;34m,[0m [0mcleaned_files_all[0m[0;34m,[0m [0mtest_size[0m[0;34m=[0m[0;36m2[0m [0;34m/[0m [0;36m9[0m[0;34m,[0m [0mrandom_state[0m[0;34m=[0m[0;36m42[0m[0;34m[0m[0;34m[0m[0m
[1;32m     27[0m )

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py[0m in [0;36mtrain_test_split[0;34m(test_size, train_size, random_state, shuffle, stratify, *arrays)[0m
[1;32m   2560[0m [0;34m[0m[0m
[1;32m   2561[0m     [0mn_samples[0m [0;34m=[0m [0m_num_samples[0m[0;34m([0m[0marrays[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2562[0;31m     n_train, n_test = _validate_shuffle_split(
[0m[1;32m   2563[0m         [0mn_samples[0m[0;34m,[0m [0mtest_size[0m[0;34m,[0m [0mtrain_size[0m[0;34m,[0m [0mdefault_test_size[0m[0;34m=[0m[0;36m0.25[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2564[0m     )

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py[0m in [0;36m_validate_shuffle_split[0;34m(n_samples, test_size, train_size, default_test_size)[0m
[1;32m   2234[0m [0;34m[0m[0m
[1;32m   2235[0m     [0;32mif[0m [0mn_train[0m [0;34m==[0m [0;36m0[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2236[0;31m         raise ValueError(
[0m[1;32m   2237[0m             [0;34m"With n_samples={}, test_size={} and train_size={}, the "[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2238[0m             [0;34m"resulting train set will be empty. Adjust any of the "[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: With n_samples=0, test_size=0.2222222222222222 and train_size=None, the resulting train set will be empty. Adjust any of the aforementioned parameters.

## === cell 18
class PairedImageDataset(Dataset):
    def __init__(
        self, train_files, cleaned_files, transform_in=None, transform_target=None
    ):
        self.train_files = train_files
        self.cleaned_files = cleaned_files
        self.transform_in = transform_in
        self.transform_target = transform_target

    def __len__(self):
        return len(self.train_files)

    def __getitem__(self, idx):
        train_img = Image.open(self.train_files[idx]).convert("RGB")
        cleaned_img = Image.open(self.cleaned_files[idx]).convert("RGB")

        if self.transform_in:
            train_img = self.transform_in(train_img)
        if self.transform_target:
            cleaned_img = self.transform_target(cleaned_img)

        return train_img, cleaned_img
