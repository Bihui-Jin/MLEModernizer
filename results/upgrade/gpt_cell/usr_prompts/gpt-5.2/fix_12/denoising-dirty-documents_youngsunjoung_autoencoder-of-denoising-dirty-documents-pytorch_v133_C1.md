# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Given a dataset of images of scanned text that is noisy, remove the noise.

## Metric
Root mean squared error between the cleaned pixel intensities and the actual grayscale pixel intensities.

## Submission Format
Form the submission file by melting each images into a set of pixels, assigning each pixel an id of image_row_col (e.g. 1_2_1 is image 1, row 2, column 1). Intensity values range from 0 (black) to 1 (white). The file should contain a header and have the following format:

```
id,value
1_1_1,1
1_2_1,1
1_3_1,1
etc.
```

## Dataset
You are provided two sets of images, train and test. These images contain various styles of text, to which synthetic noise has been added to simulate real-world, messy artifacts. The training set includes the test without the noise (train_cleaned).

# 2. Python version

3.13

# 3. Installed packages

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

# 4. Data file paths

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

# 5. Target score

0.37284

# 6. Current score

0.32292

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.41103) has done: 'Diagnosis: `test_file_paths` is built from `test_dir`, but in this environment the earlier unzip paths (`/content/denoising_data/...`) don’t exist as expected, so `os.listdir(test_dir)` yields fewer than 72 PNGs (possibly zero). The loop hard-codes `range(72)`, causing `test_file_paths[i]` to raise `IndexError` when `i` exceeds the available files. The fix is to iterate over the actual number of discovered test files, and (to preserve expected behavior) also guard that `all_outputs` has at least that many predictions.

Patch summary: In cell 32 only, compute `n_files = min(len(test_file_paths), all_outputs.shape[0])` and loop over `range(n_files)` instead of `range(72)`. This removes the crash while keeping the same submission-row generation logic and output format for whatever test set is present.

Updated cells: cell 32 below.

Compatibility notes for cell k+1: No variables or interfaces are changed (same `submission.csv` writing, same `submission_data` structure); only the loop bounds are made safe, so downstream cells (if any) still find `submission.csv` created.

Assumptions: The intended test set size is 72 images, but the current runtime may expose a different count due to extraction/path issues; this patch avoids crashing and uses whatever images and predictions are actually available.'
- What this solution (achieved 0.309) has done: 'Diagnosis: Cell 32 hard-codes `test_dir = "/kaggle/working/denoising_data/test"` and immediately calls `os.listdir(test_dir)`, but that directory does not exist in this environment (the extracted test folder may be under a different resolved path, and earlier cells already computed a working `test_dir` via `_resolve_dir`). This causes the `FileNotFoundError` at `os.listdir(test_dir)`. The minimal fix is to defensively resolve `test_dir` inside cell 32 (using the already-defined `_resolve_dir` when available, otherwise searching common Kaggle locations), and only then list files. This keeps all subsequent submission generation logic identical and does not change model outputs.

Patch summary: Update cell 32 to (1) resolve `test_dir` to an existing directory before calling `os.listdir`, and (2) raise a clearer error if it still cannot be found. No changes to model, transforms, inference, or CSV formatting.

Updated cells: Cell 32 only.

Compatibility notes for cell k+1: No interface/variable changes are introduced; `submission.csv` is still written to the same path with the same schema (`id,value`). The patch only ensures `test_dir` points to a real folder.

Assumptions: `_resolve_dir` from cell 17 may exist in the runtime; if it does not, the fallback resolver in cell 32 locate the test directory under `/kaggle/working` or `/kaggle/input` based on the provided filesystem layout.'
- What this solution (achieved 0.30761) has done: 'Your current score (0.309, lower-is-better) is better than the target (0.37284), so we should *slightly worsen* performance to move closer to the target band while keeping the same architecture/training/inference semantics. The smallest safe lever is to change only the submission-time post-processing by adding a tiny amount of smoothing (a light blur) to predictions before writing pixels; this typically increases RMSE a bit without breaking format. I also keep your existing robust test directory resolution and ensure submission rows are generated for all discovered test images (no hard-coded counts). Everything else (model, training loop, loss, transforms, I/O paths) remains unchanged.'
- What this solution (achieved 0.42647) has done: 'Your current score (0.30761, lower-is-better) is already better than the target (0.37284), so we should make a very small, submission-time-only change that slightly worsens RMSE while keeping the model/training/inference unchanged. The safest minimal lever is to increase the existing post-processing blur strength a bit (still mild), which typically removes detail and raises RMSE. I also keep the robust test directory resolution and the safe `n_files = min(...)` guard exactly as-is to ensure a valid submission is always written. No changes are made to architecture, losses, training loop, or dataloaders.'
- What this solution (achieved 0.32292) has done: 'Your current score (0.42647, lower-is-better) is worse than the target (0.37284), so we should make a minimal change that is likely to improve RMSE without touching the model, training loop, losses, or data splits. The safest lever is to remove the submission-time Gaussian blur that was intentionally degrading predictions; this should recover detail and reduce RMSE. I keep the robust test directory resolution and the `n_files = min(...)` guard exactly as-is to ensure the submission is always valid and aligned. No other logic is changed.'

# 9. Code solution

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
test_dir = _resolve_dir(test_dir)

train_dataset = PairedImageDataset(train_files, cleaned_train, train_transforms)
val_dataset = PairedImageDataset(val_files, cleaned_val, val_transforms)
test_dataset = ImageDataset(test_dir, test_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)




## === cell 20
def visualize_paired_dataset(paired_loader, num_images=5):
    for train_images, cleaned_images in paired_loader:
        fig, axes = plt.subplots(num_images, 2, figsize=(8, num_images * 3))
        for i in range(num_images):
            axes[i, 0].imshow(
                train_images[i].permute(1, 2, 0).cpu().numpy(), cmap="gray"
            )
            axes[i, 0].set_title(f"Original {i+1}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(
                cleaned_images[i].permute(1, 2, 0).cpu().numpy(), cmap="gray"
            )
            axes[i, 1].set_title(f"Cleaned {i+1}")
            axes[i, 1].axis("off")

        plt.tight_layout()
        plt.show()
        break


visualize_paired_dataset(train_loader, num_images=3)
visualize_paired_dataset(val_loader, num_images=2)



## === cell 21
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




## === cell 22
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.Conv2d(1, 8, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.Conv2d(8, 16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.Conv2d(16, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.enc5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=1, stride=1, bias=False),
            nn.ConvTranspose2d(
                64,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                groups=64,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )

        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=1, stride=1, bias=False),
            nn.ConvTranspose2d(
                32,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                groups=32,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=1, stride=1, bias=False),
            nn.ConvTranspose2d(
                16,
                16,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                groups=16,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=1, stride=1, bias=False),
            nn.ConvTranspose2d(
                8,
                8,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                groups=8,
                bias=False,
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(8, 1, kernel_size=1, stride=1, bias=False),
            nn.ConvTranspose2d(
                1,
                1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                groups=1,
                bias=False,
            ),
            nn.Sigmoid(),
        )

        self.conv4 = nn.Conv2d(64, 32, kernel_size=1, stride=1, bias=False)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        enc4_out = self.enc4(enc3_out)
        enc5_out = self.enc5(enc4_out)

        dec5_out = self.dec5(enc5_out)
        dec5_out = F.interpolate(
            dec5_out, size=(26, 33), mode="bilinear", align_corners=False
        )
        dec5_out = torch.cat([dec5_out, enc4_out], dim=1)
        dec5_out = self.conv5(dec5_out)

        dec4_out = self.dec4(enc4_out)
        dec4_out = F.interpolate(
            dec4_out, size=(52, 67), mode="bilinear", align_corners=False
        )
        dec4_out = torch.cat([dec4_out, enc3_out], dim=1)
        dec4_out = self.conv4(dec4_out)

        dec3_out = self.dec3(dec4_out)
        dec3_out = F.interpolate(
            dec3_out, size=(105, 135), mode="bilinear", align_corners=False
        )

        dec2_out = self.dec2(dec3_out)
        dec2_out = F.interpolate(
            dec2_out, size=(210, 270), mode="bilinear", align_corners=False
        )

        dec1_out = self.dec1(dec2_out)
        dec1_out = F.interpolate(
            dec1_out, size=(420, 540), mode="bilinear", align_corners=False
        )

        outputs = torch.clamp(dec1_out, min=0.001, max=0.999)
        return outputs


model = DenoisingAutoencoder().to(device)



## === cell 23
summary(model, input_size=(16, 1, 420, 540), device=device)




## === cell 24
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target))




## === cell 25
class HybridLoss(nn.Module):
    def __init__(self, lambda_rmse=0.8, lambda_l1=0.2):
        super(HybridLoss, self).__init__()
        self.lambda_rmse = lambda_rmse
        self.lambda_l1 = lambda_l1
        self.rmse_loss = RMSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.lambda_rmse * self.rmse_loss(
            pred, target
        ) + self.lambda_l1 * self.l1_loss(pred, target)




## === cell 26
criterion = HybridLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)



## === cell 27
epochs = 1000
patience = 5
early_stop_counter = 0
best_val_loss = float("inf")
best_model_state = None

epoch = 0
while epoch < epochs:
    model.train()
    train_loss = 0.0
    train_rmse_loss = 0.0
    for train_images, train_cleaned_images in train_loader:
        train_images, train_cleaned_images = train_images.to(
            device
        ), train_cleaned_images.to(device)
        optimizer.zero_grad()
        outputs = model(train_images)
        loss = criterion(outputs, train_cleaned_images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_rmse_loss += RMSELoss()(outputs, train_cleaned_images).item()

    train_loss /= len(train_loader)
    train_rmse_loss /= len(train_loader)
    print(
        f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, RMSE Score : {train_rmse_loss:.4f}"
    )

    model.eval()
    val_loss = 0.0
    val_rmse_loss = 0.0
    with torch.no_grad():
        for val_images, val_cleaned_images in val_loader:
            val_images, val_cleaned_images = val_images.to(
                device
            ), val_cleaned_images.to(device)
            outputs = model(val_images)
            loss = criterion(outputs, val_cleaned_images)
            val_loss += loss.item()
            val_rmse_loss += RMSELoss()(outputs, val_cleaned_images).item()

    val_loss /= len(val_loader)
    val_rmse_loss /= len(val_loader)

    prev_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    if current_lr != prev_lr:
        print(f"Learning Rate updated: {current_lr:.6f}\n")

    if val_loss < best_val_loss:
        print(
            f"New best validation loss: {val_loss:.4f} (Previous: {best_val_loss:.4f}), RMSE Score : {val_rmse_loss:.4f}"
        )
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(
            f"Validation loss increased! Early stopping counter: {early_stop_counter}/{patience}"
        )

    if early_stop_counter >= patience:
        print(f"Early stopping triggered! after {epoch+1} epochs")
        model.load_state_dict(best_model_state)
        print(f"Best model loaded with val_loss = {best_val_loss:.4f}")
        break

    epoch += 1

if best_model_state is not None:
    torch.save(best_model_state, "best_model.pth")
    print(f"Best model saved with val_loss = {best_val_loss:.4f}")
else:
    print("No best model was saved.")



## === cell 28
best_model_path = "best_model.pth"
model.load_state_dict(torch.load(best_model_path, map_location=device))

model.eval()
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        break




## === cell 29
def visualize_images_and_outputs(images, outputs):
    num_images = images.size(0)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 3))

    for i in range(num_images):
        axes[i, 0].imshow(images[i].cpu().numpy().squeeze(), cmap="gray")
        axes[i, 0].set_title(f"Original {i + 1}", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(outputs[i].cpu().detach().numpy().squeeze(), cmap="gray")
        axes[i, 1].set_title(f"Output {i + 1}", fontsize=10)
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


visualize_images_and_outputs(images, outputs)



## === cell 30
test_dir = "/kaggle/working/denoising_data/test"
test_dir



## === cell 31
best_model_path = "best_model.pth"
model.load_state_dict(torch.load(best_model_path, map_location=device))

model.eval()
all_outputs = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        outputs = model(batch)
        all_outputs.append(outputs.cpu())

all_outputs = torch.cat(all_outputs, dim=0)
print("all_outputs shape:", all_outputs.shape)



## === cell 32
import os
import csv
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms.functional as TF  # 원래 패딩 적용에 사용한 함수

target_size = (420, 540)  # (높이, 너비)


def compute_padding(orig_size, target_size):
    orig_height, orig_width = orig_size
    target_height, target_width = target_size
    pad_top = (target_height - orig_height) // 2 if target_height > orig_height else 0
    pad_bottom = (
        target_height - orig_height - pad_top if target_height > orig_height else 0
    )
    pad_left = (target_width - orig_width) // 2 if target_width > orig_width else 0
    pad_right = target_width - orig_width - pad_left if target_width > orig_width else 0
    return pad_top, pad_bottom, pad_left, pad_right


def remove_padding(pred, orig_size, target_size):
    orig_height, orig_width = orig_size
    pad_top, _, pad_left, _ = compute_padding(orig_size, target_size)
    return pred[:, pad_top : pad_top + orig_height, pad_left : pad_left + orig_width]


def _resolve_test_dir_fallback(expected_dir: str) -> str:
    if os.path.isdir(expected_dir):
        return expected_dir

    name = os.path.basename(expected_dir.rstrip("/"))
    candidates = [
        expected_dir,
        os.path.join("/kaggle/working/denoising_data", name),
        os.path.join("/kaggle/working/denoising-dirty-documents", name),
        os.path.join("/kaggle/input/denoising-dirty-documents", name),
        os.path.join(
            "/kaggle/input/denoising-dirty-documents/denoising-dirty-documents", name
        ),
        os.path.join(
            "/kaggle/working/denoising-dirty-documents/denoising-dirty-documents", name
        ),
    ]

    for walk_base in ("/kaggle/working", "/kaggle/input"):
        if os.path.isdir(walk_base):
            for root, dirs, _ in os.walk(walk_base):
                if name in dirs:
                    candidates.append(os.path.join(root, name))

    resolved = next((p for p in candidates if os.path.isdir(p)), None)
    if resolved is None:
        raise FileNotFoundError(
            f"Could not find test image directory. Tried: {candidates}"
        )
    return resolved


test_dir = "/kaggle/working/denoising_data/test"
if "_resolve_dir" in globals():
    test_dir = _resolve_dir(test_dir)
else:
    test_dir = _resolve_test_dir_fallback(test_dir)

test_file_paths = sorted(
    [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".png")],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
)

n_files = min(len(test_file_paths), int(all_outputs.shape[0]))
submission_data = []

BLUR_KSIZE = 0  # was 5

for i in range(n_files):
    file_path = test_file_paths[i]
    image_id = os.path.splitext(os.path.basename(file_path))[0]

    orig_img = Image.open(file_path).convert("L")
    orig_width, orig_height = orig_img.size
    orig_size = (orig_height, orig_width)  # (높이, 너비)

    pred = all_outputs[i]  # (1, 420, 540)
    cropped_pred = remove_padding(pred, orig_size, target_size)  # (1, H, W)
    pred_np = cropped_pred.squeeze(0).cpu().numpy().astype(np.float32)

    if BLUR_KSIZE >= 3 and (orig_height >= BLUR_KSIZE and orig_width >= BLUR_KSIZE):
        pred_np = cv2.GaussianBlur(pred_np, (BLUR_KSIZE, BLUR_KSIZE), 0)

    pred_np = np.clip(pred_np, 0.001, 0.999)

    for row in range(orig_height):
        for col in range(orig_width):
            pixel_id = f"{image_id}_{row+1}_{col+1}"
            pixel_value = float(pred_np[row, col])
            submission_data.append((pixel_id, pixel_value))

submission_file = "submission.csv"
with open(submission_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "value"])
    writer.writerows(submission_data)

print(
    f"Submission file '{submission_file}'이(가) 생성되었습니다. Rows: {len(submission_data)} | n_files: {n_files} | blur_ksize: {BLUR_KSIZE}"
)
