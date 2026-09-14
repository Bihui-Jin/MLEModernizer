# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                train/
                    182.png (155.1 kB)
                    153.png (160.8 kB)
                    ... and 113 other files
                train_cleaned/
                    182.png (55.7 kB)
                    153.png (58.7 kB)
                    ... and 113 other files
        input/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                train/
                    182.png (155.1 kB)
                    153.png (160.8 kB)
                    ... and 113 other files
                train_cleaned/
                    182.png (55.7 kB)
                    153.png (58.7 kB)
                    ... and 113 other files
            test/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
            train/
                train/
                    182.png (155.1 kB)
                    153.png (160.8 kB)
                    ... and 113 other files
        working/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                train/
                    182.png (155.1 kB)
                    153.png (160.8 kB)
                    ... and 113 other files
                train_cleaned/
                    182.png (55.7 kB)
                    153.png (58.7 kB)
                    ... and 113 other files
```

-> data/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
Here is some information about the columns:
id (object) has 2000 unique values. Some example values: ['110_1_1', '110_3_250', '110_3_263', '110_3_262']
value (int64) has 1 unique values: [1]

-> input/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
Here is some information about the columns:
id (object) has 2000 unique values. Some example values: ['110_1_1', '110_3_250', '110_3_263', '110_3_262']
value (int64) has 1 unique values: [1]

-> working/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
Here is some information about the columns:
id (object) has 2000 unique values. Some example values: ['110_1_1', '110_3_250', '110_3_263', '110_3_262']
value (int64) has 1 unique values: [1]

# 5. Target score

0.34838

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.30334) has done: 'The script had syntax errors (non‑ASCII hyphen and stray backticks) and used notebook‑style `!unzip` commands that fail in a plain Python run. I replaced the shell unzip with Python’s `zipfile` extraction, removed the unnecessary pip‑install cell, fixed the comment cell, and updated all paths to the extracted data directory. The submission‑creation code is cleaned up and now writes a proper `submission.csv` without errors, allowing the whole pipeline to run end‑to‑end and produce a valid Kaggle submission.'

# 9. Code solution

## === cell 0
import os
import csv
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as v2
import torchvision.transforms.functional as TF

from sklearn.model_selection import train_test_split

from PIL import Image

base_dir_candidates = [
    "/kaggle/input/denoising-dirty-documents",
    "/kaggle/working/denoising-dirty-documents",
    "/kaggle/working/input/denoising-dirty-documents",
]
base_dir = None
for cand in base_dir_candidates:
    if os.path.isdir(cand):
        base_dir = cand
        break
if base_dir is None:
    raise FileNotFoundError(
        "Could not locate the denoising‑dirty‑documents data directory."
    )
print(f"Using data directory: {base_dir}")



## === cell 1
train_dir = os.path.join(base_dir, "train")
train_cleaned_dir = os.path.join(base_dir, "train_cleaned")
test_dir = os.path.join(base_dir, "test")




## === cell 2
def load_images_from_folder(folder):
    """Utility to load all PNG images from a folder using OpenCV."""
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return images




## === cell 3
train_images = load_images_from_folder(train_dir)
train_cleaned_images = load_images_from_folder(train_cleaned_dir)
test_images = load_images_from_folder(test_dir)



## === cell 4
print(f"train_images: {len(train_images)}")
print(f"train_cleaned_images: {len(train_cleaned_images)}")
print(f"test_images: {len(test_images)}")




## === cell 5
class PadToSize:
    """Pad a (C, H, W) tensor to a target size."""

    def __init__(self, target_size):
        self.target_size = target_size  # (height, width)

    def __call__(self, img):
        _, height, width = img.shape
        target_height, target_width = self.target_size

        pad_top = (target_height - height) // 2
        pad_bottom = target_height - height - pad_top
        pad_left = (target_width - width) // 2
        pad_right = target_width - width - pad_left

        return TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=0)




## === cell 6
class Grayscale:
    """Convert a tensor image to single‑channel grayscale."""

    def __call__(self, img):
        if img.shape[0] == 3:
            img = TF.rgb_to_grayscale(img)
        elif img.shape[0] == 1:
            pass
        else:
            raise ValueError(f"Unexpected channel dimension: {img.shape[0]}")
        return img




## === cell 7
train_transforms = v2.Compose(
    [
        v2.ToTensor(),
        PadToSize((420, 540)),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.1),
        v2.RandomApply([v2.ColorJitter(brightness=0.3, contrast=0.3)], p=0.1),
        Grayscale(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

val_transforms = v2.Compose(
    [
        v2.ToTensor(),
        PadToSize((420, 540)),
        Grayscale(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

test_transforms = v2.Compose(
    [
        v2.ToTensor(),
        PadToSize((420, 540)),
        Grayscale(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3793582096.py in <cell line: 0>()
      6         v2.RandomApply([v2.ColorJitter(brightness=0.3, contrast=0.3)], p=0.1),
      7         Grayscale(),
----> 8         v2.ToDtype(torch.float32, scale=True),
      9     ]
     10 )

AttributeError: module 'torchvision.transforms' has no attribute 'ToDtype'

## === cell 8
class ImageDataset(Dataset):
    """Dataset for single images (used for test set)."""

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = sorted(
            [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.lower().endswith(".png")
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image




## === cell 9
train_files = sorted(
    [
        os.path.join(train_dir, f)
        for f in os.listdir(train_dir)
        if f.lower().endswith(".png")
    ]
)
cleaned_files = sorted(
    [
        os.path.join(train_cleaned_dir, f)
        for f in os.listdir(train_cleaned_dir)
        if f.lower().endswith(".png")
    ]
)

train_files, val_files, cleaned_train, cleaned_val = train_test_split(
    train_files, cleaned_files, test_size=2 / 9, random_state=42
)




## === cell 10
class PairedImageDataset(Dataset):
    """Dataset returning a noisy image and its cleaned counterpart."""

    def __init__(self, noisy_files, cleaned_files, transform=None):
        self.noisy_files = noisy_files
        self.cleaned_files = cleaned_files
        self.transform = transform

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_img = Image.open(self.noisy_files[idx]).convert("RGB")
        cleaned_img = Image.open(self.cleaned_files[idx]).convert("RGB")
        if self.transform:
            noisy_img = self.transform(noisy_img)
            cleaned_img = self.transform(cleaned_img)
        return noisy_img, cleaned_img




## === cell 11
if os.path.isdir(train_dir) and len(train_files) > 0:
    train_dataset = PairedImageDataset(train_files, cleaned_train, train_transforms)
    val_dataset = PairedImageDataset(val_files, cleaned_val, val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
else:
    print("Training data not found – creating empty loaders.")
    train_loader = []
    val_loader = []

test_dataset = ImageDataset(test_dir, test_transforms)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1962412839.py in <cell line: 0>()
      1 if os.path.isdir(train_dir) and len(train_files) > 0:
----> 2     train_dataset = PairedImageDataset(train_files, cleaned_train, train_transforms)
      3     val_dataset = PairedImageDataset(val_files, cleaned_val, val_transforms)
      4     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
      5     val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

NameError: name 'train_transforms' is not defined

## === cell 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)




## === cell 13
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(1, 8, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(8, 16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(16, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
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
        self.conv4 = nn.Conv2d(96, 32, kernel_size=1, stride=1, bias=False)  # 64+32
        self.conv5 = nn.Conv2d(192, 64, kernel_size=1, stride=1, bias=False)  # 128+64

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
        dec5_out = torch.cat(
            [dec5_out, enc4_out], dim=1
        )  # 64+128? actually dec5_out now 64, enc4_out 64
        dec5_out = self.conv5(dec5_out)

        dec4_out = self.dec4(enc4_out)
        dec4_out = F.interpolate(
            dec4_out, size=(52, 67), mode="bilinear", align_corners=False
        )
        dec4_out = torch.cat([dec4_out, enc3_out], dim=1)  # 32+64
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

        return torch.clamp(dec1_out, min=0.001, max=0.999)




## === cell 14
model = DenoisingAutoencoder().to(device)




## === cell 15
class RMSELoss(nn.Module):
    """Root Mean Squared Error loss."""

    def forward(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target))


class HybridLoss(nn.Module):
    """Weighted sum of RMSE and L1 loss."""

    def __init__(self, lambda_rmse=0.5, lambda_l1=0.5):
        super().__init__()
        self.lambda_rmse = lambda_rmse
        self.lambda_l1 = lambda_l1
        self.rmse_loss = RMSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.lambda_rmse * self.rmse_loss(
            pred, target
        ) + self.lambda_l1 * self.l1_loss(pred, target)


criterion = HybridLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2, verbose=False
)



## === cell 16
epochs = 5
patience = 2
early_stop_counter = 0
best_val_loss = float("inf")
best_model_state = None

if isinstance(train_loader, DataLoader) and len(train_loader) > 0:
    epoch = 0
    while epoch < epochs:
        model.train()
        train_loss = 0.0
        train_rmse_loss = 0.0
        for train_imgs, train_cleaned_imgs in train_loader:
            train_imgs = train_imgs.to(device)
            train_cleaned_imgs = train_cleaned_imgs.to(device)
            optimizer.zero_grad()
            outputs = model(train_imgs)
            loss = criterion(outputs, train_cleaned_imgs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_rmse_loss += RMSELoss()(outputs, train_cleaned_imgs).item()
        train_loss /= len(train_loader)
        train_rmse_loss /= len(train_loader)
        print(
            f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} RMSE: {train_rmse_loss:.4f}"
        )

        model.eval()
        val_loss = 0.0
        val_rmse_loss = 0.0
        with torch.no_grad():
            for val_imgs, val_cleaned_imgs in val_loader:
                val_imgs = val_imgs.to(device)
                val_cleaned_imgs = val_cleaned_imgs.to(device)
                outputs = model(val_imgs)
                loss = criterion(outputs, val_cleaned_imgs)
                val_loss += loss.item()
                val_rmse_loss += RMSELoss()(outputs, val_cleaned_imgs).item()
        val_loss /= len(val_loader)
        val_rmse_loss /= len(val_loader)

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr != prev_lr:
            print(f"Learning Rate updated: {current_lr:.6f}")

        if val_loss < best_val_loss:
            print(
                f"New best val loss {val_loss:.4f} (prev {best_val_loss:.4f}), RMSE {val_rmse_loss:.4f}"
            )
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"Val loss did not improve ({early_stop_counter}/{patience})")

        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            model.load_state_dict(best_model_state)
            break

        epoch += 1

    if best_model_state is not None:
        torch.save(best_model_state, "best_model.pth")
        print(f"Best model saved (val loss = {best_val_loss:.4f})")
else:
    print("No training data – skipping training.")



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3225628172.py in <cell line: 0>()
      5 best_model_state = None
      6 
----> 7 if isinstance(train_loader, DataLoader) and len(train_loader) > 0:
      8     epoch = 0
      9     while epoch < epochs:

NameError: name 'train_loader' is not defined

## === cell 17
if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    print("Loaded best model checkpoint.")
else:
    print("Checkpoint not found; using current model.")
model.eval()



## === cell 18
all_outputs = []
model.to(device)
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        outputs = model(batch)  # (B, 1, 420, 540)
        all_outputs.append(outputs.cpu())
if len(all_outputs) == 0:
    num_test = len(test_loader.dataset)
    all_outputs = [torch.full((1, 1, 420, 540), 0.5) for _ in range(num_test)]
else:
    all_outputs = torch.cat(all_outputs, dim=0)  # (N, 1, 420, 540)
print("All outputs shape:", all_outputs.shape)




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3654276478.py in <cell line: 0>()
      2 model.to(device)
      3 with torch.no_grad():
----> 4     for batch in test_loader:
      5         batch = batch.to(device)
      6         outputs = model(batch)  # (B, 1, 420, 540)

NameError: name 'test_loader' is not defined

## === cell 19
def compute_padding(orig_size, target_size):
    orig_h, orig_w = orig_size
    tgt_h, tgt_w = target_size
    pad_top = (tgt_h - orig_h) // 2 if tgt_h > orig_h else 0
    pad_left = (tgt_w - orig_w) // 2 if tgt_w > orig_w else 0
    return pad_top, pad_left


def remove_padding(pred, orig_size, target_size):
    """
    pred: Tensor (C, target_h, target_w)
    Returns: Tensor (C, orig_h, orig_w) cropped to original region
    """
    orig_h, orig_w = orig_size
    pad_top, pad_left = compute_padding(orig_size, target_size)
    return pred[:, pad_top : pad_top + orig_h, pad_left : pad_left + orig_w]




## === cell 20
test_file_paths = sorted(
    [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith(".png")
    ],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
)

submission_data = []
target_h, target_w = 420, 540

for idx, file_path in enumerate(test_file_paths):
    img_id = os.path.splitext(os.path.basename(file_path))[0]

    with Image.open(file_path) as img:
        orig_w, orig_h = img.size
    orig_size = (orig_h, orig_w)

    pred = all_outputs[idx]  # (1, 420, 540)
    cropped_pred = remove_padding(
        pred, orig_size, (target_h, target_w)
    )  # (1, orig_h, orig_w)
    pred_np = cropped_pred.squeeze(0).numpy()  # (orig_h, orig_w)

    for r in range(orig_h):
        for c in range(orig_w):
            pixel_id = f"{img_id}_{r+1}_{c+1}"
            pixel_value = float(pred_np[r, c])
            submission_data.append((pixel_id, pixel_value))

submission_file = "submission.csv"
with open(submission_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "value"])
    writer.writerows(submission_data)

print(f"Created '{submission_file}' with {len(submission_data)} rows.")

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/505682365.py in <cell line: 0>()
     18     orig_size = (orig_h, orig_w)
     19 
---> 20     pred = all_outputs[idx]  # (1, 420, 540)
     21     cropped_pred = remove_padding(
     22         pred, orig_size, (target_h, target_w)

IndexError: list index out of range
