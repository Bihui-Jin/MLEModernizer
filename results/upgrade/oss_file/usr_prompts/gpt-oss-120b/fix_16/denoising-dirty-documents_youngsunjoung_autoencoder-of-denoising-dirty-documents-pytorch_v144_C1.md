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

0.36675

# 6. Current score

0.40651

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.30189) has done: 'The changes fix the initial non‑Python description, replace the shell‑style unzip with Python‑based extraction using the correct Kaggle input paths, centralize the data directory to a working folder, and adjust the training loop to run a reasonable number of epochs with early stopping. These fixes remove the syntax errors, ensure files are found, and guarantee that a proper `submission.csv` with the required columns is created.'
- What this solution (achieved 0.44371) has done: 'The fix addresses the channel‑mismatch that caused the model to crash during the forward pass (and consequently prevented training, checkpoint saving, and submission generation). In **cell 15** the 1×1 convolution `self.conv5` is changed to accept the actual 64‑channel decoder output instead of the wrong 128 channels. This resolves the `RuntimeError` in both the summary and training loops, allowing the model to train, save a checkpoint, run inference, and produce a valid `submission.csv` with the required id/value format. No other logic is altered, preserving the original architecture and training strategy.'
- What this solution (achieved 0.50904) has done: 'The fix adds a small helper `compute_padding` used when converting the model’s padded predictions back to the original image size, eliminating the `NameError`. This change only affects post‑processing and keeps the core model and training logic unchanged, ensuring a valid `submission.csv` is produced.'
- What this solution (achieved 0.42245) has done: 'I add a fixed random seed for reproducibility, lower the optimizer learning rate to 1e‑3, switch the hybrid loss to use only RMSE (which aligns directly with the competition metric), and allow more training epochs with a longer patience for early stopping. These minimal adjustments should improve validation performance and move the RMSE closer to the target without altering the core model architecture or inference pipeline.'
- What this solution (achieved 0.4245) has done: 'I lower the optimizer learning rate and remove weight decay, then allow the training loop to run for more epochs with a longer early‑stopping patience. These modest hyper‑parameter tweaks let the model train longer and converge to a lower validation RMSE, moving the score closer to the target without altering the architecture or training logic.'
- What this solution (achieved 0.41099) has done: 'I reduce unnecessary augmentations, lower the learning rate, and allow more training epochs with a longer early‑stopping patience. These minimal tweaks should help the model converge to a lower validation RMSE, moving the score closer to the target 0.36675 without altering the core architecture or training logic.'
- What this solution (achieved 0.42049) has done: 'I slightly lower the optimizer learning rate, add a tiny weight decay, and give the training loop more epochs with a longer early‑stopping patience. These minimal hyper‑parameter tweaks keep the exact model architecture and loss unchanged while allowing the network to converge a bit further, which should reduce the validation RMSE toward the target value.'
- What this solution (achieved 0.40651) has done: 'The changes preload all images (training, validation, and test) into RAM after applying the same transforms, eliminating repeated disk I/O during every epoch and dramatically speeding up training and inference. Unused image‑loading cells are turned into no‑ops, and the data loaders are given `pin_memory=True` and multiple workers to further accelerate GPU data transfer. The core model architecture, loss, optimizer, and training logic remain unchanged, preserving identical results while fitting comfortably within the 600‑second limit.'

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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import v2
import math
import torchvision.transforms.functional as TF
import torch.nn.functional as F

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 1
import subprocess, sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "torchinfo"])
from torchinfo import summary




## === cell 2
import zipfile

INPUT_ROOT = "/kaggle/input/denoising-dirty-documents"
WORK_ROOT = "/kaggle/working/denoising_data"

os.makedirs(WORK_ROOT, exist_ok=True)


def unzip_to(src, dst):
    with zipfile.ZipFile(src, "r") as z:
        z.extractall(dst)


unzip_to(os.path.join(INPUT_ROOT, "train.zip"), WORK_ROOT)
unzip_to(os.path.join(INPUT_ROOT, "test.zip"), WORK_ROOT)
unzip_to(os.path.join(INPUT_ROOT, "train_cleaned.zip"), WORK_ROOT)




## === cell 3
train_dir = os.path.join(WORK_ROOT, "train")
train_cleaned_dir = os.path.join(WORK_ROOT, "train_cleaned")
test_dir = os.path.join(WORK_ROOT, "test")




## === cell 7
class PadToSize:
    def __init__(self, target_size):
        self.target_size = target_size  # (height, width)

    def __call__(self, img):
        _, height, width = img.shape
        target_height, target_width = self.target_size

        pad_top = (target_height - height) // 2
        pad_bottom = target_height - height - pad_top
        pad_left = (target_width - width) // 2
        pad_right = target_width - pad_left

        return TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=0)




## === cell 8
class Grayscale:
    def __call__(self, img):
        pil_img = TF.to_pil_image(img) if isinstance(img, torch.Tensor) else img
        grayscale_img = pil_img.convert("L")
        return TF.to_tensor(grayscale_img)




## === cell 9
train_transforms = v2.Compose(
    [
        v2.ToImage(),
        PadToSize((420, 540)),
        Grayscale(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

val_transforms = v2.Compose(
    [
        v2.ToImage(),
        PadToSize((420, 540)),
        Grayscale(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

test_transforms = v2.Compose(
    [
        v2.ToImage(),
        PadToSize((420, 540)),
        Grayscale(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)




## === cell 10
class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.image_files = sorted(
            [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.lower().endswith(".png")
            ]
        )
        self.tensors = [
            self.transform(Image.open(p).convert("RGB")) for p in self.image_files
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        return self.tensors[idx]




## === cell 11
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




## === cell 12
class PairedImageDataset(Dataset):
    """
    Loads and transforms all paired images once during initialization.
    This avoids per‑epoch disk reads and speeds up training.
    """

    def __init__(self, train_files, cleaned_files, transform=None):
        self.transform = transform
        self.train_tensors = [
            self.transform(Image.open(p).convert("RGB")) for p in train_files
        ]
        self.cleaned_tensors = [
            self.transform(Image.open(p).convert("RGB")) for p in cleaned_files
        ]

    def __len__(self):
        return len(self.train_tensors)

    def __getitem__(self, idx):
        return self.train_tensors[idx], self.cleaned_tensors[idx]




## === cell 13
train_dataset = PairedImageDataset(train_files, cleaned_train, train_transforms)
val_dataset = PairedImageDataset(val_files, cleaned_val, val_transforms)
test_dataset = ImageDataset(test_dir, test_transforms)

train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
)




## === cell 14
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## === cell 15
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(1, 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(8, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=1, bias=False),
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
            nn.Dropout(0.3),
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=1, bias=False),
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
            nn.Dropout(0.3),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=1, bias=False),
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
            nn.Dropout(0.2),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(32, 8, kernel_size=1, bias=False),
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
            nn.Dropout(0.1),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=1, bias=False),
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
        self.conv5 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=1, bias=False)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        dec5 = self.dec5(enc5)
        dec5 = F.interpolate(
            dec5, size=enc4.shape[2:], mode="bilinear", align_corners=False
        )
        dec5 = self.conv5(dec5)
        dec5 = torch.cat([dec5, enc4], dim=1)

        dec4 = self.dec4(dec5)
        dec4 = F.interpolate(
            dec4, size=enc3.shape[2:], mode="bilinear", align_corners=False
        )
        dec4 = self.conv4(dec4)
        dec4 = torch.cat([dec4, enc3], dim=1)

        dec3 = self.dec3(dec4)
        dec3 = F.interpolate(
            dec3, size=enc2.shape[2:], mode="bilinear", align_corners=False
        )
        dec3 = torch.cat([dec3, enc2], dim=1)

        dec2 = self.dec2(dec3)
        dec2 = F.interpolate(
            dec2, size=enc1.shape[2:], mode="bilinear", align_corners=False
        )
        dec2 = torch.cat([dec2, enc1], dim=1)

        dec1 = self.dec1(dec2)
        dec1 = F.interpolate(
            dec1, size=x.shape[2:], mode="bilinear", align_corners=False
        )

        return torch.clamp(dec1, 0.001, 0.999)


model = DenoisingAutoencoder().to(device)




## === cell 16
summary(model, input_size=(1, 1, 420, 540), device=device)




## === cell 17
class RMSELoss(nn.Module):
    def forward(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target))




## === cell 18
class HybridLoss(nn.Module):
    def __init__(self, lambda_rmse=1.0, lambda_l1=0.1):
        super().__init__()
        self.lambda_rmse = lambda_rmse
        self.lambda_l1 = lambda_l1
        self.rmse = RMSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        return self.lambda_rmse * self.rmse(pred, target) + self.lambda_l1 * self.l1(
            pred, target
        )


criterion = HybridLoss(lambda_rmse=1.0, lambda_l1=0.1)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)




## === cell 19
max_epochs = 300  # unchanged; early stopping will halt earlier if no improvement
patience = 40
best_val = float("inf")
best_state = None
early_stop = 0

for epoch in range(max_epochs):
    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    print(
        f"Epoch {epoch+1}/{max_epochs} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}"
    )

    if val_loss < best_val:
        best_val = val_loss
        best_state = model.state_dict()
        early_stop = 0
        print("  New best model saved")
    else:
        early_stop += 1
        print(f"  No improvement (early stop counter {early_stop}/{patience})")
        if early_stop >= patience:
            print("Early stopping triggered")
            break

if best_state is not None:
    torch.save(best_state, "best_model.pth")
    print("Best model checkpoint written to best_model.pth")
else:
    print("No model was saved (unexpected)")




## === cell 20
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()




## === cell 21
all_outputs = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device, non_blocking=True)
        preds = model(batch)  # (B, 1, 420, 540)
        all_outputs.append(preds.cpu())
all_outputs = torch.cat(all_outputs, dim=0)  # (N_test, 1, 420, 540)
print("Inference completed. Output tensor shape:", all_outputs.shape)




## === cell 22
import csv


def compute_padding(h, w, target_h=420, target_w=540):
    """
    h, w : original image height and width
    Returns (pad_top, pad_left) that were added to reach the target size.
    """
    pad_top = (target_h - h) // 2
    pad_left = (target_w - w) // 2
    return pad_top, pad_left


NOISE_STD = 0.0  # keep at 0 for best score

submission_rows = []
test_paths = sorted(
    [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith(".png")
    ],
    key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
)

for idx, path in enumerate(test_paths):
    img_id = os.path.splitext(os.path.basename(path))[0]
    pil_img = Image.open(path).convert("L")
    w, h = pil_img.size  # width, height
    pad_top, pad_left = compute_padding(h, w)

    pred_tensor = all_outputs[idx].squeeze(0)  # (420, 540)

    if NOISE_STD > 0:
        pred_tensor = torch.clamp(
            pred_tensor + torch.randn_like(pred_tensor) * NOISE_STD, 0.001, 0.999
        )

    cropped = pred_tensor[pad_top : pad_top + h, pad_left : pad_left + w]
    np_img = cropped.numpy()

    rows = np.arange(1, h + 1).reshape(-1, 1)
    cols = np.arange(1, w + 1).reshape(1, -1)
    ids = np.char.add(
        np.char.add(
            np.char.add(
                np.char.add(np.char.add(str(img_id) + "_", rows.astype(str)), "_"),
                cols.astype(str),
            ),
            "",
        ),
        "",
    )
    submission_rows.extend(zip(ids.ravel(), np_img.ravel().astype(float)))

submission_path = os.path.join("/kaggle/working", "submission.csv")
with open(submission_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "value"])
    writer.writerows(submission_rows)

print(f"Submission file written to {submission_path} with {len(submission_rows)} rows.")
