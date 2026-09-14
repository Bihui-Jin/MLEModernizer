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

0.28886

# 6. Current score

0.4142

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.39043) has done: 'I fix the data loading by pointing directly to the already‑extracted dataset instead of trying to unzip missing files, and keep the rest of the pipeline unchanged. This resolves the `FileNotFoundError` that prevented any subsequent cells from running, allowing the model to train and a submission CSV to be generated.'
- What this solution (achieved 0.4361) has done: 'I adjust the loss weighting to focus solely on RMSE (the competition metric), remove weight decay from the Adam optimizer, and give the model a bit more room to train before early stopping by increasing the patience. These small hyper‑parameter tweaks should lower the validation RMSE and move the score closer to the target while keeping the core model unchanged.'
- What this solution (achieved 0.43978) has done: 'The fix focuses on the two main slow spots: data loading and submission writing. We add parallel data loading (`num_workers`, `pin_memory`) to speed each epoch, and completely rewrite the pixel‑wise CSV generation to use vectorized NumPy/Pandas operations instead of slow Python nested loops. These changes keep the model, transforms, training loop, and loss exactly the same, preserving correctness while reducing runtime well below the 600 s limit.'
- What this solution (achieved 0.4347) has done: 'I fix the CSV generation in cell 20. The original use of `np.char.add` was incorrect, causing a TypeError and resulting in an empty submission. I replace that with a straightforward nested‑loop that builds each “image_row_col” id and its corresponding pixel value, then writes them to the CSV. This ensures the file has the required 5,789,880 rows and a valid format, allowing the notebook to complete end‑to‑end.'
- What this solution (achieved 0.4771) has done: 'I fixed the transform definition by removing the unsupported `p` argument from `RandomAffine` and wrapped it in a `RandomApply` to retain the intended probability. This resolves the `RandomAffine` error, which also restores the creation of `train_loader`, allows the training loop to execute, saves the best model, runs inference, and finally generates a correctly‑formatted CSV submission.'
- What this solution (achieved 0.4737) has done: 'I adjust the data augmentation so that the noisy input image receives the random transforms while the clean target image is kept deterministic (no random flips, rotations, etc.). This alignment prevents label mismatch during training, which should lower the validation RMSE and move the score closer to the target. The change only modifies how transforms are applied in the paired dataset and keeps the rest of the model, training loop, and submission logic unchanged.'
- What this solution (achieved 0.39945) has done: 'I slightly increase the learning‑rate (to let the model learn faster) and give the early‑stopping logic a bit more patience so the training can continue a few more epochs if it is still improving. These minimal tweaks keep the exact architecture, loss and overall pipeline unchanged, yet they usually lower the validation RMSE, moving the score toward the target 0.28886.'
- What this solution (achieved 0.4142) has done: 'The changes add deterministic seeds at the start and rewrite the submission‑writing step to open the CSV once and batch‑write rows per image instead of opening and writing for every pixel. This removes the massive per‑row file‑open overhead while keeping all data transformations identical, so the produced submission file is unchanged in content and order. The rest of the pipeline, model architecture, training loop, and evaluation remain untouched.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
import csv  # added to enable writing the submission file
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 1
data_root = "/kaggle/input/denoising-dirty-documents"
os.makedirs(data_root, exist_ok=True)

expected_dirs = ["train", "train_cleaned", "test"]
missing = [d for d in expected_dirs if not os.path.isdir(os.path.join(data_root, d))]
if missing:
    raise FileNotFoundError(f"Missing expected data directories: {missing}")




## === cell 2
train_dir = os.path.join(data_root, "train")
train_cleaned_dir = os.path.join(data_root, "train_cleaned")
test_dir = os.path.join(data_root, "test")




## === cell 3
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)  # BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return images




## === cell 4
train_images = load_images_from_folder(train_dir)
train_cleaned_images = load_images_from_folder(train_cleaned_dir)
test_images = load_images_from_folder(test_dir)




## === cell 5
print(f"train_images: {len(train_images)}")
print(f"train_cleaned_images: {len(train_cleaned_images)}")
print(f"test_images: {len(test_images)}")




## === cell 6
class PadToSize:
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




## === cell 7
class Grayscale:
    def __call__(self, img):
        pil_img = TF.to_pil_image(img) if isinstance(img, torch.Tensor) else img
        grayscale_img = pil_img.convert("L")
        return TF.to_tensor(grayscale_img)




## === cell 8
train_noisy_transforms = v2.Compose(
    [
        v2.ToImage(),
        PadToSize((420, 540)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomApply(
            [
                v2.RandomAffine(
                    degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05), shear=5
                )
            ],
            p=0.5,
        ),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.4),
        Grayscale(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

train_clean_transforms = v2.Compose(
    [
        v2.ToImage(),
        PadToSize((420, 540)),
        Grayscale(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

val_transforms = train_clean_transforms
test_transforms = train_clean_transforms




## === cell 9
class ImageDataset(Dataset):
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




## === cell 10
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




## === cell 11
class PairedImageDataset(Dataset):
    """
    Returns a pair (noisy_image, clean_image).
    The noisy image gets the random augmentation pipeline,
    while the clean image receives only deterministic transforms
    to keep the target aligned with the original pixel positions.
    """

    def __init__(
        self, noisy_files, clean_files, transform_noisy=None, transform_clean=None
    ):
        self.noisy_files = noisy_files
        self.clean_files = clean_files
        self.transform_noisy = transform_noisy
        self.transform_clean = transform_clean

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_img = Image.open(self.noisy_files[idx]).convert("RGB")
        clean_img = Image.open(self.clean_files[idx]).convert("RGB")
        if self.transform_noisy:
            noisy_img = self.transform_noisy(noisy_img)
        if self.transform_clean:
            clean_img = self.transform_clean(clean_img)
        return noisy_img, clean_img




## === cell 12
num_workers = min(4, os.cpu_count())
train_dataset = PairedImageDataset(
    train_files,
    cleaned_train,
    transform_noisy=train_noisy_transforms,
    transform_clean=train_clean_transforms,
)
val_dataset = PairedImageDataset(
    val_files,
    cleaned_val,
    transform_noisy=val_transforms,
    transform_clean=val_transforms,
)
test_dataset = ImageDataset(test_dir, test_transforms)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True
)




## === cell 13
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




## === cell 14
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(1, 8, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(8, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.MaxPool2d(2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.MaxPool2d(2),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.MaxPool2d(2),
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.0),
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
            nn.Dropout(0.0),
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=1, bias=False),
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
            nn.Dropout(0.0),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=1, bias=False),
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
            nn.Dropout(0.0),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=1, bias=False),
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
            nn.Dropout(0.0),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(8, 1, kernel_size=1, bias=False),
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

        self.conv4 = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=1, bias=False)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        dec5 = self.dec5(enc5)
        dec5 = F.interpolate(dec5, size=(26, 33), mode="bilinear", align_corners=False)
        dec5 = torch.cat([dec5, enc4], dim=1)
        dec5 = self.conv5(dec5)

        dec4 = self.dec4(enc4)
        dec4 = F.interpolate(dec4, size=(52, 67), mode="bilinear", align_corners=False)
        dec4 = torch.cat([dec4, enc3], dim=1)
        dec4 = self.conv4(dec4)

        dec3 = self.dec3(dec4)
        dec3 = F.interpolate(
            dec3, size=(105, 135), mode="bilinear", align_corners=False
        )

        dec2 = self.dec2(dec3)
        dec2 = F.interpolate(
            dec2, size=(210, 270), mode="bilinear", align_corners=False
        )

        dec1 = self.dec1(dec2)
        dec1 = F.interpolate(
            dec1, size=(420, 540), mode="bilinear", align_corners=False
        )

        return torch.clamp(dec1, min=0.001, max=0.999)


model = DenoisingAutoencoder().to(device)




## === cell 15
from torchinfo import summary

summary(model, input_size=(16, 1, 420, 540), device=device)




## === cell 16
class RMSELoss(nn.Module):
    def forward(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target))


criterion = RMSELoss()  # pure RMSE matches the competition metric

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)




## === cell 17
epochs = 1000
early_stop_patience = 150
early_stop_counter = 0
best_val_loss = float("inf")
best_model_state = None

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_rmse = 0.0
    for noisy, clean in train_loader:
        noisy, clean = noisy.to(device, non_blocking=True), clean.to(
            device, non_blocking=True
        )
        optimizer.zero_grad()
        outputs = model(noisy)
        loss = criterion(outputs, clean)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_rmse += RMSELoss()(outputs, clean).item()
    train_loss /= len(train_loader)
    train_rmse /= len(train_loader)

    model.eval()
    val_loss = 0.0
    val_rmse = 0.0
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device, non_blocking=True), clean.to(
                device, non_blocking=True
            )
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            val_loss += loss.item()
            val_rmse += RMSELoss()(outputs, clean).item()
    val_loss /= len(val_loader)
    val_rmse /= len(val_loader)

    scheduler.step(val_loss)

    print(
        f"Epoch {epoch+1}/{epochs} - "
        f"Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f} - "
        f"Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if early_stop_counter >= early_stop_patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

if best_model_state is not None:
    torch.save(best_model_state, "best_model.pth")
    print(f"Best model saved (val loss {best_val_loss:.4f})")
else:
    print("No model was saved.")




## === cell 18
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

all_outputs = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device, non_blocking=True)
        preds = model(batch)  # (B, 1, 420, 540)
        all_outputs.append(preds.cpu())
all_outputs = torch.cat(all_outputs, dim=0)  # (N, 1, 420, 540)
print("All predictions shape:", all_outputs.shape)




## === cell 19
def compute_padding(orig_size, target_size):
    orig_h, orig_w = orig_size
    tgt_h, tgt_w = target_size
    pad_top = (tgt_h - orig_h) // 2 if tgt_h > orig_h else 0
    pad_bottom = tgt_h - orig_h - pad_top if tgt_h > orig_h else 0
    pad_left = (tgt_w - orig_w) // 2 if tgt_w > orig_w else 0
    pad_right = tgt_w - orig_w - pad_left if tgt_w > orig_w else 0
    return pad_top, pad_bottom, pad_left, pad_right


def remove_padding(pred, orig_size, target_size):
    orig_h, orig_w = orig_size
    pad_top, _, pad_left, _ = compute_padding(orig_size, target_size)
    return pred[:, pad_top : pad_top + orig_h, pad_left : pad_left + orig_w]


test_file_paths = sorted(
    [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith(".png")
    ],
    key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
)

assert (
    len(test_file_paths) == all_outputs.shape[0]
), "Mismatch between number of test files and predictions"

target_h, target_w = 420, 540
submission_path = "submission.csv"

with open(submission_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "value"])
    for idx, file_path in enumerate(test_file_paths):
        image_id = os.path.splitext(os.path.basename(file_path))[0]

        with Image.open(file_path) as img:
            orig_w, orig_h = img.size

        pred_tensor = all_outputs[idx]  # (1, 420, 540)
        cropped = remove_padding(pred_tensor, (orig_h, orig_w), (target_h, target_w))
        pred_np = cropped.squeeze(0).numpy()  # (orig_h, orig_w)

        rows = [
            (f"{image_id}_{r+1}_{c+1}", float(pred_np[r, c]))
            for r in range(orig_h)
            for c in range(orig_w)
        ]
        writer.writerows(rows)

print(f"Submission file created at: {submission_path}")
