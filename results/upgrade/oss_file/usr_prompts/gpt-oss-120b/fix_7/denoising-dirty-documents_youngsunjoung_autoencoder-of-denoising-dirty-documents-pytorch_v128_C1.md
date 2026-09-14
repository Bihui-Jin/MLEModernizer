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

0.28128

# 6. Current score

0.42271

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.38381) has done: 'The script had syntax errors from raw text and stray markdown, and it used Colab‑specific paths (`/content`) and notebook “!” commands that don’t work in a plain Python run. I removed the problematic text, replaced the unzip commands with Python’s zipfile extraction, switched all data folders to a writable local directory (`./denoising_data`), and eliminated the notebook‑specific install cell. These fixes let the pipeline run end‑to‑end and generate a proper `submission.csv` file while preserving the original model and training logic.'
- What this solution (achieved 0.43149) has done: 'The changes tighten the training objective to the exact competition metric by using pure RMSE loss (removing the L1 component) and lower the learning rate to 1e‑3 for more stable convergence. The early‑stopping patience is increased slightly so the model can train a few more epochs if needed, which should improve validation loss and move the RMSE closer to the target of 0.28128 while preserving the original architecture and overall pipeline.'
- What this solution (achieved 0.4277) has done: 'I slightly reduce the optimizer weight‑decay (to let the model fit the data a bit better) and add a simple test‑time augmentation: predict both the original and a horizontally‑flipped version of each test batch, flip the second output back, and average the two predictions. This small change keeps the core architecture unchanged while aiming to lower the RMSE, moving the score toward the target.'
- What this solution (achieved 0.42271) has done: 'I keep the core architecture and training loop unchanged but fix the way the clean target images are transformed. In the original code the same random augmentations were applied to both the noisy input and the clean target, which harms learning. I modify `PairedImageDataset` to accept separate transforms for the noisy and clean images and use only deterministic transforms (the validation pipeline) for the clean side while keeping the original noisy‑side augmentations. This small correction should improve validation RMSE and move the score toward the target.'

# 9. Code solution

## === cell 0
import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
import csv



## === cell 1
data_root = "./denoising_data"
os.makedirs(data_root, exist_ok=True)

zip_paths = [
    "/kaggle/input/denoising-dirty-documents/train.zip",
    "/kaggle/input/denoising-dirty-documents/train_cleaned.zip",
    "/kaggle/input/denoising-dirty-documents/test.zip",
]

for zp in zip_paths:
    with zipfile.ZipFile(zp, "r") as zf:
        zf.extractall(data_root)



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

print(f"train_images: {len(train_images)}")
print(f"train_cleaned_images: {len(train_cleaned_images)}")
print(f"test_images: {len(test_images)}")




## === cell 5
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




## === cell 6
class Grayscale:
    def __call__(self, img):
        pil_img = TF.to_pil_image(img) if isinstance(img, torch.Tensor) else img
        gray = pil_img.convert("L")
        return TF.to_tensor(gray)




## === cell 7
train_transforms = v2.Compose(
    [
        v2.ToImage(),
        PadToSize((420, 540)),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.4),
        v2.RandomApply([v2.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
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




## === cell 8
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
    """
    Returns a pair (noisy, clean). Allows distinct transforms for the noisy
    input (where we keep augmentations) and the clean target (deterministic).
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




## === cell 11
train_dataset = PairedImageDataset(
    train_files,
    cleaned_train,
    transform_noisy=train_transforms,
    transform_clean=val_transforms,
)

val_dataset = PairedImageDataset(
    val_files,
    cleaned_val,
    transform_noisy=val_transforms,
    transform_clean=val_transforms,
)

test_dataset = ImageDataset(test_dir, test_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



## === cell 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




## === cell 13
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
            nn.Dropout(0.3),
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
            nn.Dropout(0.2),
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
            nn.Dropout(0.1),
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
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        d5 = self.dec5(e5)
        d5 = F.interpolate(d5, size=(26, 33), mode="bilinear", align_corners=False)
        d5 = torch.cat([d5, e4], dim=1)
        d5 = self.conv5(d5)

        d4 = self.dec4(e4)
        d4 = F.interpolate(d4, size=(52, 67), mode="bilinear", align_corners=False)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.conv4(d4)

        d3 = self.dec3(d4)
        d3 = F.interpolate(d3, size=(105, 135), mode="bilinear", align_corners=False)

        d2 = self.dec2(d3)
        d2 = F.interpolate(d2, size=(210, 270), mode="bilinear", align_corners=False)

        d1 = self.dec1(d2)
        d1 = F.interpolate(d1, size=(420, 540), mode="bilinear", align_corners=False)

        return torch.clamp(d1, 0.001, 0.999)


model = DenoisingAutoencoder().to(device)




## === cell 14
class RMSELoss(nn.Module):
    def forward(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target))




## === cell 15
criterion = RMSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)



## === cell 16
epochs = 200
patience = 8
best_val_loss = float("inf")
best_state = None
early_stop = 0

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for noisy, clean in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        out = model(noisy)
        loss = criterion(out, clean)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            out = model(noisy)
            loss = criterion(out, clean)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    print(
        f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.5f} - Val loss: {val_loss:.5f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = model.state_dict()
        early_stop = 0
        print("  New best model saved.")
    else:
        early_stop += 1
        print(f"  No improvement ({early_stop}/{patience})")
        if early_stop >= patience:
            print("Early stopping triggered.")
            break

if best_state is not None:
    torch.save(best_state, "best_model.pth")
    print(f"Best model checkpoint saved (val loss {best_val_loss:.5f})")
else:
    print("No model checkpoint was saved.")



## === cell 17
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
all_outputs = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)  # (N,1,420,540)
        preds = model(batch)
        batch_flipped = torch.flip(batch, dims=[3])  # flip width dimension
        preds_flipped = model(batch_flipped)
        preds_flipped = torch.flip(preds_flipped, dims=[3])  # unflip
        avg_preds = (preds + preds_flipped) / 2.0
        all_outputs.append(avg_preds.cpu())
all_outputs = torch.cat(all_outputs, dim=0)  # shape (N_test, 1, 420, 540)




## === cell 18
def compute_padding(orig_size, target_size):
    oh, ow = orig_size
    th, tw = target_size
    pad_top = (th - oh) // 2 if th > oh else 0
    pad_bottom = th - oh - pad_top if th > oh else 0
    pad_left = (tw - ow) // 2 if tw > ow else 0
    pad_right = tw - ow - pad_left if tw > ow else 0
    return pad_top, pad_bottom, pad_left, pad_right


def remove_padding(pred, orig_size, target_size):
    oh, ow = orig_size
    pad_top, _, pad_left, _ = compute_padding(orig_size, target_size)
    return pred[:, pad_top : pad_top + oh, pad_left : pad_left + ow]




## === cell 19
submission_rows = []
target_size = (420, 540)

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
    w, h = pil_img.size
    orig_sz = (h, w)  # (height, width)

    pred_tensor = all_outputs[idx]  # (1, 420, 540)
    cropped = remove_padding(pred_tensor, orig_sz, target_size)
    pred_np = cropped.squeeze(0).numpy()

    for r in range(h):
        for c in range(w):
            pixel_id = f"{img_id}_{r+1}_{c+1}"
            pixel_val = float(pred_np[r, c])
            submission_rows.append((pixel_id, pixel_val))

submission_path = "submission.csv"
with open(submission_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "value"])
    writer.writerows(submission_rows)

print(f"Submission file written to {submission_path}")
