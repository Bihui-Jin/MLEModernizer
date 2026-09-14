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

0.29275

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import transforms, v2
import torchvision.transforms.functional as TF
import csv



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/3428454127.py in <cell line: 0>()
     11 from sklearn.model_selection import train_test_split
     12 import matplotlib.pyplot as plt
---> 13 from torchvision import transforms, v2
     14 import torchvision.transforms.functional as TF
     15 import csv

ImportError: cannot import name 'v2' from 'torchvision' (/usr/local/lib/python3.11/dist-packages/torchvision/__init__.py)

## === cell 2
try:
    from torchinfo import summary
except ImportError:
    pass  # Not critical for execution.



## === cell 3
base_dir = "/kaggle/input/denoising-dirty-documents"
train_dir = os.path.join(base_dir, "train")
train_cleaned_dir = os.path.join(base_dir, "train_cleaned")
test_dir = os.path.join(base_dir, "test")




## === cell 4
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)  # BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
train_sizes = [img.shape[:2] for img in train_images]
train_cleaned_sizes = [img.shape[:2] for img in train_cleaned_images]
test_sizes = [img.shape[:2] for img in test_images]

unique_train_sizes = np.unique(train_sizes, axis=0)
unique_train_cleaned_sizes = np.unique(train_cleaned_sizes, axis=0)
unique_test_sizes = np.unique(test_sizes, axis=0)

print(f"train_images sizes:\n{unique_train_sizes}")
print(f"train_cleaned_images sizes:\n{unique_train_cleaned_sizes}")
print(f"test_images sizes:\n{unique_test_sizes}")




## === cell 8
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




## === cell 9
class Grayscale:
    def __call__(self, img):
        pil_img = TF.to_pil_image(img) if isinstance(img, torch.Tensor) else img
        gray = pil_img.convert("L")
        return TF.to_tensor(gray)




## === cell 10
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




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2346121324.py in <cell line: 0>()
----> 1 train_transforms = v2.Compose(
      2     [
      3         v2.ToImage(),
      4         PadToSize((420, 540)),
      5         v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.4),

NameError: name 'v2' is not defined

## === cell 11
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




## === cell 12
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




## === cell 13
class PairedImageDataset(Dataset):
    def __init__(self, noisy_paths, clean_paths, transform=None):
        self.noisy_paths = noisy_paths
        self.clean_paths = clean_paths
        self.transform = transform

    def __len__(self):
        return len(self.noisy_paths)

    def __getitem__(self, idx):
        noisy = Image.open(self.noisy_paths[idx]).convert("RGB")
        clean = Image.open(self.clean_paths[idx]).convert("RGB")
        if self.transform:
            noisy = self.transform(noisy)
            clean = self.transform(clean)
        return noisy, clean




## === cell 14
train_dataset = PairedImageDataset(train_files, cleaned_train, train_transforms)
val_dataset = PairedImageDataset(val_files, cleaned_val, val_transforms)
test_dataset = ImageDataset(test_dir, test_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1443139496.py in <cell line: 0>()
----> 1 train_dataset = PairedImageDataset(train_files, cleaned_train, train_transforms)
      2 val_dataset = PairedImageDataset(val_files, cleaned_val, val_transforms)
      3 test_dataset = ImageDataset(test_dir, test_transforms)
      4 
      5 train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

NameError: name 'train_transforms' is not defined

## === cell 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## === cell 16
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1, bias=False),
            nn.Conv2d(1, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1, bias=False),
            nn.Conv2d(8, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.Conv2d(16, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.Conv2d(32, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.Conv2d(64, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 1, bias=False),
            nn.ConvTranspose2d(
                64, 64, 3, 2, 1, output_padding=1, groups=64, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 1, bias=False),
            nn.ConvTranspose2d(
                32, 32, 3, 2, 1, output_padding=1, groups=32, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 1, bias=False),
            nn.ConvTranspose2d(
                16, 16, 3, 2, 1, output_padding=1, groups=16, bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 1, bias=False),
            nn.ConvTranspose2d(8, 8, 3, 2, 1, output_padding=1, groups=8, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(8, 1, 1, bias=False),
            nn.ConvTranspose2d(1, 1, 3, 2, 1, output_padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.conv4 = nn.Conv2d(64, 32, 1, bias=False)
        self.conv5 = nn.Conv2d(128, 64, 1, bias=False)

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



## === cell 17
if "summary" in globals():
    summary(model, input_size=(1, 1, 420, 540), device=device)




## === cell 18
class RMSELoss(nn.Module):
    def forward(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target))


class HybridLoss(nn.Module):
    def __init__(self, lambda_rmse=0.8, lambda_l1=0.2):
        super().__init__()
        self.lambda_rmse = lambda_rmse
        self.lambda_l1 = lambda_l1
        self.rmse = RMSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        return self.lambda_rmse * self.rmse(pred, target) + self.lambda_l1 * self.l1(
            pred, target
        )


criterion = HybridLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)



## === cell 19
max_epochs = 20
patience = 5
best_val_loss = float("inf")
best_state = None
early_stop = 0

for epoch in range(max_epochs):
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
        f"Epoch {epoch+1}/{max_epochs} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = model.state_dict()
        early_stop = 0
        print("  New best model saved.")
    else:
        early_stop += 1
        print(f"  No improvement. Early stop counter: {early_stop}/{patience}")

    if early_stop >= patience:
        print("Early stopping triggered.")
        break

if best_state is not None:
    torch.save(best_state, "best_model.pth")
    print("Best model checkpoint stored as best_model.pth")
else:
    print("No model was saved.")



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4021047538.py in <cell line: 0>()
      9     model.train()
     10     train_loss = 0.0
---> 11     for noisy, clean in train_loader:
     12         noisy, clean = noisy.to(device), clean.to(device)
     13         optimizer.zero_grad()

NameError: name 'train_loader' is not defined

## === cell 20
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
all_outputs = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)  # (B,1,420,540)
        all_outputs.append(out.cpu())
all_outputs = torch.cat(all_outputs, dim=0)  # (N,1,420,540)
print("Inference completed. Output tensor shape:", all_outputs.shape)




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3778528055.py in <cell line: 0>()
      1 # Load best model and run inference on the whole test set
----> 2 model.load_state_dict(torch.load("best_model.pth", map_location=device))
      3 model.eval()
      4 all_outputs = []
      5 with torch.no_grad():

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: 'best_model.pth'

## === cell 21
def compute_padding(orig_size, target_size):
    orig_h, orig_w = orig_size
    tgt_h, tgt_w = target_size
    pad_top = (tgt_h - orig_h) // 2 if tgt_h > orig_h else 0
    pad_left = (tgt_w - orig_w) // 2 if tgt_w > orig_w else 0
    return pad_top, pad_left


def remove_padding(pred, orig_size, target_size):
    pad_top, pad_left = compute_padding(orig_size, target_size)
    h, w = orig_size
    return pred[:, pad_top : pad_top + h, pad_left : pad_left + w]




## === cell 22
test_files = sorted(
    [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith(".png")
    ],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
)

submission_rows = []
target_h, target_w = 420, 540

for idx, fp in enumerate(test_files):
    img_id = os.path.splitext(os.path.basename(fp))[0]
    pil_img = Image.open(fp).convert("L")
    orig_w, orig_h = pil_img.size  # width, height
    orig_size = (orig_h, orig_w)  # (height, width)

    pred = all_outputs[idx]  # (1, 420, 540)
    cropped = remove_padding(pred, orig_size, (target_h, target_w)).squeeze(0).numpy()

    for r in range(orig_h):
        for c in range(orig_w):
            pixel_id = f"{img_id}_{r+1}_{c+1}"
            pixel_val = float(cropped[r, c])
            submission_rows.append((pixel_id, pixel_val))

submission_path = "submission.csv"
with open(submission_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "value"])
    writer.writerows(submission_rows)

print(f"Submission file written to {submission_path}")

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2754854540.py in <cell line: 0>()
     18     orig_size = (orig_h, orig_w)  # (height, width)
     19 
---> 20     pred = all_outputs[idx]  # (1, 420, 540)
     21     cropped = remove_padding(pred, orig_size, (target_h, target_w)).squeeze(0).numpy()
     22 

NameError: name 'all_outputs' is not defined
