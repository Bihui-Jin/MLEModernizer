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

0.37763

# 6. Current score

0.44896

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.42244) has done: 'I fixed the path errors (the data lives under /kaggle/input/denoising‑dirty‑documents), removed stray markdown backticks, and re‑ordered the code so that the dataset, dataloaders, model, training loop, and submission generation run sequentially. The script now creates a valid submission.csv with the required “id,value” columns and train the auto‑encoder (early‑stopping stop before 1000 epochs). This resolves the runtime failures and enables a proper RMSE‑based evaluation, moving the solution toward the target score.'
- What this solution (achieved 0.44635) has done: 'I keep the overall model and training pipeline unchanged but make small, targeted tweaks that are likely to lower the validation RMSE and therefore move the score closer to the target. Specifically, I reduce the optimizer weight‑decay (to avoid over‑regularising), lower the learning rate a bit, add a modest L1 component to the loss (lambda_l1 = 0.1) to improve pixel‑wise accuracy, and give the early‑stopping logic a larger patience window so the model can train a few more epochs if needed. These minimal changes preserve the core architecture and training semantics while encouraging a modest performance gain.'
- What this solution (achieved 0.46767) has done: 'I lower the regularisation that may hurt RMSE by removing the L 1 term from the loss and setting the optimizer’s weight‑decay to 0. I also give the model a bit more training time: increase the maximum epochs and the early‑stopping patience so it can continue until validation loss truly stops improving. These minimal tweaks keep the exact same network architecture and data pipeline but should push the validation RMSE down toward the target value.'
- What this solution (achieved 0.49184) has done: 'I slightly adjust the preprocessing and loss to make the model focus more on the exact pixel‑wise reconstruction, which should lower the validation RMSE and move the score toward the target. Specifically, I remove the random blur/color jitter augmentations from the training transforms (they add unnecessary noise for this denoising task) and re‑introduce a modest L1 component (λ = 0.1) while adding a tiny weight‑decay (1e‑5) to help regularisation. These are minimal changes that keep the core architecture and training loop intact.'
- What this solution (achieved 0.44896) has done: 'I lower regularisation and remove the auxiliary L1 term, which should let the auto‑encoder focus on minimizing pure RMSE. I also disable dropout (set its probability to 0) and give early stopping a bit more patience so the model can train a few extra epochs if needed. These minimal tweaks keep the exact architecture and training loop unchanged while aiming to reduce the validation RMSE and move the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split



## === cell 1
train_dir = "/kaggle/input/denoising-dirty-documents/train"
train_cleaned_dir = "/kaggle/input/denoising-dirty-documents/train_cleaned"
test_dir = "/kaggle/input/denoising-dirty-documents/test"




## === cell 2
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


class Grayscale:
    def __call__(self, img):
        pil_img = TF.to_pil_image(img) if isinstance(img, torch.Tensor) else img
        gray = pil_img.convert("L")
        return TF.to_tensor(gray)


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




## === cell 3
class ImageDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        self.files = sorted(
            [
                os.path.join(dir_path, f)
                for f in os.listdir(dir_path)
                if f.endswith(".png")
            ]
        )
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class PairedImageDataset(Dataset):
    def __init__(self, noisy_files, clean_files, transform=None):
        self.noisy_files = noisy_files
        self.clean_files = clean_files
        self.transform = transform

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy = Image.open(self.noisy_files[idx]).convert("RGB")
        clean = Image.open(self.clean_files[idx]).convert("RGB")
        if self.transform:
            noisy = self.transform(noisy)
            clean = self.transform(clean)
        return noisy, clean




## === cell 4
all_noisy = sorted(
    [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".png")]
)
all_clean = sorted(
    [
        os.path.join(train_cleaned_dir, f)
        for f in os.listdir(train_cleaned_dir)
        if f.endswith(".png")
    ]
)

train_noisy, val_noisy, train_clean, val_clean = train_test_split(
    all_noisy, all_clean, test_size=2 / 9, random_state=42
)




## === cell 5
train_dataset = PairedImageDataset(train_noisy, train_clean, transform=train_transforms)
val_dataset = PairedImageDataset(val_noisy, val_clean, transform=val_transforms)
test_dataset = ImageDataset(test_dir, transform=test_transforms)

train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
)




## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## === cell 7
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1, bias=False),
            nn.Conv2d(1, 8, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1, bias=False),
            nn.Conv2d(8, 16, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.0),  # dropout disabled
            nn.MaxPool2d(2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.Conv2d(16, 32, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.0),  # dropout disabled
            nn.MaxPool2d(2),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.Conv2d(32, 64, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.0),  # dropout disabled
            nn.MaxPool2d(2),
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.Conv2d(64, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.0),  # dropout disabled
            nn.MaxPool2d(2),
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 1),
            nn.ConvTranspose2d(
                64, 64, 3, 2, 1, output_padding=1, groups=64, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.0),
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 1),
            nn.ConvTranspose2d(
                32, 32, 3, 2, 1, output_padding=1, groups=32, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.0),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 1),
            nn.ConvTranspose2d(
                16, 16, 3, 2, 1, output_padding=1, groups=16, bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.0),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 1),
            nn.ConvTranspose2d(8, 8, 3, 2, 1, output_padding=1, groups=8, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(0.0),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(8, 1, 1),
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
        d5 = F.interpolate(d5, (26, 33), mode="bilinear", align_corners=False)
        d5 = torch.cat([d5, e4], dim=1)
        d5 = self.conv5(d5)
        d4 = self.dec4(e4)
        d4 = F.interpolate(d4, (52, 67), mode="bilinear", align_corners=False)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.conv4(d4)
        d3 = self.dec3(d4)
        d3 = F.interpolate(d3, (105, 135), mode="bilinear", align_corners=False)
        d2 = self.dec2(d3)
        d2 = F.interpolate(d2, (210, 270), mode="bilinear", align_corners=False)
        d1 = self.dec1(d2)
        d1 = F.interpolate(d1, (420, 540), mode="bilinear", align_corners=False)
        return torch.clamp(d1, 0.001, 0.999)


model = DenoisingAutoencoder().to(device)




## === cell 8
class RMSELoss(nn.Module):
    def forward(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target))


class HybridLoss(nn.Module):
    def __init__(self, lambda_rmse=1.0, lambda_l1=0.0):
        super().__init__()
        self.lambda_rmse = lambda_rmse
        self.lambda_l1 = lambda_l1
        self.rmse = RMSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        return self.lambda_rmse * self.rmse(pred, target) + self.lambda_l1 * self.l1(
            pred, target
        )


criterion = HybridLoss(lambda_rmse=1.0, lambda_l1=0.0)
optimizer = optim.Adam(model.parameters(), lr=4e-4, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)




## === cell 9
epochs = 2000
patience = 30  # a little more patience for early stopping
best_val_loss = float("inf")
best_state = None
early_stop_cnt = 0

for epoch in range(epochs):
    model.train()
    train_rmse = 0.0
    for noisy, clean in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        out = model(noisy)
        loss = criterion(out, clean)
        loss.backward()
        optimizer.step()
        train_rmse += RMSELoss()(out, clean).item()
    train_rmse /= len(train_loader)

    model.eval()
    val_rmse = 0.0
    val_loss = 0.0
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            out = model(noisy)
            loss = criterion(out, clean)
            val_loss += loss.item()
            val_rmse += RMSELoss()(out, clean).item()
    val_loss /= len(val_loader)
    val_rmse /= len(val_loader)

    scheduler.step(val_loss)

    print(
        f"Epoch {epoch+1}: Train RMSE {train_rmse:.4f} | Val RMSE {val_rmse:.4f} | Val Loss {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = model.state_dict()
        early_stop_cnt = 0
    else:
        early_stop_cnt += 1
        if early_stop_cnt >= patience:
            print("Early stopping triggered")
            break

torch.save(best_state, "best_model.pth")
print("Best model saved.")




## === cell 10
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

all_outputs = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        outs = model(batch)
        all_outputs.append(outs.cpu())
all_outputs = torch.cat(all_outputs, dim=0)  # shape: (num_test, 1, 420, 540)
print("All test outputs shape:", all_outputs.shape)




## === cell 11
def compute_padding(orig_size, target_size):
    oh, ow = orig_size
    th, tw = target_size
    pad_top = (th - oh) // 2 if th > oh else 0
    pad_left = (tw - ow) // 2 if tw > ow else 0
    return pad_top, pad_left


def remove_padding(pred, orig_size, target_size):
    oh, ow = orig_size
    pad_top, pad_left = compute_padding(orig_size, target_size)
    return pred[:, pad_top : pad_top + oh, pad_left : pad_left + ow]


test_file_paths = sorted(
    [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".png")],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
)

submission_rows = []
target_h, target_w = 420, 540

for idx, file_path in enumerate(test_file_paths):
    image_id = os.path.splitext(os.path.basename(file_path))[0]
    pil_img = Image.open(file_path).convert("L")
    orig_w, orig_h = pil_img.size
    orig_size = (orig_h, orig_w)

    pred = all_outputs[idx]  # (1, 420, 540)
    cropped = remove_padding(pred, orig_size, (target_h, target_w))
    pred_np = cropped.squeeze(0).numpy()

    for r in range(orig_h):
        for c in range(orig_w):
            pixel_id = f"{image_id}_{r+1}_{c+1}"
            pixel_val = float(pred_np[r, c])
            submission_rows.append((pixel_id, pixel_val))

submission_path = "submission.csv"
with open(submission_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "value"])
    writer.writerows(submission_rows)

print(f"Submission file created at {submission_path}")
