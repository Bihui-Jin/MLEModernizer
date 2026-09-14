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

0.30876

# 6. Current score

0.41596

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.43607) has done: 'I correct the data‑path handling (the zip files don’t exist in the Kaggle environment) and add the missing `csv` import so the script can locate the image folders, run the training pipeline, and finally write a proper `submission.csv` file.'
- What this solution (achieved 0.46779) has done: 'I lower the learning rate, give the loss function full weight to RMSE (the competition metric), and extend early‑stopping patience so the model can train longer and converge to a lower error, which should move the RMSE from 0.436 closer to the target 0.308.'
- What this solution (achieved 0.44849) has done: 'I remove the random augmentations from the training transforms (so the model learns directly from the noisy‑clean pairs), lower the optimizer learning rate slightly, add a small L1 component to the loss for smoother reconstructions, and give the training loop a bit more room (more epochs and a longer early‑stop patience). These minimal tweaks keep the architecture unchanged while encouraging a lower validation RMSE, moving the score toward the target.'
- What this solution (achieved 0.43495) has done: 'I tighten the training loss to focus solely on RMSE (remove the extra L1 term) and lower the optimizer learning rate to allow finer convergence. I also give the early‑stopping patience a bit more room (30 epochs) so the model can train a little longer if needed. These small adjustments keep the architecture untouched while nudging validation RMSE closer to the target.'
- What this solution (achieved 0.47711) has done: 'The changes preload all images (applying transforms once) and use faster DataLoaders, instantiate the RMSE loss only once, and vectorize the submission creation to avoid costly Python‑level loops, all while keeping the model architecture, training procedure, and evaluation unchanged.'
- What this solution (achieved 0.50267) has done: 'I lower the dropout rates to 0 so the network can fully learn the data, increase the optimizer learning‑rate to 1e‑4 (and remove weight decay) to let the model converge faster, and add a simple weight‑initialisation step after the model is created. These minimal adjustments keep the architecture and training loop unchanged while encouraging a lower validation RMSE, moving the score toward the target 0.30876.'
- What this solution (achieved 0.51868) has done: 'I fine‑tune the training setup by (1) adding a modest L1 component to the hybrid loss (λₗ₁ = 0.1) to encourage smoother reconstructions, (2) lowering the Adam learning rate to 5e‑5 for more stable convergence, and (3) extending the early‑stopping patience to 100 epochs so the model can keep improving later in training. These small adjustments keep the architecture and overall pipeline unchanged while nudging the validation RMSE toward the target score.'
- What this solution (achieved 0.46391) has done: 'Implemented modest hyper‑parameter tweaks to move validation RMSE toward the target: removed the L1 component from the hybrid loss (set λₗ₁ to 0), increased the Adam learning rate to 1e‑4 for faster convergence, shortened the training schedule to 500 epochs and reduced early‑stopping patience to 30 epochs. These small adjustments keep the model architecture and overall pipeline unchanged while encouraging a lower RMSE, bringing the score closer to the desired 0.30876.'
- What this solution (achieved 0.50751) has done: 'The update removes the overly‑restrictive 0.001–0.999 clamping in the autoencoder’s output (allowing true 0/1 values) and adds a tiny L2 weight‑decay to the Adam optimizer for a bit more regularisation. Both tweaks are tiny, keep the model architecture unchanged, and are aimed at nudging the validation RMSE closer to the target 0.30876.'
- What this solution (achieved 0.41596) has done: 'I adjust the training loss to plain MSE (which aligns better with the RMSE evaluation), raise the learning rate modestly, remove the tiny weight‑decay, and give the early‑stopping logic a bit more patience so the model can train longer. These small, targeted tweaks keep the architecture unchanged while encouraging a lower validation RMSE, moving the score toward the target.'

# 9. Code solution

## === cell 0
import os
import sys
import zipfile
import csv
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



## === cell 1
try:
    from torchinfo import summary
except ImportError:
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchinfo"])
    from torchinfo import summary




## === cell 2
base_input = "/kaggle/input/denoising-dirty-documents"

train_dir = os.path.join(base_input, "train")
train_cleaned_dir = os.path.join(base_input, "train_cleaned")
test_dir = os.path.join(base_input, "test")

for p in (train_dir, train_cleaned_dir, test_dir):
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Expected directory {p} does not exist.")




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
train_sizes = [img.shape[:2] for img in train_images]
train_cleaned_sizes = [img.shape[:2] for img in train_cleaned_images]
test_sizes = [img.shape[:2] for img in test_images]




## === cell 7
unique_train_sizes = np.unique(train_sizes, axis=0)
unique_train_cleaned_sizes = np.unique(train_cleaned_sizes, axis=0)
unique_test_sizes = np.unique(test_sizes, axis=0)




## === cell 8
print(f"train_images:\n {unique_train_sizes}")
print(f"train_cleaned_images:\n {unique_train_cleaned_sizes}")
print(f"test_images:\n {unique_test_sizes}")




## === cell 9
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




## === cell 10
class Grayscale:
    def __call__(self, img):
        pil_img = TF.to_pil_image(img) if isinstance(img, torch.Tensor) else img
        grayscale_img = pil_img.convert("L")
        return TF.to_tensor(grayscale_img)




## === cell 11
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




## === cell 12
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
        self.tensors = []
        for path in self.image_files:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            self.tensors.append(img)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx]




## === cell 13
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
    train_files, cleaned_files, test_size=0.1, random_state=42
)


class PairedImageDataset(Dataset):
    def __init__(self, train_files, cleaned_files, transform=None):
        self.transform = transform
        self.train_tensors = []
        self.cleaned_tensors = []
        for t_path, c_path in zip(train_files, cleaned_files):
            t_img = Image.open(t_path).convert("RGB")
            c_img = Image.open(c_path).convert("RGB")
            if self.transform:
                t_img = self.transform(t_img)
                c_img = self.transform(c_img)
            self.train_tensors.append(t_img)
            self.cleaned_tensors.append(c_img)

    def __len__(self):
        return len(self.train_tensors)

    def __getitem__(self, idx):
        return self.train_tensors[idx], self.cleaned_tensors[idx]




## === cell 14
train_dataset = PairedImageDataset(train_files, cleaned_train, train_transforms)
val_dataset = PairedImageDataset(val_files, cleaned_val, val_transforms)
test_dataset = ImageDataset(test_dir, test_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    pin_memory=True,
    num_workers=2,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    pin_memory=True,
    num_workers=2,
    persistent_workers=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    pin_memory=True,
    num_workers=2,
    persistent_workers=True,
)




## === cell 15
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




## === cell 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




## === cell 17
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
            nn.Dropout(p=0.0),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(16, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.0),
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
            nn.Dropout(p=0.0),
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
            nn.Dropout(p=0.0),
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
            nn.Dropout(p=0.0),
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
            nn.Dropout(p=0.0),
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

        return torch.clamp(dec1_out, min=0.0, max=1.0)


model = DenoisingAutoencoder().to(device)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


model.apply(init_weights)




## === cell 18
summary(model, input_size=(16, 1, 420, 540), device=device)




## === cell 19
class RMSELoss(nn.Module):
    def forward(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target))




## === cell 20
class HybridLoss(nn.Module):
    def __init__(self, lambda_rmse=1.0, lambda_l1=0.0):
        super().__init__()
        self.lambda_rmse = lambda_rmse
        self.lambda_l1 = lambda_l1
        self.rmse_loss = RMSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.lambda_rmse * self.rmse_loss(
            pred, target
        ) + self.lambda_l1 * self.l1_loss(pred, target)




## === cell 21
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)




## === cell 22
rmse_metric = RMSELoss()
epochs = 500  # keep the upper bound
patience = 70  # give the model more epochs before early stopping
early_stop_counter = 0
best_val_loss = float("inf")
best_model_state = None

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_rmse = 0.0
    for img, cleaned in train_loader:
        img, cleaned = img.to(device), cleaned.to(device)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, cleaned)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_rmse += rmse_metric(out, cleaned).item()
    train_loss /= len(train_loader)
    train_rmse /= len(train_loader)
    print(
        f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} RMSE: {train_rmse:.4f}"
    )

    model.eval()
    val_loss = 0.0
    val_rmse = 0.0
    with torch.no_grad():
        for img, cleaned in val_loader:
            img, cleaned = img.to(device), cleaned.to(device)
            out = model(img)
            loss = criterion(out, cleaned)
            val_loss += loss.item()
            val_rmse += rmse_metric(out, cleaned).item()
    val_loss /= len(val_loader)
    val_rmse /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f} RMSE: {val_rmse:.4f}")

    prev_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(val_loss)
    cur_lr = optimizer.param_groups[0]["lr"]
    if cur_lr != prev_lr:
        print(f"Learning Rate updated to {cur_lr:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        early_stop_counter = 0
        print("New best model saved.")
    else:
        early_stop_counter += 1
        print(f"No improvement. Early stop counter: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        print("Early stopping triggered.")
        break

if best_model_state is not None:
    torch.save(best_model_state, "best_model.pth")
    print(f"Best model checkpoint saved (val loss {best_val_loss:.4f}).")
else:
    print("No model was saved.")




## === cell 23
best_model_path = "best_model.pth"
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()




## === cell 24
all_outputs = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        preds = model(batch)  # (B,1,420,540)
        all_outputs.append(preds.cpu())
all_outputs = torch.cat(all_outputs, dim=0)  # (N,1,420,540)
print("All predictions shape:", all_outputs.shape)




## === cell 25
def compute_padding(orig_size, target_size):
    orig_h, orig_w = orig_size
    tar_h, tar_w = target_size
    pad_top = (tar_h - orig_h) // 2 if tar_h > orig_h else 0
    pad_bottom = tar_h - orig_h - pad_top if tar_h > orig_h else 0
    pad_left = (tar_w - orig_w) // 2 if tar_w > orig_w else 0
    pad_right = tar_w - orig_w - pad_left if tar_w > orig_w else 0
    return pad_top, pad_bottom, pad_left, pad_right


def remove_padding(pred, orig_size, target_size):
    orig_h, orig_w = orig_size
    pad_top, _, pad_left, _ = compute_padding(orig_size, target_size)
    return pred[:, pad_top : pad_top + orig_h, pad_left : pad_left + orig_w]




## === cell 26
test_file_paths = sorted(
    [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith(".png")
    ],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
)

rows = []
for idx, file_path in enumerate(test_file_paths):
    image_id = os.path.splitext(os.path.basename(file_path))[0]
    orig_w, orig_h = Image.open(file_path).convert("L").size
    orig_size = (orig_h, orig_w)

    pred_tensor = all_outputs[idx]  # (1,420,540)
    cropped = remove_padding(pred_tensor, orig_size, (420, 540))
    pred_np = cropped.squeeze(0).numpy()  # (h,w)

    rows_idx = np.arange(orig_h).reshape(-1, 1)
    cols_idx = np.arange(orig_w).reshape(1, -1)
    ids = np.core.defchararray.add(
        np.core.defchararray.add(
            np.core.defchararray.add(image_id + "_", (rows_idx + 1).astype(str)), "_"
        ),
        (cols_idx + 1).astype(str),
    )
    ids_flat = ids.ravel()
    values_flat = pred_np.ravel().astype(float)
    rows.extend(zip(ids_flat, values_flat))

submission_path = "submission.csv"
with open(submission_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "value"])
    writer.writerows(rows)

print(f"Submission file '{submission_path}' created with {len(rows)} rows.")
