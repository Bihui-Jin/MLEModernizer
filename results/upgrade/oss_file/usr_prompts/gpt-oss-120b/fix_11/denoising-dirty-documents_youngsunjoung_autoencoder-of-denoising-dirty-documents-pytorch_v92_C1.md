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

0.28044

# 6. Current score

0.47252

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.35944) has done: 'The changes fix syntax errors, replace shell commands with Python‑based unzipping, ensure correct paths, and clean up the submission loop so it writes a proper CSV for all test images. These fixes make the notebook run end‑to‑end and produce a valid `submission.csv` while keeping the original model and training logic unchanged.'
- What this solution (achieved 0.42839) has done: 'I lower the optimizer learning rate to 1e‑3 and give the hybrid loss more emphasis on RMSE (λ₁=0.9, λ₂=0.1). These minimal tweaks keep the architecture and training loop unchanged while encouraging finer convergence on the validation set, which should reduce the RMSE toward the target score.'
- What this solution (achieved 0.43654) has done: 'I adjust the model’s output clamping to the true [0, 1] range, switch the hybrid loss to pure RMSE (λ₁=1.0, λ₂=0.0), lower the optimizer learning rate to 5e‑4, and give the training loop more epochs and patience so the model can converge further. These minimal hyper‑parameter tweaks keep the architecture unchanged while encouraging finer learning, which should reduce the RMSE toward the target score.'
- What this solution (achieved 0.4147) has done: 'I remove the stochastic augmentations from the training transform (they can hinder precise denoising) and lower the optimizer learning rate slightly while allowing a few more epochs. These small hyper‑parameter tweaks keep the model architecture unchanged but give the network a clearer signal, which should lower the RMSE toward the target.'
- What this solution (achieved 0.49009) has done: 'I lower the optimizer learning rate to 1e‑4 (with a slightly reduced weight decay) and allow the training loop to run longer with more early‑stopping patience (200 epochs, patience 20). These modest hyper‑parameter tweaks keep the model architecture and loss unchanged while giving the network extra time to converge, which should reduce the validation RMSE and move the score closer to the target.'
- What this solution (achieved 0.42843) has done: 'I raise the learning rate to 5e‑4 and remove weight decay (they were overly restrictive and caused under‑fitting, leading to the high RMSE). I also add a small L1 component (λ_rmse = 0.9, λ_l1 = 0.1) to the hybrid loss, which helps the model converge to sharper pixel‑wise predictions without changing the architecture or training loop.'
- What this solution (achieved 0.44569) has done: 'We tighten the training objective by using pure RMSE (remove the L1 term) and lower the learning rate for finer convergence. Additionally, we give the model a few more epochs (250) and a slightly longer early‑stopping patience (30) so it can keep improving without altering the network architecture or overall workflow. These modest hyper‑parameter tweaks are expected to lower the validation RMSE and move the score closer to the target.'
- What this solution (achieved 0.42262) has done: 'I add a tiny L1 component to the loss (λ_rmse = 0.95, λ_l1 = 0.05) and give the model a few more training epochs (300) with a slightly longer early‑stopping patience (40). These modest hyper‑parameter tweaks keep the architecture unchanged while encouraging a bit sharper pixel‑wise learning and allowing the optimizer more time to converge, which should lower the validation RMSE and move the score toward the target.'
- What this solution (achieved 0.47252) has done: 'I set a fixed random seed for reproducibility, switch the loss to pure RMSE (removing the L 1 component), use a slightly lower learning rate with AdamW (no weight decay), and give the model a few more training epochs with a longer early‑stopping patience. These modest hyper‑parameter tweaks keep the architecture and training loop unchanged while allowing finer convergence, which should move the RMSE closer to the target.'

# 9. Code solution

## === cell 0
import os
import zipfile
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
np.random.seed(42)



## === cell 1
try:
    from torchinfo import summary
except Exception:
    summary = None



## === cell 2
BASE_INPUT = "/kaggle/input/denoising-dirty-documents"
DATA_ROOT = "./denoising_data"
os.makedirs(DATA_ROOT, exist_ok=True)

zip_paths = [
    os.path.join(BASE_INPUT, "train.zip"),
    os.path.join(BASE_INPUT, "test.zip"),
    os.path.join(BASE_INPUT, "train_cleaned.zip"),
]

for zp in zip_paths:
    with zipfile.ZipFile(zp, "r") as zf:
        zf.extractall(DATA_ROOT)



## === cell 3
train_dir = os.path.join(DATA_ROOT, "train")
train_cleaned_dir = os.path.join(DATA_ROOT, "train_cleaned")
test_dir = os.path.join(DATA_ROOT, "test")




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
print("First train image shape:", train_images[0].shape)



## === cell 8
print("First cleaned image shape:", train_cleaned_images[0].shape)



## === cell 9
print("First test image shape:", test_images[0].shape)



## === cell 10
train_sizes = [img.shape[:2] for img in train_images]
train_cleaned_sizes = [img.shape[:2] for img in train_cleaned_images]
test_sizes = [img.shape[:2] for img in test_images]



## === cell 11
unique_train_sizes = np.unique(train_sizes, axis=0)
unique_train_cleaned_sizes = np.unique(train_cleaned_sizes, axis=0)
unique_test_sizes = np.unique(test_sizes, axis=0)



## === cell 12
print(f"train_images unique sizes:\n {unique_train_sizes}")
print(f"train_cleaned_images unique sizes:\n {unique_train_cleaned_sizes}")
print(f"test_images unique sizes:\n {unique_test_sizes}")




## === cell 13
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




## === cell 14
class Grayscale:
    def __call__(self, img):
        pil_img = TF.to_pil_image(img) if isinstance(img, torch.Tensor) else img
        grayscale_img = pil_img.convert("L")
        return TF.to_tensor(grayscale_img)




## === cell 15
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




## === cell 16
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




## === cell 17
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

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)




## === cell 20
def visualize_paired_dataset(paired_loader, num_images=5):
    for train_imgs, cleaned_imgs in paired_loader:
        fig, axes = plt.subplots(num_images, 2, figsize=(8, num_images * 3))
        for i in range(num_images):
            axes[i, 0].imshow(train_imgs[i].permute(1, 2, 0).cpu().numpy(), cmap="gray")
            axes[i, 0].set_title(f"Original {i+1}")
            axes[i, 0].axis("off")
            axes[i, 1].imshow(
                cleaned_imgs[i].permute(1, 2, 0).cpu().numpy(), cmap="gray"
            )
            axes[i, 1].set_title(f"Cleaned {i+1}")
            axes[i, 1].axis("off")
        plt.tight_layout()
        plt.show()
        break




## === cell 21
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## === cell 22
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



## === cell 23
if summary is not None:
    summary(model, input_size=(16, 1, 420, 540), device=device)




## === cell 24
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target))




## === cell 25
class HybridLoss(nn.Module):
    def __init__(self, lambda_rmse=1.0, lambda_l1=0.0):
        super(HybridLoss, self).__init__()
        self.lambda_rmse = lambda_rmse
        self.lambda_l1 = lambda_l1
        self.rmse_loss = RMSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.lambda_rmse * self.rmse_loss(
            pred, target
        ) + self.lambda_l1 * self.l1_loss(pred, target)


criterion = HybridLoss(lambda_rmse=1.0, lambda_l1=0.0)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)



## === cell 26
epochs = 400  # a bit more iterations for finer learning
patience = 60  # longer patience for early stopping
early_stop_counter = 0
best_val_loss = float("inf")
best_model_state = None

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_rmse = 0.0
    for train_imgs, clean_imgs in train_loader:
        train_imgs = train_imgs.to(device)
        clean_imgs = clean_imgs.to(device)
        optimizer.zero_grad()
        outs = model(train_imgs)
        loss = criterion(outs, clean_imgs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_rmse += RMSELoss()(outs, clean_imgs).item()
    train_loss /= len(train_loader)
    train_rmse /= len(train_loader)
    print(
        f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} RMSE: {train_rmse:.4f}"
    )

    model.eval()
    val_loss = 0.0
    val_rmse = 0.0
    with torch.no_grad():
        for val_imgs, val_clean in val_loader:
            val_imgs = val_imgs.to(device)
            val_clean = val_clean.to(device)
            outs = model(val_imgs)
            loss = criterion(outs, val_clean)
            val_loss += loss.item()
            val_rmse += RMSELoss()(outs, val_clean).item()
    val_loss /= len(val_loader)
    val_rmse /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f} RMSE: {val_rmse:.4f}")

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        early_stop_counter = 0
        print("  New best model saved.")
    else:
        early_stop_counter += 1
        print(f"  No improvement. Early stop counter: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        print("Early stopping triggered.")
        break

if best_model_state is not None:
    torch.save(best_model_state, "best_model.pth")
    print("Best model checkpoint saved.")
else:
    print("No best model to save.")



## === cell 27
best_model_path = "best_model.pth"
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        _ = model(batch)
        break




## === cell 28
def visualize_images_and_outputs(images, outputs):
    num = images.size(0)
    fig, axes = plt.subplots(num, 2, figsize=(10, num * 3))
    for i in range(num):
        axes[i, 0].imshow(images[i].cpu().numpy().squeeze(), cmap="gray")
        axes[i, 0].set_title(f"Original {i+1}")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(outputs[i].cpu().detach().numpy().squeeze(), cmap="gray")
        axes[i, 1].set_title(f"Output {i+1}")
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.show()


with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        preds = model(batch)
        visualize_images_and_outputs(batch, preds)
        break



## === cell 29
print("Test directory:", test_dir)



## === cell 30
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()
all_outputs = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        outs = model(batch)  # (B,1,420,540)
        all_outputs.append(outs.cpu())
all_outputs = torch.cat(all_outputs, dim=0)
print("Aggregated outputs shape:", all_outputs.shape)



## === cell 31
import csv

target_size = (420, 540)  # (height, width)


def compute_padding(orig_size, target_size):
    orig_h, orig_w = orig_size
    tar_h, tar_w = target_size
    pad_top = (tar_h - orig_h) // 2 if tar_h > orig_h else 0
    pad_left = (tar_w - orig_w) // 2 if tar_w > orig_w else 0
    return pad_top, pad_left


def remove_padding(pred, orig_size, target_size):
    orig_h, orig_w = orig_size
    pad_top, pad_left = compute_padding(orig_size, target_size)
    return pred[:, pad_top : pad_top + orig_h, pad_left : pad_left + orig_w]


test_file_paths = sorted(
    [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith(".png")
    ],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
)

assert (
    len(test_file_paths) == all_outputs.shape[0]
), f"Mismatch: {len(test_file_paths)} files vs {all_outputs.shape[0]} predictions"

submission_rows = []

for idx, path in enumerate(test_file_paths):
    image_id = os.path.splitext(os.path.basename(path))[0]
    pil_img = Image.open(path).convert("L")
    orig_w, orig_h = pil_img.size
    orig_size = (orig_h, orig_w)  # (height, width)

    pred = all_outputs[idx]  # (1,420,540)
    cropped = remove_padding(pred, orig_size, target_size)  # (1,orig_h,orig_w)
    pixel_array = cropped.squeeze(0).numpy()

    for row in range(orig_h):
        for col in range(orig_w):
            pid = f"{image_id}_{row+1}_{col+1}"
            submission_rows.append((pid, float(pixel_array[row, col])))

submission_path = "/kaggle/working/submission.csv"
with open(submission_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "value"])
    writer.writerows(submission_rows)

print(f"Submission file created at: {submission_path}")
