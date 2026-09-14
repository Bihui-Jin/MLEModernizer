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

0.31506

# 6. Current score

0.47217

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.37626) has done: 'Implemented fixes to eliminate syntax errors, replace notebook‑style `!` commands with standard Python unzip logic, and ensure the submission file is correctly generated. Added a safe import for torchinfo (and optional install), turned the introductory text into a comment, removed stray markdown delimiters, and kept the original model‑training pipeline unchanged. The script now runs end‑to‑end, creates a valid `submission.csv`, and retains the intended denoising architecture.'
- What this solution (achieved 0.43222) has done: 'I slightly lower the optimizer learning rate and give the early‑stopping logic a larger patience window so the model can train a bit longer and converge to a lower RMSE, moving the score from ~0.376 toward the target ~0.315. These changes keep the overall architecture and training pipeline intact.'
- What this solution (achieved 0.45487) has done: 'I lower the learning rate, remove the auxiliary L1 term from the hybrid loss, and give the training loop more epochs and a larger early‑stopping patience so the model can converge closer to the target RMSE without altering the architecture or core pipeline.'
- What this solution (achieved 0.40936) has done: 'The changes remove unnecessary data augmentations that can hurt denoising performance, lessen weight decay for a gentler regularisation, and give early‑stopping more patience so the model can train longer and converge to a lower RMSE, moving the score closer to the target.'
- What this solution (achieved 0.41278) has done: 'The changes lower the optimizer learning rate and reduce weight‑decay to let the model converge more gently, and increase the early‑stopping patience so training can run longer when improvement is slow. These adjustments are expected to reduce the validation RMSE, moving the score closer to the target 0.31506 while keeping the original architecture and pipeline unchanged.'
- What this solution (achieved 0.48276) has done: 'I modestly adjust the training hyper‑parameters so the model can converge a little further without changing its architecture or overall pipeline.  
- Lower the Adam learning rate to 5e‑5 and remove weight decay, giving the optimizer finer steps.  
- Let the `ReduceLROnPlateau` scheduler be a bit less eager (patience = 5).  
- Add a tiny L1 component (λₗ₁ = 0.05) to the hybrid loss; this regularises the output slightly and often improves RMSE while keeping the loss‑driven early‑stopping logic unchanged.  

These targeted tweaks preserve the core logic and should move the validation RMSE closer to the target 0.31506.'
- What this solution (achieved 0.42126) has done: 'To lower the validation RMSE and move the score toward the target, we slightly increase the learning rate (allowing faster convergence) and remove the auxiliary L1 term, which lets the model focus on the primary RMSE objective. We also give the learning‑rate scheduler a bit more patience (patience = 10) so it can react less aggressively to temporary plateaus. These minimal hyper‑parameter tweaks keep the original architecture and training loop intact while encouraging a lower RMSE.'
- What this solution (achieved 0.47122) has done: 'I slightly lower the Adam learning rate to let the model converge more gently, add a tiny L1 regularisation term (λₗ₁ = 0.05) to encourage smoother predictions, give the learning‑rate scheduler a bit more patience, and extend the early‑stopping patience. These minimal tweaks keep the architecture unchanged while aiming to reduce the validation RMSE toward the target.'
- What this solution (achieved 0.48463) has done: 'The fix adds a missing `HybridLoss` implementation, safeguards model checkpoint loading, and ensures a checkpoint is always saved so the pipeline can continue to inference and generate a proper CSV submission. No core model architecture or training logic is changed, keeping the original approach intact while allowing the script to run end‑to‑end and move the validation RMSE toward the target.'
- What this solution (achieved 0.47217) has done: 'I tighten the training hyper‑parameters so the model can converge a bit better without changing its architecture.  
- Use a slightly larger learning rate (1e‑4) and a tiny weight‑decay (1e‑5) for smoother updates.  
- Add a small L1 regularisation (λ = 0.05) which often helps denoising models reduce RMSE.  
- Give the learning‑rate scheduler a bit more patience (20) and extend early‑stopping patience (150) so training can continue long enough to benefit from the new settings.  

These minimal tweaks keep the core pipeline intact while aiming to lower the validation RMSE toward the target.'
- What this solution (achieved 0.4283) has done: 'The fix adds a missing `HybridLoss` implementation (combining RMSE and optional L1 regularisation) and ensures the optimizer, scheduler and loss are defined before training starts. The training loop’s epoch limit is reduced to keep runtime reasonable while still allowing the model to learn, which should lower the validation RMSE toward the target.'
- What this solution (achieved 0.47217) has done: 'I slightly lower the Adam learning rate, remove the L1 regularisation term, and set weight‑decay to zero, which lets the model optimise the pure RMSE loss more precisely. I also increase the maximum epoch count and early‑stop patience so training can continue until validation loss stabilises, and make the LR‑scheduler a bit less eager. These minimal hyper‑parameter tweaks keep the architecture unchanged while aiming to lower the validation RMSE toward the target.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
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




## === cell 1
try:
    from torchinfo import summary
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchinfo"])
    from torchinfo import summary




## === cell 2
def unzip_to(src_path, dst_dir):
    with zipfile.ZipFile(src_path, "r") as zip_ref:
        zip_ref.extractall(dst_dir)


base_input = "/kaggle/input/denoising-dirty-documents"
dst_root = "/content/denoising_data"
os.makedirs(dst_root, exist_ok=True)

unzip_to(os.path.join(base_input, "train.zip"), dst_root)
unzip_to(os.path.join(base_input, "test.zip"), dst_root)
unzip_to(os.path.join(base_input, "train_cleaned.zip"), dst_root)




## === cell 3
train_dir = "/content/denoising_data/train"
train_cleaned_dir = "/content/denoising_data/train_cleaned"
test_dir = "/content/denoising_data/test"




## === cell 4
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
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




## === cell 8
unique_train_sizes = np.unique(train_sizes, axis=0)
unique_train_cleaned_sizes = np.unique(train_cleaned_sizes, axis=0)
unique_test_sizes = np.unique(test_sizes, axis=0)




## === cell 9
print(f"train_images:\n {unique_train_sizes}")
print(f"train_cleaned_images:\n {unique_train_cleaned_sizes}")
print(f"test_images:\n {unique_test_sizes}")




## === cell 10
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




## === cell 11
class Grayscale:
    def __call__(self, img):
        pil_img = TF.to_pil_image(img) if isinstance(img, torch.Tensor) else img
        grayscale_img = pil_img.convert("L")
        return TF.to_tensor(grayscale_img)




## === cell 12
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




## === cell 13
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




## === cell 14
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




## === cell 15
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




## === cell 16
train_dataset = PairedImageDataset(train_files, cleaned_train, train_transforms)
val_dataset = PairedImageDataset(val_files, cleaned_val, val_transforms)
test_dataset = ImageDataset(test_dir, test_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)




## === cell 17
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




## === cell 18
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

        return torch.clamp(dec1_out, min=0.001, max=0.999)


model = DenoisingAutoencoder().to(device)




## === cell 19
summary(model, input_size=(16, 1, 420, 540), device=device)




## === cell 20
class RMSELoss(nn.Module):
    def forward(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target))




## === cell 21
criterion = HybridLoss(lambda_l1=0.0)  # pure RMSE loss
optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=30,  # slightly larger patience
)




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/4294443001.py in <cell line: 0>()
      3 # • Use a smaller learning rate (5e-5) and no weight decay for finer convergence.
      4 # • Make the LR scheduler a bit less eager (patience=30).
----> 5 criterion = HybridLoss(lambda_l1=0.0)  # pure RMSE loss
      6 optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=0.0)
      7 scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

NameError: name 'HybridLoss' is not defined

## === cell 22
epochs = 300
patience = 250  # early‑stop patience increased
early_stop_counter = 0
best_val_loss = float("inf")
best_model_state = None

epoch = 0
while epoch < epochs:
    model.train()
    train_loss = 0.0
    train_rmse = 0.0
    for train_imgs, train_cleaned_imgs in train_loader:
        train_imgs = train_imgs.to(device)
        train_cleaned_imgs = train_cleaned_imgs.to(device)

        optimizer.zero_grad()
        outputs = model(train_imgs)
        loss = criterion(outputs, train_cleaned_imgs)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_rmse += RMSELoss()(outputs, train_cleaned_imgs).item()

    train_loss /= len(train_loader)
    train_rmse /= len(train_loader)
    print(
        f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} RMSE: {train_rmse:.4f}"
    )

    model.eval()
    val_loss = 0.0
    val_rmse = 0.0
    with torch.no_grad():
        for val_imgs, val_cleaned_imgs in val_loader:
            val_imgs = val_imgs.to(device)
            val_cleaned_imgs = val_cleaned_imgs.to(device)
            outputs = model(val_imgs)
            loss = criterion(outputs, val_cleaned_imgs)
            val_loss += loss.item()
            val_rmse += RMSELoss()(outputs, val_cleaned_imgs).item()

    val_loss /= len(val_loader)
    val_rmse /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f} RMSE: {val_rmse:.4f}")

    prev_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(val_loss)
    cur_lr = optimizer.param_groups[0]["lr"]

    if cur_lr != prev_lr:
        print(f"Learning rate changed to {cur_lr:.6f}")
        early_stop_counter = max(early_stop_counter - 1, 0)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        early_stop_counter = 0
        print(f"New best model saved (val loss {best_val_loss:.4f})")
    else:
        early_stop_counter += 1
        print(f"No improvement. Early‑stop counter: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        print("Early stopping triggered.")
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        break

    epoch += 1

if best_model_state is None:
    best_model_state = model.state_dict()
torch.save(best_model_state, "best_model.pth")
print(f"Best model checkpoint saved (val loss {best_val_loss:.4f})")




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/27435706.py in <cell line: 0>()
     15         train_cleaned_imgs = train_cleaned_imgs.to(device)
     16 
---> 17         optimizer.zero_grad()
     18         outputs = model(train_imgs)
     19         loss = criterion(outputs, train_cleaned_imgs)

NameError: name 'optimizer' is not defined

## === cell 23
if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
else:
    print("Warning: best_model.pth not found; using current model weights.")
model.eval()




## === cell 24
all_outputs = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        preds = model(batch)  # (B, 1, 420, 540)
        all_outputs.append(preds.cpu())
all_outputs = torch.cat(all_outputs, dim=0)  # (N_test, 1, 420, 540)
print("All predictions shape:", all_outputs.shape)




## === cell 25
import csv

target_size = (420, 540)  # (height, width)


def compute_padding(orig_size, target_size):
    orig_h, orig_w = orig_size
    tar_h, tar_w = target_size
    pad_top = (tar_h - orig_h) // 2 if tar_h > orig_h else 0
    pad_left = (tar_w - orig_w) // 2 if tar_w > orig_w else 0
    return pad_top, pad_left


def remove_padding(pred, orig_size):
    """
    pred: Tensor (1, target_h, target_w)
    orig_size: (orig_h, orig_w)
    """
    orig_h, orig_w = orig_size
    pad_top, pad_left = compute_padding(orig_size, target_size)
    return pred[:, pad_top : pad_top + orig_h, pad_left : pad_left + orig_w]


test_file_paths = sorted(
    [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith(".png")
    ],
    key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
)

submission_rows = []

for idx, file_path in enumerate(test_file_paths):
    img_id = os.path.splitext(os.path.basename(file_path))[0]
    orig_w, orig_h = Image.open(file_path).convert("L").size
    orig_size = (orig_h, orig_w)

    pred_tensor = all_outputs[idx]  # (1, 420, 540)
    cropped = remove_padding(pred_tensor, orig_size)  # (1, orig_h, orig_w)
    pixel_array = cropped.squeeze(0).numpy()  # (orig_h, orig_w)

    for r in range(orig_h):
        for c in range(orig_w):
            pixel_id = f"{img_id}_{r+1}_{c+1}"
            pixel_val = float(pixel_array[r, c])
            submission_rows.append((pixel_id, pixel_val))

submission_path = "submission.csv"
with open(submission_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "value"])
    writer.writerows(submission_rows)

print(f"Submission file written to {submission_path}")
