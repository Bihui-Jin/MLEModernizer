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

0.30869

# 6. Current score

0.42652

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.39032) has done: 'Implemented fixes to eliminate syntax errors, replace shell‑style commands with pure Python (using `subprocess` for pip install and `zipfile` for extraction), and trimmed the training loop to a safe number of epochs to ensure the notebook finishes within the runtime limit. These changes allow the pipeline to run end‑to‑end and produce a correctly formatted `submission.csv` without crashes.'
- What this solution (achieved 0.415) has done: 'The script failed because it tried to unzip files that are already present in the Kaggle input directories, causing `FileNotFoundError`.  
I replaced the extraction step with logic that automatically finds the existing data root (checking the common Kaggle paths) and sets `data_root` accordingly. This lets all subsequent cells locate the train, test, and cleaned folders, allowing the model to train, generate predictions, and write a proper `submission.csv`.'
- What this solution (achieved 0.38572) has done: 'The fix adds the missing import for `summary`, introduces a small L1 component to the loss (which helps the model train more effectively), slightly reduces the optimizer’s weight decay, and allows a few more training epochs with a longer patience for early stopping. These minimal changes keep the original architecture intact while improving validation RMSE, moving the score closer to the target.'
- What this solution (achieved 0.42053) has done: 'The changes tighten regularization and let the model train a bit longer, which should lower the validation RMSE and bring the score nearer the target while keeping the original architecture unchanged.'
- What this solution (achieved 0.44193) has done: 'I slightly reduce the L1 regularization weight and the optimizer’s weight decay, which usually helps the model focus more on the primary RMSE loss and can improve validation RMSE, moving the score closer to the target.'
- What this solution (achieved 0.46077) has done: 'The changes lower the L1 regularization weight, remove weight decay, and allow more training epochs with a higher early‑stopping patience, which should reduce over‑regularization and let the model converge closer to the target RMSE.'
- What this solution (achieved 0.43833) has done: 'The changes add a small weight‑decay to the Adam optimizer, lower the L1 regularisation weight in the hybrid loss, and tighten the early‑stopping schedule (fewer epochs and a smaller patience). These tweaks keep the model architecture untouched while reducing over‑fitting, which should lower the validation RMSE and move the score closer to the target of 0.30869.'
- What this solution (achieved 0.43457) has done: 'I lower the L1 regularisation weight, raise the learning‑rate and remove weight‑decay, and give the model more epochs plus a longer early‑stopping patience. These small hyper‑parameter tweaks keep the exact architecture and training loop unchanged while allowing the network to converge to a lower validation RMSE, moving the score nearer the target 0.30869.'
- What this solution (achieved 0.41844) has done: 'The changes remove the unnecessary output clamping and drop the L1 component from the loss, aligning the training objective directly with the competition’s RMSE metric. This should lower validation error and move the score closer to the target while keeping the model architecture and training loop unchanged.'
- What this solution (achieved 0.44881) has done: 'I keep the overall architecture and training loop unchanged but reduce regularisation that harms reconstruction and give the model more time to learn. Specifically, I set all Dropout layers to 0 % (removing stochastic noise), add a tiny weight‑decay to the Adam optimizer, and extend the training budget (more epochs and a larger patience) so the early‑stopping can capture a better checkpoint. These minimal tweaks are expected to lower the validation RMSE and move the score from 0.418 → closer to the target 0.309 while still producing a correct `submission.csv`.'
- What this solution (achieved 0.42652) has done: 'The changes preload all training/validation images into memory, speed up data loading with multiple workers, enable CuDNN benchmarking, and set deterministic seeds—all without altering the model architecture, loss, or training schedule, preserving identical results while keeping the total runtime under the 600‑second limit.'

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
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)



## === cell 1
possible_roots = [
    "/kaggle/input/denoising-dirty-documents",
    "/kaggle/working/denoising-dirty-documents",
    "/kaggle/input/denoising-dirty-documents/denoising-dirty-documents",
]
data_root = None
for p in possible_roots:
    if os.path.isdir(p):
        data_root = p
        break
if data_root is None:
    raise FileNotFoundError("Unable to locate the dataset directory.")

for sub in ["train", "test", "train_cleaned"]:
    if not os.path.isdir(os.path.join(data_root, sub)):
        raise FileNotFoundError(
            f"Expected sub‑directory '{sub}' not found in {data_root}"
        )

print(f"Data root set to: {data_root}")



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
            img = cv2.imread(img_path)
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
    train_files, cleaned_files, test_size=2 / 9, random_state=42
)




## === cell 14
class PairedImageDataset(Dataset):
    def __init__(self, train_files, cleaned_files, transform=None):
        self.transform = transform
        self.train_files = train_files
        self.cleaned_files = cleaned_files

        self.train_tensors = []
        self.cleaned_tensors = []
        for t_path, c_path in zip(self.train_files, self.cleaned_files):
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




## === cell 15
train_dataset = PairedImageDataset(train_files, cleaned_train, train_transforms)
val_dataset = PairedImageDataset(val_files, cleaned_val, val_transforms)
test_dataset = ImageDataset(test_dir, test_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)



## === cell 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

torch.backends.cudnn.benchmark = True




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
            nn.Dropout(p=0.0),  # reduced from 0.1
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(16, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.0),  # reduced from 0.2
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.0),  # reduced from 0.3
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.0),  # reduced from 0.3
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
            nn.Dropout(p=0.0),  # reduced from 0.3
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
            nn.Dropout(p=0.0),  # reduced from 0.3
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
            nn.Dropout(p=0.0),  # reduced from 0.2
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
            nn.Dropout(p=0.0),  # reduced from 0.1
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

        outputs = dec1_out
        return outputs


model = DenoisingAutoencoder().to(device)



## === cell 18
from torchinfo import summary

summary(model, input_size=(16, 1, 420, 540), device=device)



## === cell 19
criterion = nn.MSELoss()



## === cell 20
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)



## === cell 21
epochs = 400
patience = 60  # early‑stopping patience
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
        train_rmse_loss += torch.sqrt(F.mse_loss(outputs, train_cleaned_images)).item()
    train_loss /= len(train_loader)
    train_rmse_loss /= len(train_loader)
    print(
        f"Epoch [{epoch+1}/{epochs}], Train Loss (MSE): {train_loss:.6f}, RMSE: {train_rmse_loss:.6f}"
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
            val_rmse_loss += torch.sqrt(F.mse_loss(outputs, val_cleaned_images)).item()
    val_loss /= len(val_loader)
    val_rmse_loss /= len(val_loader)

    prev_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]
    if current_lr != prev_lr:
        print(f"Learning Rate updated: {current_lr:.6f}")

    if val_loss < best_val_loss:
        print(
            f"New best val loss (MSE): {val_loss:.6f} (prev {best_val_loss:.6f}), RMSE: {val_rmse_loss:.6f}"
        )
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(
            f"Val loss did not improve. Early stop counter: {early_stop_counter}/{patience}"
        )

    if early_stop_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        model.load_state_dict(best_model_state)
        break

    epoch += 1

if best_model_state is not None:
    torch.save(best_model_state, "best_model.pth")
    print(f"Best model saved with val_loss = {best_val_loss:.6f}")
else:
    print("No best model saved.")



## === cell 22
best_model_path = "best_model.pth"
model.load_state_dict(torch.load(best_model_path))
model.eval()
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        break  # preview first batch only




## === cell 23
def visualize_images_and_outputs(images, outputs):
    num_images = images.size(0)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 3))
    for i in range(num_images):
        axes[i, 0].imshow(images[i].cpu().numpy().squeeze(), cmap="gray")
        axes[i, 0].set_title(f"Original {i+1}", fontsize=10)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(outputs[i].cpu().detach().numpy().squeeze(), cmap="gray")
        axes[i, 1].set_title(f"Output {i+1}", fontsize=10)
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.show()


visualize_images_and_outputs(images, outputs)



## === cell 24
print("Test directory:", test_dir)



## === cell 25
model.load_state_dict(torch.load(best_model_path))
model.eval()
all_outputs = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        outs = model(batch)  # (B, 1, 420, 540)
        all_outputs.append(outs.cpu())
all_outputs = torch.cat(all_outputs, dim=0)
print("All outputs shape:", all_outputs.shape)



## === cell 26
import csv
from PIL import Image

target_size = (420, 540)  # (height, width)


def compute_padding(orig_size, target_size):
    orig_h, orig_w = orig_size
    tgt_h, tgt_w = target_size
    pad_top = (tgt_h - orig_h) // 2 if tgt_h > orig_h else 0
    pad_left = (tgt_w - orig_w) // 2 if tgt_w > orig_w else 0
    return pad_top, pad_left


def remove_padding(pred_tensor, orig_size):
    orig_h, orig_w = orig_size
    pad_top, pad_left = compute_padding(orig_size, target_size)
    return pred_tensor[:, pad_top : pad_top + orig_h, pad_left : pad_left + orig_w]


test_file_paths = sorted(
    [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith(".png")
    ],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
)

submission_data = []

for idx, file_path in enumerate(test_file_paths):
    image_id = os.path.splitext(os.path.basename(file_path))[0]

    with Image.open(file_path) as orig_img:
        orig_w, orig_h = orig_img.size
    orig_size = (orig_h, orig_w)

    pred = all_outputs[idx]  # already on CPU
    cropped_pred = remove_padding(pred, orig_size)  # (1, orig_h, orig_w)
    pred_np = cropped_pred.squeeze(0).numpy()  # (orig_h, orig_w)

    pixel_vals = pred_np.ravel().tolist()

    ids = [
        f"{image_id}_{r}_{c}"
        for r in range(1, orig_h + 1)
        for c in range(1, orig_w + 1)
    ]

    submission_data.extend(zip(ids, pixel_vals))

submission_file = "submission.csv"
with open(submission_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "value"])
    writer.writerows(submission_data)

print(f"Submission file '{submission_file}' created with {len(submission_data)} rows.")
