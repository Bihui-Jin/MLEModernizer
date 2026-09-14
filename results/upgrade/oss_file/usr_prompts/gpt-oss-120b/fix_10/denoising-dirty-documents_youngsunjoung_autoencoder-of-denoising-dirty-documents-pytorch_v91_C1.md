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

0.26862

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.32179) has done: 'I make the data‑folder paths robust (so the loaders actually see the images) and write the submission file to the Kaggle working directory. This fixes the “no CSV generated” problem without touching the model or training logic, keeping the core pipeline unchanged.'
- What this solution (achieved 0.43258) has done: 'The changes lower the learning rate, reduce regularisation, give the RMSE component of the hybrid loss a higher weight, and allow a slightly longer early‑stopping patience. These tweaks keep the model architecture and training loop unchanged while nudging the validation RMSE down toward the target score.'
- What this solution (achieved 0.4407) has done: 'The changes lower the learning rate, add a tiny weight decay, give a slightly higher L1 contribution in the hybrid loss, and allow a longer early‑stopping patience. These adjustments keep the architecture and training loop unchanged while nudging validation RMSE toward the target score.'
- What this solution (achieved 0.42997) has done: 'The changes lower the learning rate, add a bit more weight‑decay, raise the RMSE contribution in the hybrid loss and give early‑stopping more patience. These tweaks keep the model and training loop unchanged while nudging validation RMSE down toward the target score.'
- What this solution (achieved 0.43979) has done: 'I slightly lower the learning rate, remove weight decay, and use a pure RMSE loss (λ_rmse = 1.0, λ_l1 = 0) while giving the early‑stopping patience a bit more room. These minimal adjustments keep the model architecture and training loop unchanged but should let the network converge to a lower validation RMSE, moving the score toward the target.'
- What this solution (achieved 0.47387) has done: 'I slightly adjust the training hyper‑parameters: add a small L‑1 component to the loss, lower the learning‑rate and use AdamW with a tiny weight decay, and give the early‑stopping and LR‑scheduler more patience so the model can converge further. These minimal changes keep the architecture and training loop unchanged while nudging the validation RMSE lower toward the target.'

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




## === cell 1
from torchinfo import summary




## === cell 2
possible_base_dirs = [
    "/kaggle/input/denoising-dirty-documents",
    "/kaggle/working/denoising-dirty-documents",
    "/content/denoising_data",
]

base_dir = next((p for p in possible_base_dirs if os.path.isdir(p)), None)
if base_dir is None:
    raise FileNotFoundError("Unable to locate the dataset directory.")

train_dir = os.path.join(base_dir, "train")
train_cleaned_dir = os.path.join(base_dir, "train_cleaned")
test_dir = os.path.join(base_dir, "test")

print(f"Using dataset root: {base_dir}")
print(f"train_dir: {train_dir}")
print(f"train_cleaned_dir: {train_cleaned_dir}")
print(f"test_dir: {test_dir}")




## === cell 3
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):  # PNG파일 로드
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)  # OpenCV로 이미지 읽기(BGR형식)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
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
        self.target_size = target_size  # (높이, 너비)

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
        grayscale_img = pil_img.convert("L")  # RGB -> Grayscale
        return TF.to_tensor(grayscale_img)




## === cell 11
train_transforms = v2.Compose(
    [
        v2.ToImage(),
        PadToSize((420, 540)),  # 크기 통일
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




## === cell 12
class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = sorted(
            [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".png")
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




## === cell 13
train_files = sorted(
    [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".png")]
)
cleaned_files = sorted(
    [
        os.path.join(train_cleaned_dir, f)
        for f in os.listdir(train_cleaned_dir)
        if f.endswith(".png")
    ]
)

train_files, val_files, cleaned_train, cleaned_val = train_test_split(
    train_files, cleaned_files, test_size=2 / 9, random_state=42
)




## === cell 14
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




## === cell 15
train_dataset = PairedImageDataset(train_files, cleaned_train, train_transforms)
val_dataset = PairedImageDataset(val_files, cleaned_val, val_transforms)
test_dataset = ImageDataset(test_dir, test_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)




## === cell 16
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




## === cell 17
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




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

        outputs = torch.clamp(dec1_out, min=0.001, max=0.999)
        return outputs


model = DenoisingAutoencoder().to(device)




## === cell 19
summary(model, input_size=(16, 1, 420, 540), device=device)




## === cell 20
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target))




## === cell 21
class HybridLoss(nn.Module):
    def __init__(
        self, lambda_rmse=1.0, lambda_l1=0.0
    ):  # pure RMSE focuses on the metric
        super(HybridLoss, self).__init__()
        self.lambda_rmse = lambda_rmse
        self.lambda_l1 = lambda_l1
        self.rmse_loss = RMSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.lambda_rmse * self.rmse_loss(
            pred, target
        ) + self.lambda_l1 * self.l1_loss(pred, target)




## === cell 22
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=6, verbose=True
)




## === cell 23
epochs = 1000
patience = 120  # give the model more epochs before early stopping
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
        train_rmse_loss += RMSELoss()(outputs, train_cleaned_images).item()
    train_loss /= len(train_loader)
    train_rmse_loss /= len(train_loader)
    print(
        f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, RMSE Score : {train_rmse_loss:.4f}"
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
            val_rmse_loss += RMSELoss()(outputs, val_cleaned_images).item()
    val_loss /= len(val_loader)
    val_rmse_loss /= len(val_loader)

    prev_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]
    if current_lr != prev_lr:
        print(f"Learning Rate updated: {current_lr:.6f}\\n")

    if val_loss < best_val_loss:
        print(
            f"New best validation loss: {val_loss:.4f} (Previous: {best_val_loss:.4f}), RMSE Score : {val_rmse_loss:.4f}"
        )
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(
            f"Validation loss increased! Early stopping counter: {early_stop_counter}/{patience}"
        )

    if early_stop_counter >= patience:
        print(f"Early stopping triggered! after {epoch+1} epochs")
        model.load_state_dict(best_model_state)
        print(f"Best model loaded with val_loss = {best_val_loss:.4f}")
        break
    epoch += 1

if best_model_state is not None:
    torch.save(best_model_state, "best_model.pth")
    print(f"Best model saved with val_loss = {best_val_loss:.4f}")
else:
    print("No best model was saved.")




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3727333865.py in <cell line: 0>()
     16         optimizer.zero_grad()
     17         outputs = model(train_images)
---> 18         loss = criterion(outputs, train_cleaned_images)
     19         loss.backward()
     20         optimizer.step()

NameError: name 'criterion' is not defined

## === cell 24
best_model_path = "best_model.pth"
model.load_state_dict(torch.load(best_model_path))
model.eval()
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        break  # verify loading; full inference later




## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3292436470.py in <cell line: 0>()
      1 best_model_path = "best_model.pth"
----> 2 model.load_state_dict(torch.load(best_model_path))
      3 model.eval()
      4 with torch.no_grad():
      5     for images in test_loader:

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

## === cell 25
def visualize_images_and_outputs(images, outputs):
    num_images = images.size(0)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 3))
    for i in range(num_images):
        axes[i, 0].imshow(images[i].cpu().numpy().squeeze(), cmap="gray")
        axes[i, 0].set_title(f"Original {i + 1}", fontsize=10)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(outputs[i].cpu().detach().numpy().squeeze(), cmap="gray")
        axes[i, 1].set_title(f"Output {i + 1}", fontsize=10)
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.show()


visualize_images_and_outputs(images, outputs)




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3198286226.py in <cell line: 0>()
     13 
     14 
---> 15 visualize_images_and_outputs(images, outputs)
     16 
     17 

NameError: name 'images' is not defined

## === cell 26
print(f"Test directory: {test_dir}")




## === cell 27
best_model_path = "best_model.pth"
model.load_state_dict(torch.load(best_model_path))
model.eval()
all_outputs = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        outputs = model(batch)
        all_outputs.append(outputs.cpu())
all_outputs = torch.cat(all_outputs, dim=0)
print("all_outputs shape:", all_outputs.shape)




## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1263734711.py in <cell line: 0>()
      1 best_model_path = "best_model.pth"
----> 2 model.load_state_dict(torch.load(best_model_path))
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

## === cell 28
import csv
import torchvision.transforms.functional as TF

target_size = (420, 540)  # (height, width)


def compute_padding(orig_size, target_size):
    orig_h, orig_w = orig_size
    tgt_h, tgt_w = target_size
    pad_top = (tgt_h - orig_h) // 2 if tgt_h > orig_h else 0
    pad_bottom = (tgt_h - orig_h - pad_top) if tgt_h > orig_h else 0
    pad_left = (tgt_w - orig_w) // 2 if tgt_w > orig_w else 0
    pad_right = (tgt_w - orig_w - pad_left) if tgt_w > orig_w else 0
    return pad_top, pad_bottom, pad_left, pad_right


def remove_padding(pred, orig_size, target_size):
    orig_h, orig_w = orig_size
    pad_top, _, pad_left, _ = compute_padding(orig_size, target_size)
    return pred[:, pad_top : pad_top + orig_h, pad_left : pad_left + orig_w]


test_file_paths = sorted(
    [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".png")],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
)

submission_data = []
for idx, file_path in enumerate(test_file_paths):
    image_id = os.path.splitext(os.path.basename(file_path))[0]
    orig_img = Image.open(file_path).convert("L")
    orig_width, orig_height = orig_img.size
    orig_size = (orig_height, orig_width)
    print("Processing image_id:", image_id, "orig_size:", orig_size)

    pred = all_outputs[idx]  # (1, 420, 540)
    cropped_pred = remove_padding(pred, orig_size, target_size)
    print("Cropped shape:", cropped_pred.shape)

    pred_np = cropped_pred.squeeze(0).cpu().numpy()
    for row in range(orig_height):
        for col in range(orig_width):
            pixel_id = f"{image_id}_{row+1}_{col+1}"
            pixel_value = float(pred_np[row, col])
            submission_data.append((pixel_id, pixel_value))

submission_file = "/kaggle/working/submission.csv"
with open(submission_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "value"])
    writer.writerows(submission_data)

print(f"Submission file '{submission_file}' created.")

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3130303485.py in <cell line: 0>()
     34     print("Processing image_id:", image_id, "orig_size:", orig_size)
     35 
---> 36     pred = all_outputs[idx]  # (1, 420, 540)
     37     cropped_pred = remove_padding(pred, orig_size, target_size)
     38     print("Cropped shape:", cropped_pred.shape)

NameError: name 'all_outputs' is not defined
