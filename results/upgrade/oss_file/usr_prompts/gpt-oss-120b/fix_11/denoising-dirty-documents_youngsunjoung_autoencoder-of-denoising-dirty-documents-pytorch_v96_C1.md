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

0.30468

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.44476) has done: 'The changes remove the non‑code header that caused a syntax error, replace the notebook‑style shell commands with Python zipfile extraction, drop stray markdown backticks, and slightly shorten training (5 epochs) to ensure the script finishes within the runtime limits while keeping the original model and loss exactly the same. These fixes let the pipeline run end‑to‑end and produce a valid `submission.csv` file.'
- What this solution (achieved 0.37547) has done: 'I lower the learning rate and train for a few more epochs (while keeping the same model and loss) so the network can converge better without changing its architecture. This modest increase in training time should reduce the validation RMSE and move the score closer to the target, while still producing a valid `submission.csv` file.'
- What this solution (achieved 0.45098) has done: 'I lower the learning rate to 2e‑3 so the model can converge more smoothly and extend the maximum epochs to 15 (keeping the same early‑stopping logic). These small adjustments should modestly improve validation RMSE, moving the score closer to the target without altering the core architecture or loss.'
- What this solution (achieved 0.4397) has done: 'I slightly lower the optimizer learning rate, raise the number of training epochs, extend the early‑stopping patience, and give the loss function a higher weight on the RMSE component (the competition metric). These modest adjustments keep the model architecture unchanged but should let the network converge better toward the target RMSE, moving the score closer to 0.30468.'
- What this solution (achieved 0.43269) has done: 'I narrow the loss focus to pure RMSE (the competition metric) by setting the L1 weight to 0 and slightly lower the Adam learning rate to 5e‑4 so training can converge more smoothly without altering the model architecture. These minimal tweaks keep the original pipeline intact while encouraging a lower validation RMSE, moving the score toward the target.'
- What this solution (achieved 0.45533) has done: 'I lower regularization that may be hurting performance (remove weight decay and dropout), increase the batch size for more stable gradients, and give the early‑stopping logic a bit more patience so the model can train longer if it improves. These small tweaks keep the same architecture and loss while aiming to reduce the validation RMSE closer to the target.'

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
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
import torch.nn.functional as F




## === cell 1
try:
    from torchinfo import summary
except ImportError:
    summary = None




## === cell 2
data_root = "/kaggle/input/denoising-dirty-documents"
extract_dir = "/content/denoising_data"
os.makedirs(extract_dir, exist_ok=True)

for zip_name in ["train.zip", "test.zip", "train_cleaned.zip"]:
    zip_path = os.path.join(data_root, zip_name)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)




## === cell 3
train_dir = os.path.join(extract_dir, "train")
train_cleaned_dir = os.path.join(extract_dir, "train_cleaned")
test_dir = os.path.join(extract_dir, "test")




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
print(f"train_images unique sizes:\n {unique_train_sizes}")
print(f"train_cleaned_images unique sizes:\n {unique_train_cleaned_sizes}")
print(f"test_images unique sizes:\n {unique_test_sizes}")




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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)




## === cell 17
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## === cell 18
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
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(8, 16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.0),  # reduced dropout
            nn.MaxPool2d(2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(16, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.0),  # reduced dropout
            nn.MaxPool2d(2),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.0),  # reduced dropout
            nn.MaxPool2d(2),
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.0),  # reduced dropout
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
            nn.Dropout(0.0),  # reduced dropout
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
            nn.Dropout(0.0),  # reduced dropout
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
            nn.Dropout(0.0),  # reduced dropout
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=1, bias=False),
            nn.ConvTransposed(
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
            nn.Dropout(0.0),  # reduced dropout
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




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3320390893.py in <cell line: 0>()
    164 
    165 
--> 166 model = DenoisingAutoencoder().to(device)
    167 
    168 

/tmp/ipykernel_55/3320390893.py in __init__(self)
     93         self.dec2 = nn.Sequential(
     94             nn.ConvTranspose2d(16, 8, kernel_size=1, bias=False),
---> 95             nn.ConvTransposed(
     96                 8,
     97                 8,

AttributeError: module 'torch.nn' has no attribute 'ConvTransposed'

## === cell 19
if summary is not None:
    summary(model, input_size=(4, 1, 420, 540), device=device)




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4153699746.py in <cell line: 0>()
      1 if summary is not None:
----> 2     summary(model, input_size=(4, 1, 420, 540), device=device)
      3 
      4 

NameError: name 'model' is not defined

## === cell 20
class RMSELoss(nn.Module):
    def forward(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target))




## === cell 21
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




## === cell 22
criterion = HybridLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2389053917.py in <cell line: 0>()
      1 criterion = HybridLoss()
----> 2 optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=0)
      3 scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      4     optimizer, mode="min", factor=0.5, patience=3
      5 )

NameError: name 'model' is not defined

## === cell 23
epochs = 60
patience = 10  # retain original patience
early_stop_counter = 0
best_val_loss = float("inf")
best_model_state = None

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for train_imgs, clean_imgs in train_loader:
        train_imgs = train_imgs.to(device)
        clean_imgs = clean_imgs.to(device)
        optimizer.zero_grad()
        outputs = model(train_imgs)
        loss = criterion(outputs, clean_imgs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} – Train loss: {train_loss:.4f}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_imgs, val_clean_imgs in val_loader:
            val_imgs = val_imgs.to(device)
            val_clean_imgs = val_clean_imgs.to(device)
            outputs = model(val_imgs)
            loss = criterion(outputs, val_clean_imgs)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f"               – Val loss: {val_loss:.4f}")

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        early_stop_counter = 0
        print("   New best model saved.")
    else:
        early_stop_counter += 1
        print(
            f"   No improvement – early stop counter: {early_stop_counter}/{patience}"
        )

    if early_stop_counter >= patience:
        print("Early stopping triggered.")
        break

if best_model_state is not None:
    torch.save(best_model_state, "best_model.pth")
    print("Best model checkpoint saved.")
else:
    print("No model checkpoint to save.")




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1959317059.py in <cell line: 0>()
      7 
      8 for epoch in range(epochs):
----> 9     model.train()
     10     train_loss = 0.0
     11     for train_imgs, clean_imgs in train_loader:

NameError: name 'model' is not defined

## === cell 24
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()




## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2474996498.py in <cell line: 0>()
----> 1 model.load_state_dict(torch.load("best_model.pth", map_location=device))
      2 model.eval()
      3 
      4 

NameError: name 'model' is not defined

## === cell 25
all_outputs = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        preds = model(batch)  # (B,1,420,540)
        all_outputs.append(preds.cpu())
all_outputs = torch.cat(all_outputs, dim=0)  # (N,1,420,540)
print("All outputs shape:", all_outputs.shape)




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2960936652.py in <cell line: 0>()
      3     for batch in test_loader:
      4         batch = batch.to(device)
----> 5         preds = model(batch)  # (B,1,420,540)
      6         all_outputs.append(preds.cpu())
      7 all_outputs = torch.cat(all_outputs, dim=0)  # (N,1,420,540)

NameError: name 'model' is not defined

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


def remove_padding(pred_tensor, orig_size, target_size):
    pad_top, pad_left = compute_padding(orig_size, target_size)
    h, w = orig_size
    return pred_tensor[:, pad_top : pad_top + h, pad_left : pad_left + w]


test_file_paths = sorted(
    [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith(".png")
    ],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
)

submission_rows = []
for idx, path in enumerate(test_file_paths):
    img_id = os.path.splitext(os.path.basename(path))[0]

    pil_img = Image.open(path).convert("L")
    orig_w, orig_h = pil_img.size
    orig_size = (orig_h, orig_w)

    pred = all_outputs[idx]  # (1, 420, 540)
    cropped = remove_padding(pred, orig_size, target_size)  # (1, H, W)

    pred_np = cropped.squeeze(0).numpy()
    for r in range(orig_h):
        for c in range(orig_w):
            pixel_id = f"{img_id}_{r+1}_{c+1}"
            pixel_val = float(pred_np[r, c])
            submission_rows.append((pixel_id, pixel_val))

submission_path = "/kaggle/working/submission.csv"
with open(submission_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "value"])
    writer.writerows(submission_rows)

print(f"Submission file written to {submission_path}")

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_55/1909687856.py in <cell line: 0>()
     36     orig_size = (orig_h, orig_w)
     37 
---> 38     pred = all_outputs[idx]  # (1, 420, 540)
     39     cropped = remove_padding(pred, orig_size, target_size)  # (1, H, W)
     40 

IndexError: list index out of range
