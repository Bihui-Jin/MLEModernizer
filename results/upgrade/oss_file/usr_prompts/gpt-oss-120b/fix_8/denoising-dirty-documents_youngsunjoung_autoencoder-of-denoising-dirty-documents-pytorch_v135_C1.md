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

0.35827

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.41716) has done: 'I fixed the runtime error by iterating over the actual number of test images instead of a hard‑coded 72, and I made the unzip and torchinfo installation steps safe for a script environment. These changes ensure the pipeline runs end‑to‑end and creates a valid `submission.csv` with the correct id/value format.'
- What this solution (achieved 0.47062) has done: 'Implemented fixes:
- Imported `train_test_split` to resolve the NameError.
- Corrected dataset split variables and ensured loaders (`test_loader`) are created.
- Added a lightweight training loop (5 epochs) with validation loss tracking and model checkpointing, improving model performance toward the target score.
- Adjusted subsequent cells to use the trained model and generate a proper submission CSV.'
- What this solution (achieved 0.44469) has done: 'I reduce unnecessary data augmentations that can hurt denoising performance, add a small L1 component to the loss, and train for more epochs so the model can converge better. These tweaks keep the original architecture unchanged while moving the validation RMSE closer to the target.'

# 9. Code solution

## === cell 0
try:
    from torchinfo import summary
except ImportError:
    import subprocess, sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchinfo"])
    from torchinfo import summary



## === cell 1
import zipfile, pathlib, os, sys, subprocess
from sklearn.model_selection import train_test_split  # added import

base_dir = pathlib.Path("/kaggle/input/denoising-dirty-documents")
if not base_dir.exists():
    raise FileNotFoundError(f"Expected data directory {base_dir} not found.")



## === cell 2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader, random_split
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
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 3
train_dir = base_dir / "train"
train_cleaned_dir = base_dir / "train_cleaned"
test_dir = base_dir / "test"




## === cell 4
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return images




## === cell 5
train_images = load_images_from_folder(str(train_dir))
train_cleaned_images = load_images_from_folder(str(train_cleaned_dir))
test_images = load_images_from_folder(str(test_dir))



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




## === cell 14
train_files = sorted(
    [
        os.path.join(str(train_dir), f)
        for f in os.listdir(str(train_dir))
        if f.endswith(".png")
    ]
)
cleaned_files = sorted(
    [
        os.path.join(str(train_cleaned_dir), f)
        for f in os.listdir(str(train_cleaned_dir))
        if f.endswith(".png")
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
test_dataset = ImageDataset(str(test_dir), test_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)




## === cell 17
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




## === cell 18
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




## === cell 19
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
        enc1_out = self.enc1(x)  # -> (8, 210, 270)
        enc2_out = self.enc2(enc1_out)  # -> (16, 105, 135)
        enc3_out = self.enc3(enc2_out)  # -> (32, 52, 68)
        enc4_out = self.enc4(enc3_out)  # -> (64, 26, 34)
        enc5_out = self.enc5(enc4_out)  # -> (128, 13, 17)

        dec5_out = self.dec5(enc5_out)  # -> (64, 26, 34)
        dec5_out = torch.cat([dec5_out, enc4_out], dim=1)  # (128, 26, 34)
        dec5_out = self.conv5(dec5_out)  # -> (64, 26, 34)

        dec4_out = self.dec4(enc4_out)  # -> (32, 52, 68)
        dec4_out = torch.cat([dec4_out, enc3_out], dim=1)  # (64, 52, 68)
        dec4_out = self.conv4(dec4_out)  # -> (32, 52, 68)

        dec3_out = self.dec3(dec4_out)  # -> (16, 104, 136)
        dec3_out = F.interpolate(
            dec3_out, size=(105, 135), mode="bilinear", align_corners=False
        )

        dec2_out = self.dec2(dec3_out)  # -> (8, 210, 270)
        dec1_out = self.dec1(dec2_out)  # -> (1, 420, 540)

        outputs = torch.clamp(dec1_out, min=0.001, max=0.999)
        return outputs


model = DenoisingAutoencoder().to(device)



## === cell 20
summary(model, input_size=(16, 1, 420, 540), device=device)




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/torchinfo/torchinfo.py in forward_pass(model, x, batch_dim, cache_forward_pass, device, mode, **kwargs)
    294             if isinstance(x, (list, tuple)):
--> 295                 _ = model(*x, **kwargs)
    296             elif isinstance(x, dict):

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1844         try:
-> 1845             return inner()
   1846         except Exception:

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in inner()
   1792 
-> 1793             result = forward_call(*args, **kwargs)
   1794             if _global_forward_hooks or self._forward_hooks:

/tmp/ipykernel_11/1937434355.py in forward(self, x)
    133         dec5_out = self.dec5(enc5_out)  # -> (64, 26, 34)
--> 134         dec5_out = torch.cat([dec5_out, enc4_out], dim=1)  # (128, 26, 34)
    135         dec5_out = self.conv5(dec5_out)  # -> (64, 26, 34)

RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 32 but got size 33 for tensor number 1 in the list.

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/34055172.py in <cell line: 0>()
----> 1 summary(model, input_size=(16, 1, 420, 540), device=device)
      2 
      3 

/usr/local/lib/python3.11/dist-packages/torchinfo/torchinfo.py in summary(model, input_size, input_data, batch_dim, cache_forward_pass, col_names, col_width, depth, device, dtypes, mode, row_settings, verbose, **kwargs)
    221         input_data, input_size, batch_dim, device, dtypes
    222     )
--> 223     summary_list = forward_pass(
    224         model, x, batch_dim, cache_forward_pass, device, model_mode, **kwargs
    225     )

/usr/local/lib/python3.11/dist-packages/torchinfo/torchinfo.py in forward_pass(model, x, batch_dim, cache_forward_pass, device, mode, **kwargs)
    302     except Exception as e:
    303         executed_layers = [layer for layer in summary_list if layer.executed]
--> 304         raise RuntimeError(
    305             "Failed to run torchinfo. See above stack traces for more details. "
    306             f"Executed layers up to: {executed_layers}"

RuntimeError: Failed to run torchinfo. See above stack traces for more details. Executed layers up to: [Sequential: 1, Conv2d: 2, Conv2d: 2, BatchNorm2d: 2, ReLU: 2, MaxPool2d: 2, Sequential: 1, Conv2d: 2, Conv2d: 2, BatchNorm2d: 2, ReLU: 2, Dropout: 2, MaxPool2d: 2, Sequential: 1, Conv2d: 2, Conv2d: 2, BatchNorm2d: 2, ReLU: 2, Dropout: 2, MaxPool2d: 2, Sequential: 1, Conv2d: 2, Conv2d: 2, BatchNorm2d: 2, ReLU: 2, Dropout: 2, MaxPool2d: 2, Sequential: 1, Conv2d: 2, Conv2d: 2, BatchNorm2d: 2, ReLU: 2, Dropout: 2, MaxPool2d: 2, Sequential: 1, ConvTranspose2d: 2, ConvTranspose2d: 2, BatchNorm2d: 2, ReLU: 2, Dropout: 2]

## === cell 21
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target))




## === cell 22
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




## === cell 23
criterion = HybridLoss(lambda_rmse=1.0, lambda_l1=0.0)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)



## === cell 24
best_model_path = "best_model.pth"
best_val_loss = float("inf")
num_epochs = 18

for epoch in range(1, num_epochs + 1):
    model.train()
    train_losses = []
    for noisy, clean in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    avg_train_loss = np.mean(train_losses)

    model.eval()
    val_losses = []
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            loss = criterion(output, clean)
            val_losses.append(loss.item())
    avg_val_loss = np.mean(val_losses) if val_losses else float("inf")
    scheduler.step(avg_val_loss)

    print(
        f"Epoch {epoch}/{num_epochs} - Train loss: {avg_train_loss:.6f} - Val loss: {avg_val_loss:.6f}"
    )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved new best model with val loss {best_val_loss:.6f}")

if not os.path.exists(best_model_path):
    torch.save(model.state_dict(), best_model_path)

print(f"Training complete. Best validation loss: {best_val_loss:.6f}")



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1600280799.py in <cell line: 0>()
      9         noisy, clean = noisy.to(device), clean.to(device)
     10         optimizer.zero_grad()
---> 11         output = model(noisy)
     12         loss = criterion(output, clean)
     13         loss.backward()

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_11/1937434355.py in forward(self, x)
    132         # Decoder with correct up‑sampling sizes
    133         dec5_out = self.dec5(enc5_out)  # -> (64, 26, 34)
--> 134         dec5_out = torch.cat([dec5_out, enc4_out], dim=1)  # (128, 26, 34)
    135         dec5_out = self.conv5(dec5_out)  # -> (64, 26, 34)
    136 

RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 32 but got size 33 for tensor number 1 in the list.

## === cell 25
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()
print("Model loaded for inference.")




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4240884181.py in <cell line: 0>()
----> 1 model.load_state_dict(torch.load(best_model_path, map_location=device))
      2 model.eval()
      3 print("Model loaded for inference.")
      4 
      5 

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

## === cell 26
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


with torch.no_grad():
    sample_batch = next(iter(test_loader)).to(device)
    sample_outputs = model(sample_batch)
visualize_images_and_outputs(sample_batch, sample_outputs)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1208210599.py in <cell line: 0>()
     15 with torch.no_grad():
     16     sample_batch = next(iter(test_loader)).to(device)
---> 17     sample_outputs = model(sample_batch)
     18 visualize_images_and_outputs(sample_batch, sample_outputs)
     19 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_11/1937434355.py in forward(self, x)
    132         # Decoder with correct up‑sampling sizes
    133         dec5_out = self.dec5(enc5_out)  # -> (64, 26, 34)
--> 134         dec5_out = torch.cat([dec5_out, enc4_out], dim=1)  # (128, 26, 34)
    135         dec5_out = self.conv5(dec5_out)  # -> (64, 26, 34)
    136 

RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 32 but got size 33 for tensor number 1 in the list.

## === cell 27
print(f"Test directory: {test_dir}")



## === cell 28
all_outputs = []
model.eval()
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        outputs = model(batch)  # (B,1,420,540)
        all_outputs.append(outputs.cpu())
all_outputs = torch.cat(all_outputs, dim=0)  # shape (N,1,420,540)
print("all_outputs shape:", all_outputs.shape)



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/983633199.py in <cell line: 0>()
      4     for batch in test_loader:
      5         batch = batch.to(device)
----> 6         outputs = model(batch)  # (B,1,420,540)
      7         all_outputs.append(outputs.cpu())
      8 all_outputs = torch.cat(all_outputs, dim=0)  # shape (N,1,420,540)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_11/1937434355.py in forward(self, x)
    132         # Decoder with correct up‑sampling sizes
    133         dec5_out = self.dec5(enc5_out)  # -> (64, 26, 34)
--> 134         dec5_out = torch.cat([dec5_out, enc4_out], dim=1)  # (128, 26, 34)
    135         dec5_out = self.conv5(dec5_out)  # -> (64, 26, 34)
    136 

RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 32 but got size 33 for tensor number 1 in the list.

## === cell 29
import csv
from PIL import Image

target_size = (420, 540)  # (height, width)


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
        os.path.join(str(test_dir), f)
        for f in os.listdir(str(test_dir))
        if f.endswith(".png")
    ],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
)

submission_data = []

for idx, file_path in enumerate(test_file_paths):
    image_id = os.path.splitext(os.path.basename(file_path))[0]

    orig_img = Image.open(file_path).convert("L")
    orig_width, orig_height = orig_img.size
    orig_size = (orig_height, orig_width)  # (height, width)

    pred = all_outputs[idx]  # (1, 420, 540)
    cropped_pred = remove_padding(pred, orig_size, target_size)  # (1, H, W)

    pred_np = cropped_pred.squeeze(0).cpu().numpy()

    for row in range(orig_height):
        for col in range(orig_width):
            pixel_id = f"{image_id}_{row+1}_{col+1}"
            pixel_value = float(pred_np[row, col])
            submission_data.append((pixel_id, pixel_value))

submission_file = "submission.csv"
with open(submission_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "value"])
    writer.writerows(submission_data)

print(
    f"Submission file '{submission_file}' has been created with {len(submission_data)} rows."
)

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/3145964414.py in <cell line: 0>()
     39     orig_size = (orig_height, orig_width)  # (height, width)
     40 
---> 41     pred = all_outputs[idx]  # (1, 420, 540)
     42     cropped_pred = remove_padding(pred, orig_size, target_size)  # (1, H, W)
     43 

IndexError: list index out of range
