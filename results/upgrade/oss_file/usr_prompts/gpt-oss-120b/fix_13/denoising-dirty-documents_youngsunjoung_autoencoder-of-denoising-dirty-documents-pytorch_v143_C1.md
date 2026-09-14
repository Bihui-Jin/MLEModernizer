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

0.27246

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.32655) has done: 'The changes add all required imports, define missing classes and variables, correctly set up paths, datasets, dataloaders, the model, loss, optimizer and training loop, and finally generate a properly‑formatted `submission.csv`. This fixes the NameError crashes, ensures the training runs, and writes a valid submission file that the competition accept.'
- What this solution (achieved 0.43003) has done: 'I fixed the typo causing the model construction to fail (`nn.ConvTransposed` → `nn.ConvTranspose2d`) and ensured the model is instantiated before any downstream code runs. This resolves the `AttributeError`, allows the training loop to execute, and enables inference and submission generation without indexing errors.'
- What this solution (achieved 0.43177) has done: 'I adjust the loss to focus on pure RMSE (the competition metric), and align the learning‑rate scheduler, early‑stopping and model‑checkpointing to the validation RMSE instead of the hybrid loss. I also extend training a bit (more epochs and patience) so the model can converge further. These changes keep the architecture unchanged while steering learning toward a lower RMSE, moving the score closer to the target.'
- What this solution (achieved 0.4744) has done: 'The fix replaces the deprecated `v2.ToDtype` with the current `v2.ConvertImageDtype` so the transforms work, which also restores the definitions of `train_transforms`, `val_transforms`, and `test_transforms`. With the transforms corrected, the dataloaders are created successfully, allowing the training loop, inference, and submission generation to run end‑to‑end and produce a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import cv2
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as v2
from torchvision.transforms import functional as TF
from torchinfo import summary
from sklearn.model_selection import train_test_split
from PIL import Image

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

base_input = "/kaggle/input/denoising-dirty-documents"
train_dir = os.path.join(base_input, "train")
train_cleaned_dir = os.path.join(base_input, "train_cleaned")
test_dir = os.path.join(base_input, "test")




## === cell 1
def load_images_from_folder(folder):
    """Load all PNG images from a folder. Returns an empty list if the folder does not exist."""
    images = []
    if not os.path.isdir(folder):
        return images
    for filename in os.listdir(folder):
        if filename.lower().endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
    return images




## === cell 2
train_images = load_images_from_folder(train_dir)
train_cleaned_images = load_images_from_folder(train_cleaned_dir)
test_images = load_images_from_folder(test_dir)

print(
    f"Loaded {len(train_images)} train, {len(train_cleaned_images)} cleaned, {len(test_images)} test images"
)




## === cell 3
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




## === cell 4
train_transforms = v2.Compose(
    [
        v2.PILToTensor(),
        PadToSize((420, 540)),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.4),
        v2.RandomApply([v2.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
        Grayscale(),
        v2.ConvertImageDtype(torch.float32),  # replaced deprecated ToDtype
    ]
)

val_transforms = v2.Compose(
    [
        v2.PILToTensor(),
        PadToSize((420, 540)),
        Grayscale(),
        v2.ConvertImageDtype(torch.float32),
    ]
)

test_transforms = v2.Compose(
    [
        v2.PILToTensor(),
        PadToSize((420, 540)),
        Grayscale(),
        v2.ConvertImageDtype(torch.float32),
    ]
)




## === cell 5
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


class PairedImageDataset(Dataset):
    def __init__(self, noisy_files, clean_files, transform=None):
        self.noisy_files = noisy_files
        self.clean_files = clean_files
        self.transform = transform

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_img = Image.open(self.noisy_files[idx]).convert("RGB")
        clean_img = Image.open(self.clean_files[idx]).convert("RGB")
        if self.transform:
            noisy_img = self.transform(noisy_img)
            clean_img = self.transform(clean_img)
        return noisy_img, clean_img




## === cell 6
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



## === cell 7
train_dataset = PairedImageDataset(
    train_files, cleaned_train, transform=train_transforms
)
val_dataset = PairedImageDataset(val_files, cleaned_val, transform=val_transforms)
test_dataset = ImageDataset(test_dir, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)




## === cell 8
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(1, 8, kernel_size=1, bias=False),
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
            nn.ConvTransposed(
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
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        dec5 = self.dec5(enc5)
        dec5 = F.interpolate(dec5, size=(26, 33), mode="bilinear", align_corners=False)
        dec5 = torch.cat([dec5, enc4], dim=1)
        dec5 = self.conv5(dec5)

        dec4 = self.dec4(enc4)
        dec4 = F.interpolate(dec4, size=(52, 67), mode="bilinear", align_corners=False)
        dec4 = torch.cat([dec4, enc3], dim=1)
        dec4 = self.conv4(dec4)

        dec3 = self.dec3(dec4)
        dec3 = F.interpolate(
            dec3, size=(105, 135), mode="bilinear", align_corners=False
        )

        dec2 = self.dec2(dec3)
        dec2 = F.interpolate(
            dec2, size=(210, 270), mode="bilinear", align_corners=False
        )

        dec1 = self.dec1(dec2)
        dec1 = F.interpolate(
            dec1, size=(420, 540), mode="bilinear", align_corners=False
        )

        return torch.clamp(dec1, 0.001, 0.999)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenoisingAutoencoder().to(device)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/790260393.py in <cell line: 0>()
    158 
    159 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
--> 160 model = DenoisingAutoencoder().to(device)
    161 

/tmp/ipykernel_11/790260393.py in __init__(self)
    107         self.dec1 = nn.Sequential(
    108             nn.ConvTranspose2d(8, 1, kernel_size=1, bias=False),
--> 109             nn.ConvTransposed(
    110                 1,
    111                 1,

AttributeError: module 'torch.nn' has no attribute 'ConvTransposed'

## === cell 9
summary(model, input_size=(1, 1, 420, 540), device=device)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2487729042.py in <cell line: 0>()
----> 1 summary(model, input_size=(1, 1, 420, 540), device=device)
      2 
      3 

NameError: name 'model' is not defined

## === cell 10
class RMSELoss(nn.Module):
    def forward(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target))


class HybridLoss(nn.Module):
    def __init__(self, lambda_rmse=1.0, lambda_l1=0.1):
        super().__init__()
        self.lambda_rmse = lambda_rmse
        self.lambda_l1 = lambda_l1
        self.rmse_loss = RMSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.lambda_rmse * self.rmse_loss(
            pred, target
        ) + self.lambda_l1 * self.l1_loss(pred, target)


criterion = HybridLoss(lambda_rmse=1.0, lambda_l1=0.1)
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2, verbose=True
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2585862664.py in <cell line: 0>()
     19 
     20 criterion = HybridLoss(lambda_rmse=1.0, lambda_l1=0.1)
---> 21 optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
     22 scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
     23     optimizer, mode="min", factor=0.5, patience=2, verbose=True

NameError: name 'model' is not defined

## === cell 11
epochs = 30  # extended training for better convergence
patience = 30  # allow more epochs before early stopping
early_stop_counter = 0
best_val_rmse = float("inf")
best_model_state = None

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_rmse = 0.0
    for noisy, clean in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        out = model(noisy)
        loss = criterion(out, clean)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_rmse += RMSELoss()(out, clean).item()
    train_loss /= len(train_loader)
    train_rmse /= len(train_loader)

    model.eval()
    val_loss = 0.0
    val_rmse = 0.0
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            out = model(noisy)
            loss = criterion(out, clean)
            val_loss += loss.item()
            val_rmse += RMSELoss()(out, clean).item()
    val_loss /= len(val_loader)
    val_rmse /= len(val_loader)

    scheduler.step(val_rmse)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} RMSE: {train_rmse:.4f} | "
        f"Val Loss: {val_loss:.4f} RMSE: {val_rmse:.4f}"
    )

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_model_state = model.state_dict()
        early_stop_counter = 0
        print("  ** New best RMSE model saved **")
    else:
        early_stop_counter += 1
        print(
            f"  No RMSE improvement. Early stop counter: {early_stop_counter}/{patience}"
        )

    if early_stop_counter >= patience:
        print("Early stopping triggered.")
        break

if best_model_state is not None:
    torch.save(best_model_state, "best_model.pth")
    print("Best model checkpoint saved.")



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3089301817.py in <cell line: 0>()
      6 
      7 for epoch in range(epochs):
----> 8     model.train()
      9     train_loss = 0.0
     10     train_rmse = 0.0

NameError: name 'model' is not defined

## === cell 12
if os.path.isfile("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    print("Best model loaded from checkpoint.")
else:
    print("No checkpoint found; using the last trained model.")

model.eval()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1009260071.py in <cell line: 0>()
      5     print("No checkpoint found; using the last trained model.")
      6 
----> 7 model.eval()
      8 

NameError: name 'model' is not defined

## === cell 13
all_outputs = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        outs = model(batch)  # shape: (B, 1, 420, 540)
        all_outputs.append(outs.cpu())
all_outputs = torch.cat(all_outputs, dim=0)  # (N_test, 1, 420, 540)
print("All outputs shape:", all_outputs.shape)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2226299249.py in <cell line: 0>()
      3     for batch in test_loader:
      4         batch = batch.to(device)
----> 5         outs = model(batch)  # shape: (B, 1, 420, 540)
      6         all_outputs.append(outs.cpu())
      7 all_outputs = torch.cat(all_outputs, dim=0)  # (N_test, 1, 420, 540)

NameError: name 'model' is not defined

## === cell 14
def compute_padding(orig_size, target_size):
    orig_h, orig_w = orig_size
    tgt_h, tgt_w = target_size
    pad_top = (tgt_h - orig_h) // 2 if tgt_h > orig_h else 0
    pad_left = (tgt_w - orig_w) // 2 if tgt_w > orig_w else 0
    return pad_top, pad_left


def remove_padding(pred, orig_size, target_size):
    pad_top, pad_left = compute_padding(orig_size, target_size)
    orig_h, orig_w = orig_size
    return pred[:, pad_top : pad_top + orig_h, pad_left : pad_left + orig_w]


target_size = (420, 540)  # (height, width)

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

    orig_img = Image.open(file_path).convert("L")
    orig_w, orig_h = orig_img.size  # (width, height)
    orig_size = (orig_h, orig_w)

    pred = all_outputs[idx]  # (1, 420, 540)
    cropped = remove_padding(pred, orig_size, target_size)  # (1, h, w)
    pred_np = cropped.squeeze(0).numpy()  # (h, w)

    for r in range(orig_h):
        for c in range(orig_w):
            pixel_id = f"{image_id}_{r+1}_{c+1}"
            pixel_value = float(pred_np[r, c])
            submission_data.append((pixel_id, pixel_value))

submission_file = "submission.csv"
with open(submission_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "value"])
    writer.writerows(submission_data)

print(f"Submission file '{submission_file}' created.")

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/3802767308.py in <cell line: 0>()
     32     orig_size = (orig_h, orig_w)
     33 
---> 34     pred = all_outputs[idx]  # (1, 420, 540)
     35     cropped = remove_padding(pred, orig_size, target_size)  # (1, h, w)
     36     pred_np = cropped.squeeze(0).numpy()  # (h, w)

IndexError: list index out of range
