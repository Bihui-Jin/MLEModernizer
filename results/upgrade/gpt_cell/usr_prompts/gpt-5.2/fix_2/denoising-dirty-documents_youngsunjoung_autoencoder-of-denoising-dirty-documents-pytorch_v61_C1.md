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

0.28531

# 6. Current score

0.36131

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.36131) has done: 'The crash happens because cell 32 hard-codes `for i in range(72)`, but the test folder in this environment contains fewer images (e.g., ~30), so `test_file_paths[i]` goes out of bounds. The minimal fix is to iterate over the actual number of discovered test files (and thus predictions) using `range(len(test_file_paths))`, while also guarding against any mismatch between `test_file_paths` and `all_outputs`. This preserves the existing submission-building logic and output format, but makes the loop robust and deterministic for any test set size present. No other cells need changes, and the produced `submission.csv` remains the same schema (`id,value`).'

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
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import v2
import math
import torchvision.transforms.functional as TF
import torch.nn.functional as F


## === cell 1
!pip install torchinfo
from torchinfo import summary


## === cell 2
!unzip /kaggle/input/denoising-dirty-documents/train.zip -d /content/denoising_data
!unzip /kaggle/input/denoising-dirty-documents/test.zip -d /content/denoising_data
!unzip /kaggle/input/denoising-dirty-documents/train_cleaned.zip -d /content/denoising_data


## === cell 3
train_dir = "/content/denoising_data/train"
train_cleaned_dir = "/content/denoising_data/train_cleaned"
test_dir = "/content/denoising_data/test"


## === cell 4
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):   # PNG파일 로드
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)  # OpenCV로 이미지 읽기(BGR형식)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
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
train_images[0]


## === cell 8
train_cleaned_images[0]


## === cell 9
test_images[0]


## === cell 10
train_sizes = [img.shape[:2] for img in train_images]
train_cleaned_sizes = [img.shape[:2] for img in train_cleaned_images]
test_sizes = [img.shape[:2] for img in test_images]


## === cell 11
unique_train_sizes = np.unique(train_sizes, axis=0)
unique_train_cleaned_sizes = np.unique(train_cleaned_sizes, axis=0)
unique_test_sizes = np.unique(test_sizes, axis=0)


## === cell 12
print(f"train_images:\n {unique_train_sizes}")
print(f"train_cleaned_images:\n {unique_train_cleaned_sizes}")
print(f"test_images:\n {unique_test_sizes}")


## === cell 13
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


## === cell 14
class Grayscale:
    def __call__(self, img):
        pil_img = TF.to_pil_image(img) if isinstance(img, torch.Tensor) else img

        grayscale_img = pil_img.convert("L")  # RGB -> Grayscale

        return TF.to_tensor(grayscale_img)


## === cell 15
train_transforms = v2.Compose(
    [
        v2.ToImage(),
        PadToSize((420, 540)),                                                  # 크기 통일
        v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.4),                # 가우시안 블러 (40% 확률 적용)
        v2.RandomApply([v2.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),  # 밝기 & 대비 조절 (50% 확률 적용)
        Grayscale(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

val_transforms = v2.Compose(
    [
        v2.ToImage(),
        PadToSize((420, 540)),                                                  # 크기 통일
        Grayscale(),  
        v2.ToDtype(torch.float32, scale=True), 
    ]
)

test_transforms = v2.Compose(
    [
        v2.ToImage(),
        PadToSize((420, 540)),                                                  # 크기 통일
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
            [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.png')]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')  # RGB로 열기

        if self.transform:
            image = self.transform(image)

        return image


## === cell 17
train_files = sorted(
    [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.png')]
)
cleaned_files = sorted(
    [os.path.join(train_cleaned_dir, f) for f in os.listdir(train_cleaned_dir) if f.endswith('.png')]
)

train_files, val_files, cleaned_train, cleaned_val = train_test_split(
    train_files, cleaned_files, test_size=2/9, random_state=42
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
        train_img = Image.open(self.train_files[idx]).convert('RGB')
        cleaned_img = Image.open(self.cleaned_files[idx]).convert('RGB')

        if self.transform:
            train_img = self.transform(train_img)
            cleaned_img = self.transform(cleaned_img)

        return train_img, cleaned_img


## === cell 19
train_dataset = PairedImageDataset(train_files, cleaned_train, train_transforms)
val_dataset = PairedImageDataset(val_files, cleaned_val, val_transforms)
test_dataset = ImageDataset(test_dir, test_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # 훈련 데이터 셔플 O
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)     # 검증 데이터 셔플 X
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)   # 테스트 데이터 셔플 X


## === cell 20
def visualize_paired_dataset(paired_loader, num_images=5):
    for train_images, cleaned_images in paired_loader:
        fig, axes = plt.subplots(num_images, 2, figsize=(8, num_images * 3))  # num_images 행, 2열 (위: train, 아래: cleaned)
        for i in range(num_images):
            axes[i, 0].imshow(train_images[i].permute(1, 2, 0).cpu().numpy(), cmap='gray')  # [C, H, W] -> [H, W, C]
            axes[i, 0].set_title(f"Original {i+1}")
            axes[i, 0].axis('off')  # 축 비활성화

            axes[i, 1].imshow(cleaned_images[i].permute(1, 2, 0).cpu().numpy(), cmap='gray')  # [C, H, W] -> [H, W, C]
            axes[i, 1].set_title(f"Cleaned {i+1}")
            axes[i, 1].axis('off')  # 축 비활성화

        plt.tight_layout()
        plt.show()
        break  # 한 번만 실행하도록 break

visualize_paired_dataset(train_loader, num_images=3)
visualize_paired_dataset(val_loader, num_images=2)


## === cell 21
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


## === cell 22
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.Conv2d(1, 8, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (8, 210, 270)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.Conv2d(8, 16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (16, 105, 135)
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.Conv2d(16, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (32, 52, 67)
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (64, 26, 33)
        )

        self.enc5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (128, 13, 16)
        )

        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=1, stride=1, bias=False),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )

        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=1, stride=1, bias=False),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=1, stride=1, bias=False),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1, groups=16, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=1, stride=1, bias=False),
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1, groups=8, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(8, 1, kernel_size=1, stride=1, bias=False),
            nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1, groups=1, bias=False),
            nn.Sigmoid()  # 픽셀 값을 0~1로 정규화
        )

        self.conv4 = nn.Conv2d(64, 32, kernel_size=1, stride=1, bias=False)  # (64 → 32)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=False)  # (128 → 64)
    
    def forward(self, x):
        enc1_out = self.enc1(x)  # (8, 210, 270)
        enc2_out = self.enc2(enc1_out)  # (16, 105, 135)
        enc3_out = self.enc3(enc2_out)  # (32, 52, 67)
        enc4_out = self.enc4(enc3_out)  # (64, 26, 33)
        enc5_out = self.enc5(enc4_out)  # (128, 13, 16)

        dec5_out = self.dec5(enc5_out)  # (64, 26, 33)
        dec5_out = F.interpolate(dec5_out, size=(26, 33), mode='bilinear', align_corners=False)
        dec5_out = torch.cat([dec5_out, enc4_out], dim=1)
        dec5_out = self.conv5(dec5_out)  # (128 → 64)

        dec4_out = self.dec4(enc4_out)  # (32, 52, 67)
        dec4_out = F.interpolate(dec4_out, size=(52, 67), mode='bilinear', align_corners=False)
        dec4_out = torch.cat([dec4_out, enc3_out], dim=1)  # Skip Connection
        dec4_out = self.conv4(dec4_out)  # (64 → 32)

        dec3_out = self.dec3(dec4_out)  # (16, 104, 134)
        dec3_out = F.interpolate(dec3_out, size=(105, 135), mode='bilinear', align_corners=False)

        dec2_out = self.dec2(dec3_out)  # (8, 210, 270)
        dec2_out = F.interpolate(dec2_out, size=(210, 270), mode='bilinear', align_corners=False)

        dec1_out = self.dec1(dec2_out)  # (1, 420, 540)
        dec1_out = F.interpolate(dec1_out, size=(420, 540), mode='bilinear', align_corners=False)

        outputs = torch.clamp(dec1_out, min=0.001, max=0.999)
        return outputs

model = DenoisingAutoencoder().to(device)


## === cell 23
summary(model, input_size=(16, 1, 420, 540), device=device)


## === cell 24
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target))


## === cell 25
class HybridLoss(nn.Module):
    def __init__(self, lambda_rmse=0.8, lambda_l1=0.2):
        super(HybridLoss, self).__init__()
        self.lambda_rmse = lambda_rmse
        self.lambda_l1 = lambda_l1
        self.rmse_loss = RMSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.lambda_rmse * self.rmse_loss(pred, target) + self.lambda_l1 * self.l1_loss(pred, target)


## === cell 26
criterion = HybridLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)     # 스케쥴러가 적용될 것을 감안해 0.01부터 시작
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)


## === cell 27
epochs = 1000  # 최대 학습 에포크 수
patience = 5  # 몇 번 연속 val_loss가 증가하면 멈출지
early_stop_counter = 0  # Early Stopping 카운터
best_val_loss = float('inf')  # 초기 Best Loss를 매우 큰 값으로 설정
best_model_state = None  # Best 모델의 가중치를 저장할 변수

epoch = 0
while epoch < epochs:
    model.train()
    train_loss = 0.0
    train_rmse_loss = 0.0
    for train_images, train_cleaned_images in train_loader:
        train_images, train_cleaned_images = train_images.to(device), train_cleaned_images.to(device)
        optimizer.zero_grad()
        outputs = model(train_images)
        loss = criterion(outputs, train_cleaned_images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_rmse_loss += RMSELoss()(outputs, train_cleaned_images).item()  # RMSE 계산

    train_loss /= len(train_loader)
    train_rmse_loss /= len(train_loader)  # 평균 RMSE 계산
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, RMSE Score : {train_rmse_loss:.4f}")

    model.eval()
    val_loss = 0.0
    val_rmse_loss = 0.0
    with torch.no_grad():
        for val_images, val_cleaned_images in val_loader:
            val_images, val_cleaned_images = val_images.to(device), val_cleaned_images.to(device)
            outputs = model(val_images)
            loss = criterion(outputs, val_cleaned_images)
            val_loss += loss.item()
            val_rmse_loss += RMSELoss()(outputs, val_cleaned_images).item()  # RMSE 계산

    val_loss /= len(val_loader)  # 평균 Loss 계산
    val_rmse_loss /= len(val_loader)  # 평균 RMSE 계산

    prev_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    if current_lr != prev_lr:
        print(f"Learning Rate updated: {current_lr:.6f}\n")

    if val_loss < best_val_loss:
        print(f"New best validation loss: {val_loss:.4f} (Previous: {best_val_loss:.4f}), RMSE Score : {val_rmse_loss:.4f}")
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"Validation loss increased! Early stopping counter: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        print(f"Early stopping triggered! after {epoch+1} epochs")
        model.load_state_dict(best_model_state)
        print(f"Best model loaded with val_loss = {best_val_loss:.4f}")
        break

    epoch += 1

if best_model_state is not None:  # Best 모델 저장
    torch.save(best_model_state, "best_model.pth")
    print(f"Best model saved with val_loss = {best_val_loss:.4f}")
else:
    print("No best model was saved.")


## === cell 28
best_model_path = "best_model.pth"  # 저장된 모델 경로
model.load_state_dict(torch.load(best_model_path))  # Best Model 불러오기

model.eval()  # 평가 모드
with torch.no_grad():
    for images in test_loader:
        images = images.to(device) # 원본 이미지
        outputs = model(images) # 출력
        break


## === cell 29
def visualize_images_and_outputs(images, outputs):
    """
    이미지와 출력 이미지를 열로 구분하여 시각화.
    :param images: 원본 이미지 텐서
    :param outputs: 모델 출력 텐서
    """
    num_images = images.size(0)  # 전체 이미지 개수
    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 3))  # num_images 행, 2열

    for i in range(num_images):
        axes[i, 0].imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
        axes[i, 0].set_title(f"Original {i + 1}", fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(outputs[i].cpu().detach().numpy().squeeze(), cmap='gray')
        axes[i, 1].set_title(f"Output {i + 1}", fontsize=10)
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

visualize_images_and_outputs(images, outputs)


## === cell 30
test_dir = "/content/denoising_data/test"
test_dir


## === cell 31
best_model_path = "best_model.pth"  # 저장된 모델 경로
model.load_state_dict(torch.load(best_model_path))  # Best Model 불러오기

model.eval()
all_outputs = []  # 예측 결과를 저장할 리스트
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        outputs = model(batch)  # outputs의 shape: (batch_size, 1, 420, 540)
        all_outputs.append(outputs.cpu())

all_outputs = torch.cat(all_outputs, dim=0)
print("all_outputs shape:", all_outputs.shape)


## === cell 32
import os
import csv
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF  # 원래 패딩 적용에 사용한 함수


target_size = (420, 540)  # (높이, 너비)


def compute_padding(orig_size, target_size):
    """
    원본 이미지 크기(orig_size)를 target_size로 패딩했을 때 추가된 패딩 값을 계산합니다.
    :param orig_size: (높이, 너비) 튜플 (원본 이미지 크기)
    :param target_size: (높이, 너비) 튜플 (패딩이 적용된 크기)
    :return: (pad_top, pad_bottom, pad_left, pad_right)
    """
    orig_height, orig_width = orig_size
    target_height, target_width = target_size
    pad_top = (target_height - orig_height) // 2 if target_height > orig_height else 0
    pad_bottom = (
        target_height - orig_height - pad_top if target_height > orig_height else 0
    )
    pad_left = (target_width - orig_width) // 2 if target_width > orig_width else 0
    pad_right = target_width - orig_width - pad_left if target_width > orig_width else 0
    return pad_top, pad_bottom, pad_left, pad_right


def remove_padding(pred, orig_size, target_size):
    """
    모델 출력(pred)에서 원본 이미지 영역만 크롭합니다.
    :param pred: 모델 예측 결과 텐서, shape: (C, target_height, target_width)
    :param orig_size: 원본 이미지 크기 (높이, 너비)
    :param target_size: 패딩 적용된 크기 (높이, 너비)
    :return: 패딩이 제거된 텐서 (원본 영역만), shape: (C, orig_height, orig_width)
    """
    orig_height, orig_width = orig_size
    pad_top, _, pad_left, _ = compute_padding(orig_size, target_size)
    return pred[:, pad_top : pad_top + orig_height, pad_left : pad_left + orig_width]


test_file_paths = sorted(
    [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".png")],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
)

num_files = len(test_file_paths)
num_preds = (
    int(all_outputs.shape[0])
    if isinstance(all_outputs, torch.Tensor)
    else len(all_outputs)
)
num_to_process = min(num_files, num_preds)

submission_data = []

for i in range(num_to_process):
    file_path = test_file_paths[i]
    image_id = os.path.splitext(os.path.basename(file_path))[0]

    orig_img = Image.open(file_path).convert("L")
    orig_width, orig_height = orig_img.size
    orig_size = (orig_height, orig_width)  # (높이, 너비)
    print("orig_size", orig_size)

    pred = all_outputs[i]  # shape: (1, 420, 540)

    cropped_pred = remove_padding(pred, orig_size, target_size)
    print("crop size", cropped_pred.shape)

    pred_np = cropped_pred.squeeze(0).cpu().numpy()

    plt.figure()
    plt.imshow(pred_np, cmap="gray")
    plt.title(
        f"{i}th Cropped Output for image {image_id}\n(Original height: {orig_height} width: {orig_width})"
    )
    plt.axis("off")
    plt.show()

    for row in range(orig_height):
        for col in range(orig_width):
            pixel_id = f"{image_id}_{row+1}_{col+1}"
            pixel_value = pred_np[row, col]
            submission_data.append((pixel_id, pixel_value))

submission_file = "submission.csv"
with open(submission_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "value"])  # CSV 헤더 작성
    for row in submission_data:
        writer.writerow(row)

print(f"Submission file '{submission_file}'이(가) 생성되었습니다.")
