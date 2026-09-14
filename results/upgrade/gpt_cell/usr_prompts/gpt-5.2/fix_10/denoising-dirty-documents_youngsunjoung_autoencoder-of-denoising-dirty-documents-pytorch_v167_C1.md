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

0.32751

# 6. Current score

0.28598

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.3723) has done: 'Diagnosis: Cell 32 assumes there are 72 test images (`for i in range(72)`), but this dataset only contains ~30 test PNGs, so indexing `test_file_paths[i]` eventually goes out of range and raises `IndexError`. The root cause is the hard-coded loop bound being larger than the actual number of discovered files. A secondary inefficiency/fragility is using `test_file_paths.index(file_path)` inside the loop, which is unnecessary and can misbehave if duplicates ever existed; the loop index `i` already matches `all_outputs` ordering because both are created from sorted test files.  

Patch summary: Replace the hard-coded `range(72)` with `range(len(test_file_paths))` and use `pred = all_outputs[i]` directly, keeping all existing submission generation logic intact. This fixes the crash while preserving the same semantics for all available test images and maintains the expected `submission.csv` output format.  

Updated cells: Only cell 32 is changed.  

Compatibility notes for cell k+1: No changes to variables/interfaces expected by later cells; `submission.csv` is still produced in the same location with the same columns (`id`, `value`).  

Assumptions: `all_outputs` was generated from the same `test_loader` that loads `test_dir` in sorted order, so `all_outputs[i]` corresponds to `test_file_paths[i]` (as intended by the original code).'
- What this solution (achieved 0.43448) has done: 'Diagnosis: The crash in cell 34 is a `SyntaxError` caused by an accidental Markdown fence (a line containing'
- What this solution (achieved 0.43095) has done: 'To move your RMSE down toward the 0.32751 target with minimal risk, I’m keeping your model and training loop intact and focusing on two score-relevant issues: (1) your train/val split currently breaks the pairing between noisy and cleaned images (hurting learning), and (2) your submission rows must be in the exact same order as `sampleSubmission.csv` to avoid accidental misalignment. The patch fixes the split to be by shared image id so each noisy image matches its cleaned counterpart, and it generates the submission by filling `sampleSubmission.csv`’s `id` order using a fast id→prediction lookup. These are minimal semantic changes (no architecture/loss changes) but typically yield a sizable RMSE improvement and stabilize the submission.'
- What this solution (achieved 0.41826) has done: 'Your current gap to the target RMSE is sizeable (0.43095 vs 0.32751, lower is better), so we need a small but score-relevant fix without changing the model/loss/training loop. The biggest score leak in your pipeline is that you apply heavy random augmentations to the noisy inputs but not to the cleaned targets, which misaligns pixels (the metric is pixelwise RMSE) and forces the model to learn an impossible mapping. I keep the same transforms and architecture, but ensure that any *geometric* transform is applied identically to input and target while keeping photometric/noise-only augmentations on the input; this preserves core logic while making supervision consistent and typically improves RMSE substantially. I also clamp submission values to [0,1] (evaluation expects that range) without changing the model output semantics.'
- What this solution (achieved 0.43651) has done: 'To reduce RMSE toward your 0.32751 target without changing the model/loss/training loop, I’m keeping everything intact and focusing on one score-critical mismatch: your transforms currently output tensors shaped (1,H,W), but the dataloader batches them into (B,1,H,W) while your model expects exactly that; however, your custom `PadToSize` assumes a tensor input and reads `img.shape` as (C,H,W), which is not guaranteed with `v2.ToImage()` when the input is a PIL image. I make `PadToSize` robust to both PIL and tensor inputs (and always pad based on (H,W)), which fixes subtle incorrect padding/centering that directly harms pixelwise RMSE. I also ensure the training/validation targets are strictly grayscale tensors with the same dtype/scale as inputs (already intended, but enforced consistently), so supervision is perfectly aligned. These are minimal changes to preprocessing only; architecture, losses, and training logic remain unchanged.'
- What this solution (achieved 0.28891) has done: 'Your current RMSE (0.43651) is worse than the target (0.32751), so we need a small, score-relevant fix without touching the model/loop/loss. The biggest remaining issue is that your training input augmentation includes `ColorJitter` even though you later convert everything to grayscale; this adds unnecessary randomness that can make learning harder without adding useful signal. I remove the grayscale-ineffective `ColorJitter` from the input-only augmentation (keeping blur/noise augmentation intact), and I keep all dataset pairing and submission ordering logic unchanged. This is a minimal preprocessing change that typically improves RMSE for pixelwise denoising while preserving the core approach.'
- What this solution (achieved 0.28598) has done: 'Your current RMSE (0.28891) is already better than the target (0.32751), so to move *toward* the target band (±10%) we should slightly reduce denoising strength rather than improve it. With minimal, score-relevant change and without touching the model/architecture/training, I adjust only the final prediction post-processing: change the output clamp from a tight `[0.001, 0.999]` to the evaluation-expected `[0.0, 1.0]` (this typically increases pixel variance and slightly worsens RMSE). I also keep the submission mapping exactly in `sampleSubmission.csv` order and ensure values are clipped to `[0,1]` in the submission, preserving format validity.'

# 9. Code solution

## === cell 0
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
import torch.nn.functional as F



## === cell 1
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 2
try:
    import torchinfo  # noqa: F401
except Exception:
    pass
from torchinfo import summary



## === cell 3
os.makedirs("/content/denoising_data", exist_ok=True)
os.system(
    "unzip -o /kaggle/input/denoising-dirty-documents/train.zip -d /content/denoising_data"
)
os.system(
    "unzip -o /kaggle/input/denoising-dirty-documents/test.zip -d /content/denoising_data"
)
os.system(
    "unzip -o /kaggle/input/denoising-dirty-documents/train_cleaned.zip -d /content/denoising_data"
)



## === cell 4
train_dir = "/content/denoising_data/train"
train_cleaned_dir = "/content/denoising_data/train_cleaned"
test_dir = "/content/denoising_data/test"




## === cell 5
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return images




## === cell 6
train_images = load_images_from_folder(train_dir)
train_cleaned_images = load_images_from_folder(train_cleaned_dir)
test_images = load_images_from_folder(test_dir)



## === cell 7
print(f"train_images: {len(train_images)}")
print(f"train_cleaned_images: {len(train_cleaned_images)}")
print(f"test_images: {len(test_images)}")



## === cell 8
train_images[0]



## === cell 9
train_cleaned_images[0]



## === cell 10
test_images[0]



## === cell 11
train_sizes = [img.shape[:2] for img in train_images]
train_cleaned_sizes = [img.shape[:2] for img in train_cleaned_images]
test_sizes = [img.shape[:2] for img in test_images]



## === cell 12
unique_train_sizes = np.unique(train_sizes, axis=0)
unique_train_cleaned_sizes = np.unique(train_cleaned_sizes, axis=0)
unique_test_sizes = np.unique(test_sizes, axis=0)



## === cell 13
print(f"train_images:\n {unique_train_sizes}")
print(f"train_cleaned_images:\n {unique_train_cleaned_sizes}")
print(f"test_images:\n {unique_test_sizes}")




## === cell 14
class PadToSize:
    def __init__(self, target_size):
        self.target_size = target_size  # (H, W)

    def __call__(self, img):
        target_h, target_w = self.target_size

        if isinstance(img, torch.Tensor):
            if img.ndim == 3:
                h, w = int(img.shape[-2]), int(img.shape[-1])
            elif img.ndim == 2:
                h, w = int(img.shape[-2]), int(img.shape[-1])
            else:
                raise ValueError(
                    f"Unexpected tensor shape for PadToSize: {tuple(img.shape)}"
                )
        else:
            w, h = img.size

        pad_top = (target_h - h) // 2 if target_h > h else 0
        pad_bottom = (target_h - h - pad_top) if target_h > h else 0
        pad_left = (target_w - w) // 2 if target_w > w else 0
        pad_right = (target_w - w - pad_left) if target_w > w else 0

        return TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=0)




## === cell 15
class Grayscale:
    def __call__(self, img):
        pil_img = TF.to_pil_image(img) if isinstance(img, torch.Tensor) else img
        grayscale_img = pil_img.convert("L")
        return TF.to_tensor(grayscale_img)




## === cell 16
base_transforms = v2.Compose(
    [
        v2.ToImage(),
        PadToSize((420, 540)),
        Grayscale(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

train_input_only_aug = v2.Compose(
    [
        v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.4),
    ]
)

train_transforms = v2.Compose([base_transforms, train_input_only_aug])
val_transforms = base_transforms
test_transforms = base_transforms




## === cell 17
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




## === cell 18
train_files_all = sorted(
    [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".png")]
)
cleaned_files_all = sorted(
    [
        os.path.join(train_cleaned_dir, f)
        for f in os.listdir(train_cleaned_dir)
        if f.endswith(".png")
    ]
)

train_map = {os.path.splitext(os.path.basename(p))[0]: p for p in train_files_all}
clean_map = {os.path.splitext(os.path.basename(p))[0]: p for p in cleaned_files_all}
common_ids = sorted(set(train_map.keys()) & set(clean_map.keys()), key=lambda x: int(x))

ids_train, ids_val = train_test_split(common_ids, test_size=2 / 9, random_state=42)

train_files = [train_map[i] for i in ids_train]
cleaned_train = [clean_map[i] for i in ids_train]
val_files = [train_map[i] for i in ids_val]
cleaned_val = [clean_map[i] for i in ids_val]

print(
    f"paired train={len(train_files)}, val={len(val_files)}, total_common={len(common_ids)}"
)




## === cell 19
class PairedImageDataset(Dataset):
    def __init__(
        self, train_files, cleaned_files, transform_input=None, transform_target=None
    ):
        self.train_files = train_files
        self.cleaned_files = cleaned_files
        self.transform_input = transform_input
        self.transform_target = transform_target

    def __len__(self):
        return len(self.train_files)

    def __getitem__(self, idx):
        train_img = Image.open(self.train_files[idx]).convert("RGB")
        cleaned_img = Image.open(self.cleaned_files[idx]).convert("RGB")

        if self.transform_input:
            train_img = self.transform_input(train_img)
        if self.transform_target:
            cleaned_img = self.transform_target(cleaned_img)

        return train_img, cleaned_img




## === cell 20
train_dataset = PairedImageDataset(
    train_files,
    cleaned_train,
    transform_input=train_transforms,
    transform_target=val_transforms,
)
val_dataset = PairedImageDataset(
    val_files,
    cleaned_val,
    transform_input=val_transforms,
    transform_target=val_transforms,
)
test_dataset = ImageDataset(test_dir, test_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)




## === cell 21
def visualize_paired_dataset(paired_loader, num_images=5):
    for train_images_b, cleaned_images_b in paired_loader:
        fig, axes = plt.subplots(num_images, 2, figsize=(8, num_images * 3))
        for i in range(num_images):
            axes[i, 0].imshow(
                train_images_b[i].permute(1, 2, 0).cpu().numpy(), cmap="gray"
            )
            axes[i, 0].set_title(f"Original {i+1}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(
                cleaned_images_b[i].permute(1, 2, 0).cpu().numpy(), cmap="gray"
            )
            axes[i, 1].set_title(f"Cleaned {i+1}")
            axes[i, 1].axis("off")

        plt.tight_layout()
        plt.show()
        break


visualize_paired_dataset(train_loader, num_images=3)
visualize_paired_dataset(val_loader, num_images=2)



## === cell 22
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




## === cell 23
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.Conv2d(1, 8, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.Conv2d(8, 16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.Conv2d(16, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.enc5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
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

        outputs = torch.clamp(dec1_out, min=0.0, max=1.0)
        return outputs


model = DenoisingAutoencoder().to(device)



## === cell 24
summary(model, input_size=(16, 1, 420, 540), device=device)




## === cell 25
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target))




## === cell 26
class HybridLoss(nn.Module):
    def __init__(self, lambda_rmse=0.8, lambda_l1=0.2):
        super(HybridLoss, self).__init__()
        self.lambda_rmse = lambda_rmse
        self.lambda_l1 = lambda_l1
        self.rmse_loss = RMSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.lambda_rmse * self.rmse_loss(
            pred, target
        ) + self.lambda_l1 * self.l1_loss(pred, target)




## === cell 27
criterion = HybridLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)



## === cell 28
epochs = 1000
patience = 5
early_stop_counter = 0
best_val_loss = float("inf")
best_model_state = None

epoch = 0
while epoch < epochs:
    model.train()
    train_loss = 0.0
    train_rmse_loss = 0.0
    for train_images_batch, train_cleaned_images_batch in train_loader:
        train_images_batch = train_images_batch.to(device)
        train_cleaned_images_batch = train_cleaned_images_batch.to(device)

        optimizer.zero_grad()
        outputs = model(train_images_batch)
        loss = criterion(outputs, train_cleaned_images_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_rmse_loss += RMSELoss()(outputs, train_cleaned_images_batch).item()

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
            val_images = val_images.to(device)
            val_cleaned_images = val_cleaned_images.to(device)

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
        print(f"Learning Rate updated: {current_lr:.6f}\n")
        early_stop_counter -= 1

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



## === cell 29
best_model_path = "best_model.pth"
model.load_state_dict(torch.load(best_model_path, map_location=device))

model.eval()
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        break




## === cell 30
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



## === cell 31
test_dir = "/content/denoising_data/test"
test_dir



## === cell 32
best_model_path = "best_model.pth"
model.load_state_dict(torch.load(best_model_path, map_location=device))

model.eval()
all_outputs = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        outputs = model(batch)
        all_outputs.append(outputs.cpu())

all_outputs = torch.cat(all_outputs, dim=0)
print("all_outputs shape:", all_outputs.shape)



## === cell 33
import torchvision.transforms.functional as TF  # padding/cropping ops

target_size = (420, 540)  # (H, W)


def compute_padding(orig_size, target_size):
    orig_height, orig_width = orig_size
    target_height, target_width = target_size
    pad_top = (target_height - orig_height) // 2 if target_height > orig_height else 0
    pad_bottom = (
        (target_height - orig_height - pad_top) if target_height > orig_height else 0
    )
    pad_left = (target_width - orig_width) // 2 if target_width > orig_width else 0
    pad_right = (
        (target_width - orig_width - pad_left) if target_width > orig_width else 0
    )
    return pad_top, pad_bottom, pad_left, pad_right


def remove_padding(pred, orig_size, target_size):
    orig_height, orig_width = orig_size
    pad_top, _, pad_left, _ = compute_padding(orig_size, target_size)
    return pred[:, pad_top : pad_top + orig_height, pad_left : pad_left + orig_width]


test_file_paths = sorted(
    [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".png")],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
)

sample_path = "/kaggle/input/denoising-dirty-documents/sampleSubmission.csv"
if not os.path.exists(sample_path):
    sample_path = "/kaggle/input/sampleSubmission.csv"
sample_sub = pd.read_csv(sample_path)

pred_by_image = {}
for i, file_path in enumerate(test_file_paths):
    image_id = os.path.splitext(os.path.basename(file_path))[0]

    orig_img = Image.open(file_path).convert("L")
    orig_width, orig_height = orig_img.size
    orig_size = (orig_height, orig_width)

    pred = all_outputs[i]
    cropped_pred = remove_padding(pred, orig_size, target_size)
    pred_by_image[image_id] = cropped_pred.squeeze(0).numpy().astype(np.float32)


def value_for_id(pixel_id: str) -> float:
    img_id, r, c = pixel_id.split("_")
    r = int(r) - 1
    c = int(c) - 1
    v = float(pred_by_image[img_id][r, c])
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0
    return v


sample_sub["value"] = sample_sub["id"].map(value_for_id).astype(np.float32)

submission_file = "submission.csv"
sample_sub.to_csv(submission_file, index=False)
print(
    f"Submission file '{submission_file}'이(가) 생성되었습니다. rows={len(sample_sub)} cols={list(sample_sub.columns)}"
)
print(sample_sub.head())
