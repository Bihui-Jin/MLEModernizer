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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.866424901783016

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.09342) has done: 'I fixed the model initialization to use the new torchvision API (loading ImageNet weights then replacing the classifier) and made the EfficientNet load safely even when the custom checkpoint is missing. I also removed the unavailable `Cutout` transform from the Albumentations pipeline, which caused the augmentation definition to fail. With these corrections the script runs end‑to‑end and writes a proper `submission.csv` that can be submitted to Kaggle.'
- What this solution (achieved 0.72982) has done: 'I keep the original pipeline but add a lightweight fine‑tuning phase that trains only the classifier head of the EfficientNet‑B4 on the provided training set (freezing the feature extractor). This small amount of training should dramatically raise the accuracy from the random‑guess baseline while preserving the core model architecture. The new cells load the training CSV, create a simple Albumentations‑based dataset, run a short training loop, and then perform the same TTA inference and CSV export as before.'
- What this solution (achieved 0.71263) has done: 'I increase the fine‑tuning duration (6 epochs) with a smaller learning rate and a StepLR scheduler, add a reproducible seed, use a lightweight validation augmentation (Resize + Normalize) so the model is evaluated on data processed like training, and fix the test‑time preprocessing to match the training pipeline (remove the extra ToTensor that caused double normalization). These minimal changes keep the EfficientNet‑B4 architecture unchanged while expectedly raising accuracy toward the target score.'
- What this solution (achieved 0.68984) has done: 'I cache image data in memory to eliminate repeated disk I/O during training and inference, and increase DataLoader workers for parallel loading. These changes keep the model architecture, training loops, and augmentations unchanged while dramatically reducing I/O overhead, allowing the script to finish well within the 600‑second limit.'
- What this solution (achieved 0.69245) has done: 'The changes keep the exact model, training, and augmentation logic but speed up data loading and inference.  
1. Added `persistent_workers=True` to the DataLoaders so worker processes stay alive across epochs, removing their recreation overhead.  
2. Re‑implemented test‑time inference using a batched `TestDataset` and a DataLoader, applying the deterministic validation augmentation once per image per TTA round and accumulating logits in batches. This eliminates the per‑image Python loop and reduces overhead while preserving the five‑fold TTA averaging.'

# 9. Code solution

## === cell 0
try:
    import subprocess, sys, pathlib

    wheel_path = pathlib.Path(
        "../input/efficientnet-pytorch-070/efficientnet_pytorch-0.7.0-py3-none-any.whl"
    )
    if wheel_path.is_file():
        subprocess.check_call([sys.executable, "-m", "pip", "install", str(wheel_path)])
except Exception:
    pass



## === cell 1
import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

import albumentations as A
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
np.random.seed(42)

torch.backends.cudnn.benchmark = True



## === cell 2
model_path = "../input/en-b4-tta-calr-8/model(12).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_images_path = "../input/cassava-leaf-disease-classification/train_images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

skip_training = False  # flag to control training

if os.path.isfile(model_path):
    model = efficientnet_b4(pretrained=False, num_classes=5)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    skip_training = True
else:
    model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 5)
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.features[-1].parameters():
        param.requires_grad = True

model.to(device)
model.eval()



## === cell 3
sub_aug = A.Compose(
    [
        A.Resize(height=256, width=256),
        A.Transpose(p=0.8),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.8),
        A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=0, p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        A.CoarseDropout(max_holes=20, max_height=10, max_width=10, p=0.5),
    ],
    p=1.0,
)

val_aug = A.Compose(
    [
        A.Resize(height=256, width=256),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)




## === cell 4
class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, aug):
        self.df = df.reset_index(drop=True)
        self.aug = aug
        self.img_dir = img_dir
        self.paths = (
            self.df["image_id"].apply(lambda x: os.path.join(img_dir, x)).tolist()
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.paths[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        if self.aug:
            image = self.aug(image=image)["image"]
        tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        label = int(row["label"])
        return tensor, label


train_df = pd.read_csv(train_csv_path)
train_split, val_split = train_test_split(
    train_df, test_size=0.1, stratify=train_df["label"], random_state=42
)

train_dataset = CassavaDataset(train_split, train_images_path, sub_aug)
val_dataset = CassavaDataset(val_split, train_images_path, val_aug)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

if not skip_training:
    scaler = torch.cuda.amp.GradScaler()  # for mixed‑precision
    model.train()
    epochs = 20  # modestly extended training
    for epoch in range(epochs):
        running_loss = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        val_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1}: train loss={epoch_loss:.4f}, val acc={val_acc:.4f}")

        model.train()
        scheduler.step()

model.eval()  # ensure model is in eval mode before inference



## === cell 5
sample_sub = pd.read_csv(sample_sub_path)


class TestDataset(Dataset):
    """
    Loads all test images into memory once to avoid repeated disk I/O.
    Returns the raw numpy image (H, W, C) for each sample.
    """

    def __init__(self, df, img_dir):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.paths = (
            self.df["image_id"].apply(lambda x: os.path.join(img_dir, x)).tolist()
        )
        self.images = [np.array(Image.open(p).convert("RGB")) for p in self.paths]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, "image_id"]
        image = self.images[idx]
        return img_id, image


test_dataset = TestDataset(sample_sub, test_images_path)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(
    1, 3, 1, 1
)
std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(
    1, 3, 1, 1
)


def preprocess_batch(batch_images):
    """
    Vectorized batch preprocessing: resize to 256×256 and normalize.
    Accepts a list of numpy uint8 images, returns a torch tensor on the target device.
    """
    batch_np = np.stack(batch_images)  # (B, H, W, C)
    batch_tensor = (
        torch.from_numpy(batch_np).permute(0, 3, 1, 2).float() / 255.0
    )  # (B, C, H, W)
    batch_resized = F.interpolate(
        batch_tensor, size=(256, 256), mode="bilinear", align_corners=False
    )
    batch_norm = (batch_resized - mean) / std
    return batch_norm.to(device)


logits_sum = {
    img_id: torch.zeros(5, device=device) for img_id in sample_sub["image_id"]
}

model.eval()
with torch.no_grad():
    for batch_ids, batch_images in test_loader:
        tensor_batch = preprocess_batch(batch_images)  # (B,3,256,256)
        outputs = model(tensor_batch)  # (B,5)

        for idx, img_id in enumerate(batch_ids):
            logits_sum[img_id] += outputs[idx]

predictions = []
for img_id in sample_sub["image_id"]:
    avg_logits = logits_sum[img_id] / 5.0  # average over the (implicit) 5‑fold TTA
    _, pred_label = torch.max(avg_logits, dim=0)
    predictions.append([img_id, int(pred_label.item())])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])
sub_df.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
print(sub_df.head())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1860667846.py in <cell line: 0>()
     67 with torch.no_grad():
     68     for batch_ids, batch_images in test_loader:
---> 69         tensor_batch = preprocess_batch(batch_images)  # (B,3,256,256)
     70         outputs = model(tensor_batch)  # (B,5)
     71 

/tmp/ipykernel_55/1860667846.py in preprocess_batch(batch_images)
     56         batch_tensor, size=(256, 256), mode="bilinear", align_corners=False
     57     )
---> 58     batch_norm = (batch_resized - mean) / std
     59     return batch_norm.to(device)
     60 

RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
