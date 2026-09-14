# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.13

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
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
tqdm==4.67.1

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

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from concurrent.futures import ThreadPoolExecutor  # parallel image loading

torch.manual_seed(42)
np.random.seed(42)

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")  # improve matmul speed on newer GPUs




## === cell 1
num_tta = 5
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
test_csv_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)




## === cell 2
test_df = pd.read_csv(test_csv_path)
print("Test rows:", test_df.shape[0])




## === cell 3
effnet_inference_transform = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 4
def _load_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image {path} not found")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class CassavaTestDataset(Dataset):
    """
    Pre‑load all test images once (using a thread pool) to avoid per‑batch disk I/O.
    This change is purely an I/O optimisation; the returned tensors and ordering stay identical.
    """

    def __init__(self, dataframe, image_dir):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.image_paths = [
            os.path.join(self.image_dir, img_name)
            for img_name in self.dataframe.iloc[:, 0]
        ]

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            self.images = list(executor.map(_load_image, self.image_paths))

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  # image_id column
        image = self.images[idx]  # already loaded
        return image, img_name




## === cell 5
tta_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Resize(384, 384, p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 6
def tta_predict_single_model(model, images, tta_transform, device, n_tta=5):
    """
    `images` can be a single NumPy array (H,W,3) or a list/tuple of such arrays.
    Returns a Tensor of shape (B, num_classes) where B = number of images.
    """
    if isinstance(images, np.ndarray):
        images = [images]  # treat single image as batch of size 1
    batch_size = len(images)

    aug_batches = []
    for _ in range(n_tta):
        transformed = [tta_transform(image=img)["image"] for img in images]
        aug_batches.append(torch.stack(transformed))  # (B, C, H, W)

    aug_tensor = torch.cat(aug_batches, dim=0).to(device, non_blocking=True)

    with torch.no_grad(), torch.cuda.amp.autocast():
        outputs = model(aug_tensor)  # (B * n_tta, num_classes)
        probs = F.softmax(outputs, dim=1)  # (B * n_tta, num_classes)

    probs = probs.view(n_tta, batch_size, -1).mean(dim=0)  # (B, num_classes)
    return probs




## === cell 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

efficientnet_model_7 = models.efficientnet_v2_s(
    weights=EfficientNet_V2_S_Weights.DEFAULT
)
num_features = efficientnet_model_7.classifier[1].in_features
efficientnet_model_7.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features, 5)
)
efficientnet_model_7 = efficientnet_model_7.to(device)

efficientnet_model_7 = torch.compile(efficientnet_model_7)




## === cell 8
class CassavaTrainDataset(Dataset):
    """
    Pre‑load all raw training images into memory (using a thread pool) to avoid
    per‑epoch disk reads. Transformations are still applied lazily per __getitem__,
    preserving exact same augmentation behavior as before.
    """

    def __init__(self, dataframe, image_dir, transforms):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_paths = [
            os.path.join(self.image_dir, img_name)
            for img_name in self.dataframe["image_id"]
        ]

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            self.images = list(executor.map(_load_image, self.image_paths))

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        label = int(row["label"])
        image = self.images[idx]  # already in RAM
        augmented = self.transforms(image=image)
        img_tensor = augmented["image"]  # torch.Tensor
        return img_tensor, label


train_transform = A.Compose(
    [
        A.RandomResizedCrop(size=(384, 384), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 9
train_df = pd.read_csv(train_csv_path)
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

train_split_df, val_split_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=42,
    stratify=train_df["label"],
)

train_dataset = CassavaTrainDataset(
    dataframe=train_split_df,
    image_dir=train_image_dir,
    transforms=train_transform,
)

val_dataset = CassavaTrainDataset(
    dataframe=val_split_df,
    image_dir=train_image_dir,
    transforms=effnet_inference_transform,  # deterministic resize/normalize
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,  # larger batch reduces iteration count (memory permitting)
    shuffle=True,
    num_workers=4,  # fewer workers now that images are in RAM
    persistent_workers=True,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=256,  # larger batch for faster validation
    shuffle=False,
    num_workers=0,  # no extra workers needed (images pre‑loaded)
    persistent_workers=False,
    pin_memory=True,
)




## === cell 10
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(efficientnet_model_7.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()  # mixed‑precision scaler

epochs = 12  # training epochs remain unchanged
for epoch in range(1, epochs + 1):
    efficientnet_model_7.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = efficientnet_model_7(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    efficientnet_model_7.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = efficientnet_model_7(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch}: Train loss {epoch_loss:.4f} | Val acc {val_acc:.4f}")




## === cell 11
efficientnet_model_7.eval()  # ensure eval mode before inference

test_dataset = CassavaTestDataset(test_df, test_image_dir)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,  # larger batch reduces number of forward passes
    shuffle=False,
    num_workers=4,
    persistent_workers=True,
    collate_fn=lambda batch: batch,  # identity collate (simple)
    pin_memory=True,
)

ensemble_predictions = []
image_names = []

with torch.no_grad():
    for batch in test_loader:
        images, names = zip(*batch)
        probs = tta_predict_single_model(
            efficientnet_model_7, list(images), tta_transform, device, n_tta=num_tta
        )  # (B, num_classes)
        preds = probs.argmax(dim=1).cpu().tolist()
        ensemble_predictions.extend(preds)
        image_names.extend(names)




## === cell 12
submission_df = pd.DataFrame({"image_id": image_names, "label": ensemble_predictions})
submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")
