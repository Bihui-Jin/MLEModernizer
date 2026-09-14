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

# 5. Target score

0.8839528558476881

# 6. Current score

0.5426

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05493) has done: 'I fix the Albumentations `RandomResizedCrop` API break by switching to the v2 signature (tuple `size=(h, w)`), which unblocks TTA creation. Then I fix the missing checkpoint paths by removing dependencies on non-existent `/kaggle/input/*` model weight files and instead run a lightweight, deterministic EfficientNetV2-S inference using built-in torchvision weights, keeping the same overall inference/TTA structure. Finally, I fix the submission length/alignment bug by ensuring we write exactly one prediction per row in `sample_submission.csv` and that `image_id` stays a plain string (not a nested tuple/list), producing a valid `submission.csv`.'
- What this solution (achieved 0.7713) has done: 'The main timeout comes from test-time augmentation doing *two* forward passes through EfficientNet per TTA (one via `model(augmented)` and again via `extract_features(model, augmented)`), plus storing unused 1000-class probabilities. I make inference compute features and 1000-class logits in a *single* backbone forward per TTA by reusing the same feature tensor to produce logits (using the existing classifier head), preserving identical semantics. I also enable CPU-side OpenCV thread control and faster DataLoader settings (more workers where safe, persistent workers, pinned memory, prefetch) to reduce I/O stalls without changing any training/inference math. Finally, I keep determinism for training, while allowing CuDNN benchmark only in fixed-shape inference as you already intended.'
- What this solution (achieved 0.79447) has done: 'You’re currently well below the target (0.7713 vs 0.88395), so we should make small, legitimate changes that improve accuracy without changing your core approach (frozen EfficientNetV2-S feature extractor + trained linear head + TTA). The biggest accuracy issue is that you train the head on *plain resized* images but infer on *heavily augmented RandomResizedCrop/Dropout/etc.*, creating a train–test transform mismatch that hurts performance; we make TTA use the same “base” preprocessing as training (Resize+Normalize) and keep only light, label-preserving flips/rotations. We also apply the head to each TTA sample’s features and average probabilities across TTAs (instead of averaging features), which is a minimal inference-time change that usually improves accuracy for a linear head while keeping the same model and loss. Finally, we train the head a bit longer (still head-only, same optimizer/loss) to move score upward toward the target band.'
- What this solution (achieved 0.5867) has done: 'The main timeout bottleneck is per-image Python TTA augmentation and repeated feature extraction during head training/validation; both cause huge CPU overhead and too many GPU forward passes. I keep the exact model/head/loss/training logic, but cache frozen EfficientNet features for train/val once (identical to recomputing them every epoch) and train the linear head on those cached tensors. For TTA, I remove per-image Python loops by batching albumentations with `ReplayCompose` so each image’s TTA parameters are generated once and then applied consistently, and I use a DataLoader collate that avoids expensive list conversions. These changes are provably equivalent in semantics (same transforms distribution, same frozen backbone outputs, same optimization steps for the head given identical features) while drastically cutting overhead.'
- What this solution (achieved 0.54372) has done: 'Your current score (0.5867) is far below the target (0.88395), so we should make small, high-impact fixes that improve generalization without changing the backbone+linear-head core approach. The biggest issue is that the linear head is trained on CLAHE-processed images while TTA/inference does not use CLAHE, creating a major train–test preprocessing mismatch; I make train/val transforms match inference by removing CLAHE (keeping Resize+Normalize+ToTensorV2). Next, because cassava labels are imbalanced, I switch the head loss to class-weighted CrossEntropy computed from the train split (same loss family/semantics, just better weighting), which typically lifts accuracy substantially for head-only training. Finally, I keep everything else (feature caching, head training loop, TTA batching, submission alignment) unchanged to minimize risk and runtime impact.'
- What this solution (achieved 0.54111) has done: 'We’re far below the target (0.54372 vs 0.88395), so we should make a small, high-impact generalization improvement while keeping your frozen EfficientNetV2-S + linear head + feature caching + TTA pipeline intact. The biggest missing piece is that the head is trained on plain resized images without any label-preserving augmentation, but inference uses TTA with flips/rotate/scale—this mismatch and lack of augmentation typically hurts a head-only setup a lot. I add light, label-preserving training-time augmentations (the same family as your TTA: flip + mild ShiftScaleRotate) while keeping the exact same normalization/ToTensorV2 and the same head training loop/loss. I also keep validation transform deterministic (resize+normalize only) to ensure the val accuracy signal remains stable.'
- What this solution (achieved 0.5426) has done: 'Your current score (0.54111) is far below the target (0.88395), so we should make a minimal but high-impact *training correctness* fix while keeping your frozen EfficientNetV2-S + cached features + linear head + TTA pipeline intact. The biggest issue is that you precompute frozen features using the training loader **with random augmentations**, meaning the same image gets a different feature every time you re-run, and the cached features no longer correspond to a stable dataset—this hurts head fitting and generalization. I precompute features from a deterministic (no-augmentation) view of the train/val images (Resize+Normalize only), while keeping your head training loop/loss/optimizer/scheduler unchanged. This preserves the core logic (still head-only training on frozen EfficientNet features) but makes the cached feature dataset well-defined and typically yields a large accuracy jump toward your target.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights

import cv2
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

try:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass



## === cell 1
num_tta = 5



## === cell 2
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"

train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)



## === cell 3
test_df = pd.read_csv(sample_sub_path)
test_df.head()



## === cell 4
efficientnet_transforms = A.Compose(
    [
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 5
class CassavaTestDataset(Dataset):
    def __init__(self, dataframe, image_dir):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  # image_id
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, str(img_name)




## === cell 6
class CassavaTrainDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image_id"]
        label = int(self.df.loc[idx, "label"])
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image=image)["image"]
        return image, label




## === cell 7
tta_transform = A.ReplayCompose(
    [
        A.Resize(384, 384),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=10,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 8
def tta_predict_single_model(model, image, tta_transform, device, n_tta=5):
    model.eval()
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected image as np.ndarray, got {type(image)}")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Image must have shape (H, W, 3)")

    aug_tensors = [tta_transform(image=image)["image"] for _ in range(n_tta)]
    batch = torch.stack(aug_tensors, dim=0)

    if torch.cuda.is_available():
        batch = batch.to(device, non_blocking=True).to(
            memory_format=torch.channels_last
        )
    else:
        batch = batch.to(device)

    with torch.inference_mode():
        output = model(batch)  # (n_tta, num_classes)
        probs = F.softmax(output, dim=1)
        avg_probs = probs.mean(dim=0, keepdim=True)  # (1, num_classes)
    return avg_probs




## === cell 9
def collate_images_and_names(batch):
    images, names = zip(*batch)
    return list(images), list(names)


def collate_images_and_labels(batch):
    images, labels = zip(*batch)
    return list(images), torch.as_tensor(labels, dtype=torch.long)


test_dataset = CassavaTestDataset(test_df, test_image_dir)

_cpu = os.cpu_count() or 1
_num_workers = min(8, _cpu)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,  # unchanged semantics
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
    collate_fn=collate_images_and_names,
)




## === cell 10
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 11
weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
model = models.efficientnet_v2_s(weights=weights)
model = model.to(device)
model.eval()




## === cell 12
def extract_features(model, x: torch.Tensor) -> torch.Tensor:
    feats = model.features(x)
    feats = model.avgpool(feats)
    feats = torch.flatten(feats, 1)
    return feats


feature_dim = 1280
head = nn.Linear(feature_dim, 5).to(device)

for p in model.parameters():
    p.requires_grad = False
for p in head.parameters():
    p.requires_grad = True



## === cell 13
train_df_full = pd.read_csv(train_csv_path)
train_df, val_df = train_test_split(
    train_df_full,
    test_size=0.1,
    random_state=42,
    stratify=train_df_full["label"],
)

train_transform = A.Compose(
    [
        A.Resize(384, 384),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=10,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

val_transform = efficientnet_transforms

train_ds = CassavaTrainDataset(train_df, train_image_dir, train_transform)
val_ds = CassavaTrainDataset(val_df, train_image_dir, val_transform)

_num_workers_tv = min(8, _cpu)
train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=_num_workers_tv,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers_tv > 0),
    prefetch_factor=4 if _num_workers_tv > 0 else None,
)
val_loader = DataLoader(
    val_ds,
    batch_size=64,
    shuffle=False,
    num_workers=_num_workers_tv,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers_tv > 0),
    prefetch_factor=4 if _num_workers_tv > 0 else None,
)



## === cell 14
_label_counts = train_df["label"].value_counts().sort_index()
_counts = np.zeros(5, dtype=np.int64)
for k, v in _label_counts.items():
    if 0 <= int(k) < 5:
        _counts[int(k)] = int(v)
_counts = np.maximum(_counts, 1)  # safety against any missing class in the split
class_weights = 1.0 / _counts.astype(np.float32)
class_weights = (
    class_weights / class_weights.mean()
)  # normalize scale (keeps loss magnitude stable)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

criterion = nn.CrossEntropyLoss(weight=class_weights_t)
optimizer = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.3)


def _precompute_features(dataloader, desc):
    model.eval()
    feats_all = []
    y_all = []
    with torch.inference_mode():
        for x, y in tqdm(dataloader, total=len(dataloader), desc=desc):
            x = x.to(device, non_blocking=True)
            if torch.cuda.is_available():
                x = x.to(memory_format=torch.channels_last)
            feats = extract_features(model, x).detach()
            feats_all.append(feats.cpu())
            y_all.append(y.cpu())
    feats_all = torch.cat(feats_all, dim=0)
    y_all = torch.cat(y_all, dim=0)
    return feats_all, y_all


train_ds_feat = CassavaTrainDataset(train_df, train_image_dir, efficientnet_transforms)
val_ds_feat = CassavaTrainDataset(val_df, train_image_dir, efficientnet_transforms)

train_loader_feat = DataLoader(
    train_ds_feat,
    batch_size=64,
    shuffle=False,  # deterministic order for caching
    num_workers=_num_workers_tv,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers_tv > 0),
    prefetch_factor=4 if _num_workers_tv > 0 else None,
)
val_loader_feat = DataLoader(
    val_ds_feat,
    batch_size=64,
    shuffle=False,
    num_workers=_num_workers_tv,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers_tv > 0),
    prefetch_factor=4 if _num_workers_tv > 0 else None,
)

train_feats_cpu, train_y_cpu = _precompute_features(
    train_loader_feat, "Precompute train features (deterministic)"
)
val_feats_cpu, val_y_cpu = _precompute_features(
    val_loader_feat, "Precompute val features (deterministic)"
)

head_train_ds = TensorDataset(train_feats_cpu, train_y_cpu)
head_val_ds = TensorDataset(val_feats_cpu, val_y_cpu)

head_train_loader = DataLoader(
    head_train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=0,  # tensors already in memory; workers add overhead
    pin_memory=torch.cuda.is_available(),
)
head_val_loader = DataLoader(
    head_val_ds,
    batch_size=256,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


def evaluate_acc():
    head.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for feats, y in head_val_loader:
            feats = feats.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = head(feats)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    head.train()
    return correct / max(1, total)


epochs = 14
head.train()
for ep in range(1, epochs + 1):
    pbar = tqdm(
        head_train_loader,
        total=len(head_train_loader),
        desc=f"Train head epoch {ep}/{epochs}",
    )
    for feats, y in pbar:
        feats = feats.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = head(feats)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        pbar.set_postfix(
            loss=float(loss.detach()), lr=float(optimizer.param_groups[0]["lr"])
        )

    scheduler.step()

val_acc = evaluate_acc()
print(f"Validation accuracy (head-only, no TTA): {val_acc:.4f}")




## === cell 15
def _tta_batch_to_tensor(images_list, n_tta, device):
    B = len(images_list)
    replays = [
        [tta_transform(image=img)["replay"] for _ in range(n_tta)]
        for img in images_list
    ]
    tta_tensors = []
    for t in range(n_tta):
        imgs_t = [
            A.ReplayCompose.replay(replays[b][t], image=images_list[b])["image"]
            for b in range(B)
        ]
        tta_tensors.append(torch.stack(imgs_t, dim=0))  # (B,C,H,W)
    batch = torch.cat(tta_tensors, dim=0)  # (B*T,C,H,W)
    if torch.cuda.is_available():
        batch = batch.to(device, non_blocking=True).to(
            memory_format=torch.channels_last
        )
    else:
        batch = batch.to(device)
    return batch


def evaluate_val_acc_with_tta_batched(n_tta=5, batch_size=64, num_workers=None):
    model.eval()
    head.eval()

    class _ValReadOnly(Dataset):
        def __init__(self, df, image_dir):
            self.df = df.reset_index(drop=True)
            self.image_dir = image_dir

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            img_name = self.df.loc[idx, "image_id"]
            label = int(self.df.loc[idx, "label"])
            img_path = os.path.join(self.image_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image, label

    ds = _ValReadOnly(val_df, train_image_dir)
    _nw = _num_workers_tv if num_workers is None else num_workers
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(_nw > 0),
        prefetch_factor=4 if _nw > 0 else None,
        collate_fn=collate_images_and_labels,
    )

    correct = 0
    total = 0
    with torch.inference_mode():
        for images_list, labels in loader:
            B = len(images_list)
            labels = labels.to(device, non_blocking=True)

            batch = _tta_batch_to_tensor(images_list, n_tta=n_tta, device=device)
            feats = extract_features(model, batch)  # (B*T,1280)
            logits_5 = head(feats)  # (B*T,5)
            probs_5 = F.softmax(logits_5, dim=1).view(B, n_tta, -1)  # (B,T,5)
            avg_probs_5 = probs_5.mean(dim=1)  # (B,5)

            pred = avg_probs_5.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.numel()

    return correct / max(1, total)


if torch.cuda.is_available():
    val_tta_acc = evaluate_val_acc_with_tta_batched(n_tta=num_tta, batch_size=64)
    print(f"Validation accuracy (with TTA={num_tta}): {val_tta_acc:.4f}")
else:
    subset_val_df = val_df.iloc[:500].copy()

    def evaluate_subset_val_acc_with_tta_batched(n_tta=5, batch_size=32):
        model.eval()
        head.eval()

        class _ValReadOnly(Dataset):
            def __init__(self, df, image_dir):
                self.df = df.reset_index(drop=True)
                self.image_dir = image_dir

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                img_name = self.df.loc[idx, "image_id"]
                label = int(self.df.loc[idx, "label"])
                img_path = os.path.join(self.image_dir, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    raise FileNotFoundError(f"Could not read image: {img_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image, label

        ds = _ValReadOnly(subset_val_df, train_image_dir)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_images_and_labels,
        )

        correct = 0
        total = 0
        with torch.inference_mode():
            for images_list, labels in loader:
                B = len(images_list)
                labels = labels.to(device)

                batch = _tta_batch_to_tensor(images_list, n_tta=n_tta, device=device)
                feats = extract_features(model, batch)
                logits_5 = head(feats)
                probs_5 = F.softmax(logits_5, dim=1).view(B, n_tta, -1)
                avg_probs_5 = probs_5.mean(dim=1)

                pred = avg_probs_5.argmax(dim=1).cpu()
                correct += (pred == labels.cpu()).sum().item()
                total += labels.numel()

        return correct / max(1, total)

    val_tta_acc = evaluate_subset_val_acc_with_tta_batched(n_tta=num_tta, batch_size=32)
    print(
        f"Validation accuracy (with TTA={num_tta}, first 500 only on CPU): {val_tta_acc:.4f}"
    )



## === cell 16
ensemble_predictions = []
image_names = []

model.eval()
head.eval()

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    model = model.to(memory_format=torch.channels_last)
    head = head.to(memory_format=torch.channels_last)

with torch.inference_mode():
    for images_list, img_names in tqdm(test_loader, total=len(test_loader)):
        B = len(images_list)

        batch = _tta_batch_to_tensor(images_list, n_tta=num_tta, device=device)
        feats = extract_features(model, batch)  # (B*T,1280)
        logits_5 = head(feats)  # (B*T,5)
        probs_5 = F.softmax(logits_5, dim=1).view(B, num_tta, -1)  # (B,T,5)
        avg_probs_5 = probs_5.mean(dim=1)  # (B,5)
        preds = avg_probs_5.argmax(dim=1).detach().cpu().tolist()

        ensemble_predictions.extend([int(p) for p in preds])
        image_names.extend([str(n) for n in img_names])

len(ensemble_predictions), len(image_names), len(test_df)



## === cell 17
pred_map = dict(zip(image_names, ensemble_predictions))
ordered_preds = [int(pred_map[iid]) for iid in test_df["image_id"].astype(str).tolist()]

submission_df = pd.DataFrame(
    {"image_id": test_df["image_id"].astype(str), "label": ordered_preds}
)

assert len(submission_df) == len(test_df)
submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")
submission_df.head()
