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

0.79447

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05493) has done: 'I fix the Albumentations `RandomResizedCrop` API break by switching to the v2 signature (tuple `size=(h, w)`), which unblocks TTA creation. Then I fix the missing checkpoint paths by removing dependencies on non-existent `/kaggle/input/*` model weight files and instead run a lightweight, deterministic EfficientNetV2-S inference using built-in torchvision weights, keeping the same overall inference/TTA structure. Finally, I fix the submission length/alignment bug by ensuring we write exactly one prediction per row in `sample_submission.csv` and that `image_id` stays a plain string (not a nested tuple/list), producing a valid `submission.csv`.'
- What this solution (achieved 0.7713) has done: 'The main timeout comes from test-time augmentation doing *two* forward passes through EfficientNet per TTA (one via `model(augmented)` and again via `extract_features(model, augmented)`), plus storing unused 1000-class probabilities. I make inference compute features and 1000-class logits in a *single* backbone forward per TTA by reusing the same feature tensor to produce logits (using the existing classifier head), preserving identical semantics. I also enable CPU-side OpenCV thread control and faster DataLoader settings (more workers where safe, persistent workers, pinned memory, prefetch) to reduce I/O stalls without changing any training/inference math. Finally, I keep determinism for training, while allowing CuDNN benchmark only in fixed-shape inference as you already intended.'
- What this solution (achieved 0.79447) has done: 'You’re currently well below the target (0.7713 vs 0.88395), so we should make small, legitimate changes that improve accuracy without changing your core approach (frozen EfficientNetV2-S feature extractor + trained linear head + TTA). The biggest accuracy issue is that you train the head on *plain resized* images but infer on *heavily augmented RandomResizedCrop/Dropout/etc.*, creating a train–test transform mismatch that hurts performance; we make TTA use the same “base” preprocessing as training (Resize+Normalize) and keep only light, label-preserving flips/rotations. We also apply the head to each TTA sample’s features and average probabilities across TTAs (instead of averaging features), which is a minimal inference-time change that usually improves accuracy for a linear head while keeping the same model and loss. Finally, we train the head a bit longer (still head-only, same optimizer/loss) to move score upward toward the target band.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
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

        return image, img_name




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
tta_transform = A.Compose(
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
    tta_predictions = []

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected image as np.ndarray, got {type(image)}")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Image must have shape (H, W, 3)")

    with torch.no_grad():
        for _ in range(n_tta):
            augmented = tta_transform(image=image)["image"]
            augmented = augmented.unsqueeze(0).to(device)  # (1, C, H, W)
            output = model(augmented)
            probs = F.softmax(output, dim=1)
            tta_predictions.append(probs)

    avg_probs = torch.mean(torch.stack(tta_predictions), dim=0)
    return avg_probs




## === cell 9
def identity_collate(batch):
    return batch


test_dataset = CassavaTestDataset(test_df, test_image_dir)

_cpu = os.cpu_count() or 1
_num_workers = min(8, _cpu)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
    collate_fn=identity_collate,
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
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

val_transform = train_transform

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
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)


def evaluate_acc():
    head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            feats = extract_features(model, x)
            logits = head(feats)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    head.train()
    return correct / max(1, total)


epochs = 8
head.train()
for ep in range(1, epochs + 1):
    pbar = tqdm(
        train_loader, total=len(train_loader), desc=f"Train head epoch {ep}/{epochs}"
    )
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.no_grad():
            feats = extract_features(model, x)

        logits = head(feats)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=float(loss.detach().cpu()))

val_acc = evaluate_acc()
print(f"Validation accuracy (head-only): {val_acc:.4f}")



## === cell 15
ensemble_predictions = []
image_names = []

model.eval()
head.eval()

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    model = model.to(memory_format=torch.channels_last)
    head = head.to(memory_format=torch.channels_last)

with torch.inference_mode():
    for batch in tqdm(test_loader, total=len(test_loader)):
        image, img_name = batch[0]  # (image, img_name)
        if isinstance(img_name, (tuple, list)):
            img_name = img_name[0]
        img_name = str(img_name)

        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected image as np.ndarray, got {type(image)}")
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Image must have shape (H, W, 3)")

        tta_probs_5 = []
        tta_probs_1000 = (
            []
        )  # kept to preserve evaluation semantics (though unused later)

        for _ in range(num_tta):
            augmented = tta_transform(image=image)["image"].unsqueeze(0)  # CPU tensor

            if torch.cuda.is_available():
                augmented = augmented.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last
                )
            else:
                augmented = augmented.to(device)

            feats = extract_features(model, augmented)  # (1,1280)
            out_1000 = model.classifier(feats)  # (1,1000), same as model(augmented)
            probs_1000 = F.softmax(out_1000, dim=1)
            tta_probs_1000.append(probs_1000)

            logits_5 = head(feats)
            probs_5 = F.softmax(logits_5, dim=1)
            tta_probs_5.append(probs_5)

        probs_1000 = torch.mean(torch.stack(tta_probs_1000, dim=0), dim=0)  # (1,1000)
        avg_probs_5 = torch.mean(torch.stack(tta_probs_5, dim=0), dim=0)  # (1,5)

        final_pred = int(avg_probs_5.argmax(dim=1).cpu().item())
        ensemble_predictions.append(final_pred)
        image_names.append(img_name)

len(ensemble_predictions), len(image_names), len(test_df)



## === cell 16
pred_map = dict(zip(image_names, ensemble_predictions))
ordered_preds = [int(pred_map[iid]) for iid in test_df["image_id"].astype(str).tolist()]

submission_df = pd.DataFrame(
    {"image_id": test_df["image_id"].astype(str), "label": ordered_preds}
)

assert len(submission_df) == len(test_df)
submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")
submission_df.head()
