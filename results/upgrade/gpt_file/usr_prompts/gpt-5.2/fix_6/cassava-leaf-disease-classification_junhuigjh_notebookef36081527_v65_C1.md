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

No external packages required in the script and installed.

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

0.8691447567240859

# 6. Current score

0.63901

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.60277) has done: 'I fix the runtime crash by resizing inputs for `vit_b_16` to its required `image_size` (224) instead of 518, which currently triggers the assertion error. I also correct a subtle dataset bug where the label check uses `"label" in row` (checks values, not index), which can silently mis-handle labels; this is score-neutral but improves correctness. Finally, I keep the rest of the pipeline identical so it runs end-to-end and writes a valid `submission.csv` in the required format.'
- What this solution (achieved 0.63901) has done: 'I fix the crash by making the ViT feature extractor compatible with the installed torchvision `vit_b_16` implementation (it doesn’t have `ln`; the final norm is `encoder.ln`). I keep the rest of the pipeline the same (same models, same feature concatenation, same DecisionTree training/prediction), only adjusting the ViT embedding function to be robust across torchvision versions. I also remove the accidental duplicate placeholder definition of `vit_b16_embed` to avoid confusion and ensure the correct function is used. This should run end-to-end and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.63901) has done: 'You’re currently training the DecisionTree on raw concatenated CNN logits + ResNet/ViT embeddings where the feature scales are wildly different, so the tree’s split decisions are dominated by whichever block has larger numeric ranges. To move accuracy up toward your target with minimal disruption, I add a deterministic feature standardization step (fit on train features, apply to test features) while keeping the exact same models, embeddings, and DecisionTree hyperparameters. This preserves the core logic (same feature extractors + same classifier) but improves the tree’s ability to use all feature blocks. I also ensure the scaler uses float32 for speed and stable memory use and keep the submission writing identical.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, vit_b_16
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

NUM_CLASSES = 5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

torch_transforms_512 = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

torch_transforms_vit = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)




## === cell 1
class SimpleCNN512(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def build_model1():
    return SimpleCNN512(NUM_CLASSES)


def build_model2():
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.eval()
    return m


def build_model3():
    m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    m.eval()
    return m


model1 = build_model1().to(device).eval()
model2 = build_model2().to(device).eval()
model3 = build_model3().to(device).eval()


@torch.no_grad()
def resnet18_embed(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = m.conv1(x)
    x = m.bn1(x)
    x = m.relu(x)
    x = m.maxpool(x)

    x = m.layer1(x)
    x = m.layer2(x)
    x = m.layer3(x)
    x = m.layer4(x)

    x = m.avgpool(x)
    x = torch.flatten(x, 1)
    return x


@torch.no_grad()
def vit_b16_embed(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(m, "forward_features"):
        feats = m.forward_features(x)
        if isinstance(feats, torch.Tensor):
            return feats

    x = m._process_input(x)  # [B, num_patches, hidden_dim]
    n = x.shape[0]
    batch_class_token = m.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)
    x = m.encoder(x)

    if hasattr(m, "encoder") and hasattr(m.encoder, "ln"):
        x = m.encoder.ln(x)
    elif hasattr(m, "ln"):
        x = m.ln(x)

    return x[:, 0]




## === cell 2
class CassavaImageDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        path = os.path.join(self.img_dir, image_id)
        img = Image.open(path).convert("RGB")
        x = self.transform(img)

        y = row["label"] if "label" in self.df.columns else -1
        return image_id, x, int(y)


@torch.no_grad()
def extract_features(df, img_dir, batch_size=32):
    ds_512 = CassavaImageDataset(df, img_dir, torch_transforms_512)
    ds_vit = CassavaImageDataset(df, img_dir, torch_transforms_vit)

    loader_512 = DataLoader(
        ds_512,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    loader_vit = DataLoader(
        ds_vit,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    feats = []
    image_ids = []

    for (ids1, x512, _), (ids3, xvit, _) in zip(loader_512, loader_vit):
        if list(ids1) != list(ids3):
            raise RuntimeError(
                "Dataset ordering mismatch between 512 and VIT transforms."
            )
        image_ids.extend(list(ids1))

        x512 = x512.to(device, non_blocking=True)
        xvit = xvit.to(device, non_blocking=True)

        p1 = model1(x512).detach().cpu().numpy().astype(np.float32)  # [B, 5]
        p2 = (
            resnet18_embed(model2, x512).detach().cpu().numpy().astype(np.float32)
        )  # [B, 512]
        p3 = (
            vit_b16_embed(model3, xvit).detach().cpu().numpy().astype(np.float32)
        )  # [B, 768]

        feats.append(np.concatenate([p1, p2, p3], axis=1))

    feats = (
        np.vstack(feats)
        if len(feats)
        else np.zeros((0, 5 + 512 + 768), dtype=np.float32)
    )
    return image_ids, feats




## === cell 3
train_df = pd.read_csv(TRAIN_CSV)
train_ids, train_feats = extract_features(
    train_df[["image_id", "label"]], TRAIN_IMG_DIR, batch_size=32
)
train_labels = train_df["label"].to_numpy()

if len(train_feats) != len(train_labels):
    raise RuntimeError(
        f"Feature/label length mismatch: feats={len(train_feats)} labels={len(train_labels)}"
    )

scaler = StandardScaler(with_mean=True, with_std=True)
train_feats_scaled = scaler.fit_transform(train_feats).astype(np.float32, copy=False)

decision_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=9,
    min_samples_split=9,
    random_state=SEED,
)
decision_tree.fit(train_feats_scaled, train_labels)



## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB)
test_df = sample_sub[["image_id"]].copy()
test_df["label"] = -1  # placeholder for dataset API

test_ids, test_feats = extract_features(
    test_df[["image_id", "label"]], TEST_IMG_DIR, batch_size=32
)

id_to_row = {img_id: i for i, img_id in enumerate(test_ids)}
order_idx = []
missing = []
for img_id in sample_sub["image_id"].tolist():
    if img_id in id_to_row:
        order_idx.append(id_to_row[img_id])
    else:
        missing.append(img_id)

if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images on disk. First few: {missing[:5]}"
    )

test_feats_ordered = test_feats[np.array(order_idx)]

test_feats_scaled = scaler.transform(test_feats_ordered).astype(np.float32, copy=False)

prediction = decision_tree.predict(test_feats_scaled).astype(int)

submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": prediction}
)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print(
    f"Wrote submission.csv with shape={submission.shape} to {os.path.abspath('submission.csv')}"
)
