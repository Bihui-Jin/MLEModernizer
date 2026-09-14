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

# 5. Code solution

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
from torchvision.transforms import v2
from torchvision.models import (
    vit_h_14,
    efficientnet_v2_l,
    densenet121,
    ViT_H_14_Weights,
    EfficientNet_V2_L_Weights,
    DenseNet121_Weights,
)

from sklearn.ensemble import RandomForestClassifier

os.environ["PYTHONHASHSEED"] = "0"


def seed_everything(seed: int = 11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(11)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing dir: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing dir: {TEST_IMG_DIR}"

N_CLASSES = 5

_CPU_COUNT = os.cpu_count() or 2

DL_WORKERS = min(8, max(2, _CPU_COUNT // 2))
print("DataLoader workers:", DL_WORKERS)

torch.set_num_threads(max(1, min(8, _CPU_COUNT)))
torch.set_num_interop_threads(1)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

Image.MAX_IMAGE_PIXELS = None
Image.LOAD_TRUNCATED_IMAGES = True




## === cell 1
DN_WEIGHTS = DenseNet121_Weights.DEFAULT
VIT_WEIGHTS = ViT_H_14_Weights.DEFAULT
EFF_WEIGHTS = EfficientNet_V2_L_Weights.DEFAULT


def _get_mean_std(
    weights, fallback_mean=(0.485, 0.456, 0.406), fallback_std=(0.229, 0.224, 0.225)
):
    meta = getattr(weights, "meta", {}) or {}
    mean = meta.get("mean", None)
    std = meta.get("std", None)
    if mean is None or std is None:
        return list(fallback_mean), list(fallback_std)
    return list(mean), list(std)


_dn_mean, _dn_std = _get_mean_std(DN_WEIGHTS)
_vit_mean, _vit_std = _get_mean_std(VIT_WEIGHTS)
_eff_mean, _eff_std = _get_mean_std(EFF_WEIGHTS)

_t_pil_to_tensor = transforms.functional.pil_to_tensor
_t_pad = transforms.functional.pad

torch_transforms_ResNet = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_dn_mean, std=_dn_std),
    ]
)


def invert_square_pad_tensor(img):
    width, height = img.size
    img_t = _t_pil_to_tensor(img)  # uint8 CHW
    img_t = torch.roll(img_t, shifts=(height // 2, width // 2), dims=(1, 2))
    max_side = max(width, height)
    pad_l = (max_side - width) // 2
    pad_t = (max_side - height) // 2
    pad_r = (max_side - width) - pad_l
    pad_b = (max_side - height) - pad_t
    img_t = _t_pad(img_t, [pad_l, pad_t, pad_r, pad_b], padding_mode="reflect")
    return img_t  # uint8 CHW


VIT_IMAGE_SIZE = 224
torch_transforms_VIT = transforms.Compose(
    [
        v2.Lambda(invert_square_pad_tensor),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((VIT_IMAGE_SIZE, VIT_IMAGE_SIZE)),
        v2.Normalize(_vit_mean, _vit_std),
    ]
)

torch_transforms_EfficientNet = transforms.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((480, 480)),
        v2.Normalize(_eff_mean, _eff_std),
    ]
)

print("Transforms initialized (ViT size placeholder =", VIT_IMAGE_SIZE, ").")




## === cell 2
def build_models(num_classes=5, device=device):
    m_resnet_like = densenet121(weights=DN_WEIGHTS)
    m_resnet_like.classifier = nn.Linear(
        m_resnet_like.classifier.in_features, num_classes
    )

    m_vit = vit_h_14(weights=VIT_WEIGHTS)
    if hasattr(m_vit, "heads") and hasattr(m_vit.heads, "head"):
        in_f = m_vit.heads.head.in_features
        m_vit.heads.head = nn.Linear(in_f, num_classes)
    else:
        m_vit.heads = nn.Sequential(nn.Linear(m_vit.hidden_dim, num_classes))

    m_eff = efficientnet_v2_l(weights=EFF_WEIGHTS)
    if isinstance(m_eff.classifier, nn.Sequential):
        in_f = m_eff.classifier[-1].in_features
        m_eff.classifier[-1] = nn.Linear(in_f, num_classes)
    else:
        in_f = m_eff.classifier.in_features
        m_eff.classifier = nn.Linear(in_f, num_classes)

    m_aux = densenet121(weights=DN_WEIGHTS)
    m_aux.classifier = nn.Linear(m_aux.classifier.in_features, num_classes)

    for m in (m_resnet_like, m_vit, m_eff, m_aux):
        m.to(device)
        m.eval()

    if torch.cuda.is_available():
        m_resnet_like = m_resnet_like.to(memory_format=torch.channels_last)
        m_aux = m_aux.to(memory_format=torch.channels_last)
        m_eff = m_eff.to(memory_format=torch.channels_last)

    return m_resnet_like, m_aux, m_vit, m_eff


model1, model2, model3, model4 = build_models(N_CLASSES, device)
print("Models initialized (eval mode).")

VIT_IMAGE_SIZE = int(getattr(model3, "image_size", 224))
torch_transforms_VIT = transforms.Compose(
    [
        v2.Lambda(invert_square_pad_tensor),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((VIT_IMAGE_SIZE, VIT_IMAGE_SIZE)),
        v2.Normalize(_vit_mean, _vit_std),
    ]
)
print("Adjusted ViT transform resize to model3.image_size =", VIT_IMAGE_SIZE)

softmax = nn.Softmax(dim=1)




## === cell 3
class CassavaImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, return_label: bool):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.return_label = return_label

        self._image_ids = self.df["image_id"].to_numpy()
        self._labels = (
            self.df["label"].to_numpy(dtype=np.int64) if return_label else None
        )

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, idx: int):
        image_id = self._image_ids[idx]
        img_path = os.path.join(self.img_dir, image_id)

        with Image.open(img_path) as im:
            img = im.convert("RGB")

        x1 = torch_transforms_ResNet(img)
        x3 = torch_transforms_VIT(img)
        x4 = torch_transforms_EfficientNet(img)

        if self.return_label:
            y = int(self._labels[idx])
            return image_id, x1, x3, x4, y
        return image_id, x1, x3, x4


@torch.inference_mode()
def predict_features(dataloader: DataLoader):
    image_ids = []
    feats = []
    labels = []

    use_cuda = torch.cuda.is_available()

    for batch in dataloader:
        if len(batch) == 5:
            b_ids, x1, x3, x4, y = batch
            if torch.is_tensor(y):
                labels.append(y.detach().cpu().numpy().astype(np.int64))
            else:
                labels.append(np.asarray(y, dtype=np.int64))
        else:
            b_ids, x1, x3, x4 = batch

        if use_cuda:
            x1 = x1.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            x4 = x4.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            x3 = x3.to(device, non_blocking=True)
        else:
            x1 = x1.to(device)
            x3 = x3.to(device)
            x4 = x4.to(device)

        p1 = softmax(model1(x1)).cpu().numpy()
        p2 = p1  # preserve original semantics exactly
        p3 = softmax(model3(x3)).cpu().numpy()
        p4 = softmax(model4(x4)).cpu().numpy()

        f = np.concatenate([p1, p2, p3, p4], axis=1)
        feats.append(f)
        image_ids.extend(list(b_ids))

    feats = np.concatenate(feats, axis=0)
    if labels:
        labels = np.concatenate(labels, axis=0).reshape(-1)
        return image_ids, feats, labels
    return image_ids, feats




## === cell 4
train_df = pd.read_csv(TRAIN_CSV)
train_df = train_df[["image_id", "label"]].copy()

train_df_fit = train_df

BATCH_SIZE = 16

train_ds = CassavaImageDataset(train_df_fit, TRAIN_IMG_DIR, return_label=True)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=DL_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(DL_WORKERS > 0),
    prefetch_factor=2 if DL_WORKERS > 0 else None,
)

train_ids, train_feats, train_labels = predict_features(train_loader)
print(
    "Train features:",
    train_feats.shape,
    "labels:",
    train_labels.shape,
    train_labels.dtype,
)

decision_tree = RandomForestClassifier(
    n_estimators=40, criterion="gini", max_depth=8, random_state=11, n_jobs=-1
)
decision_tree.fit(train_feats, train_labels)
print("Stacker trained.")




## === cell 5
sample_sub = pd.read_csv(SAMPLE_SUB)
test_df = sample_sub[["image_id"]].copy()

test_ds = CassavaImageDataset(test_df, TEST_IMG_DIR, return_label=False)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=DL_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(DL_WORKERS > 0),
    prefetch_factor=2 if DL_WORKERS > 0 else None,
)

test_ids, test_feats = predict_features(test_loader)
print("Test features:", test_feats.shape, "test ids:", len(test_ids))

prediction = decision_tree.predict(test_feats).astype(int)

submission = pd.DataFrame({"image_id": test_ids, "label": prediction})
submission = submission.merge(sample_sub[["image_id"]], on="image_id", how="right")
submission["label"] = submission["label"].fillna(0).astype(int)

assert (
    submission.shape[0] == sample_sub.shape[0]
), "Submission row count mismatch vs sample_submission."
submission = submission[["image_id", "label"]]
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
