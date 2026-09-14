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

DL_WORKERS = min(4, max(2, _CPU_COUNT // 2))
print("DataLoader workers:", DL_WORKERS)

torch.set_num_threads(max(1, min(8, _CPU_COUNT)))
torch.set_num_interop_threads(1)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

Image.MAX_IMAGE_PIXELS = None
Image.LOAD_TRUNCATED_IMAGES = True


def _seed_worker(worker_id: int):
    base_seed = 11
    s = base_seed + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)




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

_CAN_COMPILE = hasattr(torch, "compile")
if _CAN_COMPILE:
    try:
        model1 = torch.compile(model1, mode="reduce-overhead", fullgraph=False)
        model3 = torch.compile(model3, mode="reduce-overhead", fullgraph=False)
        model4 = torch.compile(model4, mode="reduce-overhead", fullgraph=False)
        print("torch.compile enabled for inference.")
    except Exception as e:
        print(
            "torch.compile not available/failed; continuing without compile. Error:",
            repr(e),
        )




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

        im = Image.open(img_path)
        try:
            img = im.convert("RGB")
        finally:
            im.close()

        x1 = torch_transforms_ResNet(img)
        x3 = torch_transforms_VIT(img)
        x4 = torch_transforms_EfficientNet(img)

        if self.return_label:
            y = int(self._labels[idx])
            return image_id, x1, x3, x4, y
        return image_id, x1, x3, x4


@torch.inference_mode()
def predict_features(dataloader: DataLoader, has_labels: bool):
    n = len(dataloader.dataset)
    n_feat = N_CLASSES * 4
    feats = np.empty((n, n_feat), dtype=np.float32)
    image_ids = [None] * n
    labels = np.empty((n,), dtype=np.int64) if has_labels else None

    use_cuda = torch.cuda.is_available()
    pos = 0

    max_bs = getattr(dataloader, "batch_size", None) or 1
    batch_buf = np.empty((max_bs, n_feat), dtype=np.float32)

    for batch in dataloader:
        if has_labels:
            b_ids, x1, x3, x4, y = batch
        else:
            b_ids, x1, x3, x4 = batch

        bs = len(b_ids)

        if use_cuda:
            x1 = x1.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            x4 = x4.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            x3 = x3.to(device, non_blocking=True)
        else:
            x1 = x1.to(device)
            x3 = x3.to(device)
            x4 = x4.to(device)

        p1 = torch.softmax(model1(x1), dim=1).detach().cpu().numpy()
        p2 = p1  # preserve identical semantics (model2 is unused; p2==p1 in original)
        p3 = torch.softmax(model3(x3), dim=1).detach().cpu().numpy()
        p4 = torch.softmax(model4(x4), dim=1).detach().cpu().numpy()

        buf = batch_buf[:bs]
        buf[:, 0:N_CLASSES] = p1
        buf[:, N_CLASSES : 2 * N_CLASSES] = p2
        buf[:, 2 * N_CLASSES : 3 * N_CLASSES] = p3
        buf[:, 3 * N_CLASSES : 4 * N_CLASSES] = p4

        feats[pos : pos + bs, :] = buf
        image_ids[pos : pos + bs] = list(b_ids)

        if has_labels:
            if torch.is_tensor(y):
                labels[pos : pos + bs] = (
                    y.detach().cpu().numpy().astype(np.int64, copy=False)
                )
            else:
                labels[pos : pos + bs] = np.asarray(y, dtype=np.int64)

        pos += bs

    if has_labels:
        return image_ids, feats, labels
    return image_ids, feats




## === cell 4
train_df = pd.read_csv(TRAIN_CSV)
train_df = train_df[["image_id", "label"]].copy()
train_df_fit = train_df

BATCH_SIZE = 16

CACHE_DIR = "/kaggle/working"
TRAIN_CACHE = os.path.join(CACHE_DIR, "cache_train_feats_v1.npz")
TEST_CACHE = os.path.join(CACHE_DIR, "cache_test_feats_v1.npz")

train_ds = CassavaImageDataset(train_df_fit, TRAIN_IMG_DIR, return_label=True)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=DL_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(DL_WORKERS > 0),
    prefetch_factor=2 if DL_WORKERS > 0 else None,
    worker_init_fn=_seed_worker,
)

if os.path.exists(TRAIN_CACHE):
    cached = np.load(TRAIN_CACHE, allow_pickle=True)
    train_ids = cached["image_ids"].tolist()
    train_feats = cached["feats"]
    train_labels = cached["labels"]
    print("Loaded cached train features:", train_feats.shape)
else:
    train_ids, train_feats, train_labels = predict_features(
        train_loader, has_labels=True
    )
    np.savez_compressed(
        TRAIN_CACHE,
        image_ids=np.array(train_ids, dtype=object),
        feats=train_feats,
        labels=train_labels,
    )
    print("Saved cached train features:", train_feats.shape)

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
    worker_init_fn=_seed_worker,
)

if os.path.exists(TEST_CACHE):
    cached = np.load(TEST_CACHE, allow_pickle=True)
    test_ids = cached["image_ids"].tolist()
    test_feats = cached["feats"]
    print("Loaded cached test features:", test_feats.shape)
else:
    test_ids, test_feats = predict_features(test_loader, has_labels=False)
    np.savez_compressed(
        TEST_CACHE,
        image_ids=np.array(test_ids, dtype=object),
        feats=test_feats,
    )
    print("Saved cached test features:", test_feats.shape)

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
