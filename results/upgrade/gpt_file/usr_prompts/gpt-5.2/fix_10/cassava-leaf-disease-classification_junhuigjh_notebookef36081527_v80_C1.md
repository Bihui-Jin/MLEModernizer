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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageChops, ImageOps, ImageFile
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.models import (
    vit_h_14,
    ViT_H_14_Weights,
    efficientnet_v2_l,
    EfficientNet_V2_L_Weights,
    resnet50,
    ResNet50_Weights,
    densenet121,
    DenseNet121_Weights,
)

try:
    from torchvision.io import read_image, ImageReadMode

    _HAS_TVIO = True
except Exception:
    _HAS_TVIO = False

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


ImageFile.LOAD_TRUNCATED_IMAGES = True

cpu = os.cpu_count() or 1
torch.set_num_threads(max(1, min(8, cpu // 2)))
torch.set_num_interop_threads(1)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.empty_cache()

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(os.path.join(p, "train.csv")) and os.path.isdir(
        os.path.join(p, "train_images")
    ):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate cassava dataset root. Expected train.csv + train_images under one of: "
        + ", ".join(DATA_ROOT_CANDIDATES)
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

print("Using DATA_ROOT:", DATA_ROOT)
print("Device:", device)




## === cell 1
def invert_square_pad(img: Image.Image) -> Image.Image:
    width, height = img.size
    img2 = ImageChops.offset(img, width // 2, height // 2)

    max_side = max(width, height)
    pad_left = (max_side - width) // 2
    pad_top = (max_side - height) // 2
    pad_right = (max_side - width) - pad_left
    pad_bottom = (max_side - height) - pad_top

    if pad_left == pad_right == pad_top == pad_bottom == 0:
        return img2

    base = img2
    mirror_x = ImageOps.mirror(base)
    mirror_y = ImageOps.flip(base)
    mirror_xy = ImageOps.mirror(mirror_y)

    w, h = base.size
    canvas = Image.new("RGB", (w * 3, h * 3))
    tiles = [
        [mirror_xy, mirror_y, mirror_xy],
        [mirror_x, base, mirror_x],
        [mirror_xy, mirror_y, mirror_xy],
    ]
    for r in range(3):
        for c in range(3):
            canvas.paste(tiles[r][c], (c * w, r * h))

    left = w + (w - width) // 2 - pad_left
    top = h + (h - height) // 2 - pad_top
    right = left + width + pad_left + pad_right
    bottom = top + height + pad_top + pad_bottom

    padded = canvas.crop((left, top, right, bottom))
    return padded


torch_transforms_ResNet = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

torch_transforms_VIT = transforms.Compose(
    [
        v2.Lambda(invert_square_pad),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((518, 518)),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

torch_transforms_EfficientNet = transforms.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((480, 480)),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

torch_transforms_DenseNet = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)



## === cell 2
NUM_CLASSES = 5


def build_torch_model_backbone(name: str):
    if name == "vit_h_14":
        m = vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)
        in_f = m.heads.head.in_features
        m.heads.head = nn.Linear(in_f, NUM_CLASSES)
        return m
    if name == "efficientnet_v2_l":
        m = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, NUM_CLASSES)
        return m
    if name == "resnet50":
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, NUM_CLASSES)
        return m
    if name == "densenet121":
        m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        in_f = m.classifier.in_features
        m.classifier = nn.Linear(in_f, NUM_CLASSES)
        return m
    raise ValueError(f"Unknown backbone: {name}")


model1 = build_torch_model_backbone("densenet121").to(device).eval()
model2 = build_torch_model_backbone("resnet50").to(device).eval()
model3 = build_torch_model_backbone("vit_h_14").to(device).eval()
model4 = build_torch_model_backbone("efficientnet_v2_l").to(device).eval()

if device.type == "cuda":
    model1 = model1.to(memory_format=torch.channels_last)
    model2 = model2.to(memory_format=torch.channels_last)
    model4 = model4.to(memory_format=torch.channels_last)

print("Models initialized (torchvision pretrained backbones).")




## === cell 3
class CassavaImageDataset(Dataset):
    """
    Speed optimization (correctness-preserving):
    - Decode image exactly once; keep as PIL.Image for identical downstream transforms.
    - Optionally use torchvision.io.read_image for faster decode, then convert to PIL
      to preserve the exact same transform semantics (Resize/ToTensor/etc).
    """

    def __init__(self, df, image_dir):
        df = df.reset_index(drop=True)
        self.image_ids = df["image_id"].tolist()
        self.has_label = "label" in df.columns
        self.labels = df["label"].astype(np.int64).tolist() if self.has_label else None
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        y = int(self.labels[idx]) if self.has_label else -1
        fp = os.path.join(self.image_dir, image_id)

        if _HAS_TVIO:
            t = read_image(fp, mode=ImageReadMode.RGB)
            im = Image.fromarray(t.permute(1, 2, 0).numpy(), mode="RGB")
        else:
            im = Image.open(fp).convert("RGB")
        return image_id, im, y


_tf_dn = torch_transforms_DenseNet
_tf_rn = torch_transforms_ResNet
_tf_vit_no_pad = transforms.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((518, 518)),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
_tf_en = torch_transforms_EfficientNet


def quad_collate_fn(batch):
    """
    Speed optimization (correctness-preserving):
    - Compute invert_square_pad exactly once per image (needed only for ViT path).
    - Preallocate batch tensors to reduce Python list overhead and GC churn.
    """
    bs = len(batch)
    image_ids = [None] * bs
    ys = torch.empty((bs,), dtype=torch.long)

    x1_out = x2_out = x3_out = x4_out = None

    for i, (image_id, im, y) in enumerate(batch):
        image_ids[i] = image_id
        ys[i] = y

        t1 = _tf_dn(im)
        t2 = _tf_rn(im)
        im_vit = invert_square_pad(im)
        t3 = _tf_vit_no_pad(im_vit)
        t4 = _tf_en(im)

        if x1_out is None:
            x1_out = torch.empty((bs,) + tuple(t1.shape), dtype=t1.dtype)
            x2_out = torch.empty((bs,) + tuple(t2.shape), dtype=t2.dtype)
            x3_out = torch.empty((bs,) + tuple(t3.shape), dtype=t3.dtype)
            x4_out = torch.empty((bs,) + tuple(t4.shape), dtype=t4.dtype)

        x1_out[i].copy_(t1)
        x2_out[i].copy_(t2)
        x3_out[i].copy_(t3)
        x4_out[i].copy_(t4)

    return image_ids, (x1_out, x2_out, x3_out, x4_out), ys


@torch.inference_mode()
def _predict_features_from_4tensors(x1, x2, x3, x4) -> np.ndarray:
    """
    Core logic preserved: softmax probs for 4 backbones concatenated to 20-dim features.
    Speed optimization (correctness-preserving):
    - Use non_blocking H2D copies with pinned memory.
    - Keep channels_last on CUDA where already used.
    """
    if device.type == "cuda":
        x1 = x1.contiguous(memory_format=torch.channels_last).to(
            device, non_blocking=True
        )
        x2 = x2.contiguous(memory_format=torch.channels_last).to(
            device, non_blocking=True
        )
        x3 = x3.to(device, non_blocking=True)
        x4 = x4.contiguous(memory_format=torch.channels_last).to(
            device, non_blocking=True
        )
    else:
        x1, x2, x3, x4 = x1.to(device), x2.to(device), x3.to(device), x4.to(device)

    p1 = F.softmax(model1(x1), dim=1)
    p2 = F.softmax(model2(x2), dim=1)
    p3 = F.softmax(model3(x3), dim=1)
    p4 = F.softmax(model4(x4), dim=1)

    out_t = torch.cat((p1, p2, p3, p4), dim=1).to("cpu")
    return out_t.numpy().astype(np.float32, copy=False)


def extract_features_dataframe(
    df: pd.DataFrame, image_dir: str, *, loader_bs: int
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    ds = CassavaImageDataset(df, image_dir)

    cpu = os.cpu_count() or 1
    if device.type == "cuda":
        num_workers = min(8, max(4, cpu // 2))
    else:
        num_workers = min(4, max(0, cpu // 2))

    g = torch.Generator()
    g.manual_seed(SEED)

    dl_kwargs = dict(
        batch_size=loader_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=quad_collate_fn,
        persistent_workers=(num_workers > 0),
        worker_init_fn=_seed_worker,
        generator=g,
        drop_last=False,
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 8 if device.type == "cuda" else 4
    dl = DataLoader(ds, **dl_kwargs)

    X = np.empty((len(ds), NUM_CLASSES * 4), dtype=np.float32)
    has_label = "label" in df.columns
    y = np.empty((len(ds),), dtype=np.int64) if has_label else None
    image_ids_all = [None] * len(ds)

    ofs = 0
    for image_ids, (x1, x2, x3, x4), ys in dl:
        feats = _predict_features_from_4tensors(x1, x2, x3, x4)
        bs = feats.shape[0]
        X[ofs : ofs + bs] = feats
        image_ids_all[ofs : ofs + bs] = image_ids
        if has_label:
            y[ofs : ofs + bs] = ys.numpy()
        ofs += bs
        if (ofs % 400 == 0) or (ofs == len(ds)):
            print(f"Features: {ofs}/{len(ds)}", end="\r")
    print()
    return X, y, image_ids_all




## === cell 4
train_df = pd.read_csv(TRAIN_CSV)
if not {"image_id", "label"}.issubset(train_df.columns):
    raise ValueError("train.csv must contain columns: image_id, label")

MAX_META_TRAIN = 6000
if len(train_df) > MAX_META_TRAIN:
    meta_train_df, _ = train_test_split(
        train_df,
        train_size=MAX_META_TRAIN,
        stratify=train_df["label"],
        random_state=SEED,
    )
else:
    meta_train_df = train_df

meta_train_df = meta_train_df.reset_index(drop=True)
print("Meta-train size:", len(meta_train_df))

loader_bs = 96 if device.type == "cuda" else 16
X_meta, y_meta, _ = extract_features_dataframe(
    meta_train_df, TRAIN_DIR, loader_bs=loader_bs
)



## === cell 5
decision_tree = RandomForestClassifier(
    n_estimators=90,
    criterion="gini",
    max_depth=6,
    random_state=42,
    n_jobs=-1,
)
decision_tree.fit(X_meta, y_meta)
print("Meta-model fitted.")



## === cell 6
if not os.path.isdir(TEST_DIR):
    raise FileNotFoundError(f"Missing test_images directory at: {TEST_DIR}")

with os.scandir(TEST_DIR) as it:
    image_ids = sorted(
        [e.name for e in it if e.is_file() and e.name.lower().endswith(".jpg")]
    )

print("Test images:", len(image_ids))

test_df = pd.DataFrame({"image_id": image_ids})

loader_bs = 128 if device.type == "cuda" else 16
combined_output, _, image_ids_out = extract_features_dataframe(
    test_df, TEST_DIR, loader_bs=loader_bs
)

if image_ids_out != image_ids:
    raise RuntimeError(
        "Image order mismatch during feature extraction; submission would misalign."
    )

prediction = decision_tree.predict(combined_output).astype(int)
print("Predictions:", prediction.shape, "unique:", np.unique(prediction))

submission = pd.DataFrame({"image_id": image_ids, "label": prediction})

if os.path.exists(SAMPLE_SUB):
    sample = pd.read_csv(SAMPLE_SUB)
    if "image_id" in sample.columns and len(sample) == len(submission):
        submission = sample[["image_id"]].merge(submission, on="image_id", how="left")
        if submission["label"].isna().any():
            raise RuntimeError(
                "Submission merge produced NaNs; check image_id alignment."
            )
        submission["label"] = submission["label"].astype(int)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
submission.head()
