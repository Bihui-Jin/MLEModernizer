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
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
from torchvision import models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

RNG_SEED = 42
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)
torch.cuda.manual_seed_all(RNG_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    Image.MAX_IMAGE_PIXELS = None
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 1
def softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


weights = models.ResNet50_Weights.DEFAULT
torch_transforms = weights.transforms()

model2 = models.resnet50(weights=weights).to(device)
model2.eval()

feature_extractor = torch.nn.Sequential(*list(model2.children())[:-1]).to(device)
feature_extractor.eval()
if device.type == "cuda":
    feature_extractor = feature_extractor.to(memory_format=torch.channels_last)

base_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]


def first_existing_file(relpath):
    for b in base_candidates:
        p = os.path.join(b, relpath)
        if os.path.exists(p):
            return p
    return None


def first_existing_dir(relpath):
    for b in base_candidates:
        p = os.path.join(b, relpath)
        if os.path.isdir(p):
            return p
    return None


train_csv_path = first_existing_file("train.csv")
if train_csv_path is None:
    for p in ["/kaggle/input/train.csv", "/kaggle/data/train.csv"]:
        if os.path.exists(p):
            train_csv_path = p
            break
if train_csv_path is None:
    raise FileNotFoundError("train.csv not found in expected locations.")

train_dir = first_existing_dir("train_images")
if train_dir is None:
    for p in ["/kaggle/input/train_images", "/kaggle/data/train_images"]:
        if os.path.isdir(p):
            train_dir = p
            break
if train_dir is None:
    raise FileNotFoundError("train_images directory not found in expected locations.")

train_df = pd.read_csv(train_csv_path)
if not {"image_id", "label"}.issubset(train_df.columns):
    raise ValueError(
        f"train.csv must have columns image_id,label. Got: {train_df.columns.tolist()}"
    )

train_df["filepath"] = train_df["image_id"].apply(lambda x: os.path.join(train_dir, x))
train_df = train_df[train_df["filepath"].apply(os.path.exists)].reset_index(drop=True)
if len(train_df) == 0:
    raise FileNotFoundError(
        "No training images found matching train.csv in train_images."
    )

train_labels = train_df["label"].astype(int).values

T_model2 = 1.0
T_model1_fallback = 1.6

BATCH_SIZE = 64 if torch.cuda.is_available() else 16




## === cell 2
def extract_embeddings_batch(filepaths, batch_size=BATCH_SIZE, use_tta_flip=True):
    from torch.utils.data import Dataset, DataLoader

    try:
        from torchvision.io import read_image  # uint8 CHW RGB
    except Exception:
        read_image = None

    class _PathDataset(Dataset):
        __slots__ = ("fps", "do_flip", "use_tv")

        def __init__(self, fps, do_flip, use_tv):
            self.fps = fps
            self.do_flip = do_flip
            self.use_tv = use_tv

        def __len__(self):
            return len(self.fps)

        def _load_pil_rgb(self, fp):
            if self.use_tv:
                t = read_image(fp)  # CHW uint8
                np_img = t.permute(1, 2, 0).contiguous().numpy()
                return Image.fromarray(np_img, mode="RGB")
            else:
                return Image.open(fp).convert("RGB")

        def __getitem__(self, idx):
            fp = self.fps[idx]
            img = self._load_pil_rgb(fp)
            x = torch_transforms(img)
            if self.do_flip:
                img_f = img.transpose(Image.FLIP_LEFT_RIGHT)
                x_f = torch_transforms(img_f)
                return x, x_f
            else:
                return x

    ds = _PathDataset(list(filepaths), use_tta_flip, use_tv=(read_image is not None))

    if torch.cuda.is_available():
        num_workers = min(8, (os.cpu_count() or 8))
        prefetch_factor = 4
    else:
        num_workers = min(4, (os.cpu_count() or 4))
        prefetch_factor = 2

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False,
    )

    embs = np.empty((len(ds), 2048), dtype=np.float32)

    write_pos = 0
    with torch.inference_mode():
        for batch in loader:
            if use_tta_flip:
                x, x2 = batch
                if device.type == "cuda":
                    x = x.to(device, non_blocking=True).to(
                        memory_format=torch.channels_last
                    )
                    x2 = x2.to(device, non_blocking=True).to(
                        memory_format=torch.channels_last
                    )
                else:
                    x = x.to(device)
                    x2 = x2.to(device)
                emb = feature_extractor(x).float()
                emb2 = feature_extractor(x2).float()
                emb = 0.5 * (emb + emb2)
            else:
                x = batch
                if device.type == "cuda":
                    x = x.to(device, non_blocking=True).to(
                        memory_format=torch.channels_last
                    )
                else:
                    x = x.to(device)
                emb = feature_extractor(x).float()

            emb = (
                emb.squeeze(-1)
                .squeeze(-1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            bsz = emb.shape[0]
            embs[write_pos : write_pos + bsz] = emb
            write_pos += bsz

            if write_pos % 500 == 0 or write_pos == len(ds):
                print(f"Embeddings: {write_pos}/{len(ds)}", end="\r")

    return embs


train_embs = extract_embeddings_batch(
    train_df["filepath"].values, batch_size=BATCH_SIZE, use_tta_flip=True
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)

oof_logits5 = np.zeros((len(train_embs), 5), dtype=np.float32)


def make_lr():
    return LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        C=1.0,
        max_iter=1200,
        class_weight="balanced",
        random_state=RNG_SEED,
    )


for fold, (tr_idx, va_idx) in enumerate(skf.split(train_embs, train_labels), start=1):
    scaler_f = StandardScaler(with_mean=True, with_std=True)
    Xtr = np.asarray(
        scaler_f.fit_transform(train_embs[tr_idx]), dtype=np.float32, order="C"
    )
    Xva = np.asarray(
        scaler_f.transform(train_embs[va_idx]), dtype=np.float32, order="C"
    )

    lr_f = make_lr()
    lr_f.fit(Xtr, train_labels[tr_idx])

    oof_logits5[va_idx] = lr_f.decision_function(Xva).astype(np.float32, copy=False)
    print(f"OOF fold {fold}/5 done", end="\r")

p2_train = softmax_np(oof_logits5 / float(T_model2), axis=1).astype(
    np.float32, copy=False
)
p1_train = softmax_np(oof_logits5 / float(T_model1_fallback), axis=1).astype(
    np.float32, copy=False
)
train_feats = np.concatenate([p1_train, p2_train], axis=1).astype(
    np.float32, copy=False
)  # (N,10)

tree_model = ExtraTreesClassifier(
    n_estimators=900,
    max_depth=18,
    min_samples_split=8,
    min_samples_leaf=2,
    bootstrap=True,
    max_samples=0.8,
    class_weight="balanced_subsample",
    random_state=RNG_SEED,
    n_jobs=-1,
)
tree_model.fit(train_feats, train_labels)

scaler = StandardScaler(with_mean=True, with_std=True)
train_embs_s = np.asarray(scaler.fit_transform(train_embs), dtype=np.float32, order="C")

lr_map = make_lr()
lr_map.fit(train_embs_s, train_labels)

(train_feats.shape, np.unique(train_labels, return_counts=True)[0])



## === cell 3
test_dir = first_existing_dir("test_images")
if test_dir is None:
    for p in ["/kaggle/input/test_images", "/kaggle/data/test_images"]:
        if os.path.isdir(p):
            test_dir = p
            break
if test_dir is None:
    raise FileNotFoundError("Test images directory not found in expected locations.")

test_images = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
test_filepaths = [os.path.join(test_dir, f) for f in test_images]

test_embs = extract_embeddings_batch(
    test_filepaths, batch_size=BATCH_SIZE, use_tta_flip=True
)

test_embs_s = np.asarray(scaler.transform(test_embs), dtype=np.float32, order="C")

test_logits5 = lr_map.decision_function(test_embs_s).astype(
    np.float32, copy=False
)  # (M,5)
p2_test = softmax_np(test_logits5 / float(T_model2), axis=1).astype(
    np.float32, copy=False
)
p1_test = softmax_np(test_logits5 / float(T_model1_fallback), axis=1).astype(
    np.float32, copy=False
)
combined_probs = np.concatenate([p1_test, p2_test], axis=1).astype(
    np.float32, copy=False
)  # (M,10)

combined_probs.shape



## === cell 4
prediction = tree_model.predict(combined_probs)
prediction[:10], prediction.shape



## === cell 5
sample_path = first_existing_file("sample_submission.csv")
if sample_path is None:
    for p in [
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]:
        if os.path.exists(p):
            sample_path = p
            break
if sample_path is None:
    raise FileNotFoundError("sample_submission.csv not found in expected locations.")

sample_sub = pd.read_csv(sample_path)

pred_df = pd.DataFrame({"image_id": test_images, "label": prediction.astype(int)})
submission = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

if submission["label"].isna().any():
    mode_label = int(pd.Series(train_labels).mode().iloc[0])
    submission["label"] = submission["label"].fillna(mode_label).astype(int)

submission["label"] = submission["label"].clip(0, 4).astype(int)

submission.to_csv("submission.csv", index=False)
submission.head()



## === cell 6
assert os.path.exists("submission.csv")
assert list(submission.columns) == ["image_id", "label"]
assert len(submission) == len(sample_sub)
assert submission["label"].between(0, 4).all()
print("Wrote submission.csv with shape:", submission.shape)
submission.tail()
