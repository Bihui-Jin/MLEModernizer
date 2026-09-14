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
from PIL import Image

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


def extract_embeddings_batch(filepaths, batch_size=BATCH_SIZE, use_tta_flip=True):
    embs = np.zeros((len(filepaths), 2048), dtype=np.float32)
    feature_extractor.eval()
    with torch.no_grad():
        for start in range(0, len(filepaths), batch_size):
            end = min(start + batch_size, len(filepaths))
            batch_files = filepaths[start:end]

            imgs = []
            imgs_flip = [] if use_tta_flip else None

            for fp in batch_files:
                img = Image.open(fp).convert("RGB")
                imgs.append(torch_transforms(img))
                if use_tta_flip:
                    imgs_flip.append(
                        torch_transforms(img.transpose(Image.FLIP_LEFT_RIGHT))
                    )

            x = torch.stack(imgs, dim=0).to(device)
            emb = feature_extractor(x).detach().float()  # (B,2048,1,1)

            if use_tta_flip:
                x2 = torch.stack(imgs_flip, dim=0).to(device)
                emb2 = feature_extractor(x2).detach().float()
                emb = 0.5 * (emb + emb2)

            emb = emb.cpu().numpy().reshape(-1, 2048).astype(np.float32)
            embs[start:end] = emb

            if end % 500 == 0 or end == len(filepaths):
                print(f"Embeddings: {end}/{len(filepaths)}", end="\r")
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
    Xtr = scaler_f.fit_transform(train_embs[tr_idx]).astype(np.float32)
    Xva = scaler_f.transform(train_embs[va_idx]).astype(np.float32)

    lr_f = make_lr()
    lr_f.fit(Xtr, train_labels[tr_idx])

    oof_logits5[va_idx] = lr_f.decision_function(Xva).astype(np.float32)
    print(f"OOF fold {fold}/5 done", end="\r")

p2_train = softmax_np(oof_logits5 / float(T_model2), axis=1).astype(np.float32)
p1_train = softmax_np(oof_logits5 / float(T_model1_fallback), axis=1).astype(np.float32)
train_feats = np.concatenate([p1_train, p2_train], axis=1).astype(np.float32)  # (N,10)

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
train_embs_s = scaler.fit_transform(train_embs).astype(np.float32)

lr_map = make_lr()
lr_map.fit(train_embs_s, train_labels)

(train_feats.shape, np.unique(train_labels, return_counts=True)[0])



## === cell 2
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

test_embs_s = scaler.transform(test_embs).astype(np.float32)

test_logits5 = lr_map.decision_function(test_embs_s).astype(np.float32)  # (M,5)
p2_test = softmax_np(test_logits5 / float(T_model2), axis=1).astype(np.float32)
p1_test = softmax_np(test_logits5 / float(T_model1_fallback), axis=1).astype(np.float32)
combined_probs = np.concatenate([p1_test, p2_test], axis=1).astype(np.float32)  # (M,10)

combined_probs.shape



## === cell 3
prediction = tree_model.predict(combined_probs)
prediction[:10], prediction.shape



## === cell 4
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



## === cell 5
assert os.path.exists("submission.csv")
assert list(submission.columns) == ["image_id", "label"]
assert len(submission) == len(sample_sub)
assert submission["label"].between(0, 4).all()
print("Wrote submission.csv with shape:", submission.shape)
submission.tail()
