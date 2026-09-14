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

0.8566032033847084

# 6. Current score

0.77055

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.59492) has done: 'I fix the runtime failures by removing hard dependencies on missing Kaggle input models/files and instead training the same DecisionTreeClassifier on probabilities produced by a lightweight, locally available image model. To keep the core approach intact (probabilities → concatenate temps → decision tree), the pipeline still generate per-image softmax probabilities at two temperatures, concatenate them, and fit/predict with a decision tree. I also make all paths robust to both `/kaggle/input/...` and `/kaggle/data/...` layouts, ensure deterministic execution, and guarantee a correctly formatted `submission.csv` is written. This should run end-to-end in the provided environment and yield a reasonable accuracy score (likely below top deep models but valid and improved over “no submission”).'
- What this solution (achieved 0.60538) has done: 'Your current score is far below the target, and the biggest limiter is that the pipeline uses an ImageNet classifier’s 1000-way logits randomly projected to 5 classes, which has weak alignment to cassava labels. To keep your core approach intact (image → two-temperature probabilities → concatenate → DecisionTreeClassifier), I switch the feature extractor to a pretrained ImageNet model with much stronger representations (ResNet50) and extract its penultimate embedding, then map to 5-class probabilities with a fixed random projection at two temperatures just like you do now. This is a minimal, safe change that typically yields a large accuracy gain without changing the downstream decision tree logic or submission semantics. I also make the model’s preprocessing match the chosen weights to avoid silent normalization mismatch, which improves score stability.'
- What this solution (achieved 0.76196) has done: 'We keep your core pipeline intact (ResNet50 embedding → fixed random projection → two-temperature 5-way probabilities → concatenate → DecisionTreeClassifier) and make only changes that improve label alignment while staying deterministic. The biggest win with minimal risk is to replace the random projection with a learned linear map from embeddings to 5 classes (multinomial logistic regression) trained on the same training labels; we still compute two-temperature probabilities from that 5-logit output exactly like before and still fit/predict with the decision tree. This tends to move accuracy materially upward toward your target without changing the downstream tree logic or submission semantics. We also switch feature extraction to efficient batched inference (same model/transform) so it finishes within time limits.'
- What this solution (achieved 0.77055) has done: 'Your current score (0.76196) is below the target (0.8566), so we should improve accuracy with minimal risk while keeping the same core pipeline (ResNet50 embeddings → logistic-regression 5-logits → two-temperature probabilities → concatenate → decision tree). The biggest low-change gain is to add the standard ImageNet test-time augmentation used for pretrained models (horizontal flip) and average the embeddings from original+flipped images; this preserves the exact model and training approach but typically improves robustness. I’m also switching the final classifier from `DecisionTreeClassifier` to `ExtraTreesClassifier` (still a tree-based classifier with the same fit/predict semantics on the same 10-dim probability features) which usually generalizes better with minimal tuning and is fast enough. Everything else (paths, submission formatting, temperatures, LR mapping, and deterministic seeds) stays the same.'

# 9. Code solution

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

lr_map = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    C=2.0,
    max_iter=300,
    random_state=RNG_SEED,
)
lr_map.fit(train_embs, train_labels)

train_logits5 = lr_map.decision_function(train_embs).astype(np.float32)  # (N,5)
p2_train = softmax_np(train_logits5 / float(T_model2), axis=1).astype(np.float32)
p1_train = softmax_np(train_logits5 / float(T_model1_fallback), axis=1).astype(
    np.float32
)
train_feats = np.concatenate([p1_train, p2_train], axis=1).astype(np.float32)  # (N,10)

tree_model = ExtraTreesClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=2,
    bootstrap=False,
    random_state=RNG_SEED,
    n_jobs=-1,
)
tree_model.fit(train_feats, train_labels)

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

test_logits5 = lr_map.decision_function(test_embs).astype(np.float32)  # (M,5)
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
