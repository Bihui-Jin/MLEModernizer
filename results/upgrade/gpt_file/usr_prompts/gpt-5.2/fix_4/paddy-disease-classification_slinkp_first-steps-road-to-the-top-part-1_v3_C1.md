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
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.8778801843317973

# 6. Current score

0.20369

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.17487) has done: 'I remove the internet/Kaggle-API dependent setup (rustup/curl, fastkaggle setup_comp, competition_submit, push_notebook) and instead point the code directly at the already-present dataset directory under `/kaggle/input/paddy-disease-classification`. I also fix the broken environment issues by avoiding runtime `pip install` and by using a lightweight, no-external-packages classifier based on the provided `train.csv` metadata (variety + age) so it runs reliably in this Kaggle environment. The script still produce a valid `submission.csv` with the required `image_id,label` columns and correct row alignment to `sample_submission.csv`. This is primarily a correctness/stability fix so you get a valid submission; without fastai/timm available by default, the original image model can’t run end-to-end here.'
- What this solution (achieved 0.17487) has done: 'Your current score is far below the target, so the minimal path toward the target is to keep the same Naive Bayes-style core logic but add one strong, still-simple signal: the image folder name for each `image_id` in `train_images` (this is available and is effectively a supervised feature). I also fix the handling of test metadata (there is no `test.csv` here) by extracting the predicted label purely from the `test_images` filenames, and then mapping any unknowns back to the metadata-based Naive Bayes as a fallback. This preserves your existing approach (log-priors + log-likelihoods + argmax) while giving it a much more informative “feature” that should move accuracy substantially toward the target. The submission format and row alignment remain identical to `sample_submission.csv`.'
- What this solution (achieved 0.20369) has done: 'Your current score (0.17487) is far below the target (0.87788), so we need a meaningful but still minimal improvement while keeping the same Naive Bayes core logic. The biggest issue is that the “train folder label map” cannot help at test time (test images are not in labeled class folders), so it effectively never triggers and you fall back to constant-UNK metadata, producing near-prior guesses. I keep your exact Naive Bayes structure but add a lightweight, legitimate visual signal by extracting simple color statistics from each image (no external packages), and model those as Gaussian class-conditional likelihoods added to your existing log-scores. This preserves the same training approach (compute per-class statistics, then argmax of log-scores) and should move accuracy substantially toward the target while staying within runtime constraints.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageStat

RNG = np.random.default_rng(42)

BASE = Path("/kaggle/input/paddy-disease-classification")
if not BASE.exists():
    BASE = Path("/kaggle/data/paddy-disease-classification")

train_csv = BASE / "train.csv"
sample_csv = BASE / "sample_submission.csv"

assert train_csv.exists(), f"Missing train.csv at {train_csv}"
assert sample_csv.exists(), f"Missing sample_submission.csv at {sample_csv}"

train_df = pd.read_csv(train_csv)
ss = pd.read_csv(sample_csv)

assert {"image_id", "label", "variety", "age"}.issubset(train_df.columns)
assert {"image_id", "label"}.issubset(ss.columns)

train_df["age"] = pd.to_numeric(train_df["age"], errors="coerce")
train_df["variety"] = train_df["variety"].astype("string")
train_df["label"] = train_df["label"].astype("string")



## === cell 1
labels = sorted(train_df["label"].dropna().unique().tolist())
label_to_idx = {c: i for i, c in enumerate(labels)}
idx_to_label = {i: c for c, i in label_to_idx.items()}
n_classes = len(labels)

age = train_df["age"].fillna(train_df["age"].median())
age_bins = pd.cut(age, bins=[-np.inf, 40, 60, 80, 100, 120, np.inf], labels=False)
train_df = train_df.copy()
train_df["age_bin"] = age_bins.astype("Int64")

alpha = 1.0  # Laplace smoothing
class_counts = (
    train_df["label"].value_counts().reindex(labels, fill_value=0).astype(float).values
)
log_prior = np.log((class_counts + alpha) / (class_counts.sum() + alpha * n_classes))

varieties = train_df["variety"].fillna("UNK").unique().tolist()
var_to_idx = {v: i for i, v in enumerate(varieties)}
n_var = len(varieties)

all_agebins = list(range(6))
n_agebin = len(all_agebins)

var_counts = np.zeros((n_classes, n_var), dtype=np.float64)
age_counts = np.zeros((n_classes, n_agebin), dtype=np.float64)

for _, r in train_df.iterrows():
    c = label_to_idx.get(str(r["label"]))
    v = str(r["variety"]) if pd.notna(r["variety"]) else "UNK"
    a = int(r["age_bin"]) if pd.notna(r["age_bin"]) else None
    if c is None:
        continue
    if v not in var_to_idx:
        continue
    var_counts[c, var_to_idx[v]] += 1.0
    if a is not None and 0 <= a < n_agebin:
        age_counts[c, a] += 1.0

log_var_lik = np.log(
    (var_counts + alpha) / (var_counts.sum(axis=1, keepdims=True) + alpha * n_var)
)
log_age_lik = np.log(
    (age_counts + alpha) / (age_counts.sum(axis=1, keepdims=True) + alpha * n_agebin)
)



## === cell 2
train_img_dir = BASE / "train_images"
test_img_dir = BASE / "test_images"
assert train_img_dir.exists(), f"Missing train_images at {train_img_dir}"
assert test_img_dir.exists(), f"Missing test_images at {test_img_dir}"

train_img_label_map = {}
for class_dir in train_img_dir.iterdir():
    if not class_dir.is_dir():
        continue
    class_name = class_dir.name
    if class_name not in label_to_idx:
        continue
    for p in class_dir.glob("*.jpg"):
        train_img_label_map[p.name] = class_name

covered = train_df["image_id"].map(train_img_label_map).notna().mean()
print(f"Train image_id coverage from folders: {covered:.3f}")

mode_label = train_df["label"].mode().iloc[0]




## === cell 3
def img_feats(path: Path):
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im = im.resize((96, 96))
            st = ImageStat.Stat(im)
            mean = np.array(st.mean, dtype=np.float64)  # (R,G,B)
            std = np.array(st.stddev, dtype=np.float64)
            return np.concatenate([mean, std], axis=0)  # 6-d
    except Exception:
        return None


feat_dim = 6
feat_sum = np.zeros((n_classes, feat_dim), dtype=np.float64)
feat_sumsq = np.zeros((n_classes, feat_dim), dtype=np.float64)
feat_n = np.zeros((n_classes,), dtype=np.float64)

for img_name, lab in train_img_label_map.items():
    c = label_to_idx[lab]
    p = train_img_dir / lab / img_name
    f = img_feats(p)
    if f is None or not np.all(np.isfinite(f)):
        continue
    feat_sum[c] += f
    feat_sumsq[c] += f * f
    feat_n[c] += 1.0

eps = 1e-3
feat_mu = np.zeros((n_classes, feat_dim), dtype=np.float64)
feat_var = np.ones((n_classes, feat_dim), dtype=np.float64)

for c in range(n_classes):
    if feat_n[c] >= 2:
        mu = feat_sum[c] / feat_n[c]
        var = feat_sumsq[c] / feat_n[c] - mu * mu
        feat_mu[c] = mu
        feat_var[c] = np.maximum(var, eps)
    elif feat_n[c] == 1:
        feat_mu[c] = feat_sum[c]
        feat_var[c] = np.ones((feat_dim,), dtype=np.float64) * 500.0
    else:
        pass

gauss_const = -0.5 * np.log(2.0 * np.pi * feat_var)  # (C,D)

img_w = 1.0



## === cell 4
test_df = ss[["image_id"]].copy()

if "UNK" not in var_to_idx:
    var_to_idx["UNK"] = n_var
    n_var += 1
    var_counts = np.pad(var_counts, ((0, 0), (0, 1)))
    log_var_lik = np.log(
        (var_counts + alpha) / (var_counts.sum(axis=1, keepdims=True) + alpha * n_var)
    )

test_df["variety"] = "UNK"
test_df["age_bin"] = pd.NA

pred_labels = []
for _, r in test_df.iterrows():
    img_id = r["image_id"]

    v = "UNK"
    vi = var_to_idx.get(v, var_to_idx["UNK"])
    if pd.notna(r.get("age_bin", pd.NA)):
        ai = int(r["age_bin"])
        ai = ai if 0 <= ai < n_agebin else None
    else:
        ai = None

    scores = log_prior.copy()
    scores = scores + log_var_lik[:, vi]
    if ai is not None:
        scores = scores + log_age_lik[:, ai]

    fp = img_feats(test_img_dir / img_id)
    if fp is not None and np.all(np.isfinite(fp)):
        diff = fp[None, :] - feat_mu  # (C,D)
        img_ll = (gauss_const - 0.5 * (diff * diff) / feat_var).sum(axis=1)  # (C,)
        scores = scores + img_w * img_ll

    pred = idx_to_label[int(np.argmax(scores))]
    pred_labels.append(pred)



## === cell 5
sub = ss.copy()
sub["label"] = pred_labels
sub["label"] = sub["label"].fillna(mode_label)

out_path = Path("submission.csv")
sub.to_csv(out_path, index=False)

assert out_path.exists() and out_path.suffix == ".csv"
assert list(sub.columns) == ["image_id", "label"]
assert len(sub) == len(ss)
print(sub.head())
print(f"Wrote {out_path} with {len(sub)} rows.")
