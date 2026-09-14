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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.9669828776458648

# 6. Current score

0.5597

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the dataset path resolution so `BASE_DIR` always points to the folder that actually contains `sample_submission.csv` and `test.csv`, which is why your code currently crashes. Then I make the submission-building cell robust by re-reading `test.csv`/`sample_submission.csv` from that resolved directory and ensuring the exact required columns/order. These changes are score-neutral (still outputs the same constant probabilities) but make the notebook run end-to-end and reliably write a valid `submission.csv`. I also keep your original core approach intact and only adjust the failing path logic and the dependent variable usage.'
- What this solution (achieved 0.54982) has done: 'I fix the image filename/path bug that causes `Image.open()` to look for files like `Train_0` without the required `.jpg` suffix, and I make image path resolution robust across both the `images/` and nested `plant-pathology-2020-fgvc7/images/` layouts. I also add a safe fallback that tries common extensions if an id already includes/omits one, so the feature builder doesn’t crash mid-run. These changes are execution/stability fixes and do not change the model/training logic. The script then run end-to-end and write a valid `submission.csv` with the exact required columns/order.'
- What this solution (achieved 0.54156) has done: 'We keep your current feature extractor and LogisticRegression OneVsRest pipeline intact, but make two minimal, score-relevant upgrades: (1) use an L2-regularized logistic model with a slightly higher C to reduce underfitting on these simple image statistics, and (2) add class-weight balancing so rare classes (notably multiple_diseases) aren’t ignored, which tends to lift mean per-column ROC AUC. These changes preserve the same training approach (single fit on all train, same model family) and prediction semantics (probabilities via predict_proba), while usually improving AUC meaningfully from the ~0.55 range. We also add deterministic settings (random_state) for stability without changing the core logic. The script still runs end-to-end and writes a valid `submission.csv` with the required columns/order.'
- What this solution (achieved 0.55714) has done: 'You’re far below the target AUC, so we should make a small, legitimate improvement that keeps your exact model family and training flow intact. The main score-limiting issue is that the 4 labels are mutually exclusive (one-hot), but your current OneVsRest logistic setup doesn’t enforce probabilities summing to 1, which tends to hurt mean column-wise ROC AUC here. We switch to a single multinomial LogisticRegression (same linear model, same scaling, same “fit once then predict_proba” approach) and use its calibrated 4-class probabilities directly. This is a minimal change (no new feature extraction, no new training loop) and typically yields a large AUC jump on this dataset.'
- What this solution (achieved 0.5597) has done: 'We’re far below the target AUC, so the smallest score-relevant change is to make the linear classifier better match the metric without changing the feature extractor or the “fit once then predict_proba” flow. The current `class_weight="balanced"` and relatively large `C=3.0` can overcorrect on this dataset and hurt mean per-column ROC AUC, especially for mutually exclusive classes; we revert to unweighted multinomial logistic regression and use a slightly more regularized `C=1.0` to improve probability ranking stability. Everything else (features, scaling, multinomial LR, submission formatting, paths) stays the same. This is a minimal hyperparameter adjustment that typically yields a meaningful AUC lift while preserving core logic.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

BASE_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "../input/plant-pathology-2020-fgvc7",
    "../kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/data",
    "/kaggle/input",
]


def _resolve_base_dir(candidates):
    for c in candidates:
        if not os.path.exists(c):
            continue

        if os.path.isfile(os.path.join(c, "sample_submission.csv")) and os.path.isfile(
            os.path.join(c, "test.csv")
        ):
            return c

        nested = os.path.join(c, "plant-pathology-2020-fgvc7")
        if os.path.isdir(nested):
            if os.path.isfile(
                os.path.join(nested, "sample_submission.csv")
            ) and os.path.isfile(os.path.join(nested, "test.csv")):
                return nested

    return None


BASE_DIR = _resolve_base_dir(BASE_CANDIDATES)

if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate dataset directory containing sample_submission.csv and test.csv. "
        f"Tried: {BASE_CANDIDATES}"
    )

print("Resolved BASE_DIR:", BASE_DIR)
print(
    "Files present:",
    [
        f
        for f in ["train.csv", "test.csv", "sample_submission.csv"]
        if os.path.exists(os.path.join(BASE_DIR, f))
    ],
)



## === cell 1
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

train = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
test = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

img_dir = os.path.join(BASE_DIR, "images")
if not os.path.isdir(img_dir):
    nested_img_dir = os.path.join(BASE_DIR, "plant-pathology-2020-fgvc7", "images")
    if os.path.isdir(nested_img_dir):
        img_dir = nested_img_dir

if not os.path.isdir(img_dir):
    raise FileNotFoundError(
        f"Could not find images directory. Tried: {os.path.join(BASE_DIR,'images')}"
    )

print("Resolved img_dir:", img_dir)


def _image_path(image_id: str) -> str:
    candidates = [image_id]
    if "." not in os.path.basename(image_id):
        candidates = [
            f"{image_id}.jpg",
            f"{image_id}.jpeg",
            f"{image_id}.png",
            image_id,
        ]

    for name in candidates:
        p = os.path.join(img_dir, name)
        if os.path.isfile(p):
            return p

    raise FileNotFoundError(
        f"Image file not found for image_id='{image_id}'. Tried: "
        + ", ".join([os.path.join(img_dir, n) for n in candidates])
    )


def _extract_features(image_id: str, size=(96, 96)) -> np.ndarray:
    """
    Simple, fast features:
      - per-channel mean/std in RGB
      - per-channel mean/std in HSV
      - grayscale mean/std
      - downsampled grayscale thumbnail (captures coarse texture/shape)
    """
    p = _image_path(image_id)
    with Image.open(p) as im:
        im = im.convert("RGB")
        im_small = im.resize(size, resample=Image.BILINEAR)

        arr = np.asarray(im_small, dtype=np.float32) / 255.0  # (H,W,3)

        rgb_mean = arr.mean(axis=(0, 1))
        rgb_std = arr.std(axis=(0, 1))

        hsv = np.asarray(im_small.convert("HSV"), dtype=np.float32)
        hsv[..., 0] = hsv[..., 0] / 255.0
        hsv[..., 1] = hsv[..., 1] / 255.0
        hsv[..., 2] = hsv[..., 2] / 255.0
        hsv_mean = hsv.mean(axis=(0, 1))
        hsv_std = hsv.std(axis=(0, 1))

        gray = np.asarray(im_small.convert("L"), dtype=np.float32) / 255.0
        gray_mean = np.array([gray.mean()], dtype=np.float32)
        gray_std = np.array([gray.std()], dtype=np.float32)

        thumb = (
            np.asarray(
                im.resize((24, 24), resample=Image.BILINEAR).convert("L"),
                dtype=np.float32,
            )
            / 255.0
        )
        thumb = thumb.ravel()

        feat = np.concatenate(
            [rgb_mean, rgb_std, hsv_mean, hsv_std, gray_mean, gray_std, thumb], axis=0
        )
        return feat


def _build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    feats = []
    for iid in df["image_id"].values:
        feats.append(_extract_features(iid))
    return np.vstack(feats)


X_train = _build_feature_matrix(train)
y_train = train[target_cols].values.astype(np.float32)

X_test = _build_feature_matrix(test)

y_class = np.argmax(y_train, axis=1).astype(np.int64)

base_lr = LogisticRegression(
    max_iter=4000,
    solver="lbfgs",
    C=1.0,
    multi_class="multinomial",
    class_weight=None,
    random_state=42,
)

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", base_lr),
    ]
)

clf.fit(X_train, y_class)
proba = clf.predict_proba(X_test).astype(np.float32)  # shape (n_test, 4)

class_to_index = {c: i for i, c in enumerate(clf.named_steps["lr"].classes_)}
proba_ordered = np.zeros((proba.shape[0], len(target_cols)), dtype=np.float32)
for k in range(len(target_cols)):
    proba_ordered[:, k] = proba[:, class_to_index[k]]

proba_ordered = np.clip(proba_ordered, 1e-6, 1 - 1e-6)

sub = pd.DataFrame(proba_ordered, columns=target_cols)
sub.insert(0, "image_id", test["image_id"].values)

sub = sub[["image_id"] + target_cols]

assert sub.shape[0] == test.shape[0], "Submission row count must match test.csv"
assert list(sub.columns) == ["image_id"] + target_cols, "Submission columns mismatch"

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
