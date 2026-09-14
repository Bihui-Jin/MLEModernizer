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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.91123

# 6. Current score

0.42924

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I remove the failing `fastai2` install/import (it’s incompatible with this Kaggle environment) and replace it with a lightweight scikit-learn baseline that can run with the provided installed packages. The baseline uses the provided `train.csv`/`test.csv` only (no image loading) to build label priors and outputs valid probabilities per class for every `image_id`, ensuring a correct submission file is written. This fixes the runtime errors (imports, missing modules like `Path/pd/os`, and unavailable deep learning stack) and guarantees an end-to-end run that produces `submission.csv` with the required columns. Since your current score was “Not yielded”, the priority is producing a valid submission; this yield a non-zero score (though likely below the target because images aren’t used).'
- What this solution (achieved 0.47286) has done: 'Your current 0.5 score comes from predicting the same constant probabilities for every test image, which yields ROC AUC ≈ 0.5 because there’s no ranking signal. To move toward the 0.91123 target without changing the overall “no-image, scikit-learn/pandas-only” approach, I add a minimal signal based on the `image_id` string itself: extract simple character n-grams and train one-vs-rest logistic regression models for the 4 labels. This preserves the same training/prediction semantics (probabilities per class) but introduces per-row variation so AUC can improve above chance. I also keep a tiny amount of smoothing via blending with the label priors to stabilize probabilities and reduce the risk of extreme outputs.'
- What this solution (achieved 0.42924) has done: 'Your current score is far below the target, so we need more real signal; the simplest legitimate improvement (without changing the modeling family or training loop) is to use the provided images to generate features and keep the same one-vs-rest LogisticRegression setup. I add lightweight image feature extraction (resize + normalized RGB + simple edge magnitude + HSV-ish stats) using only PIL/numpy, then concatenate those numeric features with your existing TF‑IDF char n-grams on `image_id`. I also ensure the solver is appropriate for the mixed sparse/dense feature matrix and keep the same probability blending with priors for stability. This should materially raise AUC toward the target while staying within the “scikit-learn baseline” core logic and producing the same submission schema.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import sparse

from PIL import Image



## === cell 1
data_path = Path("/kaggle/input/plant-pathology-2020-fgvc7")

if not data_path.exists():
    data_path = Path("/kaggle/data/plant-pathology-2020-fgvc7")

data_path



## === cell 2
train_path = data_path / "train.csv"
test_path = data_path / "test.csv"
sample_path = data_path / "sample_submission.csv"
images_dir = data_path / "images"

df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

df.shape, test_df.shape, sample_sub.shape, images_dir.exists()



## === cell 3
df.head()



## === cell 4
target_cols = [c for c in sample_sub.columns if c != "image_id"]
target_cols



## === cell 5
imglabels = target_cols  # expected order: healthy, multiple_diseases, rust, scab

df["labels"] = df.apply(
    lambda x: imglabels[np.asarray(x[imglabels].values, dtype=float).argmax()], axis=1
)
df["labels"].value_counts()



## === cell 6
priors = df[imglabels].mean(axis=0).astype(float)
priors = priors.clip(0.0, 1.0)
priors




## === cell 7
def _image_path(image_id: str) -> Path:
    return images_dir / str(image_id)


def extract_image_features(image_ids, size=(64, 64)):
    """
    Returns a dense float32 matrix of shape (n, d).
    Features: resized RGB (flattened), per-channel mean/std, edge magnitude mean/std,
    and simple HSV-like stats computed from RGB.
    """
    n = len(image_ids)
    h, w = size[1], size[0]
    rgb_dim = h * w * 3
    d = rgb_dim + 14
    X = np.zeros((n, d), dtype=np.float32)

    for i, iid in enumerate(image_ids):
        p = _image_path(iid)
        try:
            im = Image.open(p).convert("RGB").resize((w, h), resample=Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0  # (h,w,3)
        except Exception:
            continue

        flat = arr.reshape(-1)
        X[i, :rgb_dim] = flat

        ch_mean = arr.mean(axis=(0, 1))
        ch_std = arr.std(axis=(0, 1))

        gray = (
            0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
        ).astype(np.float32)
        gx = np.diff(gray, axis=1, prepend=gray[:, :1])
        gy = np.diff(gray, axis=0, prepend=gray[:1, :])
        mag = np.sqrt(gx * gx + gy * gy)
        edge_mean = float(mag.mean())
        edge_std = float(mag.std())

        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin + 1e-8

        hue = np.zeros_like(cmax, dtype=np.float32)
        mask = cmax == r
        hue[mask] = ((g[mask] - b[mask]) / delta[mask]) % 6.0
        mask = cmax == g
        hue[mask] = ((b[mask] - r[mask]) / delta[mask]) + 2.0
        mask = cmax == b
        hue[mask] = ((r[mask] - g[mask]) / delta[mask]) + 4.0
        hue = (hue / 6.0).astype(np.float32)

        sat = (delta / (cmax + 1e-8)).astype(np.float32)
        val = cmax.astype(np.float32)

        hsv_mean = np.array([hue.mean(), sat.mean(), val.mean()], dtype=np.float32)
        hsv_std = np.array([hue.std(), sat.std(), val.std()], dtype=np.float32)

        X[i, rgb_dim : rgb_dim + 3] = ch_mean
        X[i, rgb_dim + 3 : rgb_dim + 6] = ch_std
        X[i, rgb_dim + 6] = edge_mean
        X[i, rgb_dim + 7] = edge_std
        X[i, rgb_dim + 8 : rgb_dim + 11] = hsv_mean
        X[i, rgb_dim + 11 : rgb_dim + 14] = hsv_std

    return X




## === cell 8
X_train_text = df["image_id"].astype(str).values
X_test_text = test_df["image_id"].astype(str).values

vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1)
Xtr_txt = vectorizer.fit_transform(X_train_text)
Xte_txt = vectorizer.transform(X_test_text)

Xtr_img = extract_image_features(df["image_id"].astype(str).values, size=(64, 64))
Xte_img = extract_image_features(test_df["image_id"].astype(str).values, size=(64, 64))

scaler = StandardScaler(with_mean=True, with_std=True)
Xtr_img_s = scaler.fit_transform(Xtr_img).astype(np.float32)
Xte_img_s = scaler.transform(Xte_img).astype(np.float32)

Xtr = sparse.hstack([Xtr_txt, sparse.csr_matrix(Xtr_img_s)], format="csr")
Xte = sparse.hstack([Xte_txt, sparse.csr_matrix(Xte_img_s)], format="csr")

preds = np.zeros((len(test_df), len(imglabels)), dtype=float)

for j, c in enumerate(imglabels):
    y = df[c].astype(int).values

    if y.min() == y.max():
        preds[:, j] = float(priors[c])
        continue

    clf = LogisticRegression(
        solver="saga",
        C=2.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=0,
        n_jobs=1,
    )
    clf.fit(Xtr, y)
    preds[:, j] = clf.predict_proba(Xte)[:, 1]

alpha = 0.95  # slightly more model signal now that we have real features
prior_vec = priors[imglabels].values.reshape(1, -1)
preds = alpha * preds + (1.0 - alpha) * prior_vec
preds = np.clip(preds, 1e-6, 1 - 1e-6)

submission = pd.DataFrame({"image_id": test_df["image_id"].astype(str)})
for j, c in enumerate(imglabels):
    submission[c] = preds[:, j].astype(float)

submission = submission[sample_sub.columns]
submission.head(10)



## === cell 9
out_path = Path("submission.csv")
submission.to_csv(out_path, index=False)

out_path, out_path.exists(), submission.shape



## === cell 10
assert out_path.suffix == ".csv"
assert list(submission.columns) == list(sample_sub.columns)
assert submission.isna().sum().sum() == 0
assert len(submission) == len(test_df)
submission.describe(include="all")
