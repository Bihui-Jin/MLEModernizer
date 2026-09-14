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

0.9704488894833836

# 6. Current score

0.57299

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.57299) has done: 'I fix the image loading bug by ensuring we append the `.jpg` extension when needed and by searching a couple of common alternative image directories, which unblock feature extraction. Then I make the pipeline robust to PIL not being installed by defaulting to `imageio` (available via scikit-image dependency) or matplotlib as a last resort, keeping the exact same feature logic (64×64 RGB flattened) and model (StandardScaler + MultiOutput LogisticRegression). Finally, I ensure predictions are generated and written into a submission with the exact `sample_submission.csv` column order and a `.csv` suffix.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
CANDIDATE_BASES = [
    "../input/plant-pathology-2020-fgvc7",
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "../input",
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(relpath_options):
    for base in CANDIDATE_BASES:
        for rel in relpath_options:
            p = os.path.join(base, rel)
            if os.path.exists(p):
                return p
    for rel in relpath_options:
        if os.path.exists(rel):
            return rel
    raise FileNotFoundError(
        f"Could not find any of: {relpath_options} in bases: {CANDIDATE_BASES}"
    )


train_csv_path = find_file(["train.csv", "plant-pathology-2020-fgvc7/train.csv"])
test_csv_path = find_file(["test.csv", "plant-pathology-2020-fgvc7/test.csv"])
sample_sub_path = find_file(
    ["sample_submission.csv", "plant-pathology-2020-fgvc7/sample_submission.csv"]
)


def find_images_dir():
    candidates = []
    for base in CANDIDATE_BASES:
        candidates += [
            os.path.join(base, "images"),
            os.path.join(base, "plant-pathology-2020-fgvc7", "images"),
            os.path.join(
                base,
                "plant-pathology-2020-fgvc7",
                "plant-pathology-2020-fgvc7",
                "images",
            ),
        ]
    for d in candidates:
        if os.path.isdir(d):
            return d
    raise FileNotFoundError("Could not find images directory in expected locations.")


images_dir = find_images_dir()

print("train_csv:", train_csv_path)
print("test_csv:", test_csv_path)
print("sample_submission:", sample_sub_path)
print("images_dir:", images_dir)



## === cell 2
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from PIL import Image  # type: ignore

    _READER = "PIL"
except Exception:
    Image = None
    try:
        import imageio.v2 as imageio  # type: ignore

        _READER = "IMAGEIO"
    except Exception:
        imageio = None
        import matplotlib.image as mpimg  # type: ignore

        _READER = "MPL"

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
sub = pd.read_csv(sample_sub_path)

assert "image_id" in sub.columns, "sample_submission must have image_id"
for c in TARGET_COLS:
    if c not in sub.columns:
        raise ValueError(f"Missing target column in sample_submission: {c}")


def _resolve_image_path(image_id):
    candidates = []
    s = str(image_id)

    candidates.append(os.path.join(images_dir, s))
    if not s.lower().endswith(".jpg"):
        candidates.append(os.path.join(images_dir, s + ".jpg"))

    alt_dirs = []
    for base in CANDIDATE_BASES:
        alt_dirs += [
            os.path.join(base, "images"),
            os.path.join(base, "plant-pathology-2020-fgvc7", "images"),
            os.path.join(
                base,
                "plant-pathology-2020-fgvc7",
                "plant-pathology-2020-fgvc7",
                "images",
            ),
        ]
    for d in alt_dirs:
        candidates.append(os.path.join(d, s))
        if not s.lower().endswith(".jpg"):
            candidates.append(os.path.join(d, s + ".jpg"))

    for p in candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"Image not found for image_id={image_id}. Tried: {candidates[:5]} ... total={len(candidates)}"
    )


def load_image_features(image_id, size=(64, 64)):
    img_path = _resolve_image_path(image_id)

    if _READER == "PIL":
        im = Image.open(img_path).convert("RGB").resize(size, resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    else:
        if _READER == "IMAGEIO":
            arr = imageio.imread(img_path).astype(np.float32)
        else:
            arr = mpimg.imread(img_path).astype(np.float32)

        if arr.max() > 1.5:
            arr = arr / 255.0

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[:, :, :3]
        else:
            arr = np.repeat(arr[:, :, :1], 3, axis=2)

        h, w = arr.shape[:2]
        yy = (np.linspace(0, h - 1, size[0])).astype(int)
        xx = (np.linspace(0, w - 1, size[1])).astype(int)
        arr = arr[yy][:, xx]

    return arr.reshape(-1)


X_train = np.vstack([load_image_features(iid) for iid in train_df["image_id"].values])
y_train = train_df[TARGET_COLS].astype(int).values
X_test = np.vstack([load_image_features(iid) for iid in test_df["image_id"].values])

print("Reader:", _READER)
print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)



## === cell 3
base_clf = LogisticRegression(
    solver="lbfgs",
    max_iter=500,
    n_jobs=None,
)

model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", MultiOutputClassifier(base_clf)),
    ]
)

model.fit(X_train, y_train)

probas_list = model.named_steps["clf"].predict_proba(
    model.named_steps["scaler"].transform(X_test)
)

pred = np.zeros((X_test.shape[0], len(TARGET_COLS)), dtype=np.float64)
for j, p in enumerate(probas_list):
    if p.shape[1] == 2:
        pred[:, j] = p[:, 1]
    else:
        pred[:, j] = p[:, 0]

pred = np.clip(pred, 0.0, 1.0)



## === cell 4
pred_df = pd.DataFrame(pred, columns=TARGET_COLS)
pred_df["image_id"] = test_df["image_id"].values

out = sub[["image_id"] + TARGET_COLS].merge(
    pred_df, on="image_id", how="left", suffixes=("", "_pred")
)

for c in TARGET_COLS:
    pc = f"{c}_pred"
    if pc in out.columns:
        out[c] = out[pc].astype(float)
        out.drop(columns=[pc], inplace=True)
    out[c] = out[c].fillna(0.25)

assert (
    out.shape[0] == sub.shape[0]
), "Submission row count mismatch vs sample_submission"
assert list(out.columns) == ["image_id"] + TARGET_COLS, "Submission columns mismatch"
out.to_csv("submission.csv", index=False)

print(out.head())
print("Wrote submission.csv with shape:", out.shape)
