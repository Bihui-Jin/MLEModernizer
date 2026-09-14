# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.9

# 3. Installed packages

No external packages required in the script and installed.

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

# 5. Code solution

## === cell 0
import os, random, warnings

warnings.filterwarnings("ignore")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from PIL import Image

seed = 42
np.random.seed(seed)
random.seed(seed)

print("Setup complete.")




## === cell 1
BASE_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"
IMG_DIR = os.path.join(BASE_PATH, "images")
train_csv = os.path.join(BASE_PATH, "train.csv")
test_csv = os.path.join(BASE_PATH, "test.csv")
sample_sub_csv = os.path.join(BASE_PATH, "sample_submission.csv")

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
sub_df = pd.read_csv(sample_sub_csv)


def img_path(st):
    return os.path.join(IMG_DIR, f"{st}.jpg")




## === cell 2
IMG_SIZE = 64  # kept small for speed
HIST_BINS = 256  # higher‑resolution colour histogram
DOWNSAMPLE_SIZE = 12  # larger down‑sampled raw‑pixel descriptor


def load_features(paths):
    X = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            arr = np.asarray(img, dtype=np.float32) / 255.0  # (64,64,3)

            hists = []
            for ch in range(3):
                hist, _ = np.histogram(
                    arr[:, :, ch], bins=HIST_BINS, range=(0.0, 1.0), density=True
                )
                hists.append(hist)
            hist_feats = np.concatenate(hists)  # 3 * HIST_BINS

            means = arr.mean(axis=(0, 1))
            stds = arr.std(axis=(0, 1))
            stats = np.concatenate([means, stds])  # 6

            small_img = Image.fromarray((arr * 255).astype(np.uint8)).resize(
                (DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE), Image.BILINEAR
            )
            small_arr = np.asarray(small_img, dtype=np.float32) / 255.0
            raw_feats = small_arr.flatten()  # DOWNSAMPLE_SIZE*DOWNSAMPLE_SIZE*3

            feats = np.concatenate([hist_feats, stats, raw_feats])
        except Exception:
            feats = np.zeros(
                3 * HIST_BINS + 6 + DOWNSAMPLE_SIZE * DOWNSAMPLE_SIZE * 3,
                dtype=np.float32,
            )
        X.append(feats)
    return np.stack(X)


train_paths = train_df["image_id"].apply(img_path).values
test_paths = test_df["image_id"].apply(img_path).values

X_all = load_features(train_paths)
y_all = train_df.loc[:, "healthy":].values  # 4 label columns

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.10, random_state=seed
)




## === cell 3
base_clf = GradientBoostingClassifier(
    n_estimators=2000,  # a bit more trees for higher capacity
    learning_rate=0.02,
    max_depth=6,
    subsample=0.8,  # stochastic gradient boosting
    max_features=0.8,  # feature sub‑sampling
    random_state=seed,
)
model = OneVsRestClassifier(base_clf, n_jobs=-1)

print("Training model...")
model.fit(X_train, y_train)

val_pred = model.predict_proba(X_val)
val_auc = np.mean(
    [roc_auc_score(y_val[:, i], val_pred[:, i]) for i in range(y_val.shape[1])]
)
print(f"Validation mean ROC‑AUC: {val_auc:.5f}")




## === cell 4
X_test = load_features(test_paths)
test_pred = model.predict_proba(X_test)  # (n_test, 4)

sub_df.loc[:, "healthy":] = test_pred
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print(f"Submission saved to {sub_path}")
sub_df.head()
