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

# 5. Code solution

## === cell 0
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.linear_model import LogisticRegression

BASE_INPUT = Path("/kaggle/input")
WORKING_DIR = Path("/kaggle/working")



## === cell 1
train_path = BASE_INPUT / "plant-pathology-2020-fgvc7" / "train.csv"
train_df = pd.read_csv(train_path)

prior_probs = {
    "healthy": train_df["healthy"].mean(),
    "multiple_diseases": train_df["multiple_diseases"].mean(),
    "rust": train_df["rust"].mean(),
    "scab": train_df["scab"].mean(),
}
print("Prior probabilities:", prior_probs)



## === cell 2
test_path = BASE_INPUT / "plant-pathology-2020-fgvc7" / "test.csv"
test_df = pd.read_csv(test_path)

sub = pd.DataFrame(
    {
        "image_id": test_df["image_id"],
        "healthy": 0.0,
        "multiple_diseases": 0.0,
        "rust": 0.0,
        "scab": 0.0,
    }
)
print("Initial submission shape:", sub.shape)



## === cell 3
possible_files = [
    "fork-of-plant-2020-tpu-915e9c_version1.csv",
    "plant-pathology-pytorch-efficientnet-b4-gpu_version_7.csv",
    "public-first-score-tpu-incepresnetv2-enb7_version8.csv",
    "plant-pathology-2020-efficientnetb7-0-980-score.csv",
    "tf-zoo-models-on-tpu.csv",
]

for extra_path in WORKING_DIR.glob("*.csv"):
    if extra_path.name not in possible_files:
        possible_files.append(extra_path.name)

search_dirs = [BASE_INPUT / "plant-pathology-2020-fgvc7", WORKING_DIR]
dsub = []  # list of external prediction DataFrames

for fdir in search_dirs:
    for fname in possible_files:
        fpath = fdir / fname
        if fpath.is_file():
            try:
                df = pd.read_csv(fpath)
                dsub.append(df)
                print(f"Loaded {fname} from {fdir}")
            except Exception as e:
                print(f"Error loading {fname} from {fdir}: {e}")
        else:
            pass

print(f"Number of external prediction files loaded: {len(dsub)}")



## === cell 4
images_dir = BASE_INPUT / "plant-pathology-2020-fgvc7" / "images"


def extract_features(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0  # (H, W, 3)

        means = arr.mean(axis=(0, 1))
        stds = arr.std(axis=(0, 1))

        medians = np.median(arr, axis=(0, 1))

        sum_rgb = means.sum() + 1e-6
        r_ratio = means[0] / sum_rgb
        g_ratio = means[1] / sum_rgb
        b_ratio = means[2] / sum_rgb

        return np.concatenate([means, stds, medians, [r_ratio, g_ratio, b_ratio]])
    except Exception:
        return np.zeros(12, dtype=np.float32)


X_train = []
y_train = {c: [] for c in ["healthy", "multiple_diseases", "rust", "scab"]}

for _, row in train_df.iterrows():
    img_path = images_dir / f"{row['image_id']}.jpg"
    feats = extract_features(img_path)
    X_train.append(feats)
    for c in y_train:
        y_train[c].append(row[c])

X_train = np.array(X_train)

models = {}
for c in y_train:
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=30.0)
    lr.fit(X_train, y_train[c])
    models[c] = lr

X_test = []
test_ids = []
for img_id in test_df["image_id"]:
    img_path = images_dir / f"{img_id}.jpg"
    X_test.append(extract_features(img_path))
    test_ids.append(img_id)

X_test = np.array(X_test)

preds = {"image_id": test_ids}
for c, model in models.items():
    preds[c] = model.predict_proba(X_test)[:, 1]  # probability of class 1

model_pred_df = pd.DataFrame(preds)

dsub = [model_pred_df]

print("Added model predictions to blending pool; total sources now:", len(dsub))



## === cell 5
prior_weight = 0.0

model_weight = 1.0

for d in dsub:
    d = d.set_index("image_id")
    sub = sub.set_index("image_id")
    for col in ["healthy", "multiple_diseases", "rust", "scab"]:
        if col in d.columns:
            sub[col] += d[col]
    sub = sub.reset_index()

sub["healthy"] = (sub["healthy"] + prior_weight * prior_probs["healthy"]) / model_weight
sub["multiple_diseases"] = (
    sub["multiple_diseases"] + prior_weight * prior_probs["multiple_diseases"]
) / model_weight
sub["rust"] = (sub["rust"] + prior_weight * prior_probs["rust"]) / model_weight
sub["scab"] = (sub["scab"] + prior_weight * prior_probs["scab"]) / model_weight



## === cell 6
output_path = WORKING_DIR / "submission.csv"
sub.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
