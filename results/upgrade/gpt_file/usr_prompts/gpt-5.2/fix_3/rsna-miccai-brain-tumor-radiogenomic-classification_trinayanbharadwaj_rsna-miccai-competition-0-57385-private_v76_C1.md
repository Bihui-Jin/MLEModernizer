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
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Target score

-1.0

# 6. Current score

0.49882

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.57294) has done: 'I remove/guard imports that crash the notebook environment (the protobuf-related `MessageFactory` error comes from unused packages like `pympler`) and fix missing/incorrect dependencies (`resize` not imported in the right scope, and a leading-space indentation bug in `create_sub`). Since the referenced pre-trained model file is not available in your dataset, I keep the same inference pipeline structure but replace `load_model(...)` with a small CNN built/trained on-the-fly from the provided `train/` + `train_labels.csv`, using the same 5-slice-per-case averaging idea to produce test probabilities. I also fix logic issues in `create_sub` (it was computing `prediction` inside a loop incorrectly) and ensure the submission IDs are formatted exactly like the sample (`00002` etc.) with `MGMT_value` clipped to `[0,1]`. The final script run end-to-end on Kaggle, avoid the known-bad training cases `[00109, 00123, 00709]`, and write `submission.csv` with the correct columns.'
- What this solution (achieved 0.49882) has done: 'The crash comes from importing TensorFlow (it triggers a protobuf `MessageFactory.GetPrototype` issue in this Kaggle image), so I keep your exact slice-loading + 5-slice averaging pipeline but replace the TF/Keras model with a lightweight, dependency-safe scikit-learn model that runs in this environment. I also fix a logic bug in `load_case_T2W_slices`: selecting `mri_types[3]` is not reliably T2w because directory ordering is not guaranteed; I explicitly pick the `T2w` folder by name to avoid silently training on the wrong sequence (this should improve AUC toward your target while staying within the same “use T2w slices” core idea). Finally, I keep the submission formatting identical to the sample and ensure `submission.csv` is always written with valid probabilities.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import pydicom as dicom
from skimage.transform import resize

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

SEED = 42
np.random.seed(SEED)



## === cell 1
BASE_PATH = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")
LABELS_CSV = os.path.join(BASE_PATH, "train_labels.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

labels_df = pd.read_csv(LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

print("Train labels:", labels_df.shape, labels_df.columns.tolist())
print("Sample submission:", sample_sub.shape, sample_sub.columns.tolist())




## === cell 2
def _safe_norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mx = float(np.max(x)) if x.size else 0.0
    if mx <= 0:
        return np.zeros_like(x, dtype=np.float32)
    return x / mx


def load_case_T2W_slices(
    case_path, img_px_size=150, n_slices=5, min_sum=100000, min_norm_sum=2500
):
    """
    Returns: list of up to n_slices images (H,W,3) float32 in [0,1]

    Bug fix (score-relevant, minimal): previously used mri_types[3] after sorting directories,
    which is not guaranteed to correspond to T2w. We now explicitly select the 'T2w' folder.
    """
    t2w_path = os.path.join(case_path, "T2w")
    if not os.path.isdir(t2w_path):
        return []

    img_files = sorted(
        [
            f.path
            for f in os.scandir(t2w_path)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )
    out = []
    for fp in img_files:
        if len(out) >= n_slices:
            break
        try:
            ds = dicom.dcmread(fp)
            arr = ds.pixel_array
        except Exception:
            continue
        if arr is None:
            continue
        if float(np.sum(arr)) <= min_sum:
            continue

        resized_img = resize(
            arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        resized_img = _safe_norm01(resized_img)
        stacked = np.stack((resized_img,) * 3, axis=-1)

        if float(np.sum(stacked)) <= min_norm_sum:
            continue

        out.append(stacked.astype(np.float32))
    return out


def build_dataset_from_ids(root_dir, ids, img_px_size=150, n_slices=5):
    """
    Builds (X, groups) where:
    - X: (N, H, W, 3)
    - groups: case id strings for grouping/aggregation
    Uses up to n_slices per case; if fewer are found, repeats last slice to reach n_slices.
    """
    X_list, g_list = [], []
    for case_id in ids:
        case_folder = os.path.join(root_dir, str(case_id).zfill(5))
        if not os.path.isdir(case_folder):
            continue

        imgs = load_case_T2W_slices(
            case_folder, img_px_size=img_px_size, n_slices=n_slices
        )
        if len(imgs) == 0:
            continue

        while len(imgs) < n_slices:
            imgs.append(imgs[-1])

        for im in imgs[:n_slices]:
            X_list.append(im)
            g_list.append(str(case_id).zfill(5))

    X = (
        np.stack(X_list, axis=0).astype(np.float32)
        if len(X_list)
        else np.zeros((0, img_px_size, img_px_size, 3), dtype=np.float32)
    )
    return X, np.array(g_list)




## === cell 3
bad_cases = {109, 123, 709}
labels_df = labels_df[~labels_df["BraTS21ID"].isin(bad_cases)].reset_index(drop=True)

train_ids = labels_df["BraTS21ID"].tolist()
test_ids = sample_sub["BraTS21ID"].tolist()

train_ids_str = [str(i).zfill(5) for i in train_ids]
test_ids_str = [str(i).zfill(5) for i in test_ids]

print("Train cases:", len(train_ids_str), "Test cases:", len(test_ids_str))



## === cell 4
IMG_PX_SIZE = 150
N_SLICES = 5

label_map = dict(
    zip(
        labels_df["BraTS21ID"].astype(int).astype(str).str.zfill(5),
        labels_df["MGMT_value"].astype(np.float32),
    )
)

X_train_slices, g_train = build_dataset_from_ids(
    TRAIN_DIR, train_ids, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
)
y_train_slices = np.array([label_map.get(g, np.nan) for g in g_train], dtype=np.float32)

mask = ~np.isnan(y_train_slices)
X_train_slices = X_train_slices[mask]
y_train_slices = y_train_slices[mask]
g_train = g_train[mask]

print("Loaded train slices:", X_train_slices.shape, "labels:", y_train_slices.shape)



## === cell 5
X_flat = X_train_slices.reshape((X_train_slices.shape[0], -1)).astype(np.float32)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_flat,
    y_train_slices,
    test_size=0.15,
    random_state=SEED,
    stratify=y_train_slices,
)

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(max_iter=1000, random_state=SEED, solver="lbfgs")),
    ]
)

clf.fit(X_tr, y_tr)
val_proba = clf.predict_proba(X_val)[:, 1]
print(
    "Validation proba stats:",
    float(np.min(val_proba)),
    float(np.mean(val_proba)),
    float(np.max(val_proba)),
)



## === cell 6
X_test_slices, g_test = build_dataset_from_ids(
    TEST_DIR, [int(x) for x in test_ids_str], img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
)
print("Loaded test slices:", X_test_slices.shape, "groups:", g_test.shape)

X_test_flat = X_test_slices.reshape((X_test_slices.shape[0], -1)).astype(np.float32)



## === cell 7
test_pred_slices = clf.predict_proba(X_test_flat)[:, 1].astype(np.float32)

pred_df = pd.DataFrame({"BraTS21ID": g_test, "p": test_pred_slices})
pred_case = (
    pred_df.groupby("BraTS21ID", as_index=False)["p"]
    .mean()
    .rename(columns={"p": "MGMT_value"})
)

sub_df = sample_sub.copy()
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
pred_case["BraTS21ID"] = pred_case["BraTS21ID"].astype(str).str.zfill(5)

sub_df = sub_df.drop(columns=["MGMT_value"]).merge(
    pred_case, on="BraTS21ID", how="left"
)
sub_df["MGMT_value"] = (
    sub_df["MGMT_value"].fillna(0.5).clip(0.0, 1.0).astype(np.float32)
)

print(sub_df.head())
print(
    "Submission rows:",
    len(sub_df),
    "missing preds filled:",
    int(sub_df["MGMT_value"].isna().sum()),
)



## === cell 8
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with columns:", sub_df.columns.tolist())
print("submission.csv size (bytes):", os.path.getsize("submission.csv"))
