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

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
pydicom==3.0.1
Pympler==1.1
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0

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

0.62

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.46471) has done: 'I remove the incompatible/unused imports that trigger the protobuf `MessageFactory.GetPrototype` crash and keep only the libraries actually needed for inference and submission writing. Since the referenced pretrained `.h5` models are not available in your input paths, I replace the missing-model inference with a deterministic, lightweight heuristic that produces valid probability predictions from one representative DICOM slice per case (this restores end-to-end execution and creates a proper `submission.csv`). I also fix the `resize`/`randrange` NameErrors by ensuring the required functions are imported and used correctly. Finally, I ensure the submission uses the exact `BraTS21ID,MGMT_value` columns and matches the sample submission order.'
- What this solution (achieved 0.62) has done: 'Your current target score is `-1.0`, which is not a meaningful AUC destination (AUC is typically in `[0, 1]`), so I assume you actually want to move *upward* from `0.46471` toward a more competitive score while keeping your lightweight “single-slice heuristic” core logic intact. The smallest legitimate improvement is to (1) load a slightly more informative slice (center slice) rather than the first “bright” one found, (2) use all four MRI modalities and fuse their simple statistics (still no model/architecture change), and (3) calibrate the fused z-score using the provided training labels (a tiny logistic regression on the same heuristic features) to better align probabilities for ROC-AUC. This keeps the same overall approach (DICOM → resize → simple stats → sigmoid probability) but makes it more robust and typically improves AUC without adding heavy compute.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import pydicom as dicom
from skimage.transform import resize
from sklearn.linear_model import LogisticRegression

np.random.seed(0)

DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TRAIN_LABELS_PATH = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

print("Using TEST_DIR:", TEST_DIR)
print("Using TRAIN_DIR:", TRAIN_DIR)
print("Train labels exists:", os.path.exists(TRAIN_LABELS_PATH))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB_PATH))




## === cell 1
def _safe_normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx <= mn + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _choose_middle_readable_dcm(dcm_files):
    if not dcm_files:
        return None
    mid = len(dcm_files) // 2
    offsets = [0, -1, 1, -2, 2, -3, 3]
    for off in offsets:
        i = mid + off
        if i < 0 or i >= len(dcm_files):
            continue
        fp = dcm_files[i]
        try:
            ds = dicom.dcmread(fp, stop_before_pixels=False, force=True)
            px = ds.pixel_array
            if px is None:
                continue
            if not np.isfinite(px).all():
                continue
            return fp
        except Exception:
            continue
    return dcm_files[mid]


def load_one_slice_per_case(
    path_root: str, modality: str = "T2w", img_px_size: int = 299
):
    """
    Minimal, robust loader: for each case folder, pick ONE DICOM slice from a chosen modality.
    Returns:
      case_ids: list[str] in the same order as returned images
      images: np.ndarray [N, img_px_size, img_px_size, 3] float32 in [0,1]
    """
    case_dirs = sorted([f.path for f in os.scandir(path_root) if f.is_dir()])
    case_ids, images = [], []

    for case_path in case_dirs:
        case_id = os.path.basename(case_path)  # already zero-padded string like '00002'
        modality_dir = os.path.join(case_path, modality)
        if not os.path.isdir(modality_dir):
            continue

        dcm_files = sorted(
            [
                f.path
                for f in os.scandir(modality_dir)
                if f.is_file() and f.name.lower().endswith(".dcm")
            ]
        )
        if not dcm_files:
            continue

        chosen = _choose_middle_readable_dcm(dcm_files)
        if chosen is None:
            continue

        try:
            ds = dicom.dcmread(chosen, stop_before_pixels=False, force=True)
            img = ds.pixel_array.astype(np.float32)
        except Exception:
            continue

        img = resize(
            img, (img_px_size, img_px_size), anti_aliasing=True, preserve_range=True
        ).astype(np.float32)
        img01 = _safe_normalize01(img)
        stacked = np.stack([img01, img01, img01], axis=-1).astype(np.float32)

        case_ids.append(case_id)
        images.append(stacked)

    images = (
        np.stack(images, axis=0)
        if len(images)
        else np.zeros((0, img_px_size, img_px_size, 3), dtype=np.float32)
    )
    return case_ids, images


def load_all_modalities_one_slice(
    path_root: str, img_px_size: int = 299, modalities=("FLAIR", "T1w", "T1wCE", "T2w")
):
    """
    Change rationale: still "one slice per case", but across all 4 modalities, which is a minimal
    information increase and usually improves AUC while keeping the same heuristic approach.
    Returns:
      case_ids: list[str]
      imgs_by_mod: dict[modality] -> np.ndarray [N,H,W,3]
    """
    case_dirs = sorted([f.path for f in os.scandir(path_root) if f.is_dir()])
    case_ids = [os.path.basename(p) for p in case_dirs]

    imgs_by_mod = {}
    kept_case_ids = None

    for mod in modalities:
        ids, imgs = load_one_slice_per_case(
            path_root, modality=mod, img_px_size=img_px_size
        )
        id_to_img = {i: im for i, im in zip(ids, imgs)}
        if kept_case_ids is None:
            kept_case_ids = [i for i in case_ids if i in id_to_img]
        else:
            kept_case_ids = [i for i in kept_case_ids if i in id_to_img]
        imgs_by_mod[mod] = id_to_img

    aligned = {}
    for mod in modalities:
        aligned[mod] = (
            np.stack([imgs_by_mod[mod][i] for i in kept_case_ids], axis=0)
            if kept_case_ids
            else np.zeros((0, img_px_size, img_px_size, 3), dtype=np.float32)
        )

    print(
        f"Loaded {len(kept_case_ids)} cases with modalities={modalities}. Example batch shape={aligned[modalities[0]].shape if modalities else None}"
    )
    return kept_case_ids, aligned




## === cell 2
test_case_ids, test_imgs_by_mod = load_all_modalities_one_slice(
    TEST_DIR, img_px_size=299
)

if len(test_case_ids) == 0:
    raise RuntimeError(
        "No test cases were loaded. Check TEST_DIR path and folder structure."
    )




## === cell 3
def _img_stats(images: np.ndarray) -> np.ndarray:
    """
    Returns per-image simple stats: mean, std, p10, p90
    """
    means = images.mean(axis=(1, 2, 3))
    stds = images.std(axis=(1, 2, 3))
    p10 = np.quantile(images, 0.10, axis=(1, 2, 3))
    p90 = np.quantile(images, 0.90, axis=(1, 2, 3))
    return np.stack([means, stds, p10, p90], axis=1).astype(np.float32)


def build_feature_matrix_from_modalities(
    case_ids, imgs_by_mod, modalities=("FLAIR", "T1w", "T1wCE", "T2w")
) -> np.ndarray:
    feats = []
    for mod in modalities:
        feats.append(_img_stats(imgs_by_mod[mod]))
    X = np.concatenate(feats, axis=1).astype(np.float32)  # [N, 4 stats * 4 mods = 16]
    return X


def fit_calibrator_on_train(
    img_px_size=299, modalities=("FLAIR", "T1w", "T1wCE", "T2w")
):
    labels = pd.read_csv(TRAIN_LABELS_PATH, dtype={"BraTS21ID": str})
    bad = {"00109", "00123", "00709"}
    labels = labels[~labels["BraTS21ID"].isin(bad)].reset_index(drop=True)

    train_case_ids, train_imgs_by_mod = load_all_modalities_one_slice(
        TRAIN_DIR, img_px_size=img_px_size, modalities=modalities
    )

    label_map = dict(
        zip(labels["BraTS21ID"].astype(str), labels["MGMT_value"].astype(int))
    )
    keep_ids = [i for i in train_case_ids if i in label_map]
    if len(keep_ids) == 0:
        raise RuntimeError("No training cases intersected with labels after loading.")

    idx = {cid: k for k, cid in enumerate(train_case_ids)}
    aligned_imgs = {}
    for mod in modalities:
        arr = train_imgs_by_mod[mod]
        aligned_imgs[mod] = arr[[idx[cid] for cid in keep_ids], ...]

    X_train = build_feature_matrix_from_modalities(
        keep_ids, aligned_imgs, modalities=modalities
    )
    y_train = np.array([label_map[cid] for cid in keep_ids], dtype=np.int32)

    mu = X_train.mean(axis=0, keepdims=True)
    sig = X_train.std(axis=0, keepdims=True) + 1e-6
    Xs = (X_train - mu) / sig

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=200,
        C=1.0,
        class_weight=None,
        random_state=0,
    )
    clf.fit(Xs, y_train)
    print(
        f"Calibrator fit on {len(y_train)} cases. Train positive rate={y_train.mean():.3f}"
    )
    return clf, mu.astype(np.float32), sig.astype(np.float32)


modalities = ("FLAIR", "T1w", "T1wCE", "T2w")
calibrator, X_mu, X_sig = fit_calibrator_on_train(
    img_px_size=299, modalities=modalities
)

X_test = build_feature_matrix_from_modalities(
    test_case_ids, test_imgs_by_mod, modalities=modalities
)
X_test_s = (X_test - X_mu) / X_sig
prediction = calibrator.predict_proba(X_test_s)[:, 1].astype(np.float32)
prediction = np.clip(prediction, 1e-4, 1 - 1e-4)

print("Predictions:", prediction[:10], " ...")
print("Prediction range:", float(prediction.min()), float(prediction.max()))




## === cell 4
def create_sub_from_case_ids(case_ids_list, preds):
    df = pd.DataFrame({"BraTS21ID": case_ids_list, "MGMT_value": preds.astype(float)})
    return df


sub_df = create_sub_from_case_ids(test_case_ids, prediction)

sample = pd.read_csv(SAMPLE_SUB_PATH, dtype={"BraTS21ID": str})
sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float)

print(sub_df.head())
print("Submission shape:", sub_df.shape)
print("Any missing predictions:", sub_df["MGMT_value"].isna().any())



## === cell 5
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print("Columns:", list(sub_df.columns))
print("MGMT_value summary:", sub_df["MGMT_value"].describe())
