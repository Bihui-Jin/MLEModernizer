# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Identify fractures in CT scans of the cervical spine (neck) at both the level of a single vertebrae and the entire patient.

## Metric
Weighted multi-label logarithmic loss. Each fracture sub-type is its own row for every exam, and you are expected to predict a probability for a fracture at each of the seven cervical vertebrae designated as C1, C2, C3, C4, C5, C6 and C7. There is also an any label, `patient_overall`, which indicates that a fracture of ANY kind described before exists in the examination. Fractures in the skull base, thoracic spine, ribs, and clavicles are ignored. The any label is weighted more highly than specific fracture level sub-types.

For each exam Id, you must submit a set of predicted probabilities (a separate row for each cervical level subtype). We then take the log loss for each predicted probability versus its true label.

The binary weighted log loss function for label j on exam i is specified as:

$$
L_{i j}=-w_j *\left[y_{i j} * \log \left(p_{i j}\right)+\left(1-y_{i j}\right) * \log \left(1-p_{i j}\right)\right]
$$

Finally, loss is averaged across all rows.

## Submission Format
There will be 8 rows per image Id. The label indicated by a particular row will look like [image Id]_[Sub-type Name], as follows. There is also a target column, `fractured`, indicating the probability of whether a fracture exists at the specified level. For each image ID in the test set, you must predict a probability for each of the different possible sub-types and the patient overall. The file should contain a header and have the following format:

```
row_id,fractured
1_C1,0
1_C2,0
1_C3,0
1_C4,0.6
1_C5,0
1_C6,0.9
1_C7,0.01
1_patient_overall,0.99
2_C1,0
etc.
```

## Dataset
**train.csv** Metadata for the train test set.

- `StudyInstanceUID` - The study ID. There is one unique study ID for each patient scan.
- `patient_overall` - One of the target columns. The patient level outcome, i.e. if any of the vertebrae are fractured.
- `C[1-7]` - The other target columns. Whether the given vertebrae is fractured. See [this diagram](https://en.wikipedia.org/wiki/Vertebral_column#/media/File:Gray_111_-_Vertebral_column-coloured.png) for the real location of each vertbrae in the spine.

**test.csv** Metadata for the test set prediction structure. Only the first few rows of the test set are available for download.

- `row_id` - The row ID. This will match the same column in the sample submission file.
- `StudyInstanceUID` - The study ID.
- `prediction_type` - Which one of the eight target columns needs a prediction in this row.

**[train/test]_images/[StudyInstanceUID]/[slice_number].dcm** The image data, organized with one folder per scan. Expect to see roughly 1,500 scans in the hidden test set.\

Each image is in [the dicom file format](https://www.dicomstandard.org/). The DICOM image files are ≤ 1 mm slice thickness, axial orientation, and bone kernel. Note that some of the DICOM files are JPEG compressed. You may require additional resources to read the pixel array of these files, such as GDCM and pylibjpeg.

**sample_submission.csv** A valid sample submission.

- `row_id` - The row ID. See the test.csv for what prediction needs to be filed in that row.
- `fractured` - The target column.

**train_bounding_boxes.csv** Bounding boxes for a subset of the training set.

**segmentations/** Pixel level annotations for a subset of the training set. This data is provided in the [nifti file format](https://nifti.nimh.nih.gov/).

A portion of the imaging datasets have been segmented automatically using a 3D UNET model, and radiologists modified and approved the segmentations. The provided segmentation labels have values of 1 to 7 for C1 to C7 (seven cervical vertebrae) and 8 to 19 for T1 to T12 (twelve thoracic vertebrae are located in the center of your upper and middle back), and 0 for everything else. As we focused on the cervical spine, all scans have C1 to C7 labels but not all thoracic labels.

Please be aware that the NIFTI files consist of segmentation in the sagittal plane, while the DICOM files are in the axial plane. Please use the NIFTI header information to determine the appropriate orientation such that the DICOM images and segmentation match. Otherwise, you run the risk of having the segmentations flipped in the Z axis and mirrored in the X axis.

# 2. Python version

3.11

# 3. Installed packages

cloudpathlib==0.21.1
cuda-pathfinder==1.3.2
geopandas==0.14.4
jmespath==1.0.1
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
path==17.1.1
path.py==12.5.0
pathos==0.3.2
pathspec==0.12.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
simpleitk==2.5.2
sklearn-pandas==2.2.0
testpath==0.6.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (276 lines)
            sample_submission.csv (14537 lines)
            sample_submission.csv.zip (50.3 kB)
            segmentations.zip (3.0 MB)
            test.csv (14537 lines)
            test.csv.zip (94.5 kB)
            test.zip (160 Bytes)
            test_images.zip (181.9 GB)
            train.csv (203 lines)
            train.csv.zip (1.2 kB)
            train.zip (162 Bytes)
            train_bounding_boxes.csv (691 lines)
            train_bounding_boxes.csv.zip (11.5 kB)
            train_images.zip (20.3 GB)
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
            segmentations/
                1.2.826.0.1.3680043.12292.nii (89.4 MB)
                1.2.826.0.1.3680043.24617.nii (307.2 MB)
                ... and 7 other files
            test_images/
                1.2.826.0.1.3680043.10001/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 266 other files
                1.2.826.0.1.3680043.10005/
                    1.dcm (525.1 kB)
                    10.dcm (525.1 kB)
                    ... and 257 other files
                ... and 1816 other folders
            train_images/
                1.2.826.0.1.3680043.10014/
                    1.dcm (240.9 kB)
                    10.dcm (250.8 kB)
                    ... and 256 other files
                1.2.826.0.1.3680043.10058/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 574 other files
                ... and 201 other folders
        input/
            description.md (276 lines)
            sample_submission.csv (14537 lines)
            sample_submission.csv.zip (50.3 kB)
            segmentations.zip (3.0 MB)
            test.csv (14537 lines)
            test.csv.zip (94.5 kB)
            test.zip (160 Bytes)
            test_images.zip (181.9 GB)
            train.csv (203 lines)
            train.csv.zip (1.2 kB)
            train.zip (162 Bytes)
            train_bounding_boxes.csv (691 lines)
            train_bounding_boxes.csv.zip (11.5 kB)
            train_images.zip (20.3 GB)
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
            segmentations/
                1.2.826.0.1.3680043.12292.nii (89.4 MB)
                1.2.826.0.1.3680043.24617.nii (307.2 MB)
                ... and 7 other files
            test_images/
                1.2.826.0.1.3680043.10001/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 266 other files
                1.2.826.0.1.3680043.10005/
                    1.dcm (525.1 kB)
                    10.dcm (525.1 kB)
                    ... and 257 other files
                ... and 1816 other folders
            train_images/
                1.2.826.0.1.3680043.10014/
                    1.dcm (240.9 kB)
                    10.dcm (250.8 kB)
                    ... and 256 other files
                1.2.826.0.1.3680043.10058/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 574 other files
                ... and 201 other folders
        working/
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
```

-> data/rsna-2022-cervical-spine-fracture-detection/sample_submission.csv has 14536 rows and 2 columns.
The columns are: row_id, fractured

-> data/rsna-2022-cervical-spine-fracture-detection/test.csv has 14536 rows and 3 columns.
The columns are: StudyInstanceUID, prediction_type, row_id

-> data/rsna-2022-cervical-spine-fracture-detection/train.csv has 202 rows and 9 columns.
The columns are: StudyInstanceUID, patient_overall, C1, C2, C3, C4, C5, C6, C7

-> data/rsna-2022-cervical-spine-fracture-detection/train_bounding_boxes.csv has 690 rows and 6 columns.
The columns are: StudyInstanceUID, x, y, width, height, slice_number

-> data/sample_submission.csv has 14536 rows and 2 columns.
The columns are: row_id, fractured

-> data/test.csv has 14536 rows and 3 columns.
The columns are: StudyInstanceUID, prediction_type, row_id

-> data/train.csv has 202 rows and 9 columns.
The columns are: StudyInstanceUID, patient_overall, C1, C2, C3, C4, C5, C6, C7

-> data/train_bounding_boxes.csv has 690 rows and 6 columns.
The columns are: StudyInstanceUID, x, y, width, height, slice_number

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd

DATA_ROOT = "/kaggle/data"
COMP_DIR = os.path.join(DATA_ROOT, "rsna-2022-cervical-spine-fracture-detection")

TRAIN_CSV = os.path.join(COMP_DIR, "train.csv")
TEST_CSV = os.path.join(COMP_DIR, "test.csv")
SAMPLE_SUB = os.path.join(COMP_DIR, "sample_submission.csv")

TRAIN_IMG_DIR = os.path.join(COMP_DIR, "train_images")
TEST_IMG_DIR = os.path.join(COMP_DIR, "test_images")

SAVE_CSV = "submission.csv"

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(TEST_CSV), f"Missing {TEST_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

print("Using data from:", COMP_DIR)
print("Will write:", os.path.abspath(SAVE_CSV))




## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)[["row_id"]]

label_cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
c_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

print(
    "Train shape:",
    train_df.shape,
    "Test rows:",
    test_df.shape,
    "Sample rows:",
    sample_df.shape,
)




## === cell 2
def count_slices_for_uids(
    img_root: str,
    uids,
    max_header_reads_per_uid: int = 64,  # bounded escalation; preserves logic, reduces worst-case I/O
    head_tail_reads: int = 8,  # fast path: read only a few headers
):
    """
    Returns a robust estimate of number of slices in a study.
    Prefer: (max InstanceNumber - min InstanceNumber + 1) from DICOM headers (no pixel decode).
    Fallback: number of .dcm files.
    """
    try:
        import pydicom  # available on Kaggle images typically
        from pydicom.errors import InvalidDicomError
    except Exception:
        pydicom = None
        InvalidDicomError = Exception

    out = {}
    join = os.path.join

    for uid in uids:
        d = join(img_root, uid)
        try:
            dcm_files = []
            with os.scandir(d) as it:
                for e in it:
                    if e.is_file():
                        n = e.name
                        if n.lower().endswith(".dcm"):
                            dcm_files.append(n)
        except FileNotFoundError:
            out[uid] = 0
            continue

        if not dcm_files:
            out[uid] = 0
            continue

        dcm_files.sort()
        fallback_n = int(len(dcm_files))

        if pydicom is None:
            out[uid] = fallback_n
            continue

        def _read_inst(fname: str):
            fp = join(d, fname)
            try:
                ds = pydicom.dcmread(
                    fp, stop_before_pixels=True, specific_tags=["InstanceNumber"]
                )
                inst = getattr(ds, "InstanceNumber", None)
                if inst is None:
                    return None
                return int(inst)
            except (
                InvalidDicomError,
                FileNotFoundError,
                PermissionError,
                OSError,
                ValueError,
            ):
                return None
            except Exception:
                return None

        inst_min = None
        inst_max = None

        k = min(head_tail_reads, fallback_n)
        probe_files = dcm_files[:k] + (dcm_files[-k:] if fallback_n > k else [])
        read_n = 0
        for f in probe_files:
            inst = _read_inst(f)
            read_n += 1
            if inst is None:
                continue
            inst_min = inst if inst_min is None else min(inst_min, inst)
            inst_max = inst if inst_max is None else max(inst_max, inst)

        if inst_min is None or inst_max is None:
            inst_min = None
            inst_max = None
            read_n = 0
            for f in dcm_files:
                if read_n >= max_header_reads_per_uid:
                    break
                inst = _read_inst(f)
                read_n += 1
                if inst is None:
                    continue
                inst_min = inst if inst_min is None else min(inst_min, inst)
                inst_max = inst if inst_max is None else max(inst_max, inst)

        if inst_min is not None and inst_max is not None and inst_max >= inst_min:
            est = int(inst_max - inst_min + 1)
            est = max(1, min(est, int(max(2 * fallback_n, fallback_n + 10))))
            out[uid] = est
        else:
            out[uid] = fallback_n

    return out


train_uids = train_df["StudyInstanceUID"].tolist()
test_uids = sorted(test_df["StudyInstanceUID"].unique().tolist())

train_slice_count = count_slices_for_uids(TRAIN_IMG_DIR, train_uids)
test_slice_count = count_slices_for_uids(TEST_IMG_DIR, test_uids)

train_df["n_slices"] = (
    train_df["StudyInstanceUID"].map(train_slice_count).fillna(0).astype(np.int32)
)
test_uid_to_slices = pd.Series(test_slice_count, name="n_slices")

print(
    "n_slices train min/med/max:",
    int(train_df["n_slices"].min()),
    float(train_df["n_slices"].median()),
    int(train_df["n_slices"].max()),
)
print(
    "n_slices test min/med/max:",
    int(test_uid_to_slices.min()),
    float(test_uid_to_slices.median()),
    int(test_uid_to_slices.max()),
)




## === cell 3
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_logistic_1d_irls(x, y, l2=1.0, iters=50, laplace=1.0):
    """
    Logistic regression on 1D feature x with bias b using IRLS/Newton steps.
    Minimize mean NLL + 0.5*l2*w^2 (do not regularize bias).
    Returns (b, w).
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    p0 = (y.sum() + laplace) / (len(y) + 2.0 * laplace)
    p0 = float(np.clip(p0, 1e-6, 1 - 1e-6))
    b = math.log(p0 / (1.0 - p0))
    w = 0.0

    for _ in range(iters):
        z = b + w * x
        p = sigmoid(z)

        r = np.clip(p * (1.0 - p), 1e-6, None)
        gb = (p - y).mean()
        gw = ((p - y) * x).mean() + l2 * w

        hbb = r.mean()
        hbw = (r * x).mean()
        hww = (r * x * x).mean() + l2

        det = hbb * hww - hbw * hbw
        if not np.isfinite(det) or det <= 1e-12:
            break

        db = (hww * gb - hbw * gw) / det
        dw = (-hbw * gb + hbb * gw) / det

        b -= db
        w -= dw

        if max(abs(db), abs(dw)) < 1e-8:
            break

    return float(b), float(w)


def predict_logistic_1d(x, b, w):
    x = x.astype(np.float64)
    return sigmoid(b + w * x)


def weighted_logloss_binary(y_true, p_pred, w=1.0, eps=1e-6):
    y_true = y_true.astype(np.float64)
    p_pred = np.clip(p_pred.astype(np.float64), eps, 1.0 - eps)
    return float(
        w * (-(y_true * np.log(p_pred) + (1.0 - y_true) * np.log(1.0 - p_pred))).mean()
    )


def kaggle_rowwise_weighted_logloss(
    y_true_by_label, p_pred_by_label, weight_by_label, eps=1e-6
):
    tot = 0.0
    wsum = 0.0
    for c, y in y_true_by_label.items():
        w = float(weight_by_label.get(c, 1.0))
        p = np.clip(p_pred_by_label[c].astype(np.float64), eps, 1.0 - eps)
        y = y.astype(np.float64)
        tot += w * (-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))).mean()
        wsum += w
    return float(tot / max(wsum, 1e-12))


def noisy_or_any(p_mat, gamma=1.0, eps=1e-6):
    """
    p_mat: (n, 7) probabilities for C1..C7
    gamma>1 reduces overconfidence when C-levels are correlated.
    """
    p = np.clip(p_mat.astype(np.float64), eps, 1.0 - eps)
    log_q = np.log1p(-p)  # log(1-p)
    any_p = 1.0 - np.exp(np.clip(gamma * log_q.sum(axis=1), -50, 0))
    return np.clip(any_p, eps, 1.0 - eps)


x_train_raw = np.log1p(train_df["n_slices"].to_numpy(dtype=np.float64))
x_test_raw = np.log1p(test_uid_to_slices.to_numpy(dtype=np.float64))

x_mean = float(x_train_raw.mean())
x_std = float(x_train_raw.std())
if not np.isfinite(x_std) or x_std <= 1e-12:
    x_std = 1.0

x_train = (x_train_raw - x_mean) / x_std
x_test_uid = (x_test_raw - x_mean) / x_std

prevalence = {c: float(train_df[c].mean()) for c in label_cols}

l2_by_label = {"patient_overall": 0.5}
for c in c_cols:
    p = prevalence[c]
    if p < 0.03:
        l2_by_label[c] = 10.0
    elif p < 0.06:
        l2_by_label[c] = 5.0
    else:
        l2_by_label[c] = 2.0


def loo_preds_for_label(x, y, l2, iters=50, laplace=1.0):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    out = np.zeros(n, dtype=np.float64)
    idx = np.arange(n)

    for i in range(n):
        m = idx != i
        b, w = fit_logistic_1d_irls(x[m], y[m], l2=l2, iters=iters, laplace=laplace)
        out[i] = predict_logistic_1d(x[i : i + 1], b, w)[0]
    return out


weight_by_label = {c: 1.0 for c in label_cols}
weight_by_label["patient_overall"] = 2.0

alphas = {}
params = {}
p0_by_label = {}
p_loo_by_label = {}
y_by_label = {}

for c in label_cols:
    y = train_df[c].to_numpy(dtype=np.float64)
    y_by_label[c] = y
    l2 = float(l2_by_label[c])

    p_loo = loo_preds_for_label(x_train, y, l2=l2, iters=50, laplace=1.0)
    p_loo_by_label[c] = p_loo

    p0 = float((y.sum() + 1.0) / (len(y) + 2.0))
    p0_by_label[c] = p0

    b, w = fit_logistic_1d_irls(x_train, y, l2=l2, iters=50, laplace=1.0)
    params[c] = (b, w)

alpha_grid = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0)

for c in label_cols:
    alphas[c] = 1.0


def blended_preds_for_label(c, alpha):
    return alpha * p_loo_by_label[c] + (1.0 - alpha) * p0_by_label[c]


def apply_patient_overall_constraint(p_pred_by_label, eps=1e-6):
    out = {k: v.copy() for k, v in p_pred_by_label.items()}
    mx = np.maximum.reduce([out[c] for c in c_cols])
    out["patient_overall"] = np.maximum(out["patient_overall"], mx)
    out["patient_overall"] = np.clip(out["patient_overall"], eps, 1.0 - eps)
    return out


def c_only_weighted_logloss(y_by_label, p_pred_by_label, eps=1e-6):
    tot = 0.0
    for c in c_cols:
        y = y_by_label[c]
        p = np.clip(p_pred_by_label[c].astype(np.float64), eps, 1.0 - eps)
        y = y.astype(np.float64)
        tot += (-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))).mean()
    return float(tot / 7.0)


for _pass in range(2):
    for c in c_cols:
        best_a = None
        best_loss = None
        for a in alpha_grid:
            p_pred_by_label = {}
            for cc in c_cols:
                aa = a if (cc == c) else alphas[cc]
                p_pred_by_label[cc] = blended_preds_for_label(cc, aa)

            loss = c_only_weighted_logloss(y_by_label, p_pred_by_label, eps=1e-6)
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_a = a
        alphas[c] = float(best_a)

gamma_grid = (1.0, 1.2, 1.5, 1.8, 2.2)
best_po = {"loss": None, "alpha_po": None, "gamma": None}

p_c_loo_tuned = np.stack(
    [blended_preds_for_label(c, alphas[c]) for c in c_cols], axis=1
)

p_po_direct = blended_preds_for_label("patient_overall", 1.0)
p_po_prior = np.full_like(p_po_direct, p0_by_label["patient_overall"], dtype=np.float64)

alpha_po_grid = (0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0)
beta_grid = (0.0, 0.25, 0.5, 0.75, 1.0)
best_beta = None

for gamma in gamma_grid:
    p_any = noisy_or_any(p_c_loo_tuned, gamma=gamma, eps=1e-6)
    for alpha_po in alpha_po_grid:
        p_dir = alpha_po * p_po_direct + (1.0 - alpha_po) * p_po_prior
        for beta in beta_grid:
            p_pred_by_label = {c: p_c_loo_tuned[:, i] for i, c in enumerate(c_cols)}
            p_pred_by_label["patient_overall"] = beta * p_any + (1.0 - beta) * p_dir
            p_pred_by_label = apply_patient_overall_constraint(
                p_pred_by_label, eps=1e-6
            )
            loss = kaggle_rowwise_weighted_logloss(
                y_by_label, p_pred_by_label, weight_by_label, eps=1e-6
            )
            if best_po["loss"] is None or loss < best_po["loss"]:
                best_po = {
                    "loss": float(loss),
                    "alpha_po": float(alpha_po),
                    "gamma": float(gamma),
                }
                best_beta = float(beta)

alphas["patient_overall"] = best_po["alpha_po"]
patient_overall_gamma = best_po["gamma"]
patient_overall_beta = best_beta

print(
    "Tuned patient_overall: alpha_po(prior-vs-direct)=",
    alphas["patient_overall"],
    "gamma(anyC)=",
    patient_overall_gamma,
    "beta(anyC-vs-direct)=",
    patient_overall_beta,
    "LOO objective=",
    best_po["loss"],
)

for c in label_cols:
    b, w = params[c]
    p_med = float(predict_logistic_1d(np.array([np.median(x_train)]), b, w)[0])
    y = y_by_label[c]
    if c in c_cols:
        p_blend = blended_preds_for_label(c, alphas[c])
        loss_c = weighted_logloss_binary(y, p_blend, w=weight_by_label[c], eps=1e-6)
        print(
            f"{c:15s} prev={prevalence[c]:.4f} l2={l2_by_label[c]:.2f} "
            f"alpha={alphas[c]:.2f} b={b:+.4f} w={w:+.4f} p(median)={p_med:.4f} "
            f"LOO_loss(label)~{loss_c:.5f}"
        )
    else:
        loss_c = weighted_logloss_binary(
            y, blended_preds_for_label(c, 1.0), w=weight_by_label[c], eps=1e-6
        )
        print(
            f"{c:15s} prev={prevalence[c]:.4f} l2={l2_by_label[c]:.2f} "
            f"alpha_po={alphas[c]:.2f} b={b:+.4f} w={w:+.4f} p(median)={p_med:.4f} "
            f"LOO_loss(direct_only)~{loss_c:.5f}"
        )




## === cell 4
eps = 1e-4  # logloss safety clip

test_uid_arr = test_uid_to_slices.index.to_numpy()
x_test_arr = x_test_uid

uid_pred = {uid: {} for uid in test_uid_arr}

p_c_test = []
for c in c_cols:
    b, w = params[c]
    p_model = predict_logistic_1d(x_test_arr, b, w)

    y_train = train_df[c].to_numpy(dtype=np.float64)
    p0 = (y_train.sum() + 1.0) / (len(y_train) + 2.0)
    p = alphas[c] * p_model + (1.0 - alphas[c]) * p0

    p = np.clip(p, eps, 1.0 - eps).astype(np.float64)
    p_c_test.append(p)
    for uid, pv in zip(test_uid_arr, p):
        uid_pred[uid][c] = float(pv)

p_c_test = np.stack(p_c_test, axis=1)  # (n_test, 7)

b_po, w_po = params["patient_overall"]
p_po_model = predict_logistic_1d(x_test_arr, b_po, w_po)

y0 = train_df["patient_overall"].to_numpy(dtype=np.float64)
p0_po = float((y0.sum() + 1.0) / (len(y0) + 2.0))
p_po_direct = (
    alphas["patient_overall"] * p_po_model + (1.0 - alphas["patient_overall"]) * p0_po
)

p_po_any = noisy_or_any(p_c_test, gamma=patient_overall_gamma, eps=eps)
p_po = patient_overall_beta * p_po_any + (1.0 - patient_overall_beta) * p_po_direct
p_po = np.clip(p_po, eps, 1.0 - eps).astype(np.float64)

for uid, pv in zip(test_uid_arr, p_po):
    uid_pred[uid]["patient_overall"] = float(pv)

for uid in test_uid_arr:
    mx = 0.0
    for c in c_cols:
        v = uid_pred[uid][c]
        if v > mx:
            mx = v
    po = uid_pred[uid]["patient_overall"]
    if po < mx:
        uid_pred[uid]["patient_overall"] = float(np.clip(mx, eps, 1.0 - eps))

sub_work = test_df[["row_id", "StudyInstanceUID", "prediction_type"]].copy()

pred_records = []
for uid, dct in uid_pred.items():
    for t, v in dct.items():
        pred_records.append((uid, t, v))
pred_df = pd.DataFrame(
    pred_records, columns=["StudyInstanceUID", "prediction_type", "fractured"]
)

sub_work = sub_work.merge(
    pred_df, on=["StudyInstanceUID", "prediction_type"], how="left"
)

sub = sample_df.merge(sub_work[["row_id", "fractured"]], on="row_id", how="left")

p_med_model = float(predict_logistic_1d(np.array([np.median(x_train)]), b_po, w_po)[0])
fill_val = float(
    np.clip(
        patient_overall_beta * float(np.clip(p_med_model, eps, 1.0 - eps))
        + (1.0 - patient_overall_beta) * float(np.clip(p0_po, eps, 1.0 - eps)),
        eps,
        1.0 - eps,
    )
)

sub["fractured"] = sub["fractured"].fillna(fill_val).astype(np.float32)

assert sub.shape[0] == sample_df.shape[0], "Row count mismatch vs sample_submission"
assert list(sub.columns) == ["row_id", "fractured"], "Submission columns mismatch"
assert sub["fractured"].between(0.0, 1.0).all(), "Probabilities out of range"

sub.to_csv(SAVE_CSV, index=False)
print("Saved submission:", SAVE_CSV)
print(sub.head())
print("Rows:", len(sub))
print(
    "fractured min/mean/max:",
    float(sub["fractured"].min()),
    float(sub["fractured"].mean()),
    float(sub["fractured"].max()),
)
