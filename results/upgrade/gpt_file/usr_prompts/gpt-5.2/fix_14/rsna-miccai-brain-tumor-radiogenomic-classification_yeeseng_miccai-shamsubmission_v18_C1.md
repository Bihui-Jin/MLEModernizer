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

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I remove the dependency on the missing `../input/miccai-testsubmissions/testPredictions_all.csv` file that causes the first runtime error, and instead build predictions directly from the competition’s provided `sample_submission.csv` (which guarantees correct IDs and format). This also fixes the downstream `NameError` by ensuring the prediction dictionary is always defined. Since no model code exists here and the previous intent was to output 0.5 when predictions are unavailable, the score-neutral/stable fix is to submit constant 0.5 probabilities for all test IDs. Finally, I keep paths aligned with Kaggle (`/kaggle/input/...`) and ensure the output is a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.54647) has done: 'Your current score (0.5 AUC) comes from predicting a constant probability for everyone, which is expected to be near-random and far from any meaningful target; your provided target score (-1.0) is not achievable under AUC (AUC is in [0,1]), so I interpret the goal as “increase score” while keeping changes minimal. To improve AUC without changing the overall approach (still a very lightweight, non-deep-learning pipeline), I replace the constant 0.5 with a simple radiomics-like feature: mean pixel intensity from a single middle DICOM slice of the FLAIR series per subject, then fit a logistic regression on train and predict probabilities on test. I also keep the known-bad training cases excluded as suggested by the competition and ensure the submission IDs/ordering exactly match `sample_submission.csv`. This should move the score upward from 0.5 while staying within Kaggle constraints and producing a valid `submission.csv`.'
- What this solution (achieved 0.54353) has done: 'Your target score (-1.0) is not attainable for AUC (valid range is [0, 1]), so the only sensible “toward target” direction under a higher-is-better metric is to increase the AUC from your current 0.54647 with minimal, low-risk changes. I keep your exact pipeline (single handcrafted intensity feature → StandardScaler → LogisticRegression) but make that feature slightly more informative by averaging the mean intensity across all four modalities (FLAIR, T1w, T1wCE, T2w) using the same “middle slice” logic you already use. This preserves the training approach, model family, and semantics while typically boosting separability a bit on this dataset. I also add a tiny safeguard to fall back to FLAIR-only if some modalities are missing for a subject, keeping runtime stable and still producing a valid `submission.csv`.'
- What this solution (achieved 0.54353) has done: 'Your target score of -1.0 is outside the valid AUC range [0, 1], so the only sensible “toward target” direction under a higher-is-better metric is to increase your current 0.54353. To do that with minimal changes and identical core approach (single handcrafted intensity feature → StandardScaler → LogisticRegression), I (1) normalize the raw DICOM pixel arrays using DICOM RescaleSlope/RescaleIntercept when present (a small but often meaningful correctness fix for intensity-based features), and (2) slightly strengthen LogisticRegression regularization settings via `class_weight="balanced"` to better handle class imbalance without changing the model family or training loop. Everything else (feature definition as mean intensity across modalities from the middle slice, missing-value handling, submission formatting) stays the same, and the script still write a valid `submission.csv`.'
- What this solution (achieved 0.54118) has done: 'The timeout is dominated by repeated filesystem listing/sorting and repeated DICOM header parsing done separately for each modality and each subject. I keep the exact same feature definition (mean intensity of the middle slice per modality, averaged across modalities) and the same sklearn pipeline, but (1) avoid reading `InstanceNumber` entirely when selecting the “middle” slice because file names already encode slice order in this dataset, (2) eliminate per-subject repeated directory listings by selecting the middle file via a fast nth-element scan, and (3) add a simple on-disk feature cache so subsequent runs don’t recompute features. These are correctness-preserving because the chosen middle slice remains the same (lexicographic filename order), pixel decoding and clipping are unchanged, and model training/evaluation semantics are identical.'
- What this solution (achieved 0.52118) has done: 'Your target score of -1.0 is impossible for ROC-AUC (valid range is [0, 1]), so with “higher is better” and no current score yielded, the safest way to move score upward is to fix a likely generalization issue without changing the core pipeline. I keep the exact same feature definition (mean intensity of middle slice per modality), preprocessing (median impute + StandardScaler), and model family (LogisticRegression), but adjust only the regularization strength `C` to a slightly stronger value to reduce overfitting from noisy single-slice features. I also add a deterministic sanity-check AUC via a stratified CV estimate (no effect on training) so you can see whether the adjustment is likely to help before submitting. Submission formatting/ID alignment stays identical and still writes a valid `submission.csv`.'
- What this solution (achieved 0.59882) has done: 'Your pipeline doesn’t currently “yield” a score in your environment because it depends on `sklearn`, which is not in the installed-packages list you provided; so the smallest score-improving change is to replace the sklearn LogisticRegression with a tiny pure-NumPy logistic regression trained on the exact same features (same semantics: scaled features + logistic model + predicted probabilities). I keep the feature extraction, caching, NaN handling, and submission formatting identical, and I keep class-imbalance handling by using sample weights (equivalent intent to `class_weight="balanced"`). I also preserve the deterministic behavior (fixed seed) and keep runtime under 600s by using a small fixed-iteration batch gradient descent on just 5 features. This should produce a valid `submission.csv` and typically improve over constant predictions, moving AUC upward (since your target score is invalid for AUC, the only sensible “toward target” direction under “higher is better” is to increase score).'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import os
import io
import struct
from functools import lru_cache

DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
if not os.path.exists(sample_path):
    sample_path = "/kaggle/input/sample_submission.csv"

train_labels_path = os.path.join(DATA_ROOT, "train_labels.csv")
if not os.path.exists(train_labels_path):
    train_labels_path = "/kaggle/input/train_labels.csv"

train_dir = os.path.join(DATA_ROOT, "train")
test_dir = os.path.join(DATA_ROOT, "test")

sample_df = pd.read_csv(sample_path)
assert (
    "BraTS21ID" in sample_df.columns and "MGMT_value" in sample_df.columns
), "Submission columns not found."
sample_df["BraTS21ID"] = sample_df["BraTS21ID"].astype(str).str.zfill(5)

train_labels = pd.read_csv(train_labels_path)
train_labels["BraTS21ID"] = train_labels["BraTS21ID"].astype(str).str.zfill(5)
assert set(["BraTS21ID", "MGMT_value"]).issubset(train_labels.columns)

print("Loaded sample_submission:", sample_df.shape, "train_labels:", train_labels.shape)
print(
    "Train dir exists:",
    os.path.isdir(train_dir),
    "Test dir exists:",
    os.path.isdir(test_dir),
)




## === cell 1
def _read_dicom_pixel_array(path):
    """
    Minimal DICOM pixel reader for common RSNA-MICCAI BraTS DICOMs.
    Supports uncompressed little-endian syntaxes; returns float32 array or raises.

    Keeps prior correctness: applies RescaleSlope/RescaleIntercept when present.
    """

    def _read_exact(f, n):
        b = f.read(n)
        if len(b) != n:
            raise ValueError("Unexpected EOF.")
        return b

    def read_u16(f):
        return struct.unpack("<H", _read_exact(f, 2))[0]

    def read_u32(f):
        return struct.unpack("<I", _read_exact(f, 4))[0]

    def _parse_number(val_bytes):
        s = val_bytes.replace(b"\x00", b"").decode("ascii", errors="ignore").strip()
        if s == "":
            return None
        try:
            return float(s)
        except Exception:
            return None

    with open(path, "rb") as f:
        pre = _read_exact(f, 132)
        if pre[128:132] != b"DICM":
            raise ValueError("Not a standard DICOM file with DICM marker.")

        transfer_syntax = None
        while True:
            hdr = f.read(4)
            if len(hdr) < 4:
                raise ValueError("Unexpected EOF in meta header.")
            group, elem = struct.unpack("<HH", hdr)
            if group != 0x0002:
                f.seek(-4, io.SEEK_CUR)
                break

            vr = _read_exact(f, 2).decode("ascii", errors="ignore")
            if vr in ("OB", "OW", "OF", "SQ", "UT", "UN"):
                _read_exact(f, 2)  # reserved
                length = read_u32(f)
            else:
                length = read_u16(f)

            value = _read_exact(f, length)
            if (group, elem) == (0x0002, 0x0010):
                transfer_syntax = value.rstrip(b"\x00").decode("ascii", errors="ignore")

        if transfer_syntax is None:
            transfer_syntax = "1.2.840.10008.1.2"  # Implicit VR Little Endian

        if transfer_syntax not in (
            "1.2.840.10008.1.2",  # Implicit VR Little Endian
            "1.2.840.10008.1.2.1",  # Explicit VR Little Endian
        ):
            raise ValueError(f"Unsupported TransferSyntaxUID: {transfer_syntax}")

        explicit = transfer_syntax == "1.2.840.10008.1.2.1"

        rows = None
        cols = None
        bits_alloc = None
        pixel_repr = 0
        samples_per_pixel = 1
        rescale_slope = 1.0
        rescale_intercept = 0.0

        while True:
            tag_bytes = f.read(4)
            if len(tag_bytes) < 4:
                break
            group, elem = struct.unpack("<HH", tag_bytes)

            if explicit:
                vr = _read_exact(f, 2).decode("ascii", errors="ignore")
                if vr in ("OB", "OW", "OF", "SQ", "UT", "UN"):
                    _read_exact(f, 2)  # reserved
                    length = read_u32(f)
                else:
                    length = read_u16(f)
            else:
                length = read_u32(f)
                vr = None  # not used

            tag = (group, elem)

            if tag == (0x0028, 0x0010):  # Rows
                v = _read_exact(f, length)
                rows = int.from_bytes(v, "little", signed=False)
            elif tag == (0x0028, 0x0011):  # Columns
                v = _read_exact(f, length)
                cols = int.from_bytes(v, "little", signed=False)
            elif tag == (0x0028, 0x0100):  # BitsAllocated
                v = _read_exact(f, length)
                bits_alloc = int.from_bytes(v, "little", signed=False)
            elif tag == (0x0028, 0x0103):  # PixelRepresentation
                v = _read_exact(f, length)
                pixel_repr = int.from_bytes(v, "little", signed=False)
            elif tag == (0x0028, 0x0002):  # SamplesPerPixel
                v = _read_exact(f, length)
                samples_per_pixel = int.from_bytes(v, "little", signed=False)
            elif tag == (0x0028, 0x1053):  # RescaleSlope
                v = _read_exact(f, length)
                num = _parse_number(v)
                if num is not None and np.isfinite(num) and num != 0:
                    rescale_slope = float(num)
            elif tag == (0x0028, 0x1052):  # RescaleIntercept
                v = _read_exact(f, length)
                num = _parse_number(v)
                if num is not None and np.isfinite(num):
                    rescale_intercept = float(num)
            elif tag == (0x7FE0, 0x0010):  # PixelData
                if rows is None or cols is None or bits_alloc is None:
                    raise ValueError(
                        "Missing Rows/Columns/BitsAllocated before PixelData."
                    )
                if samples_per_pixel != 1:
                    raise ValueError("Only SamplesPerPixel=1 supported.")

                pixel_bytes = _read_exact(f, length)

                if bits_alloc == 16:
                    dtype = np.int16 if pixel_repr == 1 else np.uint16
                elif bits_alloc == 8:
                    dtype = np.int8 if pixel_repr == 1 else np.uint8
                else:
                    raise ValueError(f"Unsupported BitsAllocated: {bits_alloc}")

                arr = np.frombuffer(pixel_bytes, dtype=dtype)
                expected = rows * cols
                if arr.size < expected:
                    raise ValueError("PixelData too short.")
                arr = arr[:expected].reshape(rows, cols).astype(np.float32)

                arr = arr * np.float32(rescale_slope) + np.float32(rescale_intercept)
                return arr
            else:
                if length:
                    f.seek(length, io.SEEK_CUR)

        raise ValueError("PixelData not found.")


def _get_middle_slice_dcm(series_dir):
    try:
        with os.scandir(series_dir) as it:
            names = [e.name for e in it if e.is_file() and e.name.endswith(".dcm")]
    except FileNotFoundError:
        return None

    if not names:
        return None

    names.sort()
    return os.path.join(series_dir, names[len(names) // 2])


def subject_feature_mean_intensity(subject_dir, modality="FLAIR"):
    series_dir = os.path.join(subject_dir, modality)
    dcm_path = _get_middle_slice_dcm(series_dir)
    if dcm_path is None:
        return np.nan
    img = _read_dicom_pixel_array(dcm_path)

    lo, hi = np.percentile(img, [1.0, 99.0])
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        img = np.clip(img, lo, hi)

    return float(np.mean(img))


def subject_feature_mean_intensity_multi_vec(
    subject_dir, modalities=("FLAIR", "T1w", "T1wCE", "T2w")
):
    out = np.empty((len(modalities),), dtype=np.float32)
    for j, m in enumerate(modalities):
        try:
            v = subject_feature_mean_intensity(subject_dir, modality=m)
        except Exception:
            v = np.nan
        out[j] = v if np.isfinite(v) else np.nan
    return out




## === cell 2
bad_ids = set(["00109", "00123", "00709"])


def _list_subject_ids(folder):
    if not os.path.isdir(folder):
        return []
    ids = []
    with os.scandir(folder) as it:
        for e in it:
            if e.is_dir():
                ids.append(str(e.name).zfill(5))
    return sorted(ids)


train_ids = _list_subject_ids(train_dir)

test_ids = sample_df["BraTS21ID"].tolist()

train_ids = [i for i in train_ids if i not in bad_ids]
train_labels_use = train_labels[train_labels["BraTS21ID"].isin(train_ids)].copy()

train_ids = train_labels_use["BraTS21ID"].tolist()

print(
    "Train subjects (after excluding bad):",
    len(train_ids),
    "Test subjects:",
    len(test_ids),
)

FEATURE_CACHE_PATH = "/kaggle/working/feature_cache_mean_intensity_multi_vec_v2.csv"

_cache = None
if os.path.exists(FEATURE_CACHE_PATH):
    try:
        _cache_df = pd.read_csv(FEATURE_CACHE_PATH)
        needed = {"key", "m0", "m1", "m2", "m3"}
        if needed.issubset(_cache_df.columns):
            _cache = {}
            for _, r in _cache_df.iterrows():
                _cache[str(r["key"])] = np.array(
                    [r["m0"], r["m1"], r["m2"], r["m3"]], dtype=np.float32
                )
        else:
            _cache = {}
    except Exception:
        _cache = {}
else:
    _cache = {}


def _cache_key(split, sid):
    return f"{split}:{sid}"


def _get_or_compute_feature_vec(
    split, sid, base_dir, modalities=("FLAIR", "T1w", "T1wCE", "T2w")
):
    k = _cache_key(split, sid)
    if k in _cache:
        v = _cache[k]
        if isinstance(v, np.ndarray) and v.shape == (4,):
            return v.astype(np.float32, copy=False)
    subj_dir = os.path.join(base_dir, sid)
    try:
        v = subject_feature_mean_intensity_multi_vec(subj_dir, modalities=modalities)
    except Exception:
        v = np.full((4,), np.nan, dtype=np.float32)
    _cache[k] = v.astype(np.float32, copy=False)
    return _cache[k]


modalities = ("FLAIR", "T1w", "T1wCE", "T2w")

X_train_base = np.empty((len(train_ids), len(modalities)), dtype=np.float32)
y_train = train_labels_use["MGMT_value"].astype(int).to_numpy()

for idx, sid in enumerate(train_ids):
    X_train_base[idx, :] = _get_or_compute_feature_vec(
        "train", sid, train_dir, modalities=modalities
    )

X_test_base = np.empty((len(test_ids), len(modalities)), dtype=np.float32)
for idx, sid in enumerate(test_ids):
    X_test_base[idx, :] = _get_or_compute_feature_vec(
        "test", sid, test_dir, modalities=modalities
    )

try:
    keys = list(_cache.keys())
    arr = np.vstack([_cache[k].reshape(1, -1) for k in keys]).astype(np.float32)
    _df_cache = pd.DataFrame(
        {
            "key": keys,
            "m0": arr[:, 0],
            "m1": arr[:, 1],
            "m2": arr[:, 2],
            "m3": arr[:, 3],
        }
    )
    _df_cache.to_csv(FEATURE_CACHE_PATH, index=False)
except Exception:
    pass

X_train = np.concatenate(
    [X_train_base, np.nanmean(X_train_base, axis=1, keepdims=True)], axis=1
).astype(np.float32)
X_test = np.concatenate(
    [X_test_base, np.nanmean(X_test_base, axis=1, keepdims=True)], axis=1
).astype(np.float32)

med = np.nanmedian(X_train, axis=0)
med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)

for j in range(X_train.shape[1]):
    X_train[:, j] = np.where(np.isfinite(X_train[:, j]), X_train[:, j], med[j])
    X_test[:, j] = np.where(np.isfinite(X_test[:, j]), X_test[:, j], med[j])

print("Feature stats - train mean/std:", float(X_train.mean()), float(X_train.std()))
print("Feature stats - test  mean/std:", float(X_test.mean()), float(X_test.std()))




## === cell 3
def _standardize_fit_transform(X):
    mu = X.mean(axis=0, keepdims=True).astype(np.float32)
    sd = X.std(axis=0, keepdims=True).astype(np.float32)
    sd = np.where(sd > 0, sd, 1.0).astype(np.float32)
    return (X - mu) / sd, mu, sd


def _standardize_transform(X, mu, sd):
    return (X - mu) / sd


def _sigmoid(z):
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logreg_l2_balanced(X, y, l2=1.0, lr=0.1, n_iter=400, seed=0):
    """
    Simple batch GD logistic regression with L2 penalty on weights (not intercept).
    Uses balanced sample weights (same intent as class_weight="balanced").
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = rng.normal(0, 0.01, size=(d,)).astype(np.float32)
    b = np.float32(0.0)

    y = y.astype(np.float32)
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    w_pos = (n / 2.0) / max(pos, 1.0)
    w_neg = (n / 2.0) / max(neg, 1.0)
    sw = np.where(y > 0.5, w_pos, w_neg).astype(np.float32)

    for _ in range(int(n_iter)):
        z = X @ w + b
        p = _sigmoid(z).astype(np.float32)

        diff = (p - y) * sw
        grad_w = (X.T @ diff) / n + l2 * w
        grad_b = float(diff.mean())

        w = (w - lr * grad_w).astype(np.float32)
        b = np.float32(b - lr * grad_b)

    return w, b


def predict_proba_logreg(X, w, b):
    return _sigmoid(X @ w + b).astype(np.float32)


def roc_auc_score_numpy(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)
    order = np.argsort(y_score)
    y_true = y_true[order]

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan

    ranks = np.empty_like(y_score, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j

    sum_ranks_pos = float(ranks[y_true == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _stratified_kfold_indices(y, n_splits=5, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    folds = [[] for _ in range(n_splits)]
    for i, ix in enumerate(pos_idx):
        folds[i % n_splits].append(ix)
    for i, ix in enumerate(neg_idx):
        folds[i % n_splits].append(ix)

    out = []
    for k in range(n_splits):
        val_idx = np.array(sorted(folds[k]), dtype=int)
        train_idx = np.array(
            sorted(np.setdiff1d(np.arange(len(y)), val_idx)), dtype=int
        )
        out.append((train_idx, val_idx))
    return out


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _fit_temperature_on_oof(
    y, p, temps=np.array([0.7, 0.85, 1.0, 1.2, 1.4], dtype=np.float32)
):
    y = y.astype(np.float32)
    z = _logit(p).astype(np.float32)

    def logloss(y_, p_):
        p_ = np.clip(p_, 1e-6, 1 - 1e-6)
        return float(-np.mean(y_ * np.log(p_) + (1 - y_) * np.log(1 - p_)))

    best_t = 1.0
    best_ll = 1e9
    for t in temps:
        p_cal = _sigmoid(z / float(t)).astype(np.float32)
        ll = logloss(y, p_cal)
        if ll < best_ll:
            best_ll = ll
            best_t = float(t)
    return best_t, best_ll


X_train_s, mu, sd = _standardize_fit_transform(X_train.astype(np.float32))
X_test_s = _standardize_transform(X_test.astype(np.float32), mu, sd)

X_train_s = np.clip(X_train_s, -5.0, 5.0).astype(np.float32)
X_test_s = np.clip(X_test_s, -5.0, 5.0).astype(np.float32)

l2_scaled = (1.0 / max(len(X_train_s), 1)) * 526.0  # keep prior magnitude

splits = _stratified_kfold_indices(y_train, n_splits=5, seed=0)
oof = np.zeros((len(y_train),), dtype=np.float32)
for tr_idx, va_idx in splits:
    w_fold, b_fold = fit_logreg_l2_balanced(
        X_train_s[tr_idx],
        y_train[tr_idx],
        l2=float(l2_scaled),
        lr=0.15,
        n_iter=450,
        seed=0,
    )
    oof[va_idx] = predict_proba_logreg(X_train_s[va_idx], w_fold, b_fold)

temp, oof_ll = _fit_temperature_on_oof(y_train.astype(np.float32), oof)
oof_auc = roc_auc_score_numpy(y_train, oof)
print("OOF AUC (sanity):", oof_auc, "OOF logloss:", oof_ll, "Chosen temperature:", temp)

w_lr, b_lr = fit_logreg_l2_balanced(
    X_train_s, y_train, l2=float(l2_scaled), lr=0.15, n_iter=450, seed=0
)

test_proba_raw = predict_proba_logreg(X_test_s, w_lr, b_lr)
test_proba = _sigmoid(_logit(test_proba_raw) / float(temp)).astype(np.float32)
test_proba = np.clip(test_proba, 1e-6, 1 - 1e-6)

print(
    "Pred proba summary:",
    float(test_proba.min()),
    float(test_proba.mean()),
    float(test_proba.max()),
)

train_pred = predict_proba_logreg(X_train_s, w_lr, b_lr)
train_auc = roc_auc_score_numpy(y_train, train_pred)
print("Train AUC (in-sample, for sanity only):", train_auc)



## === cell 4
pred_map = {sid: float(p) for sid, p in zip(test_ids, test_proba)}

sub = sample_df.copy()
sub["MGMT_value"] = sub["BraTS21ID"].map(pred_map).astype(float)
sub["MGMT_value"] = sub["MGMT_value"].fillna(0.5).astype(float)
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())



## === cell 5
assert sub.columns.tolist() == ["BraTS21ID", "MGMT_value"]
assert len(sub) == len(sample_df)
assert sub["MGMT_value"].between(0.0, 1.0).all()

print("Unique IDs in submission:", sub["BraTS21ID"].nunique())
print(
    "Any missing preds filled with 0.5:",
    int(sub["BraTS21ID"].isin(pred_map.keys()).sum()),
    "/",
    len(sub),
)
