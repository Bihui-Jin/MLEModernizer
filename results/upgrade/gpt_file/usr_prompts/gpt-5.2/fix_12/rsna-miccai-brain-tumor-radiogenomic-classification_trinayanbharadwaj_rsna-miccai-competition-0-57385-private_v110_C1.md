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

0.64059

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.55176) has done: 'I remove/guard the imports that trigger the protobuf `MessageFactory.GetPrototype` crash and keep only the packages actually needed for loading DICOMs and running inference. Because the referenced pretrained `.h5` models are not available in your environment, I minimally replace that missing step by training the same kind of lightweight Keras CNN directly on the provided `train/` DICOMs (using the same “pick up to 6 informative slices” logic), then run predictions on `test/` and write `submission.csv`. I also fix logic bugs in `create_sub` (it was producing a single scalar prediction for all cases) and ensure IDs are formatted exactly as required. The result run end-to-end within the Kaggle environment and produce a valid submission CSV.'
- What this solution (achieved 0.68294) has done: 'The timeout is dominated by repeatedly scanning directories, repeatedly parsing DICOM headers, and re-loading slices multiple times across train/val/full-train/test passes. I keep the exact same slice-selection heuristics, feature extraction, and logistic-regression training, but add a disk-backed cache for extracted subject features so each subject is decoded only once. I also avoid building large argument lists for multiprocessing, use `os.listdir`/pre-filtering to reduce `scandir` overhead, and tune pool `chunksize` for fewer IPC roundtrips while preserving determinism. These changes are equivalent (they reuse identical computed features) but remove redundant work that pushes runtime past 600s.'
- What this solution (achieved 0.68294) has done: 'You’re already above the (invalid) target_score of -1.0, so any “improvement toward target” would mean lowering performance, which is not meaningful here; instead I make only stability/correctness tweaks that should preserve (or slightly improve) your current score without changing the core feature/model logic. Specifically, I (1) make T2w selection explicit instead of relying on a sorted-folder index (prevents occasional modality mismatch), (2) ensure the fast header/pixel-sum filter never rejects valid slices due to partial PixelData reads, and (3) keep caching semantics identical while making train/test split detection robust for nested paths. These are minimal changes focused on preventing silent data issues that can hurt AUC while keeping the same logistic-regression training and feature extraction.'
- What this solution (achieved 0.68294) has done: 'Your current score (0.68294 AUC) is already well above the provided target_score (-1.0), so changing the model to “move toward target” would mean intentionally degrading performance, which is not meaningful. Instead, I make only minimal stability/correctness tweaks that should preserve (or slightly improve) AUC by preventing silent feature mismatches: (1) make the multiprocessing start method portable by falling back from `fork` to `spawn` if needed, (2) ensure cached feature files are invalidated if the feature dimension ever changes (prevents mixing old/bad cache with new runs), and (3) harden T2w modality selection to prefer exact `T2w` but also accept common casing/spacing variants. These changes keep the same feature extraction, slice heuristics, and logistic regression training while reducing the chance of corrupted/empty features that can lower AUC.'
- What this solution (achieved 0.64882) has done: 'Your current score (0.68294 AUC) is already far above the provided target_score (-1.0), so any change that “moves toward target” would technically mean intentionally degrading performance; instead, I keep the exact same feature extraction and logistic-regression training, and make only minimal correctness/stability fixes that typically preserve or slightly improve AUC by avoiding silent data mixups. Concretely, I (1) ensure we always train on the full available training set (remove the random cap that can unnecessarily hurt score), (2) make modality selection prefer exact `T2w/` but also fall back robustly (already mostly done; we just make it deterministic if multiple candidates exist), and (3) harden DICOM loading to use `pydicom.dcmread(..., force=True, stop_before_pixels=True)` for header scan where available to reduce occasional header parse failures that can lead to empty/zero features. These are minimal changes that keep the same pipeline and semantics while reducing avoidable feature noise. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.64529) has done: 'Your current AUC (0.64882) is already far above the provided target_score (-1.0), so changing the model to “move toward target” would technically mean intentionally degrading performance; instead, I keep the exact same feature extraction and logistic-regression training, and make only minimal correctness/stability fixes that tend to preserve or slightly improve AUC by preventing silent data issues. Concretely, I (1) make DICOM slice ordering deterministic by sorting using `InstanceNumber` when available (fallback to filename), so the “first 6 informative slices” heuristic is applied consistently across subjects, and (2) make the fast pixel-sum header scan robust to VR `UN`/undefined lengths by simply skipping the fast-sum shortcut when the header can’t reliably provide offsets (we still fully decode pixels as before). These changes don’t alter the model/metric semantics, they just reduce noisy slice selection and accidental filtering that can hurt AUC. The script still runs end-to-end within the same paths and writes a valid `submission.csv`.'
- What this solution (achieved 0.64059) has done: 'Your current AUC (0.64529) is already far above the provided target_score (-1.0), so any true “move toward target” would mean intentionally degrading performance; instead, the safest way to avoid accidental score loss is to make the pipeline more deterministic and prevent rare empty-feature cases. I (1) make feature extraction robust by explicitly handling NaNs/Infs before computing percentiles/statistics (so one bad slice can’t poison a subject’s features), and (2) make the training split stratified (same train/val sizes, but avoids occasional label-imbalance in the validation fold that can mislead you into choosing worse hyperparameters later). These changes preserve the exact model (same handcrafted features + L2 logistic regression), training loop, and prediction semantics, and they still write a valid `submission.csv` end-to-end.'
- What this solution (achieved 0.64059) has done: 'Your current AUC (0.64059) is already far above the provided target_score (-1.0), so any change that truly “moves toward target” would mean intentionally degrading performance; instead, the smallest safe way to avoid further score loss is to fix two correctness issues that can silently hurt AUC without changing the model/feature core logic. I (1) make the DICOM fallback reader handle implicit VR little endian (many MR DICOMs are implicit VR), so we don’t drop slices/subjects when pydicom fails, and (2) make modality selection prefer an exact `T2w` folder name (then fall back) to avoid rare modality mismatches. These changes preserve the same slice-selection heuristic (“first 6 informative slices”), the same handcrafted 13-dim features, and the same L2 logistic regression training/prediction. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.64059) has done: 'I make two minimal, score-relevant fixes that keep your exact feature extraction and logistic-regression core logic unchanged while reducing silent data issues that can hurt AUC. First, I make `_read_dicom_header_minimal` correctly handle implicit-VR DICOMs in its manual parser (it currently assumes explicit VR and can mis-parse headers), which improves the reliability of the “fast pixel-sum skip” and reduces accidental slice rejection. Second, I simplify the fast-sum gating so it’s only applied when the header parse is trustworthy; otherwise we fall back to the existing full pixel decode (same as you already do) rather than incorrectly skipping a slice. These changes are small, deterministic, and should nudge the AUC upward (or at least prevent avoidable drops) without changing the model/training semantics, and the script still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

try:
    import pydicom as dicom  # type: ignore
except Exception:
    dicom = None

try:
    from skimage.transform import resize  # type: ignore
except Exception:
    resize = None

import struct
import multiprocessing as mp


def _read_dicom_pixels_fallback(path):
    """
    Minimal DICOM reader for uncompressed 2D images.
    Supports Explicit VR Little Endian and (common) Implicit VR Little Endian.
    Returns float32 2D array or None on failure.
    """
    try:
        with open(path, "rb") as f:
            data = f.read()
        if len(data) < 132 or data[128:132] != b"DICM":
            return None

        def u16(off):
            return struct.unpack_from("<H", data, off)[0]

        def u32(off):
            return struct.unpack_from("<I", data, off)[0]

        def is_likely_implicit_vr(offset):
            if offset + 6 >= len(data):
                return True
            vr = data[offset + 4 : offset + 6]
            return not (65 <= vr[0] <= 90 and 65 <= vr[1] <= 90)

        implicit_vr = is_likely_implicit_vr(132)

        off = 132
        rows = cols = None
        bits_alloc = None
        pixel_repr = 0
        samples_per_pixel = 1
        photometric = None
        pixel_data = None

        while off + 8 <= len(data):
            group = u16(off)
            elem = u16(off + 2)

            if implicit_vr:
                off_tag = off
                off += 4
                if off + 4 > len(data):
                    break
                length = u32(off)
                off += 4
            else:
                vr = data[off + 4 : off + 6]
                off_tag = off
                off += 6
                if vr in (b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN"):
                    off += 2  # reserved
                    if off + 4 > len(data):
                        break
                    length = u32(off)
                    off += 4
                else:
                    if off + 2 > len(data):
                        break
                    length = u16(off)
                    off += 2

            value = data[off : off + length] if off + length <= len(data) else None

            if (group, elem) == (0x0028, 0x0010) and value is not None:  # Rows
                if length == 2:
                    rows = u16(off)
                else:
                    try:
                        rows = int(value.rstrip(b"\x00 ").decode(errors="ignore"))
                    except Exception:
                        pass
            elif (group, elem) == (0x0028, 0x0011) and value is not None:  # Columns
                if length == 2:
                    cols = u16(off)
                else:
                    try:
                        cols = int(value.rstrip(b"\x00 ").decode(errors="ignore"))
                    except Exception:
                        pass
            elif (group, elem) == (
                0x0028,
                0x0100,
            ) and value is not None:  # BitsAllocated
                if length == 2:
                    bits_alloc = u16(off)
                else:
                    try:
                        bits_alloc = int(value.rstrip(b"\x00 ").decode(errors="ignore"))
                    except Exception:
                        pass
            elif (group, elem) == (
                0x0028,
                0x0103,
            ) and value is not None:  # PixelRepresentation
                if length == 2:
                    pixel_repr = u16(off)
                else:
                    try:
                        pixel_repr = int(value.rstrip(b"\x00 ").decode(errors="ignore"))
                    except Exception:
                        pass
            elif (group, elem) == (
                0x0028,
                0x0002,
            ) and value is not None:  # SamplesPerPixel
                if length == 2:
                    samples_per_pixel = u16(off)
                else:
                    try:
                        samples_per_pixel = int(
                            value.rstrip(b"\x00 ").decode(errors="ignore")
                        )
                    except Exception:
                        pass
            elif (group, elem) == (
                0x0028,
                0x0004,
            ) and value is not None:  # PhotometricInterpretation
                try:
                    photometric = value.rstrip(b"\x00 ").decode(errors="ignore")
                except Exception:
                    photometric = None
            elif (group, elem) == (0x7FE0, 0x0010):  # PixelData
                if value is None:
                    break
                pixel_data = value
                break

            off += length
            if off <= off_tag:
                break

        if pixel_data is None or rows is None or cols is None or bits_alloc is None:
            return None
        if samples_per_pixel != 1:
            return None
        if bits_alloc not in (8, 16):
            return None

        if bits_alloc == 8:
            dt = np.int8 if pixel_repr == 1 else np.uint8
        else:
            dt = np.int16 if pixel_repr == 1 else np.uint16

        arr = np.frombuffer(pixel_data, dtype=dt)
        if arr.size < rows * cols:
            return None
        arr = arr[: rows * cols].reshape((rows, cols)).astype(np.float32)

        if photometric is not None and photometric.upper() == "MONOCHROME1":
            arr = np.max(arr) - arr

        return arr
    except Exception:
        return None




## === cell 1
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.exists(TRAIN_DIR), f"Missing: {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"Missing: {TEST_DIR}"
assert os.path.exists(LABELS_CSV), f"Missing: {LABELS_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"

labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
labels_df = labels_df.set_index("BraTS21ID")

sample_df = pd.read_csv(SAMPLE_SUB)
sample_df["BraTS21ID"] = sample_df["BraTS21ID"].astype(str).str.zfill(5)

print("Train labels:", labels_df.shape, "Test sample:", sample_df.shape)



## === cell 2
MRI_TYPE_INDEX_T2W = 3  # backwards compatibility fallback
IMG_PX_SIZE = 150
SLICES_PER_CASE = 6

BAD_TRAIN_IDS = set(["00109", "00123", "00709"])  # as per competition note

FEATURE_CACHE_DIR = "/kaggle/working/feature_cache_v1"
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)


def _feature_cache_path(base_dir, sid):
    base_norm = os.path.normpath(base_dir)
    tail = os.path.basename(base_norm)
    split = (
        "train"
        if tail.lower() == "train"
        else ("test" if tail.lower() == "test" else tail.lower())
    )
    return os.path.join(
        FEATURE_CACHE_DIR,
        f"{split}_{sid}_t2w_{IMG_PX_SIZE}_{SLICES_PER_CASE}.npy",
    )


def _resize2d(arr, out_hw):
    h, w = arr.shape
    oh, ow = out_hw
    if h == oh and w == ow:
        return arr.astype(np.float32, copy=False)
    y_idx = (np.linspace(0, h - 1, oh)).astype(np.int32)
    x_idx = (np.linspace(0, w - 1, ow)).astype(np.int32)
    return arr[np.ix_(y_idx, x_idx)].astype(np.float32, copy=False)


def _read_dicom_header_minimal(path, max_bytes=512 * 1024):
    if dicom is not None:
        try:
            d = dicom.dcmread(path, stop_before_pixels=True, force=True)
            rows = int(getattr(d, "Rows", 0) or 0)
            cols = int(getattr(d, "Columns", 0) or 0)
            bits_alloc = int(getattr(d, "BitsAllocated", 0) or 0)
            pixel_repr = int(getattr(d, "PixelRepresentation", 0) or 0)
            samples_per_pixel = int(getattr(d, "SamplesPerPixel", 1) or 1)
            photometric = str(getattr(d, "PhotometricInterpretation", "") or "")
            if rows > 0 and cols > 0 and bits_alloc in (8, 16):
                return {
                    "rows": rows,
                    "cols": cols,
                    "bits_alloc": bits_alloc,
                    "pixel_repr": pixel_repr,
                    "samples_per_pixel": samples_per_pixel,
                    "photometric": photometric,
                    "pixel_data_offset": None,
                    "pixel_data_len": None,
                    "implicit_vr": None,
                }
        except Exception:
            pass

    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes)
        if len(data) < 132 or data[128:132] != b"DICM":
            return None

        def u16(off):
            return struct.unpack_from("<H", data, off)[0]

        def u32(off):
            return struct.unpack_from("<I", data, off)[0]

        def is_likely_implicit_vr(offset=132):
            if offset + 6 >= len(data):
                return True
            vr = data[offset + 4 : offset + 6]
            return not (65 <= vr[0] <= 90 and 65 <= vr[1] <= 90)

        implicit_vr = is_likely_implicit_vr(132)

        off = 132
        rows = cols = None
        bits_alloc = None
        pixel_repr = 0
        samples_per_pixel = 1
        photometric = None
        pixel_data_offset = None
        pixel_data_len = None

        while off + 8 <= len(data):
            group = u16(off)
            elem = u16(off + 2)
            off_tag = off

            if implicit_vr:
                off += 4
                if off + 4 > len(data):
                    break
                length = u32(off)
                off += 4
            else:
                vr = data[off + 4 : off + 6]
                off += 6
                if vr in (b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN"):
                    off += 2
                    if off + 4 > len(data):
                        break
                    length = u32(off)
                    off += 4
                else:
                    if off + 2 > len(data):
                        break
                    length = u16(off)
                    off += 2

            value_off = off
            value_end = off + int(length)

            if value_end > len(data):
                break

            if (group, elem) == (0x0028, 0x0010):  # Rows
                if length == 2:
                    rows = u16(value_off)
            elif (group, elem) == (0x0028, 0x0011):  # Columns
                if length == 2:
                    cols = u16(value_off)
            elif (group, elem) == (0x0028, 0x0100):  # BitsAllocated
                if length == 2:
                    bits_alloc = u16(value_off)
            elif (group, elem) == (0x0028, 0x0103):  # PixelRepresentation
                if length == 2:
                    pixel_repr = u16(value_off)
            elif (group, elem) == (0x0028, 0x0002):  # SamplesPerPixel
                if length == 2:
                    samples_per_pixel = u16(value_off)
            elif (group, elem) == (0x0028, 0x0004):  # PhotometricInterpretation
                if length > 0:
                    photometric = (
                        data[value_off:value_end]
                        .rstrip(b"\x00 ")
                        .decode(errors="ignore")
                    )
            elif (group, elem) == (0x7FE0, 0x0010):  # PixelData
                if length != 0xFFFFFFFF:
                    pixel_data_offset = value_off
                    pixel_data_len = int(length)
                break

            off = value_end
            if off <= off_tag:
                break

        if rows is None or cols is None or bits_alloc is None:
            return None

        return {
            "rows": rows,
            "cols": cols,
            "bits_alloc": bits_alloc,
            "pixel_repr": pixel_repr,
            "samples_per_pixel": samples_per_pixel,
            "photometric": photometric,
            "pixel_data_offset": pixel_data_offset,
            "pixel_data_len": pixel_data_len,
            "implicit_vr": implicit_vr,
        }
    except Exception:
        return None


def _dicom_pixel_sum_fast(path, hdr):
    try:
        if (
            hdr is None
            or hdr.get("pixel_data_offset") is None
            or hdr.get("pixel_data_len") is None
        ):
            return None
        if hdr["samples_per_pixel"] != 1:
            return None
        bits_alloc = hdr["bits_alloc"]
        if bits_alloc not in (8, 16):
            return None
        pixel_repr = hdr["pixel_repr"]
        if bits_alloc == 8:
            dt = np.int8 if pixel_repr == 1 else np.uint8
        else:
            dt = np.int16 if pixel_repr == 1 else np.uint16

        n = int(hdr["rows"] * hdr["cols"])
        byte_len = n * (1 if bits_alloc == 8 else 2)
        if int(hdr["pixel_data_len"]) < byte_len:
            return None
        with open(path, "rb") as f:
            f.seek(int(hdr["pixel_data_offset"]))
            raw = f.read(byte_len)
        if len(raw) != byte_len:
            return None

        arr = np.frombuffer(raw, dtype=dt, count=n)
        if arr.size != n:
            return None
        return float(arr.sum(dtype=np.int64))
    except Exception:
        return None


def _safe_read_dcm(path):
    if dicom is not None:
        try:
            d = dicom.dcmread(path, force=True)
            arr = d.pixel_array.astype(np.float32)
            return arr
        except Exception:
            pass
    return _read_dicom_pixels_fallback(path)


def _dcm_sort_key(path):
    if dicom is not None:
        try:
            d = dicom.dcmread(path, stop_before_pixels=True, force=True)
            inst = getattr(d, "InstanceNumber", None)
            if inst is not None:
                return (0, int(inst))
        except Exception:
            pass
    return (1, os.path.basename(path))


def load_case_slices(
    case_dir,
    mri_type_index=MRI_TYPE_INDEX_T2W,
    img_px_size=IMG_PX_SIZE,
    slices_per_case=SLICES_PER_CASE,
):
    """
    Returns: np.ndarray shape (slices_per_case, img_px_size, img_px_size) normalized to [0,1]
    """
    try:
        mri_types = sorted(
            [
                os.path.join(case_dir, name)
                for name in os.listdir(case_dir)
                if os.path.isdir(os.path.join(case_dir, name))
            ]
        )
    except Exception:
        return None

    exact_t2w = [p for p in mri_types if os.path.basename(p) == "T2w"]
    if exact_t2w:
        mri_path = exact_t2w[0]
    else:

        def _norm_mod_name(p):
            return os.path.basename(p).strip().lower().replace(" ", "").replace("_", "")

        t2_candidates = sorted(
            [p for p in mri_types if _norm_mod_name(p) in ("t2w", "t2")]
        )
        if t2_candidates:
            mri_path = t2_candidates[0]
        else:
            if len(mri_types) <= mri_type_index:
                return None
            mri_path = mri_types[mri_type_index]

    try:
        dcm_paths = [
            os.path.join(mri_path, fn)
            for fn in os.listdir(mri_path)
            if fn.lower().endswith(".dcm")
            and os.path.isfile(os.path.join(mri_path, fn))
        ]
        dcm_paths = sorted(dcm_paths, key=_dcm_sort_key)
    except Exception:
        return None

    selected = []
    for p in dcm_paths:
        hdr = _read_dicom_header_minimal(p)

        fast_sum = None
        if (
            hdr is not None
            and hdr.get("pixel_data_offset") is not None
            and hdr.get("pixel_data_len") is not None
        ):
            fast_sum = _dicom_pixel_sum_fast(p, hdr)
        if fast_sum is not None and fast_sum <= 100000:
            continue

        arr = _safe_read_dcm(p)
        if arr is None:
            continue

        if float(np.nansum(arr)) > 100000:
            r = _resize2d(arr, (img_px_size, img_px_size))
            mx = float(np.nanmax(r)) if np.isfinite(r).any() else 0.0
            if mx > 0:
                r = r / mx
            if float(np.nansum(r)) > 2000:
                selected.append(r.astype(np.float32, copy=False))
                if len(selected) >= slices_per_case:
                    break

    if len(selected) == 0:
        selected = [np.zeros((img_px_size, img_px_size), dtype=np.float32)]

    while len(selected) < slices_per_case:
        selected.append(selected[-1].copy())

    return np.stack(selected[:slices_per_case], axis=0)


def _percentiles_10_50_90(x_flat):
    x_flat = np.asarray(x_flat, dtype=np.float32)
    if not np.isfinite(x_flat).all():
        x_flat = np.nan_to_num(x_flat, nan=0.0, posinf=0.0, neginf=0.0)
    return np.percentile(x_flat, [10, 50, 90]).astype(np.float32)


def extract_slice_features(slice2d):
    x = slice2d.astype(np.float32, copy=False)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_flat = x.ravel()

    mean = float(np.mean(x_flat))
    std = float(np.std(x_flat))
    q10, q50, q90 = _percentiles_10_50_90(x_flat)
    mx = float(np.max(x_flat))
    mn = float(np.min(x_flat))

    h, w = x.shape
    ch0, ch1 = int(0.25 * h), int(0.75 * h)
    cw0, cw1 = int(0.25 * w), int(0.75 * w)
    center = x[ch0:ch1, cw0:cw1]
    cmean = float(np.mean(center))
    cstd = float(np.std(center))

    top = x[:ch0, :]
    bottom = x[ch1:, :]
    left = x[ch0:ch1, :cw0]
    right = x[ch0:ch1, cw1:]
    border_sum = float(top.sum() + bottom.sum() + left.sum() + right.sum())
    border_n = float(top.size + bottom.size + left.size + right.size)
    bmean = border_sum / border_n if border_n > 0 else 0.0

    border = np.concatenate([top.ravel(), bottom.ravel(), left.ravel(), right.ravel()])
    bstd = float(np.std(border))

    hi = float(np.mean(x_flat > 0.8))
    mid = float(np.mean(x_flat > 0.5))

    feats = np.array(
        [mean, std, q10, q50, q90, mn, mx, cmean, bmean, cstd, bstd, hi, mid],
        dtype=np.float32,
    )
    return feats


def extract_subject_features(vol_slices):
    feats = np.stack(
        [extract_slice_features(vol_slices[i]) for i in range(vol_slices.shape[0])],
        axis=0,
    )
    return feats.mean(axis=0).astype(np.float32)


def _load_one_subject(args):
    sid, base_dir, has_labels, labels_map = args
    case_dir = os.path.join(base_dir, sid)
    if not os.path.isdir(case_dir):
        return None

    cache_path = _feature_cache_path(base_dir, sid)
    x = None
    if os.path.exists(cache_path):
        try:
            x = np.load(cache_path).astype(np.float32, copy=False)
            if x.ndim != 1 or x.shape[0] != 13:
                x = None
        except Exception:
            x = None

    if x is None:
        vol = load_case_slices(case_dir)
        if vol is None:
            return None
        x = extract_subject_features(vol)
        try:
            np.save(cache_path, x.astype(np.float32, copy=False))
        except Exception:
            pass

    if has_labels:
        y = float(labels_map[sid])
        return sid, x, y
    else:
        return sid, x, None


def load_subject_features(subject_ids, base_dir, labels_indexed=None):
    X_list, y_list, kept_ids = [], [], []

    has_labels = labels_indexed is not None
    labels_map = None
    if has_labels:
        labels_map = labels_indexed["MGMT_value"].to_dict()

    n_workers = min(4, max(1, (os.cpu_count() or 2) // 2))
    if len(subject_ids) < 8 or n_workers == 1:
        for sid in subject_ids:
            out = _load_one_subject((sid, base_dir, has_labels, labels_map))
            if out is None:
                continue
            sid_o, x, y = out
            kept_ids.append(sid_o)
            X_list.append(x)
            if has_labels:
                y_list.append(y)
    else:
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers, maxtasksperchild=32) as pool:
            it = pool.imap(
                _load_one_subject,
                ((sid, base_dir, has_labels, labels_map) for sid in subject_ids),
                chunksize=8,
            )
            for out in it:
                if out is None:
                    continue
                sid_o, x, y = out
                kept_ids.append(sid_o)
                X_list.append(x)
                if has_labels:
                    y_list.append(y)

    X = np.stack(X_list, axis=0) if len(X_list) else np.empty((0, 13), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32) if has_labels else None
    return X, y, kept_ids




## === cell 3
all_train_ids = sorted(
    [
        name
        for name in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, name))
    ]
)
all_train_ids = [sid for sid in all_train_ids if sid not in BAD_TRAIN_IDS]
all_train_ids = [sid for sid in all_train_ids if sid in labels_df.index]

MAX_TRAIN_SUBJECTS = None
if MAX_TRAIN_SUBJECTS is not None and len(all_train_ids) > MAX_TRAIN_SUBJECTS:
    rng = np.random.default_rng(SEED)
    all_train_ids = sorted(
        rng.choice(all_train_ids, size=MAX_TRAIN_SUBJECTS, replace=False).tolist()
    )

rng = np.random.default_rng(SEED)
y_all = labels_df.loc[all_train_ids, "MGMT_value"].astype(int).values
pos_ids = [sid for sid, y in zip(all_train_ids, y_all) if y == 1]
neg_ids = [sid for sid, y in zip(all_train_ids, y_all) if y == 0]
rng.shuffle(pos_ids)
rng.shuffle(neg_ids)

split_pos = int(0.85 * len(pos_ids))
split_neg = int(0.85 * len(neg_ids))
train_ids = pos_ids[:split_pos] + neg_ids[:split_neg]
val_ids = pos_ids[split_pos:] + neg_ids[split_neg:]
rng.shuffle(train_ids)
rng.shuffle(val_ids)

print("Subjects train/val:", len(train_ids), len(val_ids))

X_train, y_train, kept_train_ids = load_subject_features(
    train_ids, TRAIN_DIR, labels_df
)
X_val, y_val, kept_val_ids = load_subject_features(val_ids, TRAIN_DIR, labels_df)

print("Loaded X_train:", X_train.shape, "y_train:", y_train.shape)
print("Loaded X_val:", X_val.shape, "y_val:", y_val.shape)




## === cell 4
def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def standardize_fit(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    return mu.astype(np.float32), sigma.astype(np.float32)


def standardize_apply(X, mu, sigma):
    return ((X - mu) / sigma).astype(np.float32)


def train_logreg_l2(X, y, lr=0.1, steps=800, l2=1.0):
    N, D = X.shape
    w = np.zeros((D,), dtype=np.float32)
    b = np.float32(0.0)

    for _ in range(steps):
        z = X @ w + b
        p = sigmoid(z).astype(np.float32)
        grad_w = (X.T @ (p - y)) / N + l2 * w
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w.astype(np.float32)
        b = np.float32(b - lr * grad_b)
    return w, b


def roc_auc_score_np(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(y_score)
    y_true_sorted = y_true[order]
    ranks = np.arange(1, len(y_true_sorted) + 1, dtype=np.float64)
    rank_sum_pos = np.sum(ranks[y_true_sorted == 1])
    auc = (rank_sum_pos - pos * (pos + 1) / 2) / (pos * neg)
    return float(auc)


mu, sigma = standardize_fit(X_train)
Xtr = standardize_apply(X_train, mu, sigma)
Xva = standardize_apply(X_val, mu, sigma)

w, b = train_logreg_l2(Xtr, y_train, lr=0.1, steps=800, l2=1.0)

val_pred = sigmoid(Xva @ w + b)
val_auc = roc_auc_score_np(y_val, val_pred)
print("Validation subject-level AUC:", val_auc)



## === cell 5
X_full, y_full, kept_full_ids = load_subject_features(
    all_train_ids, TRAIN_DIR, labels_df
)
mu_f, sigma_f = standardize_fit(X_full)
X_full_s = standardize_apply(X_full, mu_f, sigma_f)
w_f, b_f = train_logreg_l2(X_full_s, y_full, lr=0.1, steps=900, l2=1.0)

train_prior = float(np.mean(y_full)) if y_full is not None and len(y_full) else 0.5
print("Train prior:", train_prior, "Full train N:", X_full.shape[0])



## === cell 6
test_ids = sample_df["BraTS21ID"].tolist()
X_test, _, kept_test_ids = load_subject_features(
    test_ids, TEST_DIR, labels_indexed=None
)

test_pred_map = {}
if X_test.shape[0] > 0:
    X_test_s = standardize_apply(X_test, mu_f, sigma_f)
    preds = sigmoid(X_test_s @ w_f + b_f).astype(np.float32)
    for sid, p in zip(kept_test_ids, preds):
        test_pred_map[sid] = float(p)

sub = sample_df.copy()
sub["MGMT_value"] = (
    sub["BraTS21ID"].map(test_pred_map).fillna(train_prior).astype(np.float32)
)
sub["MGMT_value"] = sub["MGMT_value"].clip(0.0, 1.0)

print(sub.head())
print("Submission shape:", sub.shape)



## === cell 7
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.describe(include="all"))
