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

3.10

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

0.70941

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the immediate runtime errors by updating the deprecated `pydicom.read_file` call to `pydicom.dcmread` and by guarding TensorFlow/Keras imports that can fail in this environment due to a known protobuf mismatch (they’re not needed for the fallback). Since the referenced pretrained model file doesn’t exist, I keep the same “predict probability per BraTS21ID” evaluation semantics by replacing the missing model inference with a simple, deterministic baseline: the global mean of `MGMT_value` from the training labels. Finally, I ensure the submission is written as `submission.csv` with the exact required columns and row alignment to `sample_submission.csv`.'
- What this solution (achieved 0.5) has done: 'I fix the immediate crash by removing the TensorFlow import attempt that triggers a protobuf incompatibility in this Kaggle environment (it isn’t needed for your current baseline submission path). I keep your existing core logic (global-mean baseline prediction) unchanged to avoid unnecessary score changes since you’re already at 0.5 and the target score provided is not meaningful for AUC. I also make the data root resolution more robust (try common Kaggle paths) while keeping the same default path, and ensure the submission is always written as `submission.csv` with the exact required columns and ID formatting.'
- What this solution (achieved 0.34588) has done: 'Your current 0.5 AUC comes from predicting a constant probability (the train mean), which is effectively random-ranking on AUC and tends to land near 0.5. To move the score upward toward a more realistic target, the smallest legitimate improvement without changing the overall approach is to replace the constant with a simple, deterministic image-derived signal per subject (still producing one probability per BraTS21ID). I keep your DICOM loading and folder traversal logic, but actually use it to compute a lightweight per-case feature (mean intensity of a small set of central slices from one modality) and map it to a probability via rank normalization. This preserves the “no external model file” fallback while giving the submission non-constant ordering, which should increase AUC above 0.5.'
- What this solution (achieved 0.34588) has done: 'Your current rank-normalized intensity feature likely overfits to scanner/sequence brightness differences that don’t correlate reliably with MGMT, which can push AUC below a constant baseline. To move the score back upward toward the safer ~0.5 region with minimal semantic change (still one deterministic probability per case), I keep the exact same feature extraction, but blend the ranked signal with the global-mean prior so predictions are less extreme/noisy. This preserves your lightweight inference path and should reduce harmful ordering while still avoiding a fully-constant submission. I also make the rank step stable for ties by using an average-rank transform, which can slightly reduce arbitrary noise without changing the overall approach.'
- What this solution (achieved 0.65412) has done: 'Your current score (0.34588 AUC) is worse than the safer constant-baseline behavior (usually ~0.5 AUC), so the smallest change to move upward toward that region is to reduce the influence of the noisy rank-based intensity signal. I keep the exact same feature extraction and rank-normalization logic, but make the blend much more conservative by lowering `alpha`, so predictions are closer to the global-mean prior while still non-constant. I also compute the ranked signal from the training set (same feature) and flip its direction if it is negatively correlated with the training labels, which is a minimal, legitimate calibration step that often prevents “anti-signal” submissions. These tweaks preserve your overall approach (one lightweight per-case feature + rank mapping + blending) while aiming to increase AUC toward ~0.5.'
- What this solution (achieved 0.65412) has done: 'Your current AUC (0.65412) is already far better than a constant baseline and your provided target score (-1.0) is not meaningful for an AUC (which is typically in [0,1]). To minimize the risk of accidentally moving away from your current good score, I keep the exact same feature extraction and rank/blending logic, but make two small “stability” fixes: compute the rank-flip correlation on the same filled-feature values (so NaNs don’t distort the flip decision), and add a tiny deterministic monotonic “jitter” based on BraTS21ID to break rank ties consistently (AUC is rank-based, so this can help when many cases share identical means). These changes preserve your evaluation semantics (one probability per case, still mostly near the global mean due to small alpha) while slightly reducing arbitrary ordering noise that can hurt leaderboard AUC. The script still runs end-to-end and writes a valid `submission.csv` with correct columns and ID formatting.'
- What this solution (achieved 0.65412) has done: 'Your current AUC (0.65412) is already strong, and the provided target score (-1.0) is not achievable/meaningful for an AUC metric, so the safest way to move “toward the target” without risking a big drop is to keep the same logic and only make tiny stability/calibration fixes. I (1) compute the train-side flip decision using the same tie-broken ranked signal you use at test time (so the flip choice is consistent), and (2) add the same tiny deterministic ID-based jitter to the train ranks as well to avoid tie-heavy artifacts that can mis-estimate correlation. I also cache per-case features during the run so repeated feature extraction (train for flip + test for prediction) doesn’t waste time and stays within the 600s budget, without changing the feature itself. The submission writing and columns/format remain identical.'
- What this solution (achieved 0.65412) has done: 'Your current AUC (0.65412) is already strong, and the provided target score (-1.0) is not meaningful for an AUC metric, so the safest way to avoid moving away from your current score is to keep the same core pipeline and only make tiny, stability-focused tweaks. I (1) compute the rank-flip decision using the exact same NaN-filling and tie-breaking that you apply at test time (so the flip direction is consistent), and (2) use a deterministic ID-based jitter that’s computed the same way for both train and test to reduce tie-related randomness. I also add a small guard to ensure we never attempt to use non-finite probabilities, without changing the model/feature logic. The script still run end-to-end and write `submission.csv` with the required columns and formatting.'
- What this solution (achieved 0.70824) has done: 'Your current score (0.65412 AUC) is already strong, and the provided target (-1.0) is not meaningful for an AUC metric, so the safest “toward target” action is to keep performance stable with only tiny robustness improvements. I keep the exact same core pipeline (T2w mean-intensity feature → average-rank → optional flip from train correlation → blend with global mean) and only make two minimal fixes that reduce unintended noise: (1) compute the intensity feature on a brain-focused center crop (still the same mean-intensity feature, just less background-dominated), and (2) make DICOM loading more consistent by applying VOI LUT/windowing when available and handling MONOCHROME1 inversion. These changes should slightly stabilize ordering (AUC is rank-based) without changing the model/training approach or risking big score swings, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.70941) has done: 'Your current AUC (0.70824) is already strong, and since the provided target score (-1.0) is not meaningful for an AUC metric, the safest way to avoid moving away from your current score is to keep the exact same pipeline and only reduce avoidable feature noise. I make DICOM sorting more robust (so slice ordering is consistent even when filenames vary) and standardize pixel spacing by resizing with correct aspect ratio handling (still producing the same mean-intensity feature, just less distorted). I also compute a slightly more stable per-case feature by trimming extreme slice means (reduces the impact of a few corrupted/odd slices) while keeping “mean intensity of central slices” semantics. The submission format, blending, ranking, flip logic, and output file path/name remain unchanged.'
- What this solution (achieved 0.68706) has done: 'Your current AUC (0.70941) is already strong, while the provided target (-1.0) is not meaningful for an AUC metric, so the safest way to avoid moving away from your current score is to keep the exact same feature/rank/flip/blend pipeline and only reduce avoidable feature noise. I make one minimal, directly score-relevant tweak: compute the central-slice intensity feature from a robust per-slice statistic (median) instead of mean, which is less sensitive to bright artifacts and residual background and typically stabilizes rank ordering (AUC is rank-based). Everything else (DICOM loading, slice selection, ranking + tiny ID-jitter, train-correlation flip, alpha blending, and submission format/path) remains unchanged. This stays within the same “lightweight intensity feature” semantics and should nudge performance upward or at least reduce variance without risky architectural changes.'
- What this solution (achieved 0.70941) has done: 'Your current AUC (0.68706) is already strong, and the provided target score (-1.0) is not meaningful for an AUC metric; the safest way to avoid moving away from your current score is to keep the exact same feature→rank→flip→blend pipeline and only make a tiny, directly score-relevant stability tweak. Specifically, I revert the per-slice statistic from `median` back to `mean` (still on the same center crop and same slice selection), because switching to median is a larger behavior change and can unpredictably hurt rank ordering on this dataset. Everything else (VOI LUT/MONOCHROME handling, robust slice sort, trimming, average-rank + deterministic ID jitter, train-correlation flip, alpha blending, and submission format/path) remains unchanged to minimize risk. The script still run end-to-end and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.70941) has done: 'Because your current AUC (0.70941) is already strong and the provided target score (-1.0) is not meaningful for an AUC metric, the lowest-risk way to avoid moving away from your current score is to keep the exact same feature→rank→flip→blend pipeline and only make a tiny stability tweak. The main avoidable noise left is that the train-side flip decision is based on a raw correlation proxy; switching it to an actual ROC-AUC computed on the same ranked signal is a minimal calibration change that preserves semantics but more reliably chooses whether to flip. I also keep everything else identical and add a small guard so that if the AUC computation is degenerate (all labels same), we default to no-flip rather than injecting randomness. The script still runs end-to-end and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.70941) has done: 'Your current AUC (0.70941) is already strong, while the provided target score (-1.0) is not meaningful for an AUC metric, so the safest way to avoid moving away from your current score is to keep the pipeline identical and only remove tiny sources of avoidable noise. I keep the exact same feature (T2w center-crop mean intensity on central slices), rank normalization, train-side AUC-based flip, and alpha blending. The only change is to make slice ordering more consistent by preferring DICOM `InstanceNumber`/`ImagePositionPatient` when available (falling back to filename number), which can reduce accidental slice mis-ordering that perturbs the per-case feature and therefore the rank-based AUC. The script still run end-to-end and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.70941) has done: 'Your current AUC (0.70941) is already strong, and the provided target score (-1.0) is not meaningful for an AUC metric, so the lowest-risk approach is to keep your exact pipeline and only remove a small source of ordering noise. I keep the same feature (T2w center-crop mean intensity on central slices), ranking, AUC-based flip decision, and alpha blending, but compute the flip decision more robustly by evaluating train AUC via a deterministic out-of-fold (K-fold) estimate instead of using the same data to choose the flip direction. This preserves evaluation semantics while reducing the chance of picking the wrong flip due to noise/overfitting in the flip decision. The script still runs end-to-end under the same paths and writes a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import re
import numpy as np
import pandas as pd

import pydicom
import cv2

tf = None
_TF_IMPORT_ERROR = "TensorFlow import skipped to avoid protobuf incompatibility; not needed for baseline."




## === cell 1
TYPES = ["FLAIR", "T1w", "T2w", "T1wCE"]
WHITE_THRESHOLD = 10  # out of 255 (kept, though unused)
EXCLUDE = [109, 123, 709]

_CANDIDATE_ROOTS = [
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification",
    "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification",
    "/kaggle/data/rsna-miccai-brain-tumor-radiogenomic-classification",
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/rsna-miccai-brain-tumor-radiogenomic-classification",
    "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/rsna-miccai-brain-tumor-radiogenomic-classification",
]
DATA_ROOT = None
for _p in _CANDIDATE_ROOTS:
    if os.path.exists(os.path.join(_p, "train_labels.csv")) and os.path.exists(
        os.path.join(_p, "sample_submission.csv")
    ):
        DATA_ROOT = _p
        break
if DATA_ROOT is None:
    DATA_ROOT = _CANDIDATE_ROOTS[0]

train_df = pd.read_csv(os.path.join(DATA_ROOT, "train_labels.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))

train_df = train_df[~train_df.BraTS21ID.isin(EXCLUDE)].reset_index(drop=True)


def _safe_int_from_filename(path: str) -> int:
    """
    Change (stability): more robust slice ordering key.
    Some series have filenames like 'Image-387.dcm' but we guard against unexpected patterns.
    """
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.findall(r"\d+", base)
    if not m:
        return 0
    return int(m[-1])


def load_dicom(path, size=224):
    """
    Reads a DICOM image, standardizes to [0,1] then rescales to [0,255] uint8 and resizes.

    Change (stability): apply VOI LUT/windowing when present and handle MONOCHROME1 inversion.
    Change (stability): respect PixelSpacing aspect ratio before resizing to reduce geometric distortion
    that can perturb mean intensity comparisons across cases.
    """
    dicom = pydicom.dcmread(path)

    try:
        arr = dicom.pixel_array.astype(np.float32)
    except Exception:
        arr = np.zeros((size, size), dtype=np.float32)

    try:
        from pydicom.pixel_data_handlers.util import apply_voi_lut

        arr = apply_voi_lut(arr, dicom).astype(np.float32)
    except Exception:
        pass

    try:
        if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = np.max(arr) - arr
    except Exception:
        pass

    if arr.size:
        lo = float(np.percentile(arr, 1.0))
        hi = float(np.percentile(arr, 99.0))
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        else:
            mx = float(np.max(arr))
            if mx > 0:
                arr = arr / mx
    else:
        arr = np.zeros((size, size), dtype=np.float32)

    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

    try:
        ps = getattr(dicom, "PixelSpacing", None)
        if ps is not None and len(ps) >= 2:
            row_sp = float(ps[0])
            col_sp = float(ps[1])
            if row_sp > 0 and col_sp > 0:
                scale_x = col_sp / min(row_sp, col_sp)
                scale_y = row_sp / min(row_sp, col_sp)
                if not (abs(scale_x - 1.0) < 1e-6 and abs(scale_y - 1.0) < 1e-6):
                    h, w = arr.shape[:2]
                    new_w = max(1, int(round(w * scale_x)))
                    new_h = max(1, int(round(h * scale_y)))
                    arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception:
        pass

    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)




## === cell 2
def get_all_image_paths(brats21id, image_type, folder="train"):
    """
    Returns an array of all the images of a particular type for a particular patient ID.

    Change (score-stability, minimal): sort slices using DICOM metadata when available
    (InstanceNumber, then ImagePositionPatient z), falling back to filename number.
    This keeps the same "use central slice window then compute mean intensity" logic,
    but reduces accidental slice mis-ordering noise that can perturb rank-based AUC.
    """
    assert image_type in TYPES

    patient_path = os.path.join(
        DATA_ROOT,
        folder,
        str(int(brats21id)).zfill(5),
    )

    paths = glob.glob(os.path.join(patient_path, image_type, "*"))
    if not paths:
        return np.array([], dtype=object)

    def _dicom_sort_key(p: str):
        try:
            ds = pydicom.dcmread(
                p,
                stop_before_pixels=True,
                specific_tags=["InstanceNumber", "ImagePositionPatient"],
            )
            inst = getattr(ds, "InstanceNumber", None)
            if inst is not None:
                try:
                    return (0, int(inst))
                except Exception:
                    pass
            ipp = getattr(ds, "ImagePositionPatient", None)
            if ipp is not None and len(ipp) >= 3:
                try:
                    return (1, float(ipp[2]))
                except Exception:
                    pass
        except Exception:
            pass
        return (2, _safe_int_from_filename(p))

    paths = sorted(paths, key=_dicom_sort_key)

    num_images = len(paths)
    if num_images == 0:
        return np.array([], dtype=object)

    if num_images > 10:
        start = int(num_images * 0.25)
        end = int(num_images * 0.75)
    else:
        start = 0
        end = num_images

    interval = 1
    return np.array(paths[start:end:interval], dtype=object)


def get_all_images(brats21id, image_type, folder="train", size=224):
    return [
        load_dicom(path, size)
        for path in get_all_image_paths(brats21id, image_type, folder)
    ]


IMAGE_SIZE = 224


def get_all_data_for_test(image_type):
    """
    Kept for compatibility with the original flow, but not used in the fallback submission.
    """
    X = []
    test_ids = []
    for brats_id in sample_sub["BraTS21ID"].values:
        images = get_all_images(int(brats_id), image_type, "test", IMAGE_SIZE)
        X += images
        test_ids += [int(brats_id)] * len(images)
    return np.array(X), np.array(test_ids)




## === cell 3
global_mean = float(train_df["MGMT_value"].mean())
global_mean = min(max(global_mean, 0.0), 1.0)

_FEATURE_CACHE = {}


def _case_intensity_feature(
    brats_id, image_type="T2w", folder="test", size=96, max_slices=12
):
    """
    Lightweight per-case feature: mean pixel intensity over a small set of central slices.

    Change (stability): compute the mean on a center crop to reduce background/air dominance.
    Change (stability): trim extreme slice means (small %) to reduce influence of odd/corrupt slices
    while preserving "mean intensity of central slices" semantics.

    Change (score-stability, minimal): use per-slice MEAN intensity (instead of median) on the crop.
    This keeps the original "mean intensity" semantics and often yields a smoother, less tie-heavy
    ranking than median on this dataset, which can improve AUC stability without changing the pipeline.
    """
    key = (int(brats_id), str(image_type), str(folder), int(size), int(max_slices))
    if key in _FEATURE_CACHE:
        return _FEATURE_CACHE[key]

    paths = get_all_image_paths(int(brats_id), image_type, folder=folder)
    n = len(paths)
    if n == 0:
        _FEATURE_CACHE[key] = np.nan
        return np.nan

    if n <= max_slices:
        chosen = paths
    else:
        mid = n // 2
        half = max_slices // 2
        start = max(0, mid - half)
        end = min(n, start + max_slices)
        start = max(0, end - max_slices)
        chosen = paths[start:end]

    slice_stats = []
    for p in chosen:
        img = load_dicom(p, size=size)

        h, w = img.shape[:2]
        ch = int(round(h * 0.70))
        cw = int(round(w * 0.70))
        y0 = (h - ch) // 2
        x0 = (w - cw) // 2
        crop = img[y0 : y0 + ch, x0 : x0 + cw]

        slice_stats.append(float(np.mean(crop)))

    if not slice_stats:
        out = np.nan
    else:
        sm = np.asarray(slice_stats, dtype=np.float32)
        if sm.size >= 10:
            lo = float(np.percentile(sm, 10.0))
            hi = float(np.percentile(sm, 90.0))
            sm = np.clip(sm, lo, hi)
        out = float(np.mean(sm))

    _FEATURE_CACHE[key] = out
    return out


def _average_rank_to_unit_interval(x: np.ndarray) -> np.ndarray:
    """
    Uses average ranks for ties to reduce arbitrary ordering noise.
    """
    x = np.asarray(x, dtype=np.float32)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]

    n = len(x)
    ranks_sorted = np.empty(n, dtype=np.float32)

    i = 0
    while i < n:
        j = i + 1
        while j < n and xs[j] == xs[i]:
            j += 1
        avg_rank = 0.5 * (i + (j - 1))
        ranks_sorted[i:j] = avg_rank
        i = j

    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = ranks_sorted

    if n > 1:
        return ranks / (n - 1.0)
    return np.array([0.5], dtype=np.float32)


def _id_based_unit_jitter(ids: np.ndarray) -> np.ndarray:
    """
    Deterministic tie-breaker based only on ID, used consistently on train and test.
    """
    ids = np.asarray(ids, dtype=np.int64)
    return _average_rank_to_unit_interval(ids.astype(np.float32)).astype(np.float32)


def _roc_auc_fast(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Change (score-stability, minimal): compute ROC-AUC directly (no sklearn dependency).
    Used only to choose whether to flip the ranked signal; core prediction pipeline unchanged.
    Returns NaN if AUC is undefined (all labels same).
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float32)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ys = y_score[order]
    yt = y_true[order]

    n = len(y_score)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and ys[j] == ys[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)  # 1-based average rank over [i+1, j]
        ranks[i:j] = avg_rank
        i = j

    rank_sum_pos = float(np.sum(ranks[yt == 1]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _rank_signal_for_ids(ids: np.ndarray, feats: np.ndarray) -> np.ndarray:
    """
    Helper: apply the exact same fill + average-rank + tiny ID jitter transform.
    """
    ids = np.asarray(ids, dtype=np.int64)
    feats = np.asarray(feats, dtype=np.float32)

    valid = np.isfinite(feats)
    if not valid.any():
        return np.full(len(ids), 0.5, dtype=np.float32)

    feats_filled = feats.copy()
    feats_filled[~valid] = float(np.median(feats_filled[valid]))

    r = _average_rank_to_unit_interval(feats_filled).astype(np.float32)
    id_u = _id_based_unit_jitter(ids)
    r = np.clip(r + 1e-6 * (id_u - 0.5), 0.0, 1.0).astype(np.float32)
    return r


def _compute_rank_flip_using_train(
    image_type="T2w", size=96, max_slices=12, n_splits=5
) -> float:
    """
    Change (score-stability, minimal): choose flip using a deterministic out-of-fold ROC-AUC
    estimate of the ranked signal. This preserves the same flip-via-AUC logic, but reduces
    overfitting/noise from evaluating the flip on the same data used to decide it.
    """
    train_ids = train_df["BraTS21ID"].astype(int).values
    y = train_df["MGMT_value"].values.astype(np.int8)

    f = np.empty(len(train_ids), dtype=np.float32)
    for i, bid in enumerate(train_ids):
        f[i] = _case_intensity_feature(
            bid, image_type=image_type, folder="train", size=size, max_slices=max_slices
        )

    if np.isfinite(f).sum() < 10:
        return 1.0

    order = np.argsort(train_ids, kind="mergesort")
    folds = np.empty(len(train_ids), dtype=np.int32)
    for k, idx in enumerate(order):
        folds[idx] = k % int(max(2, n_splits))

    aucs = []
    for fold in range(int(max(2, n_splits))):
        mask = folds == fold
        if mask.sum() < 5:
            continue
        y_fold = y[mask]
        if (y_fold == 1).sum() == 0 or (y_fold == 0).sum() == 0:
            continue
        r_fold = _rank_signal_for_ids(train_ids[mask], f[mask])
        auc_fold = _roc_auc_fast(y_fold, r_fold)
        if np.isfinite(auc_fold):
            aucs.append(float(auc_fold))

    if len(aucs) == 0:
        r_all = _rank_signal_for_ids(train_ids, f)
        auc_all = _roc_auc_fast(y, r_all)
        if not np.isfinite(auc_all):
            return 1.0
        return -1.0 if auc_all < 0.5 else 1.0

    auc = float(np.mean(aucs))
    return -1.0 if auc < 0.5 else 1.0


test_ids = sample_sub["BraTS21ID"].astype(int).values
feat = np.empty(len(test_ids), dtype=np.float32)
for i, bid in enumerate(test_ids):
    feat[i] = _case_intensity_feature(
        bid, image_type="T2w", folder="test", size=96, max_slices=12
    )

valid = np.isfinite(feat)
if valid.any():
    ranked = _rank_signal_for_ids(test_ids, feat)

    flip = _compute_rank_flip_using_train(
        image_type="T2w", size=96, max_slices=12, n_splits=5
    )
    if flip < 0:
        ranked = (1.0 - ranked).astype(np.float32)

    alpha = 0.05
    probs = (1.0 - alpha) * np.float32(global_mean) + alpha * ranked
else:
    probs = np.full(len(test_ids), global_mean, dtype=np.float32)

probs = np.asarray(probs, dtype=np.float32)
bad = ~np.isfinite(probs)
if bad.any():
    probs[bad] = np.float32(global_mean)

eps = 1e-4
probs = np.clip(probs, eps, 1.0 - eps).astype(float)

submission = sample_sub.copy()
submission["MGMT_value"] = probs
submission["BraTS21ID"] = submission["BraTS21ID"].astype(str).str.zfill(5)
submission["MGMT_value"] = submission["MGMT_value"].astype(float)

submission.to_csv("submission.csv", index=False)
submission.head()




## === cell 4
"""
file_path = "../input/mri-best-model-2dcnn/best_model.h5"
if tf is not None and os.path.exists(file_path):
    model_best = tf.keras.models.load_model(filepath=file_path)
    X_test, testidt = get_all_data_for_test("T1w")
    y_pred = model_best.predict(X_test, verbose=0)

    # If model outputs logits/probabilities with 2 columns, use positive class prob; if 1 column, use it.
    if y_pred.ndim == 2 and y_pred.shape[1] == 2:
        pred_img = y_pred[:, 1]
    else:
        pred_img = y_pred.reshape(-1)

    result = pd.DataFrame({"BraTS21ID": testidt, "MGMT_value": pred_img})
    result2 = result.groupby("BraTS21ID", as_index=False).mean()

    # Align ordering to sample_submission
    submission2 = sample_sub[["BraTS21ID"]].copy()
    submission2["BraTS21ID_int"] = submission2["BraTS21ID"].astype(int)
    result2["BraTS21ID_int"] = result2["BraTS21ID"].astype(int)
    submission2 = submission2.merge(result2[["BraTS21ID_int", "MGMT_value"]], on="BraTS21ID_int", how="left")
    submission2["MGMT_value"] = submission2["MGMT_value"].fillna(global_mean).clip(0, 1)
    submission2 = submission2.drop(columns=["BraTS21ID_int"])
    submission2.to_csv("submission.csv", index=False)
"""
