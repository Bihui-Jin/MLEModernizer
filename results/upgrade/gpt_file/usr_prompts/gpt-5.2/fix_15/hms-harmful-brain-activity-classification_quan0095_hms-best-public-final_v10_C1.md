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
Detect and classify harmful brain activity in electroencephalography (EEG) data: seizure (SZ), generalized periodic discharges (GPD), lateralized periodic discharges (LPD), lateralized rhythmic delta activity (LRDA), generalized rhythmic delta activity (GRDA), or "other".

## Metric
Kullback Liebler divergence between the predicted probability and the observed target.

## Submission Format
For each `eeg_id` in the test set, you must predict a probability for each of the `vote` columns. The file should contain a header and have the following format:

```
eeg_id,seizure_vote,lpd_vote,gpd_vote,lrda_vote,grda_vote,other_vote\
0,0.166,0.166,0.167,0.167,0.167,0.167\
1,0.166,0.166,0.167,0.167,0.167,0.167\
etc.
```

Your total predicted probabilities for each row must sum to one or your submission will fail.

## Dataset
**train.csv** Metadata for the train set. The expert annotators reviewed 50 second long EEG samples plus matched spectrograms covering 10 a minute window centered at the same time and labeled the central 10 seconds. Many of these samples overlapped and have been consolidated. `train.csv` provides the metadata that allows you to extract the original subsets that the raters annotated.

- `eeg_id` - A unique identifier for the entire EEG recording.
- `eeg_sub_id` - An ID for the specific 50 second long subsample this row's labels apply to.
- `eeg_label_offset_seconds` - The time between the beginning of the consolidated EEG and this subsample.
- `spectrogram_id` - A unique identifier for the entire EEG recording.
- `spectrogram_sub_id` - An ID for the specific 10 minute subsample this row's labels apply to.
- `spectogram_label_offset_seconds` - The time between the beginning of the consolidated spectrogram and this subsample.
- `label_id` - An ID for this set of labels.
- `patient_id` - An ID for the patient who donated the data.
- `expert_consensus` - The consensus annotator label. Provided for convenience only.
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The count of annotator votes for a given brain activity class. The full names of the activity classes are as follows: `lpd`: lateralized periodic discharges, `gpd`: generalized periodic discharges, `lrd`: lateralized rhythmic delta activity, and `grda`: generalized rhythmic delta activity . A detailed explanations of these patterns is [available here.](https://www.acns.org/UserFiles/file/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf)

**test.csv** Metadata for the test set. As there are no overlapping samples in the test set, many columns in the train metadata don't apply.

- `eeg_id`
- `spectrogram_id`
- `patient_id`

**sample_submission.csv**

- `eeg_id`
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The target columns. Your predictions must be probabilities. Note that the test samples had between 3 and 20 annotators.

**train_eegs/** EEG data from one or more overlapping samples. Use the metadata in train.csv to select specific annotated subsets. The column names are [the names of the individual electrode locations for EEG leads](https://en.wikipedia.org/wiki/10%E2%80%9320_system_%28EEG%29), with one exception. The EKG column is for an electrocardiogram lead that records data from the heart. All of the EEG data (for both train and test) was collected at a frequency of 200 samples per second.

**test_eegs/** Exactly 50 seconds of EEG data.

train_spectrograms/ Spectrograms assembled EEG data. Use the metadata in train.csv to select specific annotated subsets. The column names indicate the frequency in hertz and the recording regions of the EEG electrodes. The latter are abbreviated as LL = left lateral; RL = right lateral; LP = left parasagittal; RP = right parasagittal.

**test_spectrograms/** Spectrograms assembled using exactly 10 minutes of EEG data.

**example_figures/** Larger copies of the example case images used on the overview tab.

# 2. Python version

3.12

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        input/
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        working/
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
```

-> data/hms-harmful-brain-activity-classification/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/hms-harmful-brain-activity-classification/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/hms-harmful-brain-activity-classification/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> (stopped after 10 files for performance)

# 5. Target score

0.2838780478053066

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The main runtime failure is that the notebook hard-codes `device="cuda:0"` and crashes in CPU-only environments; I switch to an automatic device selection and load checkpoints with `map_location` so inference works on CPU too. Several later cells rely on variables (`result_6`, `result_7`, etc.) that never get created once earlier GPU cells crash; I simplify to a single inference pass that always runs and produces one `result` dict. To keep the pipeline stable and within time, I also remove extremely verbose per-sample printing and ensure predictions are valid probabilities (non-negative and row-sum = 1) and aligned exactly to `sample_submission.csv`’s `eeg_id` order. Finally, I keep the core architecture and feature generation logic intact, but make file/path handling and submission writing robust so a `submission.csv` is always produced.'
- What this solution (achieved 1.40995) has done: 'The timeout is dominated by two stages: per-eeg feature building (multiple heavy `pd.read_parquet` per row + SciPy filtering/spectrograms) and per-sample resizing inside `__getitem__` (6× `skimage.resize` for 9850 items). To finish under 600s without changing the model or inference semantics, the refactor (1) ensures each EEG/spec parquet is read only once per `eeg_id` during feature building, (2) uses all available CPU cores for feature building when needed, and (3) replaces slow `skimage.transform.resize` with a deterministic, numerically-equivalent `torch.nn.functional.interpolate` path executed in DataLoader workers (CPU), preserving float32 outputs. Additionally, it avoids redundant tensor conversions in the inference loop and uses faster pinned-memory transfers when on GPU; the forward passes, ensembling logic, and probability post-processing are unchanged.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.28388), so we should make small changes that legitimately reduce KL without changing the model or feature logic. The biggest likely issue is a mismatch between training-target semantics (soft labels as vote-distributions) and inference (pure softmax on logits): applying a tiny amount of probability smoothing (label-distribution style) and using the sample-submission prior instead of uniform fallback typically reduces extreme/overconfident KL penalties while preserving the same inference pipeline. I also fix the `need_build` check to avoid unnecessary rebuilds (and ensure it builds when files are missing), but keep feature generation identical. Finally, I ensure numeric stability by renormalizing after smoothing and clipping.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far above the target (0.28388), and the most likely cause is miscalibration/overconfidence from a plain softmax ensemble on logits. I keep the exact same model, features, and inference loop, but add a tiny temperature scaling to soften probabilities (a standard, semantics-preserving calibration step for KL) and make the existing prior-mix smoothing slightly stronger while staying conservative. I also ensure the “no-weights-loaded” fallback uses the same prior (not uniform), since uniform can be badly wrong and heavily penalized by KL. These are minimal post-processing changes that typically reduce KL without altering core logic or requiring retraining, and they still guarantee row-sums=1 for a valid submission.'
- What this solution (achieved 1.40995) has done: 'Your KL (1.40995, lower-is-better) is far above the target (0.28388), so the smallest safe move toward the target is to reduce overconfidence and avoid pathological class probabilities without changing the model or features. I keep the exact same feature building, dataset, model, and ensemble logic, but calibrate probabilities a bit more by slightly increasing temperature scaling and slightly strengthening the prior-mix smoothing (both directly reduce KL for this metric). I also make one correctness fix: `seed_everything` currently sets `cudnn.deterministic=True` and `benchmark=True` together (contradictory), so I set `benchmark=False` to reduce nondeterministic variability (stability, not speed). Finally, I ensure the submission is aligned to `sample_submission.csv` order (already done) and always produces valid probabilities (already done), leaving everything else intact.'
- What this solution (achieved 1.39779) has done: 'Your current KL (1.40995, lower-is-better) is far above the target (0.28388), so the most likely “minimal-change” win is fixing probability calibration rather than changing the model/features. I keep the exact same model, features, and ensemble flow, but adjust post-processing to be more vote-distribution-like: (1) apply a small Dirichlet-style floor (epsilon) to avoid near-zero probabilities that are heavily penalized by KL, and (2) slightly increase temperature and prior-mix smoothing to reduce overconfidence. I also fix the `prior` computation: averaging `sample_submission.csv` (which is typically uniform) is not a useful prior; instead I compute a class prior from `train.csv` vote distributions (legitimate, label-available) and use it only for smoothing/fallback. These are minimal semantic changes confined to calibration and should move KL substantially down toward the target while preserving the core inference logic and producing a valid `submission.csv`.'
- What this solution (achieved 1.39779) has done: 'Your KL is much worse than the target, so we should focus on a minimal, semantics-preserving calibration fix rather than changing features/models. The biggest likely issue is that your post-processing mixes with the prior but does not enforce a true probability “floor” across all classes; KL punishes near-zero probabilities heavily when the true vote distribution has mass there. I keep your exact ensemble/model/feature pipeline, but replace the current “DIRICHLET_EPS * prior add” with a proper epsilon-floor mixture `p = (1-eps)*p + eps*prior` (always guarantees every class has at least eps*prior mass), and I slightly increase temperature and smoothing to reduce overconfidence toward the target band. Everything else (paths, inference loop, CSV format, and row-sum=1 validity) remains unchanged.'
- What this solution (achieved 1.39779) has done: 'Your current KL (1.39779, lower-is-better) is far above the target (0.28388), so the smallest safe move is to further reduce overconfidence and prevent near-zero class probabilities that KL heavily penalizes. I keep the exact same feature generation, dataset, model architecture, and ensemble averaging, and only adjust the probability calibration/post-processing. Concretely, I slightly soften logits with a higher temperature and replace the two-step prior mixing with one stronger but single, well-defined mixture with the train-derived prior (plus a tiny clip) to guarantee a probability floor. This keeps evaluation semantics identical (still valid probabilities summing to 1) while typically moving KL substantially downward toward the target.'
- What this solution (achieved 1.39719) has done: 'Your current KL (1.39779, lower-is-better) is far above the target (0.28388), so we should make the smallest safe calibration changes that reduce overconfidence and avoid near-zero probabilities (which KL punishes heavily) without changing the model, features, or inference flow. I keep the exact same ensemble and feature pipeline, but (1) soften probabilities a bit more via a slightly higher temperature, and (2) replace the single strong prior-mix with a two-stage mix: a small uniform floor (prevents any class from going too close to 0) plus a moderate train-prior mix (keeps predictions realistic). This is confined to post-processing and preserves evaluation semantics (valid probabilities summing to 1) while typically lowering KL substantially. Everything still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 1.39749) has done: 'Your current KL (1.39719, lower-is-better) is far above the target (0.28388), so we should reduce overconfidence and avoid near-zero probabilities (which KL punishes) with minimal, post-processing-only changes. I keep the exact same feature generation, dataset, model, and ensemble logic, but (1) soften logits a bit more via a slightly higher temperature, and (2) tune the probability smoothing to be less “washed out” (reduce uniform/prior mixing) while adding a small explicit probability floor so no class collapses to ~0. I also ensure inference is done under `torch.inference_mode()` for consistency and keep strict submission alignment/row-sum=1 guarantees unchanged. These are small calibration adjustments that typically move KL substantially down without altering core semantics.'
- What this solution (achieved 1.39779) has done: 'Your current KL (1.39749, lower-is-better) is far above the target (0.28388), so we should make minimal, semantics-preserving changes that reduce catastrophic KL penalties from near-zero probabilities. I keep the exact model/feature pipeline and only adjust probability calibration/post-processing: (1) reduce temperature (4.40 is likely over-smoothing toward the prior/uniform and can be mismatched) and (2) replace the multi-stage smoothing (uniform + prior + floor) with one principled Dirichlet-style mixture `p = (1-eps)*p + eps*prior` plus a tiny clip, which directly prevents near-zeros while preserving the model signal. I also make the ensemble combine logits first (then softmax once), which is equivalent in intent but typically yields better-calibrated probabilities for KL than averaging post-softmax probabilities, without changing architecture or features. The submission writing, row alignment to `sample_submission.csv`, and row-sum=1 guarantees remain unchanged.'
- What this solution (achieved 1.39758) has done: 'Your current KL (1.39779, lower-is-better) is far above the target (0.28388), so we should only adjust probability calibration/post-processing to reduce overconfidence and avoid near-zero probabilities (which are heavily penalized by KL) while leaving the model, features, and inference flow intact. I keep your “average logits then softmax” ensemble exactly as-is, but add a small **logit-centering** step (per-sample mean subtraction) before temperature softmax to improve numerical calibration without changing ranking/semantics. Then I slightly strengthen the **Dirichlet-style prior mixture** and enforce a small **uniform floor** so no class collapses to ~0 even when the model is extremely confident. All outputs are still valid probabilities summing to 1 and the script still writes `submission.csv`.'
- What this solution (achieved 1.39779) has done: 'Your current KL (1.39758, lower-is-better) is far above the target (0.28388), so the most likely minimal change that moves you toward the target is fixing probability calibration/post-processing (KL is extremely sensitive to overconfident near-zero probabilities). I keep the exact same feature building, dataset, model architecture, and ensemble “average logits then softmax” logic, but adjust only the final probability smoothing to be a single, stronger, well-defined Dirichlet-style mixture with the train-derived prior (and remove the extra uniform-mix step that can wash out useful signal). I also slightly increase temperature to soften predictions (reducing KL penalties) while keeping outputs valid probabilities summing to 1. These changes are confined to post-processing and should reduce KL substantially without altering core semantics or runtime.'
- What this solution (achieved 1.41937) has done: 'Your score is far worse than the target (KL 1.39779 vs 0.28388; lower is better), so the smallest legitimate move is to fix probability calibration without touching the model/feature pipeline. I keep the exact feature extraction, dataset, model, and ensemble logic, but adjust only the final probability post-processing to better match the competition’s “vote-distribution” targets by (1) using a milder temperature (less washing-out than the current 3.20) and (2) reducing the very-strong prior mix (0.35) that can overwhelm model signal. I also compute the smoothing prior from train vote totals (equivalent but slightly more stable than mean-of-per-row distributions) and keep a tiny probability floor + renormalization so the submission is always valid. These changes are confined to calibration and should move KL substantially down toward the target while remaining minimal and safe.'

# 9. Code solution

## === cell 0
import os
import gc
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")



## === cell 1
DEBUG = False



## === cell 2
NAMES = ["LL", "LP", "RP", "RR"]
FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]



## === cell 3
from scipy import signal



## === cell 4
BASE_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
TEST_CSV = f"{BASE_PATH}/test.csv"
TRAIN_CSV = f"{BASE_PATH}/train.csv"
SAMPLE_SUB = f"{BASE_PATH}/sample_submission.csv"
TEST_SPEC_PATH = f"{BASE_PATH}/test_spectrograms/"
TEST_EEG_PATH = f"{BASE_PATH}/test_eegs/"
TRAIN_SPEC_PATH = f"{BASE_PATH}/train_spectrograms/"
TRAIN_EEG_PATH = f"{BASE_PATH}/train_eegs/"

CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
N_CLASSES = len(CLASSES)

if DEBUG:
    test = pd.read_csv(TRAIN_CSV).head(40)
    SPEC_PATH = TRAIN_SPEC_PATH
    EEG_PATH = TRAIN_EEG_PATH
else:
    test = pd.read_csv(TEST_CSV)
    SPEC_PATH = TEST_SPEC_PATH
    EEG_PATH = TEST_EEG_PATH

print("test:", test.shape)



## === cell 5
spec_directory_path = "spec_spectrograms/"
eeg_directory_path = "eeg_spectrograms/"
raw_10s_directory_path = "eeg_10s_raws/"
raw_50s_directory_path = "eeg_50s_raws/"

for p in [
    spec_directory_path,
    eeg_directory_path,
    raw_10s_directory_path,
    raw_50s_directory_path,
]:
    os.makedirs(p, exist_ok=True)



## === cell 6
SFREQ = 200
filter_range = [0.5, 40]

RAW_FEATS = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")




## === cell 7
def stft_spec_from_eeg_df(eeg: pd.DataFrame):
    EEG_LENGTH = 50

    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)

    eeg = eeg.iloc[time_start:time_stop]

    list_eeg = []
    for k in range(4):
        COLS = FEATS[k]
        img = np.zeros((128, 142, 4), dtype="float32")
        for kk in range(4):
            eeg_1 = eeg[COLS[kk]].copy()
            mean_value = float(eeg_1.mean())
            eeg_1 = eeg_1.fillna(mean_value).values

            eeg_2 = eeg[COLS[kk + 1]].copy()
            mean_value = float(eeg_2.mean())
            eeg_2 = eeg_2.fillna(mean_value).values

            new_eeg = eeg_1 - eeg_2
            fs = 200
            nperseg = 70
            noverlap = 0
            f, t, spec = signal.spectrogram(
                new_eeg, fs, nperseg=nperseg, noverlap=noverlap, nfft=256
            )

            spec = np.abs(spec)
            spec = np.log1p(spec).astype("float32")

            img[:, :, kk] += spec[:128, :]
        img = np.concatenate(
            (img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]), 1
        )
        list_eeg.append(img)

    img = np.concatenate(list_eeg, 0)
    img /= 2.0
    return img


def raw10seeg_from_eeg_df(raw_eeg: pd.DataFrame, eeg_id):
    EEG_LENGTH = 10

    def _slice_make(start_s, stop_s):
        time_start = round(start_s * 200)
        time_stop = round(stop_s * 200)
        eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(
            drop=True
        )

        list_eeg = []
        for region in RAW_FEATS.keys():
            eeg_arr = np.zeros(
                (len(RAW_FEATS[region]), eeg_default.shape[0]), dtype=np.float32
            )
            for chan_i, chan in enumerate(RAW_FEATS[region]):
                c1, c2 = chan.split("-")
                eeg_1 = eeg_default.loc[:, c1].copy()
                eeg_1 = eeg_1.fillna(float(eeg_1.mean())).values

                eeg_2 = eeg_default.loc[:, c2].copy()
                eeg_2 = eeg_2.fillna(float(eeg_2.mean())).values

                new_eeg = eeg_1 - eeg_2
                new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
                new_eeg = np.clip(new_eeg, -1024, 1024)
                eeg_arr[chan_i, :] = new_eeg

            eeg_arr = np.reshape(eeg_arr, (4, 200, EEG_LENGTH))
            eeg_arr = np.concatenate(
                [
                    eeg_arr[0, :, :],
                    eeg_arr[1, :, :],
                    eeg_arr[2, :, :],
                    eeg_arr[3, :, :],
                ],
                1,
            )
            list_eeg.append(eeg_arr)

        eeg_c = np.concatenate(list_eeg, 1)
        eeg_c /= 104.0
        return eeg_c

    eeg_c = _slice_make((50 - EEG_LENGTH) / 2, (50 + EEG_LENGTH) / 2)
    eeg_l = _slice_make(18, 28)
    eeg_r = _slice_make(22, 32)
    return eeg_l, eeg_c, eeg_r


def raw50seeg_from_eeg_df(raw_eeg: pd.DataFrame):
    EEG_LENGTH = 50

    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)
    eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(drop=True)

    list_eeg = []
    for region in RAW_FEATS.keys():
        eeg_arr = np.zeros(
            (len(RAW_FEATS[region]), eeg_default.shape[0]), dtype=np.float32
        )
        for chan_i, chan in enumerate(RAW_FEATS[region]):
            c1, c2 = chan.split("-")
            eeg_1 = eeg_default.loc[:, c1].copy()
            eeg_1 = eeg_1.fillna(float(eeg_1.mean())).values

            eeg_2 = eeg_default.loc[:, c2].copy()
            eeg_2 = eeg_2.fillna(float(eeg_2.mean())).values

            new_eeg = eeg_1 - eeg_2
            new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            new_eeg = np.clip(new_eeg, -1024, 1024).astype("float32")
            eeg_arr[chan_i, :] = new_eeg

        eeg_arr = np.reshape(eeg_arr, (4, 200, EEG_LENGTH))
        eeg_arr = np.concatenate(
            (eeg_arr[0, :, :], eeg_arr[1, :, :], eeg_arr[2, :, :], eeg_arr[3, :, :]), 1
        )
        list_eeg.append(eeg_arr)

    eeg_arr = np.concatenate(list_eeg, 1)
    eeg_arr /= 104.0
    return eeg_arr


def stft_spec_from_eeg(parquet_path):
    return stft_spec_from_eeg_df(pd.read_parquet(parquet_path))


def raw10seeg_from_eeg(parquet_path, eeg_id):
    return raw10seeg_from_eeg_df(pd.read_parquet(parquet_path), eeg_id)


def raw50seeg_from_eeg(parquet_path):
    return raw50seeg_from_eeg_df(pd.read_parquet(parquet_path))




## === cell 8
from joblib import Parallel, delayed


def save_features(row):
    eeg_id = row["eeg_id"]
    spec_id = row["spectrogram_id"]

    spec = pd.read_parquet(f"{SPEC_PATH}{spec_id}.parquet")
    eeg_df = pd.read_parquet(f"{EEG_PATH}{eeg_id}.parquet")

    spec_arr = spec.values[:, 1:].T.astype("float32")  # (Hz, Time)
    split_spec_arr = spec_arr[:, 0:300]
    np.save(f"{spec_directory_path}{eeg_id}", split_spec_arr)

    img_l, img_c, img_r = raw10seeg_from_eeg_df(eeg_df, eeg_id)
    np.save(f"{raw_10s_directory_path}{eeg_id}_l", img_l)
    np.save(f"{raw_10s_directory_path}{eeg_id}_c", img_c)
    np.save(f"{raw_10s_directory_path}{eeg_id}_r", img_r)

    img50 = raw50seeg_from_eeg_df(eeg_df)
    np.save(f"{raw_50s_directory_path}{eeg_id}", img50)

    imgstft = stft_spec_from_eeg_df(eeg_df)
    np.save(f"{eeg_directory_path}{eeg_id}", imgstft)


need_build = True
if len(test) > 0:
    sample_eeg_id = str(test.iloc[0]["eeg_id"])
    need_build = not (
        os.path.exists(os.path.join(spec_directory_path, sample_eeg_id + ".npy"))
        and os.path.exists(os.path.join(eeg_directory_path, sample_eeg_id + ".npy"))
        and os.path.exists(os.path.join(raw_50s_directory_path, sample_eeg_id + ".npy"))
        and os.path.exists(
            os.path.join(raw_10s_directory_path, sample_eeg_id + "_l.npy")
        )
        and os.path.exists(
            os.path.join(raw_10s_directory_path, sample_eeg_id + "_c.npy")
        )
        and os.path.exists(
            os.path.join(raw_10s_directory_path, sample_eeg_id + "_r.npy")
        )
    )

if need_build:
    n_jobs = max(1, min(os.cpu_count() or 2, 8))
    _ = Parallel(n_jobs=n_jobs, prefer="processes", batch_size=8)(
        delayed(save_features)(row) for _, row in test.iterrows()
    )




## === cell 9
class Config:
    seed = 2024
    num_folds = 5


def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seed_everything(Config.seed)



## === cell 10
import timm
import torch.utils.data as data
from torch.utils.data import DataLoader



## === cell 11
import torch.nn.functional as F


def _resize_np_to_hw(img: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
    return x.squeeze(0).squeeze(0).numpy().astype("float32", copy=False)




## === cell 12
class ImageFolder(data.Dataset):
    def __init__(self, df, test_imgsize):
        super().__init__()
        self.spec_data_path = spec_directory_path
        self.eeg_data_path = eeg_directory_path
        self.raw_50s_data_path = raw_50s_directory_path
        self.raw_10s_data_path = raw_10s_directory_path
        self.df = df.reset_index(drop=True)
        self.test_imgsize = test_imgsize  # (H, W)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        eeg_id = int(row.eeg_id)

        spec_img = np.load(
            os.path.join(self.spec_data_path, f"{eeg_id}.npy"), mmap_mode="r"
        ).astype("float32", copy=False)
        raw_50s_img = np.load(
            os.path.join(self.raw_50s_data_path, f"{eeg_id}.npy"), mmap_mode="r"
        ).astype("float32", copy=False)
        raw_10s_l_img = np.load(
            os.path.join(self.raw_10s_data_path, f"{eeg_id}_l.npy"), mmap_mode="r"
        ).astype("float32", copy=False)
        raw_10s_c_img = np.load(
            os.path.join(self.raw_10s_data_path, f"{eeg_id}_c.npy"), mmap_mode="r"
        ).astype("float32", copy=False)
        raw_10s_r_img = np.load(
            os.path.join(self.raw_10s_data_path, f"{eeg_id}_r.npy"), mmap_mode="r"
        ).astype("float32", copy=False)
        eeg_img = np.load(
            os.path.join(self.eeg_data_path, f"{eeg_id}.npy"), mmap_mode="r"
        ).astype("float32", copy=False)

        eeg_img = _resize_np_to_hw(eeg_img, self.test_imgsize)
        spec_img = _resize_np_to_hw(spec_img, self.test_imgsize)
        raw_10s_l_img = _resize_np_to_hw(raw_10s_l_img, self.test_imgsize)
        raw_10s_c_img = _resize_np_to_hw(raw_10s_c_img, self.test_imgsize)
        raw_10s_r_img = _resize_np_to_hw(raw_10s_r_img, self.test_imgsize)
        raw_50s_img = _resize_np_to_hw(raw_50s_img, self.test_imgsize)

        eeg_img = np.expand_dims(eeg_img, -1)
        spec_img = np.expand_dims(spec_img, -1)
        raw_50s_img = np.expand_dims(raw_50s_img, -1)
        raw_10s_l_img = np.expand_dims(raw_10s_l_img, -1)
        raw_10s_c_img = np.expand_dims(raw_10s_c_img, -1)
        raw_10s_r_img = np.expand_dims(raw_10s_r_img, -1)

        eps = 1e-6

        spec_img = np.clip(spec_img, np.exp(-4), np.exp(8))
        spec_img = np.log(spec_img)
        spec_img = np.nan_to_num(spec_img, nan=0.0)

        img_mean = spec_img.mean(axis=(0, 1))
        img_std = spec_img.std(axis=(0, 1))
        spec_img = (spec_img - img_mean) / (img_std + eps)

        return (
            spec_img,
            eeg_img,
            raw_50s_img,
            raw_10s_l_img,
            raw_10s_c_img,
            raw_10s_r_img,
            eeg_id,
        )




## === cell 13
class Net(nn.Module):
    def __init__(self, back_bone, device_id=None):
        super().__init__()
        self.spec_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )
        self.eeg_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )
        self.raw_50s_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )
        self.raw_10s_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )

        for m in [
            self.spec_model,
            self.eeg_model,
            self.raw_50s_model,
            self.raw_10s_model,
        ]:
            m.fc_norm = nn.Identity()
            m.head_drop = nn.Identity()
            m.head = nn.Identity()

        self.head = nn.Linear(384 * 4, 6)
        self.head1 = nn.Linear(384, 6)
        self.head2 = nn.Linear(384, 6)
        self.head3 = nn.Linear(384, 6)
        self.head4 = nn.Linear(384, 6)

    def forward(self, spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_imgs):
        spec_imgs = spec_imgs.transpose(1, 2).transpose(1, 3).contiguous()
        eeg_imgs = eeg_imgs.transpose(1, 2).transpose(1, 3).contiguous()
        raw_50s_imgs = raw_50s_imgs.transpose(1, 2).transpose(1, 3).contiguous()
        raw_10s_imgs = raw_10s_imgs.transpose(1, 2).transpose(1, 3).contiguous()

        spec_feature = self.spec_model.forward_features(spec_imgs)[:, 0]
        eeg_feature = self.eeg_model.forward_features(eeg_imgs)[:, 0]
        raw_50s_feature = self.raw_50s_model.forward_features(raw_50s_imgs)[:, 0]
        raw_10s_feature = self.raw_10s_model.forward_features(raw_10s_imgs)[:, 0]

        feature = torch.cat(
            (spec_feature, eeg_feature, raw_50s_feature, raw_10s_feature), 1
        )
        logits = self.head(feature)
        logits_1 = self.head1(spec_feature)
        logits_2 = self.head2(eeg_feature)
        logits_3 = self.head3(raw_50s_feature)
        logits_4 = self.head4(raw_10s_feature)
        return logits, logits_1, logits_2, logits_3, logits_4




## === cell 14
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model_weights = [
    "/kaggle/input/hms-stage2/fold_0_spec_raw_50_10_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_1_spec_raw_50_10_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_2_spec_raw_50_10_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_3_spec_raw_50_10_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_4_spec_raw_50_10_bestlb.pth",
]
backbone = "vit_small_patch14_reg4_dinov2.lvd142m"

vit_models = []
for w in model_weights:
    if os.path.exists(w):
        m = Net(backbone).to(device)
        state = torch.load(w, map_location=device)
        m.load_state_dict(state, strict=True)
        m.eval()
        vit_models.append(m)

print("loaded models:", len(vit_models))



## === cell 15
sub_template = pd.read_csv(SAMPLE_SUB)

train_df_for_prior = pd.read_csv(TRAIN_CSV, usecols=CLASSES)
vote_totals = train_df_for_prior[CLASSES].sum(axis=0).values.astype(np.float64)
vote_totals = np.clip(vote_totals, 1e-12, None)
prior = vote_totals / vote_totals.sum()

TEMPERATURE = 2.20

LOGIT_CENTER = True

test_data = ImageFolder(test, (518, 518))
num_workers = (
    min(4, os.cpu_count() or 2)
    if device.type == "cuda"
    else min(2, os.cpu_count() or 2)
)
test_loader = DataLoader(
    test_data,
    batch_size=8 if device.type == "cpu" else 32,
    pin_memory=(device.type == "cuda"),
    num_workers=num_workers,
    drop_last=False,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)

result = {}

with torch.inference_mode():
    for (
        spec_imgs,
        eeg_imgs,
        raw_50s_imgs,
        raw_10s_l_imgs,
        raw_10s_c_imgs,
        raw_10s_r_imgs,
        eeg_ids,
    ) in test_loader:
        spec_imgs = spec_imgs.to(device=device, dtype=torch.float32, non_blocking=True)
        eeg_imgs = eeg_imgs.to(device=device, dtype=torch.float32, non_blocking=True)
        raw_50s_imgs = raw_50s_imgs.to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        raw_10s_l_imgs = raw_10s_l_imgs.to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        raw_10s_c_imgs = raw_10s_c_imgs.to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        raw_10s_r_imgs = raw_10s_r_imgs.to(
            device=device, dtype=torch.float32, non_blocking=True
        )

        if len(vit_models) == 0:
            probs = (
                torch.tensor(prior, device=device, dtype=torch.float32)
                .unsqueeze(0)
                .repeat(spec_imgs.size(0), 1)
            )
        else:
            ensemble_logits = torch.zeros((spec_imgs.size(0), N_CLASSES), device=device)
            for model in vit_models:
                logits_l, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_l_imgs
                )
                logits_c, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs
                )
                logits_r, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_r_imgs
                )
                ensemble_logits += (logits_l + logits_c + logits_r) / 3.0
            ensemble_logits = ensemble_logits / len(vit_models)

            if LOGIT_CENTER:
                ensemble_logits = ensemble_logits - ensemble_logits.mean(
                    dim=1, keepdim=True
                )

            probs = (ensemble_logits / TEMPERATURE).softmax(dim=1)

        probs = probs.detach().cpu().numpy()

        for j, eeg_id in enumerate(eeg_ids):
            eeg_id_int = int(eeg_id)
            result[eeg_id_int] = probs[j].astype(np.float64)

del vit_models
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("predicted eeg_ids:", len(result))



## === cell 16
sub = sub_template.copy()
pred_mat = np.zeros((len(sub), N_CLASSES), dtype=np.float64)

PRIOR_DIRICHLET_EPS = 0.18

MIN_PROB_CLIP = 1e-8

for i, eeg_id in enumerate(sub["eeg_id"].values):
    eeg_id_int = int(eeg_id)
    p = result.get(eeg_id_int, prior).copy()

    p = np.clip(p, MIN_PROB_CLIP, 1.0)
    p = p / p.sum()

    p = (1.0 - PRIOR_DIRICHLET_EPS) * p + PRIOR_DIRICHLET_EPS * prior
    p = np.clip(p, MIN_PROB_CLIP, 1.0)
    p = p / p.sum()

    pred_mat[i] = p

for k, c in enumerate(CLASSES):
    sub[c] = pred_mat[:, k]

row_sums = sub[CLASSES].sum(axis=1).values
if not np.allclose(row_sums, 1.0, atol=1e-6):
    sub[CLASSES] = sub[CLASSES].div(sub[CLASSES].sum(axis=1), axis=0)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("wrote submission.csv with shape:", sub.shape)



## === cell 17
if not DEBUG:
    for p in [
        spec_directory_path,
        eeg_directory_path,
        raw_50s_directory_path,
        raw_10s_directory_path,
    ]:
        try:
            pass
        except Exception:
            pass
