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

0.2876283318496322

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the runtime failure by making the device selection robust (CPU fallback when no NVIDIA driver is available) and by loading model weights safely onto the chosen device. I also remove the notebook-only shell `!rm` commands (they crash in a `.py` run) and replace them with equivalent Python cleanup so the pipeline runs end-to-end in Kaggle’s script environment. Finally, I ensure we always produce a valid `submission.csv` with probabilities that sum to 1 per row (including a safe fallback to the sample submission distribution if inference cannot run due to missing weights). These changes are execution/stability fixes and keep the model/inference logic the same when the weights and environment support it.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.2876), so we need a real inference improvement with minimal semantic changes. The biggest issue is that your script is generating EEG STFT features twice with two different shapes, and in cell 12 you delete the first set right after creating it—this wastes time and risks feeding mismatched features vs what the weights expect. I remove the redundant first feature-generation pass and keep only the later STFT function (the one that matches the model’s expected input), while still generating spec + raw 10s/50s exactly as before. I also make the random seed setup actually deterministic (your current `deterministic=True` + `benchmark=True` conflicts) to stabilize the score and avoid run-to-run variance.'
- What this solution (achieved 1.40995) has done: 'Your current KL score (1.40995, lower-is-better) is far worse than the target (0.2876), and the main likely cause is that the inference feature pipeline doesn’t match what the provided fold weights expect (so the model outputs are poorly calibrated/incorrect). I make the smallest “metric-aligned” fix: aggregate predictions at the **eeg_id level** (since the competition evaluates per eeg_id) and apply a tiny, safe probability floor + renormalization (KL is very sensitive to zeros). I also ensure the input sizes and normalization stay identical, but I fix one subtle alignment issue: `eeg_id` keys should be consistently treated as strings everywhere to avoid missing-lookups and falling back to uniform predictions. These changes keep your architecture and inference loops intact and should move the score substantially toward the target without changing the modeling approach.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far worse than the target (0.2876), so we should make small, metric-aligned inference fixes rather than changing the model. The biggest likely score killer here is inconsistent test-time normalization: you standardize only the spectrogram input, while the EEG-STFT and raw EEG inputs are left on mismatched scales vs what the pretrained fold weights typically expect, which can severely distort softmax probabilities and inflate KL. I add the same kind of safe per-sample standardization (mean/std with eps) to the other three inputs (eeg_img, raw_50s, raw_10s) while preserving shapes and the rest of your pipeline. I also add a tiny probability “prior mix” with the sample_submission distribution (a common, legitimate KL stabilizer) to reduce overconfident wrong predictions without changing architecture or weights.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.2876), so we need a small but meaningful inference-side correction without changing the model architecture or overall pipeline. The biggest low-risk gain is to match the fold weights’ expected spectrogram preprocessing: the saved spectrogram arrays are currently raw/linear and then you take `log()` of clipped values, which is inconsistent and can severely distort inputs; we instead save `log1p` spectrograms (and then keep dataset normalization consistent by removing the extra `log()` step). Next, we also standardize the saved EEG spectrogram (STFT) the same way as the other inputs already are (mean/std per sample), which is metric-aligned for KL and doesn’t alter model structure. Finally, we keep your prior-mixing, but make it slightly stronger (still small) to reduce overconfident wrong predictions, which typically improves KL without changing the model.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far worse than the target (0.2876), so we should make small, metric-aligned inference fixes without changing the model or training approach. The biggest low-risk improvement is to calibrate the ensemble probabilities (KL is very sensitive to overconfident wrong predictions) by applying a mild temperature scaling and slightly stronger prior-mixing, while keeping outputs valid probabilities that sum to 1. I also ensure we average the three 10s crops (l/c/r) and folds exactly as before, but add a tiny epsilon-floor + renormalization right after averaging to avoid numerically tiny probabilities hurting KL. These changes keep architecture/weights/feature extraction intact and should move the score meaningfully toward the target without “over-optimizing”.'
- What this solution (achieved 1.40995) has done: 'We need to move KL down from 1.40995 toward 0.2876 (lower is better), so the smallest likely win is to fix an input mismatch that makes the pretrained fold weights behave incorrectly rather than changing the model. Your `raw10seeg_from_eeg()` currently returns arrays shaped like `(200, 16*EEG_LENGTH)` (after concatenation) which are then resized into an image; most HMS ViT solutions expect the 10s raw input as a true 2D “image” `(200, 16)` (time x channels) per crop, not stretched by EEG_LENGTH. I minimally correct the raw-10s construction to keep **exactly the same channels, filtering, clipping, and crop timing**, but output `(200, 16)` by stacking the 4 differential channels across the 4 regions (LL/RL/LP/RP) instead of concatenating time segments. I keep your inference/ensemble/temperature/prior-mix unchanged so the core logic remains intact, and still write a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'We need to move KL down from 1.40995 toward 0.2876 (lower is better), so the smallest high-impact fix is to ensure the inference inputs match what the pretrained fold weights were trained on. Your `raw10seeg_from_eeg()` was changed to output `(time, channels)=(2000,16)` but the rest of your pipeline (notably `resize(...)` and ViT “image” expectations) is much more consistent with the original “stitched 200x(16*10)” representation used by many HMS public weights. I revert only the 10s-raw construction back to the stitched layout (while keeping your other improvements: log1p spec, per-sample standardization, temperature scaling, epsilon floor, eeg_id aggregation, and prior-mixing), which should materially reduce the mismatch-driven overconfident errors that blow up KL. No model architecture, loss, or ensembling logic is changed; we just restore the expected raw-10s feature shape.'
- What this solution (achieved 1.40995) has done: 'We need to move KL down from 1.40995 toward 0.2876 (lower is better), so we should fix the most likely silent inference mismatch rather than changing the model. The biggest risk here is that `resize()` is currently returning `float64` by default, and then you standardize and pass these into ViT; this can subtly but materially change the activation/statistics compared to what the fold weights expect (and also wastes memory/bandwidth). I make the resizing and post-processing explicitly `float32` (no semantic change, just dtype consistency), and I also enforce a consistent per-sample normalization after resize on all inputs in a numerically stable way (still the same mean/std standardization you already do). Finally, I keep your temperature/prior-mix logic intact, but add a very small extra probability floor right before writing submission to avoid any extremely tiny values that can spike KL on edge cases.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far worse than the target (0.2876), so the most likely issue is a silent mismatch between the test-time raw-10s feature layout and what the provided fold weights were trained on. I make one minimal, score-relevant correction: build the 10s raw “stitched image” by chunking the 2000-sample window into 10 blocks of 200 samples per channel (so the final shape is 200×160), instead of relying on a reshape that unintentionally interleaves time. This preserves the exact same channels, filtering, clipping, crop timing, resizing, model, and ensembling logic, but aligns the raw-10s construction with the typical HMS public-weight expectation. Everything else (standardization, temperature, prior-mix, epsilon-floor, submission writing) is kept the same to avoid unnecessary score variance.'
- What this solution (achieved 1.40995) has done: 'We need to move KL down from 1.40995 toward 0.2876 (lower is better), so the smallest high-impact change is to reduce any silent input mismatch between your saved features and what the pretrained fold weights expect. I (1) make the raw-50s construction use the same correct “stitched by 1-second blocks” logic already fixed for raw-10s (your current `reshape(4,200,EEG_LENGTH)` interleaves time and likely breaks the pretrained model’s assumptions), and (2) ensure the raw-10s crop windows are exactly 10 seconds long and not truncated to 9 seconds by a fencepost bug in `_crop_to_stitched`. These are minimal, score-relevant fixes that keep your architecture/inference/ensembling/temperature/prior-mix intact while aligning feature layout and duration, which should materially reduce KL. Everything else (paths, submission format, probability normalization, cleanup) remains unchanged.'
- What this solution (achieved 1.40995) has done: 'We need to move KL down from 1.40995 toward 0.2876 (lower is better), so the smallest likely gain is to fix a silent preprocessing mismatch rather than changing the model or ensembling. Your `raw50seeg_from_eeg()` currently returns a width of 800 (four regions concatenated), but most public HMS ViT pipelines (and corresponding fold weights) use the same “stitched image” width of 200 by averaging the four regional images instead of concatenating them, so I change raw-50s to return (200,200) by averaging regions (keeping the exact same channels/filtering/clipping/stitching). I also make the region naming consistent (use `RL` instead of `RR`) to avoid accidental ordering drift, while keeping your raw-10s logic untouched. Finally, I keep your temperature scaling, prior-mix, epsilon floor, and submission formatting exactly the same.'
- What this solution (achieved 1.40995) has done: 'We need to move KL down substantially (1.40995 → 0.2876; lower is better), so the smallest high-impact fix is to correct a likely silent input-mismatch rather than changing the model or ensembling. Your `raw50seeg_from_eeg()` currently averages the 4 regional stitched images into (200,200), but this is inconsistent with how the rest of the pipeline treats raw inputs (and with many fold-weight pipelines that expect a wider “stitched” raw-50s image that is later resized to the model’s square input). I revert raw-50s back to the original concatenation of the 4 regions into (200,800) while keeping the exact same channels, filtering, clipping, and stitching; this preserves core logic but aligns the raw-50s geometry with the downstream `resize(...)` step and typical pretrained expectations. Everything else (log1p spec, standardization, temperature scaling, prior-mix, probability flooring, eeg_id aggregation, and submission writing) stays unchanged.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995; lower is better) is far from the target (0.2876), so we need a small inference-side fix that corrects a likely pretrained-weights input mismatch rather than changing the model. The biggest remaining mismatch risk is **the channel/region ordering**: `RAW_FEATS` is defined in `LL, RL, LP, RP` order, while `NAMES`/most pipelines use `LL, LP, RP, RL`; since you iterate `RAW_FEATS.keys()` for both raw-10s and raw-50s, this silently permutes regions and can badly degrade logits. I make `RAW_FEATS` an `OrderedDict` in the `NAMES` order and iterate by `NAMES` explicitly (core logic identical: same signals, filtering, clipping, stitching; only consistent ordering). I also slightly increase the probability prior-mix `alpha` (0.14 → 0.20) as a minimal KL-calibration stabilizer to reduce overconfident wrong predictions without changing architecture or ensembling.'

# 9. Code solution

## === cell 0
import os
import gc
import random
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")



## === cell 1
DEBUG = False



## === cell 2
NAMES = ["LL", "LP", "RP", "RL"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]

import librosa  # noqa: F401



## === cell 3
from scipy.signal import butter, lfilter  # noqa: F401
from scipy import signal



## === cell 4
NAMES = ["LL", "LP", "RP", "RL"]
SFREQ = 200
filter_range = [0.5, 40]

RAW_FEATS = OrderedDict(
    [
        ("LL", ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"]),
        ("LP", ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"]),
        ("RP", ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"]),
        ("RL", ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"]),
    ]
)

b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")


def raw10seeg_from_eeg(parquet_path, eeg_id):
    """
    Keep stitched 10s raw construction, but ensure region iteration order matches NAMES.
    """
    EEG_LENGTH = 10  # number of 1-second blocks; 10s crop -> 2000 samples at 200Hz
    raw_eeg = pd.read_parquet(parquet_path)

    time_temp = 0

    def _crop_to_stitched(time_start, time_stop):
        eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(
            drop=True
        )
        n_samples = eeg_default.shape[0]  # expected 2000

        target = EEG_LENGTH * 200  # 2000
        if n_samples >= target:
            eeg_default = eeg_default.iloc[:target].reset_index(drop=True)
            n_samples = target

        block = 200  # 1-second blocks
        n_blocks = n_samples // block  # expected 10

        list_eeg = []
        for region in NAMES:
            chans = RAW_FEATS[region]
            eeg = np.zeros((len(chans), n_samples), dtype=np.float32)
            for chan_i, chan in enumerate(chans):
                eeg_1 = eeg_default.loc[:, chan.split("-")[0]]
                mean_value = eeg_1.mean()
                eeg_1.fillna(value=mean_value, inplace=True)
                eeg_1 = eeg_1.values

                eeg_2 = eeg_default.loc[:, chan.split("-")[1]]
                mean_value = eeg_2.mean()
                eeg_2.fillna(value=mean_value, inplace=True)
                eeg_2 = eeg_2.values

                new_eeg = eeg_1 - eeg_2
                del eeg_1
                del eeg_2
                new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
                new_eeg = np.clip(new_eeg, -1024, 1024).astype("float32")
                eeg[chan_i, :] = new_eeg

            eeg = eeg.reshape(4, n_blocks, block)
            eeg = np.transpose(eeg, (0, 2, 1))
            eeg = np.concatenate([eeg[i, :, :] for i in range(4)], axis=1)
            list_eeg.append(eeg)

        eeg = np.concatenate(list_eeg, 1)  # (200, 160)
        eeg /= 104.0
        return eeg.astype(np.float32)

    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)
    eeg_c = _crop_to_stitched(time_start, time_stop)

    time_start = round(time_temp + 18 * 200)
    time_stop = round(time_temp + 28 * 200)
    eeg_l = _crop_to_stitched(time_start, time_stop)

    time_start = round(time_temp + 22 * 200)
    time_stop = round(time_temp + 32 * 200)
    eeg_r = _crop_to_stitched(time_start, time_stop)

    return eeg_l, eeg_c, eeg_r


def raw50seeg_from_eeg(parquet_path):
    """
    Keep concatenated (200, 800) stitched geometry, but ensure region iteration order matches NAMES.
    """
    EEG_LENGTH = 50
    raw_eeg = pd.read_parquet(parquet_path)

    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)

    eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(drop=True)

    target = EEG_LENGTH * 200  # 10000
    if eeg_default.shape[0] >= target:
        eeg_default = eeg_default.iloc[:target].reset_index(drop=True)

    n_samples = eeg_default.shape[0]
    block = 200
    n_blocks = n_samples // block  # expected 50

    list_eeg = []
    for region in NAMES:
        chans = RAW_FEATS[region]
        eeg = np.zeros((len(chans), n_samples), dtype=np.float32)
        for chan_i, chan in enumerate(chans):
            eeg_1 = eeg_default.loc[:, chan.split("-")[0]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values

            eeg_2 = eeg_default.loc[:, chan.split("-")[1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values

            new_eeg = eeg_1 - eeg_2
            del eeg_1
            del eeg_2
            new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            new_eeg = np.clip(new_eeg, -1024, 1024).astype("float32")
            eeg[chan_i, :] = new_eeg

        eeg = eeg.reshape(4, n_blocks, block)  # (4, 50, 200)
        eeg = np.transpose(eeg, (0, 2, 1))  # (4, 200, 50)
        eeg = np.concatenate([eeg[i, :, :] for i in range(4)], axis=1)  # (200, 200)
        list_eeg.append(eeg.astype(np.float32))

    eeg = np.concatenate(list_eeg, 1)  # (200, 800)
    eeg /= 104.0
    return eeg.astype(np.float32)




## === cell 5
CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
N_CLASSES = len(CLASSES)

if DEBUG is True:
    test = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )[:40]
    SPEC_PATH = (
        "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    )
    EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
else:
    test = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    )
    SPEC_PATH = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

test["eeg_id"] = test["eeg_id"].astype(str)
test["spectrogram_id"] = test["spectrogram_id"].astype(str)

print(test.shape)

spec_directory_path = "spec_spectrograms/"
os.makedirs(spec_directory_path, exist_ok=True)

eeg_directory_path = "eeg_spectrograms/"
os.makedirs(eeg_directory_path, exist_ok=True)

raw_10s_directory_path = "eeg_10s_raws/"
os.makedirs(raw_10s_directory_path, exist_ok=True)

raw_50s_directory_path = "eeg_50s_raws/"
os.makedirs(raw_50s_directory_path, exist_ok=True)

from joblib import Parallel, delayed

EEG_IDS = test.eeg_id.unique()




## === cell 6
class Config:
    seed = 2024
    num_folds = 5




## === cell 7
def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(Config.seed)



## === cell 8
import timm



## === cell 9
import torch.utils.data as data  # noqa: F401
from torch.utils.data import DataLoader



## === cell 10
from skimage.transform import resize




## === cell 11
def _rm_tree_or_files(path):
    if not os.path.exists(path):
        return
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass
        return
    for root, dirs, files in os.walk(path, topdown=False):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                os.remove(fp)
            except OSError:
                pass
        for dn in dirs:
            dp = os.path.join(root, dn)
            try:
                os.rmdir(dp)
            except OSError:
                pass


os.makedirs("eeg_spectrograms/", exist_ok=True)




## === cell 12
def stft_spec_from_eeg(parquet_path):
    EEG_LENGTH = 50
    eeg = pd.read_parquet(parquet_path)

    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)

    eeg = eeg.iloc[time_start:time_stop]

    list_eeg = []
    for k in range(4):
        COLS = FEATS[k]
        img = np.zeros((128, 256, 4), dtype="float32")
        for kk in range(4):
            eeg_1 = eeg[COLS[kk]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values

            eeg_2 = eeg[COLS[kk + 1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values

            new_eeg = eeg_1 - eeg_2
            del eeg_1
            del eeg_2
            fs = 200
            nperseg = 39
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
    return img.astype(np.float32)




## === cell 13
def save(row):
    eeg_id = row["eeg_id"]
    spec_id = row["spectrogram_id"]

    spec = pd.read_parquet(f"{SPEC_PATH}{spec_id}.parquet")
    spec_arr = spec.values[:, 1:].T.astype("float32")  # (Hz, Time) = (400, 300)
    split_spec_arr = spec_arr[:, 0:300]
    split_spec_arr = np.nan_to_num(split_spec_arr, nan=0.0)
    split_spec_arr = np.log1p(np.clip(split_spec_arr, 0.0, None)).astype("float32")
    np.save(f"{spec_directory_path}{eeg_id}", split_spec_arr)

    img_l, img_c, img_r = raw10seeg_from_eeg(f"{EEG_PATH}{eeg_id}.parquet", eeg_id)
    np.save(f"{raw_10s_directory_path}{eeg_id}_l", img_l)
    np.save(f"{raw_10s_directory_path}{eeg_id}_c", img_c)
    np.save(f"{raw_10s_directory_path}{eeg_id}_r", img_r)

    img50 = raw50seeg_from_eeg(f"{EEG_PATH}{eeg_id}.parquet")
    np.save(f"{raw_50s_directory_path}{eeg_id}", img50)

    img_stft = stft_spec_from_eeg(f"{EEG_PATH}{eeg_id}.parquet")
    np.save(f"{eeg_directory_path}{eeg_id}", img_stft)


_ = Parallel(n_jobs=4)(delayed(save)(row) for _, row in test.iterrows())




## === cell 14
class ImageFolder(data.Dataset):
    def __init__(self, df, test_imgsize):
        super().__init__()
        df["eeg_id"] = df["eeg_id"]
        self.spec_data_path = spec_directory_path
        self.eeg_data_path = eeg_directory_path
        self.raw_50s_data_path = raw_50s_directory_path
        self.raw_10s_data_path = raw_10s_directory_path
        self.df = df.reset_index(drop=True)
        self.test_imgsize = test_imgsize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        eeg_id = str(row.eeg_id)
        spec_image_path = os.path.join(self.spec_data_path, eeg_id + ".npy")
        eeg_image_path = os.path.join(self.eeg_data_path, eeg_id + ".npy")
        raw_50s_image_path = os.path.join(self.raw_50s_data_path, eeg_id + ".npy")
        raw_10s_l_image_path = os.path.join(self.raw_10s_data_path, eeg_id + "_l.npy")
        raw_10s_c_image_path = os.path.join(self.raw_10s_data_path, eeg_id + "_c.npy")
        raw_10s_r_image_path = os.path.join(self.raw_10s_data_path, eeg_id + "_r.npy")

        spec_img = np.load(spec_image_path).astype("float32")
        raw_50s_img = np.load(raw_50s_image_path).astype("float32")
        raw_10s_l_img = np.load(raw_10s_l_image_path).astype("float32")
        raw_10s_c_img = np.load(raw_10s_c_image_path).astype("float32")
        raw_10s_r_img = np.load(raw_10s_r_image_path).astype("float32")
        eeg_img = np.load(eeg_image_path).astype("float32")

        spec_img = resize(
            spec_img, self.test_imgsize, preserve_range=True, anti_aliasing=True
        ).astype("float32")
        raw_10s_l_img = resize(
            raw_10s_l_img, self.test_imgsize, preserve_range=True, anti_aliasing=True
        ).astype("float32")
        raw_10s_c_img = resize(
            raw_10s_c_img, self.test_imgsize, preserve_range=True, anti_aliasing=True
        ).astype("float32")
        raw_10s_r_img = resize(
            raw_10s_r_img, self.test_imgsize, preserve_range=True, anti_aliasing=True
        ).astype("float32")
        raw_50s_img = resize(
            raw_50s_img, self.test_imgsize, preserve_range=True, anti_aliasing=True
        ).astype("float32")

        eeg_img = np.expand_dims(eeg_img, -1).astype("float32")
        spec_img = np.expand_dims(spec_img, -1).astype("float32")
        raw_50s_img = np.expand_dims(raw_50s_img, -1).astype("float32")
        raw_10s_l_img = np.expand_dims(raw_10s_l_img, -1).astype("float32")
        raw_10s_c_img = np.expand_dims(raw_10s_c_img, -1).astype("float32")
        raw_10s_r_img = np.expand_dims(raw_10s_r_img, -1).astype("float32")

        eps = 1e-6

        def _standardize(x):
            x = np.nan_to_num(x, nan=0.0).astype("float32")
            m = x.mean(axis=(0, 1), dtype=np.float32)
            s = x.std(axis=(0, 1), dtype=np.float32)
            return ((x - m) / (s + eps)).astype("float32")

        spec_img = _standardize(spec_img)
        eeg_img = _standardize(eeg_img)
        raw_50s_img = _standardize(raw_50s_img)
        raw_10s_l_img = _standardize(raw_10s_l_img)
        raw_10s_c_img = _standardize(raw_10s_c_img)
        raw_10s_r_img = _standardize(raw_10s_r_img)

        return (
            spec_img,
            eeg_img,
            raw_50s_img,
            raw_10s_l_img,
            raw_10s_c_img,
            raw_10s_r_img,
            eeg_id,
        )




## === cell 15
class Net(nn.Module):
    def __init__(self, back_bone, device_id):
        super().__init__()
        self.spec_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )
        self.eeg_model = timm.create_model(
            back_bone,
            num_classes=6,
            pretrained=False,
            in_chans=1,
            dynamic_img_pad=True,
            dynamic_img_size=True,
        )
        self.raw_50s_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )
        self.raw_10s_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )

        self.device_id = device_id

        self.spec_model.fc_norm = nn.Identity()
        self.spec_model.head_drop = nn.Identity()
        self.spec_model.head = nn.Identity()

        self.eeg_model.fc_norm = nn.Identity()
        self.eeg_model.head_drop = nn.Identity()
        self.eeg_model.head = nn.Identity()

        self.raw_50s_model.fc_norm = nn.Identity()
        self.raw_50s_model.head_drop = nn.Identity()
        self.raw_50s_model.head = nn.Identity()

        self.raw_10s_model.fc_norm = nn.Identity()
        self.raw_10s_model.head_drop = nn.Identity()
        self.raw_10s_model.head = nn.Identity()

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




## === cell 16
device = "cuda:0" if torch.cuda.is_available() else "cpu"

vit_models = []
model_weights = [
    "/kaggle/input/hms-stage2/fold_0_exp_3_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_1_exp_3_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_2_exp_3_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_3_exp_3_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_4_exp_3_bestlb.pth",
]
model_types = ["vit_small", "vit_small", "vit_small", "vit_small", "vit_small"]

weights_available = all(os.path.exists(p) for p in model_weights)

if weights_available:
    for i in range(len(model_types)):
        if model_types[i] == "vit_small":
            model = Net("vit_small_patch14_reg4_dinov2.lvd142m", device).to(device)
            state = torch.load(model_weights[i], map_location=device)
            model.load_state_dict(state, strict=True)
            model.eval()
            vit_models.append(model)

result_sum = {}
result_cnt = {}

TEMPERATURE = 1.35  # keep unchanged

if weights_available:
    test_data = ImageFolder(test, (518, 518))
    test_loader = DataLoader(
        test_data, batch_size=32, pin_memory=False, num_workers=4, drop_last=False
    )

    with torch.no_grad():
        for (
            spec_imgs,
            eeg_imgs,
            raw_50s_imgs,
            raw_10s_l_imgs,
            raw_10s_c_imgs,
            raw_10s_r_imgs,
            eeg_ids,
        ) in test_loader:
            spec_imgs = spec_imgs.to(device).float()
            eeg_imgs = eeg_imgs.to(device).float()
            raw_50s_imgs = raw_50s_imgs.to(device).float()
            raw_10s_l_imgs = raw_10s_l_imgs.to(device).float()
            raw_10s_c_imgs = raw_10s_c_imgs.to(device).float()
            raw_10s_r_imgs = raw_10s_r_imgs.to(device).float()

            ensemble_probs = torch.zeros((spec_imgs.shape[0], 6), device=device)
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

                probs_l = (logits_l / TEMPERATURE).softmax(dim=1)
                probs_c = (logits_c / TEMPERATURE).softmax(dim=1)
                probs_r = (logits_r / TEMPERATURE).softmax(dim=1)
                ensemble_probs += (probs_l + probs_c + probs_r) / 3.0

            ensemble_probs /= len(vit_models)

            ensemble_probs = torch.clamp(ensemble_probs, min=1e-7)
            ensemble_probs = ensemble_probs / ensemble_probs.sum(dim=1, keepdim=True)

            ensemble_probs = ensemble_probs.detach().cpu().numpy()

            for j in range(len(eeg_ids)):
                eeg_id = str(eeg_ids[j])
                if eeg_id not in result_sum:
                    result_sum[eeg_id] = np.zeros(6, dtype=np.float32)
                    result_cnt[eeg_id] = 0
                result_sum[eeg_id] += ensemble_probs[j].astype(np.float32)
                result_cnt[eeg_id] += 1



## === cell 17
for model in vit_models:
    del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()



## === cell 18
sample_sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
sample_sub["eeg_id"] = sample_sub["eeg_id"].astype(str)

if not weights_available or len(result_sum) == 0:
    df = sample_sub.copy()
else:
    preds = np.zeros((len(sample_sub), 6), dtype=np.float32)
    for i, eeg_id in enumerate(sample_sub["eeg_id"].astype(str).values):
        s = result_sum.get(eeg_id, None)
        c = result_cnt.get(eeg_id, 0)
        if s is None or c == 0:
            preds[i] = np.array([1 / 6] * 6, dtype=np.float32)
        else:
            preds[i] = (s / float(c)).astype(np.float32)

    prior = sample_sub[CLASSES].mean(axis=0).values.astype(np.float32)
    prior = np.clip(prior, 1e-6, None)
    prior = prior / prior.sum()

    alpha = 0.20
    preds = (1.0 - alpha) * preds + alpha * prior[None, :]

    preds = np.clip(preds, 1e-6, None)
    preds = preds / preds.sum(axis=1, keepdims=True)

    df = pd.DataFrame(preds, columns=CLASSES)
    df.insert(0, "eeg_id", sample_sub["eeg_id"].values)

df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print(
    "Row-wise sum stats:", df[CLASSES].sum(axis=1).min(), df[CLASSES].sum(axis=1).max()
)



## === cell 19
if DEBUG is False:
    _rm_tree_or_files("/kaggle/working/spec_spectrograms/")
    _rm_tree_or_files("/kaggle/working/eeg_spectrograms/")
    _rm_tree_or_files("/kaggle/working/eeg_50s_raws/")
    _rm_tree_or_files("/kaggle/working/eeg_10s_raws/")
    _rm_tree_or_files("/kaggle/working/squeezeformer")
