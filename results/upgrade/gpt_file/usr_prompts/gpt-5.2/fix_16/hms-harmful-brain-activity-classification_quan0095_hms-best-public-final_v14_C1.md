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

0.2875501042577512

# 6. Current score

1.4856

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime failure by making the model-weight loading robust: if the specified `/kaggle/input/hms-stage2-vitlarge/*.pth` files are not present, the script fall back to a valid, well-formed probability submission (uniform class probabilities) so you always get a `submission.csv`. I also fix the downstream `NameError` by ensuring `result_8` is always defined, even when no models are loaded. Additionally, I correct a small determinism setting (benchmark vs deterministic) and make the ensemble aggregation key consistently a string to avoid missing predictions due to dtype mismatches. These changes keep the core model/feature pipeline intact when weights exist, while guaranteeing end-to-end execution and a valid CSV in this Kaggle environment.'
- What this solution (achieved 1.39779) has done: 'Your current score is far worse than the target (lower is better), and the biggest cause is that the code falls back to uniform probabilities because the referenced weight files are not available. To move the score substantially toward the target without changing the model/feature core logic, I add a lightweight “prior calibration” fallback that uses the training label distribution (computed from `train.csv`) instead of uniform predictions when no weights are found. This keeps evaluation semantics valid (probabilities that sum to 1) and is a minimal change that typically beats uniform for KL on this competition. I also keep the original ensemble path unchanged when weights exist, and ensure numeric stability/normalization remains correct.'
- What this solution (achieved 1.43453) has done: 'Your current score (1.39779, lower is better) is far from the target (0.28755), and the main limiter is that you’re still effectively using a “global prior” fallback when weights aren’t present, which is much worse than even a simple per-patient prior on this dataset. I keep your entire feature extraction and model path unchanged, but improve only the fallback behavior by computing a patient-conditional prior from `train.csv` and using that for each test `patient_id` (with a safe global prior fallback for unseen patients). This remains a legitimate probability baseline (rows sum to 1) and typically reduces KL substantially versus a single global prior, moving the score toward the target without touching the model architecture/training logic. I also make sure `result_8` is filled for every `eeg_id` in fallback mode to avoid any alignment issues and keep the submission format identical.'
- What this solution (achieved 0.96457) has done: 'Your score is far worse than the target (lower is better), and the main reason is still that no weight files are found so you’re submitting a simple patient-conditional prior; we can move closer to the target by making that fallback prior materially better without touching your model/feature pipeline. I keep your entire EEG/spectrogram feature extraction and model code unchanged, but improve fallback predictions by (1) aggregating training labels at the EEG level (since the metric is per test `eeg_id`, and train has many overlapping subsamples per `eeg_id`) and (2) using a smoothed patient-level prior computed from those per-EEG distributions with a Dirichlet-style blend toward the global prior for stability. This is a minimal, legitimate adjustment that usually reduces KL substantially versus averaging per-row subsamples. Submission writing stays identical, still guaranteeing valid probabilities summing to 1.'
- What this solution (achieved 0.96457) has done: 'Your current score (0.96457, lower is better) is still far from the target (0.28755), so we should improve the fallback path (since no weights are found) with the smallest legitimate change. I keep your model/feature pipeline untouched and only make the fallback prior more informative by conditioning not just on `patient_id` but also on `spectrogram_id` (which clusters recording/context) using smoothed per-EEG label distributions. To avoid overfitting/noise, I blend spectrogram-conditional and patient-conditional priors with data-driven weights and a global prior backstop, preserving valid probabilities that sum to 1. This should reduce KL versus patient-only prior while remaining stable and fast (<600s) because it only reads `train.csv`.'
- What this solution (achieved 1.07853) has done: 'Your current score (0.96457, lower is better) is still far above the target (0.28755), and since no weight files are available, the only path to improve is the fallback. I keep your entire feature extraction/model code intact and only adjust the fallback prior computation to better match the evaluation granularity by (1) de-duplicating train labels to one distribution per `eeg_id` using `label_id` (avoids overweighting overlapping windows), and (2) adding a patient+spectrogram “mixture-of-experts” prior with a global backstop and slightly stronger smoothing to reduce overconfident wrong predictions (KL is harsh on near-zeros). This is a minimal, fast change (only reads `train.csv`) and preserves valid probability constraints (sum-to-1, no zeros). The submission writing/path remains unchanged and still guarantees a valid `submission.csv`.'
- What this solution (achieved 1.2772) has done: 'Your current score (1.07853, lower is better) is still far above the target (0.28755), and because no model weights are available the only lever is improving the fallback probabilities while keeping your model/feature pipeline unchanged. I keep all feature extraction and the inference path intact, but change the fallback to use a stronger, still-legitimate prior: a smoothed conditional distribution based on the patient’s *training class distribution* plus a global prior, with an additional “patient-context” term built from patient × discretized recording index (via spectrogram_id bucketing) to capture within-patient heterogeneity without exploding cardinality. I also ensure the fallback is evaluated at the correct granularity by first aggregating training rows to one distribution per (eeg_id,label_id) then to eeg_id, then building priors from those per-eeg distributions (as you already intended), and finally blending components with data-driven weights and safe clipping to avoid near-zero probabilities (KL is harsh). This is a minimal change localized to the fallback block and should move the score materially toward the target while remaining fast (only reads train.csv once) and still produces a valid submission.csv.'
- What this solution (achieved 1.42499) has done: 'Your score is far above the target (lower is better) and, since no weights are found, the only lever is the fallback probabilities. I keep your entire model/feature pipeline intact and only strengthen the fallback by (1) using a patient×(spectrogram bucket) conditional prior plus a pure spectrogram-bucket prior (not tied to patient) and (2) blending them with the patient prior and global prior using count-based weights with Dirichlet smoothing. This remains a legitimate, fast baseline (only reads `train.csv`) and should reduce KL versus the current fallback without changing any inference semantics when weights are present. I also make the spectrogram bucket definition more informative (log-scale bucketing) while staying minimal and deterministic.'
- What this solution (achieved 1.2581) has done: 'Your current score (1.42499, lower is better) is far above the target (0.28755), and given the missing weights the only lever is improving the fallback probabilities while keeping your model/inference core intact. I keep the entire model/feature code unchanged and only replace the fallback prior with a more evaluation-aligned estimate: compute the label distribution at the `(patient_id, spectrogram_id)` level (exact match, not bucketing) from de-duplicated EEG-level targets, then smoothly back off to patient-only and global priors when counts are low. I also add conservative Dirichlet smoothing + probability floor to avoid near-zero probabilities (KL is very harsh), and keep submission formatting/normalization identical. This is fast (reads only `train.csv`) and minimal, and should move the score materially toward the target compared to the current bucketed-mixture fallback.'
- What this solution (achieved 1.35605) has done: 'Your current score (1.2581, lower-is-better) is still far above the target (0.28755), and since the weight files are missing the only lever is the fallback prior. I keep the entire feature extraction + model inference path unchanged, and only improve the fallback by (1) learning a smoothed conditional prior at the exact `spectrogram_id` level (strong contextual signal) and (2) blending it with the existing `(patient_id, spectrogram_id)` and `patient_id` priors plus the global prior using count-based backoff. I also ensure the fallback is computed from de-duplicated `(eeg_id, label_id)` rows (as you already do) and add safe clipping so probabilities never hit near-zero (KL is harsh). This should move the score toward the target while remaining fast (only reads `train.csv`) and still always writes a valid `submission.csv`.'
- What this solution (achieved 1.08708) has done: 'Your current score is much worse than the target (lower is better), and since no model weights are found the submission is entirely determined by the fallback priors. I keep the whole feature extraction + model code untouched, but improve only the fallback by conditioning on `patient_id` *and* the test-context identifiers `spectrogram_id` and `eeg_id` more safely: (1) compute priors from de-duplicated `(eeg_id,label_id)` rows aggregated to per-`eeg_id`, then (2) build a stronger hierarchical backoff `(patient_id, spectrogram_id)` → `patient_id` → `spectrogram_id` → global using Dirichlet-style smoothing in “vote-count space” (not averaging already-normalized distributions). Finally, I add a tiny uniform mix and a probability floor to prevent near-zeros (KL is harsh), while still keeping row sums to 1 and preserving identical semantics when weights exist.'
- What this solution (achieved 1.41912) has done: 'Your current score (1.08708, lower-is-better) is still far above the target, and because the specified weights aren’t available the submission is entirely driven by the fallback priors. I keep your full model/feature pipeline unchanged and only adjust the fallback prior computation to better match the KL metric by (1) computing priors in vote-count space with stronger, safer smoothing and (2) adding a small “temperature” flattening step to reduce overconfident wrong class probabilities (KL punishes near-zeros heavily). I also make the test-time combination weights more conservative (more backoff to global) so the fallback generalizes better, while still guaranteeing each row sums to 1 and producing a valid `submission.csv`. No changes are made to the inference path when weights exist.'
- What this solution (achieved 1.43987) has done: 'Your score is far above the target (lower is better) and the log shows you’re still in “no weights found” mode, so the only lever (without changing the core model/inference logic) is improving the fallback probabilities. I keep your existing hierarchical `(patient_id, spectrogram_id) -> patient_id -> spectrogram_id -> global` structure, but fix a key bug: the fallback currently builds priors from `train.spectrogram_id`, which is a *subwindow id* and doesn’t match the test `spectrogram_id` (full recording id), making the spectrogram conditioning mostly useless. I minimally switch fallback aggregation to use the correct “full spectrogram id” by joining `train.csv` to the unique `(eeg_id -> spectrogram_id)` mapping from `train.csv` itself, and I also compute/condition on an `eeg_id`-level prior (from train) to help when test contains seen `eeg_id`s. This should materially reduce KL versus the current mismatched-ID fallback while keeping runtime fast and preserving the model path unchanged when weights exist.'
- What this solution (achieved 1.4856) has done: 'Your current score is far above the target (lower is better) and logs indicate you’re still running in “no weights found” fallback mode, so the only safe lever is improving the fallback probabilities without touching the model/feature extraction/inference path. I keep your existing hierarchical prior structure but fix a remaining core issue: the train→test `spectrogram_id` mismatch still exists because the fallback builds priors from train rows without explicitly consolidating to the *full recording* spectrogram id per EEG in a stable way. I compute priors in proper “vote-count space” aggregated at EEG level using deduped `(eeg_id,label_id)` rows, then build a strict hierarchical backoff `(patient_id, full_spectrogram_id) -> patient_id -> full_spectrogram_id -> global`, and I remove the unused/incorrect `eeg_id` direct-lookup shortcut (test `eeg_id`s are typically unseen and this can inject noise if any accidental collisions happen). Finally, I keep your KL-safety clipping/normalization and still always write a valid `submission.csv`.'

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
        img = np.zeros((128, 142, 4), dtype="float32")
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




## === cell 5
NAMES = ["LL", "LP", "RP", "RR"]

SFREQ = 200
filter_range = [0.5, 40]

RAW_FEATS = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")


def raw10seeg_from_eeg(parquet_path, eeg_id):
    EEG_LENGTH = 10
    raw_eeg = pd.read_parquet(parquet_path)

    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)

    eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(drop=True)
    list_eeg = []
    for region in RAW_FEATS.keys():
        eeg = np.zeros((len(RAW_FEATS[region]), eeg_default.shape[0]), dtype=np.float32)
        for chan_i, chan in enumerate(RAW_FEATS[region]):
            eeg_1 = eeg_default.loc[:, chan.split("-")[0]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values

            eeg_2 = eeg_default.loc[:, chan.split("-")[1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values

            new_eeg = eeg_1 - eeg_2
            new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            new_eeg = np.clip(new_eeg, -1024, 1024)
            eeg[chan_i, :] = new_eeg

        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate(
            [eeg[0, :, :], eeg[1, :, :], eeg[2, :, :], eeg[3, :, :]], 1
        )
        list_eeg.append(eeg)

    eeg_c = np.concatenate(list_eeg, 1)
    eeg_c /= 104

    time_temp = 0
    time_start = round(time_temp + 18 * 200)
    time_stop = round(time_temp + 28 * 200)

    eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(drop=True)
    list_eeg = []
    for region in RAW_FEATS.keys():
        eeg = np.zeros((len(RAW_FEATS[region]), eeg_default.shape[0]), dtype=np.float32)
        for chan_i, chan in enumerate(RAW_FEATS[region]):
            eeg_1 = eeg_default.loc[:, chan.split("-")[0]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values

            eeg_2 = eeg_default.loc[:, chan.split("-")[1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values

            new_eeg = eeg_1 - eeg_2
            new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            new_eeg = np.clip(new_eeg, -1024, 1024)
            eeg[chan_i, :] = new_eeg

        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate(
            [eeg[0, :, :], eeg[1, :, :], eeg[2, :, :], eeg[3, :, :]], 1
        )
        list_eeg.append(eeg)

    eeg_l = np.concatenate(list_eeg, 1)
    eeg_l /= 104

    time_temp = 0
    time_start = round(time_temp + 22 * 200)
    time_stop = round(time_temp + 32 * 200)

    eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(drop=True)
    list_eeg = []
    for region in RAW_FEATS.keys():
        eeg = np.zeros((len(RAW_FEATS[region]), eeg_default.shape[0]), dtype=np.float32)
        for chan_i, chan in enumerate(RAW_FEATS[region]):
            eeg_1 = eeg_default.loc[:, chan.split("-")[0]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values

            eeg_2 = eeg_default.loc[:, chan.split("-")[1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values

            new_eeg = eeg_1 - eeg_2
            new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            new_eeg = np.clip(new_eeg, -1024, 1024)
            eeg[chan_i, :] = new_eeg

        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate(
            [eeg[0, :, :], eeg[1, :, :], eeg[2, :, :], eeg[3, :, :]], 1
        )
        list_eeg.append(eeg)

    eeg_r = np.concatenate(list_eeg, 1)
    eeg_r /= 104

    return eeg_l, eeg_c, eeg_r


def raw50seeg_from_eeg(parquet_path):
    EEG_LENGTH = 50
    raw_eeg = pd.read_parquet(parquet_path)
    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)

    eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(drop=True)
    list_eeg = []
    for region in RAW_FEATS.keys():
        eeg = np.zeros((len(RAW_FEATS[region]), eeg_default.shape[0]), dtype=np.float32)
        for chan_i, chan in enumerate(RAW_FEATS[region]):
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

        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate(
            (eeg[0, :, :], eeg[1, :, :], eeg[2, :, :], eeg[3, :, :]), 1
        )
        list_eeg.append(eeg)

    eeg = np.concatenate(list_eeg, 1)
    eeg /= 104
    return eeg




## === cell 6
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


def save(row):
    eeg_id = row["eeg_id"]
    spec_id = row["spectrogram_id"]
    spec = pd.read_parquet(f"{SPEC_PATH}{spec_id}.parquet")

    spec_arr = spec.values[:, 1:].T.astype("float32")  # (Hz, Time)
    split_spec_arr = spec_arr[:, 0:300]
    np.save(f"{spec_directory_path}{eeg_id}", split_spec_arr)

    img_l, img_c, img_r = raw10seeg_from_eeg(f"{EEG_PATH}{eeg_id}.parquet", eeg_id)
    np.save(f"{raw_10s_directory_path}{eeg_id}_l", img_l)
    np.save(f"{raw_10s_directory_path}{eeg_id}_c", img_c)
    np.save(f"{raw_10s_directory_path}{eeg_id}_r", img_r)

    img = raw50seeg_from_eeg(f"{EEG_PATH}{eeg_id}.parquet")
    np.save(f"{raw_50s_directory_path}{eeg_id}", img)

    img = stft_spec_from_eeg(f"{EEG_PATH}{eeg_id}.parquet")
    np.save(f"{eeg_directory_path}{eeg_id}", img)


_ = Parallel(n_jobs=4)(delayed(save)(row) for _, row in test.iterrows())




## === cell 7
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



## === cell 8
import timm
import torch.utils.data as data
from torch.utils.data import DataLoader



## === cell 9
from skimage.transform import resize


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

        eeg_img = resize(
            eeg_img, self.test_imgsize, anti_aliasing=True, preserve_range=True
        ).astype("float32")
        spec_img = resize(
            spec_img, self.test_imgsize, anti_aliasing=True, preserve_range=True
        ).astype("float32")
        raw_10s_l_img = resize(
            raw_10s_l_img, self.test_imgsize, anti_aliasing=True, preserve_range=True
        ).astype("float32")
        raw_10s_c_img = resize(
            raw_10s_c_img, self.test_imgsize, anti_aliasing=True, preserve_range=True
        ).astype("float32")
        raw_10s_r_img = resize(
            raw_10s_r_img, self.test_imgsize, anti_aliasing=True, preserve_range=True
        ).astype("float32")
        raw_50s_img = resize(
            raw_50s_img, self.test_imgsize, anti_aliasing=True, preserve_range=True
        ).astype("float32")

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




## === cell 10
class Net(nn.Module):
    def __init__(self, back_bone, device_id):
        super().__init__()
        self.spec_model = timm.create_model(
            "vit_small_patch14_reg4_dinov2.lvd142m",
            num_classes=6,
            pretrained=False,
            in_chans=1,
        )
        self.eeg_model = timm.create_model(
            "vit_small_patch14_reg4_dinov2.lvd142m",
            num_classes=6,
            pretrained=False,
            in_chans=1,
        )
        self.raw_50s_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )
        self.raw_10s_model = timm.create_model(
            "vit_small_patch14_reg4_dinov2.lvd142m",
            num_classes=6,
            pretrained=False,
            in_chans=1,
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

        self.head = nn.Linear(384 * 3 + 1024, 6)
        self.head1 = nn.Linear(384, 6)
        self.head2 = nn.Linear(384, 6)
        self.head3 = nn.Linear(1024, 6)
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




## === cell 11
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

vit_models = []
model_weights = [
    "/kaggle/input/hms-stage2-vitlarge/fold_0_raw_50_10_bestlb_vitlarge.pth",
    "/kaggle/input/hms-stage2-vitlarge/fold_1_raw_50_10_bestlb_vitlarge.pth",
    "/kaggle/input/hms-stage2-vitlarge/fold_2_raw_50_10_bestlb_vitlarge.pth",
    "/kaggle/input/hms-stage2-vitlarge/fold_3_raw_50_10_bestlb_vitlarge.pth",
    "/kaggle/input/hms-stage2-vitlarge/fold_4_raw_50_10_bestlb_vitlarge.pth",
]
model_types = ["vit_large", "vit_large", "vit_large", "vit_large", "vit_large"]

result_8 = {}

available = [p for p in model_weights if os.path.exists(p)]
if len(available) == 0:
    print(
        "WARNING: No model weight files found under /kaggle/input/hms-stage2-vitlarge/."
    )
    print(
        "Will fall back to hierarchical priors. Change: rebuild FULL spectrogram_id priors using a stable eeg_id->spectrogram_id mapping, then strict backoff (patient,spec)->patient->spec->global."
    )

    train_path = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"

    map_cols = ["eeg_id", "spectrogram_id"]
    train_map = pd.read_csv(train_path, usecols=map_cols)
    train_map["eeg_id"] = train_map["eeg_id"].astype(np.int64)
    train_map["spectrogram_id"] = train_map["spectrogram_id"].astype(np.int64)
    eeg_to_full_sid = (
        train_map.groupby("eeg_id", sort=False)["spectrogram_id"]
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
    )
    del train_map
    gc.collect()

    usecols = ["eeg_id", "patient_id", "label_id"] + CLASSES
    train_df = pd.read_csv(train_path, usecols=usecols)
    train_df["eeg_id"] = train_df["eeg_id"].astype(np.int64)
    train_df["patient_id"] = train_df["patient_id"].astype(np.int64)
    train_df["label_id"] = train_df["label_id"].astype(np.int64)
    train_df["spectrogram_id_full"] = (
        train_df["eeg_id"].map(eeg_to_full_sid).astype(np.int64)
    )

    votes = train_df[CLASSES].values.astype(np.float64)
    votes_sum = votes.sum(axis=1, keepdims=True)
    votes_sum[votes_sum == 0] = 1.0

    train_df["_row_idx_"] = np.arange(len(train_df), dtype=np.int64)
    keep = (
        train_df.groupby(["eeg_id", "label_id"], sort=False)["_row_idx_"].first().values
    )

    keep_meta = train_df.loc[
        keep, ["eeg_id", "patient_id", "spectrogram_id_full"]
    ].copy()
    keep_votes = votes[keep]
    keep_strength = votes_sum[keep].reshape(-1)

    keep_meta["_i_"] = np.arange(len(keep_meta), dtype=np.int64)
    eeg_to_idxs = keep_meta.groupby("eeg_id", sort=False)["_i_"].apply(list).to_dict()
    eeg_ids = np.fromiter(eeg_to_idxs.keys(), dtype=np.int64, count=len(eeg_to_idxs))

    eeg_patient = np.empty(len(eeg_ids), dtype=np.int64)
    eeg_spec_full = np.empty(len(eeg_ids), dtype=np.int64)
    eeg_counts = np.zeros((len(eeg_ids), N_CLASSES), dtype=np.float64)
    eeg_n = np.zeros(len(eeg_ids), dtype=np.float64)

    for i, eid in enumerate(eeg_ids):
        idxs = eeg_to_idxs[int(eid)]
        eeg_patient[i] = int(keep_meta.loc[idxs[0], "patient_id"])
        eeg_spec_full[i] = int(keep_meta.loc[idxs[0], "spectrogram_id_full"])
        v = keep_votes[idxs].mean(axis=0)
        s = float(np.clip(np.mean(keep_strength[idxs]), 1.0, 20.0))
        eeg_counts[i] = v
        eeg_n[i] = s

    global_counts = (eeg_counts * eeg_n[:, None]).sum(axis=0)
    global_prior = global_counts / max(global_counts.sum(), 1e-12)
    global_prior = np.clip(global_prior, 1e-12, None)
    global_prior = global_prior / global_prior.sum()

    from collections import defaultdict

    pid_sum = defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.float64))
    pid_n = defaultdict(float)

    sid_sum = defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.float64))
    sid_n = defaultdict(float)

    pid_sid_sum = defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.float64))
    pid_sid_n = defaultdict(float)

    for i in range(len(eeg_ids)):
        pid = int(eeg_patient[i])
        sid = int(eeg_spec_full[i])
        c = eeg_counts[i] * eeg_n[i]
        w = float(eeg_n[i])

        pid_sum[pid] += c
        pid_n[pid] += w

        sid_sum[sid] += c
        sid_n[sid] += w

        pid_sid_sum[(pid, sid)] += c
        pid_sid_n[(pid, sid)] += w

    alpha_pid = 140.0
    alpha_sid = 180.0
    alpha_pid_sid = 260.0

    def dirichlet_smooth(counts, base_prob, alpha):
        counts = np.asarray(counts, dtype=np.float64)
        base_prob = np.asarray(base_prob, dtype=np.float64)
        sm = counts + alpha * base_prob
        sm = np.clip(sm, 1e-12, None)
        return sm / sm.sum()

    pid_prior_sm = {
        pid: dirichlet_smooth(csum, global_prior, alpha_pid)
        for pid, csum in pid_sum.items()
    }
    sid_prior_sm = {
        sid: dirichlet_smooth(csum, global_prior, alpha_sid)
        for sid, csum in sid_sum.items()
    }

    pid_sid_prior_sm = {}
    for (pid, sid), csum in pid_sid_sum.items():
        base = 0.70 * sid_prior_sm.get(sid, global_prior) + 0.30 * pid_prior_sm.get(
            pid, global_prior
        )
        base = np.clip(base, 1e-12, None)
        base = base / base.sum()
        pid_sid_prior_sm[(pid, sid)] = dirichlet_smooth(csum, base, alpha_pid_sid)

    def w_from_n(n, alpha):
        n = float(n)
        return n / (n + alpha)

    uniform = np.ones(N_CLASSES, dtype=np.float64) / N_CLASSES
    tiny_uniform_mix = 0.01
    temp = 1.08  # small flattening to reduce overconfident wrong preds (KL-sensitive)

    def apply_temperature(p, t):
        p = np.clip(p, 1e-12, None)
        lp = np.log(p)
        lp = lp / float(t)
        lp = lp - lp.max()
        q = np.exp(lp)
        q = np.clip(q, 1e-12, None)
        return q / q.sum()

    for _, row in test.iterrows():
        eid_str = str(row["eeg_id"])
        pid = int(row["patient_id"])
        sid = int(row["spectrogram_id"])  # test uses full recording spectrogram_id
        key = (pid, sid)

        n_pid_sid = pid_sid_n.get(key, 0.0)
        n_pid = pid_n.get(pid, 0.0)
        n_sid = sid_n.get(sid, 0.0)

        w_pid_sid = w_from_n(n_pid_sid, alpha_pid_sid)
        w_pid = w_from_n(n_pid, alpha_pid)
        w_sid = w_from_n(n_sid, alpha_sid)

        p_pid = pid_prior_sm.get(pid, global_prior)
        p_sid = sid_prior_sm.get(sid, global_prior)
        p_pid_sid = pid_sid_prior_sm.get(key, None)

        if p_pid_sid is None:
            w_pid_sid = 0.0
            p_pid_sid = global_prior

        base = 0.70 * (w_sid * p_sid + (1.0 - w_sid) * global_prior) + 0.30 * (
            w_pid * p_pid + (1.0 - w_pid) * global_prior
        )
        p = w_pid_sid * p_pid_sid + (1.0 - w_pid_sid) * base

        p = (1.0 - tiny_uniform_mix) * p + tiny_uniform_mix * uniform
        p = apply_temperature(p, temp)

        p = np.clip(p, 1e-6, None)
        p = p / p.sum()
        result_8[eid_str] = p.astype(np.float64)

    result_8["_FALLBACK_PRIOR_"] = global_prior.astype(np.float64)

else:
    for i in range(len(model_types)):
        if not os.path.exists(model_weights[i]):
            print(f"WARNING: missing weights: {model_weights[i]} -> skipping this fold")
            continue
        if model_types[i] == "vit_large":
            model = Net("vit_large_patch14_reg4_dinov2.lvd142m", device).to(device)
            state = torch.load(model_weights[i], map_location=device)
            model.load_state_dict(state, strict=True)
            model.eval()
            vit_models.append(model)

    test_data = ImageFolder(test, (518, 518))
    test_loader = DataLoader(
        test_data, batch_size=16, pin_memory=False, num_workers=4, drop_last=False
    )

    with torch.no_grad():
        for batch_idx, (
            spec_imgs,
            eeg_imgs,
            raw_50s_imgs,
            raw_10s_l_imgs,
            raw_10s_c_imgs,
            raw_10s_r_imgs,
            eeg_ids,
        ) in enumerate(test_loader):
            spec_imgs = spec_imgs.to(device).float()
            eeg_imgs = eeg_imgs.to(device).float()
            raw_50s_imgs = raw_50s_imgs.to(device).float()
            raw_10s_c_imgs = raw_10s_c_imgs.to(device).float()

            ensemble_probs = torch.zeros((spec_imgs.shape[0], 6), device=device)
            for model in vit_models:
                logits_c, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs
                )
                probs_c = logits_c.softmax(dim=1)
                ensemble_probs += probs_c

            ensemble_probs /= max(len(vit_models), 1)
            ensemble_probs = ensemble_probs.detach().cpu().numpy()

            for j in range(len(eeg_ids)):
                eeg_id = str(eeg_ids[j])  # ensure consistent key type
                if eeg_id not in result_8:
                    result_8[eeg_id] = np.zeros(6, dtype=np.float64)
                result_8[eeg_id] += ensemble_probs[j]



## === cell 12
for model in vit_models:
    del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()



## === cell 13
sample_sub_path = (
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
sample = pd.read_csv(sample_sub_path)

pred_df = pd.DataFrame({"eeg_id": sample["eeg_id"].astype(str)})
preds = np.zeros((len(pred_df), 6), dtype=np.float64)

pred_map = {str(k): v for k, v in result_8.items()}
fallback = pred_map.get("_FALLBACK_PRIOR_", np.ones(6, dtype=np.float64) / 6.0)

for i, eid in enumerate(pred_df["eeg_id"].values):
    if eid in pred_map and eid != "_FALLBACK_PRIOR_":
        preds[i] = pred_map[eid]
    else:
        preds[i] = fallback

preds = np.clip(preds, 1e-6, None)
preds = preds / preds.sum(axis=1, keepdims=True)

for ci, c in enumerate(CLASSES):
    pred_df[c] = preds[:, ci].astype(np.float32)

pred_df.to_csv("submission.csv", index=False)
print(pred_df.head())
print("Saved submission.csv with shape:", pred_df.shape)



## === cell 14
if DEBUG is False:
    os.system("rm -rf /kaggle/working/spec_spectrograms/")
    os.system("rm -rf /kaggle/working/eeg_spectrograms/")
    os.system("rm -rf /kaggle/working/eeg_50s_raws/")
    os.system("rm -rf /kaggle/working/eeg_10s_raws/")
    os.system("rm -rf /kaggle/working/squeezeformer")
