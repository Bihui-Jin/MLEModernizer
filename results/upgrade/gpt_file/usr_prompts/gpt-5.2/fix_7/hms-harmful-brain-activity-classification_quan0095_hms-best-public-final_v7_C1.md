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

0.2850204006981144

# 6. Current score

0.86242

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the runtime blockers so the notebook can execute end-to-end on a CPU-only Kaggle session and still generate a valid `submission.csv`. The main issue is hardcoding `cuda:0`; I switch to an automatic device selection and load checkpoints with `map_location` so models can run on CPU. Because your external model weight paths may not exist in this environment (and CPU inference with multiple large ViTs is too slow), I add a safe fallback that produces a valid probability submission using the empirical class prior from `train.csv` (score be worse than your target but yield a valid submission). I also remove the Jupyter `!rm` shell lines (they break in a `.py` run) and ensure probabilities are normalized to sum to 1.'
- What this solution (achieved 0.73126) has done: 'Your current score (1.41937, lower-is-better) is far worse than the target (0.2850), and the main reason is that you’re almost certainly falling back to the global class-prior submission because the external weight files don’t exist. The smallest legitimate improvement is to replace that fallback with a simple, well-calibrated per-patient prior computed from `train.csv` (still a “no-model” baseline, but much closer to realistic test distribution than a global prior). I also ensure `eeg_id` stays integer (as in sample_submission) to avoid any potential merge/alignment quirks, and I keep the probability clipping+renormalization for KL stability. No model architecture/training logic is changed; only the fallback prediction logic is improved to move the score toward your target.'
- What this solution (achieved 0.93865) has done: 'Your current submission is almost certainly coming from the fallback “patient prior” rather than the ViT ensemble (weights are missing), so the only realistic way to move the KL score toward your target is to make that fallback closer to the hidden test distribution without changing your core model logic. I keep the entire feature extraction + model code intact and only improve the fallback by (1) averaging labels at the `eeg_id` level (to match the evaluation unit) before computing priors, and (2) using a shrinkage/Dirichlet-smoothed per-patient distribution based on that patient’s effective sample size (more reliable than a fixed alpha). I also ensure the output probabilities are clipped + renormalized and that the submission rows align exactly to `test.csv` order and schema. These are minimal changes directly tied to improving the metric under the “no-weights” path.'
- What this solution (achieved 1.21915) has done: 'Your current score (0.93865, lower-is-better) is far from the target (0.2850), and given the missing external weights, the only viable path is improving the fallback so its predicted distribution is closer to the hidden test distribution. I keep all feature extraction and model code intact, and only change the fallback to (1) compute per-patient priors using an empirical-Bayes Dirichlet posterior mean at the eeg_id level (evaluation unit), and (2) blend that patient posterior with a global posterior using a small fixed mixing weight for better out-of-patient generalization. I also make the patient key lookup robust by building dicts (avoids pandas index dtype mismatches) and keep the same probability clipping+renormalization required for KL stability. This is a minimal change that directly targets the metric under the “no-weights” path and should improve toward your target without changing model semantics.'
- What this solution (achieved 0.86242) has done: 'Your score is far worse than the target (lower-is-better), and since the external model weights are missing you’re effectively submitting the fallback priors. The smallest meaningful improvement without changing any model/feature logic is to make the fallback better match the evaluation unit (`eeg_id`) and reduce calibration error by using a patient posterior built from **Dirichlet-smoothed per-eeg aggregated votes** (not per-row means), plus a small global blend. I also ensure the fallback uses proper pseudo-count updates (alpha + summed votes) and keep the same clipping+renormalization so the KL metric remains stable and the submission always validates. No architecture, feature extraction, or inference code is changed—only the no-weights fallback probability estimator.'
- What this solution (achieved 0.86242) has done: 'Your current score (0.86242, lower-is-better) is far from the target (0.2850), and since the external weights aren’t available you’re effectively relying on the fallback priors. I keep all model/feature logic untouched and only make the fallback closer to the evaluation distribution by conditioning not just on `patient_id` but also on `spectrogram_id` (test provides it), using the same Dirichlet-smoothed posterior mean and a small global blend for robustness. This is a minimal change that uses only metadata already available at inference time, and it should move KL down toward the target. I also keep the existing clipping+renormalization to ensure valid probabilities for the KL metric and submission validity.'

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

import librosa  # noqa: F401



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

    list_eeg = list()
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
    list_eeg = list()
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
    list_eeg = list()
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
    list_eeg = list()
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
    list_eeg = list()
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

DATA_ROOT = "/kaggle/input/hms-harmful-brain-activity-classification"

if DEBUG:
    test = pd.read_csv(f"{DATA_ROOT}/train.csv")[:40]
    SPEC_PATH = f"{DATA_ROOT}/train_spectrograms/"
    EEG_PATH = f"{DATA_ROOT}/train_eegs/"
else:
    test = pd.read_csv(f"{DATA_ROOT}/test.csv")
    SPEC_PATH = f"{DATA_ROOT}/test_spectrograms/"
    EEG_PATH = f"{DATA_ROOT}/test_eegs/"

print("test shape:", test.shape)

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



## === cell 7
from joblib import Parallel, delayed


def save_features(row):
    eeg_id = row["eeg_id"]
    spec_id = row["spectrogram_id"]

    spec_out = f"{spec_directory_path}{eeg_id}.npy"
    eeg_out = f"{eeg_directory_path}{eeg_id}.npy"
    raw50_out = f"{raw_50s_directory_path}{eeg_id}.npy"
    raw10_l_out = f"{raw_10s_directory_path}{eeg_id}_l.npy"
    raw10_c_out = f"{raw_10s_directory_path}{eeg_id}_c.npy"
    raw10_r_out = f"{raw_10s_directory_path}{eeg_id}_r.npy"

    if all(
        os.path.exists(x)
        for x in [spec_out, eeg_out, raw50_out, raw10_l_out, raw10_c_out, raw10_r_out]
    ):
        return

    spec = pd.read_parquet(f"{SPEC_PATH}{spec_id}.parquet")
    spec_arr = spec.values[:, 1:].T.astype("float32")  # (Hz, Time)
    split_spec_arr = spec_arr[:, 0:300]
    np.save(f"{spec_directory_path}{eeg_id}", split_spec_arr)

    img_l, img_c, img_r = raw10seeg_from_eeg(f"{EEG_PATH}{eeg_id}.parquet", eeg_id)
    np.save(f"{raw_10s_directory_path}{eeg_id}_l", img_l)
    np.save(f"{raw_10s_directory_path}{eeg_id}_c", img_c)
    np.save(f"{raw_10s_directory_path}{eeg_id}_r", img_r)

    img50 = raw50seeg_from_eeg(f"{EEG_PATH}{eeg_id}.parquet")
    np.save(f"{raw_50s_directory_path}{eeg_id}", img50)

    imgstft = stft_spec_from_eeg(f"{EEG_PATH}{eeg_id}.parquet")
    np.save(f"{eeg_directory_path}{eeg_id}", imgstft)


_ = Parallel(n_jobs=2)(delayed(save_features)(row) for _, row in test.iterrows())




## === cell 8
class Config:
    seed = 2024
    num_folds = 5


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seed_everything(Config.seed)



## === cell 9
import timm
import torch.utils.data as data
from torch.utils.data import DataLoader
from skimage.transform import resize




## === cell 10
class ImageFolder(data.Dataset):
    def __init__(self, df, test_imgsize):
        super().__init__()
        df = df.copy()
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

        spec_img = np.load(os.path.join(self.spec_data_path, eeg_id + ".npy")).astype(
            "float32"
        )
        raw_50s_img = np.load(
            os.path.join(self.raw_50s_data_path, eeg_id + ".npy")
        ).astype("float32")
        raw_10s_l_img = np.load(
            os.path.join(self.raw_10s_data_path, eeg_id + "_l.npy")
        ).astype("float32")
        raw_10s_c_img = np.load(
            os.path.join(self.raw_10s_data_path, eeg_id + "_c.npy")
        ).astype("float32")
        raw_10s_r_img = np.load(
            os.path.join(self.raw_10s_data_path, eeg_id + "_r.npy")
        ).astype("float32")
        eeg_img = np.load(os.path.join(self.eeg_data_path, eeg_id + ".npy")).astype(
            "float32"
        )

        eeg_img = resize(eeg_img, self.test_imgsize)
        spec_img = resize(spec_img, self.test_imgsize)
        raw_10s_l_img = resize(raw_10s_l_img, self.test_imgsize)
        raw_10s_c_img = resize(raw_10s_c_img, self.test_imgsize)
        raw_10s_r_img = resize(raw_10s_r_img, self.test_imgsize)
        raw_50s_img = resize(raw_50s_img, self.test_imgsize)

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




## === cell 11
class Net(nn.Module):
    def __init__(self, back_bone, device_id):
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
        return logits




## === cell 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model_weights = [
    "/kaggle/input/hms-stage2-2/fold_0_raw_20_10_bestlb.pth",
    "/kaggle/input/hms-stage2-2/fold_1_raw_20_10_bestlb.pth",
    "/kaggle/input/hms-stage2-2/fold_2_raw_20_10_bestlb.pth",
    "/kaggle/input/hms-stage2-2/fold_3_raw_20_10_bestlb.pth",
    "/kaggle/input/hms-stage2-2/fold_4_raw_20_10_bestlb.pth",
]
backbone_name = "vit_small_patch14_reg4_dinov2.lvd142m"

all_weights_exist = all(os.path.exists(w) for w in model_weights)
print("All weights exist:", all_weights_exist)




## === cell 13
def run_inference_if_possible():
    if not all_weights_exist:
        return None

    vit_models = []
    for w in model_weights:
        m = Net(backbone_name, str(device))
        sd = torch.load(w, map_location=device)
        m.load_state_dict(sd)
        m.to(device)
        m.eval()
        vit_models.append(m)

    test_data = ImageFolder(test, (518, 518))
    test_loader = DataLoader(
        test_data,
        batch_size=(4 if device.type == "cpu" else 32),
        pin_memory=(device.type == "cuda"),
        num_workers=0,
        drop_last=False,
    )

    result = {}
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
            raw_10s_c_imgs = raw_10s_c_imgs.to(device).float()

            ensemble_probs = torch.zeros((spec_imgs.shape[0], 6), device=device)
            for model in vit_models:
                logits = model(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs)
                ensemble_probs += logits.softmax(dim=1)
            ensemble_probs /= len(vit_models)

            probs_np = ensemble_probs.detach().cpu().numpy()
            for j in range(len(eeg_ids)):
                eid = str(eeg_ids[j])
                result[eid] = probs_np[j].astype(np.float64)

    for m in vit_models:
        del m
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


pred_dict = run_inference_if_possible()



## === cell 14
if pred_dict is None:
    train = pd.read_csv(
        f"{DATA_ROOT}/train.csv",
        usecols=["eeg_id", "patient_id", "spectrogram_id"] + CLASSES,
    )

    eeg_votes = (
        train.groupby(["eeg_id", "patient_id", "spectrogram_id"], sort=False)[CLASSES]
        .sum()
        .reset_index()
    )

    global_counts = eeg_votes[CLASSES].sum(axis=0).values.astype(np.float64)
    global_counts = np.clip(global_counts, 1.0, None)
    global_mean = global_counts / global_counts.sum()

    s0 = 120.0
    alpha0 = s0 * global_mean
    global_posterior = alpha0 / alpha0.sum()

    group_counts = eeg_votes.groupby(["patient_id", "spectrogram_id"], sort=False)[
        CLASSES
    ].sum()
    group_total = group_counts.sum(axis=1).astype(np.float64)

    group_counts_dict = {
        (int(pid), int(sid)): group_counts.loc[(pid, sid)].values.astype(np.float64)
        for (pid, sid) in group_counts.index
    }
    group_total_dict = {
        (int(pid), int(sid)): float(group_total.loc[(pid, sid)])
        for (pid, sid) in group_total.index
    }

    patient_counts = eeg_votes.groupby("patient_id", sort=False)[CLASSES].sum()
    patient_total = patient_counts.sum(axis=1).astype(np.float64)
    patient_counts_dict = {
        int(pid): patient_counts.loc[pid].values.astype(np.float64)
        for pid in patient_counts.index
    }
    patient_total_dict = {
        int(pid): float(patient_total.loc[pid]) for pid in patient_total.index
    }

    mix_global = 0.10
    mix_patient_backoff = (
        0.30  # when group exists but is small/noisy, blend with patient-level
    )

    test_pids = test["patient_id"].astype(np.int64).values
    test_sids = test["spectrogram_id"].astype(np.int64).values
    pred = np.zeros((len(test), N_CLASSES), dtype=np.float64)

    for i, (pid, sid) in enumerate(zip(test_pids, test_sids)):
        key = (int(pid), int(sid))
        counts_g = group_counts_dict.get(key)
        n_g = group_total_dict.get(key, 0.0)

        if counts_g is not None and n_g > 0:
            alpha_post_g = alpha0 + counts_g
            p_g = alpha_post_g / alpha_post_g.sum()

            counts_p = patient_counts_dict.get(int(pid))
            n_p = patient_total_dict.get(int(pid), 0.0)
            if counts_p is not None and n_p > 0:
                alpha_post_p = alpha0 + counts_p
                p_p = alpha_post_p / alpha_post_p.sum()
                p = (1.0 - mix_patient_backoff) * p_g + mix_patient_backoff * p_p
            else:
                p = p_g

            p = (1.0 - mix_global) * p + mix_global * global_posterior
        else:
            counts_p = patient_counts_dict.get(int(pid))
            n_p = patient_total_dict.get(int(pid), 0.0)
            if counts_p is not None and n_p > 0:
                alpha_post_p = alpha0 + counts_p
                p_p = alpha_post_p / alpha_post_p.sum()
                p = (1.0 - mix_global) * p_p + mix_global * global_posterior
            else:
                p = global_posterior.copy()

        pred[i] = p

    pred = np.clip(pred, 1e-12, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    pred_df = pd.DataFrame(pred, columns=CLASSES)
    pred_df.insert(0, "eeg_id", test["eeg_id"].astype(np.int64).values)
else:
    preds = []
    for eid in test["eeg_id"].astype(str).values:
        p = pred_dict.get(str(eid))
        if p is None:
            p = np.ones(6, dtype=np.float64) / 6.0
        preds.append(p)
    pred = np.vstack(preds).astype(np.float64)
    pred = np.clip(pred, 1e-12, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    pred_df = pd.DataFrame(pred, columns=CLASSES)
    pred_df.insert(0, "eeg_id", test["eeg_id"].astype(np.int64).values)

sub = pred_df.copy()
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)




## === cell 15
def safe_rmtree(path):
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                try:
                    os.remove(os.path.join(root, name))
                except OSError:
                    pass
            for name in dirs:
                try:
                    os.rmdir(os.path.join(root, name))
                except OSError:
                    pass
        try:
            os.rmdir(path)
        except OSError:
            pass


if not DEBUG:
    pass
