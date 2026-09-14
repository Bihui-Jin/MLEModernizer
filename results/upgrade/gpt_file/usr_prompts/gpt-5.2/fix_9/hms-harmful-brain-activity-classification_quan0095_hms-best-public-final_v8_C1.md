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

0.2840415419245986

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The crashes come from hardcoding `device="cuda:0"` in a CPU-only environment; I make device selection automatic and ensure all `torch.load` calls use `map_location=device` so weights load on CPU. I also remove notebook shell (`!rm`) usage that would break in a plain `.py` run, and replace it with safe Python cleanup. To guarantee a valid submission even if the external weight files are missing, I add a minimal fallback that outputs the sample_submission uniform probabilities (still sums to 1), while keeping the original ensemble logic when weights exist. Finally, I ensure the submission contains all `eeg_id` from `test.csv` in the correct order and each row is normalized to sum exactly to one.'
- What this solution (achieved 1.40995) has done: 'Your current 1.40995 is far worse than the 0.2840 target (lower is better), and the main reason is that the script is not actually using the intended pretrained weights (it falls back to uniform predictions when `/kaggle/input/hms-stage2-2/...` isn’t present). I keep the same feature extraction, dataset, model, and inference logic, but change the weight discovery to search common Kaggle input locations for matching `.pth` files and load them when available. I also make the inference use all three 10s crops (l/c/r) by averaging their probabilities, which is a minimal post-processing change consistent with the existing inputs and improves calibration for KL without changing architecture. Finally, I keep the submission formatting/normalization guarantees unchanged.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.28404), and the most likely cause is that you’re still not actually loading the intended fold weights (so predictions are effectively weak/uniform). I keep the exact same feature extraction, dataset, model, and inference logic, but make weight discovery more robust by (1) accepting any `fold_{k}*.pth` that matches the backbone/head shape and (2) handling common checkpoint key wrappers (`state_dict`, `model`, `module.`) so valid weights load instead of silently failing or being skipped. I also ensure we always use `torch.inference_mode()` and non-blocking device transfers for deterministic, stable inference without changing semantics. If no compatible weights are found, the code still fall back to uniform probabilities and write a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your current score is far above the target (1.40995 vs 0.28404, lower-is-better), and the most likely reason is that you are not actually loading any fold weights, so you fall back to near-uniform predictions. I keep the exact same feature extraction, dataset, model, and inference logic, but make weight discovery/load robust by (1) searching all `/kaggle/input/**` for `.pth` files, (2) trying multiple common checkpoint key patterns, and (3) allowing `strict=False` while verifying that a high-signal subset of keys (the classifier head and at least one backbone block) actually loads—so we avoid silently using random weights. This is a minimal change that should move KL much closer to the target because real pretrained ensemble predictions replace the uniform fallback. I also add a deterministic “unique test rows” guard to avoid accidental duplicate feature extraction/inference for the same `eeg_id`, which can otherwise introduce inconsistencies without changing semantics.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.28404), and the dominant likely cause is that no compatible `.pth` weights are actually being loaded, so you effectively submit near-uniform probabilities. I keep the exact same feature extraction, dataset, model, and inference, but make checkpoint discovery and loading more robust by (1) also searching for `.pt`/`.bin` and (2) remapping common key prefixes so weights saved from different wrapper modules (e.g., `net.`, `model.`, `module.`) load into your `Net` instead of being skipped. I also fix a subtle determinism issue (you set `cudnn.deterministic=True` and `benchmark=True` simultaneously) to avoid inconsistent behavior; this doesn’t change the core logic but improves stability. If no compatible weights still exist in the environment, the script continue to produce a valid `submission.csv` with normalized probabilities.'
- What this solution (achieved 1.40995) has done: 'Your KL is much worse than the target, and the most likely cause (given the current code) is that no usable pretrained weights are actually being loaded, so you effectively submit near-uniform predictions. I keep the same feature extraction, dataset, model, and inference flow, but (1) make weight discovery explicitly look for the known “hms-stage2-2” style fold files by scanning inputs for that pattern, and (2) relax the “backbone_loaded” verification bug (it currently checks prefixes incorrectly, so it can reject valid checkpoints) while still requiring the classifier head to load. Finally, I keep the submission formatting identical but add a tiny epsilon-smoothing after ensembling to reduce KL blow-ups from overconfident near-zeros (still sums to 1 and preserves semantics).'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far worse than the target (0.2840), and the biggest likely driver is that the script is still frequently not loading the intended pretrained checkpoints (so predictions are weak/near-uniform). I make the checkpoint loading more permissive (without changing model architecture) by (1) fixing the backbone-loaded verification (it currently checks the wrong key prefixes) and (2) adding a second, safe fallback remap that can drop common wrapper prefixes so real weights load instead of being rejected. I also increase the probability-floor smoothing slightly (still normalized to sum to 1) to reduce KL blow-ups from near-zero probabilities, which usually improves KL while keeping evaluation semantics intact. Everything else (feature extraction, model forward, ensembling, submission formatting) stays the same.'
- What this solution (achieved 1.40995) has done: 'Your KL is far worse than the target, so the smallest likely improvement is to ensure the intended pretrained checkpoints actually load instead of silently being rejected and falling back to weak/uniform predictions. I keep your exact model/feature/inference flow, but (1) fix the “backbone_loaded” verification bug (currently it never matches because `startswith()` is used with a tuple incorrectly) so valid checkpoints aren’t discarded, and (2) make weight discovery prefer the existing “hms-stage2-2” fold files while avoiding loading lots of unrelated checkpoints. Finally, I keep your probability smoothing but reduce it slightly (still normalized) to avoid hurting good models with excessive uniform-mixing, which should move KL closer to the 0.284 target without changing evaluation semantics.'

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)



## === cell 2
NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]

try:
    import librosa  # noqa: F401
except Exception as e:
    print("Warning: librosa import failed:", repr(e))



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

if DEBUG:
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

test = test.drop_duplicates(subset=["eeg_id"]).reset_index(drop=True)

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


def save(row):
    eeg_id = row["eeg_id"]
    spec_id = row["spectrogram_id"]

    spec = pd.read_parquet(f"{SPEC_PATH}{spec_id}.parquet")
    spec_arr = spec.values[:, 1:].T.astype("float32")  # (Hz, Time) = (400, 300)

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


needed_npy = os.path.join(spec_directory_path, str(test.iloc[0]["eeg_id"]) + ".npy")
if not os.path.exists(needed_npy):
    _ = Parallel(n_jobs=4)(delayed(save)(row) for _, row in test.iterrows())
else:
    print("Cached npy features found; skipping feature extraction.")




## === cell 8
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
def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _strip_module_prefix(state):
    if not isinstance(state, dict):
        return state
    if any(k.startswith("module.") for k in state.keys()):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _remap_common_prefixes_to_net(state):
    if not isinstance(state, dict) or len(state) == 0:
        return state

    keys = list(state.keys())
    prefixes = (
        "net.",
        "model.",
        "student.",
        "teacher.",
        "module.net.",
        "module.model.",
    )
    for pref in prefixes:
        if sum(k.startswith(pref) for k in keys) > 0.8 * len(keys):
            return {k.replace(pref, "", 1): v for k, v in state.items()}

    if sum(k.startswith("module.") for k in keys) > 0.8 * len(keys):
        return {k.replace("module.", "", 1): v for k, v in state.items()}

    return state


def _drop_leading_wrapper(state, wrapper):
    if not isinstance(state, dict) or len(state) == 0:
        return state
    if any(k.startswith(wrapper) for k in state.keys()):
        return {k.replace(wrapper, "", 1): v for k, v in state.items()}
    return state


def _try_load_weights_into_model(model, weight_path, device):
    try:
        ckpt = torch.load(weight_path, map_location=device)
        state = _extract_state_dict(ckpt)
        state = _strip_module_prefix(state)
        state = _remap_common_prefixes_to_net(state)
        state = _strip_module_prefix(state)

        for wrap in ("ema.", "model.", "net.", "student.", "teacher."):
            state = _drop_leading_wrapper(state, wrap)

        if not isinstance(state, dict) or len(state) == 0:
            return False

        model_state = model.state_dict()
        incompatible = model.load_state_dict(state, strict=False)

        loaded_keys = set(model_state.keys()) - set(incompatible.missing_keys)
        head_loaded = ("head.weight" in loaded_keys) and ("head.bias" in loaded_keys)

        backbone_prefixes = (
            "spec_model.",
            "eeg_model.",
            "raw_50s_model.",
            "raw_10s_model.",
        )
        backbone_loaded = any(k.startswith(backbone_prefixes) for k in loaded_keys)

        load_ratio = len(loaded_keys) / max(1, len(model_state))
        if head_loaded and backbone_loaded and load_ratio > 0.50:
            return True
        return False
    except Exception:
        return False


def discover_model_weights():
    preferred = [
        "/kaggle/input/hms-stage2-2/fold_0_raw_20_10_bestlb.pth",
        "/kaggle/input/hms-stage2-2/fold_1_raw_20_10_bestlb.pth",
        "/kaggle/input/hms-stage2-2/fold_2_raw_20_10_bestlb.pth",
        "/kaggle/input/hms-stage2-2/fold_3_raw_20_10_bestlb.pth",
        "/kaggle/input/hms-stage2-2/fold_4_raw_20_10_bestlb.pth",
    ]
    if all(os.path.exists(p) for p in preferred):
        return preferred

    candidates = []
    exts = (".pth", ".pt", ".bin")
    for root, _, files in os.walk("/kaggle/input"):
        lroot = root.lower()
        root_bonus = 0
        if "hms" in lroot:
            root_bonus += 1
        if "stage2" in lroot:
            root_bonus += 2
        if "hms-stage2-2" in lroot:
            root_bonus += 10
        for fn in files:
            if fn.endswith(exts):
                full = os.path.join(root, fn)
                score = root_bonus
                lfn = fn.lower()
                if "raw_20_10" in lfn:
                    score += 6
                if "bestlb" in lfn:
                    score += 5
                if "fold_" in lfn or "fold" in lfn:
                    score += 3
                if "best" in lfn:
                    score += 2
                if "lb" in lfn:
                    score += 1
                candidates.append((score, full))
    candidates.sort(key=lambda x: (-x[0], x[1]))

    weights_by_fold = {}
    nonfold = []
    for score, p in candidates:
        bn = os.path.basename(p)
        fold = None
        parts = bn.replace("-", "_").split("_")
        for i, tok in enumerate(parts):
            if tok == "fold" and i + 1 < len(parts):
                try:
                    fold = int(parts[i + 1])
                except Exception:
                    fold = None
            if tok.startswith("fold"):
                try:
                    digits = "".join(ch for ch in tok if ch.isdigit())
                    fold = int(digits) if digits else fold
                except Exception:
                    fold = None
        if fold is not None:
            if fold not in weights_by_fold:
                weights_by_fold[fold] = p
        else:
            nonfold.append(p)

    if all(k in weights_by_fold for k in range(5)):
        return [weights_by_fold[i] for i in range(5)]
    if len(weights_by_fold) > 0:
        return [weights_by_fold[k] for k in sorted(weights_by_fold.keys())]
    if len(nonfold) > 0:
        return nonfold[:10]
    return preferred


backbone = "vit_small_patch14_reg4_dinov2.lvd142m"
model_weights = discover_model_weights()

print("Model weights discovered (order used):")
for p in model_weights:
    print("  ", p, "exists:", os.path.exists(p))

test_data = ImageFolder(test, (518, 518))
test_loader = DataLoader(
    test_data,
    batch_size=16 if device.type == "cpu" else 32,
    pin_memory=(device.type == "cuda"),
    num_workers=2,
    drop_last=False,
)

result = {}

vit_models = []
for w in model_weights:
    if not os.path.exists(w):
        continue
    m = Net(backbone, str(device)).to(device)
    ok = _try_load_weights_into_model(m, w, device)
    if ok:
        m.eval()
        vit_models.append(m)
        print("Loaded:", w)
    else:
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

print("Loaded models:", len(vit_models))

if len(vit_models) > 0:
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
            spec_imgs = spec_imgs.to(device, non_blocking=True).float()
            eeg_imgs = eeg_imgs.to(device, non_blocking=True).float()
            raw_50s_imgs = raw_50s_imgs.to(device, non_blocking=True).float()
            raw_10s_l_imgs = raw_10s_l_imgs.to(device, non_blocking=True).float()
            raw_10s_c_imgs = raw_10s_c_imgs.to(device, non_blocking=True).float()
            raw_10s_r_imgs = raw_10s_r_imgs.to(device, non_blocking=True).float()

            ensemble_probs = torch.zeros((spec_imgs.shape[0], 6), device=device)
            for m in vit_models:
                probs_l = m(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_l_imgs).softmax(
                    dim=1
                )
                probs_c = m(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs).softmax(
                    dim=1
                )
                probs_r = m(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_r_imgs).softmax(
                    dim=1
                )
                probs_avg = (probs_l + probs_c + probs_r) / 3.0
                ensemble_probs += probs_avg
            ensemble_probs /= len(vit_models)

            smooth = 1e-4
            ensemble_probs = (1.0 - 6 * smooth) * ensemble_probs + smooth

            probs_np = ensemble_probs.detach().cpu().numpy()
            for j in range(len(eeg_ids)):
                eid = str(eeg_ids[j])
                result[eid] = probs_np[j].astype("float64")

    for m in vit_models:
        del m
    del vit_models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
else:
    print(
        "No compatible weights found/loaded; using uniform probabilities fallback to produce a valid submission."
    )
    for eid in test["eeg_id"].astype(str).tolist():
        result[eid] = np.ones(6, dtype="float64") / 6.0



## === cell 13
sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
sub["eeg_id"] = sub["eeg_id"].astype(str)

preds = np.zeros((len(sub), 6), dtype="float64")
missing = 0
for i, eid in enumerate(sub["eeg_id"].tolist()):
    if eid in result:
        preds[i] = result[eid]
    else:
        missing += 1
        preds[i] = np.ones(6, dtype="float64") / 6.0

preds = np.clip(preds, 1e-8, 1.0)
preds = preds / preds.sum(axis=1, keepdims=True)

sub[CLASSES] = preds
sub.to_csv("submission.csv", index=False)
print("submission.csv written. missing_eeg_ids_filled:", missing)
print(sub.head())



## === cell 14
if not DEBUG:
    for p in [
        spec_directory_path,
        eeg_directory_path,
        raw_50s_directory_path,
        raw_10s_directory_path,
    ]:
        try:
            for fn in os.listdir(p):
                if fn.endswith(".npy"):
                    os.remove(os.path.join(p, fn))
        except Exception as e:
            print("Cleanup warning for", p, ":", repr(e))
    gc.collect()
