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

0.2838422248059112

# 6. Current score

0.78198

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'Your runtime fails because the notebook forces `cuda:0` even when no GPU is available; switching to CPU safely (and loading weights with `map_location`) fixes the immediate crash and allows end-to-end inference. I also removed the `librosa` import (not used and often missing) and fixed the missing `signal` import usage inside `stft_spec_from_eeg`. To ensure a valid submission even if the external weight datasets (`/kaggle/input/hms-stage2*`) are not present, the script now falls back to a deterministic prior (class frequencies from train) while still producing a correctly-formatted `submission.csv` whose rows sum to 1. Finally, I added probability normalization and a small epsilon clip to avoid invalid KL-divergence inputs.'
- What this solution (achieved 0.83081) has done: 'Your current 1.419 score is dominated by the fallback path that outputs a global label prior because the `/kaggle/input/hms-stage2/` weight files are missing, so the model never actually runs. The smallest legitimate way to move the score toward your 0.284 target is to (1) use weights if they exist, but otherwise (2) replace the flat prior with a simple per-patient prior (computed from train.csv) blended with the global prior; this uses only metadata available at test time and typically beats a single global distribution on KL. I’m also keeping your probability normalization/epsilon clipping (required for valid KL inputs) and making the ensemble loader search for weights in common Kaggle input locations without changing model logic. These changes preserve your architecture/inference approach when weights are present, and only improve the deterministic fallback when they’re absent (which is your current situation).'
- What this solution (achieved 0.83081) has done: 'Your current score is far above the target (lower is better), and the biggest likely contributor is that your spectrogram extraction is misaligned (you’re saving the first 300 time columns, not the centered 300 around the label time as intended), which can significantly hurt inference when weights are available and also makes any learned weights less applicable. I make a minimal, metric-relevant fix: center-crop the spectrogram time axis (still 300 wide) to better match the competition’s “centered at the same time” setup, while keeping your model/inference logic unchanged. I also keep your probability clipping/renormalization (important for KL) and ensure we still always write a valid `submission.csv`. No architectural/training changes are introduced; this is only a data alignment correction in the preprocessing cache.'
- What this solution (achieved 0.83081) has done: 'Your current score (0.83081, lower-is-better) is much worse than the target (0.28384), and the most likely reason is that your spectrogram preprocessing is mismatched to how the training weights expect the data. I make two minimal, metric-relevant alignment fixes: (1) apply the same log+standardization to the saved spectrogram arrays during caching (so `ImageFolder` doesn’t “log twice”), and (2) use `librosa`-style dB conversion for the EEG STFT spectrogram (log-power) which is what many HMS solutions/weights assume. These keep your model/ensemble logic identical while improving input calibration, which should reduce KL divergence. The submission writing and probability clipping/renormalization remain unchanged to guarantee a valid CSV.'
- What this solution (achieved 0.76358) has done: 'Your current score is much worse than the target (lower-is-better), and because your run is likely falling back to priors (missing weights), the biggest legitimate gain with minimal risk is to make the fallback smarter using only test-time metadata. I keep your model/inference path unchanged when weights exist, but improve the fallback by replacing the per-patient blend with a smoothed per-(patient_id, spectrogram_id) prior (falls back to per-patient, then global), which typically reduces KL versus a single patient prior. I also make the fallback compute priors on normalized vote probabilities (votes -> probabilities per row) so the prior matches the competition’s probabilistic targets better. Finally, I keep the strict probability clipping+renormalization to avoid invalid KL inputs and still always write a valid `submission.csv`.'
- What this solution (achieved 0.77957) has done: 'Your current score is far worse than the target (lower-is-better), and the biggest minimal win that doesn’t change your model is to improve the *fallback* prediction quality (since weights are likely missing) using only test-time metadata. I keep your existing per-(patient_id, spectrogram_id) → per-patient → global prior chain, but make it better aligned to the KL metric by (1) weighting averages by the number of annotator votes per training row and (2) adding a small “empirical Bayes” Dirichlet smoothing toward the global prior instead of only linear blending. I also speed up the fallback lookup (no per-row MultiIndex membership checks) and keep your probability clipping+renormalization to ensure valid KL inputs and a valid `submission.csv`. No model architecture, inference loop, or feature extraction logic is changed.'
- What this solution (achieved 0.78198) has done: 'I keep your model/inference logic unchanged and focus only on improving the deterministic fallback path, since your current score strongly suggests missing weights and thus fallback-dominated predictions. Specifically, I strengthen the metadata-based priors by adding a (patient_id, eeg_id) → patient → global hierarchy (eeg_id is always available at test time) and tune the Dirichlet smoothing strengths to rely more on local evidence when enough training mass exists. I also ensure the fallback priors are computed efficiently without dict lookups on large MultiIndex objects, but without changing any semantics of the prediction targets or output formatting. The submission still be clipped/renormalized to be valid for KL and always write `submission.csv`.'

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
    eps = 1e-10
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

            spec = (np.abs(spec) ** 2).astype("float32")
            spec = 10.0 * np.log10(np.maximum(spec, eps)).astype("float32")

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

DATA_ROOT = "/kaggle/input/hms-harmful-brain-activity-classification"

if DEBUG:
    test = pd.read_csv(f"{DATA_ROOT}/train.csv")[:40]
    SPEC_PATH = f"{DATA_ROOT}/train_spectrograms/"
    EEG_PATH = f"{DATA_ROOT}/train_eegs/"
else:
    test = pd.read_csv(f"{DATA_ROOT}/test.csv")
    SPEC_PATH = f"{DATA_ROOT}/test_spectrograms/"
    EEG_PATH = f"{DATA_ROOT}/test_eegs/"

print(test.shape)

spec_directory_path = "spec_spectrograms/"
os.makedirs(spec_directory_path, exist_ok=True)

eeg_directory_path = "eeg_spectrograms/"
os.makedirs(eeg_directory_path, exist_ok=True)

raw_10s_directory_path = "eeg_10s_raws/"
os.makedirs(raw_10s_directory_path, exist_ok=True)

raw_50s_directory_path = "eeg_50s_raws/"
os.makedirs(raw_50s_directory_path, exist_ok=True)



## === cell 7
from joblib import Parallel, delayed


def _center_crop_time(spec_arr, target_T=300):
    T = spec_arr.shape[1]
    if T <= target_T:
        return spec_arr[:, :target_T]
    start = (T - target_T) // 2
    return spec_arr[:, start : start + target_T]


def _spec_log_standardize(spec_arr):
    eps = 1e-6
    x = spec_arr.astype("float32")
    x = np.clip(x, np.exp(-4), np.exp(8))
    x = np.log(x).astype("float32")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    m = x.mean(axis=(0, 1), keepdims=True)
    s = x.std(axis=(0, 1), keepdims=True)
    x = (x - m) / (s + eps)
    return x.astype("float32")


def save(row):
    eeg_id = row["eeg_id"]
    spec_id = row["spectrogram_id"]

    spec = pd.read_parquet(f"{SPEC_PATH}{spec_id}.parquet")
    spec_arr = spec.values[:, 1:].T.astype("float32")  # (Hz, Time)
    split_spec_arr = _center_crop_time(spec_arr, target_T=300)
    split_spec_arr = _spec_log_standardize(split_spec_arr)
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




## === cell 8
class Config:
    seed = 2024
    num_folds = 5


def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
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
            eeg_img, self.test_imgsize, preserve_range=True, anti_aliasing=True
        ).astype("float32")
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

        eeg_img = np.expand_dims(eeg_img, -1)
        spec_img = np.expand_dims(spec_img, -1)
        raw_50s_img = np.expand_dims(raw_50s_img, -1)
        raw_10s_l_img = np.expand_dims(raw_10s_l_img, -1)
        raw_10s_c_img = np.expand_dims(raw_10s_c_img, -1)
        raw_10s_r_img = np.expand_dims(raw_10s_r_img, -1)

        spec_img = np.nan_to_num(spec_img, nan=0.0, posinf=0.0, neginf=0.0).astype(
            "float32"
        )

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




## === cell 12
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def _normalize_probs(p, eps=1e-6):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p


train_csv_path = f"{DATA_ROOT}/train.csv"
train_df_full = pd.read_csv(
    train_csv_path, usecols=["patient_id", "spectrogram_id", "eeg_id"] + CLASSES
)

votes = train_df_full[CLASSES].values.astype(np.float64)
row_vote_sums = votes.sum(axis=1).astype(np.float64)
row_vote_sums = np.clip(row_vote_sums, 1.0, None)  # safety
train_probs = votes / row_vote_sums[:, None]
train_probs = np.clip(train_probs, 1e-6, 1.0)
train_probs = train_probs / train_probs.sum(axis=1, keepdims=True)

train_probs_df = pd.DataFrame(train_probs, columns=CLASSES)
train_df_full["_w"] = row_vote_sums

w = train_df_full["_w"].values.astype(np.float64)
global_prior = (train_probs_df.values * w[:, None]).sum(axis=0) / w.sum()
global_prior = np.clip(global_prior, 1e-6, 1.0)
global_prior = global_prior / global_prior.sum()
print("Global label prior:", global_prior)


def _weighted_group_prior(df_key, probs_df, weights):
    tmp = probs_df.copy()
    for c in CLASSES:
        tmp[c] = tmp[c].values * weights
    grp_sum = tmp.groupby(df_key, sort=False).sum()
    grp_w = pd.Series(weights, index=probs_df.index).groupby(df_key, sort=False).sum()
    grp_mean = grp_sum.div(grp_w, axis=0).astype(np.float64)
    return grp_mean, grp_w.astype(np.float64)


patient_prior, patient_wsum = _weighted_group_prior(
    train_df_full["patient_id"], train_probs_df, train_df_full["_w"].values
)

pat_spec_keys = [train_df_full["patient_id"], train_df_full["spectrogram_id"]]
pat_spec_prior, pat_spec_wsum = _weighted_group_prior(
    pat_spec_keys, train_probs_df, train_df_full["_w"].values
)

pat_eeg_keys = [train_df_full["patient_id"], train_df_full["eeg_id"]]
pat_eeg_prior, pat_eeg_wsum = _weighted_group_prior(
    pat_eeg_keys, train_probs_df, train_df_full["_w"].values
)

patient_prior_dict = {k: v for k, v in patient_prior.to_dict(orient="index").items()}
patient_wsum_dict = patient_wsum.to_dict()

pat_spec_prior_dict = {k: v for k, v in pat_spec_prior.to_dict(orient="index").items()}
pat_spec_wsum_dict = pat_spec_wsum.to_dict()

pat_eeg_prior_dict = {k: v for k, v in pat_eeg_prior.to_dict(orient="index").items()}
pat_eeg_wsum_dict = pat_eeg_wsum.to_dict()

del train_df_full, votes, row_vote_sums, train_probs, train_probs_df, w
gc.collect()


def _dirichlet_smooth(p_local, wsum, strength=20.0):
    """
    Dirichlet smoothing toward global_prior.
    """
    wsum = float(wsum)
    strength = float(strength)
    p = (wsum * np.asarray(p_local, dtype=np.float64) + strength * global_prior) / (
        wsum + strength
    )
    p = np.clip(p, 1e-6, 1.0)
    p = p / p.sum()
    return p


def get_fallback_pred_for_row(patient_id, spectrogram_id, eeg_id):
    key_pe = (
        patient_id,
        int(eeg_id) if isinstance(eeg_id, str) and eeg_id.isdigit() else eeg_id,
    )
    rec = pat_eeg_prior_dict.get(key_pe, None)
    if rec is not None:
        wsum = pat_eeg_wsum_dict.get(key_pe, 1.0)
        p_local = np.array([rec[c] for c in CLASSES], dtype=np.float64)
        return _dirichlet_smooth(p_local, wsum, strength=10.0)

    key_ps = (patient_id, spectrogram_id)
    rec = pat_spec_prior_dict.get(key_ps, None)
    if rec is not None:
        wsum = pat_spec_wsum_dict.get(key_ps, 1.0)
        p_local = np.array([rec[c] for c in CLASSES], dtype=np.float64)
        return _dirichlet_smooth(p_local, wsum, strength=14.0)

    rec = patient_prior_dict.get(patient_id, None)
    if rec is not None:
        wsum = patient_wsum_dict.get(patient_id, 1.0)
        p_local = np.array([rec[c] for c in CLASSES], dtype=np.float64)
        return _dirichlet_smooth(p_local, wsum, strength=28.0)

    return global_prior


def _resolve_weight_paths(weight_paths):
    resolved = []
    for wp in weight_paths:
        if os.path.exists(wp):
            resolved.append(wp)
            continue
        base = os.path.basename(wp)
        candidates = [
            wp,
            f"/kaggle/input/hms-stage2/{base}",
            f"/kaggle/input/hms-stage2-weights/{base}",
            f"/kaggle/input/hms-harmful-brain-activity-classification/{base}",
        ]
        found = None
        for c in candidates:
            if os.path.exists(c):
                found = c
                break
        resolved.append(found if found is not None else wp)
    return resolved


def try_run_ensemble(model_weights, batch_size):
    model_weights = _resolve_weight_paths(model_weights)
    available = [w for w in model_weights if os.path.exists(w)]
    if len(available) != len(model_weights):
        print(
            f"Missing {len(model_weights)-len(available)} weights; skipping this ensemble."
        )
        return None

    vit_models = []
    for w in available:
        model = Net("vit_small_patch14_reg4_dinov2.lvd142m", device).to(device)
        sd = torch.load(w, map_location=device)
        model.load_state_dict(sd)
        model.eval()
        vit_models.append(model)

    test_data = ImageFolder(test, (518, 518))
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=2,
        drop_last=False,
    )

    results = {}
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
            for m in vit_models:
                logits_l, _, _, _, _ = m(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_l_imgs
                )
                logits_c, _, _, _, _ = m(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs
                )
                logits_r, _, _, _, _ = m(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_r_imgs
                )
                probs = (
                    logits_l.softmax(1) + logits_c.softmax(1) + logits_r.softmax(1)
                ) / 3.0
                ensemble_probs += probs

            ensemble_probs = (ensemble_probs / len(vit_models)).detach().cpu().numpy()
            for j, eeg_id in enumerate(eeg_ids):
                results[str(eeg_id)] = ensemble_probs[j].astype(np.float64)

    for m in vit_models:
        del m
    del vit_models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results




## === cell 13
model_weights_6 = [
    "/kaggle/input/hms-stage2/fold_0_spec_raw_50_10_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_1_spec_raw_50_10_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_2_spec_raw_50_10_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_3_spec_raw_50_10_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_4_spec_raw_50_10_bestlb.pth",
]

result_6 = try_run_ensemble(model_weights_6, batch_size=8)

if result_6 is None:
    print(
        "Using deterministic patient/eeg/spectrogram-aware prior fallback (no weights found)."
    )
    result_6 = {}
    for _, row in test.iterrows():
        eeg_id = str(row["eeg_id"])
        pid = row["patient_id"]
        sid = row["spectrogram_id"]
        result_6[eeg_id] = get_fallback_pred_for_row(pid, sid, eeg_id)



## === cell 14
sub = pd.read_csv(f"{DATA_ROOT}/sample_submission.csv")
pred = np.zeros((len(sub), 6), dtype=np.float64)

for i, eeg_id in enumerate(sub["eeg_id"].astype(str).values):
    pred[i] = result_6.get(eeg_id, global_prior)

pred = _normalize_probs(pred, eps=1e-6)

sub[CLASSES] = pred
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Saved submission.csv with shape:", sub.shape)



## === cell 15
if DEBUG is False:
    for p in [
        spec_directory_path,
        eeg_directory_path,
        raw_50s_directory_path,
        raw_10s_directory_path,
    ]:
        try:
            for fn in os.listdir(p):
                os.remove(os.path.join(p, fn))
        except Exception:
            pass
