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

0.2851059022544702

# 6. Current score

0.86831

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the pipeline so it runs end-to-end on Kaggle CPU-only by (1) removing the hardcoded CUDA device and loading models on the available device, (2) ensuring missing imports and notebook-only shell commands don’t crash in a .py run, and (3) making the submission creation robust and aligned to `sample_submission.csv` with per-row probabilities that sum to 1. I also add safe fallbacks: if the external weight files or `timm` models aren’t available in your environment, the script still generate a valid submission using a uniform distribution (score be worse, but it “yield” a submission). These are execution/stability fixes; they don’t change the model architecture or inference semantics when the weights are present.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.2851), and the biggest reason is that your pipeline likely falls back to uniform predictions because `timm` (and/or the external weight files) aren’t available, which makes KL divergence very high. The minimal, score-relevant fix is to ensure `timm` is installed in the Kaggle run (via pip) so the exact same model/weights inference path is used, without changing the architecture, features, or inference math. I also make the saved-feature generation more robust and faster (deduplicate by `eeg_id` and use all CPU cores), which reduces the risk of timeouts that can silently trigger fallback behavior. Everything else (model definition, softmax, ensembling, submission formatting) is kept identical.'
- What this solution (achieved 1.41937) has done: 'I fix the hard crash caused by missing external weight files by switching to a safe, score-relevant fallback that does not require those unavailable `.pth` files: compute per-class priors from `train.csv` and use them as constant probabilities for every test `eeg_id` (this is far better than uniform for KL on this competition). I also fix the submission construction so the probability columns are guaranteed numeric floats and each row sums to 1 (the earlier “sum to 1” failure is typically caused by object dtypes after concatenation). Finally, I keep the original feature-generation and model code intact, but gate model inference so it only runs when weights are actually present; otherwise it still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 1.47425) has done: 'Your current score is dominated by the class-prior fallback because the external `.pth` weight files are not available in your environment, so the model never runs; the smallest score-relevant improvement (without changing core model/inference logic) is to make that fallback smarter. I replace the global prior with a patient-conditioned prior computed from `train.csv` (same labels, no leakage) and use it for each test row via `patient_id`, falling back to the global prior only for unseen patients. This keeps evaluation semantics (probabilities) identical and should move KL substantially toward your target. I also make sure `eeg_id` string/int alignment can’t silently miss lookups, preserving a valid submission with rows summing to 1.'
- What this solution (achieved 0.99771) has done: 'Your score is far above the target (lower is better), and since the external `.pth` weights are missing your submission is entirely driven by the fallback. To move the KL score toward the target without changing the core model/inference logic, I make the fallback closer to the true label distribution by (1) aggregating train labels per `eeg_id` (matching the test granularity) and (2) using a simple mixture of patient-conditioned prior and global prior (smoothing) so patients with few/noisy samples don’t overfit. This keeps evaluation semantics identical (valid per-row probabilities summing to 1) and only changes the fallback probabilities used when models can’t run. The rest of your pipeline remains unchanged, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.97449) has done: 'Your current score (0.99771, lower-is-better) is far worse than the target (0.2851), and since the external `.pth` weights are missing your submission is dominated by the fallback priors. The most minimal, score-relevant improvement (without touching model/feature logic) is to make the fallback closer to the evaluation granularity by using the *training distribution aggregated at `eeg_id`* and then *smoothing per-patient priors with a Dirichlet-style prior* (global prior) based on the patient’s total vote mass, not just number of EEGs. I also add a small amount of probability tempering (mix with global prior) to reduce overconfident priors, which typically improves KL. Everything else (model code, feature extraction, submission format) is kept the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.96976) has done: 'Your current score is far worse than the target (lower-is-better), and since the external `.pth` weights are missing the submission is entirely driven by the fallback priors. The minimal score-relevant improvement is to make the fallback better calibrated to the test distribution by conditioning not just on `patient_id` but also on the test-time `spectrogram_id` (available in `test.csv`) using the same Dirichlet-smoothed vote aggregation from `train.csv`. I keep your model/feature pipeline unchanged and only adjust the fallback probability computation when `result_5` is empty; this should reduce KL substantially without altering any model architecture or inference semantics. I also add robust smoothing and mixing so the fallback never becomes overconfident (important for KL), while still shifting toward more informative priors.'
- What this solution (achieved 0.90489) has done: 'Your current score (0.96976, lower-is-better) is much worse than the target (0.2851), and because the external `.pth` weights are missing your submission is entirely driven by the fallback priors. The smallest score-relevant change (without touching model/feature/inference core logic) is to improve the fallback calibration by using *patient+spectrogram conditioned priors with a better smoothing strength* and a *slightly stronger global-mix* to reduce overconfidence (KL heavily penalizes overconfident wrong probabilities). Concretely, we keep your same eeg-level aggregation, but adjust `alpha` downward (less wash-out) and `temper_mix` upward a bit (more safety against sharp priors), and add a final tiny “uniform floor” epsilon-mix to avoid extremely small probabilities. Everything else stays the same and the script still runs end-to-end and writes a valid `submission.csv` with row sums exactly 1.'
- What this solution (achieved 0.86831) has done: 'Your current score (0.90489, lower-is-better) is still far from the target (0.2851), and since external `.pth` weights are missing the only thing affecting score is the fallback. I keep your entire model/feature/inference pipeline unchanged, and only make the fallback closer to the true (hidden) label distribution by using per-`eeg_id` vote proportions and then learning a tiny linear calibration that maps fallback priors (global/patient/spectrogram) to better probabilities on a validation split of the training metadata (no leakage, no architecture change). This is a minimal change in the “no weights present” branch only, and it typically reduces KL because it fixes systematic bias and improves calibration while keeping probabilities well-behaved. I also keep your existing smoothing/mixing, and simply apply the learned calibration + renormalization before writing `submission.csv`.'

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

try:
    import librosa  # noqa: F401
except Exception:
    librosa = None



## === cell 3
from scipy import signal  # required by stft_spec_from_eeg and filters
from scipy.signal import (
    butter,
    lfilter,
)  # noqa: F401 (kept; original code imported these)




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
class Config:
    seed = 2024
    num_folds = 5




## === cell 8
def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seed_everything(Config.seed)



## === cell 9
try:
    import timm  # noqa: F401
except Exception:
    timm = None

if timm is None:
    import sys
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "timm==0.9.12"]
    )
    import timm  # noqa: F401
else:
    required_names = [
        "vit_small_patch14_reg4_dinov2.lvd142m",
        "vit_base_patch14_reg4_dinov2.lvd142m",
    ]
    missing = []
    for name in required_names:
        try:
            timm.create_model(name, pretrained=False, num_classes=6, in_chans=1)
        except Exception:
            missing.append(name)
    if missing:
        import sys
        import subprocess
        import importlib

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "timm==0.9.12"]
        )
        timm = importlib.reload(timm)



## === cell 10
import torch.utils.data as data
from torch.utils.data import DataLoader



## === cell 11
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




## === cell 12
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

        self.head = nn.Linear(384 * 2 + 768 * 2, 6)
        self.head1 = nn.Linear(384, 6)
        self.head2 = nn.Linear(384, 6)
        self.head3 = nn.Linear(768, 6)
        self.head4 = nn.Linear(768, 6)

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




## === cell 13
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

vit_models = []
model_weights = [
    "/kaggle/input/hms-bestlb-vitbase/fold_0_exp_6_bestlb.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_1_exp_6_bestlb.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_2_exp_6_bestlb.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_3_exp_6_bestlb.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_4_exp_6_bestlb.pth",
]
model_types = ["vit_base", "vit_base", "vit_base", "vit_base", "vit_base"]

weights_present = all(os.path.exists(w) for w in model_weights)
if not weights_present:
    missing = [w for w in model_weights if not os.path.exists(w)]
    print(
        "[WARN] Missing external weight files; will skip model inference and use class-prior fallback."
    )
    print("[WARN] Missing:", missing[:3], ("..." if len(missing) > 3 else ""))
else:
    required_model_names = [
        "vit_small_patch14_reg4_dinov2.lvd142m",
        "vit_base_patch14_reg4_dinov2.lvd142m",
    ]
    for name in required_model_names:
        _ = timm.create_model(name, pretrained=False, num_classes=6, in_chans=1)

    for i in range(len(model_types)):
        if model_types[i] == "vit_base":
            print(model_weights[i])
            model = Net("vit_base_patch14_reg4_dinov2.lvd142m", device).to(device)
            state = torch.load(model_weights[i], map_location=device)
            model.load_state_dict(state, strict=True)
            model.eval()
            vit_models.append(model)



## === cell 14
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


if weights_present and len(vit_models) > 0:
    test_unique = test.drop_duplicates(subset=["eeg_id"]).reset_index(drop=True)
    _ = Parallel(n_jobs=-1, prefer="processes")(
        delayed(save)(row) for _, row in test_unique.iterrows()
    )
else:
    print(
        "[INFO] Skipping feature generation because model inference is disabled (no weights)."
    )



## === cell 15
result_5 = {}

if weights_present and len(vit_models) > 0:
    test_data = ImageFolder(test, (518, 518))
    test_loader = DataLoader(
        test_data, batch_size=32, pin_memory=False, num_workers=4, drop_last=False
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
            raw_10s_l_imgs = raw_10s_l_imgs.to(device).float()
            raw_10s_c_imgs = raw_10s_c_imgs.to(device).float()
            raw_10s_r_imgs = raw_10s_r_imgs.to(device).float()

            ensemble_probs = torch.zeros((spec_imgs.shape[0], 6), device=device)
            for model in vit_models:
                model.eval()
                logits_l, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_l_imgs
                )
                logits_c, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs
                )
                logits_r, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_r_imgs
                )

                probs_l = logits_l.softmax(dim=1)
                probs_c = logits_c.softmax(dim=1)
                probs_r = logits_r.softmax(dim=1)
                ensemble_probs += (probs_l + probs_c + probs_r) / 3.0

            ensemble_probs /= float(len(vit_models))
            ensemble_probs = ensemble_probs.detach().cpu().numpy()

            for j in range(len(eeg_ids)):
                eeg_id = str(eeg_ids[j])
                if eeg_id not in result_5:
                    result_5[eeg_id] = np.zeros((6,), dtype=np.float64)
                result_5[eeg_id] += ensemble_probs[j].astype(np.float64)
else:
    print(
        "[INFO] Model inference skipped; will use class-prior fallback at submission time."
    )



## === cell 16
for model in vit_models:
    del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()



## === cell 17
sample_sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
test_meta = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
)


def _softmax_np(z):
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def _fit_dirichlet_logit_calibrator(P, Y, l2=1e-3, iters=400, lr=0.2, seed=2024):
    rng = np.random.default_rng(seed)
    n, k = P.shape
    eps = 1e-12
    X = np.log(np.clip(P, eps, 1.0))  # (n,k)
    W = 0.01 * rng.standard_normal((k, k))
    b = np.zeros((k,), dtype=np.float64)

    for _ in range(iters):
        Z = X @ W + b  # (n,k)
        Q = _softmax_np(Z)  # (n,k)
        G = (Q - Y) / float(n)  # (n,k)
        gW = X.T @ G + l2 * W
        gb = G.sum(axis=0)
        W -= lr * gW
        b -= lr * gb

    return W, b


def _apply_calibrator(P, W, b):
    eps = 1e-12
    X = np.log(np.clip(P, eps, 1.0))
    Q = _softmax_np(X @ W + b)
    return Q


if len(result_5) == 0:
    train = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv",
        usecols=["eeg_id", "patient_id", "spectrogram_id"] + CLASSES,
    )
    train["patient_id"] = train["patient_id"].astype(np.int64)
    train["spectrogram_id"] = train["spectrogram_id"].astype(np.int64)
    train["eeg_id"] = train["eeg_id"].astype(np.int64)
    train[CLASSES] = train[CLASSES].astype(np.float64)

    eeg_agg = (
        train.groupby("eeg_id", as_index=False)
        .agg(
            {
                **{c: "sum" for c in CLASSES},
                "patient_id": "first",
                "spectrogram_id": "first",
            }
        )
        .reset_index(drop=True)
    )

    y_counts = eeg_agg[CLASSES].values.astype(np.float64)
    y_counts = np.clip(y_counts, 1e-12, None)
    y_true = y_counts / y_counts.sum(axis=1, keepdims=True)

    global_counts = y_counts.sum(axis=0).astype(np.float64)
    global_counts = np.clip(global_counts, 1e-12, None)
    global_prior = global_counts / global_counts.sum()

    patient_counts_df = (
        eeg_agg.groupby("patient_id", as_index=True)[CLASSES].sum().astype(np.float64)
    )
    patient_vote_mass = patient_counts_df.sum(axis=1).astype(np.float64)

    spec_counts_df = (
        eeg_agg.groupby("spectrogram_id", as_index=True)[CLASSES]
        .sum()
        .astype(np.float64)
    )
    spec_vote_mass = spec_counts_df.sum(axis=1).astype(np.float64)

    alpha = 80.0

    patient_smoothed = patient_counts_df.add(alpha * global_prior, axis=1)
    patient_smoothed = patient_smoothed.clip(lower=1e-12)
    patient_smoothed = patient_smoothed.div(patient_smoothed.sum(axis=1), axis=0)

    spec_smoothed = spec_counts_df.add(alpha * global_prior, axis=1)
    spec_smoothed = spec_smoothed.clip(lower=1e-12)
    spec_smoothed = spec_smoothed.div(spec_smoothed.sum(axis=1), axis=0)

    patient_prior = {int(pid): row.values for pid, row in patient_smoothed.iterrows()}
    patient_mass = {int(pid): float(m) for pid, m in patient_vote_mass.items()}

    spec_prior = {int(sid): row.values for sid, row in spec_smoothed.iterrows()}
    spec_mass = {int(sid): float(m) for sid, m in spec_vote_mass.items()}

    temper_mix = 0.08
    uniform_mix = 0.01

    ids = eeg_agg["eeg_id"].values.astype(np.int64)
    pid_arr = eeg_agg["patient_id"].values.astype(np.int64)
    sid_arr = eeg_agg["spectrogram_id"].values.astype(np.int64)

    def _fallback_one(pid, sid):
        p_global = global_prior

        if int(pid) in patient_prior:
            mp = float(patient_mass.get(int(pid), 0.0))
            wp = mp / (mp + float(alpha))
            p_patient = wp * patient_prior[int(pid)] + (1.0 - wp) * p_global
        else:
            p_patient = p_global

        if int(sid) in spec_prior:
            ms = float(spec_mass.get(int(sid), 0.0))
            ws = ms / (ms + float(alpha))
            p_spec = ws * spec_prior[int(sid)] + (1.0 - ws) * p_global
        else:
            p_spec = p_global

        mp_eff = (
            float(patient_mass.get(int(pid), 0.0)) if int(pid) in patient_prior else 0.0
        )
        ms_eff = float(spec_mass.get(int(sid), 0.0)) if int(sid) in spec_prior else 0.0
        denom = mp_eff + ms_eff + 1e-12
        lam_p = mp_eff / denom
        lam_s = ms_eff / denom
        if denom <= 1e-6:
            p = p_global
        else:
            p = lam_p * p_patient + lam_s * p_spec

        p = (1.0 - float(temper_mix)) * p + float(temper_mix) * p_global
        p = (1.0 - float(uniform_mix)) * p + float(uniform_mix) * (np.ones(6) / 6.0)
        p = np.clip(p, 1e-12, None)
        p = p / p.sum()
        return p

    P0 = np.vstack(
        [_fallback_one(pid, sid) for pid, sid in zip(pid_arr, sid_arr)]
    ).astype(np.float64)

    rng = np.random.default_rng(Config.seed)
    idx = np.arange(len(eeg_agg))
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]

    W_cal, b_cal = _fit_dirichlet_logit_calibrator(
        P0[tr_idx], y_true[tr_idx], l2=1e-3, iters=350, lr=0.25, seed=Config.seed
    )
    cal_mix = 0.75

    print(
        "[INFO] Using eeg-level (patient + spectrogram) Dirichlet-smoothed prior fallback + learned calibrator; global prior:",
        dict(zip(CLASSES, global_prior.round(6))),
    )
else:
    global_prior = None
    patient_prior = None
    patient_mass = None
    spec_prior = None
    spec_mass = None
    alpha = None
    temper_mix = None
    uniform_mix = None
    W_cal, b_cal, cal_mix = None, None, None

sample_sub["eeg_id"] = sample_sub["eeg_id"].astype(np.int64)
test_meta["eeg_id"] = test_meta["eeg_id"].astype(np.int64)
test_meta["patient_id"] = test_meta["patient_id"].astype(np.int64)
test_meta["spectrogram_id"] = test_meta["spectrogram_id"].astype(np.int64)

eeg_to_patient = dict(zip(test_meta["eeg_id"].values, test_meta["patient_id"].values))
eeg_to_spec = dict(zip(test_meta["eeg_id"].values, test_meta["spectrogram_id"].values))

pred_mat = np.zeros((len(sample_sub), 6), dtype=np.float64)
for i, eeg_id in enumerate(sample_sub["eeg_id"].values):
    key = str(eeg_id)
    if key in result_5:
        pred_mat[i] = result_5[key]
    else:
        if global_prior is None:
            pred_mat[i] = 1.0 / 6.0
        else:
            pid = int(eeg_to_patient.get(int(eeg_id), -1))
            sid = int(eeg_to_spec.get(int(eeg_id), -1))

            p_global = global_prior

            if pid in patient_prior:
                mp = float(patient_mass.get(pid, 0.0))
                wp = mp / (mp + float(alpha))
                p_patient = wp * patient_prior[pid] + (1.0 - wp) * p_global
            else:
                p_patient = p_global

            if sid in spec_prior:
                ms = float(spec_mass.get(sid, 0.0))
                ws = ms / (ms + float(alpha))
                p_spec = ws * spec_prior[sid] + (1.0 - ws) * p_global
            else:
                p_spec = p_global

            mp_eff = float(patient_mass.get(pid, 0.0)) if pid in patient_prior else 0.0
            ms_eff = float(spec_mass.get(sid, 0.0)) if sid in spec_prior else 0.0
            denom = mp_eff + ms_eff + 1e-12
            lam_p = mp_eff / denom
            lam_s = ms_eff / denom
            if denom <= 1e-6:
                p = p_global
            else:
                p = lam_p * p_patient + lam_s * p_spec

            p = (1.0 - float(temper_mix)) * p + float(temper_mix) * p_global
            p = (1.0 - float(uniform_mix)) * p + float(uniform_mix) * (np.ones(6) / 6.0)

            p = np.clip(p, 1e-12, None)
            p = p / p.sum()
            p_cal = _apply_calibrator(p.reshape(1, -1), W_cal, b_cal).reshape(-1)
            p = (1.0 - float(cal_mix)) * p + float(cal_mix) * p_cal

            pred_mat[i] = p

eps = 1e-12
pred_mat = np.clip(pred_mat, eps, None)
pred_mat = pred_mat / pred_mat.sum(axis=1, keepdims=True)

submission = pd.DataFrame({"eeg_id": sample_sub["eeg_id"].values})
for k, c in enumerate(CLASSES):
    submission[c] = pred_mat[:, k].astype(np.float64)

row_sums = submission[CLASSES].sum(axis=1).values
submission[CLASSES] = (submission[CLASSES].values / row_sums.reshape(-1, 1)).astype(
    np.float64
)

submission.to_csv("submission.csv", index=False)

max_abs_err = np.max(np.abs(submission[CLASSES].sum(axis=1).values - 1.0))
print(submission.head())
print("Max |row_sum-1|:", max_abs_err)
print("Wrote submission.csv with shape:", submission.shape)



## === cell 18
if not DEBUG:
    import shutil

    for p in [
        "/kaggle/working/spec_spectrograms/",
        "/kaggle/working/eeg_spectrograms/",
        "/kaggle/working/eeg_50s_raws/",
        "/kaggle/working/eeg_10s_raws/",
        "/kaggle/working/squeezeformer",
    ]:
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
        except Exception as e:
            print(f"[WARN] Could not remove {p}: {e}")
