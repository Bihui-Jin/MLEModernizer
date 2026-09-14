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

0.84744

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the runtime blockers so the notebook can execute end-to-end on a CPU-only Kaggle session and still generate a valid `submission.csv`. The main issue is hardcoding `cuda:0`; I switch to an automatic device selection and load checkpoints with `map_location` so models can run on CPU. Because your external model weight paths may not exist in this environment (and CPU inference with multiple large ViTs is too slow), I add a safe fallback that produces a valid probability submission using the empirical class prior from `train.csv` (score be worse than your target but yield a valid submission). I also remove the Jupyter `!rm` shell lines (they break in a `.py` run) and ensure probabilities are normalized to sum to 1.'
- What this solution (achieved 0.73126) has done: 'Your current score (1.41937, lower-is-better) is far worse than the target (0.2850), and the main reason is that you’re almost certainly falling back to the global class-prior submission because the external weight files don’t exist. The smallest legitimate improvement is to replace that fallback with a simple, well-calibrated per-patient prior computed from `train.csv` (still a “no-model” baseline, but much closer to realistic test distribution than a global prior). I also ensure `eeg_id` stays integer (as in sample_submission) to avoid any potential merge/alignment quirks, and I keep the probability clipping+renormalization for KL stability. No model architecture/training logic is changed; only the fallback prediction logic is improved to move the score toward your target.'
- What this solution (achieved 0.93865) has done: 'Your current submission is almost certainly coming from the fallback “patient prior” rather than the ViT ensemble (weights are missing), so the only realistic way to move the KL score toward your target is to make that fallback closer to the hidden test distribution without changing your core model logic. I keep the entire feature extraction + model code intact and only improve the fallback by (1) averaging labels at the `eeg_id` level (to match the evaluation unit) before computing priors, and (2) using a shrinkage/Dirichlet-smoothed per-patient distribution based on that patient’s effective sample size (more reliable than a fixed alpha). I also ensure the output probabilities are clipped + renormalized and that the submission rows align exactly to `test.csv` order and schema. These are minimal changes directly tied to improving the metric under the “no-weights” path.'
- What this solution (achieved 1.21915) has done: 'Your current score (0.93865, lower-is-better) is far from the target (0.2850), and given the missing external weights, the only viable path is improving the fallback so its predicted distribution is closer to the hidden test distribution. I keep all feature extraction and model code intact, and only change the fallback to (1) compute per-patient priors using an empirical-Bayes Dirichlet posterior mean at the eeg_id level (evaluation unit), and (2) blend that patient posterior with a global posterior using a small fixed mixing weight for better out-of-patient generalization. I also make the patient key lookup robust by building dicts (avoids pandas index dtype mismatches) and keep the same probability clipping+renormalization required for KL stability. This is a minimal change that directly targets the metric under the “no-weights” path and should improve toward your target without changing model semantics.'
- What this solution (achieved 0.86242) has done: 'Your score is far worse than the target (lower-is-better), and since the external model weights are missing you’re effectively submitting the fallback priors. The smallest meaningful improvement without changing any model/feature logic is to make the fallback better match the evaluation unit (`eeg_id`) and reduce calibration error by using a patient posterior built from **Dirichlet-smoothed per-eeg aggregated votes** (not per-row means), plus a small global blend. I also ensure the fallback uses proper pseudo-count updates (alpha + summed votes) and keep the same clipping+renormalization so the KL metric remains stable and the submission always validates. No architecture, feature extraction, or inference code is changed—only the no-weights fallback probability estimator.'
- What this solution (achieved 0.86242) has done: 'Your current score (0.86242, lower-is-better) is far from the target (0.2850), and since the external weights aren’t available you’re effectively relying on the fallback priors. I keep all model/feature logic untouched and only make the fallback closer to the evaluation distribution by conditioning not just on `patient_id` but also on `spectrogram_id` (test provides it), using the same Dirichlet-smoothed posterior mean and a small global blend for robustness. This is a minimal change that uses only metadata already available at inference time, and it should move KL down toward the target. I also keep the existing clipping+renormalization to ensure valid probabilities for the KL metric and submission validity.'
- What this solution (achieved 0.85255) has done: 'Your current score is much worse than the target (lower is better), and since the external weights are not present you are scoring entirely on the metadata-based fallback. I keep your whole feature pipeline + model code unchanged and only improve the fallback to better match the evaluation unit by (a) aggregating train votes at `eeg_id` (not `eeg_id,spectrogram_id`), and (b) building a **test-time prior conditioned on `patient_id` + a coarse spectrogram-derived bucket** (computed from train spectrogram parquet content), with Dirichlet smoothing and a small global blend for robustness. This uses only information available at inference time and typically moves KL down versus patient/spectrogram-id matching (which does not transfer to test because spectrogram_ids differ). The rest stays the same, and we still clip+renormalize probabilities to guarantee a valid submission.'
- What this solution (achieved 0.85215) has done: 'Your current score (0.85255, lower-is-better) is far above the target (0.2850), and since the external weights are missing you’re effectively scoring on the metadata fallback. I keep your entire feature extraction + model/inference code unchanged and only adjust the fallback to use a stronger, still-legitimate prior: a **hierarchical Dirichlet posterior** conditioned on `patient_id` plus a simple EEG-derived bucket computed from the already-extracted raw EEG features (`raw_50s`), which is available at test time. This replaces the expensive per-spectrogram parquet bucketing (slow and noisy) with a more directly relevant EEG summary, and blends (patient+bucket) → patient → global in a stable way. I keep probability clipping+renormalization and the exact submission schema so it always validates.'
- What this solution (achieved 0.84744) has done: 'Your current score (0.85215, lower-is-better) is far above the target (0.2850), and given `all_weights_exist=False` you’re effectively scoring on the metadata fallback. I keep your model/feature extraction unchanged and only strengthen the fallback by (1) using the *test-available* `spectrogram_id` (strong proxy signal) via a hierarchical Dirichlet posterior: (patient_id, spectrogram_id) → spectrogram_id → patient_id → global, and (2) using vote-sum pseudo-counts (not normalized means) with light smoothing + clipping/renorm for KL stability. This is a minimal, inference-only change that should move the KL score downward toward your target while still producing a valid `submission.csv`. The rest of the pipeline (feature saving, dataset, Net, inference-if-weights) remains intact.'
- What this solution (achieved 0.84744) has done: 'Your current score (0.84744, lower-is-better) is far above the target (0.2850), and given `all_weights_exist=False` the submission is coming from the metadata fallback. The smallest change likely to reduce KL is to stop using `spectrogram_id` for conditioning (it does not generalize from train to test because ids differ) and instead condition on a stable, test-computable EEG “bucket” derived from the already-saved `raw_50s` features. I keep your whole feature extraction and model/inference code intact, and only change the fallback to a hierarchical Dirichlet posterior: (patient_id, bucket) → patient_id → global, with the same clipping+renormalization to guarantee a valid submission. This should move the score downward toward the target without changing any core model semantics.'
- What this solution (achieved 0.84744) has done: 'Your current score (0.84744, lower-is-better) is far above the target (0.2850), and because the external weights are missing you’re entirely dependent on the fallback. The smallest change likely to reduce KL without touching your model/feature code is to replace the coarse 4-bin “EEG bucket” with a slightly richer, still-cheap bucket built from the already-saved `raw_50s` arrays (variance, line-length proxy, and bandpower ratios via Welch), which is more aligned with seizure/periodic patterns. Then we keep the same hierarchical Dirichlet posterior logic but condition on `(patient_id, rich_bucket)` → `patient_id` → global, with the same smoothing, clipping, and renormalization to guarantee valid submissions. Everything else (feature extraction, model definition, inference path, and submission schema) stays unchanged.'
- What this solution (achieved 0.84744) has done: 'Your current score (0.84744, lower-is-better) is far above the target (0.2850), and because weights are missing you’re entirely scoring on the fallback prior; the most direct way to move KL down is to make that fallback use a stronger, test-available grouping signal without changing any model/feature code. I keep your full feature extraction, dataset, model, and inference path untouched, and only adjust the fallback to use a small k-means clustering over the already-saved `raw_50s` features to create a more meaningful “EEG phenotype” bucket than the current hand-crafted thresholds. Then I use the same hierarchical Dirichlet posterior as before but conditioned on `(patient_id, cluster)` → `patient_id` → global, with the same clipping+renormalization to ensure valid KL-safe probabilities. This is minimal, runs fast on CPU (sampling + small k-means), and should reduce KL versus the current coarse bucket while preserving evaluation semantics.'
- What this solution (achieved 0.84744) has done: 'Your current score (0.84744, lower-is-better) is far above the target (0.2850), and since the external weights are missing you’re entirely scored on the fallback. I keep your full feature extraction + model code untouched and only adjust the fallback to use a stronger, still test-available grouping: a hierarchical Dirichlet posterior conditioned on `(patient_id, eeg_cluster)` **and** `(patient_id, spectrogram_id)` with robust backoff and small global blending, while keeping the same KL-safe clipping+renormalization. This is a minimal inference-only change and should move the KL downward because it lets the fallback adapt when either the EEG phenotype cluster or spectrogram context is informative for that patient. The submission format, row order, and probability normalization remain exactly correct.'

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
def _raw50_embed_from_npy(raw50_npy_path, sfreq=200):
    """
    Minimal test-time embedding from raw_50s.npy used ONLY in fallback.
    Chosen to be cheap and stable: amplitude + line length + bandpower ratios.
    """
    x = np.load(raw50_npy_path).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)
    sig = x.reshape(-1)

    rms = float(np.sqrt(np.mean(sig * sig) + 1e-12))
    ll = float(np.mean(np.abs(np.diff(sig))) + 1e-12)

    f, pxx = signal.welch(sig, fs=sfreq, nperseg=512, noverlap=256, scaling="density")
    pxx = np.maximum(pxx, 1e-20)

    def band_power(lo, hi):
        m = (f >= lo) & (f < hi)
        if not np.any(m):
            return 1e-20
        return float(np.trapz(pxx[m], f[m]) + 1e-20)

    p_delta = band_power(0.5, 4.0)
    p_theta = band_power(4.0, 8.0)
    p_alpha = band_power(8.0, 13.0)
    p_beta = band_power(13.0, 30.0)
    p_total = p_delta + p_theta + p_alpha + p_beta

    r_delta = p_delta / p_total
    r_theta = p_theta / p_total
    r_alpha = p_alpha / p_total
    r_beta = p_beta / p_total

    emb = np.array(
        [np.log1p(rms), np.log1p(ll), r_delta, r_theta, r_alpha, r_beta],
        dtype=np.float32,
    )
    return emb


def _kmeans_fit(X, k, seed=2024, n_iter=25):
    """
    Tiny numpy k-means (no external deps). Used only in fallback when weights missing.
    """
    rs = np.random.RandomState(seed)
    n = X.shape[0]
    if n == 0:
        return None, None
    k = int(min(k, n))
    idx = rs.choice(n, size=k, replace=False)
    C = X[idx].copy()

    for _ in range(n_iter):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        a = d2.argmin(axis=1)
        C_new = C.copy()
        for j in range(k):
            m = a == j
            if np.any(m):
                C_new[j] = X[m].mean(axis=0)
            else:
                C_new[j] = X[rs.randint(0, n)]
        if np.max(np.abs(C_new - C)) < 1e-6:
            C = C_new
            break
        C = C_new
    return C, a


def _kmeans_predict(X, C):
    d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1).astype(np.int16)


if pred_dict is None:
    train = pd.read_csv(
        f"{DATA_ROOT}/train.csv",
        usecols=["eeg_id", "patient_id", "spectrogram_id"] + CLASSES,
    )

    votes_eeg = train.groupby("eeg_id", sort=False)[CLASSES].sum()
    meta_eeg = (
        train.groupby("eeg_id", sort=False)[["patient_id", "spectrogram_id"]]
        .agg(lambda x: x.iloc[0])
        .astype(np.int64)
    )
    eeg_votes = meta_eeg.join(votes_eeg, how="inner").reset_index()

    train_eids = eeg_votes["eeg_id"].astype(np.int64).values
    rs = np.random.RandomState(Config.seed)
    max_fit = 2500 if not DEBUG else min(200, len(train_eids))
    fit_idx = rs.choice(
        len(train_eids), size=min(max_fit, len(train_eids)), replace=False
    )
    fit_eids = train_eids[fit_idx]

    X_fit = []
    for eid in fit_eids:
        pth = os.path.join(raw_50s_directory_path, f"{int(eid)}.npy")
        try:
            X_fit.append(_raw50_embed_from_npy(pth, sfreq=200))
        except Exception:
            continue
    X_fit = np.asarray(X_fit, dtype=np.float32)

    if X_fit.shape[0] >= 50:
        mu = X_fit.mean(axis=0, keepdims=True)
        sd = X_fit.std(axis=0, keepdims=True) + 1e-6
        X_fit_z = (X_fit - mu) / sd
        K = 32
        C, _ = _kmeans_fit(X_fit_z, k=K, seed=Config.seed, n_iter=30)
    else:
        mu = np.zeros((1, 6), dtype=np.float32)
        sd = np.ones((1, 6), dtype=np.float32)
        C = np.zeros((1, 6), dtype=np.float32)

    def eeg_cluster_for_id(eid_int):
        pth = os.path.join(raw_50s_directory_path, f"{int(eid_int)}.npy")
        try:
            x = _raw50_embed_from_npy(pth, sfreq=200).reshape(1, -1)
            xz = (x - mu) / sd
            return int(_kmeans_predict(xz.astype(np.float32), C)[0])
        except Exception:
            return 0

    eeg_votes["eeg_cluster"] = np.asarray(
        [eeg_cluster_for_id(eid) for eid in train_eids], dtype=np.int16
    )

    global_counts = eeg_votes[CLASSES].sum(axis=0).values.astype(np.float64)
    global_counts = np.clip(global_counts, 1.0, None)
    global_mean = global_counts / global_counts.sum()
    s0 = 120.0
    alpha0 = s0 * global_mean
    global_posterior = alpha0 / alpha0.sum()

    pc_counts = eeg_votes.groupby(["patient_id", "eeg_cluster"], sort=False)[
        CLASSES
    ].sum()
    pc_total = pc_counts.sum(axis=1).astype(np.float64)
    pc_counts_dict = {
        (int(pid), int(cl)): pc_counts.loc[(pid, cl)].values.astype(np.float64)
        for (pid, cl) in pc_counts.index
    }
    pc_total_dict = {
        (int(pid), int(cl)): float(pc_total.loc[(pid, cl)])
        for (pid, cl) in pc_total.index
    }

    p_counts = eeg_votes.groupby("patient_id", sort=False)[CLASSES].sum()
    p_total = p_counts.sum(axis=1).astype(np.float64)
    p_counts_dict = {
        int(pid): p_counts.loc[pid].values.astype(np.float64) for pid in p_counts.index
    }
    p_total_dict = {int(pid): float(p_total.loc[pid]) for pid in p_total.index}

    ps_counts = eeg_votes.groupby(["patient_id", "spectrogram_id"], sort=False)[
        CLASSES
    ].sum()
    ps_total = ps_counts.sum(axis=1).astype(np.float64)
    ps_counts_dict = {
        (int(pid), int(sid)): ps_counts.loc[(pid, sid)].values.astype(np.float64)
        for (pid, sid) in ps_counts.index
    }
    ps_total_dict = {
        (int(pid), int(sid)): float(ps_total.loc[(pid, sid)])
        for (pid, sid) in ps_total.index
    }

    s_counts = eeg_votes.groupby("spectrogram_id", sort=False)[CLASSES].sum()
    s_total = s_counts.sum(axis=1).astype(np.float64)
    s_counts_dict = {
        int(sid): s_counts.loc[sid].values.astype(np.float64) for sid in s_counts.index
    }
    s_total_dict = {int(sid): float(s_total.loc[sid]) for sid in s_total.index}

    mix_global = 0.04
    mix_backoff_pc_to_p = 0.20
    mix_backoff_ps_to_p = 0.25
    mix_ps_vs_pc = 0.50  # combine two conditionals; equal weighting is stable

    test_eids = test["eeg_id"].astype(np.int64).values
    test_pids = test["patient_id"].astype(np.int64).values
    test_sids = test["spectrogram_id"].astype(np.int64).values
    test_clusters = np.asarray(
        [eeg_cluster_for_id(eid) for eid in test_eids], dtype=np.int16
    )

    pred = np.zeros((len(test), N_CLASSES), dtype=np.float64)

    for i, (pid, sid, cl) in enumerate(zip(test_pids, test_sids, test_clusters)):
        pid = int(pid)
        sid = int(sid)
        cl = int(cl)

        p_est_pc = None
        counts_pc = pc_counts_dict.get((pid, cl))
        n_pc = pc_total_dict.get((pid, cl), 0.0)
        counts_p = p_counts_dict.get(pid)
        n_p = p_total_dict.get(pid, 0.0)

        if counts_pc is not None and n_pc > 0:
            p_pc = alpha0 + counts_pc
            p_pc = p_pc / p_pc.sum()
            if counts_p is not None and n_p > 0:
                p_p = alpha0 + counts_p
                p_p = p_p / p_p.sum()
                p_est_pc = (
                    1.0 - mix_backoff_pc_to_p
                ) * p_pc + mix_backoff_pc_to_p * p_p
            else:
                p_est_pc = p_pc
        elif counts_p is not None and n_p > 0:
            p_p = alpha0 + counts_p
            p_est_pc = p_p / p_p.sum()

        p_est_ps = None
        counts_ps = ps_counts_dict.get((pid, sid))
        n_ps = ps_total_dict.get((pid, sid), 0.0)
        counts_s = s_counts_dict.get(sid)
        n_s = s_total_dict.get(sid, 0.0)

        if counts_ps is not None and n_ps > 0:
            p_ps = alpha0 + counts_ps
            p_ps = p_ps / p_ps.sum()
            if counts_p is not None and n_p > 0:
                p_p = alpha0 + counts_p
                p_p = p_p / p_p.sum()
                p_est_ps = (
                    1.0 - mix_backoff_ps_to_p
                ) * p_ps + mix_backoff_ps_to_p * p_p
            else:
                p_est_ps = p_ps
        elif counts_p is not None and n_p > 0:
            p_p = alpha0 + counts_p
            p_est_ps = p_p / p_p.sum()
        elif counts_s is not None and n_s > 0:
            p_s = alpha0 + counts_s
            p_est_ps = p_s / p_s.sum()

        if p_est_pc is None and p_est_ps is None:
            p_est = global_posterior.copy()
        elif p_est_pc is None:
            p_est = p_est_ps
        elif p_est_ps is None:
            p_est = p_est_pc
        else:
            p_est = (1.0 - mix_ps_vs_pc) * p_est_pc + mix_ps_vs_pc * p_est_ps

        p_est = (1.0 - mix_global) * p_est + mix_global * global_posterior
        pred[i] = p_est

    pred = np.clip(pred, 1e-12, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    pred_df = pd.DataFrame(pred, columns=CLASSES)
    pred_df.insert(0, "eeg_id", test_eids)
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
print(
    "Row sums (min/max):",
    sub[CLASSES].sum(axis=1).min(),
    sub[CLASSES].sum(axis=1).max(),
)




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
