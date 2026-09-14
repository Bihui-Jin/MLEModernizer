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

albumentations==2.0.8
geopandas==0.14.4
librosa==0.11.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
PyWavelets==1.8.0
sklearn-pandas==2.2.0
timm==1.0.19
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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

0.6909634962658863

# 6. Current score

0.79914

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I replace the failing model‑loading inference with a simple baseline that uses the class distribution from the training data as predictions for every test sample. This removes the missing‑file error, guarantees the output probabilities sum to 1, and creates a valid `submission.csv` file.'
- What this solution (achieved 1.68479) has done: 'I replace the simple global‑frequency baseline with a patient‑aware baseline: compute class vote distributions per patient from the training data and use them for any test rows that share a patient_id; fall back to the overall class prior when a patient is unseen. This keeps the core logic unchanged, guarantees probabilities sum to 1, and should lower the KL score toward the target.'
- What this solution (achieved 1.68479) has done: 'I keep the existing data‑loading and dataset code, but replace the simple patient‑prior baseline with an actual model inference step. The script now loads the EfficientNet weights, runs the test loader through the model, and writes the softmax probabilities to `submission.csv`. If the weight file is missing or loading fails, it gracefully falls back to the previous patient‑wise prior so the code always produces a valid submission.'
- What this solution (achieved 1.68479) has done: 'I keep the overall pipeline unchanged but enhance the fallback baseline: in addition to the patient‑wise prior it also try an eeg_id‑wise prior (which is often more specific) before falling back to the global class prior. This small refinement should produce predictions that are closer to the true label distribution and therefore reduce the KL divergence toward the target while preserving the original logic and file output.'
- What this solution (achieved 1.68479) has done: 'I added all missing imports, defined the device, corrected the use of undefined variables (glob, np, pd, nn, Dataset, DataLoader, label_cols), reordered the loading of spectrogram and EEG data before the dataset class, and fixed the dataset constructor to receive the pre‑loaded dictionaries. These fixes let the script run end‑to‑end, produce a properly normalised probability matrix, and write a valid `submission.csv` file. The fallback patient/eeg‑wise prior remains unchanged, providing a reasonable score while keeping the core logic intact.'
- What this solution (achieved 1.68479) has done: 'The fix corrects the shape mismatch in the dataset by extracting a 100 × 256 region from each spectrogram (center‑cropping and padding when needed) before assigning it to the input tensor. This eliminates the broadcasting error, allows the DataLoader to produce valid samples, and lets the rest of the pipeline run to generate a properly normalised submission CSV. No other logic is altered, preserving the original model‑fallback behavior and keeping the score‑related code unchanged.'
- What this solution (achieved 1.68479) has done: 'Implemented a streamlined pipeline that restores missing imports, defines required paths and configuration, and focuses on the reliable fallback‑prior prediction method. The script now loads the train and test metadata, computes global, patient‑wise, and eeg‑wise vote distributions, builds probability predictions that are guaranteed to sum to 1, and writes a valid `submission.csv` file.'
- What this solution (achieved 0.76617) has done: 'I smooth the class counts by adding a small constant before normalising, and blend each prior‑based prediction ≈ 90 % with the global class prior ≈ 10 % to avoid overly confident zeros. These tiny adjustments keep the original patient/eeg‑wise logic while making the probabilities more calibrated, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.78418) has done: 'I slightly adjust the smoothing constant and the blend weight used for combining the patient/eeg‑specific priors with the global prior.  
Using a smaller Laplace smoothing (0.5 instead of 1) reduces over‑confidence in the priors, and decreasing the specific‑prior blend from 0.9 to 0.8 makes the predictions a bit more conservative, which should lower the KL divergence and move the score closer to the target while keeping the overall logic unchanged.'
- What this solution (achieved 0.86468) has done: 'I slightly increase the Laplace smoothing (to 1.0) and reduce the reliance on patient/eeg‑specific priors (set SPECIFIC_WEIGHT to 0.6, giving GLOBAL_WEIGHT 0.4). These minimal tweaks make the predictions more conservative and better calibrated, which should lower the KL‑divergence toward the target while keeping the original logic unchanged.'
- What this solution (achieved 0.79914) has done: 'I lower the Laplace smoothing to 0.5 (less aggressive smoothing) and make the blend rely more on the global prior by setting `SPECIFIC_WEIGHT` = 0.75 and `GLOBAL_WEIGHT` = 0.25. After the weighted combination I also add a tiny epsilon to each probability before renormalising so that no entry is exactly zero, which yields a slightly better‑calibrated distribution and should decrease the KL‑divergence toward the target score. The rest of the pipeline and logic remain unchanged.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import librosa
import pywt
import matplotlib.pyplot as plt
import timm

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
LABEL_COLS = TARGETS  # same names for label columns


class Paths:
    BASE = "/kaggle/input/hms-harmful-brain-activity-classification"
    TRAIN_CSV = os.path.join(BASE, "train.csv")
    TEST_CSV = os.path.join(BASE, "test.csv")
    OUTPUT_DIR = "/kaggle/working"  # write submission here


paths = Paths()


class Config:
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    MODEL = "efficientnet_b0"  # placeholder, not used in fallback


config = Config()

device = torch.device("cpu")



## === cell 1
train_df = pd.read_csv(paths.TRAIN_CSV)
test_df = pd.read_csv(paths.TEST_CSV)
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")



## === cell 2
SMOOTH = 0.5

class_counts = train_df[TARGETS].sum() + SMOOTH
global_prior = class_counts / class_counts.sum()

patient_votes = train_df.groupby("patient_id")[TARGETS].sum() + SMOOTH
patient_prior = patient_votes.div(patient_votes.sum(axis=1), axis=0).fillna(0)
patient_totals = patient_votes.sum(axis=1)

eeg_votes = train_df.groupby("eeg_id")[TARGETS].sum() + SMOOTH
eeg_prior = eeg_votes.div(eeg_votes.sum(axis=1), axis=0).fillna(0)
eeg_totals = eeg_votes.sum(axis=1)



## === cell 3
test_patient_ids = test_df["patient_id"].values
test_eeg_ids = test_df["eeg_id"].values

predictions = np.zeros((len(test_df), len(TARGETS)), dtype=np.float32)

SPECIFIC_WEIGHT = 0.75
GLOBAL_WEIGHT = 1.0 - SPECIFIC_WEIGHT

for i, (pid, eid) in enumerate(zip(test_patient_ids, test_eeg_ids)):
    has_patient = pid in patient_prior.index
    has_eeg = eid in eeg_prior.index

    if has_patient and has_eeg:
        pw = patient_totals.loc[pid]
        ew = eeg_totals.loc[eid]
        combined = patient_prior.loc[pid].values * pw + eeg_prior.loc[eid].values * ew
        pred = combined / (pw + ew)
    elif has_patient:
        pred = patient_prior.loc[pid].values
    elif has_eeg:
        pred = eeg_prior.loc[eid].values
    else:
        pred = global_prior.values

    pred = SPECIFIC_WEIGHT * pred + GLOBAL_WEIGHT * global_prior.values
    predictions[i] = pred

epsilon = 1e-6
predictions = predictions + epsilon
row_sums = predictions.sum(axis=1, keepdims=True)
predictions = predictions / np.where(row_sums == 0, 1, row_sums)



## === cell 4
sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
sub[TARGETS] = predictions
sub_path = os.path.join(paths.OUTPUT_DIR, "submission.csv")
sub.to_csv(sub_path, index=False)
print(f"Submission saved to {sub_path}")
print(f"Submission shape: {sub.shape}")
print(sub.head())
