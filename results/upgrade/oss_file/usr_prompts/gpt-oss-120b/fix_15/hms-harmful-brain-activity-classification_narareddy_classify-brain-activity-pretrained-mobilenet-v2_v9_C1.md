# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

geopandas==0.14.4
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
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
transformers==4.53.3

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

1.132911383265961

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I added the missing imports, replaced the unavailable pretrained MobileNet with a lightweight dummy neural network that matches the data shape, and rewrote the test‑prediction generation so it creates a row for every entry in `test.csv` (ensuring the submission length matches the ground‑truth file). These changes fix the runtime errors and guarantee a valid `.csv` submission while keeping the overall pipeline intact.'
- What this solution (achieved 1.39403) has done: 'Implemented two key fixes:  
1. Adjusted `DummyModel` input dimension to match the flattened image tensor size (3 × 224 × 224).  
2. Replaced uniform test predictions with the overall class‑probability distribution computed from the training data, yielding more informed probabilities and a lower KL‑divergence score.'

# 9. Code solution

## === cell 0
import pandas as pd

pd.set_option("display.max_columns", None)
import numpy as np
import os
import ast
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

data_path = "/kaggle/input/hms-harmful-brain-activity-classification"
sg_path = f"{data_path}/train_spectrograms"
sg_test_path = f"{data_path}/test_spectrograms"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 1
train_df_M = pd.read_csv(f"{data_path}/train.csv")
print(train_df_M.shape)
train_df_M.head()



## === cell 2
print(
    "Train Data:\n",
    train_df_M["expert_consensus"]
    .value_counts()
    .rename("Count")
    .to_frame()
    .assign(Percentage=lambda x: round((x / x.sum()) * 100)),
)



## === cell 3
train_df = train_df_M.sample(5000, random_state=52).reset_index(drop=True)
print(
    "Train Data:\n",
    train_df["expert_consensus"]
    .value_counts()
    .rename("Count")
    .to_frame()
    .assign(Percentage=lambda x: round((x / x.sum()) * 100)),
)



## === cell 4
train_checked_count = 0
null_count = 0
for index, row in train_df.iterrows():
    parquet_path = f"{sg_path}/{row['spectrogram_id']}.parquet"
    parquet_df = pd.read_parquet(parquet_path)
    filtered_df = parquet_df[
        (parquet_df["time"] >= row["spectrogram_label_offset_seconds"])
        & (parquet_df["time"] < row["spectrogram_label_offset_seconds"] + 600)
    ]
    if filtered_df.isnull().any().any():
        train_df.at[index, "SG_NullData_Ind"] = True
    else:
        train_df.at[index, "SG_NullData_Ind"] = False

print(train_df["SG_NullData_Ind"].value_counts())
print("\ncheck expert consensus where sg is not null...")
print(
    train_df[train_df["SG_NullData_Ind"] == False]["expert_consensus"]
    .value_counts()
    .rename("Count")
    .to_frame()
    .assign(Percentage=lambda x: round((x / x.sum()) * 100))
)



## === cell 5
train_df = train_df[train_df["SG_NullData_Ind"] != True]
train_df.shape



## === cell 6
vote_columns = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
prob_columns = [
    "seizure_prob",
    "lpd_prob",
    "gpd_prob",
    "lrda_prob",
    "grda_prob",
    "other_prob",
]

train_df["total_votes"] = train_df[vote_columns].sum(axis=1)

for vote_col, prob_col in zip(vote_columns, prob_columns):
    train_df[prob_col] = train_df[vote_col] / train_df["total_votes"]
train_df.head()



## === cell 7
test_df = pd.read_csv(f"{data_path}/test.csv")
print("test shape:", test_df.shape)

epsilon = 1e-3

holdout = train_df.sample(frac=0.2, random_state=123).reset_index(drop=True)
train_split = train_df.drop(holdout.index).reset_index(drop=True)

patient_votes_split = (
    train_split.groupby("patient_id")[vote_columns].sum().reset_index()
)
patient_votes_split_smoothed = patient_votes_split.copy()
patient_votes_split_smoothed[vote_columns] = (
    patient_votes_split_smoothed[vote_columns] + epsilon
)
row_sums = patient_votes_split_smoothed[vote_columns].sum(axis=1)
patient_votes_split_smoothed[vote_columns] = patient_votes_split_smoothed[
    vote_columns
].div(row_sums, axis=0)
patient_probs_split = patient_votes_split_smoothed.rename(
    columns={col: f"{col}_mean" for col in vote_columns}
)

global_votes = train_df[vote_columns].sum()
global_smoothed = (global_votes + epsilon) / (
    global_votes.sum() + epsilon * len(vote_columns)
)
global_smoothed_dict = {f"{col}_mean": global_smoothed[col] for col in vote_columns}


def kl_divergence(p, q):
    """Mean KL(p||q) for two 2‑D arrays (rows = samples)."""
    eps = 1e-12
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.mean(np.sum(p * np.log(p / q), axis=1))


candidate_weights = np.arange(0.0, 1.01, 0.05)
best_weight = 0.5  # fallback
best_kl = np.inf

for pw in candidate_weights:
    gw = 1.0 - pw
    blended = holdout.copy()
    blended = blended.merge(patient_probs_split, on="patient_id", how="left")
    for col in vote_columns:
        patient_col = f"{col}_mean"
        blended[patient_col] = (
            pw * blended[patient_col] + gw * global_smoothed_dict[patient_col]
        )

    blended = blended.rename(
        columns={f"{col}_mean": f"{col}_vote" for col in vote_columns}
    )
    prob_cols = [f"{col}_vote" for col in vote_columns]
    blended[prob_cols] = blended[prob_cols].clip(lower=0)
    blended[prob_cols] = blended[prob_cols].div(blended[prob_cols].sum(axis=1), axis=0)

    true_probs = holdout[vote_columns].values / holdout["total_votes"].values[:, None]
    pred_probs = blended[prob_cols].values
    kl = kl_divergence(true_probs, pred_probs)

    if kl < best_kl:
        best_kl = kl
        best_weight = pw

print(
    f"Chosen patient weight = {best_weight:.2f} (global weight = {1-best_weight:.2f}), "
    f"KL on hold‑out = {best_kl:.5f}"
)

patient_weight = best_weight
global_weight = 1.0 - best_weight

patient_votes_full = train_df.groupby("patient_id")[vote_columns].sum().reset_index()
patient_votes_full_smoothed = patient_votes_full.copy()
patient_votes_full_smoothed[vote_columns] = (
    patient_votes_full_smoothed[vote_columns] + epsilon
)
row_sums_full = patient_votes_full_smoothed[vote_columns].sum(axis=1)
patient_votes_full_smoothed[vote_columns] = patient_votes_full_smoothed[
    vote_columns
].div(row_sums_full, axis=0)
patient_probs_full = patient_votes_full_smoothed.rename(
    columns={col: f"{col}_mean" for col in vote_columns}
)

test_preds = test_df[["eeg_id", "patient_id"]].copy()
test_preds = test_preds.merge(patient_probs_full, on="patient_id", how="left")

for col in [f"{c}_mean" for c in vote_columns]:
    test_preds[col].fillna(global_smoothed_dict[col], inplace=True)

for col in vote_columns:
    patient_col = f"{col}_mean"
    test_preds[patient_col] = (
        patient_weight * test_preds[patient_col]
        + global_weight * global_smoothed_dict[patient_col]
    )

test_preds = test_preds.rename(
    columns={f"{col}_mean": f"{col}_vote" for col in vote_columns}
)[["eeg_id"] + [f"{col}_vote" for col in vote_columns]]

prob_cols = [f"{col}_vote" for col in vote_columns]
test_preds[prob_cols] = test_preds[prob_cols].clip(lower=0)
test_preds[prob_cols] = test_preds[prob_cols].div(
    test_preds[prob_cols].sum(axis=1), axis=0
)

assert len(test_preds) == len(
    test_df
), "Length mismatch between predictions and test set"
assert set(prob_cols).issubset(
    set(test_preds.columns)
), "Missing vote columns after processing"
print("Preview of submission:")
print(test_preds.head())



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/380668565.py in <cell line: 0>()
      6 # ---- split into hold‑out for weight tuning ----
      7 holdout = train_df.sample(frac=0.2, random_state=123).reset_index(drop=True)
----> 8 train_split = train_df.drop(holdout.index).reset_index(drop=True)
      9 
     10 # ---- patient probabilities from the non‑holdout part (more honest estimate) ----

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   5579                 weight  1.0     0.8
   5580         """
-> 5581         return super().drop(
   5582             labels=labels,
   5583             axis=axis,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   4786         for axis, labels in axes.items():
   4787             if labels is not None:
-> 4788                 obj = obj._drop_axis(labels, axis, level=level, errors=errors)
   4789 
   4790         if inplace:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _drop_axis(self, labels, axis, level, errors, only_slice)
   4828                 new_axis = axis.drop(labels, level=level, errors=errors)
   4829             else:
-> 4830                 new_axis = axis.drop(labels, errors=errors)
   4831             indexer = axis.get_indexer(new_axis)
   4832 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in drop(self, labels, errors)
   7068         if mask.any():
   7069             if errors != "ignore":
-> 7070                 raise KeyError(f"{labels[mask].tolist()} not found in axis")
   7071             indexer = indexer[~mask]
   7072         return self.delete(indexer)

KeyError: '[8, 52, 55, 80, 87, 90, 93, 101, 118, 119, 127, 161, 162, 168, 178, 179, 181, 188, 198, 217, 223, 225, 235, 268, 297, 314, 323, 333, 336, 367, 377, 407, 414, 440, 446, 457, 470, 482, 489, 502, 514, 517, 520, 528, 533, 542, 547, 548, 549, 569, 586, 603, 614, 625, 685, 689, 709, 715, 724, 732, 734, 735, 738, 750, 764, 767, 770, 772, 791, 794, 834, 842, 862, 864, 866, 889, 896, 897, 902, 904, 905, 912] not found in axis'

## === cell 8
submission_path = "submission.csv"
test_preds.to_csv(submission_path, index=False)
print(f"submission file generated at {submission_path}")

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3711983024.py in <cell line: 0>()
      1 submission_path = "submission.csv"
----> 2 test_preds.to_csv(submission_path, index=False)
      3 print(f"submission file generated at {submission_path}")

NameError: name 'test_preds' is not defined
