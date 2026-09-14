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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

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

0.3391724675284024

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import sys
from pathlib import Path

sys.path.append("/kaggle/input/hms-mk-codes/")



## === cell 1
get_ipython().run_line_magic(
    "pip",
    "install /kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl --no-index --no-deps --force-reinstall",
)
get_ipython().run_line_magic(
    "pip",
    "install /kaggle/input/requirements-mk/omegaconf-2.3.0-py3-none-any.whl --no-index --no-deps",
)
get_ipython().run_line_magic(
    "pip",
    "install /kaggle/input/requirements-mk/hydra_core-1.3.2-py3-none-any.whl --no-index --no-deps",
)
get_ipython().run_line_magic(
    "pip",
    "install /kaggle/input/requirements-mk/lightning-2.2.1-py3-none-any.whl --no-deps --no-index",
)



## === cell 2
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

Path(OUT_PATH).mkdir(parents=True, exist_ok=True)



## === cell 3
get_ipython().system(
    f'cd /kaggle/input/hms-mk-codes && python -m src.convert_parquet_to_npy --data_dir="{DATA_PATH}" --out_dir="{OUT_PATH}"'
)



## === cell 4
get_ipython().system("ls -la /kaggle/input | head -n 200")



## === cell 5
checkpoints = [
    "epoch_012_val_loss_0.5037.ckpt",
    "epoch_013_val_loss_0.5103.ckpt",
    "epoch_014_val_loss_0.4728.ckpt",
    "epoch_014_val_loss_0.4944.ckpt",
    "epoch_014_val_loss_0.5053.ckpt",
]



## === cell 7
get_ipython().system(
    f'cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir="{DATA_PATH}" data.test_eegs_dir="{OUT_PATH}" ckpt_path=/kaggle/input/hms-mk-data/fold0_pseudo_log.ckpt hydra=test +model.test_output_dir="{OUT_PATH}" experiment=conv1d_pseudo +model.net.pretrained=False'
)
get_ipython().system(
    "mv /kaggle/working/submission.csv /kaggle/working/submission_fold0.csv"
)

get_ipython().system(
    f'cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir="{DATA_PATH}" data.test_eegs_dir="{OUT_PATH}" ckpt_path=/kaggle/input/hms-mk-data/fold1_pseudo_log.ckpt hydra=test +model.test_output_dir="{OUT_PATH}" experiment=conv1d_pseudo +model.net.pretrained=False'
)
get_ipython().system(
    "mv /kaggle/working/submission.csv /kaggle/working/submission_fold1.csv"
)

get_ipython().system(
    f'cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir="{DATA_PATH}" data.test_eegs_dir="{OUT_PATH}" ckpt_path=/kaggle/input/hms-mk-data/fold2_pseudo_log.ckpt hydra=test +model.test_output_dir="{OUT_PATH}" experiment=conv1d_pseudo +model.net.pretrained=False'
)
get_ipython().system(
    "mv /kaggle/working/submission.csv /kaggle/working/submission_fold2.csv"
)

get_ipython().system(
    f'cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir="{DATA_PATH}" data.test_eegs_dir="{OUT_PATH}" ckpt_path=/kaggle/input/hms-mk-data/fold3_pseudo_log.ckpt hydra=test +model.test_output_dir="{OUT_PATH}" experiment=conv1d_pseudo +model.net.pretrained=False'
)
get_ipython().system(
    "mv /kaggle/working/submission.csv /kaggle/working/submission_fold3.csv"
)

get_ipython().system(
    f'cd /kaggle/input/hms-mk-codes && python -m test paths.data_dir="{DATA_PATH}" data.test_eegs_dir="{OUT_PATH}" ckpt_path=/kaggle/input/hms-mk-data/fold4_pseudo_log.ckpt hydra=test +model.test_output_dir="{OUT_PATH}" experiment=conv1d_pseudo +model.net.pretrained=False'
)
get_ipython().system(
    "mv /kaggle/working/submission.csv /kaggle/working/submission_fold4.csv"
)



## === cell 8
import pandas as pd
import numpy as np

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


def merge_preds(
    folds=(0, 1, 2, 3, 4),
    sample_sub_path="/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv",
):
    sample = pd.read_csv(sample_sub_path)
    if "eeg_id" not in sample.columns:
        raise ValueError("sample_submission.csv is missing 'eeg_id' column.")
    for c in TARGET_COLS:
        if c not in sample.columns:
            raise ValueError(
                f"sample_submission.csv is missing required target column: {c}"
            )

    preds_accum = None
    used_folds = 0

    for fold in folds:
        fold_path = f"/kaggle/working/submission_fold{fold}.csv"
        if not os.path.exists(fold_path):
            raise FileNotFoundError(f"Missing fold prediction file: {fold_path}")

        sol = pd.read_csv(fold_path)
        if "eeg_id" not in sol.columns:
            raise ValueError(f"{fold_path} is missing 'eeg_id' column.")

        sol = sol.merge(sample[["eeg_id"]], on="eeg_id", how="right", sort=False)
        for c in TARGET_COLS:
            if c not in sol.columns:
                raise ValueError(f"{fold_path} is missing required target column: {c}")

        fold_preds = sol[TARGET_COLS].to_numpy(dtype=np.float64)

        if preds_accum is None:
            preds_accum = fold_preds
        else:
            preds_accum += fold_preds
        used_folds += 1

    if used_folds == 0:
        raise RuntimeError("No fold predictions were loaded; cannot create ensemble.")

    preds = preds_accum / used_folds

    row_sums = preds.sum(axis=1, keepdims=True)
    zero_mask = row_sums[:, 0] == 0
    if np.any(zero_mask):
        preds[zero_mask] = 1.0 / len(TARGET_COLS)
        row_sums = preds.sum(axis=1, keepdims=True)

    preds = preds / row_sums

    out = sample.copy()
    out[TARGET_COLS] = preds
    return out




## === cell 9
sol = merge_preds(folds=(0, 1, 2, 3, 4))



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2576195677.py in <cell line: 0>()
----> 1 sol = merge_preds(folds=(0, 1, 2, 3, 4))
      2 

/tmp/ipykernel_11/1804031837.py in merge_preds(folds, sample_sub_path)
     34         fold_path = f"/kaggle/working/submission_fold{fold}.csv"
     35         if not os.path.exists(fold_path):
---> 36             raise FileNotFoundError(f"Missing fold prediction file: {fold_path}")
     37 
     38         sol = pd.read_csv(fold_path)

FileNotFoundError: Missing fold prediction file: /kaggle/working/submission_fold0.csv

## === cell 10
sub_path = "/kaggle/working/submission.csv"
sol.to_csv(sub_path, index=False)
print(f"Wrote submission: {sub_path}  shape={sol.shape}")
print(sol.head())



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1202269224.py in <cell line: 0>()
      1 # Write final submission with required filename/suffix.
      2 sub_path = "/kaggle/working/submission.csv"
----> 3 sol.to_csv(sub_path, index=False)
      4 print(f"Wrote submission: {sub_path}  shape={sol.shape}")
      5 print(sol.head())

NameError: name 'sol' is not defined

## === cell 11
sol

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1793929092.py in <cell line: 0>()
----> 1 sol

NameError: name 'sol' is not defined
