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

0.3390572951476089

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I remove the failing imports and shell calls, define the target column names directly, compute simple baseline class probabilities from the training votes, and create a valid submission CSV where each test row gets the same normalized probabilities. This fixes the `ModuleNotFoundError`, ensures the script runs without external dependencies, and guarantees a correctly‑formatted submission file.'
- What this solution (achieved 1.41937) has done: 'I replace the simple global‑probability baseline with a per‑eeg_id empirical distribution: for each eeg_id present in the training set I compute its vote counts, normalize them to probabilities, and use those directly for matching test rows. If a test eeg_id does not appear in training, I fall back to the overall baseline probabilities. This adds only lightweight aggregation and merging, preserving the original workflow while moving the KL‑divergence score closer to the target.'
- What this solution (achieved 1.68479) has done: 'I keep the existing per‑eeg‑id probability baseline but add a second fallback: use the aggregated vote distribution per `patient_id`. For test rows whose `eeg_id` was never seen in training, the script now looks up the patient‑level probabilities; only if that is also missing does it fall back to the global baseline. This extra, still‑lightweight information should lower the KL‑divergence toward the target while preserving the original workflow.'
- What this solution (achieved 1.68479) has done: 'I add a lightweight spectrogram‑level probability fallback to the existing hierarchy (eeg → patient → spectrogram → global). This provides more specific priors for test rows that lack an eeg_id or patient match, while keeping the original simple averaging logic and preserving the row‑sum = 1 constraint. The change is limited to the data‑processing cell and does not alter model architecture or training steps.'
- What this solution (achieved 1.05318) has done: 'I simplify the hierarchy by dropping the spectrogram‑level fallback (which can introduce noisy zero probabilities) and add a tiny smoothing epsilon before normalising each row. This keeps the original per‑eeg and per‑patient priors, avoids zero predictions, and should lower the KL‑divergence toward the target while preserving the overall workflow.'
- What this solution (achieved 1.05318) has done: 'I add a spectrogram‑level fallback to the existing hierarchy (eeg → patient → spectrogram → global) and keep the small epsilon smoothing before normalising. This gives a more specific prior for test rows that lack both an eeg_id and patient match, which should lower the KL‑divergence and move the score closer to the target while preserving the original workflow.'

# 9. Code solution

## === cell 0
import sys

sys.path.append("/kaggle/input/hms-mk-codes/")



## === cell 1
import subprocess, shlex


def safe_install(cmd):
    try:
        subprocess.run(shlex.split(cmd), check=True, capture_output=True)
    except Exception:
        pass  # ignore failures – they are not required for the baseline


safe_install(
    "pip install /kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl --no-index --no-deps --force-reinstall"
)
safe_install(
    "pip install /kaggle/input/requirements-mk/omegaconf-2.3.0-py3-none-any.whl --no-index --no-deps"
)
safe_install(
    "pip install /kaggle/input/requirements-mk/hydra_core-1.3.2-py3-none-any.whl --no-index --no-deps"
)
safe_install(
    "pip install /kaggle/input/requirements-mk/lightning-2.2.1-py3-none-any.whl --no-deps --no-index"
)



## === cell 2
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"



## === cell 3
pass



## === cell 4
pass



## === cell 5
checkpoints = [
    "epoch_012_val_loss_0.5037.ckpt",
    "epoch_013_val_loss_0.5103.ckpt",
    "epoch_014_val_loss_0.4728.ckpt",
    "epoch_014_val_loss_0.4944.ckpt",
    "epoch_014_val_loss_0.5053.ckpt",
]



## === cell 6
TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



## === cell 7
import pandas as pd
import numpy as np
import os

train_path = os.path.join(DATA_PATH, "train.csv")
train_df = pd.read_csv(train_path)

test_path = os.path.join(DATA_PATH, "test.csv")
test_df = pd.read_csv(test_path)

global_sums = train_df[TARGET_COLS].sum()
base_probs = (global_sums / global_sums.sum()).values.astype(float)  # shape (6,)

eeg_group = train_df.groupby("eeg_id")[TARGET_COLS].sum()
eeg_group_sum = eeg_group.sum(axis=1).replace(0, np.nan)  # total votes per eeg_id
eeg_probs = eeg_group.div(eeg_group_sum, axis=0)  # rows sum to 1
eeg_weights = eeg_group_sum  # weight = total votes

patient_group = train_df.groupby("patient_id")[TARGET_COLS].sum()
patient_group_sum = patient_group.sum(axis=1).replace(0, np.nan)
patient_probs = patient_group.div(patient_group_sum, axis=0)
patient_weights = patient_group_sum

spectro_group = train_df.groupby("spectrogram_id")[TARGET_COLS].sum()
spectro_group_sum = spectro_group.sum(axis=1).replace(0, np.nan)
spectro_probs = spectro_group.div(spectro_group_sum, axis=0)
spectro_weights = spectro_group_sum

eeg_merge = test_df[["eeg_id"]].merge(
    eeg_probs, how="left", left_on="eeg_id", right_index=True
)
eeg_w_merge = test_df[["eeg_id"]].merge(
    eeg_weights.rename("weight"), how="left", left_on="eeg_id", right_index=True
)

patient_merge = test_df[["patient_id"]].merge(
    patient_probs, how="left", left_on="patient_id", right_index=True
)
patient_w_merge = test_df[["patient_id"]].merge(
    patient_weights.rename("weight"), how="left", left_on="patient_id", right_index=True
)

spectro_merge = test_df[["spectrogram_id"]].merge(
    spectro_probs, how="left", left_on="spectrogram_id", right_index=True
)
spectro_w_merge = test_df[["spectrogram_id"]].merge(
    spectro_weights.rename("weight"),
    how="left",
    left_on="spectrogram_id",
    right_index=True,
)

blended = pd.DataFrame(0.0, index=test_df.index, columns=TARGET_COLS)
total_weight = pd.Series(0.0, index=test_df.index)


def add_component(probs_df, weights_series):
    mask = probs_df.notna().any(axis=1)
    if mask.any():
        w = weights_series[mask].values[:, None]  # (n,1)
        p = probs_df.loc[mask, TARGET_COLS].fillna(0).values  # (n,6)
        blended.loc[mask, TARGET_COLS] += p * w
        total_weight[mask] += weights_series[mask].values


add_component(eeg_merge, eeg_w_merge["weight"])
add_component(patient_merge, patient_w_merge["weight"])
add_component(spectro_merge, spectro_w_merge["weight"])

no_prior_mask = total_weight == 0
blended.loc[no_prior_mask, TARGET_COLS] = base_probs
total_weight[no_prior_mask] = 1.0  # avoid division by zero

blended = blended.div(total_weight.values[:, None], axis=0)

epsilon = 1e-6
blended[TARGET_COLS] = blended[TARGET_COLS] + epsilon
row_sums = blended[TARGET_COLS].sum(axis=1)
submission = blended[TARGET_COLS].div(row_sums, axis=0)
submission.insert(0, "eeg_id", test_df["eeg_id"])

output_file = os.path.join(OUT_PATH, "submission.csv")
submission.to_csv(output_file, index=False)

print(f"Submission written to {output_file}")
