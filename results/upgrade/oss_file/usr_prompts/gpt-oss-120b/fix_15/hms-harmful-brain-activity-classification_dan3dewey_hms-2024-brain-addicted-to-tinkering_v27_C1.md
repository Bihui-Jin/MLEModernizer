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
pyarrow==19.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

1.043043

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.20868) has done: 'I add a safe import for os, ensure the submission file is written to the current working directory with a clear name, and normalise the predicted probabilities so each row sums to 1 (which guarantees a valid KL‑score and a valid Kaggle submission). These small adjustments keep the original modelling pipeline untouched while fixing the missing/invalid CSV issue and nudging the score toward the target.'
- What this solution (achieved 1.20344) has done: 'We fine‑tune the “taming” step that blends cluster centroids with the overall mean probabilities.  
By searching the blend fraction with a finer step (0.02 instead of 0.05) both for the training‑derived and validation‑derived tuning loops, we can locate a slightly better mixture that lowers the KL‑divergence, moving the score toward the target 1.043043 while keeping the original modelling pipeline unchanged. The rest of the code—including data handling, feature extraction, model training and CSV creation—remains the same.'
- What this solution (achieved 1.41937) has done: 'I fix the file‑path issue that caused the script to crash when loading `train.csv` and `test.csv`. The code now looks for the dataset in the typical Kaggle input location (or the original working directory) and selects the first existing path, ensuring the metadata can be read and the subsequent cells run correctly, producing a valid `submission.csv` with properly normalised probabilities.'
- What this solution (achieved 1.41937) has done: 'I keep the original data handling and probability calculations but add a lightweight blending step that mixes the per‑EEG probabilities with the overall global class frequencies. By giving a modest weight to the global distribution (instead of relying entirely on possibly noisy per‑EEG values) we can reduce over‑confidence and lower the KL‑divergence, moving the score toward the target while preserving the existing pipeline. The change is limited to cell 4 and adds a single blend factor and the corresponding normalization.'
- What this solution (achieved 1.41937) has done: 'The script crashed because `train_test_split` was asked to stratify on `eeg_id`, which is almost unique for each row, causing some classes to have only one sample. Removing the stratify argument resolves the error and lets the pipeline run to produce a valid `submission.csv`. No other logic is altered, preserving the original modelling approach while ensuring a proper output file.'
- What this solution (achieved 1.41937) has done: 'I keep the original pipeline but add a fine‑grained search around the best blend factor found with the coarse 0.05 step. By exploring factors in ±0.05 of the current best (with 0.01 resolution) we can usually obtain a slightly lower validation KL, which should reduce the final competition score toward the target without altering the core modeling logic.'
- What this solution (achieved 1.41937) has done: 'I keep the original pipeline but add a confidence‑based weighting to the per‑EEG blending step: EEGs with few training votes get a smaller per‑EEG contribution and rely more on the global class distribution. This modest change is expected to reduce over‑confident noisy predictions and lower the KL‑divergence, moving the score closer to the target without altering the core model logic. The script now merges per‑EEG vote counts, computes a weight (capped at 1), and uses it to adjust the blend factor for each test row before normalising and writing the submission.'
- What this solution (achieved 1.41937) has done: 'I increase the vote‑threshold used for weighting the per‑EEG blend factor (from 30 to 60). This makes the per‑EEG contribution smaller for most recordings, relying more on the stable global class distribution, which should lower the KL‑divergence and move the score closer to the target while keeping the core pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import pyarrow.dataset as pads

possible_dirs = [
    os.path.join(os.getcwd(), "data", "hms-harmful-brain-activity-classification"),
    "/kaggle/input/hms-harmful-brain-activity-classification",
    "/kaggle/working/data/hms-harmful-brain-activity-classification",
]

above_dir = None
for d in possible_dirs:
    if os.path.isdir(d):
        above_dir = d
        break

if above_dir is None:
    raise FileNotFoundError(
        "Dataset directory not found. Checked: " + ", ".join(possible_dirs)
    )

HBA_names = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
HBA_votes = [f"{name}_vote" for name in HBA_names]

np.set_printoptions(precision=6, suppress=True)




## === cell 1
def kld_score(solution, submission):
    """
    Calculate the average KL divergence score.
    Ignores the "row id" assumed in the first column.
    """
    sumsum = 0.0
    for prob_col in solution.columns.values:
        sumsum += np.nansum(
            -1.0
            * solution[prob_col]
            * np.log(submission[prob_col] / solution[prob_col])
        )
    return sumsum / (len(solution))




## === cell 2
def read_hms_meta():
    """
    Read in the train.csv and test.csv files.
    Add total_vote, _prob columns, and vote entropy to train_meta.
    Add extra cols to test to allow the same processing as train:
        eeg[spectro]_sub_id, eeg[spectro]_label_offset_seconds, label_id
    """
    test_path = os.path.join(above_dir, "test.csv")
    train_path = os.path.join(above_dir, "train.csv")

    test_meta = pd.read_csv(test_path)
    test_meta_len = len(test_meta)
    print("Test has length", test_meta_len)
    test_meta["eeg_sub_id"] = 0
    test_meta["eeg_label_offset_seconds"] = 0.0
    test_meta["spectrogram_sub_id"] = 0
    test_meta["spectrogram_label_offset_seconds"] = 0.0
    test_meta["label_id"] = test_meta.eeg_id
    if test_meta_len > 1:
        REAL_TEST = True
    else:
        REAL_TEST = False
        print("  --> not the real LB test data.\n")
        pass

    train_meta = pd.read_csv(train_path)
    train_meta_len = len(train_meta)
    print("Train has length", train_meta_len, " with:")

    train_meta["total_vote"] = (
        train_meta["seizure_vote"]
        + train_meta["lpd_vote"]
        + train_meta["gpd_vote"]
        + train_meta["lrda_vote"]
        + train_meta["grda_vote"]
        + train_meta["other_vote"]
    )
    train_meta["max_vote"] = np.max(
        np.array(
            [
                train_meta["seizure_vote"],
                train_meta["lpd_vote"],
                train_meta["gpd_vote"],
                train_meta["lrda_vote"],
                train_meta["grda_vote"],
                train_meta["other_vote"],
            ]
        ),
        axis=0,
    )

    for this_col in [
        "label_id",
        "eeg_id",
        "spectrogram_id",
        "patient_id",
        "total_vote",
    ]:
        print(
            "   ", len(train_meta[this_col].unique()), "unique " + this_col + " values."
        )

    plt.figure(figsize=(6, 3))
    plt.hist(train_meta["total_vote"], bins=55, log=True)
    plt.title("Histogram of Total Votes")
    plt.show()

    for col_pre in HBA_names:
        train_meta[col_pre + "_prob"] = (
            train_meta[col_pre + "_vote"] / train_meta["total_vote"]
        )

    def calc_entropy(row):
        the_probs = np.clip(row[16 : 21 + 1].values.astype(float), 1.0e-8, 1.0)
        return np.nansum(the_probs * -1 * np.log(the_probs))

    train_meta["entropy"] = train_meta.apply(calc_entropy, axis=1)

    return train_meta, test_meta




## === cell 3
train_meta, test_meta = read_hms_meta()




## === cell 4
from sklearn.model_selection import train_test_split

train_split, val_split = train_test_split(train_meta, test_size=0.20, random_state=42)

total_votes_sum = train_split[HBA_votes].sum().sum()
global_probs = train_split[HBA_votes].sum() / total_votes_sum

per_eeg_counts = train_split.groupby("eeg_id")[HBA_votes].sum()
per_eeg_probs = per_eeg_counts.div(per_eeg_counts.sum(axis=1), axis=0)

val_true_probs = (
    val_split[HBA_votes].div(val_split["total_vote"], axis=0).reset_index(drop=True)
)

candidate_factors = np.arange(0.0, 1.01, 0.05)
best_factor = 0.5
best_score = np.inf

for bf in candidate_factors:
    val_merge = val_split[["eeg_id"]].copy()
    val_merge = val_merge.merge(
        per_eeg_probs,
        how="left",
        left_on="eeg_id",
        right_index=True,
        suffixes=("", "_eeg"),
    )
    per_eeg_filled = val_merge[HBA_votes].fillna(0.0)

    blended = per_eeg_filled * bf + global_probs * (1.0 - bf)
    row_sums = blended.sum(axis=1).replace(0, 1e-12)
    blended_norm = blended.div(row_sums, axis=0)

    kl = np.nansum(
        -val_true_probs.values * np.log(blended_norm.values / val_true_probs.values)
    ) / len(val_true_probs)

    if kl < best_score:
        best_score = kl
        best_factor = bf

fine_start = max(0.0, best_factor - 0.05)
fine_end = min(1.0, best_factor + 0.05)
fine_candidates = np.arange(fine_start, fine_end + 1e-9, 0.01)

for bf in fine_candidates:
    val_merge = val_split[["eeg_id"]].copy()
    val_merge = val_merge.merge(
        per_eeg_probs,
        how="left",
        left_on="eeg_id",
        right_index=True,
        suffixes=("", "_eeg"),
    )
    per_eeg_filled = val_merge[HBA_votes].fillna(0.0)

    blended = per_eeg_filled * bf + global_probs * (1.0 - bf)
    row_sums = blended.sum(axis=1).replace(0, 1e-12)
    blended_norm = blended.div(row_sums, axis=0)

    kl = np.nansum(
        -val_true_probs.values * np.log(blended_norm.values / val_true_probs.values)
    ) / len(val_true_probs)

    if kl < best_score:
        best_score = kl
        best_factor = bf

print(f"Chosen blend factor: {best_factor:.2f} (validation KL={best_score:.5f})")

total_votes_sum_full = train_meta[HBA_votes].sum().sum()
global_probs_full = train_meta[HBA_votes].sum() / total_votes_sum_full

per_eeg_probs_full = train_meta.groupby("eeg_id")[HBA_votes].sum()
per_eeg_probs_full = per_eeg_probs_full.div(per_eeg_probs_full.sum(axis=1), axis=0)

per_eeg_votes_full = train_meta.groupby("eeg_id")["total_vote"].sum()

test_submit = test_meta[["eeg_id"]].copy()
test_submit = test_submit.merge(
    per_eeg_probs_full,
    how="left",
    left_on="eeg_id",
    right_index=True,
    suffixes=("", "_eeg"),
)

test_submit = test_submit.merge(
    per_eeg_votes_full.rename("eeg_total_votes"),
    how="left",
    left_on="eeg_id",
    right_index=True,
)

test_submit[HBA_votes] = test_submit[HBA_votes].fillna(0.0)
test_submit["eeg_total_votes"] = test_submit["eeg_total_votes"].fillna(0.0)

vote_threshold = 60.0
test_submit["weight"] = (test_submit["eeg_total_votes"] / vote_threshold).clip(
    upper=1.0
)

effective_factor = best_factor * test_submit["weight"].values[:, None]

per_eeg_filled_test = test_submit[HBA_votes].values
blended_test = per_eeg_filled_test * effective_factor + global_probs_full.values * (
    1 - effective_factor
)

row_sums_test = blended_test.sum(axis=1).reshape(-1, 1)
row_sums_test[row_sums_test == 0] = 1e-12
blended_norm_test = blended_test / row_sums_test

test_submit[HBA_votes] = blended_norm_test

submission_path = os.path.join(os.getcwd(), "submission.csv")
test_submit.to_csv(
    submission_path, header=True, index=False, na_rep="", float_format="%.6f"
)
print(f"Submission written to {submission_path}")
print("First few rows of submission:")
print(test_submit.head())
