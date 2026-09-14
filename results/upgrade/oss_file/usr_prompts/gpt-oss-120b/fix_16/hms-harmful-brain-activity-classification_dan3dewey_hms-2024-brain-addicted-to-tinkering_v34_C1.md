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

1.036728

# 6. Current score

1.39806

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.2031) has done: 'I compute the true mean class probabilities from the training data and use them in the KL‑taming step instead of the hard‑coded values, which should bring the KL score closer to the target. This adds a small calculation after loading the metadata and updates the `find_best_tamed_kl` function to use the computed means.'
- What this solution (achieved 1.39779) has done: 'I fixed the data path so the CSV files are correctly found, ensured the training metadata is loaded before any later calculations, rewrote the KL‑taming helper to safely use the computed overall class means, and made the submission creation use those means. These changes unblock the script, compute the required mean probabilities, and write a valid `submission.csv` with rows that sum to one.'
- What this solution (achieved 1.39779) has done: 'I keep the overall structure of the script but add a lightweight per‑EEG‑ID probability lookup: after loading the training metadata I compute the mean class probabilities for each unique `eeg_id`. When building the submission I first try to use these per‑EEG probabilities; if an `eeg_id` is not present in the training set I fall back to the global mean. Finally I renormalize each row so the probabilities sum to 1. This small, data‑driven adjustment should lower the KL divergence and move the score toward the target without altering the core modelling logic.'
- What this solution (achieved 1.3966) has done: 'I keep the existing data‑driven per‑EEG means but add a lightweight smoothing step: blend each row with the global class mean and apply a modest temperature scaling, then renormalise so rows sum to 1. This simple calibration reduces extreme probabilities, which usually lowers the KL divergence and moves the score closer to the target without altering the core modelling logic. The changes are limited to the submission‑creation cell and the helper that supplies the global mean.'
- What this solution (achieved 1.39806) has done: 'I reduce the per‑EEG influence and increase the smoothing to lower the KL‑divergence, which moves the score closer to the target (lower is better).  In cell 25 the blend weight is changed from 0.7 to 0.4 so the global mean dominates more.  In cell 29 the temperature is raised from 1.2 to 2.0 to flatten the probabilities further, and a tiny epsilon clipping is added before the final renormalisation to avoid numerical issues.  These adjustments keep the overall pipeline unchanged while making the predictions more conservative, which should reduce the KL score.'
- What this solution (achieved 1.40072) has done: 'I lower the blend weight so the global mean dominates more and increase the temperature to flatten the probability distribution further. These modest calibration tweaks should reduce extreme predictions, bringing the KL‑divergence closer to the target while keeping the core pipeline unchanged.'
- What this solution (achieved 1.40379) has done: 'I reduced the influence of per‑EEG predictions and increased the temperature smoothing so the submission leans more toward the global class distribution, which typically lowers the KL‑divergence on this task. The blend weight is changed from 0.2 to 0.05 and the temperature from 3.0 to 5.0; all other logic and file handling remain unchanged.'
- What this solution (achieved 1.39779) has done: 'I increase the influence of the per‑EEG learned probabilities (set the blend weight to 0 so predictions rely on the data‑driven per‑EEG means) and remove the aggressive temperature flattening (use temperature = 1). These minimal adjustments keep the original pipeline but should produce probabilities closer to the true distribution, lowering the KL‑divergence toward the target score.'
- What this solution (achieved 1.39669) has done: 'I adjust the calibration step to let the per‑EEG statistics influence the predictions and apply a mild temperature scaling. Specifically, `find_best_tamed_kl` now return a blend weight of 0.5 (mixing per‑EEG means with the global mean) and the temperature in cell 29 is set to 1.5. These minimal changes keep the original pipeline intact while making the probability distributions more data‑driven and slightly smoother, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.39806) has done: 'I add a lightweight confidence‑based blending: rows whose eeg_id appears many times in the training set keep more of the per‑eeg mean, while rare eeg_id values rely more on the global class distribution. This uses the existing per‑eeg statistics and only changes the calibration step, keeping the overall pipeline unchanged. I also raise the temperature marginally to 2.0 to smooth predictions a bit further, which together should lower the KL divergence toward the target.'
- What this solution (achieved 1.39779) has done: 'I adjust the calibration step to give a stronger influence to the per‑EEG probabilities and remove the aggressive temperature flattening. Specifically, I replace the count‑based weight with a fixed blend weight (70 % per‑EEG, 30 % global) and set temperature = 1.0, then renormalise the blended rows. This keeps the overall pipeline identical while making the predictions more data‑driven, which should lower the KL‑divergence toward the target.'
- What this solution (achieved 1.39806) has done: 'We reduce the per‑EEG influence by lowering the blend weight from 0.70 to 0.40 and add a mild temperature scaling ( T = 2.0 ) after blending. This flattens overly confident predictions, which typically lowers the KL‑divergence and moves the score closer to the target while keeping the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import pyarrow
import pyarrow.parquet as pq
import pyarrow.dataset as pads

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

possible_dirs = [
    "kaggle/input/hms-harmful-brain-activity-classification/",
    "data/hms-harmful-brain-activity-classification/",
    "./hms-harmful-brain-activity-classification/",
]
above_dir = next((d for d in possible_dirs if os.path.isdir(d)), None)
if above_dir is None:
    raise FileNotFoundError("Could not locate the dataset directory.")

NUM_CLUSTS = 8  # reasonable default number of clusters
SMOOTH_WIDTH = 5  # smoothing width used in feature extraction




## === cell 1
HBA_number = 6
HBA_names = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
HBA_expert_names = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]
iHBA_of_expert = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
HBA_votes = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
HBA_probs = [
    "seizure_prob",
    "lpd_prob",
    "gpd_prob",
    "lrda_prob",
    "grda_prob",
    "other_prob",
]
the4chains = ["LL", "RL", "LP", "RP"]

np.set_printoptions(precision=6, suppress=True)




## === cell 2
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




## === cell 3
def read_hms_meta():
    """
    Read in the train.csv and test.csv files.
    Add total_vote, _prob columns, and vote entropy to train_meta.
    Add extra cols to test to allow the same processing as train:
        eeg[spectro]_sub_id, eeg[spectro]_label_offset_seconds, label_id
    Make various plots of the train_meta values.
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
    REAL_TEST = test_meta_len > 1

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

    print("Calculating voting entropy values ...")

    def calc_entropy(row):
        the_probs = np.clip(row[16 : 21 + 1].values.astype(float), 1.0e-8, 1.0)
        return np.nansum(the_probs * -1 * np.log(the_probs))

    train_meta["entropy"] = train_meta.apply(calc_entropy, axis=1)

    return train_meta, test_meta




## === cell 4
train_meta, test_meta = read_hms_meta()




## === cell 5
overall_mean_probs = train_meta[HBA_probs].mean().values
print("Overall mean class probabilities (from training data):", overall_mean_probs)




## === cell 6
per_eeg_means = (
    train_meta.groupby("eeg_id")[HBA_probs].mean().reset_index().set_index("eeg_id")
)
print(f"Calculated per‑eeg_id means for {per_eeg_means.shape[0]} unique eeg_id values.")




## === cell 7
per_eeg_counts = train_meta.groupby(
    "eeg_id"
).size()  # Series indexed by eeg_id with count of rows
print(
    f"Computed per‑eeg confidence counts for {per_eeg_counts.shape[0]} eeg_id values."
)




## === cell 8
pass




## === cell 9
pass




## === cell 10
pass




## === cell 11
pass




## === cell 12
pass




## === cell 13
pass




## === cell 14
pass




## === cell 15
pass




## === cell 16
pass




## === cell 17
pass




## === cell 18
pass




## === cell 19
pass




## === cell 20
pass




## === cell 21
pass




## === cell 22
pass




## === cell 23
pass




## === cell 24
def find_best_tamed_kl(mean_all_probs=None):
    """
    Returns the global mean probabilities and the per‑eeg confidence weights.
    Blending will be performed later row‑wise using these confidence weights.
    """
    if mean_all_probs is None:
        mean_all_probs = overall_mean_probs
    return per_eeg_counts, mean_all_probs




## === cell 25
per_eeg_counts_series, global_mean = find_best_tamed_kl()
print(
    f"Using per‑eeg confidence counts (min={per_eeg_counts_series.min()}, max={per_eeg_counts_series.max()}) for KL‑taming."
)




## === cell 26
pass




## === cell 27
pass




## === cell 28
submission = test_meta[["eeg_id"]].copy()
joined = submission.join(per_eeg_means, on="eeg_id", how="left")
joined_counts = submission.join(
    per_eeg_counts_series.rename("eeg_count"), on="eeg_id", how="left"
)

for prob_col, vote_col, global_val in zip(HBA_probs, HBA_votes, overall_mean_probs):
    joined[vote_col] = joined[prob_col].fillna(global_val)

submission = joined[["eeg_id"] + HBA_votes].copy()

blend_weight = 0.40  # less reliance on per‑eeg means
temperature = 2.0  # >1 flattens the distribution

weight_series = np.full((submission.shape[0], 1), blend_weight)  # (n_rows, 1)

blended = (
    weight_series * submission[HBA_votes].values + (1 - weight_series) * global_mean
)

scaled = np.clip(blended, 1e-8, None)
scaled = scaled ** (1.0 / temperature)

scaled = scaled / scaled.sum(axis=1, keepdims=True)

submission[HBA_votes] = scaled

print("First few calibrated rows:")
print(submission.head())

try:
    submission.to_csv(
        "submission.csv", header=True, index=False, na_rep="", float_format="%.6f"
    )
    print("submission.csv written successfully.")
except Exception as e:
    print("Error writing submission.csv:", e)




## === cell 29
pass
