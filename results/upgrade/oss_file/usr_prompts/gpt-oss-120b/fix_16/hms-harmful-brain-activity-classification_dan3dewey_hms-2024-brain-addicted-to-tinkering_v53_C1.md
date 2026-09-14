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

1.015208464728662

# 6. Current score

1.64506

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39874) has done: 'I set the preprocessing flag to False, guard all model‑training blocks so they skip when the required data isn’t available, and fall back to using the overall class‑wise mean probabilities from the training set to create a valid `submission.csv`. This removes the FileNotFound errors, avoids undefined‑variable crashes, and still produces a submission whose KL score be close to the target baseline.'
- What this solution (achieved 1.40171) has done: 'I increase the number of K‑means clusters to capture finer probability patterns and adjust the taming loop to also consider the pure cluster centre (fraction = 1.0). These small tweaks should move the KL score closer to the target lower value while keeping the overall pipeline unchanged.'
- What this solution (achieved 1.40339) has done: 'I adjust the clustering step to use the full training set (instead of the small `<200` slice) so the K‑means centroids better represent the true distribution, and I limit the taming fraction search to a maximum of 0.5. This pulls the final predictions closer to the overall class‑wise mean, reducing over‑confident errors and should lower the KL divergence toward the target while keeping the original pipeline intact.'
- What this solution (achieved 1.39779) has done: 'I adjust the KL‑scoring helper to safely clip probabilities and broaden the taming search to include the pure‑mean case (fraction 0). Then I simplify the final prediction step so that, after the (optional) taming search, the submission is set directly to the overall class‑wise mean probabilities for every test row. Using the uniform‑mean baseline is expected to lower the KL score toward the target 1.015 while keeping the original pipeline intact.'
- What this solution (achieved 1.39779) has done: 'I keep the original pipeline but stop overwriting the clustered, tamed predictions with the overall mean. After the “find_best_tamed_kl” search I assign each test row the corresponding tamed cluster centre, then renormalise to guarantee rows sum to 1. I also broaden the taming‑fraction search to include the pure cluster centre (up to 1.0) so the optimizer can keep useful confidence when it helps lower KL. These minimal edits let the model use the learned cluster structure instead of a naïve uniform baseline, moving the KL score toward the target lower value.'
- What this solution (achieved 1.39779) has done: 'I limit the taming search to fractions up to 0.5 so the optimizer cannot keep overly confident cluster centres, and then blend the final predictions slightly toward the overall class‑wise mean (20 % mean, 80 % tuned centres). This reduces over‑confidence and should lower the KL divergence, moving the score closer to the target while keeping the original pipeline untouched.'
- What this solution (achieved 1.64506) has done: 'I replace the per‑patient cluster‑based prediction with the patient‑specific mean probabilities computed from the training set. This uses richer patient‑level information and keeps probabilities normalized, which should lower the KL divergence toward the target score while preserving the overall pipeline.'

# 9. Code solution

## === cell 0
NUM_CLUSTS = 12  # increased from 9 to capture finer patterns
SMOOTH_WIDTH = 5  # Odd>1: 3,5,7,9,...
TRAIN_DOWNSEL = 23  # small values for code test; set to 1 to output all.
VALID_DOWNSEL = 47  #  "
USE_PREPROC = False  # Force on‑the‑fly processing (pre‑saved files missing)
USE_LR1 = False
LR1_C = 0.10  # smaller --> fewer non-zero coeff.s
USE_LR2 = False
LR2_C = 0.10
LR_BLUR = 0.10
TRAIN_RF = False  # Skip heavy RandomForest training for fast execution

above_dir = "../input/hms-harmful-brain-activity-classification/"
above_dir_preproc = "../input/hms-2024-brain-data/"

import os

if not os.path.isdir(above_dir):
    above_dir = "./data/hms-harmful-brain-activity-classification/"
if not os.path.isdir(above_dir_preproc):
    above_dir_preproc = "./data/hms-2024-brain-data/"



## === cell 1
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



## === cell 2
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




## === cell 3
def kld_score(solution, submission):
    """
    Calculate the average KL divergence score.
    Ignores the "row id" assumed in the first column.
    """
    eps = 1e-12  # avoid log(0)
    sumsum = 0.0
    for prob_col in solution.columns.values:
        true_p = solution[prob_col].values
        pred_p = np.clip(submission[prob_col].values, eps, 1.0)
        sumsum += np.nansum(-1.0 * true_p * np.log(pred_p / true_p))
    return sumsum / (len(solution))




## === cell 4
def read_hms_meta():
    """
    Read train.csv and test.csv, add total_vote and probability columns.
    """
    test_meta = pd.read_csv(above_dir + "test.csv")
    test_meta["eeg_sub_id"] = 0
    test_meta["eeg_label_offset_seconds"] = 0.0
    test_meta["spectrogram_sub_id"] = 0
    test_meta["spectrogram_label_offset_seconds"] = 0.0
    test_meta["label_id"] = test_meta.eeg_id

    train_meta = pd.read_csv(above_dir + "train.csv")
    train_meta["total_vote"] = (
        train_meta["seizure_vote"]
        + train_meta["lpd_vote"]
        + train_meta["gpd_vote"]
        + train_meta["lrda_vote"]
        + train_meta["grda_vote"]
        + train_meta["other_vote"]
    )
    for col_pre in HBA_names:
        train_meta[col_pre + "_prob"] = (
            train_meta[col_pre + "_vote"] / train_meta["total_vote"]
        )
    return train_meta, test_meta




## === cell 5
train_meta, test_meta = read_hms_meta()



## === cell 6
mean_all_probs = train_meta[HBA_probs].mean().values  # shape (6,)



## === cell 7
clust_rows_bool = np.ones(len(train_meta), dtype=bool)  # all rows
prob_vectors = train_meta.loc[clust_rows_bool, HBA_probs]
prob_array = np.array(prob_vectors)
kmeans = KMeans(
    n_clusters=NUM_CLUSTS, init="k-means++", n_init=10, max_iter=300, random_state=0
)
kmeans.fit(prob_array)
clust_centers = kmeans.cluster_centers_
clust_centers = clust_centers / clust_centers.sum(axis=1, keepdims=True)



## === cell 8
train_meta["clust_id"] = kmeans.predict(np.array(train_meta[HBA_probs]))



## === cell 9
lrmodel1 = None
lrmodel2 = None
if USE_LR1 or USE_LR2:
    X_dummy = train_meta[HBA_probs]
    y_dummy = train_meta["clust_id"]
    if USE_LR1:
        lrmodel1 = LogisticRegression(
            penalty="l1",
            C=LR1_C,
            solver="saga",
            max_iter=500,
            multi_class="multinomial",
            n_jobs=-1,
        )
        lrmodel1.fit(X_dummy, y_dummy)
    if USE_LR2:
        lrmodel2 = LogisticRegression(
            penalty="l1",
            C=LR2_C,
            solver="saga",
            max_iter=500,
            multi_class="multinomial",
            n_jobs=-1,
        )
        lrmodel2.fit(X_dummy, y_dummy)

rfmodel = None
if TRAIN_RF:
    X_rf = train_meta[HBA_probs]
    y_rf = train_meta["clust_id"]
    rfmodel = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        max_features=0.2,
        max_samples=0.9,
        oob_score=True,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=0,
    )
    rfmodel.fit(X_rf, y_rf)




## === cell 10
def find_best_tamed_kl(solution, submission, pred_ids, clust_centers, mean_all_probs):
    """
    For each cluster, search blending fractions from 0.0 (pure mean) to 1.0 (pure cluster centre)
    and keep the fraction that yields the lowest KL on the training proxy data.
    """
    num_clusters = clust_centers.shape[0]
    num_hba = len(HBA_votes)
    best_fracs = np.zeros(num_clusters)
    best_centers = clust_centers.copy()

    for iclust in range(num_clusters):
        best_kl = np.inf
        best_center = clust_centers[iclust].copy()
        for frac in np.arange(0.0, 1.01, 0.05):
            candidate = frac * clust_centers[iclust] + (1.0 - frac) * mean_all_probs
            submission_tmp = submission.copy()
            for iprob in range(num_hba):
                submission_tmp[HBA_votes[iprob]] = candidate[iprob]
            kl = kld_score(solution, submission_tmp)
            if kl < best_kl:
                best_kl = kl
                best_center = candidate.copy()
                best_fracs[iclust] = frac
        best_centers[iclust] = best_center
    return best_fracs, best_centers




## === cell 11
train_solution = train_meta[["eeg_id"] + HBA_votes].copy()
for col in HBA_votes:
    train_solution[col] = train_meta[col.replace("_vote", "_prob")]

train_proxy_sub = pd.DataFrame(
    np.tile(mean_all_probs, (len(train_meta), 1)),
    columns=HBA_votes,
)
train_proxy_sub["eeg_id"] = train_meta["eeg_id"]

best_fracs, best_centers = find_best_tamed_kl(
    solution=train_solution,
    submission=train_proxy_sub.copy(),
    pred_ids=train_meta["clust_id"].values,
    clust_centers=clust_centers,
    mean_all_probs=mean_all_probs,
)

patient_means = train_meta.groupby("patient_id")[HBA_probs].mean().reset_index()
patient_means = patient_means.rename(columns=dict(zip(HBA_probs, HBA_votes)))

submission = pd.DataFrame(
    np.tile(mean_all_probs, (len(test_meta), 1)),
    columns=HBA_votes,
    index=test_meta.index,
)
submission.insert(0, "eeg_id", test_meta["eeg_id"])
submission["patient_id"] = test_meta["patient_id"]

submission = submission.merge(
    patient_means, on="patient_id", how="left", suffixes=("", "_patient")
)

for col in HBA_votes:
    patient_col = f"{col}_patient"
    submission[col] = np.where(
        submission[patient_col].notna(), submission[patient_col], submission[col]
    )
    submission.drop(columns=[patient_col], inplace=True)

submission.drop(columns=["patient_id"], inplace=True)
submission[HBA_votes] = submission[HBA_votes].div(
    submission[HBA_votes].sum(axis=1), axis=0
)

submission_path = "submission.csv"
submission.to_csv(submission_path, header=True, index=False, float_format="%.6f")
print(f"Submission written to {submission_path}")



## === cell 12
kl_proxy = kld_score(train_solution, train_proxy_sub)
print(f"Proxy KL on training data (uniform baseline): {kl_proxy:.4f}")
