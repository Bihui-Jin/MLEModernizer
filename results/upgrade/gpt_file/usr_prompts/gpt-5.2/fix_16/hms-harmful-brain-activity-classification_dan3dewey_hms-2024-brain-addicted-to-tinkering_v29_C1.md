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

1.036779

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.28969) has done: 'Your code already has the right end-to-end structure to generate `submission.csv`, so the main reason you likely got “Not yielded” is that it can error out while reading spectrogram parquet files (missing paths/NaNs/offset bounds) or silently produce invalid probabilities (NaNs / row sums not 1) before writing. I make minimal, score-relevant hardening changes: (1) make `above_dir` auto-detect the correct input path so file reads always work, (2) make spectrogram row indexing robust by clipping `loc_offset` into valid bounds, and (3) ensure every probability vector is strictly valid for KL (floor + renorm) both during validation scoring and final submission. These changes keep your model/feature logic intact but prevent failures and reduce catastrophic KL from occasional invalid/near-zero predictions.'

# 9. Code solution

## === cell 0
NUM_CLUSTS = 6
SMOOTH_WIDTH = 5

import os

CANDIDATE_DIRS = [
    "../input/hms-harmful-brain-activity-classification/",
    "/kaggle/input/hms-harmful-brain-activity-classification/",
    "/kaggle/data/hms-harmful-brain-activity-classification/",
    "/kaggle/data/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification/",
]
above_dir = None
for d in CANDIDATE_DIRS:
    d2 = d if d.endswith("/") else d + "/"
    if os.path.exists(os.path.join(d2, "train.csv")) and os.path.exists(
        os.path.join(d2, "test.csv")
    ):
        above_dir = d2
        break
if above_dir is None:
    above_dir = "../input/hms-harmful-brain-activity-classification/"

SEED = 42

PROB_FLOOR = 1e-6

SHOW_PLOTS = False


def _has_required_assets(root: str) -> bool:
    root = root if root.endswith("/") else root + "/"
    return (
        os.path.exists(os.path.join(root, "train.csv"))
        and os.path.exists(os.path.join(root, "test.csv"))
        and os.path.isdir(os.path.join(root, "train_spectrograms"))
        and os.path.isdir(os.path.join(root, "test_spectrograms"))
    )


if not _has_required_assets(above_dir):
    for d in CANDIDATE_DIRS:
        if _has_required_assets(d):
            above_dir = d if d.endswith("/") else d + "/"
            break

assert _has_required_assets(
    above_dir
), f"Could not find required HMS assets under any candidate dir. Last tried: {above_dir}"

print("Using data root:", above_dir)



## === cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import pyarrow
import pyarrow.dataset as pads

from sklearn.ensemble import RandomForestClassifier



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
def _sanitize_probs(arr, prob_floor=1e-6):
    """Ensure strictly positive, finite, and row-normalized probs for KL + Kaggle submission."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = np.nan_to_num(
        arr,
        nan=1.0 / arr.shape[1],
        posinf=1.0 / arr.shape[1],
        neginf=1.0 / arr.shape[1],
    )
    arr = np.clip(arr, prob_floor, 1.0)
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    return arr / row_sums


def kld_score(solution, submission, prob_floor=1e-6):
    """
    Calculate the average KL divergence score.
    Assumes columns are the prob columns (vote columns used as probabilities here).
    """
    sol_vals = solution.to_numpy(dtype=np.float64)
    sub_vals = submission.to_numpy(dtype=np.float64)

    sol_vals = _sanitize_probs(sol_vals, prob_floor=prob_floor)
    sub_vals = _sanitize_probs(sub_vals, prob_floor=prob_floor)

    kls = np.sum(sol_vals * (np.log(sol_vals) - np.log(sub_vals)), axis=1)
    return float(np.mean(kls))




## === cell 4
def read_hms_meta():
    """
    Read in the train.csv and test.csv files.
    Add total_vote, _prob columns, and vote entropy to train_meta.
    Add extra cols to test to allow the same processing as train:
        eeg[spectro]_sub_id, eeg[spectro]_label_offset_seconds, label_id
    """
    test_meta = pd.read_csv(above_dir + "test.csv")
    test_meta_len = len(test_meta)
    print("Test has length", test_meta_len)
    test_meta["eeg_sub_id"] = 0
    test_meta["eeg_label_offset_seconds"] = 0.0
    test_meta["spectrogram_sub_id"] = 0
    test_meta["spectrogram_label_offset_seconds"] = 0.0
    test_meta["label_id"] = test_meta.eeg_id

    train_meta = pd.read_csv(above_dir + "train.csv")
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

    if SHOW_PLOTS:
        print("\nHistogram of the total votes")
        plt.figure(figsize=(6, 3))
        plt.hist(train_meta["total_vote"], bins=55, log=True)
        plt.title("Histogram of Total Votes")
        plt.show()

    train_meta["total_vote"] = train_meta["total_vote"].clip(lower=1)

    for col_pre in HBA_names:
        train_meta[col_pre + "_prob"] = (
            train_meta[col_pre + "_vote"] / train_meta["total_vote"]
        )

    def calc_entropy(row):
        the_probs = np.clip(row[HBA_probs].values.astype(float), 1.0e-8, 1.0)
        the_probs = the_probs / np.sum(the_probs)
        return np.nansum(the_probs * -1 * np.log(the_probs))

    train_meta["entropy"] = train_meta.apply(calc_entropy, axis=1)

    return train_meta, test_meta




## === cell 5
def prob_prob_scatter(name1, name2, probs2plot, clust_ids, iclust_order=[0]):
    """
    Make a prob1 vs prob2 scatter plot.
    External: HBA_probs, iHBA_of_expert[ ], clust_centers
    """
    hba_clrs = [
        "orange",
        "blue",
        "red",
        "black",
        "green",
        "purple",
        "green",
        "red",
        "blue",
        "orange",
    ]  # up to 10 clusters
    if len(iclust_order) > 2:
        kmclrs = hba_clrs.copy()
        for iord, iclust in enumerate(iclust_order):
            kmclrs[iclust] = hba_clrs[iord]
    else:
        kmclrs = hba_clrs.copy()

    clstclrs = [kmclrs[ilab] for ilab in clust_ids]

    ixax = iHBA_of_expert[name1]
    iyax = iHBA_of_expert[name2]
    lenprob = len(probs2plot)
    plt.figure(figsize=(5, 5))
    plt.scatter(
        np.sqrt(probs2plot[HBA_probs[ixax]]) + 0.04 * (np.random.rand(lenprob) - 0.5),
        np.sqrt(probs2plot[HBA_probs[iyax]]) + 0.04 * (np.random.rand(lenprob) - 0.5),
        s=3,
        c=clstclrs,
        alpha=0.02,
    )
    for iclust in range(0, len(clust_centers)):
        plt.plot(
            np.sqrt([clust_centers[iclust, ixax]]),
            np.sqrt([clust_centers[iclust, iyax]]),
            c=kmclrs[iclust],
            marker="x",
            markersize=15,
        )
    plt.xlabel("sqrt( " + name1 + " )")
    plt.ylabel("sqrt( " + name2 + " )")
    plt.show()
    return kmclrs




## === cell 6
train_meta, test_meta = read_hms_meta()



## === cell 7
if SHOW_PLOTS:
    plt.figure(figsize=(5, 2))
    plt.hist(train_meta["spectrogram_sub_id"], bins=55, log=True)
    plt.title("Histogram of spectrogram_sub_id")
    plt.show()

    plt.figure(figsize=(5, 2))
    plt.hist(train_meta["eeg_sub_id"], bins=55, log=True)
    plt.title("Histogram of eeg_sub_id")
    plt.show()



## === cell 8
num_clusts = NUM_CLUSTS  # 6 to 10

clust_rows_bool = (
    ((train_meta.eeg_sub_id < 56) & (train_meta.eeg_sub_id % 7 == 6))
    | (train_meta.eeg_sub_id == 0)
) & (train_meta.eeg_id % 4 < 2)



## === cell 9
prob_vectors = train_meta.loc[clust_rows_bool, HBA_probs]

print("\nUsing {} HBA samples for clustering.".format(len(prob_vectors)))
print(
    "These include {} unique eeg_ids".format(
        train_meta.loc[clust_rows_bool, "eeg_id"].nunique()
    ),
    "and {} unique patient ids.".format(
        train_meta.loc[clust_rows_bool, "patient_id"].nunique()
    ),
)

if SHOW_PLOTS:
    plt.figure(figsize=(6, 3))
    plt.hist(train_meta.loc[clust_rows_bool, "total_vote"], bins=100, log=True)
    plt.title("Histogram of total_vote in HBA samples clustered")
    plt.show()

prob_array = np.array(prob_vectors)

kmeans = KMeans(
    n_clusters=num_clusts, init="k-means++", n_init=10, max_iter=300, random_state=SEED
)
kmeans.fit(prob_array)

clust_centers = kmeans.cluster_centers_

for iclust in range(num_clusts):
    denom = np.sum(clust_centers[iclust, :])
    if denom > 0:
        clust_centers[iclust, :] = clust_centers[iclust, :] / denom

iclust_of_order = []
for icol in range(HBA_number):
    iclust_of_order.append(np.argmax(clust_centers[:, icol]))
clust_by_max = np.argsort(-1 * np.max(clust_centers, axis=1))
for iord in range(HBA_number, num_clusts):
    iclust_of_order.append(clust_by_max[iord])

kmnames = HBA_expert_names.copy()
for ihyb in range(1, (num_clusts - HBA_number) + 1):
    kmnames.append("Hybrid-" + str(ihyb))



## === cell 10
train_meta["clust_id"] = kmeans.predict(np.array(train_meta[HBA_probs]))

all_probs = train_meta[HBA_probs]
all_ids = train_meta["clust_id"]

if SHOW_PLOTS:
    kmclrs = prob_prob_scatter("Seizure", "GPD", all_probs, all_ids, iclust_of_order)
    kmclrs = prob_prob_scatter("LPD", "GRDA", all_probs, all_ids, iclust_of_order)
    kmclrs = prob_prob_scatter("LRDA", "Other", all_probs, all_ids, iclust_of_order)
else:
    kmclrs = [
        "orange",
        "blue",
        "red",
        "black",
        "green",
        "purple",
        "green",
        "red",
        "blue",
        "orange",
    ]

clust_counts = train_meta.clust_id.value_counts()

if SHOW_PLOTS:
    for iord, iclust in enumerate(iclust_of_order):
        if iord == 0:
            print("The close-to-unit-vector cluster centers:")
        if iord == 6:
            print("The Hybrid cluster centers:")
        plt.figure(figsize=(5, 1))
        plt.bar(HBA_expert_names, clust_centers[iclust, :], color=kmclrs[iclust])
        plt.ylim(-0.01, 1.01)
        plt.title(
            kmnames[iord]
            + "  kmclust={} has {} samples".format(iclust, clust_counts[iclust]),
            size="medium",
        )
        if iord < 5:
            plt.xticks([])
        plt.show()



## === cell 11
solution_train = train_meta[["eeg_id"] + HBA_votes].copy()
for col_pre in HBA_names:
    solution_train[col_pre + "_vote"] = train_meta[col_pre + "_prob"].to_numpy()

submission_train = solution_train.copy()
clust_ids = train_meta["clust_id"].to_numpy()
for iprob in range(HBA_number):
    this_col_probs = clust_centers[:, iprob]
    submission_train[HBA_votes[iprob]] = this_col_probs[clust_ids]

print(
    "Score if HBA samples are correctly assigned cluster prob.s:",
    np.round(kld_score(solution_train[HBA_votes], submission_train[HBA_votes]), 4),
)



## === cell 12
smooth_width = SMOOTH_WIDTH  # Here smooth_width is just for looking,
spectro_meta = train_meta[clust_rows_bool].copy()
every_nth = 37



## === cell 13
freqs = np.array(range(100)) * 0.19525 + 0.59
spect_trend = 150.0 / (1.0**2.3 + freqs ** (2.3))



## === cell 14
if SHOW_PLOTS:
    plt.figure(figsize=(10, 8))

    all_medians = []
    all_means = []
    all_clusts = []
    print(
        "Plotting {} x 4 processed spectra".format(int(len(spectro_meta) / every_nth))
    )
    for ilocrow in range(0, len(spectro_meta), every_nth):
        this_row = spectro_meta.iloc[ilocrow]
        this_clust = this_row.clust_id
        spectro_id_str = str(this_row.spectrogram_id)
        spectro_file = above_dir + "train_spectrograms/" + spectro_id_str + ".parquet"
        pads_spectro = pads.dataset(spectro_file)
        this_spectro = pads_spectro.to_table().to_pandas()

        loc_offset = int(this_row.spectrogram_label_offset_seconds / 2)
        max_off = max(0, len(this_spectro) - 151)
        loc_offset = int(np.clip(loc_offset, 0, max_off))

        for ispec in range(4):
            ibeg = ([1, 101, 201, 301])[ispec]
            iend = ibeg + 100
            middle4s = (
                this_spectro.iloc[loc_offset + 149, ibeg:iend]
                + this_spectro.iloc[loc_offset + 150, ibeg:iend]
            ) / (2.0 * spect_trend)
            middle4s = np.clip(middle4s, 0.001, 1000.0)
            middle4s = middle4s.replace([np.nan, -np.inf, np.inf], 0.001)
            spect_mean = np.log10(np.mean(middle4s))
            all_means.append(spect_mean)
            spect_median = np.log10(np.median(middle4s))
            all_medians.append(spect_median)
            all_clusts.append(this_clust)
            middle4s = np.log10(middle4s)
            middle4s = middle4s - spect_mean
            middle4s = middle4s.rolling(
                smooth_width, min_periods=1, center=True, closed=None
            ).mean()
            plt.plot((freqs), middle4s, c=kmclrs[this_clust], lw=3, alpha=0.01)
        if (ilocrow / every_nth + 1) % 100 == 0:
            print("... {} done...".format(int(ilocrow / every_nth) + 1))

    plt.plot([0.0, 20.0], [0.0, 0.0], c="black", lw=3, alpha=0.2)
    downsel_freqs = freqs[int((smooth_width - 1) / 2) : 100 : smooth_width]
    plt.plot(downsel_freqs, len(downsel_freqs) * [0.0], ".k")
    plt.ylim(-1.0, 1.0)
    plt.xlim(0.0, 20.5)
    plt.title("Middle-4s Spectra (color-coded by the {} clusters)".format(num_clusts))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("log10[] Amplitude (/ref-spectrum, /mean, and smoothed n=3) ]")
    plt.savefig("middle4s_spectra.png")
    plt.show()



## === cell 15
_SPECTRO_CACHE = {}
_SPECTRO_CACHE_ORDER = []
_SPECTRO_CACHE_MAX = 64


def _cache_get(key):
    return _SPECTRO_CACHE.get(key, None)


def _cache_put(key, val):
    if key in _SPECTRO_CACHE:
        return
    _SPECTRO_CACHE[key] = val
    _SPECTRO_CACHE_ORDER.append(key)
    if len(_SPECTRO_CACHE_ORDER) > _SPECTRO_CACHE_MAX:
        k0 = _SPECTRO_CACHE_ORDER.pop(0)
        _SPECTRO_CACHE.pop(k0, None)


def _load_spectrogram_df(traintest, spectrogram_id):
    """
    Robust parquet loading for spectrograms.

    Change (execution safety -> prevents "Not yielded"): always read from validated above_dir,
    and return None on any issue (handled upstream).
    """
    sid = str(int(spectrogram_id))
    key = (traintest, sid)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    rel = f"{traintest}_spectrograms/{sid}.parquet"
    p = os.path.join(above_dir, rel)
    if not os.path.exists(p):
        return None

    df = None
    try:
        df = pd.read_parquet(p)
    except Exception:
        try:
            df = pads.dataset(p).to_table().to_pandas()
        except Exception:
            df = None
    if df is None or len(df) == 0:
        return None

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    _cache_put(key, df)
    return df


def assemble_features(meta_frame, traintest="train", smooth_width=5, SHOW_PLOT=True):
    """
    Create a dataframe of spectrogram features from the meta_frame rows.
    Will include clust_id (i.e, the y) if it is in the input meta_frame.

    Change (submission validity + score stability): clip offsets into bounds, and for any bad spectrogram
    emit stable global features (prevents NaNs/inf features => bad probs => huge KL).
    """
    freqs = np.array(range(100)) * 0.19525 + 0.59
    spect_trend = 150.0 / (1.0**2.3 + freqs ** (2.3))
    freqs4 = np.array(4 * list(freqs))
    spect_trend4 = np.array(4 * list(spect_trend))
    if SHOW_PLOT:
        plt.figure(figsize=(10, 8))
    feats_frame = []
    last_spectro_id_str = "starting"
    this_spectro = None

    baseinds = np.arange(int((smooth_width - 1) / 2), 100, smooth_width)
    select_inds = np.concatenate(
        (baseinds, 100 + baseinds, 200 + baseinds, 300 + baseinds)
    )
    nband = len(select_inds)
    feat_band_cols = list(range(nband))
    aux_cols = (
        ["Mean", "Median"]
        + [c + "mean" for c in the4chains]
        + [c + "median" for c in the4chains]
    )
    all_feat_cols = feat_band_cols + aux_cols

    def _smooth1d(x, w):
        if w <= 1:
            return x
        k = np.ones(int(w), dtype=np.float64) / float(w)
        x = np.asarray(x, dtype=np.float64)
        pad = int(w) // 2
        xp = np.pad(x, (pad, pad), mode="edge")
        return np.convolve(xp, k, mode="valid")

    global_feat_values = np.zeros(nband, dtype=np.float64)
    global_aux = np.zeros(2 + 2 * len(the4chains), dtype=np.float64)

    try:
        _rows = meta_frame.head(80)
        tmp_feats = []
        last_id = None
        tmp_spec = None
        for r in _rows.itertuples(index=False):
            sid = str(int(r.spectrogram_id))
            if sid != last_id:
                tmp_spec = _load_spectrogram_df(traintest, r.spectrogram_id)
                last_id = sid
            if tmp_spec is None or len(tmp_spec) < 151 or tmp_spec.shape[1] < 401:
                continue

            loc_offset = int(getattr(r, "spectrogram_label_offset_seconds") / 2)
            max_off = max(0, len(tmp_spec) - 151)
            loc_offset = int(np.clip(loc_offset, 0, max_off))

            v1 = tmp_spec.iloc[loc_offset + 149, -400:].to_numpy(
                dtype=np.float64, copy=False
            )
            v2 = tmp_spec.iloc[loc_offset + 150, -400:].to_numpy(
                dtype=np.float64, copy=False
            )
            m4 = (v1 + v2) / (2.0 * spect_trend4)
            m4 = np.nan_to_num(m4, nan=0.001, posinf=0.001, neginf=0.001)
            m4 = np.clip(m4, 0.001, 1000.0)

            spect_mean = float(np.mean(m4))
            spect_median = float(np.median(m4))
            the4means = []
            the4medians = []
            for ispec in range(4):
                ibeg = ([0, 100, 200, 300])[ispec]
                iend = ibeg + 100
                the4means.append(float(np.mean(m4[ibeg:iend])))
                the4medians.append(float(np.median(m4[ibeg:iend])))

            v = np.log10(m4 / spect_mean)
            v = _smooth1d(v, smooth_width)
            vds = v[select_inds].astype(np.float64, copy=False)

            aux = np.array(
                [np.log10(spect_mean), np.log10(spect_median)]
                + list(np.log10(np.array(the4means)))
                + list(np.log10(np.array(the4medians))),
                dtype=np.float64,
            )
            tmp_feats.append(np.concatenate([vds, aux]))
        if len(tmp_feats) > 0:
            tmp_feats = np.vstack(tmp_feats)
            global_feat_values = np.nanmean(tmp_feats[:, :nband], axis=0)
            global_aux = np.nanmean(tmp_feats[:, nband:], axis=0)
            global_feat_values = np.nan_to_num(
                global_feat_values, nan=0.0, posinf=0.0, neginf=0.0
            )
            global_aux = np.nan_to_num(global_aux, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        pass

    for r in meta_frame.itertuples(index=False):
        spectro_id_str = str(int(r.spectrogram_id))

        spec_ok = True
        if spectro_id_str != last_spectro_id_str:
            this_spectro = _load_spectrogram_df(traintest, r.spectrogram_id)
            if (
                this_spectro is None
                or len(this_spectro) < 151
                or this_spectro.shape[1] < 401
            ):
                spec_ok = False
                this_spectro = None
        last_spectro_id_str = spectro_id_str

        if spec_ok:
            try:
                loc_offset = int(getattr(r, "spectrogram_label_offset_seconds") / 2)
                max_off = max(0, len(this_spectro) - 151)
                loc_offset = int(np.clip(loc_offset, 0, max_off))

                v1 = this_spectro.iloc[loc_offset + 149, -400:].to_numpy(
                    dtype=np.float64, copy=False
                )
                v2 = this_spectro.iloc[loc_offset + 150, -400:].to_numpy(
                    dtype=np.float64, copy=False
                )
                middle4s = (v1 + v2) / (2.0 * spect_trend4)
                middle4s = np.nan_to_num(
                    middle4s, nan=0.001, posinf=0.001, neginf=0.001
                )
                middle4s = np.clip(middle4s, 0.001, 1000.0)

                spect_mean = float(np.mean(middle4s))
                spect_median = float(np.median(middle4s))
                the4means = []
                the4medians = []
                for ispec in range(4):
                    ibeg = ([0, 100, 200, 300])[ispec]
                    iend = ibeg + 100
                    the4means.append(float(np.mean(middle4s[ibeg:iend])))
                    the4medians.append(float(np.median(middle4s[ibeg:iend])))

                middle4s_log = np.log10(middle4s / spect_mean)
                middle4s_log = _smooth1d(middle4s_log, smooth_width)

                freqs4ds = freqs4[select_inds]
                middle4sds = middle4s_log[select_inds]

                these_feats = pd.DataFrame(
                    [middle4sds.astype(np.float64, copy=False)], columns=feat_band_cols
                )

                spect_mean = np.log10(spect_mean)
                spect_median = np.log10(spect_median)
                the4means = np.log10(np.array(the4means))
                the4medians = np.log10(np.array(the4medians))
                these_feats["Mean"] = float(spect_mean)
                these_feats["Median"] = float(spect_median)
                for ispec in range(4):
                    these_feats[the4chains[ispec] + "mean"] = float(the4means[ispec])
                    these_feats[the4chains[ispec] + "median"] = float(
                        the4medians[ispec]
                    )
            except Exception:
                spec_ok = False

        if not spec_ok:
            these_feats = pd.DataFrame([global_feat_values], columns=feat_band_cols)
            these_feats["Mean"] = float(global_aux[0])
            these_feats["Median"] = float(global_aux[1])
            for ispec in range(4):
                these_feats[the4chains[ispec] + "mean"] = float(global_aux[2 + ispec])
                these_feats[the4chains[ispec] + "median"] = float(
                    global_aux[2 + 4 + ispec]
                )

        these_feats = these_feats.reindex(columns=all_feat_cols)

        if "clust_id" in meta_frame.columns:
            these_feats["clust_id"] = getattr(r, "clust_id")

        if len(feats_frame) == 0:
            feats_frame = these_feats.copy()
        else:
            feats_frame = pd.concat([feats_frame, these_feats], ignore_index=True)

        if len(feats_frame) % 100 == 0:
            print("... {} done...".format(len(feats_frame)))

        if SHOW_PLOT:
            this_clr = "blue"
            plt.title("Middle-4s Feature Values (smooth={})".format(smooth_width))
            if "clust_id" in meta_frame.columns:
                this_clust = getattr(r, "clust_id")
                this_clr = kmclrs[this_clust]
                plt.title(
                    "Middle-4s Feature Values"
                    + " (color-coded by the {} clusters, smooth={})".format(
                        num_clusts, smooth_width
                    )
                )
            the_alpha = np.clip(0.05 * 350 / len(meta_frame), 0.003, 0.5)

            try:
                yplot = middle4sds
                xplot = freqs4ds
            except Exception:
                yplot = global_feat_values
                xplot = freqs4[select_inds]

            plt.plot(
                xplot + smooth_width * 0.15 * (np.random.rand(len(xplot)) - 0.5),
                yplot,
                ".",
                c=this_clr,
                markersize=10,
                alpha=the_alpha,
            )
            plt.ylim(-1.5, 1.5)
            plt.xlabel(
                "<-- Means are band < 0" + 20 * " " + "Frequency Bands (Hz)" + 50 * " "
            )
            plt.ylabel(
                "Amplitude (/ref-spectrum, /mean, and smoothed n={})".format(
                    smooth_width
                )
            )
    if SHOW_PLOT:
        plt.savefig("middle4s_features.png")
        plt.show()
    return feats_frame.reset_index().drop(columns=["index"])




## === cell 16
Xy_train_meta = (train_meta[clust_rows_bool])[::23].copy()
Xy_train_meta = Xy_train_meta.reset_index().drop(columns=["index"])
print("Number of samples used for training =", len(Xy_train_meta))

Xy_train_feats = assemble_features(
    Xy_train_meta, traintest="train", smooth_width=SMOOTH_WIDTH, SHOW_PLOT=SHOW_PLOTS
)
Xy_train_meta.to_csv("Xy_train_meta.csv", header=True, index=False, float_format="%.6f")
Xy_train_feats.to_csv(
    "Xv_train_feats.csv", header=True, index=False, float_format="%.6f"
)



## === cell 17
Xy_train_feats



## === cell 18
valid_rows_bool = (
    ((train_meta.eeg_sub_id < 21) & (train_meta.eeg_sub_id % 7 == 4))
    | (train_meta.eeg_sub_id == 0)
) & (train_meta.eeg_id % 4 > 1)

Xy_valid_meta = (train_meta[valid_rows_bool])[::47].copy()
Xy_valid_meta = Xy_valid_meta.reset_index().drop(columns=["index"])
print("Number of samples used for Validation =", len(Xy_valid_meta))

Xy_valid_feats = assemble_features(
    Xy_valid_meta, traintest="train", smooth_width=SMOOTH_WIDTH, SHOW_PLOT=SHOW_PLOTS
)
Xy_valid_meta.to_csv("Xy_valid_meta.csv", header=True, index=False, float_format="%.6f")
Xy_valid_feats.to_csv(
    "Xv_valid_feats.csv", header=True, index=False, float_format="%.6f"
)



## === cell 19
Xy_valid_feats



## === cell 20
X = Xy_train_feats.drop(columns=["clust_id"]).copy()
X.columns = X.columns.astype(str)
y = Xy_train_feats.clust_id

ave_oob = []
nfits = 5
for ifit in range(nfits):
    rfmodel = RandomForestClassifier(
        n_estimators=100,
        max_leaf_nodes=int(4 * NUM_CLUSTS),
        max_features=0.5,
        max_samples=0.7,
        oob_score=True,
        class_weight="balanced_subsample",
        n_jobs=-1,
        verbose=0,
        random_state=SEED + ifit,
    ).fit(X, y)
    ave_oob.append(rfmodel.oob_score_)
print(
    "RF model ave OOB score = {:.1f}% +/- {:.1f}".format(
        100 * np.mean(ave_oob), 100 * np.std(ave_oob)
    )
)

print("\nRF model score for X,y = {:.1f}%".format(100 * rfmodel.score(X, y)))



## === cell 21
if SHOW_PLOTS:
    sort_inds = rfmodel.feature_importances_.argsort()
    plt.figure(figsize=(4, 5))
    plt.barh(
        rfmodel.feature_names_in_[sort_inds], rfmodel.feature_importances_[sort_inds]
    )
    plt.ylim(int(2 * len(sort_inds) / 3), len(sort_inds) + 0.2)
    plt.title("Feature Importances (top third)")
    plt.show()



## === cell 22
X_train_infer = Xy_train_feats.drop(columns=["clust_id"]).copy()
X_train_infer.columns = X_train_infer.columns.astype(str)
X_train_infer = X_train_infer.reindex(columns=rfmodel.feature_names_in_, fill_value=0.0)

Xy_train_meta["pred_id"] = rfmodel.predict(X_train_infer)

if SHOW_PLOTS:
    maxprobs_train = np.max(rfmodel.predict_proba(X), axis=1)
    plt.figure(figsize=(6, 2))
    plt.hist(maxprobs_train, bins=20)
    plt.xlim(0.0, 1.0)
    plt.title("Histogram of max(proba) for model-training samples")
    plt.show()

solution = Xy_train_meta[["eeg_id"] + HBA_votes].copy()
for col_pre in HBA_names:
    solution[col_pre + "_vote"] = Xy_train_meta[col_pre + "_prob"].to_numpy()

submission = solution.copy()

pred_ids = rfmodel.predict(X_train_infer)
pred_ids = rfmodel.classes_.searchsorted(pred_ids)



## === cell 23
tamed_centers = clust_centers.copy()
mean_all_probs = np.array([0.208319, 0.132120, 0.128532, 0.138913, 0.179294, 0.212822])

best_fracs = 1.0 * np.ones(NUM_CLUSTS)
best_centers = clust_centers.copy()
last_kl = 10.0

for this_frac in np.arange(0.0, 1.0, 0.05):
    tamed_fracs = this_frac * np.ones(NUM_CLUSTS)
    for icent in range(NUM_CLUSTS):
        this_cent = clust_centers[icent, :]
        this_cent = (
            tamed_fracs[icent] * this_cent + (1.0 - tamed_fracs[icent]) * mean_all_probs
        )
        tamed_centers[icent, :] = this_cent

    tamed_centers = _sanitize_probs(tamed_centers, prob_floor=PROB_FLOOR)

    for iprob in range(HBA_number):
        this_col_probs = tamed_centers[:, iprob]
        submission[HBA_votes[iprob]] = this_col_probs[pred_ids]

    submission[HBA_votes] = _sanitize_probs(
        submission[HBA_votes].to_numpy(dtype=np.float64), prob_floor=PROB_FLOOR
    )

    this_kl = kld_score(
        solution[HBA_votes], submission[HBA_votes], prob_floor=PROB_FLOOR
    )
    if this_kl < last_kl:
        best_fracs = tamed_fracs.copy()
        best_centers = tamed_centers.copy()
        last_kl = this_kl
    else:
        break

best_centers = _sanitize_probs(best_centers, prob_floor=PROB_FLOOR)

for iprob in range(HBA_number):
    this_col_probs = best_centers[:, iprob]
    submission[HBA_votes[iprob]] = this_col_probs[pred_ids]

submission[HBA_votes] = _sanitize_probs(
    submission[HBA_votes].to_numpy(dtype=np.float64), prob_floor=PROB_FLOOR
)

this_kl = kld_score(solution[HBA_votes], submission[HBA_votes], prob_floor=PROB_FLOOR)
print("Tamed fractions:\n", best_fracs, "\nand centers:\n", best_centers)
print("\nKL from tamed centers (train-subsample): {:.4f}".format(this_kl))



## === cell 24
X_valid_infer = Xy_valid_feats.drop(columns=["clust_id"]).copy()
X_valid_infer.columns = X_valid_infer.columns.astype(str)
X_valid_infer = X_valid_infer.reindex(columns=rfmodel.feature_names_in_, fill_value=0.0)

Xy_valid_meta["pred_id"] = rfmodel.predict(X_valid_infer)

if SHOW_PLOTS:
    maxprobs_valid = np.max(rfmodel.predict_proba(X_valid_infer), axis=1)
    plt.figure(figsize=(6, 2))
    plt.hist(maxprobs_valid, bins=20)
    plt.xlim(0.0, 1.0)
    plt.title("Histogram of max(proba) for validation samples")
    plt.show()

solution = Xy_valid_meta[["eeg_id"] + HBA_votes].copy()
for col_pre in HBA_names:
    solution[col_pre + "_vote"] = Xy_valid_meta[col_pre + "_prob"].to_numpy()

submission = solution.copy()

pred_ids = rfmodel.predict(X_valid_infer)
pred_ids = rfmodel.classes_.searchsorted(pred_ids)



## === cell 25
for iprob in range(HBA_number):
    this_col_probs = best_centers[:, iprob]
    submission[HBA_votes[iprob]] = this_col_probs[pred_ids]

submission[HBA_votes] = _sanitize_probs(
    submission[HBA_votes].to_numpy(dtype=np.float64), prob_floor=PROB_FLOOR
)

this_kl = kld_score(solution[HBA_votes], submission[HBA_votes], prob_floor=PROB_FLOOR)
print("\nValidation KL using best_centers from train-taming: {:.4f}".format(this_kl))



## === cell 26
best_centers = _sanitize_probs(best_centers, prob_floor=PROB_FLOOR)

sample_sub = pd.read_csv(above_dir + "sample_submission.csv")

Xy_test_feats = assemble_features(
    test_meta, traintest="test", smooth_width=SMOOTH_WIDTH, SHOW_PLOT=False
)

X_test_infer = Xy_test_feats.copy()
X_test_infer.columns = X_test_infer.columns.astype(str)
X_test_infer = X_test_infer.reindex(columns=rfmodel.feature_names_in_, fill_value=0.0)

pred_ids = rfmodel.predict(X_test_infer)
pred_ids = rfmodel.classes_.searchsorted(pred_ids)

test_submit = sample_sub[["eeg_id"]].copy()

for new_col in HBA_votes:
    test_submit[new_col] = 1 / HBA_number

for iprob in range(HBA_number):
    this_col_probs = best_centers[:, iprob]
    test_submit[HBA_votes[iprob]] = this_col_probs[pred_ids]

probs = _sanitize_probs(
    test_submit[HBA_votes].to_numpy(dtype=np.float64), prob_floor=PROB_FLOOR
).astype(np.float64)
test_submit[HBA_votes] = probs

left = sample_sub[["eeg_id"]].drop_duplicates(subset=["eeg_id"], keep="first")
right = test_submit.drop_duplicates(subset=["eeg_id"], keep="first")
test_submit = left.merge(right, on="eeg_id", how="left")

for c in HBA_votes:
    if c not in test_submit.columns:
        test_submit[c] = 1.0 / HBA_number
test_submit[HBA_votes] = test_submit[HBA_votes].fillna(1.0 / HBA_number)

test_submit[HBA_votes] = _sanitize_probs(
    test_submit[HBA_votes].to_numpy(dtype=np.float64), prob_floor=PROB_FLOOR
)

full = sample_sub[["eeg_id"]].merge(
    test_submit, on="eeg_id", how="left", suffixes=("", "_pred")
)
for c in HBA_votes:
    if c not in full.columns:
        full[c] = 1.0 / HBA_number
full[HBA_votes] = full[HBA_votes].fillna(1.0 / HBA_number)

full[HBA_votes] = _sanitize_probs(
    full[HBA_votes].to_numpy(dtype=np.float64), prob_floor=PROB_FLOOR
)

full = full.reindex(columns=sample_sub.columns)
full["eeg_id"] = full["eeg_id"].astype(sample_sub["eeg_id"].dtype, copy=False)

row_sums = full[HBA_votes].sum(axis=1).to_numpy()
assert len(full) == len(
    sample_sub
), "Submission rowcount mismatch vs sample_submission.csv"
assert list(full.columns) == list(
    sample_sub.columns
), "Submission columns mismatch vs sample_submission.csv"
assert np.isfinite(full[HBA_votes].to_numpy()).all(), "Non-finite probabilities present"
assert np.max(np.abs(row_sums - 1.0)) < 1e-6, "Row probabilities do not sum to 1"

print(full.head())
print(
    "Row-sum check (min/max):",
    full[HBA_votes].sum(axis=1).min(),
    full[HBA_votes].sum(axis=1).max(),
)
print("Any NaNs in submission?:", full.isna().any().any())

full.to_csv("submission.csv", header=True, index=False, na_rep="", float_format="%.6f")
print("Wrote submission.csv with shape:", full.shape)
print("Submission columns:", list(full.columns))
