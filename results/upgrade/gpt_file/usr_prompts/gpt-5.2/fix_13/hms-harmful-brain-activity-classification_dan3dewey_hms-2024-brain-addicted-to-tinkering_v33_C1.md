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

1.033695

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.14745) has done: 'Your code didn’t yield a Kaggle score mainly because it is unlikely to finish within runtime limits due to heavy plotting and repeated full-parquet-to-pandas loads during feature assembly; that prevents reliably producing `submission.csv`. I keep the exact modeling logic (KMeans on vote-prob vectors + RandomForest on spectrogram-derived features + “tamed” cluster centers + soft cluster-proba -> class-proba) but make minimal performance fixes: disable expensive plotting by default, read only the needed spectrogram rows/columns with `pyarrow.parquet.read_table` (instead of loading whole parquet files), and cache loaded spectrograms. I also add a small safety fix to `calc_entropy` so it always selects the correct probability columns by name (avoids fragile positional slicing), and enforce correct RF class/proba alignment when combining with cluster centers (prevents wrong-column probability mixing which can silently hurt KL). These changes are directly score-relevant because they ensure a valid submission is produced and prevent misaligned probability vectors that inflate KL.'

# 9. Code solution

## === cell 0
NUM_CLUSTS = 6  # 6 to 10
SMOOTH_WIDTH = 5  # Odd>1: 3,5,7,9,...

above_dir = "../input/hms-harmful-brain-activity-classification/"

DO_PLOTS = False

RANDOM_STATE = 42



## === cell 1
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
def kld_score(solution, submission, eps=1e-15):
    """
    Score-relevant: robust KL divergence for probability vectors (train/valid checks).
    """
    cols = [c for c in solution.columns if c != "eeg_id"]
    p = solution[cols].to_numpy(dtype=np.float64)
    q = submission[cols].to_numpy(dtype=np.float64)

    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)

    kl = np.sum(p * (np.log(p) - np.log(q)), axis=1)
    return float(np.mean(kl))




## === cell 4
def read_hms_meta():
    """
    Read train/test metadata and compute per-row probability targets for clustering.
    """
    test_meta = pd.read_csv(above_dir + "test.csv")
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

    if DO_PLOTS:
        print("\nHistogram of the total votes")
        plt.figure(figsize=(6, 3))
        plt.hist(train_meta["total_vote"], bins=55, log=True)
        plt.title("Histogram of Total Votes")
        plt.show()

    print("\nComputing the probabilites of the different HBAs ...")
    for col_pre in HBA_names:
        train_meta[col_pre + "_prob"] = (
            train_meta[col_pre + "_vote"] / train_meta["total_vote"]
        )

    print("Calculating voting entropy values ...")

    def calc_entropy(row):
        the_probs = np.clip(row[HBA_probs].values.astype(float), 1.0e-8, 1.0)
        return float(np.nansum(the_probs * -1.0 * np.log(the_probs)))

    train_meta["entropy"] = train_meta.apply(calc_entropy, axis=1)

    return train_meta, test_meta




## === cell 5
def prob_prob_scatter(name1, name2, probs2plot, clust_ids, iclust_order=[0]):
    """
    Plot helper (kept; gated by DO_PLOTS).
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

    clstclrs = []
    for ilab in clust_ids:
        clstclrs.append(kmclrs[ilab])

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
    return kmclrs  # returns cluster colors appropriate for iclust




## === cell 6
train_meta, test_meta = read_hms_meta()



## === cell 7
if DO_PLOTS:
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
) & (
    train_meta.eeg_id % 4 < 2
)  # include odd and even eeg_ids



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
if DO_PLOTS:
    plt.figure(figsize=(6, 3))
    plt.hist(train_meta.loc[clust_rows_bool, "total_vote"], bins=100, log=True)
    plt.title("Histogram of total_vote in HBA samples clustered")
    plt.show()

prob_array = np.array(prob_vectors)
kmeans = KMeans(
    n_clusters=num_clusts,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=RANDOM_STATE,
)
kmeans.fit(prob_array)

clust_centers = kmeans.cluster_centers_

for iclust in range(num_clusts):
    clust_centers[iclust, :] = clust_centers[iclust, :] / np.sum(
        clust_centers[iclust, :]
    )

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

if DO_PLOTS:
    kmclrs = prob_prob_scatter("Seizure", "GPD", all_probs, all_ids, iclust_of_order)
    kmclrs = prob_prob_scatter("LPD", "GRDA", all_probs, all_ids, iclust_of_order)
    kmclrs = prob_prob_scatter("LRDA", "Other", all_probs, all_ids, iclust_of_order)

    clust_counts = train_meta.clust_id.value_counts()

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



## === cell 11
solution_train = pd.DataFrame({"eeg_id": train_meta["eeg_id"].values})
for col_pre in HBA_names:
    solution_train[col_pre + "_vote"] = train_meta[col_pre + "_prob"].values

submission_train = solution_train.copy()
clust_ids = train_meta["clust_id"].to_numpy()
if False:
    clust_ids = np.random.choice(9, size=len(train_meta), replace=True, p=None)
for iprob in range(HBA_number):
    this_col_probs = clust_centers[:, iprob]
    submission_train[HBA_votes[iprob]] = this_col_probs[clust_ids]

print(
    "Score if HBA samples are correctly assigned cluster prob.s:",
    np.round(kld_score(solution_train, submission_train), 4),
)



## === cell 12
smooth_width = SMOOTH_WIDTH  # Here smooth_width is just for looking,
spectro_meta = train_meta[clust_rows_bool].copy()
every_nth = 37



## === cell 13
freqs = np.array(range(100)) * 0.19525 + 0.59
spect_trend = 150.0 / (1.0**2.3 + freqs ** (2.3))



## === cell 14
if DO_PLOTS:
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
        if True:
            spectro_id_str = str(this_row.spectrogram_id)
            spectro_file = (
                above_dir + "train_spectrograms/" + spectro_id_str + ".parquet"
            )
            pads_spectro = pads.dataset(spectro_file)
            this_spectro = pads_spectro.to_table().to_pandas()
            loc_offset = int(this_row.spectrogram_label_offset_seconds / 2)
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
                middle4spre = middle4s - spect_mean
                middle4s = middle4spre.rolling(
                    smooth_width, min_periods=smooth_width, center=True, closed=None
                ).mean()
                for ibin in range(int((smooth_width - 1) / 2)):  # assumes width is odd
                    middle4s[ibin] = middle4spre[ibin]
                plt.plot((freqs), middle4s, c=kmclrs[this_clust], lw=3, alpha=0.01)
        if (ilocrow / every_nth + 1) % 100 == 0:
            print("... {} done...".format(int(ilocrow / every_nth) + 1))

    plt.plot([0.0, 20.0], [0.0, 0.0], c="black", lw=3, alpha=0.2)
    downsel_freqs = np.insert(
        freqs[int((smooth_width - 1) / 2) : 100 : smooth_width], 0, freqs[0]
    )
    plt.plot(downsel_freqs, len(downsel_freqs) * [0.0], ".k")
    plt.ylim(-1.0, 1.0)  # log already taken
    plt.xlim(0.0, 20.5)
    plt.title(
        "Middle-4s Spectra (color-coded by the {} clusters, smooth={})".format(
            num_clusts, smooth_width
        )
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("log10[] Amplitude (/ref-spectrum, /mean, and smoothed n=3) ]")
    plt.savefig("middle4s_spectra.png")
    plt.show()



## === cell 15
if DO_PLOTS:
    print("\nMedian of the Means(below): {:.4f}".format(np.median(all_means)))
    plt.figure(figsize=(6, 2))
    plt.hist(np.clip(all_means, -2.0, 2.0), bins=100)
    plt.xlim(-2.05, 2.05)
    plt.xlabel("Mean")
    plt.title("Histogram of the Means of the log(Spectra/Ref-spect)")
    plt.show()

    print("\nMedian of the Medians(below): {:.4f}".format(np.median(all_medians)))
    plt.figure(figsize=(6, 2))
    plt.hist(np.clip(all_medians, -2.0, 2.0), bins=100)
    plt.xlim(-2.05, 2.05)
    plt.xlabel("log10( Median )")
    plt.title("Histogram of the Medians of the log(Spectra/Ref-spect)")
    plt.show()

    mmclrs = []
    for ilab in all_clusts:
        mmclrs.append(kmclrs[ilab])
    plt.figure(figsize=(3, 3))
    plt.scatter(all_medians, all_means, s=2, c=mmclrs, alpha=0.3)
    plt.xlim(-1.0, 1)
    plt.ylim(-1.0, 1)
    plt.xlabel("Median")
    plt.ylabel("Mean")
    plt.show()



## === cell 16
_PARQUETFILE_CACHE = {}
_PARQUETFILE_CACHE_ORDER = []
_MAX_CACHED_PF = 256

_PF_STARTS_CACHE = {}


def _clear_spectro_cache():
    _PARQUETFILE_CACHE.clear()
    _PARQUETFILE_CACHE_ORDER.clear()
    _PF_STARTS_CACHE.clear()


def _get_pf(parquet_path):
    pf = _PARQUETFILE_CACHE.get(parquet_path)
    if pf is not None:
        return pf
    pf = pq.ParquetFile(parquet_path)
    _PARQUETFILE_CACHE[parquet_path] = pf
    _PARQUETFILE_CACHE_ORDER.append(parquet_path)
    if len(_PARQUETFILE_CACHE_ORDER) > _MAX_CACHED_PF:
        old = _PARQUETFILE_CACHE_ORDER.pop(0)
        _PARQUETFILE_CACHE.pop(old, None)
        _PF_STARTS_CACHE.pop(old, None)
    return pf


def _rowgroup_starts_cached(pf, parquet_path):
    starts = _PF_STARTS_CACHE.get(parquet_path)
    if starts is not None:
        return starts
    meta = pf.metadata
    starts = [0]
    acc = 0
    for i in range(meta.num_row_groups):
        acc += meta.row_group(i).num_rows
        starts.append(acc)
    _PF_STARTS_CACHE[parquet_path] = starts
    return starts  # len = num_row_groups+1, cumulative ends


def _find_rowgroup(starts, row):
    for i in range(len(starts) - 1):
        if row < starts[i + 1]:
            return i
    return len(starts) - 2


def _table_to_numpy_float64(tbl: pyarrow.Table) -> np.ndarray:
    """
    Bugfix: pyarrow.Table doesn't have .to_numpy() in this environment.
    Convert via pandas without changing numeric values/semantics.
    """
    df = tbl.to_pandas(split_blocks=True, self_destruct=True)
    return df.to_numpy(dtype=np.float64, copy=False)


def _read_spectro_two_rows_400cols(parquet_path, r1, r2):
    """
    Runtime-critical: fetch ONLY two rows and ONLY 400 feature columns using
    Parquet row-group reads + Arrow take, avoiding whole-file pandas conversion.
    Bugfix: replace Table.to_numpy (not available) with safe conversion.
    """
    r1 = int(r1)
    r2 = int(r2)
    lo_req = min(r1, r2)
    hi_req = max(r1, r2)

    if not os.path.exists(parquet_path):
        return None, None

    pf = _get_pf(parquet_path)
    nrows = pf.metadata.num_rows
    if nrows <= 0:
        return None, None

    lo = int(np.clip(lo_req, 0, nrows - 1))
    hi = int(np.clip(hi_req, 0, nrows - 1))

    all_cols = pf.schema_arrow.names
    if len(all_cols) < 401:
        return None, None
    use_cols = all_cols[1:401]  # 400 feature columns

    starts = _rowgroup_starts_cached(pf, parquet_path)
    rg_lo = _find_rowgroup(starts, lo)
    rg_hi = _find_rowgroup(starts, hi)

    tbl = pf.read_row_groups(row_groups=list(range(rg_lo, rg_hi + 1)), columns=use_cols)
    base = starts[rg_lo]
    i_lo = int(lo - base)
    i_hi = int(hi - base)

    take = sorted({i_lo, i_hi})
    tbl2 = tbl.take(pyarrow.array(take))

    arr2 = _table_to_numpy_float64(tbl2)
    a0 = arr2[0]
    a1 = arr2[1] if arr2.shape[0] > 1 else arr2[0]

    if i_lo == i_hi:
        row_lo = row_hi = a0
    else:
        row_lo = a0 if take[0] == i_lo else a1
        row_hi = a0 if take[0] == i_hi else a1

    if r1 <= r2:
        return row_lo, row_hi
    else:
        return row_hi, row_lo


def assemble_features(meta_frame, traintest="train", smooth_width=5, SHOW_PLOT=True):
    """
    Create a dataframe of spectrogram features from the meta_frame rows.
    Will include clust_id (i.e, the y) if it is in the input meta_frame.
    Assumes these are available: above_dir, num_clusts
    """
    freqs = np.array(range(100)) * 0.19525 + 0.59
    spect_trend = 150.0 / (1.0**2.3 + freqs ** (2.3))
    freqs[0] = 0.0  # separate the (unsmoothed) first bin from others
    freqs4 = np.array(4 * list(freqs))
    spect_trend4 = np.array(4 * list(spect_trend))

    if SHOW_PLOT and DO_PLOTS:
        plt.figure(figsize=(10, 8))

    feats_rows = []
    for irow in meta_frame.index:
        this_row = meta_frame.loc[irow]
        spectro_id_str = str(int(this_row.spectrogram_id))  # make sure int

        loc_offset = int(this_row.spectrogram_label_offset_seconds / 2)
        r1 = loc_offset + 149
        r2 = loc_offset + 150

        spectro_file = (
            above_dir + traintest + "_spectrograms/" + spectro_id_str + ".parquet"
        )
        row1, row2 = _read_spectro_two_rows_400cols(spectro_file, r1, r2)

        if row1 is None or row2 is None:
            middle4s = np.ones(400, dtype=np.float64)
        else:
            middle4s = (row1 + row2) / (2.0 * spect_trend4)

        middle4s = np.clip(middle4s, 0.001, 1000.0)
        middle4s = np.nan_to_num(middle4s, nan=0.001, posinf=0.001, neginf=0.001)

        spect_mean = float(np.mean(middle4s))  # over all 4 spectra
        spect_median = float(np.median(middle4s))

        the4means = []
        the4medians = []
        for ispec in range(4):
            ibeg = ([0, 100, 200, 300])[ispec]
            iend = ibeg + 100
            the4means.append(float(np.mean(middle4s[ibeg:iend])))
            the4medians.append(float(np.median(middle4s[ibeg:iend])))

        middle4spre = np.log10(middle4s / spect_mean)

        w = smooth_width
        kernel = np.ones(w, dtype=np.float64) / w
        middle4s_sm = middle4spre.copy()
        for band in range(4):
            seg = middle4spre[band * 100 : (band + 1) * 100]
            sm = np.convolve(seg, kernel, mode="same")
            middle4s_sm[band * 100 : (band + 1) * 100] = sm
            for ibin in range(int((w - 1) / 2)):
                middle4s_sm[band * 100 + ibin] = seg[ibin]
                middle4s_sm[band * 100 + (99 - ibin)] = seg[99 - ibin]

        baseinds = np.insert(
            np.arange(int((smooth_width - 1) / 2), 100, smooth_width), 0, 0
        )
        select_inds = np.concatenate(
            (baseinds, 100 + baseinds, 200 + baseinds, 300 + baseinds)
        )
        freqs4ds = freqs4[select_inds]
        middle4sds = middle4s_sm[select_inds]

        row_dict = {f"f_{i}": float(middle4sds[i]) for i in range(len(middle4sds))}
        row_dict["Mean"] = float(np.log10(spect_mean))
        row_dict["Median"] = float(np.log10(spect_median))
        the4means = np.log10(np.array(the4means, dtype=np.float64))
        the4medians = np.log10(np.array(the4medians, dtype=np.float64))
        for ispec in range(4):
            row_dict[the4chains[ispec] + "mean"] = float(the4means[ispec])
            row_dict[the4chains[ispec] + "median"] = float(the4medians[ispec])

        if "clust_id" in meta_frame.columns:
            row_dict["clust_id"] = int(this_row.clust_id)

        feats_rows.append(row_dict)

        if len(feats_rows) % 400 == 0:
            print("... {} done...".format(len(feats_rows)))

        if SHOW_PLOT and DO_PLOTS:
            this_clr = "blue"
            if "clust_id" in meta_frame.columns:
                this_clust = int(this_row.clust_id)
                this_clr = kmclrs[this_clust]
                plt.title(
                    "Middle-4s Feature Values"
                    + " (color-coded by the {} clusters, smooth={})".format(
                        num_clusts, smooth_width
                    )
                )
            the_alpha = np.clip(0.05 * 350 / len(meta_frame), 0.003, 0.5)
            plt.plot(
                freqs4ds + smooth_width * 0.15 * (np.random.rand(len(freqs4ds)) - 0.5),
                middle4sds,
                ".",
                c=this_clr,
                markersize=10,
                alpha=the_alpha,
            )
            plt.plot(
                -1.2 + smooth_width * 0.15 * (np.random.rand(4) - 0.5),
                the4means + (0.0 * (np.random.rand(4) - 0.5)),
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

    feats_frame = pd.DataFrame(feats_rows)
    if SHOW_PLOT and DO_PLOTS:
        plt.savefig("middle4s_features.png")
        plt.show()
    return feats_frame.reset_index(drop=True)




## === cell 17
_clear_spectro_cache()

Xy_train_meta = (train_meta[clust_rows_bool])[::23].copy()
Xy_train_meta = Xy_train_meta.reset_index().drop(columns=["index"])
print("Number of samples used for training =", len(Xy_train_meta))

Xy_train_feats = assemble_features(
    Xy_train_meta, traintest="train", smooth_width=SMOOTH_WIDTH, SHOW_PLOT=False
)
Xy_train_meta.to_csv("Xy_train_meta.csv", header=True, index=False, float_format="%.6f")
Xy_train_feats.to_csv(
    "Xv_train_feats.csv", header=True, index=False, float_format="%.6f"
)



## === cell 18
Xy_train_feats



## === cell 19
valid_rows_bool = (
    ((train_meta.eeg_sub_id < 21) & (train_meta.eeg_sub_id % 7 == 4))
    | (train_meta.eeg_sub_id == 0)
) & (
    train_meta.eeg_id % 4 > 1
)  # include odd and even eeg_ids

_clear_spectro_cache()

Xy_valid_meta = (train_meta[valid_rows_bool])[::47].copy()
Xy_valid_meta = Xy_valid_meta.reset_index().drop(columns=["index"])
print("Number of samples used for Validation =", len(Xy_valid_meta))

Xy_valid_feats = assemble_features(
    Xy_valid_meta, traintest="train", smooth_width=SMOOTH_WIDTH, SHOW_PLOT=False
)
Xy_valid_meta.to_csv("Xy_valid_meta.csv", header=True, index=False, float_format="%.6f")
Xy_valid_feats.to_csv(
    "Xv_valid_feats.csv", header=True, index=False, float_format="%.6f"
)



## === cell 20
Xy_valid_feats




## === cell 21
def probs_from_centers(center_matrix, cluster_proba, eps=1e-15):
    """
    For KL metric, use soft cluster probabilities.
    """
    pred = cluster_proba @ center_matrix
    pred = np.clip(pred, eps, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)
    return pred




## === cell 22
X = Xy_train_feats.drop(columns=["clust_id"])
y = Xy_train_feats.clust_id

ave_oob = []
nfits = 1
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
        random_state=RANDOM_STATE + ifit,
    ).fit(X, y)
    ave_oob.append(rfmodel.oob_score_)
print(
    "RF model ave OOB score = {:.1f}% +/- {:.1f}".format(
        100 * np.mean(ave_oob), 100 * np.std(ave_oob)
    )
)

print("\nRF model score for X,y = {:.1f}%".format(100 * rfmodel.score(X, y)))



## === cell 23
if DO_PLOTS:
    sort_inds = rfmodel.feature_importances_.argsort()
    plt.figure(figsize=(4, 5))
    plt.barh(
        rfmodel.feature_names_in_[sort_inds], rfmodel.feature_importances_[sort_inds]
    )
    plt.ylim(int(2 * len(sort_inds) / 3), len(sort_inds) + 0.2)
    plt.title("Feature Importances (top third)")
    plt.show()




## === cell 24
def rf_predict_proba_full(rf, X, n_classes):
    proba = rf.predict_proba(X)
    full = np.zeros((len(X), n_classes), dtype=np.float64)
    for j, cls in enumerate(rf.classes_):
        full[:, int(cls)] = proba[:, j]
    row_sums = full.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    full = full / row_sums
    return full


Xy_train_meta["pred_id"] = rfmodel.predict(Xy_train_feats.drop(columns=["clust_id"]))

if DO_PLOTS:
    maxprobs_train = np.max(rfmodel.predict_proba(X), axis=1)
    plt.figure(figsize=(6, 2))
    plt.hist(maxprobs_train, bins=20)
    plt.xlim(0.0, 1.0)
    plt.title("Histogram of max(proba) for model-training samples")
    plt.show()

solution = pd.DataFrame({"eeg_id": Xy_train_meta["eeg_id"].values})
for col_pre in HBA_names:
    solution[col_pre + "_vote"] = Xy_train_meta[col_pre + "_prob"].values

submission = solution.copy()

pred_ids = Xy_train_meta["pred_id"]
pred_proba = rf_predict_proba_full(rfmodel, X, NUM_CLUSTS)




## === cell 25
def find_best_tamed_kl(pred_proba, solution, submission, mean_all_probs):
    """
    Optimize taming using soft cluster probabilities (better aligned with KL).
    """
    mean_all_probs = np.asarray(mean_all_probs, dtype=np.float64)
    mean_all_probs = mean_all_probs / np.sum(mean_all_probs)

    best_fracs = 0.0 * np.ones(NUM_CLUSTS, dtype=np.float64)

    tamed_centers = clust_centers.copy().astype(np.float64)
    for iclust in range(NUM_CLUSTS):
        tamed_centers[iclust, :] = mean_all_probs

    tmp_pred = probs_from_centers(tamed_centers, pred_proba)
    for iprob in range(HBA_number):
        submission[HBA_votes[iprob]] = tmp_pred[:, iprob]
    best_kl_global = kld_score(solution, submission)

    for iclust in range(NUM_CLUSTS):
        last_kl = np.inf
        local_best_frac = best_fracs[iclust]
        local_best_centers = tamed_centers.copy()

        for this_frac in np.arange(0.0, 1.0 + 1e-9, 0.05):
            tamed_centers_trial = tamed_centers.copy()
            this_cent = (
                this_frac * clust_centers[iclust, :]
                + (1.0 - this_frac) * mean_all_probs
            )
            this_cent = this_cent / np.sum(this_cent)
            tamed_centers_trial[iclust, :] = this_cent

            tmp_pred = probs_from_centers(tamed_centers_trial, pred_proba)
            for iprob in range(HBA_number):
                submission[HBA_votes[iprob]] = tmp_pred[:, iprob]
            this_kl = kld_score(solution, submission)

            if this_kl <= last_kl + 1e-12:
                last_kl = this_kl
                local_best_frac = this_frac
                local_best_centers = tamed_centers_trial.copy()
            else:
                break

        tamed_centers = local_best_centers.copy()
        best_fracs[iclust] = local_best_frac

    best_centers = tamed_centers.copy()
    tmp_pred = probs_from_centers(best_centers, pred_proba)
    for iprob in range(HBA_number):
        submission[HBA_votes[iprob]] = tmp_pred[:, iprob]
    this_kl = kld_score(solution, submission)
    return best_fracs, best_centers, this_kl


mean_all_probs_train = train_meta[HBA_probs].mean(axis=0).to_numpy(dtype=np.float64)

best_fracs, best_centers, this_kl = find_best_tamed_kl(
    pred_proba, solution, submission, mean_all_probs=mean_all_probs_train
)
print("Tamed fractions:\n", best_fracs, "\nand centers:\n", best_centers)
print("\nKL from tamed centers (soft cluster proba): {:.4f}".format(this_kl))



## === cell 26
Xy_valid_meta["pred_id"] = rfmodel.predict(Xy_valid_feats.drop(columns=["clust_id"]))

if DO_PLOTS:
    maxprobs_valid = np.max(
        rfmodel.predict_proba(Xy_valid_feats.drop(columns=["clust_id"])), axis=1
    )
    plt.figure(figsize=(6, 2))
    plt.hist(maxprobs_valid, bins=20)
    plt.xlim(0.0, 1.0)
    plt.title("Histogram of max(proba) for validation samples")
    plt.show()

solution = pd.DataFrame({"eeg_id": Xy_valid_meta["eeg_id"].values})
for col_pre in HBA_names:
    solution[col_pre + "_vote"] = Xy_valid_meta[col_pre + "_prob"].values

submission = solution.copy()

pred_ids = Xy_valid_meta["pred_id"]
pred_proba = rf_predict_proba_full(
    rfmodel, Xy_valid_feats.drop(columns=["clust_id"]), NUM_CLUSTS
)

best_fracs, best_centers, this_kl = find_best_tamed_kl(
    pred_proba, solution, submission, mean_all_probs=mean_all_probs_train
)
print("Tamed fractions:\n", best_fracs, "\nand centers:\n", best_centers)
print("\nKL from tamed centers (soft cluster proba): {:.4f}".format(this_kl))



## === cell 27
_clear_spectro_cache()

Xy_test_feats = assemble_features(
    test_meta, traintest="test", smooth_width=SMOOTH_WIDTH, SHOW_PLOT=False
)

test_cluster_proba = rf_predict_proba_full(rfmodel, Xy_test_feats, NUM_CLUSTS)

test_probs = probs_from_centers(best_centers, test_cluster_proba)

test_submit = test_meta[["eeg_id"]].copy()
for iprob in range(HBA_number):
    test_submit[HBA_votes[iprob]] = test_probs[:, iprob]

row_sums = test_submit[HBA_votes].sum(axis=1).values
test_submit[HBA_votes] = test_submit[HBA_votes].div(row_sums, axis=0)

sample_sub = pd.read_csv(above_dir + "sample_submission.csv")
test_submit = sample_sub[["eeg_id"]].merge(test_submit, on="eeg_id", how="left")
for c in HBA_votes:
    if c not in test_submit.columns:
        raise ValueError(f"Missing required submission column: {c}")
test_submit = test_submit[["eeg_id"] + HBA_votes]

test_submit[HBA_votes] = test_submit[HBA_votes].fillna(1.0 / HBA_number)
row_sums = test_submit[HBA_votes].sum(axis=1).values
test_submit[HBA_votes] = test_submit[HBA_votes].div(row_sums, axis=0)

print(test_submit.head())

test_submit.to_csv(
    "submission.csv", header=True, index=False, na_rep="", float_format="%.6f"
)
print("\nWrote submission.csv with shape:", test_submit.shape)
print(
    "Row-sum min/max:",
    test_submit[HBA_votes].sum(axis=1).min(),
    test_submit[HBA_votes].sum(axis=1).max(),
)
