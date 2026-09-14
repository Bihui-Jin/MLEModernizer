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

N/A

# 9. Code solution

## === cell 0
NUM_CLUSTS = 6  # 6 to 10
SMOOTH_WIDTH = 5  # Odd>1: 3,5,7,9,...

above_dir = "/kaggle/input/hms-harmful-brain-activity-classification/"



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
    sumsum = 0.0
    for prob_col in solution.columns.values:
        sumsum += np.nansum(
            -1.0
            * solution[prob_col]
            * np.log(submission[prob_col] / solution[prob_col])
        )
    return sumsum / (len(solution))




## === cell 4
def read_hms_meta():
    """
    Read in the train.csv and test.csv files.
    Add total_vote, _prob columns, and vote entropy to train_meta.
    Add extra cols to test to allow the same processing as train:
        eeg[spectro]_sub_id, eeg[spectro]_label_offset_seconds, label_id
    Make various plots of the train_meta values.
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

    print("\nHistogram of the total votes")
    plt.figure(figsize=(6, 3))
    plt.hist(train_meta["total_vote"], bins=55, log=True)
    plt.title("Histogram of Total Votes")
    plt.show()

    allvt = train_meta.expert_consensus.value_counts()
    less9 = train_meta[train_meta["total_vote"] < 9].expert_consensus.value_counts()
    more9 = train_meta[train_meta["total_vote"] > 9].expert_consensus.value_counts()
    vnot3 = train_meta[train_meta["total_vote"] != 3].expert_consensus.value_counts()

    print("\nHistograms of the probabilites of the different HBAs:")
    print("   (note that the large Prob=0 bin is not included.)")
    for col_pre in HBA_names:
        train_meta[col_pre + "_prob"] = (
            train_meta[col_pre + "_vote"] / train_meta["total_vote"]
        )
        if False:
            plt.figure(figsize=(6, 1.5))
            plt.hist(train_meta[col_pre + "_prob"], bins=20, range=(0.02, 1))
            plt.ylim(0, len(train_meta) / 5)
            plt.title("Histogram of   " + col_pre + "_prob")
            plt.show()

    print("Calculating voting entropy values ...")

    def calc_entropy(row):
        the_probs = np.clip(row[16 : 21 + 1].values.astype(float), 1.0e-8, 1.0)
        return np.nansum(the_probs * -1 * np.log(the_probs))

    train_meta["entropy"] = train_meta.apply(calc_entropy, axis=1)

    return train_meta, test_meta




## === cell 5
def prob_prob_scatter(name1, name2, probs2plot, clust_ids, iclust_order=[0]):
    """
    Make a prob1 vs prob2 scatter plot.
    - name1, name2 are 2 of the 6 HBA expert_consensus labels.
    - probs2plot is 6-column dataframe, e.g., train_meta[HBA_probs]
    - clust_ids is an array of cluster id integers, e.g., train_meta["clust_id"]
    Include an x at the cluster centers in the chosen axes.
    Use sqrt scaling to emphasize lower values.
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
        kmclrs = hba_clrs

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
    return kmclrs  # returns cluster colors appropriate for iclust




## === cell 6
train_meta, test_meta = read_hms_meta()




## === cell 7
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

clust_rows_bool = ((train_meta.eeg_sub_id < 56) & (train_meta.eeg_sub_id % 7 == 6)) | (
    train_meta.eeg_sub_id == 0
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
plt.figure(figsize=(6, 3))
plt.hist(train_meta.loc[clust_rows_bool, "total_vote"], bins=100, log=True)
plt.title("Histogram of total_vote in HBA samples clustered")
plt.show()

prob_array = np.array(prob_vectors)
kmeans = KMeans(
    n_clusters=num_clusts, init="k-means++", n_init=10, max_iter=300, random_state=None
)
kmeans.fit(prob_array)

clust_centers = kmeans.cluster_centers_
clust_centers = clust_centers / clust_centers.sum(axis=1, keepdims=True)

for iclust in range(HBA_number):
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

if True:
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




## === cell 11
solution_train = train_meta[["eeg_id"] + HBA_votes]
for col_pre in HBA_names:
    solution_train.loc[:, col_pre + "_vote"] = train_meta[col_pre + "_prob"]

submission_train = solution_train.copy()
clust_ids = train_meta["clust_id"]
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
plt.figure(figsize=(10, 8))

all_medians = []
all_means = []
all_clusts = []
print("Plotting {} x 4 processed spectra".format(int(len(spectro_meta) / every_nth)))
for ilocrow in range(0, len(spectro_meta), every_nth):
    this_row = spectro_meta.iloc[ilocrow]
    this_clust = this_row.clust_id
    if True:
        spectro_id_str = str(this_row.spectrogram_id)
        spectro_file = above_dir + "train_spectrograms/" + spectro_id_str + ".parquet"
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
    if SHOW_PLOT:
        plt.figure(figsize=(10, 8))
    feats_frame = []
    last_spectro_id_str = "starting"
    for irow in meta_frame.index:
        this_row = meta_frame.loc[irow]
        spectro_id_str = str(int(this_row.spectrogram_id))  # make sure int
        if spectro_id_str != last_spectro_id_str:
            spectro_file = (
                above_dir + traintest + "_spectrograms/" + spectro_id_str + ".parquet"
            )
            pads_spectro = pads.dataset(spectro_file)
            this_spectro = pads_spectro.to_table().to_pandas()
        last_spectro_id_str = spectro_id_str
        loc_offset = int(this_row.spectrogram_label_offset_seconds / 2)
        middle4s = (
            this_spectro.iloc[loc_offset + 149, 1:]
            + this_spectro.iloc[loc_offset + 150, 1:]
        ) / (
            2.0 * spect_trend4
        )  # 4 copies of trend
        middle4s = np.clip(middle4s, 0.001, 1000.0)
        middle4s = middle4s.replace([np.nan, -np.inf, np.inf], 0.001)
        spect_mean = np.mean(middle4s)  # over all 4 spectra
        spect_median = np.median(middle4s)
        the4means = []
        the4medians = []
        for ispec in range(4):
            ibeg = ([0, 100, 200, 300])[ispec]
            iend = ibeg + 100
            the4means.append(np.mean(middle4s[ibeg:iend]))
            the4medians.append(np.median(middle4s[ibeg:iend]))
        middle4spre = np.log10(middle4s / spect_mean)
        middle4s = middle4spre.rolling(
            smooth_width, min_periods=smooth_width, center=True, closed=None
        ).mean()
        for ioff in range(0, 400, 100):
            for ibin in range(int((smooth_width - 1) / 2)):  # assumes width is odd
                middle4s[ibin + ioff] = middle4spre[ibin + ioff]
        baseinds = np.insert(
            np.arange(int((smooth_width - 1) / 2), 100, smooth_width), 0, 0
        )
        select_inds = np.concatenate(
            (baseinds, 100 + baseinds, 200 + baseinds, 300 + baseinds)
        )
        freqs4ds = freqs4[select_inds]
        middle4sds = middle4s[select_inds]
        these_feats = middle4sds.to_frame().T  # already log10()
        spect_mean = np.log10(spect_mean)
        spect_median = np.log10(spect_median)
        the4means = np.log10(the4means)
        the4medians = np.log10(the4medians)
        these_feats["Mean"] = spect_mean
        these_feats["Median"] = spect_median
        for ispec in range(4):
            these_feats[the4chains[ispec] + "mean"] = the4means[ispec]
            these_feats[the4chains[ispec] + "median"] = the4medians[ispec]
        if "clust_id" in meta_frame.columns:
            these_feats["clust_id"] = this_row.clust_id
        if len(feats_frame) == 0:
            feats_frame = these_feats.copy()
        else:
            feats_frame = pd.concat([feats_frame, these_feats])
        if len(feats_frame) % 100 == 0:
            print("... {} done...".format(len(feats_frame)))
        if SHOW_PLOT:
            this_clr = "blue"
            plt.title("Middle-4s Feature Values (smooth={})".format(smooth_width))
            if "clust_id" in meta_frame.columns:
                this_clust = this_row.clust_id
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
    if SHOW_PLOT:
        plt.savefig("middle4s_features.png")
        plt.show()
    return feats_frame.reset_index().drop(columns=["index"])




## === cell 17
Xy_train_meta = (train_meta[clust_rows_bool])[::23].copy()
Xy_train_meta = Xy_train_meta.reset_index().drop(columns=["index"])
print("Number of samples used for training =", len(Xy_train_meta))

Xy_train_feats = assemble_features(
    Xy_train_meta, traintest="train", smooth_width=SMOOTH_WIDTH, SHOW_PLOT=True
)
Xy_train_meta.to_csv("Xy_train_meta.csv", header=True, index=False, float_format="%.6f")
Xy_train_feats.to_csv(
    "Xv_train_feats.csv", header=True, index=False, float_format="%.6f"
)




## === cell 18
Xy_train_feats




## === cell 19
valid_rows_bool = ((train_meta.eeg_sub_id < 21) & (train_meta.eeg_sub_id % 7 == 4)) | (
    train_meta.eeg_sub_id == 0
) & (
    train_meta.eeg_id % 4 > 1
)  # include odd and even eeg_ids

Xy_valid_meta = (train_meta[valid_rows_bool])[::47].copy()
Xy_valid_meta = Xy_valid_meta.reset_index().drop(columns=["index"])
print("Number of samples used for Validation =", len(Xy_valid_meta))

Xy_valid_feats = assemble_features(
    Xy_valid_meta, traintest="train", smooth_width=SMOOTH_WIDTH, SHOW_PLOT=True
)
Xy_valid_meta.to_csv("Xy_valid_meta.csv", header=True, index=False, float_format="%.6f")
Xy_valid_feats.to_csv(
    "Xv_valid_feats.csv", header=True, index=False, float_format="%.6f"
)




## === cell 20
Xy_valid_feats




## === cell 21
X = Xy_train_feats.drop(columns=["clust_id"])
y = Xy_train_meta["clust_id"]

ave_oob = []
nfits = 5
for ifit in range(nfits):
    rfmodel = RandomForestClassifier(
        n_estimators=200,
        max_leaf_nodes=int(6 * NUM_CLUSTS),
        max_features=0.5,
        max_samples=0.7,
        oob_score=True,
        class_weight="balanced_subsample",
        n_jobs=-1,
        verbose=0,
        random_state=42,
    ).fit(X, y)
    ave_oob.append(rfmodel.oob_score_)
print(
    "RF model ave OOB score = {:.1f}% +/- {:.1f}".format(
        100 * np.mean(ave_oob), 100 * np.std(ave_oob)
    )
)

print("\nRF model score for X,y = {:.1f}%".format(100 * rfmodel.score(X, y)))




## === cell 22
sort_inds = rfmodel.feature_importances_.argsort()
plt.figure(figsize=(4, 5))
plt.barh(rfmodel.feature_names_in_[sort_inds], rfmodel.feature_importances_[sort_inds])
plt.ylim(int(2 * len(sort_inds) / 3), len(sort_inds) + 0.2)
plt.title("Feature Importances (top third)")
plt.show()




## === cell 23
pred_ids = rfmodel.predict(Xy_train_feats.drop(columns=["clust_id"]))

solution = Xy_train_meta[["eeg_id"] + HBA_votes].copy()
for col_pre in HBA_names:
    solution.loc[:, col_pre + "_vote"] = Xy_train_meta[col_pre + "_prob"]

submission = solution.copy()
for iprob in range(HBA_number):
    this_col_probs = clust_centers[:, iprob]
    submission[HBA_votes[iprob]] = this_col_probs[pred_ids]

print("Initial KL on training data:", np.round(kld_score(solution, submission), 4))




## === cell 24
def find_best_tamed_kl():
    """
    Adjust the taming fraction for each cluster center to optimize KL
    Assumed inputs in environment:
        submission, pred_ids, solution
    Assumed useful values available:
        clust_centers, NUM_CLUSTS, HBA_number, HBA_votes
    """
    mean_all_probs = np.array(
        [0.208319, 0.132120, 0.128532, 0.138913, 0.179294, 0.212822]
    )
    tamed_fracs = 0.0 * np.ones(NUM_CLUSTS)
    tamed_centers = clust_centers.copy()
    for iclust in range(NUM_CLUSTS):
        tamed_centers[iclust, :] = mean_all_probs
    for iclust in range(NUM_CLUSTS):
        last_kl = 10.0
        best_fracs = tamed_fracs.copy()
        best_centers = tamed_centers.copy()
        for this_frac in np.arange(0.0, 1.0, 0.05):
            tamed_fracs[iclust] = this_frac
            this_cent = (
                tamed_fracs[iclust] * clust_centers[iclust, :]
                + (1.0 - tamed_fracs[iclust]) * mean_all_probs
            )
            tamed_centers[iclust, :] = this_cent

            for iprob in range(HBA_number):
                this_col_probs = tamed_centers[:, iprob]
                submission[HBA_votes[iprob]] = this_col_probs[pred_ids]
            this_kl = kld_score(solution, submission)
            if this_kl < last_kl:
                best_fracs = tamed_fracs.copy()
                best_centers = tamed_centers.copy()
                last_kl = this_kl
            else:
                tamed_fracs[iclust] = best_fracs[iclust]
                tamed_centers[iclust, :] = best_centers[iclust, :]
                break
    return best_fracs, best_centers


best_fracs, best_centers = find_best_tamed_kl()

for iprob in range(HBA_number):
    this_col_probs = best_centers[:, iprob]
    submission[HBA_votes[iprob]] = this_col_probs[pred_ids]
this_kl = kld_score(solution, submission)
print("Tamed fractions:\n", best_fracs, "\nand centers:\n", best_centers)
print("\nKL from tamed centers: {:.4f}".format(this_kl))




## === cell 25
pred_ids_valid = rfmodel.predict(Xy_valid_feats.drop(columns=["clust_id"]))

solution_valid = Xy_valid_meta[["eeg_id"] + HBA_votes].copy()
for col_pre in HBA_names:
    solution_valid.loc[:, col_pre + "_vote"] = Xy_valid_meta[col_pre + "_prob"]

submission_valid = solution_valid.copy()
for iprob in range(HBA_number):
    this_col_probs = best_centers[:, iprob]
    submission_valid[HBA_votes[iprob]] = this_col_probs[pred_ids_valid]

print(
    "Validation KL after taming:",
    np.round(kld_score(solution_valid, submission_valid), 4),
)




## === cell 26
print("Best taming fractions (re‑print):", best_fracs)




## === cell 27
Xy_test_feats = assemble_features(
    test_meta, traintest="test", smooth_width=SMOOTH_WIDTH, SHOW_PLOT=False
)

pred_ids_test = rfmodel.predict(Xy_test_feats)

test_submit = test_meta[["eeg_id"]].copy()
for new_col in HBA_votes:
    test_submit[new_col] = 1.0 / HBA_number  # fallback uniform

for iprob in range(HBA_number):
    this_col_probs = best_centers[:, iprob]
    test_submit[HBA_votes[iprob]] = this_col_probs[pred_ids_test]

prob_sum = test_submit[HBA_votes].sum(axis=1)
test_submit[HBA_votes] = test_submit[HBA_votes].div(prob_sum, axis=0)

print(test_submit.head())

test_submit.to_csv(
    "submission.csv", header=True, index=False, na_rep="", float_format="%.6f"
)
