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

1.057977

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
NUM_CLUSTS = 10
SMOOTH_WIDTH = 5

above_dir = "../input/hms-harmful-brain-activity-classification/"



## === cell 1
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

import pyarrow
import pyarrow.dataset as pads



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

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)




## === cell 3
def _resolve_above_dir(preferred):
    preferred = preferred if preferred.endswith("/") else preferred + "/"
    candidates = [
        preferred,
        "/kaggle/input/hms-harmful-brain-activity-classification/",
        "/kaggle/input/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification/",
        "/kaggle/data/hms-harmful-brain-activity-classification/",
        "/kaggle/data/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification/",
        "/kaggle/data/input/hms-harmful-brain-activity-classification/",
        "/kaggle/data/input/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification/",
        "/kaggle/data/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification/",
        "/kaggle/data/input/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification/",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
            os.path.join(c, "test.csv")
        ):
            return c if c.endswith("/") else c + "/"

    for root in ["/kaggle/input", "/kaggle/data", "/kaggle/data/input", "/"]:
        if os.path.isdir(root):
            for dirpath, dirnames, filenames in os.walk(root):
                if "train.csv" in filenames and "test.csv" in filenames:
                    return dirpath if dirpath.endswith("/") else dirpath + "/"
    return preferred


above_dir = _resolve_above_dir(above_dir)
print("Using above_dir =", above_dir)




## === cell 4
def kld_score(solution, submission, eps=1e-15):
    """
    Calculate the average KL divergence score.
    Kaggle metric is KL(solution || submission). We make it numerically safe by clipping.
    """
    sumsum = 0.0
    for prob_col in solution.columns.values:
        p = np.clip(solution[prob_col].to_numpy(dtype=np.float64), eps, 1.0)
        q = np.clip(submission[prob_col].to_numpy(dtype=np.float64), eps, 1.0)
        sumsum += np.nansum(-1.0 * p * np.log(q / p))
    return sumsum / (len(solution))




## === cell 5
def read_hms_meta():
    """
    Read in the train.csv and test.csv files.
    Add total_vote, *_prob columns, and vote entropy to train_meta.
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
    if test_meta_len > 1:
        REAL_TEST = True
    else:
        REAL_TEST = False
        print("  --> not the real LB test data.\n")

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

    print("\nComputing *_prob columns from votes:")
    for col_pre in HBA_names:
        train_meta[col_pre + "_prob"] = (
            train_meta[col_pre + "_vote"] / train_meta["total_vote"]
        )

    print("Calculating voting entropy values ...")

    def calc_entropy(row):
        the_probs = np.clip(row[HBA_probs].values.astype(float), 1.0e-8, 1.0)
        return np.nansum(the_probs * -1 * np.log(the_probs))

    train_meta["entropy"] = train_meta.apply(calc_entropy, axis=1)

    return train_meta, test_meta




## === cell 6
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
        kmclrs = hba_clrs.copy()

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




## === cell 7
train_meta, test_meta = read_hms_meta()



## === cell 8
if False:
    plt.figure(figsize=(5, 2))
    plt.hist(train_meta["spectrogram_sub_id"], bins=55, log=True)
    plt.title("Histogram of spectrogram_sub_id")
    plt.show()

    plt.figure(figsize=(5, 2))
    plt.hist(train_meta["eeg_sub_id"], bins=55, log=True)
    plt.title("Histogram of eeg_sub_id")
    plt.show()



## === cell 9
num_clusts = NUM_CLUSTS  # 6 to 10

clust_rows_bool = (
    ((train_meta.eeg_sub_id < 56) & (train_meta.eeg_sub_id % 7 == 6))
    | (train_meta.eeg_sub_id == 0)
) & (
    train_meta.eeg_id % 4 < 2
)  # include odd and even eeg_ids



## === cell 10
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

prob_array = np.array(prob_vectors)
kmeans = KMeans(
    n_clusters=num_clusts,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=GLOBAL_SEED,
)
kmeans.fit(prob_array)

clust_centers = kmeans.cluster_centers_

clust_centers = np.clip(clust_centers, 1e-12, 1.0)
clust_centers = clust_centers / clust_centers.sum(axis=1, keepdims=True)

iclust_of_order = []
for icol in range(HBA_number):
    iclust_of_order.append(np.argmax(clust_centers[:, icol]))
clust_by_max = np.argsort(-1 * np.max(clust_centers, axis=1))
for iord in range(HBA_number, num_clusts):
    iclust_of_order.append(clust_by_max[iord])

kmnames = HBA_expert_names.copy()
for ihyb in range(1, (num_clusts - HBA_number) + 1):
    kmnames.append("Hybrid-" + str(ihyb))



## === cell 11
train_meta["clust_id"] = kmeans.predict(np.array(train_meta[HBA_probs]))

all_probs = train_meta[HBA_probs]
all_ids = train_meta["clust_id"]

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



## === cell 12
solution_train = train_meta[["eeg_id"]].copy()
for col_pre in HBA_names:
    solution_train[col_pre + "_vote"] = train_meta[col_pre + "_prob"]

submission_train = solution_train.copy()
clust_ids = train_meta["clust_id"]
for iprob in range(HBA_number):
    this_col_probs = clust_centers[:, iprob]
    submission_train[HBA_votes[iprob]] = this_col_probs[clust_ids]

print(
    "Score if HBA samples are correctly assigned cluster prob.s:",
    np.round(kld_score(solution_train[HBA_votes], submission_train[HBA_votes]), 4),
)



## === cell 13
smooth_width = SMOOTH_WIDTH  # Here smooth_width is just for looking,
spectro_meta = train_meta[clust_rows_bool].copy()
every_nth = 37



## === cell 14
freqs = np.array(range(100)) * 0.19525 + 0.59
spect_trend = 150.0 / (1.0**2.3 + freqs ** (2.3))



## === cell 15
if False:
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
        max_base = max(0, len(this_spectro) - 151)  # needs +150 below
        loc_offset = int(np.clip(loc_offset, 0, max_base))
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
    plt.ylim(-1.0, 1.0)  # log already taken
    plt.xlim(0.0, 20.5)
    plt.title("Middle-4s Spectra (color-coded by the {} clusters)".format(num_clusts))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("log10[] Amplitude (/ref-spectrum, /mean, and smoothed n=3) ]")
    plt.savefig("middle4s_spectra.png")
    plt.show()



## === cell 16
pass




## === cell 17
def _spectrogram_file(traintest, spectrogram_id):
    sid = str(int(spectrogram_id))
    candidates = [
        os.path.join(above_dir, f"{traintest}_spectrograms", f"{sid}.parquet"),
        os.path.join(
            above_dir,
            "hms-harmful-brain-activity-classification",
            f"{traintest}_spectrograms",
            f"{sid}.parquet",
        ),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def _load_spectrogram_df(path):
    try:
        ds = pads.dataset(path)
        df = ds.to_table().to_pandas()
        return df
    except Exception:
        return None


def assemble_features(meta_frame, traintest="train", smooth_width=5, SHOW_PLOT=True):
    """
    Create a dataframe of spectrogram features from the meta_frame rows.
    Will include clust_id (i.e, the y) if it is in the input meta_frame.
    Assumes these are available: above_dir, num_clusts

    CHANGE (score+stability, no semantic change): cache loaded spectrogram parquet tables by spectrogram_id
    so we don't repeatedly hit disk / parquet decode; this prevents timeouts that can otherwise stop
    before writing submission.csv (which yielded "Not yielded").
    """
    freqs = np.array(range(100)) * 0.19525 + 0.59
    spect_trend = 150.0 / (1.0**2.3 + freqs ** (2.3))
    freqs4 = np.array(4 * list(freqs))
    spect_trend4 = np.array(4 * list(spect_trend))

    if SHOW_PLOT:
        plt.figure(figsize=(10, 8))

    feats_frame = []
    n_spec_bins = 400

    spectro_cache = {}
    cache_hits = 0
    cache_misses = 0

    default_middle4s = pd.Series(np.full(n_spec_bins, 0.001, dtype=np.float64))

    for irow in meta_frame.index:
        this_row = meta_frame.loc[irow]
        spectro_id_str = str(int(this_row.spectrogram_id))  # make sure int

        if spectro_id_str in spectro_cache:
            this_spectro = spectro_cache[spectro_id_str]
            cache_hits += 1
        else:
            spectro_file = _spectrogram_file(traintest, this_row.spectrogram_id)
            this_spectro = _load_spectrogram_df(spectro_file)
            spectro_cache[spectro_id_str] = this_spectro
            cache_misses += 1

        if (
            this_spectro is None
            or len(this_spectro) < 151
            or this_spectro.shape[1] < (1 + n_spec_bins)
        ):
            middle4s_raw = default_middle4s.copy()
        else:
            loc_offset = int(this_row.spectrogram_label_offset_seconds / 2)
            max_base = max(0, len(this_spectro) - 151)  # needs +150 below
            loc_offset = int(np.clip(loc_offset, 0, max_base))

            middle4s_raw = (
                this_spectro.iloc[loc_offset + 149, 1:]
                + this_spectro.iloc[loc_offset + 150, 1:]
            )

            if len(middle4s_raw) != n_spec_bins:
                middle4s_raw = middle4s_raw.iloc[:n_spec_bins]
                if len(middle4s_raw) < n_spec_bins:
                    middle4s_raw = pd.concat(
                        [
                            middle4s_raw,
                            pd.Series(np.full(n_spec_bins - len(middle4s_raw), np.nan)),
                        ],
                        ignore_index=True,
                    )
            middle4s_raw = middle4s_raw.reset_index(drop=True)

        middle4s = middle4s_raw / (2.0 * spect_trend4)  # 4 copies of trend
        middle4s = middle4s.replace([np.nan, -np.inf, np.inf], 0.001)
        middle4s = np.clip(middle4s, 0.001, 1000.0)

        spect_mean = np.mean(middle4s)  # over all 4 spectra
        spect_median = np.median(middle4s)

        the4means = []
        the4medians = []
        for ispec in range(4):
            ibeg = ([0, 100, 200, 300])[ispec]
            iend = ibeg + 100
            the4means.append(np.mean(middle4s[ibeg:iend]))
            the4medians.append(np.median(middle4s[ibeg:iend]))

        middle4s = np.log10(middle4s / spect_mean)
        middle4s = middle4s.rolling(
            smooth_width, min_periods=1, center=True, closed=None
        ).mean()

        baseinds = np.arange(int((smooth_width - 1) / 2), 100, smooth_width)
        select_inds = np.concatenate(
            (baseinds, 100 + baseinds, 200 + baseinds, 300 + baseinds)
        )
        freqs4ds = freqs4[select_inds]
        middle4sds = middle4s.iloc[select_inds].reset_index(drop=True)

        these_feats = middle4sds.to_frame().T  # 0..N-1 integer columns here
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
            feats_frame = pd.concat([feats_frame, these_feats], ignore_index=True)

        if len(feats_frame) % 200 == 0:
            print(
                "... {} done... (cache hits={}, misses={})".format(
                    len(feats_frame), cache_hits, cache_misses
                )
            )

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
                -1.0 * freqs4ds[0] + smooth_width * 0.15 * (np.random.rand(4) - 0.5),
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
    return feats_frame.reset_index(drop=True)




## === cell 18
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



## === cell 19
Xy_train_feats



## === cell 20
valid_rows_bool = (
    ((train_meta.eeg_sub_id < 21) & (train_meta.eeg_sub_id % 7 == 4))
    | (train_meta.eeg_sub_id == 0)
) & (
    train_meta.eeg_id % 4 > 1
)  # include odd and even eeg_ids

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



## === cell 21
Xy_valid_feats



## === cell 22
pass



## === cell 23
X = Xy_train_feats.drop(columns=["clust_id"]).copy()
X.columns = X.columns.astype(str)
y = Xy_train_feats.clust_id

X_valid = Xy_valid_feats.drop(columns=["clust_id"]).copy()
X_valid.columns = X_valid.columns.astype(str)
X_valid = X_valid.reindex(columns=X.columns, fill_value=0.0)

ave_oob = []
best_oob = -1.0
rfmodel = None  # ensure defined for later cells
nfits = 5
for ifit in range(nfits):
    model = RandomForestClassifier(
        n_estimators=100,
        max_leaf_nodes=int(4 * NUM_CLUSTS),
        max_features=0.5,
        max_samples=0.7,
        oob_score=True,
        class_weight="balanced_subsample",
        n_jobs=-1,
        verbose=0,
        random_state=GLOBAL_SEED + ifit,
    ).fit(X, y)
    ave_oob.append(model.oob_score_)
    if model.oob_score_ > best_oob:
        best_oob = float(model.oob_score_)
        rfmodel = model

print(
    "RF model ave OOB score = {:.1f}% +/- {:.1f}".format(
        100 * np.mean(ave_oob), 100 * np.std(ave_oob)
    )
)
print("Using best OOB model = {:.1f}%".format(100 * best_oob))
print("\nRF model score for X,y = {:.1f}%".format(100 * rfmodel.score(X, y)))



## === cell 24
if False:
    sort_inds = rfmodel.feature_importances_.argsort()
    plt.figure(figsize=(4, 5))
    plt.barh(
        rfmodel.feature_names_in_[sort_inds], rfmodel.feature_importances_[sort_inds]
    )
    plt.ylim(int(2 * len(sort_inds) / 3), len(sort_inds) + 0.2)
    plt.title("Feature Importances (top third)")
    plt.show()




## === cell 25
def clusterprob_to_hbaprob(cluster_probs, centers):
    return cluster_probs @ centers


def predict_proba_all_classes(model, X, n_classes):
    proba = model.predict_proba(X)
    out = np.zeros((len(X), n_classes), dtype=np.float64)
    for j, c in enumerate(model.classes_):
        if 0 <= int(c) < n_classes:
            out[:, int(c)] = proba[:, j]
    return out


mean_all_probs = train_meta[HBA_probs].mean(axis=0).to_numpy(dtype=np.float64)
mean_all_probs = mean_all_probs / mean_all_probs.sum()
print("Empirical mean_all_probs =", np.round(mean_all_probs, 6))

Xy_train_meta["pred_id"] = rfmodel.predict(X)

solution = Xy_train_meta[["eeg_id"]].copy()
for col_pre in HBA_names:
    solution[col_pre + "_vote"] = Xy_train_meta[col_pre + "_prob"]

train_cluster_proba = predict_proba_all_classes(rfmodel, X, num_clusts)
train_hba_pred = clusterprob_to_hbaprob(train_cluster_proba, clust_centers)

submission = solution.copy()
for i, col in enumerate(HBA_votes):
    submission[col] = train_hba_pred[:, i]

print(
    "Train KL using soft cluster mixture (untamed centers):",
    np.round(kld_score(solution[HBA_votes], submission[HBA_votes]), 4),
)




## === cell 26
def _fix_submission_probs(df, prob_cols, eps=1e-6):
    arr = df[prob_cols].to_numpy(dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    row_sums = arr.sum(axis=1, keepdims=True)
    bad = row_sums[:, 0] <= 0.0
    if np.any(bad):
        arr[bad, :] = 1.0 / len(prob_cols)
        row_sums[bad, :] = 1.0
    arr = arr / row_sums
    arr = np.clip(arr, eps, 1.0)
    arr = arr / arr.sum(axis=1, keepdims=True)
    df.loc[:, prob_cols] = arr
    return df


solution_v = Xy_valid_meta[["eeg_id"]].copy()
for col_pre in HBA_names:
    solution_v[col_pre + "_vote"] = Xy_valid_meta[col_pre + "_prob"]

valid_cluster_proba = predict_proba_all_classes(rfmodel, X_valid, num_clusts)
valid_hba_pred = clusterprob_to_hbaprob(valid_cluster_proba, clust_centers)

submission_v = solution_v.copy()
for i, col in enumerate(HBA_votes):
    submission_v[col] = valid_hba_pred[:, i]
submission_v = _fix_submission_probs(submission_v, HBA_votes, eps=1e-6)

base_kl = kld_score(solution_v[HBA_votes], submission_v[HBA_votes])
print("Validation KL using soft cluster mixture (no blend): {:.4f}".format(base_kl))

best_alpha = 1.0
best_kl = base_kl
mean_mat = np.tile(mean_all_probs.reshape(1, -1), (len(submission_v), 1))

for alpha in np.arange(1.0, -0.001, -0.05):
    tmp = submission_v.copy()
    blended = (
        alpha * tmp[HBA_votes].to_numpy(dtype=np.float64) + (1.0 - alpha) * mean_mat
    )
    tmp.loc[:, HBA_votes] = blended
    tmp = _fix_submission_probs(tmp, HBA_votes, eps=1e-6)
    this_kl = kld_score(solution_v[HBA_votes], tmp[HBA_votes])
    if this_kl < best_kl:
        best_kl = this_kl
        best_alpha = float(alpha)

print("Best validation blend alpha =", best_alpha, "-> KL =", np.round(best_kl, 4))




## === cell 27
def tame_centers(centers, frac, mean_probs):
    out = centers.copy()
    for ic in range(out.shape[0]):
        out[ic, :] = frac * out[ic, :] + (1.0 - frac) * mean_probs
        s = out[ic, :].sum()
        if s > 0:
            out[ic, :] /= s
    out = np.clip(out, 1e-12, 1.0)
    out = out / out.sum(axis=1, keepdims=True)
    return out


last_kl = 1e9
best_frac = 1.0
best_centers = clust_centers.copy()

for frac in np.arange(0.0, 1.0001, 0.05):
    centers_t = tame_centers(clust_centers, frac, mean_all_probs)
    valid_hba_pred_t = clusterprob_to_hbaprob(valid_cluster_proba, centers_t)

    tmp = solution_v.copy()
    for i, col in enumerate(HBA_votes):
        tmp[col] = valid_hba_pred_t[:, i]

    blended = (
        best_alpha * tmp[HBA_votes].to_numpy(dtype=np.float64)
        + (1.0 - best_alpha) * mean_mat
    )
    tmp.loc[:, HBA_votes] = blended
    tmp = _fix_submission_probs(tmp, HBA_votes, eps=1e-6)

    this_kl = kld_score(solution_v[HBA_votes], tmp[HBA_votes])
    if this_kl < last_kl:
        last_kl = this_kl
        best_frac = float(frac)
        best_centers = centers_t.copy()

print(
    "Best center-taming frac =", best_frac, "-> validation KL =", np.round(last_kl, 4)
)



## === cell 28
Xy_test_feats = assemble_features(
    test_meta, traintest="test", smooth_width=SMOOTH_WIDTH, SHOW_PLOT=False
)
Xy_test_feats.columns = Xy_test_feats.columns.astype(str)
Xy_test_feats = Xy_test_feats.reindex(columns=X.columns, fill_value=0.0)

test_cluster_proba = predict_proba_all_classes(rfmodel, Xy_test_feats, num_clusts)
test_hba_pred = clusterprob_to_hbaprob(test_cluster_proba, best_centers)

test_submit = test_meta[["eeg_id"]].copy()
for i, col in enumerate(HBA_votes):
    test_submit[col] = test_hba_pred[:, i]

test_mean_mat = np.tile(mean_all_probs.reshape(1, -1), (len(test_submit), 1))
test_submit.loc[:, HBA_votes] = (
    best_alpha * test_submit[HBA_votes].to_numpy(dtype=np.float64)
    + (1.0 - best_alpha) * test_mean_mat
)

test_submit = _fix_submission_probs(test_submit, HBA_votes, eps=1e-6)

sample_sub = pd.read_csv(above_dir + "sample_submission.csv")
test_submit = sample_sub[["eeg_id"]].merge(test_submit, on="eeg_id", how="left")

for col in HBA_votes:
    if col not in test_submit.columns:
        test_submit[col] = np.nan
test_submit[HBA_votes] = test_submit[HBA_votes].fillna(1.0 / HBA_number)
test_submit = _fix_submission_probs(test_submit, HBA_votes, eps=1e-6)
test_submit = test_submit[["eeg_id"] + HBA_votes]

print(test_submit.head())
print(
    "Row-sum check: min/max =",
    test_submit[HBA_votes].sum(axis=1).min(),
    test_submit[HBA_votes].sum(axis=1).max(),
)
print(
    "Any NaNs/Infs in prob cols:",
    (~np.isfinite(test_submit[HBA_votes].to_numpy())).any(),
)
print("Submission length =", len(test_submit), " Sample length =", len(sample_sub))

test_submit.to_csv(
    "submission.csv", header=True, index=False, na_rep="", float_format="%.6f"
)
print("Wrote submission.csv with shape:", test_submit.shape)
