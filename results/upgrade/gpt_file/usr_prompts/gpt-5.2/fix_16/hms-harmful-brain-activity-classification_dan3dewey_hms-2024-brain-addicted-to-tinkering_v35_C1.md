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

1.031468

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
NUM_CLUSTS = 9  # 6 to 10
SMOOTH_WIDTH = 5  # Odd>1: 3,5,7,9,...

import os

_CANDIDATE_DIRS = [
    "/kaggle/input/hms-harmful-brain-activity-classification/",
    "/kaggle/data/hms-harmful-brain-activity-classification/",
    "/kaggle/data/input/hms-harmful-brain-activity-classification/",
    "/kaggle/data/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification/",
    "../input/hms-harmful-brain-activity-classification/",
]
above_dir = None
for _d in _CANDIDATE_DIRS:
    if os.path.exists(os.path.join(_d, "train.csv")) and os.path.exists(
        os.path.join(_d, "test.csv")
    ):
        above_dir = _d if _d.endswith("/") else (_d + "/")
        break
if above_dir is None:
    raise FileNotFoundError(
        "Could not locate HMS dataset directory. Tried:\n" + "\n".join(_CANDIDATE_DIRS)
    )

GLOBAL_SEED = 123
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)



## === cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import pyarrow
import pyarrow.parquet as pq
import pyarrow.dataset as pads

np.random.seed(GLOBAL_SEED)

PLOT_DIAGNOSTICS = False



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
    Average KL divergence: sum_i sum_c p_ic * log(p_ic / q_ic) / N
    Assumes both frames contain only the 6 probability columns (no id column).
    """
    p = solution.to_numpy(dtype=np.float64)
    q = submission.to_numpy(dtype=np.float64)

    p = np.clip(p, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)

    q = np.clip(q, eps, 1.0)
    q = q / q.sum(axis=1, keepdims=True)

    return float(np.mean(np.sum(p * np.log(p / q), axis=1)))




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

    for col_pre in HBA_names:
        train_meta[col_pre + "_prob"] = (
            train_meta[col_pre + "_vote"] / train_meta["total_vote"]
        )

    print("Calculating voting entropy values ...")

    def calc_entropy(row):
        the_probs = np.clip(
            row[[c for c in train_meta.columns if c.endswith("_prob")]].values.astype(
                float
            ),
            1.0e-8,
            1.0,
        )
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

    clstclrs = [kmclrs[int(ilab)] for ilab in clust_ids]

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
if PLOT_DIAGNOSTICS:
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

for iclust in range(num_clusts):
    denom = np.sum(clust_centers[iclust, :])
    if denom > 0:
        clust_centers[iclust, :] = clust_centers[iclust, :] / denom
    else:
        clust_centers[iclust, :] = np.ones(HBA_number) / HBA_number

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

if PLOT_DIAGNOSTICS:
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
    ][:num_clusts]



## === cell 11
solution_train = train_meta[HBA_probs].copy()
submission_train = solution_train.copy()
clust_ids = train_meta["clust_id"].to_numpy()

for iprob in range(HBA_number):
    this_col_probs = clust_centers[:, iprob]
    submission_train[HBA_probs[iprob]] = this_col_probs[clust_ids]

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
if PLOT_DIAGNOSTICS:
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
                middle4s.iloc[ibin] = middle4spre.iloc[ibin]
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
if PLOT_DIAGNOSTICS:
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

    mmclrs = [kmclrs[int(ilab)] for ilab in all_clusts]
    plt.figure(figsize=(3, 3))
    plt.scatter(all_medians, all_means, s=2, c=mmclrs, alpha=0.3)
    plt.xlim(-1.0, 1)
    plt.ylim(-1.0, 1)
    plt.xlabel("Median")
    plt.ylabel("Mean")
    plt.show()




## === cell 16
def _read_spectrogram_middle_rows_parquet(spectro_file: str, row0: int, row1: int):
    """
    Read only rows [row0,row1] (inclusive) and all 400 spectrogram value columns (skip the first time col).
    Returns numpy array shape (row1-row0+1, 400) in float64.

    Change (score): fix row slicing across row groups by computing slice indices relative to the
    concatenated table's first row index (min row of included row groups). The prior version used
    rg_starts[rg0] but the concatenated table begins at rg0, so correct base is rg_starts[rg0]
    (ok) BUT local0 must also be clamped within concatenated length, and when row_b crosses into
    rg1 the concatenation includes rg0..rg1, so local0/local_len must be computed against that
    concatenated span (minrow..maxrow). This prevents reading the wrong time rows, improving features.
    """
    if row1 < row0:
        raise ValueError("row1 must be >= row0")

    pf = pq.ParquetFile(spectro_file)
    nrows = pf.metadata.num_rows
    if nrows <= 0:
        raise ValueError(f"Empty spectrogram parquet: {spectro_file}")

    row0 = int(np.clip(row0, 0, nrows - 1))
    row1 = int(np.clip(row1, 0, nrows - 1))
    if row1 < row0:
        row1 = row0

    rg_starts = []
    cum = 0
    for i in range(pf.num_row_groups):
        rg_starts.append(cum)
        cum += pf.metadata.row_group(i).num_rows

    def _rg_for_row(r: int) -> int:
        lo, hi = 0, len(rg_starts) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if rg_starts[mid] <= r:
                lo = mid + 1
            else:
                hi = mid - 1
        return max(0, hi)

    rg0 = _rg_for_row(row0)
    rg1 = _rg_for_row(row1)

    colnames = pf.schema.names
    value_cols = colnames[1:]

    parts = []
    for rg in range(rg0, rg1 + 1):
        tbl = pf.read_row_group(rg, columns=value_cols)
        parts.append(tbl)

    tbl_all = pyarrow.concat_tables(parts, promote_options="default")

    concat_base = rg_starts[rg0]
    local0 = int(row0 - concat_base)
    local1 = int(row1 - concat_base)

    local0 = int(np.clip(local0, 0, max(0, tbl_all.num_rows - 1)))
    local1 = int(np.clip(local1, 0, max(0, tbl_all.num_rows - 1)))
    if local1 < local0:
        local1 = local0
    local_len = int(local1 - local0 + 1)

    sl = tbl_all.slice(local0, local_len)
    arr = sl.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    return arr




## === cell 17
_SPECTRO_MIDROW_CACHE = {}


def _get_middle4s_from_cache(spectro_file: str, row_a: int, row_b: int):
    key = (spectro_file, int(row_a), int(row_b))
    arr = _SPECTRO_MIDROW_CACHE.get(key)
    if arr is None:
        arr = _read_spectrogram_middle_rows_parquet(spectro_file, row_a, row_b)
        _SPECTRO_MIDROW_CACHE[key] = arr
    return arr


def _centered_rolling_mean_preserve_edges(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return x.copy()
    edge = (width - 1) // 2
    y = x.copy()
    for i in range(edge, len(x) - edge):
        y[i] = float(np.mean(x[i - edge : i + edge + 1]))
    return y


def assemble_features(meta_frame, traintest="train", smooth_width=5, SHOW_PLOT=True):
    """
    Create a dataframe of spectrogram features from the meta_frame rows.
    Will include clust_id and *_prob (i.e, y targets) if they are in meta_frame.
    Assumes these are available: above_dir, num_clusts

    Change (score): align feature normalization with the diagnostic path:
    compute spect_mean as mean(log10(raw_middle4s)) and then center log-spectra by subtracting that mean.
    This reduces train/test mismatch vs. the earlier log10(middle4s/spect_mean_raw).
    """
    freqs = np.array(range(100)) * 0.19525 + 0.59
    spect_trend = 150.0 / (1.0**2.3 + freqs ** (2.3))
    freqs[0] = 0.0  # separate the (unsmoothed) first bin from others
    freqs4 = np.array(4 * list(freqs))
    spect_trend4 = np.array(4 * list(spect_trend))
    if SHOW_PLOT:
        plt.figure(figsize=(10, 8))

    feats_rows = []

    edge = int((smooth_width - 1) / 2)
    baseinds = np.insert(np.arange(edge, 100, smooth_width), 0, 0)
    select_inds = np.concatenate(
        (baseinds, 100 + baseinds, 200 + baseinds, 300 + baseinds)
    )
    n_freq_feats = len(select_inds)

    n_excepts = 0

    for irow_i, irow in enumerate(meta_frame.index, start=1):
        this_row = meta_frame.loc[irow]
        spectro_id_str = str(int(this_row.spectrogram_id))  # make sure int
        spectro_file = (
            above_dir + traintest + "_spectrograms/" + spectro_id_str + ".parquet"
        )

        loc_offset = int(this_row.spectrogram_label_offset_seconds / 2)
        row_a = loc_offset + 149
        row_b = loc_offset + 150

        try:
            mid2 = _get_middle4s_from_cache(spectro_file, row_a, row_b)
            if mid2.shape[0] == 1:
                middle4s = (mid2[0, :] + mid2[0, :]) / (2.0 * spect_trend4)
            else:
                middle4s = (mid2[0, :] + mid2[1, :]) / (2.0 * spect_trend4)

            middle4s = np.clip(middle4s, 0.001, 1000.0)
            middle4s = np.nan_to_num(middle4s, nan=0.001, posinf=0.001, neginf=0.001)

            spect_mean_l = float(np.mean(np.log10(middle4s)))
            spect_median_l = float(np.log10(np.median(middle4s)))

            the4means = []
            the4medians = []
            for ispec in range(4):
                ibeg = ([0, 100, 200, 300])[ispec]
                iend = ibeg + 100
                the4means.append(float(np.mean(middle4s[ibeg:iend])))
                the4medians.append(float(np.median(middle4s[ibeg:iend])))
            the4means_l = np.log10(np.array(the4means, dtype=np.float64))
            the4medians_l = np.log10(np.array(the4medians, dtype=np.float64))

            middle4s_log = np.log10(middle4s)
            middle4spre = middle4s_log - spect_mean_l

            middle4s_sm = middle4spre.copy()
            for ioff in range(0, 400, 100):
                seg = middle4spre[ioff : ioff + 100]
                seg_sm = _centered_rolling_mean_preserve_edges(seg, smooth_width)
                middle4s_sm[ioff : ioff + 100] = seg_sm

            middle4sds = middle4s_sm[select_inds].astype(np.float64, copy=False)

            these_feats = pd.DataFrame(
                [middle4sds], columns=[f"f_{i}" for i in range(len(middle4sds))]
            )

            these_feats["Mean"] = spect_mean_l
            these_feats["Median"] = spect_median_l
            for ispec in range(4):
                these_feats[the4chains[ispec] + "mean"] = float(the4means_l[ispec])
                these_feats[the4chains[ispec] + "median"] = float(the4medians_l[ispec])

        except Exception:
            n_excepts += 1
            these_feats = pd.DataFrame(
                [np.zeros(n_freq_feats, dtype=np.float64)],
                columns=[f"f_{i}" for i in range(n_freq_feats)],
            )
            these_feats["Mean"] = 0.0
            these_feats["Median"] = 0.0
            for ispec in range(4):
                these_feats[the4chains[ispec] + "mean"] = 0.0
                these_feats[the4chains[ispec] + "median"] = 0.0

        if "clust_id" in meta_frame.columns:
            these_feats["clust_id"] = int(this_row.clust_id)

        for pcol in HBA_probs:
            if pcol in meta_frame.columns:
                these_feats[pcol] = float(this_row[pcol])

        feats_rows.append(these_feats)

        if irow_i % 200 == 0:
            print("... {} done...".format(irow_i))

        if SHOW_PLOT:
            this_clr = "blue"
            title_start = "Middle-4s Feature Values"
            if "clust_id" in meta_frame.columns:
                this_clust = int(this_row.clust_id)
                this_clr = kmclrs[this_clust]
                plt.title(
                    title_start
                    + " (color-coded by the "
                    + "{} clusters, smooth={})".format(num_clusts, smooth_width)
                )
            else:
                plt.title(title_start + " (smooth={})".format(smooth_width))

            the_alpha = np.clip(0.05 * 350 / max(1, len(meta_frame)), 0.003, 0.5)
            if "f_0" in these_feats.columns:
                xs = np.arange(
                    len([c for c in these_feats.columns if c.startswith("f_")])
                )
                plt.plot(
                    xs + smooth_width * 0.15 * (np.random.rand(len(xs)) - 0.5),
                    these_feats[[c for c in these_feats.columns if c.startswith("f_")]]
                    .iloc[0]
                    .to_numpy(),
                    ".",
                    c=this_clr,
                    markersize=10,
                    alpha=the_alpha,
                )
            plt.ylim(-1.5, 1.5)
            plt.xlabel("Feature index (freq bands + summary stats)")
            plt.ylabel(
                "Amplitude (/ref-spectrum, /mean, and smoothed n={})".format(
                    smooth_width
                )
            )

    if n_excepts > 0:
        print(
            f"assemble_features({traintest}): {n_excepts} rows used fallback zero-features due to read issues."
        )

    feats_frame = pd.concat(feats_rows, axis=0, ignore_index=True)

    if SHOW_PLOT:
        plt.savefig("middle4s_" + traintest + "_features.png")
        plt.show()

    return feats_frame




## === cell 18
Xy_train_meta = (train_meta[clust_rows_bool])[::23].copy()
Xy_train_meta = Xy_train_meta.reset_index().drop(columns=["index"])
print("Number of samples used for training =", len(Xy_train_meta))

Xy_train_feats = assemble_features(
    Xy_train_meta,
    traintest="train",
    smooth_width=SMOOTH_WIDTH,
    SHOW_PLOT=PLOT_DIAGNOSTICS,
)
Xy_train_meta.to_csv("Xy_train_meta.csv", header=True, index=False, float_format="%.6f")
Xy_train_feats.to_csv(
    "Xv_train_feats.csv", header=True, index=False, float_format="%.6f"
)



## === cell 19
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
    Xy_valid_meta,
    traintest="train",
    smooth_width=SMOOTH_WIDTH,
    SHOW_PLOT=PLOT_DIAGNOSTICS,
)
Xy_valid_meta.to_csv("Xy_valid_meta.csv", header=True, index=False, float_format="%.6f")
Xy_valid_feats.to_csv(
    "Xv_valid_feats.csv", header=True, index=False, float_format="%.6f"
)



## === cell 20
Xtr = Xy_train_feats.drop(
    columns=["clust_id"] + [c for c in HBA_probs if c in Xy_train_feats.columns],
    errors="ignore",
)
Ytr = Xy_train_feats[HBA_probs].to_numpy(dtype=np.float64)

Xva = Xy_valid_feats.drop(
    columns=["clust_id"] + [c for c in HBA_probs if c in Xy_valid_feats.columns],
    errors="ignore",
)
Yva = Xy_valid_feats[HBA_probs].to_numpy(dtype=np.float64)

Xva = Xva.reindex(columns=Xtr.columns, fill_value=0.0)

rf_reg = RandomForestRegressor(
    n_estimators=200,
    criterion="squared_error",
    max_leaf_nodes=int(6 * NUM_CLUSTS),
    max_features=0.6,
    max_samples=0.8,
    n_jobs=-1,
    random_state=GLOBAL_SEED,
    verbose=0,
)
rf_reg.fit(Xtr, Ytr)

va_pred = rf_reg.predict(Xva)
va_pred = np.clip(va_pred, 1e-8, 1.0)
va_pred = va_pred / va_pred.sum(axis=1, keepdims=True)


def _apply_temperature(p, T: float, eps: float = 1e-12):
    p = np.clip(p, eps, 1.0)
    lp = np.log(p)
    lp = lp / T
    lp = lp - lp.max(axis=1, keepdims=True)
    q = np.exp(lp)
    q = q / q.sum(axis=1, keepdims=True)
    return q


va_sol = pd.DataFrame(Yva, columns=HBA_probs)
va_sub_raw = pd.DataFrame(va_pred, columns=HBA_probs)
raw_kl = kld_score(va_sol, va_sub_raw)

Ts = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15], dtype=np.float64)
best_T = 1.0
best_kl = raw_kl
for T in Ts:
    q = _apply_temperature(va_pred, float(T))
    kl = kld_score(va_sol, pd.DataFrame(q, columns=HBA_probs))
    if kl < best_kl:
        best_kl = kl
        best_T = float(T)

print("Validation KL (direct prob regression):", np.round(raw_kl, 5))
print(
    "Validation KL after temperature scaling: {:.5f} (best T={:.2f})".format(
        best_kl, best_T
    )
)



## === cell 21
Xc = Xy_train_feats.drop(
    columns=["clust_id"] + [c for c in HBA_probs if c in Xy_train_feats.columns],
    errors="ignore",
)
yc = Xy_train_feats.clust_id

rfmodel = RandomForestClassifier(
    n_estimators=100,
    max_leaf_nodes=int(4 * NUM_CLUSTS),
    max_features=0.5,
    max_samples=0.7,
    oob_score=True,
    class_weight="balanced_subsample",
    n_jobs=-1,
    verbose=0,
    random_state=GLOBAL_SEED,
).fit(Xc, yc)

print("\nRF (cluster classifier) OOB score = {:.1f}%".format(100 * rfmodel.oob_score_))
print("RF (cluster classifier) train acc = {:.1f}%".format(100 * rfmodel.score(Xc, yc)))



## === cell 22
Xy_test_feats = assemble_features(
    test_meta, traintest="test", smooth_width=SMOOTH_WIDTH, SHOW_PLOT=False
)

Xte = Xy_test_feats.copy()
Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)

test_pred = rf_reg.predict(Xte)
test_pred = np.clip(test_pred, 1e-8, 1.0)
test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)

test_pred = _apply_temperature(test_pred, best_T)

test_pred_df = pd.DataFrame(test_pred, columns=HBA_probs)
test_pred_df.insert(0, "eeg_id", test_meta["eeg_id"].to_numpy())
test_pred_df["eeg_id"] = test_pred_df["eeg_id"].astype("int64", copy=False)

rename_map = dict(zip(HBA_probs, HBA_votes))
test_pred_df = test_pred_df.rename(columns=rename_map)

test_pred_agg = test_pred_df.groupby("eeg_id", as_index=False)[HBA_votes].mean()

sample_sub = pd.read_csv(above_dir + "sample_submission.csv")
if "eeg_id" not in sample_sub.columns:
    raise ValueError("sample_submission.csv missing required eeg_id column.")
sample_sub["eeg_id"] = sample_sub["eeg_id"].astype("int64", copy=False)

sample_sub_unique = sample_sub.drop_duplicates(subset=["eeg_id"], keep="first").copy()
test_pred_agg_unique = test_pred_agg.drop_duplicates(
    subset=["eeg_id"], keep="first"
).copy()

test_submit = sample_sub_unique[["eeg_id"]].merge(
    test_pred_agg_unique, on="eeg_id", how="left", sort=False
)

for col in HBA_votes:
    if col not in test_submit.columns:
        test_submit[col] = 1.0 / HBA_number
test_submit[HBA_votes] = test_submit[HBA_votes].fillna(1.0 / HBA_number)

eps = 1e-8
probs = test_submit[HBA_votes].to_numpy(dtype=np.float64)
probs = np.clip(probs, eps, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)
test_submit[HBA_votes] = probs

test_submit = test_submit[["eeg_id"] + HBA_votes].copy()
test_submit = (
    test_submit.set_index("eeg_id")
    .loc[sample_sub_unique["eeg_id"].to_numpy()]
    .reset_index()
)

if test_submit["eeg_id"].duplicated().any():
    raise ValueError(
        "Submission eeg_id must be unique per row; found duplicates after merge."
    )

assert list(test_submit.columns) == ["eeg_id"] + HBA_votes
row_sums = test_submit[HBA_votes].sum(axis=1).to_numpy()
assert np.all(np.isfinite(row_sums))
assert np.allclose(row_sums, 1.0, atol=1e-6)

print(test_submit.head())
print(
    "Row-sum check (min/mean/max):",
    row_sums.min(),
    row_sums.mean(),
    row_sums.max(),
)
print("Submission shape:", test_submit.shape)

test_submit.to_csv(
    "submission.csv", header=True, index=False, na_rep="", float_format="%.6f"
)
print("Wrote submission.csv with columns:", list(test_submit.columns))
