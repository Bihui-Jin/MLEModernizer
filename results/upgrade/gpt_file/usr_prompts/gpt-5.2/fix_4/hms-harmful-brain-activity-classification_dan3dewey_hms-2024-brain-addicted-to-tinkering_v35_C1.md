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

import pyarrow
import pyarrow.parquet as pq
import pyarrow.dataset as pads

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

np.random.seed(GLOBAL_SEED)



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

if True:
    kmclrs = prob_prob_scatter("Seizure", "GPD", all_probs, all_ids, iclust_of_order)
    kmclrs = prob_prob_scatter("LPD", "GRDA", all_probs, all_ids, iclust_of_order)
    kmclrs = prob_prob_scatter("LRDA", "Other", all_probs, all_ids, iclust_of_order)



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
plt.figure(figsize=(10, 8))

all_medians = []
all_means = []
all_clusts = []
print("Plotting {} x 4 processed spectra".format(int(len(spectro_meta) / every_nth)))
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
def assemble_features(meta_frame, traintest="train", smooth_width=5, SHOW_PLOT=True):
    """
    Create a dataframe of spectrogram features from the meta_frame rows.
    Will include clust_id and *_prob (i.e, y targets) if they are in meta_frame.
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
    this_spectro = None

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
        middle4s_sm = middle4spre.rolling(
            smooth_width, min_periods=smooth_width, center=True, closed=None
        ).mean()

        for ioff in range(0, 400, 100):
            for ibin in range(int((smooth_width - 1) / 2)):  # assumes width is odd
                middle4s_sm.iloc[ibin + ioff] = middle4spre.iloc[ibin + ioff]

        baseinds = np.insert(
            np.arange(int((smooth_width - 1) / 2), 100, smooth_width), 0, 0
        )
        select_inds = np.concatenate(
            (baseinds, 100 + baseinds, 200 + baseinds, 300 + baseinds)
        )
        freqs4ds = freqs4[select_inds]
        middle4sds = middle4s_sm.iloc[select_inds]

        these_feats = middle4sds.to_frame().T  # already log10()
        spect_mean_l = np.log10(spect_mean)
        spect_median_l = np.log10(spect_median)
        the4means_l = np.log10(the4means)
        the4medians_l = np.log10(the4medians)

        these_feats["Mean"] = spect_mean_l
        these_feats["Median"] = spect_median_l
        for ispec in range(4):
            these_feats[the4chains[ispec] + "mean"] = the4means_l[ispec]
            these_feats[the4chains[ispec] + "median"] = the4medians_l[ispec]

        if "clust_id" in meta_frame.columns:
            these_feats["clust_id"] = this_row.clust_id

        for pcol in HBA_probs:
            if pcol in meta_frame.columns:
                these_feats[pcol] = float(this_row[pcol])

        if len(feats_frame) == 0:
            feats_frame = these_feats.copy()
        else:
            feats_frame = pd.concat([feats_frame, these_feats], axis=0)

        if len(feats_frame) % 200 == 0:
            print("... {} done...".format(len(feats_frame)))

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
                the4means_l,
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
        plt.savefig("middle4s_" + traintest + "_features.png")
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



## === cell 19
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

rf_reg = RandomForestRegressor(
    n_estimators=200,
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

va_sol = pd.DataFrame(Yva, columns=HBA_probs)
va_sub = pd.DataFrame(va_pred, columns=HBA_probs)
print("Validation KL (direct prob regression):", np.round(kld_score(va_sol, va_sub), 5))



## === cell 20
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



## === cell 21
Xy_test_feats = assemble_features(
    test_meta, traintest="test", smooth_width=SMOOTH_WIDTH, SHOW_PLOT=False
)

Xte = Xy_test_feats.copy()
Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)

test_pred = rf_reg.predict(Xte)
test_pred = np.clip(test_pred, 1e-8, 1.0)
test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)

test_submit = test_meta[["eeg_id"]].copy()
for i, col in enumerate(HBA_votes):
    test_submit[col] = test_pred[:, i]

eps = 1e-8
probs = test_submit[HBA_votes].to_numpy(dtype=np.float64)
probs = np.clip(probs, eps, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)
test_submit[HBA_votes] = probs

print(test_submit.head())
print(
    "Row-sum check (min/mean/max):",
    probs.sum(axis=1).min(),
    probs.sum(axis=1).mean(),
    probs.sum(axis=1).max(),
)

test_submit.to_csv(
    "submission.csv", header=True, index=False, na_rep="", float_format="%.6f"
)
print("Wrote submission.csv with shape:", test_submit.shape)
