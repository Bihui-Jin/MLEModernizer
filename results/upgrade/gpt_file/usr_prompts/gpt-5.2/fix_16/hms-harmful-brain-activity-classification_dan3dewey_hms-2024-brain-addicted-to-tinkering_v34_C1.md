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

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.25526) has done: 'I fix the execution blockers without changing the modeling approach: the main runtime error is scikit-learn rejecting mixed-type feature names (int + str), so I make all feature column names strings consistently for train/valid/test. I also ensure the LR and RF `.predict/.predict_proba` calls use the same exact column order by explicitly aligning feature columns across datasets. Finally, I harden submission creation to guarantee the required header (including `eeg_id`), correct column order, numeric dtype, and per-row probability normalization so Kaggle accepts the file.'
- What this solution (achieved 1.42346) has done: 'To get a valid Kaggle score (instead of “Not yielded”), the main fix is to ensure the submission probabilities are strictly valid: all six class columns must be finite, strictly positive after clipping, and each row must sum to exactly 1. I also align the submission row order to `sample_submission.csv` (by `eeg_id`) to avoid any hidden ordering mismatch that can silently worsen or invalidate scoring. Finally, I make the “taming” optimization not mutate the passed `submission` frame in-place (it currently does), which can make the search unstable and can lead to inconsistent centers/prediction mapping; this preserves the same core approach but makes it deterministic and correct. These are minimal changes and keep your modeling/training logic intact.'
- What this solution (achieved 1.2471) has done: 'Your pipeline currently can fail to “yield” a Kaggle score if the submission has any tiny row-sum drift, NaNs/infs, negative values, or misalignment to `sample_submission.csv`; those issues can happen here due to mixed dtypes, missing feature columns from parquet read oddities, or downstream matrix multiplication producing slight invalid probabilities. I add a single, centralized “make_submission_valid” post-processing step applied right before writing `submission.csv`, enforcing strict positivity, finiteness, exact renormalization, and exact row order by `eeg_id`. I also make the spectrogram slice reader robust to parquet schemas where the 1..400 columns are not strings (some are ints), by falling back to reading all columns and selecting the first 400 feature columns deterministically; this preserves your feature logic but prevents silent all-zero feature rows that can poison predictions. These are minimal execution/correctness fixes (not model changes) and should produce a valid submission consistently, enabling a real score and typically improving it versus invalid/degenerate outputs.'
- What this solution (achieved 1.25473) has done: 'I make the submission pipeline “yield” reliably by ensuring the generated test feature rows are aligned 1:1 with `test.csv`/`sample_submission.csv` and that the regressor input contains only the intended feature columns (right now `eeg_id`/meta columns can accidentally leak into `X_test_reg`, causing schema mismatches or silent degradation). I also harden `assemble_features()` so it always carries `eeg_id` through (without changing feature extraction), preventing any accidental row-order drift between features and metadata. Finally, I keep your existing model/training logic intact, but ensure the test-time feature frame is constructed with the exact same column set/order used in training (which is critical for scikit-learn consistency and score stability).'

# 9. Code solution

## === cell 0
NUM_CLUSTS = 6  # 6 to 10
SMOOTH_WIDTH = 5  # Odd>1: 3,5,7,9,...
above_dir = "../input/hms-harmful-brain-activity-classification/"
GLOBAL_SEED = 42



## === cell 1
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import pyarrow
import pyarrow.parquet as pq
import pyarrow.dataset as pads

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

plt.ioff()



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
np.random.seed(GLOBAL_SEED)




## === cell 3
def kld_score(solution, submission, eps=1e-8):
    """
    Calculate the average KL divergence score.
    Assumes solution/submission have identical probability column names.
    """
    sol = solution.to_numpy(dtype=np.float64)
    sub = submission.to_numpy(dtype=np.float64)

    sol = np.clip(sol, eps, 1.0)
    sol = sol / sol.sum(axis=1, keepdims=True)

    sub = np.clip(sub, eps, 1.0)
    sub = sub / sub.sum(axis=1, keepdims=True)

    return float(np.mean(np.sum(sol * (np.log(sol) - np.log(sub)), axis=1)))




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

    probs = train_meta[HBA_probs].to_numpy(dtype=np.float64)
    probs = np.clip(probs, 1.0e-8, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    train_meta["entropy"] = np.nansum(probs * (-np.log(probs)), axis=1)

    return train_meta, test_meta




## === cell 5
def prob_prob_scatter(name1, name2, probs2plot, clust_ids, iclust_order=[0]):
    """
    Make a prob1 vs prob2 scatter plot (diagnostic only).
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
    ]
    if len(iclust_order) > 2:
        kmclrs = hba_clrs.copy()
        for iord, iclust in enumerate(iclust_order):
            kmclrs[iclust] = hba_clrs[iord]

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
if False:
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

prob_array = prob_vectors.to_numpy(dtype=np.float64)

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
train_meta["clust_id"] = kmeans.predict(
    train_meta[HBA_probs].to_numpy(dtype=np.float64)
)

all_probs = train_meta[HBA_probs]
all_ids = train_meta["clust_id"]

if False:
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



## === cell 11
solution_train = train_meta[["eeg_id"] + HBA_votes].copy()
for col_pre in HBA_names:
    solution_train.loc[:, col_pre + "_vote"] = train_meta[col_pre + "_prob"]

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
            for ibin in range(int((smooth_width - 1) / 2)):
                middle4s[ibin] = middle4spre[ibin]
            plt.plot((freqs), middle4s, c=kmclrs[this_clust], lw=3, alpha=0.01)
        if (ilocrow / every_nth + 1) % 100 == 0:
            print("... {} done...".format(int(ilocrow / every_nth) + 1))
    plt.plot([0.0, 20.0], [0.0, 0.0], c="black", lw=3, alpha=0.2)
    downsel_freqs = np.insert(
        freqs[int((smooth_width - 1) / 2) : 100 : smooth_width], 0, freqs[0]
    )
    plt.plot(downsel_freqs, len(downsel_freqs) * [0.0], ".k")
    plt.ylim(-1.0, 1.0)
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
if False:
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

    mmclrs = [kmclrs[ilab] for ilab in all_clusts]
    plt.figure(figsize=(3, 3))
    plt.scatter(all_medians, all_means, s=2, c=mmclrs, alpha=0.3)
    plt.xlim(-1.0, 1)
    plt.ylim(-1.0, 1)
    plt.xlabel("Median")
    plt.ylabel("Mean")
    plt.show()



## === cell 16
_SPECTRO_SLICE_CACHE = {}
_SPECTRO_SLICE_CACHE_ORDER = []
_SPECTRO_SLICE_CACHE_MAX = 5000  # bounded to keep memory under control


def _read_spectro_two_rows_400cols(spectro_file, r1, r2):
    """
    Return a (2, 400) numpy array for absolute rows [r1,r2] and columns [1..400]
    from the spectrogram parquet. Uses a bounded cache.

    Change (execution stability): if the parquet is missing/corrupt, return zeros.
    This keeps core feature logic identical when files are valid, but prevents crashes
    and downstream NaNs that can hurt KL or prevent a submission.
    """
    r1 = int(r1)
    r2 = int(r2)
    if r2 < r1:
        r1, r2 = r2, r1

    key = (spectro_file, r1, r2)
    if key in _SPECTRO_SLICE_CACHE:
        return _SPECTRO_SLICE_CACHE[key]

    try:
        pf = pq.ParquetFile(spectro_file)
    except Exception:
        out = np.zeros((2, 400), dtype=np.float64)
        _SPECTRO_SLICE_CACHE[key] = out
        _SPECTRO_SLICE_CACHE_ORDER.append(key)
        return out

    total_rows = pf.metadata.num_rows
    if total_rows <= 0:
        out = np.zeros((2, 400), dtype=np.float64)
        return out

    r1c = max(0, min(r1, total_rows - 1))
    r2c = max(0, min(r2, total_rows - 1))

    rg_sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
    cum = np.cumsum([0] + rg_sizes)  # cum[i] start row of row_group i

    def _rg_for_abs_row(r_abs):
        return int(np.searchsorted(cum, r_abs, side="right") - 1)

    rg1 = _rg_for_abs_row(r1c)
    rg2 = _rg_for_abs_row(r2c)

    all_schema_names = list(pf.schema.names)

    preferred_cols = [str(i) for i in range(1, 401)]
    use_preferred = all(c in all_schema_names for c in preferred_cols)

    def _select_400_cols(df):
        cols = list(df.columns)
        drop_candidates = {"time", "Time", "timestamp", "Timestamp"}
        cols2 = [c for c in cols if c not in drop_candidates]

        def _to_int_or_none(x):
            try:
                return int(x)
            except Exception:
                return None

        numeric_pairs = []
        for c in cols2:
            v = _to_int_or_none(c)
            if v is not None:
                numeric_pairs.append((v, c))
        if len(numeric_pairs) >= 400:
            numeric_pairs.sort(key=lambda t: t[0])
            sel = [c for _, c in numeric_pairs[:400]]
        else:
            sel = cols2[:400]
        if len(sel) < 400:
            sel = sel + [sel[-1]] * (400 - len(sel))
        return sel

    def _read_abs_row(pf_, rg, r_abs):
        if use_preferred:
            tab = pf_.read_row_group(rg, columns=preferred_cols)
            df = tab.to_pandas()
            local = int(r_abs - cum[rg])
            local = max(0, min(local, len(df) - 1))
            return df.iloc[local, :].to_numpy(dtype=np.float64, copy=False)

        tab = pf_.read_row_group(rg)  # all columns
        df = tab.to_pandas()
        local = int(r_abs - cum[rg])
        local = max(0, min(local, len(df) - 1))
        sel_cols = _select_400_cols(df)
        return df.loc[df.index[local], sel_cols].to_numpy(dtype=np.float64, copy=False)

    try:
        row1 = _read_abs_row(pf, rg1, r1c)
        row2 = _read_abs_row(pf, rg2, r2c)
        out = np.vstack([row1, row2]).astype(np.float64, copy=False)
    except Exception:
        out = np.zeros((2, 400), dtype=np.float64)

    if out.shape != (2, 400):
        out = np.zeros((2, 400), dtype=np.float64)

    _SPECTRO_SLICE_CACHE[key] = out
    _SPECTRO_SLICE_CACHE_ORDER.append(key)
    if len(_SPECTRO_SLICE_CACHE_ORDER) > _SPECTRO_SLICE_CACHE_MAX:
        old = _SPECTRO_SLICE_CACHE_ORDER.pop(0)
        _SPECTRO_SLICE_CACHE.pop(old, None)

    return out




## === cell 17
def assemble_features(meta_frame, traintest="train", smooth_width=5, SHOW_PLOT=True):
    """
    Create a dataframe of spectrogram features from the meta_frame rows.
    Will include clust_id (i.e, the y) if it is in the input meta_frame.

    Change (score + submission correctness): always carry through eeg_id (and only eeg_id)
    so feature rows can be verified/aligned 1:1 with meta rows, avoiding accidental
    row misalignment that can silently worsen KL.
    """
    freqs = np.array(range(100)) * 0.19525 + 0.59
    spect_trend = 150.0 / (1.0**2.3 + freqs ** (2.3))
    freqs[0] = 0.0
    freqs4 = np.array(4 * list(freqs))
    spect_trend4 = np.array(4 * list(spect_trend))

    baseinds = np.insert(
        np.arange(int((smooth_width - 1) / 2), 100, smooth_width), 0, 0
    )
    select_inds = np.concatenate(
        (baseinds, 100 + baseinds, 200 + baseinds, 300 + baseinds)
    )
    freqs4ds = freqs4[select_inds]

    if SHOW_PLOT:
        plt.figure(figsize=(10, 8))

    rows = []

    for n_done, irow in enumerate(meta_frame.index, start=1):
        this_row = meta_frame.loc[irow]
        spectro_id_str = str(int(this_row.spectrogram_id))
        spectro_file = (
            above_dir + traintest + "_spectrograms/" + spectro_id_str + ".parquet"
        )

        loc_offset = int(this_row.spectrogram_label_offset_seconds / 2)
        r1 = loc_offset + 149
        r2 = loc_offset + 150

        arr2x400 = _read_spectro_two_rows_400cols(spectro_file, r1, r2)
        middle4s = (arr2x400[0, :] + arr2x400[1, :]) / (2.0 * spect_trend4)

        middle4s = np.clip(middle4s, 0.001, 1000.0)
        middle4s = np.nan_to_num(middle4s, nan=0.001, posinf=0.001, neginf=0.001)

        spect_mean = float(np.mean(middle4s))
        spect_median = float(np.median(middle4s))

        the4means = np.empty(4, dtype=np.float64)
        the4medians = np.empty(4, dtype=np.float64)
        for ispec in range(4):
            ibeg = 100 * ispec
            iend = ibeg + 100
            seg = middle4s[ibeg:iend]
            the4means[ispec] = float(np.mean(seg))
            the4medians[ispec] = float(np.median(seg))

        middle4spre = np.log10(middle4s / spect_mean)

        middle4spre_s = pd.Series(middle4spre)
        middle4s_sm = middle4spre_s.rolling(
            smooth_width, min_periods=smooth_width, center=True, closed=None
        ).mean()
        for ioff in range(0, 400, 100):
            for ibin in range(int((smooth_width - 1) / 2)):
                middle4s_sm.iloc[ibin + ioff] = middle4spre_s.iloc[ibin + ioff]

        middle4sds = middle4s_sm.iloc[select_inds].to_numpy(dtype=np.float64)

        row_dict = {str(i): float(v) for i, v in enumerate(middle4sds)}
        row_dict["Mean"] = float(np.log10(spect_mean))
        row_dict["Median"] = float(np.log10(spect_median))
        the4means = np.log10(the4means)
        the4medians = np.log10(the4medians)
        for ispec in range(4):
            row_dict[the4chains[ispec] + "mean"] = float(the4means[ispec])
            row_dict[the4chains[ispec] + "median"] = float(the4medians[ispec])

        if "eeg_id" in meta_frame.columns:
            row_dict["eeg_id"] = int(this_row.eeg_id)

        if "clust_id" in meta_frame.columns:
            row_dict["clust_id"] = int(this_row.clust_id)

        rows.append(row_dict)

        if n_done % 200 == 0:
            print(f"... {n_done} done...")

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
                the4means,
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

    feats_frame = pd.DataFrame(rows)
    feats_frame.columns = feats_frame.columns.astype(str)

    if SHOW_PLOT:
        plt.savefig("middle4s_" + traintest + "_features.png")
        plt.show()

    return feats_frame




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
) & (train_meta.eeg_id % 4 > 1)

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
X = Xy_train_feats.drop(columns=["clust_id", "eeg_id"], errors="ignore").copy()
X.columns = X.columns.astype(str)
y = Xy_train_feats["clust_id"].astype(int)

BASE_FEATURE_COLS = list(X.columns)

lrmodel = LogisticRegression(
    C=1.0, max_iter=300, multi_class="ovr", n_jobs=-1, random_state=GLOBAL_SEED
).fit(X.reindex(columns=BASE_FEATURE_COLS), y)



## === cell 23
lrblur = 0.0

X_train_base = Xy_train_feats.drop(
    columns=["clust_id", "eeg_id"], errors="ignore"
).copy()
X_train_base.columns = X_train_base.columns.astype(str)
X_train_base = X_train_base.reindex(columns=BASE_FEATURE_COLS)
lrprobas_train = lrmodel.predict_proba(X_train_base)

Xy_train_wLRfeats = Xy_train_feats.copy()
for iadd in range(NUM_CLUSTS):
    Xy_train_wLRfeats["LRp" + str(iadd)] = lrprobas_train[:, iadd] + lrblur * (
        np.random.rand(len(lrprobas_train)) - 0.5
    )

X_valid_base = Xy_valid_feats.drop(
    columns=["clust_id", "eeg_id"], errors="ignore"
).copy()
X_valid_base.columns = X_valid_base.columns.astype(str)
X_valid_base = X_valid_base.reindex(columns=BASE_FEATURE_COLS)
lrprobas_valid = lrmodel.predict_proba(X_valid_base)

Xy_valid_wLRfeats = Xy_valid_feats.copy()
for iadd in range(NUM_CLUSTS):
    Xy_valid_wLRfeats["LRp" + str(iadd)] = lrprobas_valid[:, iadd] + lrblur * (
        np.random.rand(len(lrprobas_valid)) - 0.5
    )



## === cell 24
X = Xy_train_wLRfeats.drop(columns=["clust_id", "eeg_id"], errors="ignore").copy()
X.columns = X.columns.astype(str)

LR_FEATURE_COLS = list(X.columns)

Yprob_train = Xy_train_meta[HBA_probs].to_numpy(dtype=np.float64)
Yprob_train = np.clip(Yprob_train, 1e-8, 1.0)
Yprob_train = Yprob_train / Yprob_train.sum(axis=1, keepdims=True)

ave_oob = []
nfits = 5

rfmodels = []
for ifit in range(nfits):
    rfmodel = RandomForestRegressor(
        n_estimators=150,
        max_leaf_nodes=int(6 * NUM_CLUSTS),
        max_features=0.5,
        max_samples=0.7,
        oob_score=True,
        n_jobs=-1,
        random_state=GLOBAL_SEED + ifit,
        verbose=0,
    ).fit(X.reindex(columns=LR_FEATURE_COLS), Yprob_train)
    rfmodels.append(rfmodel)
    ave_oob.append(float(rfmodel.oob_score_))

print(
    "RF regressor ave OOB R^2 = {:.3f} +/- {:.3f}".format(
        np.mean(ave_oob), np.std(ave_oob)
    )
)




## === cell 25
def _rf_reg_predict_probs(model, Xdf, eps=1e-8):
    P = model.predict(Xdf).astype(np.float64, copy=False)
    if P.ndim == 1:
        P = P.reshape(-1, HBA_number)
    P = np.nan_to_num(
        P, nan=1.0 / HBA_number, posinf=1.0 / HBA_number, neginf=1.0 / HBA_number
    )
    P = np.clip(P, eps, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    return P


def _rf_reg_predict_probs_ensemble(models, Xdf, eps=1e-8):
    Ps = []
    for m in models:
        Ps.append(_rf_reg_predict_probs(m, Xdf, eps=eps))
    P = np.mean(np.stack(Ps, axis=0), axis=0)
    P = np.clip(P, eps, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    return P


X_train_reg = (
    Xy_train_wLRfeats.drop(columns=["clust_id", "eeg_id"], errors="ignore")
    .copy()
    .reindex(columns=LR_FEATURE_COLS)
)
P_train = _rf_reg_predict_probs_ensemble(rfmodels, X_train_reg)

sol_train = Xy_train_meta[HBA_probs].copy()
sub_train = pd.DataFrame(P_train, columns=HBA_probs)
print(
    "Train KL (direct 6-prob RF reg, ens): {:.4f}".format(
        kld_score(sol_train, sub_train)
    )
)

X_valid_reg = (
    Xy_valid_wLRfeats.drop(columns=["clust_id", "eeg_id"], errors="ignore")
    .copy()
    .reindex(columns=LR_FEATURE_COLS)
)
P_valid = _rf_reg_predict_probs_ensemble(rfmodels, X_valid_reg)
sol_valid = Xy_valid_meta[HBA_probs].copy()
sub_valid = pd.DataFrame(P_valid, columns=HBA_probs)
print(
    "Valid KL (direct 6-prob RF reg, ens): {:.4f}".format(
        kld_score(sol_valid, sub_valid)
    )
)




## === cell 26
def find_best_tamed_kl(solution, submission, pred_ids, step=0.02, eps=1e-8):
    """
    Adjust the taming fraction for each cluster center to optimize KL.

    Kept for compatibility with your existing pipeline, but not used for the
    final predictions after switching to direct 6-prob regression.
    """
    mean_all_probs = solution[HBA_votes].to_numpy(dtype=np.float64).mean(axis=0)
    mean_all_probs = np.clip(mean_all_probs, eps, 1.0)
    mean_all_probs = mean_all_probs / mean_all_probs.sum()

    tamed_fracs = 0.0 * np.ones(NUM_CLUSTS, dtype=np.float64)
    tamed_centers = clust_centers.copy()
    for iclust in range(NUM_CLUSTS):
        tamed_centers[iclust, :] = mean_all_probs

    best_fracs = tamed_fracs.copy()
    best_centers = tamed_centers.copy()
    best_kl_global = np.inf

    sol_probs = solution[HBA_votes].to_numpy(dtype=np.float64)
    sol_probs = np.clip(sol_probs, eps, 1.0)
    sol_probs = sol_probs / sol_probs.sum(axis=1, keepdims=True)

    for iclust in range(NUM_CLUSTS):
        last_kl = np.inf
        for this_frac in np.arange(0.0, 1.0 + 1e-12, step):
            tamed_fracs[iclust] = this_frac
            this_cent = (
                tamed_fracs[iclust] * clust_centers[iclust, :]
                + (1.0 - tamed_fracs[iclust]) * mean_all_probs
            )
            this_cent = np.clip(this_cent, eps, 1.0)
            this_cent = this_cent / this_cent.sum()
            tamed_centers[iclust, :] = this_cent

            sub_probs = tamed_centers[pred_ids, :]
            sub_probs = np.clip(sub_probs, eps, 1.0)
            sub_probs = sub_probs / sub_probs.sum(axis=1, keepdims=True)

            this_kl = float(
                np.mean(
                    np.sum(sol_probs * (np.log(sol_probs) - np.log(sub_probs)), axis=1)
                )
            )

            if this_kl < last_kl:
                last_kl = this_kl
                best_fracs = tamed_fracs.copy()
                best_centers = tamed_centers.copy()
                best_kl_global = min(best_kl_global, this_kl)
            else:
                tamed_fracs[iclust] = best_fracs[iclust]
                tamed_centers[iclust, :] = best_centers[iclust, :]
                break

    best_centers = np.clip(best_centers, eps, 1.0)
    best_centers = best_centers / best_centers.sum(axis=1, keepdims=True)
    return best_fracs, best_centers, best_kl_global




## === cell 27
def make_submission_valid(df, sample_sub, vote_cols, eps=1e-8):
    """
    Enforce strict Kaggle requirements: correct columns, finiteness, positivity,
    exact row order matching sample_submission, and row sums exactly 1.
    """
    out = df.copy()
    out["eeg_id"] = out["eeg_id"].astype(np.int64)
    for c in vote_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float64)

    order = sample_sub["eeg_id"].astype(np.int64).to_numpy()
    out = out.set_index("eeg_id").reindex(order).reset_index()

    probs = out[vote_cols].to_numpy(dtype=np.float64)
    probs = np.nan_to_num(
        probs,
        nan=1.0 / len(vote_cols),
        posinf=1.0 / len(vote_cols),
        neginf=1.0 / len(vote_cols),
    )

    probs = np.clip(probs, eps, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)

    out[vote_cols] = probs

    assert list(out.columns) == ["eeg_id"] + vote_cols
    assert len(out) == len(sample_sub)
    rs = out[vote_cols].to_numpy(dtype=np.float64).sum(axis=1)
    assert np.all(np.isfinite(rs))
    assert float(np.max(np.abs(rs - 1.0))) < 1e-9
    assert out.isna().sum().sum() == 0
    return out




## === cell 28
train_prior = train_meta[HBA_probs].to_numpy(dtype=np.float64)
train_prior = np.clip(train_prior, 1e-8, 1.0)
train_prior = train_prior / train_prior.sum(axis=1, keepdims=True)
EMPIRICAL_PRIOR = train_prior.mean(axis=0)
EMPIRICAL_PRIOR = np.clip(EMPIRICAL_PRIOR, 1e-8, 1.0)
EMPIRICAL_PRIOR = EMPIRICAL_PRIOR / EMPIRICAL_PRIOR.sum()

PRIOR_SMOOTH_ALPHA = 0.08  # was 0.02


def smooth_toward_prior(P, prior, alpha=0.02, eps=1e-8):
    P = np.asarray(P, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64).reshape(1, -1)
    Q = (1.0 - alpha) * P + alpha * prior
    Q = np.clip(Q, eps, 1.0)
    Q = Q / Q.sum(axis=1, keepdims=True)
    return Q




## === cell 29
sample_sub = pd.read_csv(above_dir + "sample_submission.csv")

Xy_test_feats = assemble_features(
    test_meta, traintest="test", smooth_width=SMOOTH_WIDTH, SHOW_PLOT=False
)

if "eeg_id" in Xy_test_feats.columns:
    assert np.array_equal(
        Xy_test_feats["eeg_id"].astype(np.int64).to_numpy(),
        test_meta["eeg_id"].astype(np.int64).to_numpy(),
    )

X_test_base = Xy_test_feats.drop(columns=["eeg_id"], errors="ignore").copy()
X_test_base.columns = X_test_base.columns.astype(str)
X_test_base = X_test_base.reindex(columns=BASE_FEATURE_COLS)

lrprobas = lrmodel.predict_proba(X_test_base)

Xy_test_wLRfeats = Xy_test_feats.copy()
for iadd in range(NUM_CLUSTS):
    Xy_test_wLRfeats["LRp" + str(iadd)] = lrprobas[:, iadd] + lrblur * (
        np.random.rand(len(lrprobas)) - 0.5
    )

X_test_reg = (
    Xy_test_wLRfeats.drop(columns=["eeg_id"], errors="ignore")
    .copy()
    .reindex(columns=LR_FEATURE_COLS)
)
test_probs = _rf_reg_predict_probs_ensemble(rfmodels, X_test_reg, eps=1e-8)

test_probs = smooth_toward_prior(
    test_probs, EMPIRICAL_PRIOR, alpha=PRIOR_SMOOTH_ALPHA, eps=1e-8
)

test_submit = pd.DataFrame({"eeg_id": test_meta["eeg_id"].astype(np.int64).to_numpy()})
for j, col in enumerate(HBA_votes):
    test_submit[col] = test_probs[:, j]

test_submit = test_submit[["eeg_id"] + HBA_votes]
test_submit = make_submission_valid(test_submit, sample_sub, HBA_votes, eps=1e-8)

print(test_submit.head())
print("submission shape:", test_submit.shape)

test_submit.to_csv(
    "submission.csv", header=True, index=False, na_rep="", float_format="%.6f"
)
print("Wrote submission.csv")
