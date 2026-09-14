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

2.390865

# 6. Current score

1.33763

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.33763) has done: 'Your current pipeline already computes sensible cluster-center class probabilities from train votes, but it throws that information away at test-time by assigning *random* clusters, which unnecessarily worsens KL divergence. I make the smallest change that uses the same KMeans model to assign each test row to a cluster using the only shared metadata available (patient_id and spectrogram_id), by learning a per-cluster “profile” over these IDs from train and then picking the most likely cluster for each test sample. This preserves your core logic (KMeans on vote-prob vectors; predicting via cluster centers) while replacing the randomness with a deterministic, data-driven mapping that should reduce your score toward the target. I also add a tiny probability clip + renormalization before saving to guarantee valid probabilities for Kaggle.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import pyarrow
import pyarrow.parquet as pq
import pyarrow.dataset as pads



## === cell 1
above_dir = "../input/hms-harmful-brain-activity-classification/"



## === cell 2
HBA_names = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
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
    print("Histogram of the votes entropies")
    plt.figure(figsize=(6, 3))
    plt.hist(train_meta["entropy"], bins=50, log=True)
    plt.title("Histogram of Vote Entropy (~ vote variation)")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(
        train_meta["total_vote"] + 0.7 * (np.random.rand(len(train_meta)) - 0.5),
        train_meta["entropy"] + 0.05 * (np.random.rand(len(train_meta)) - 0.5),
        s=3,
        alpha=0.02,
    )
    ref_ents = []
    for ispread in [1, 2, 3, 4, 5, 6]:
        spread_ent = np.log(ispread)
        ref_ents.append(spread_ent)
        plt.plot([ispread, 28], [spread_ent, spread_ent], lw=2, c="pink", alpha=0.5)
        plt.text(24.0, spread_ent + 0.03, "{} x p=1/{}".format(ispread, ispread))
    plt.title("Vote Entropy vs Number of Votes")
    plt.ylim(-0.05, 1.90)
    plt.ylabel("Entropy of the Votes")
    plt.xlabel("Number of Votes")
    plt.show()
    fmtstr = (len(ref_ents) - 1) * "{:.4f}, " + "{:.4f}"
    print(
        "          The entropy reference lines are at:", fmtstr.format(*ref_ents), "\n"
    )

    return train_meta, test_meta




## === cell 5
def prob_prob_scatter(name1, name2, probs2plot, clust_ids):
    """
    Make a prob1 vs prob2 scatter plot.
    Include an x at the cluster centers in chosen axes.
    Use sqrt scaling.
    External: HBA_probs, iHBA_of_expert[ ], clust_probs
    """
    kmclrs = 2 * ["red", "blue", "green", "black", "purple", "orange"]
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
    for iclust in range(0, len(clust_probs)):
        plt.plot(
            np.sqrt([clust_probs[iclust, ixax]]),
            np.sqrt([clust_probs[iclust, iyax]]),
            c=kmclrs[iclust],
            marker="x",
            markersize=15,
        )
    plt.xlabel("sqrt( " + name1 + " )")
    plt.ylabel("sqrt( " + name2 + " )")
    plt.show()




## === cell 6
train_meta, test_meta = read_hms_meta()



## === cell 7
test_meta



## === cell 8
prob_vectors = train_meta.loc[
    ((train_meta.eeg_sub_id < 50) & (train_meta.eeg_sub_id % 7 == 0)), HBA_probs
]

print("\nUsing {} HBA samples for clustering.".format(len(prob_vectors)))
plt.figure(figsize=(6, 3))
plt.hist(train_meta.loc[prob_vectors.index, "total_vote"], bins=55, log=True)
plt.title("Histogram of total_vote in HBA samples clustered")
plt.show()

prob_array = np.array(prob_vectors)
kmeans = KMeans(
    n_clusters=9, init="k-means++", n_init=10, max_iter=300, random_state=None
)
kmeans.fit(prob_array)

clust_probs = kmeans.cluster_centers_
for iclust in range(len(clust_probs)):
    clust_probs[iclust, :] = clust_probs[iclust, :] / np.sum(clust_probs[iclust, :])

train_meta["clust_id"] = kmeans.predict(np.array(train_meta[HBA_probs]))



## === cell 9
all_probs = train_meta[HBA_probs]
all_ids = train_meta["clust_id"]

prob_prob_scatter("LPD", "GRDA", all_probs, all_ids)

prob_prob_scatter("LRDA", "Other", all_probs, all_ids)

prob_prob_scatter("Seizure", "GPD", all_probs, all_ids)

print("The centers for {} clusters:".format(len(clust_probs)))
print(clust_probs)

max_probs = np.argmax(clust_probs, axis=1)
clust_names = []
for iclust in range(len(clust_probs)):
    clust_names.append(HBA_names[max_probs[iclust]])
print(clust_names)

print("\nNumber of HBA samples in each cluster (all samples assigned):")
print(train_meta.clust_id.value_counts())



## === cell 10
solution_train = train_meta[["eeg_id"] + HBA_votes]
for col_pre in HBA_names:
    solution_train.loc[:, col_pre + "_vote"] = train_meta[col_pre + "_prob"]

submission_train = solution_train.copy()
clust_ids = train_meta["clust_id"]
if False:
    clust_ids = np.random.choice(9, size=len(train_meta), replace=True, p=None)
for iprob in range(len(clust_probs[0])):
    this_col_probs = clust_probs[:, iprob]
    submission_train[HBA_votes[iprob]] = this_col_probs[clust_ids]

print(
    "Score if HBA samples are assigned cluster prob.s:",
    np.round(kld_score(solution_train, submission_train), 4),
)



## === cell 11
plt.figure(figsize=(6, 3))
plt.hist(train_meta["spectrogram_sub_id"], bins=55, log=True)
plt.show()

plt.figure(figsize=(6, 3))
plt.hist(train_meta["eeg_sub_id"], bins=55, log=True)
plt.show()



## === cell 12
train_meta[(train_meta["spectrogram_sub_id"] == 0) & (train_meta["total_vote"] > 3)]



## === cell 13
spectro_id_str = "2145546675"  # "3452193"   #"924234"
spectro_file = above_dir + "train_spectrograms/" + spectro_id_str + ".parquet"
pads_spectro = pads.dataset(spectro_file)

this_spectro = pads_spectro.to_table().to_pandas()



## === cell 14
for eloc in [0, 1, 2, 3]:
    plt.figure(figsize=(6, 2))
    plt.plot(
        this_spectro.iloc[150 - 2, eloc * 100 + 1 : eloc * 100 + 100 + 1].values,
        c="orange",
    )
    plt.plot(
        this_spectro.iloc[150, eloc * 100 + 1 : eloc * 100 + 100 + 1].values, c="black"
    )
    plt.plot(
        this_spectro.iloc[150 + 2, eloc * 100 + 1 : eloc * 100 + 100 + 1].values,
        c="lightblue",
    )
    plt.xlim(0.0, 50.0)
    plt.show()




## === cell 15
def assign_test_clusters_from_metadata(train_meta, test_meta, n_clusters=9, alpha=1.0):
    """
    Learn per-cluster categorical distributions over patient_id and spectrogram_id from train,
    then assign each test row to the cluster with maximum posterior score.

    alpha: Laplace smoothing strength (>=0). Small smoothing prevents zero-prob issues.
    """
    prior = (
        train_meta["clust_id"]
        .value_counts()
        .reindex(range(n_clusters), fill_value=0)
        .astype(float)
    )
    prior = (prior + alpha) / (prior.sum() + alpha * n_clusters)

    pt_ct = pd.crosstab(train_meta["clust_id"], train_meta["patient_id"])
    pt_ct = pt_ct.reindex(index=range(n_clusters), fill_value=0).astype(float)
    pt_denom = pt_ct.sum(axis=1).values.reshape(-1, 1)
    pt_like = (pt_ct + alpha) / (pt_denom + alpha * pt_ct.shape[1])

    sp_ct = pd.crosstab(train_meta["clust_id"], train_meta["spectrogram_id"])
    sp_ct = sp_ct.reindex(index=range(n_clusters), fill_value=0).astype(float)
    sp_denom = sp_ct.sum(axis=1).values.reshape(-1, 1)
    sp_like = (sp_ct + alpha) / (sp_denom + alpha * sp_ct.shape[1])

    pt_cols = pt_like.columns
    sp_cols = sp_like.columns

    pt_unseen = pt_like.mean(axis=1).values
    sp_unseen = sp_like.mean(axis=1).values

    test_clusters = np.zeros(len(test_meta), dtype=int)
    log_prior = np.log(np.clip(prior.values, 1e-12, 1.0))

    for i, row in enumerate(test_meta.itertuples(index=False)):
        pid = getattr(row, "patient_id")
        sid = getattr(row, "spectrogram_id")

        if pid in pt_cols:
            log_pt = np.log(np.clip(pt_like[pid].values, 1e-12, 1.0))
        else:
            log_pt = np.log(np.clip(pt_unseen, 1e-12, 1.0))

        if sid in sp_cols:
            log_sp = np.log(np.clip(sp_like[sid].values, 1e-12, 1.0))
        else:
            log_sp = np.log(np.clip(sp_unseen, 1e-12, 1.0))

        scores = log_prior + log_pt + log_sp
        test_clusters[i] = int(np.argmax(scores))

    return test_clusters




## === cell 16
test_submit = test_meta[["eeg_id"]].copy()
for new_col in HBA_votes:
    test_submit[new_col] = 1 / 6

test_clusts = assign_test_clusters_from_metadata(
    train_meta, test_meta, n_clusters=len(clust_probs), alpha=1.0
)

for iprob in range(len(clust_probs[0])):
    this_col_probs = clust_probs[:, iprob]
    test_submit[HBA_votes[iprob]] = this_col_probs[test_clusts]

probs = test_submit[HBA_votes].values.astype(float)
probs = np.clip(probs, 1e-8, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)
test_submit[HBA_votes] = probs

print(test_submit.head())

test_submit.to_csv(
    "submission.csv", header=True, index=False, na_rep="", float_format="%.6f"
)
print("Wrote submission.csv with shape:", test_submit.shape)
