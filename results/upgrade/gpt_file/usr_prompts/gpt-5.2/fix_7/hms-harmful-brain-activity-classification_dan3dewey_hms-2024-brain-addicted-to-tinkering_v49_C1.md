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

1.0214799308403597

# 6. Current score

1.3077

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.4357) has done: 'I fix the pipeline break caused by missing preprocessed CSVs by automatically falling back to on-the-fly feature assembly when those files are not present, while keeping the same feature logic and model training flow. I also make the KL-scoring helper consistent with the competition format (using only the 6 vote columns and clipping predictions away from 0) so it can be used safely for internal tuning without NaNs/infs. Finally, I ensure the trained models (`lrmodel1`, `rfmodel`, and `best_centers`) are always defined before test inference, and I enforce that the final submission probabilities are clipped and row-normalized to sum to 1 to avoid submission failure. All paths remain under the provided `../input/hms-harmful-brain-activity-classification/` dataset and the script writes `submission.csv`.'
- What this solution (achieved 1.2374) has done: 'To move the KL score down toward your target with minimal disruption, I’m keeping your feature extraction, KMeans clustering, LR+RF training, and “tamed centers” idea intact, but fixing two things that materially hurt score: (1) the “center normalization” bug (you normalize the first 6 rows instead of all clusters), and (2) using training-tuned `best_centers` at test time instead of the validation-tuned centers. I’m also making randomness deterministic (seeds for KMeans/RF and LR_BLUR noise) to stabilize results rather than chasing variance. These changes preserve your core logic and should consistently reduce KL from 1.4357 toward the 1.02 target without changing the modeling approach.'
- What this solution (achieved 1.23487) has done: 'Your current score (1.2374, lower is better) is still above the target (1.02148), so we should cautiously improve without changing your core approach (KMeans on label-prob vectors → spectrogram features → LR-proba features → RF → “tamed centers”). The biggest minimal-impact gain available is to make the taming optimization match the Kaggle metric more closely by weighting the KL objective by `total_vote` (more annotators = more reliable label distribution), while keeping the same taming search and model flow. I implement a weighted KL helper and use it only inside `find_best_tamed_kl()` (train/valid tuning), leaving your feature/model logic untouched. Finally, I keep the same submission post-processing (clip + renormalize) to guarantee a valid file.'
- What this solution (achieved 1.23975) has done: 'To move your KL score down toward the target with minimal disruption, I keep your full pipeline (KMeans on label-prob vectors → spectrogram features → LR-proba features → RF → “tamed centers”) unchanged, but make the taming step more metric-aligned and stable. Specifically, I (1) ensure the “mean_all_probs” used for taming is computed from your actual training meta (vote-weighted mean distribution) instead of a hard-coded vector, and (2) make `find_best_tamed_kl()` accept that mean as an argument so train/valid use consistent, data-derived priors. This typically reduces KL without changing the model/feature logic, because the shrinkage target becomes closer to the true marginal distribution. Submission writing, clipping, and row-normalization remain as-is to guarantee a valid CSV.'
- What this solution (achieved 1.25866) has done: 'Your current KL (1.23975, lower is better) is still above the target (1.02148), so we should make a small, metric-aligned improvement without changing your pipeline. The biggest low-risk issue is that your KMeans is fit on unweighted per-row label distributions, even though some rows have many more votes (more reliable) and the metric is defined on the vote distribution; weighting KMeans by `total_vote` keeps the same clustering approach but makes centers and assignments more consistent with the objective. I add `sample_weight=total_vote` to the KMeans fit only (no architecture/training loop changes), keep the existing “tamed centers” logic, and keep deterministic seeds and the same submission post-processing. This should typically reduce KL modestly and move you closer to the target band.'
- What this solution (achieved 1.3077) has done: 'Your current score (1.25866, lower-is-better) is still worse than the target (1.02148), so we should make a small, metric-aligned improvement without changing your pipeline. The most direct low-risk gain is to make the RandomForest training better reflect the KL metric by weighting samples with `total_vote` (more annotators → more reliable target distribution), while keeping the same model/feature logic and inference flow. I also keep the “tamed centers” tuning unchanged, but ensure the weighted objective is consistently used where it matters (RF fit) without altering architecture or adding early stopping. This should usually reduce KL modestly and move your score closer to the target band.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")




## === cell 1
def _resolve_comp_dir(preferred="../input/hms-harmful-brain-activity-classification/"):
    candidates = [
        preferred,
        "/kaggle/input/hms-harmful-brain-activity-classification/",
        "../input/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification/",
        "/kaggle/input/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification/",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
            os.path.join(c, "test.csv")
        ):
            return c if c.endswith("/") else (c + "/")
    return preferred




## === cell 2
def _csv_exists(path):
    return isinstance(path, str) and os.path.exists(path) and os.path.isfile(path)




## === cell 3
NUM_CLUSTS = 7  # 6 to 10

USE_PREPROC = True  # Read in saved meta and features frames
SMOOTH_WIDTH = 5  # Odd>1: 3,5,7,9,...
TRAIN_DOWNSEL = 23  # small values for code test; set to 1 to output all.
VALID_DOWNSEL = 47  #  "

USE_LR1 = True
LR1_C = 0.9
USE_LR2 = False
LR2_C = 0.9
LR_BLUR = 0.10

GLOBAL_SEED = 0

above_dir = "../input/hms-harmful-brain-activity-classification/"
above_dir_preproc = "../input/hms-2024-brain-data/"

above_dir = _resolve_comp_dir(above_dir)



## === cell 4
print("Using above_dir:", above_dir)
print("train.csv exists:", os.path.exists(os.path.join(above_dir, "train.csv")))
print("test.csv exists:", os.path.exists(os.path.join(above_dir, "test.csv")))



## === cell 5
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



## === cell 6
np.random.seed(GLOBAL_SEED)
rng = np.random.default_rng(GLOBAL_SEED)



## === cell 7
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




## === cell 8
def kld_score(solution, submission, eps=1e-15):
    """
    Calculate the average KL divergence score for the 6 target columns.
    - Clips submission away from 0/1 to avoid inf/nan.
    - Renormalizes submission rows to sum to 1 (submission requirement).
    """
    sol = solution[HBA_votes].astype(float).to_numpy()
    sub = submission[HBA_votes].astype(float).to_numpy()
    sub = np.clip(sub, eps, 1.0)
    sub = sub / np.clip(sub.sum(axis=1, keepdims=True), eps, None)
    sol = np.clip(sol, eps, 1.0)
    sol = sol / np.clip(sol.sum(axis=1, keepdims=True), eps, None)
    return float(np.mean(np.sum(sol * (np.log(sol) - np.log(sub)), axis=1)))


def kld_score_weighted(solution, submission, weights, eps=1e-15):
    sol = solution[HBA_votes].astype(float).to_numpy()
    sub = submission[HBA_votes].astype(float).to_numpy()

    sub = np.clip(sub, eps, 1.0)
    sub = sub / np.clip(sub.sum(axis=1, keepdims=True), eps, None)

    sol = np.clip(sol, eps, 1.0)
    sol = sol / np.clip(sol.sum(axis=1, keepdims=True), eps, None)

    w = np.asarray(weights, dtype=float).reshape(-1)
    w = np.clip(w, eps, None)
    w = w / np.sum(w)

    per_row = np.sum(sol * (np.log(sol) - np.log(sub)), axis=1)
    return float(np.sum(w * per_row))




## === cell 9
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




## === cell 10
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




## === cell 11
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
    print_every_nth = max([100, 100 * int(0.5 + len(meta_frame.index) / (100.0 * 15))])
    for irow in meta_frame.index:
        this_row = meta_frame.loc[irow]
        spectro_id_str = str(int(this_row.spectrogram_id))  # make sure int
        if spectro_id_str != last_spectro_id_str:
            if traintest != "test":
                spectro_file = (
                    above_dir + "train_spectrograms/" + spectro_id_str + ".parquet"
                )
            else:
                spectro_file = (
                    above_dir + "test_spectrograms/" + spectro_id_str + ".parquet"
                )
            pads_spectro = pads.dataset(spectro_file)
            this_spectro = pads_spectro.to_table().to_pandas()
        last_spectro_id_str = spectro_id_str
        loc_offset = int(this_row.spectrogram_label_offset_seconds / 2)
        middle4s = (
            this_spectro.iloc[loc_offset + 148, 1:]
            + this_spectro.iloc[loc_offset + 149, 1:]
            + this_spectro.iloc[loc_offset + 150, 1:]
            + this_spectro.iloc[loc_offset + 151, 1:]
        ) / (
            4.0 * spect_trend4
        )  # 4 copies of trend
        middle4s = np.clip(middle4s, 0.001, 1000.0)
        middle4s = middle4s.replace([np.nan, -np.inf, np.inf], 0.001)
        ratio4s = (
            this_spectro.iloc[loc_offset + 148, 1:]
            + this_spectro.iloc[loc_offset + 149, 1:]
            + this_spectro.iloc[loc_offset + 150, 1:]
            + this_spectro.iloc[loc_offset + 151, 1:]
        ) / (
            this_spectro.iloc[loc_offset + 149 - 56, 1:]
            + this_spectro.iloc[loc_offset + 149 - 40, 1:]
            + this_spectro.iloc[loc_offset + 149 - 24, 1:]
            + this_spectro.iloc[loc_offset + 149 + 56, 1:]
            + this_spectro.iloc[loc_offset + 149 + 40, 1:]
            + this_spectro.iloc[loc_offset + 149 + 24, 1:]
        )
        ratio4s = np.clip((6.0 / 4.0) * ratio4s, 0.01, 100.0)
        ratio4s = ratio4s.replace([np.nan, -np.inf, np.inf], 1.0)
        ratio4spre = np.log10(ratio4s)
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
        ratio4s = ratio4spre.rolling(
            smooth_width, min_periods=smooth_width, center=True, closed=None
        ).mean()
        for ioff in range(0, 400, 100):
            for ibin in range(int((smooth_width - 1) / 2)):  # assumes width is odd
                middle4s[ibin + ioff] = middle4spre[ibin + ioff]
                ratio4s[ibin + ioff] = ratio4spre[ibin + ioff]
        baseinds = np.insert(
            np.arange(int((smooth_width - 1) / 2), 100, smooth_width), 0, 0
        )
        select_inds = np.concatenate(
            (baseinds, 100 + baseinds, 200 + baseinds, 300 + baseinds)
        )
        freqs4ds = freqs4[select_inds]
        middle4sds = middle4s[select_inds]
        ratio4sds = ratio4s[select_inds]
        middle_feats = middle4sds.to_frame().T
        ratio_feats = ratio4sds.to_frame().T
        oldcols = ratio_feats.columns
        newcols = []
        for oldcol in oldcols:
            newcols.append("r" + oldcol)
        ratio_feats.columns = newcols
        these_feats = pd.concat([middle_feats, ratio_feats], axis=1)
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
        if len(feats_frame) % print_every_nth == 0:
            print("... {} done...".format(len(feats_frame)))
        if SHOW_PLOT:
            this_clr = "blue"
            title_start = "Middle-8s Feature Values of " + traintest
            plt.title(title_start + " (smooth={})".format(smooth_width))
            if "clust_id" in meta_frame.columns:
                this_clust = this_row.clust_id
                this_clr = kmclrs[this_clust]
                plt.title(
                    title_start
                    + " (color-coded by the "
                    + "{} clusters, smooth={})".format(num_clusts, smooth_width)
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
        plt.savefig("middle8s_" + traintest + "_features.png")
        plt.show()
    return feats_frame.reset_index().drop(columns=["index"])




## === cell 12
def _compute_mean_all_probs_from_meta(meta_df):
    """
    Minimal score-improvement change:
    Use the vote-weighted marginal label distribution from the actual training meta
    as the shrinkage target for taming, instead of a hard-coded vector.
    This better matches the competition KL objective and usually reduces KL.
    """
    w = meta_df["total_vote"].astype(float).to_numpy()
    w = np.clip(w, 1e-12, None)
    probs = meta_df[HBA_probs].astype(float).to_numpy()
    mean_probs = (w[:, None] * probs).sum(axis=0) / w.sum()
    mean_probs = np.clip(mean_probs, 1e-15, 1.0)
    mean_probs = mean_probs / mean_probs.sum()
    return mean_probs




## === cell 13
def find_best_tamed_kl(mean_all_probs):
    """
    Adjust the taming fraction for each cluster center to optimize KL
    Assumed inputs in environment:
        submission, pred_ids, solution
    Assumed useful values available:
        clust_centers, NUM_CLUSTS, HBA_number, HBA_votes

    Uses vote-weighted KL when solution has a 'total_vote' column (more metric-aligned).

    mean_all_probs: data-derived marginal distribution used as shrinkage target.
    """
    mean_all_probs = np.asarray(mean_all_probs, dtype=float).reshape(-1)
    mean_all_probs = np.clip(mean_all_probs, 1e-15, 1.0)
    mean_all_probs = mean_all_probs / mean_all_probs.sum()

    tamed_fracs = 0.0 * np.ones(NUM_CLUSTS)
    tamed_centers = clust_centers.copy()
    for iclust in range(NUM_CLUSTS):
        tamed_centers[iclust, :] = mean_all_probs

    if isinstance(solution, pd.DataFrame) and ("total_vote" in solution.columns):
        kl_weights = solution["total_vote"].astype(float).to_numpy()
    else:
        kl_weights = np.ones(len(solution), dtype=float)

    for iclust in range(NUM_CLUSTS):
        last_kl = 10.0
        for this_frac in np.arange(0.03, 1.00, 0.05):  # 0.03--0.98
            tamed_fracs[iclust] = this_frac
            this_cent = (
                tamed_fracs[iclust] * clust_centers[iclust, :]
                + (1.0 - tamed_fracs[iclust]) * mean_all_probs
            )
            tamed_centers[iclust, :] = this_cent

            for iprob in range(HBA_number):
                this_col_probs = tamed_centers[:, iprob]
                submission[HBA_votes[iprob]] = this_col_probs[pred_ids]

            this_kl = kld_score_weighted(solution, submission, weights=kl_weights)
            if this_kl < last_kl:
                best_fracs = tamed_fracs.copy()
                best_centers = tamed_centers.copy()
                last_kl = this_kl
            else:
                tamed_fracs[iclust] = best_fracs[iclust]
                tamed_centers[iclust, :] = best_centers[iclust, :]
                break
    return best_fracs, best_centers




## === cell 14
SHOW_CLUSTER_PLOTS = False
SHOW_SPECTRA_PLOTS = False



## === cell 15
train_meta, test_meta = read_hms_meta()



## === cell 16
plt.figure(figsize=(5, 2))
plt.hist(train_meta["spectrogram_sub_id"], bins=55, log=True)
plt.title("Histogram of spectrogram_sub_id")
plt.show()

plt.figure(figsize=(5, 2))
plt.hist(train_meta["eeg_sub_id"], bins=55, log=True)
plt.title("Histogram of eeg_sub_id")
plt.show()



## === cell 17
assert all(
    c in train_meta.columns for c in HBA_probs
), "Missing *_prob columns in train_meta."



## === cell 18
num_clusts = NUM_CLUSTS  # 6 to 10

clust_rows_bool = (
    (train_meta.eeg_sub_id < 33 + 1) & (train_meta.eeg_sub_id % 5 == 3)
) | (  # id=3,8,13,18,23,28,33
    train_meta.eeg_sub_id == 0
) & (
    (train_meta.eeg_id % 23) % 8 > 1
)  # include odd and even eeg_ids
sum(clust_rows_bool)



## === cell 19
valid_rows_bool = (
    (train_meta.eeg_sub_id < 44 + 1) & (train_meta.eeg_sub_id % 19 == 6)
) | (  # id=6,25,44
    train_meta.eeg_sub_id == 0
) & (
    (train_meta.eeg_id % 23) % 8 < 2
)  # include odd and even eeg_ids
sum(valid_rows_bool)



## === cell 20
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
plt.hist(train_meta.loc[clust_rows_bool, "total_vote"], bins=55, log=True)
plt.title("Histogram of total_vote in HBA samples clustered")
plt.show()

prob_array = np.array(prob_vectors)

kmeans = KMeans(
    n_clusters=num_clusts,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=GLOBAL_SEED,
)

kmeans.fit(
    prob_array,
    sample_weight=train_meta.loc[clust_rows_bool, "total_vote"]
    .astype(float)
    .to_numpy(),
)

clust_centers = kmeans.cluster_centers_

for iclust in range(num_clusts):
    clust_centers[iclust, :] = clust_centers[iclust, :] / np.sum(
        clust_centers[iclust, :]
    )

print("cluster centers:")
print(clust_centers)

iclust_of_order = []
for icol in range(HBA_number):
    iclust_of_order.append(np.argmax(clust_centers[:, icol]))
clust_by_max = np.argsort(-1 * np.max(clust_centers, axis=1))
for iord in range(HBA_number, num_clusts):
    iclust_of_order.append(clust_by_max[iord])

kmnames = HBA_expert_names.copy()
for ihyb in range(1, (num_clusts - HBA_number) + 1):
    kmnames.append("Hybrid-" + str(ihyb))



## === cell 21
train_meta["clust_id"] = kmeans.predict(np.array(train_meta[HBA_probs]))

all_probs = train_meta[HBA_probs]
all_ids = train_meta["clust_id"]

if SHOW_CLUSTER_PLOTS:
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
        "brown",
        "gray",
        "cyan",
        "magenta",
    ]

clust_counts = train_meta.clust_id.value_counts()

iorder_of_clust = num_clusts * [-1]
for iord, iclust in enumerate(iclust_of_order):
    iorder_of_clust[iclust] = iord
    if SHOW_CLUSTER_PLOTS:
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



## === cell 22
assert all(i >= 0 for i in iorder_of_clust), "Invalid cluster ordering map."



## === cell 23
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




## === cell 24
def _load_or_build_xy(
    meta_df, meta_path, feats_path, traintest, downsel, smooth_width, show_plot
):
    if USE_PREPROC and _csv_exists(meta_path) and _csv_exists(feats_path):
        meta = pd.read_csv(meta_path)
        feats = pd.read_csv(feats_path)
        meta["clust_id"] = kmeans.predict(np.array(meta[HBA_probs]))
        feats["clust_id"] = meta["clust_id"]
        return meta, feats, True
    meta = (meta_df)[::downsel].copy()
    meta = meta.reset_index().drop(columns=["index"])
    print(f"Number of samples used for {traintest} =", len(meta))
    feats = assemble_features(
        meta, traintest=traintest, smooth_width=smooth_width, SHOW_PLOT=show_plot
    )
    return meta, feats, False




## === cell 25
smooth_width = 9  # Here smooth_width is just for looking,

spectro_meta = train_meta[clust_rows_bool].copy()
every_nth = 19

freqs = np.array(range(100)) * 0.19525 + 0.59
downsel_freqs = np.insert(
    freqs[int((smooth_width - 1) / 2) : 100 : smooth_width], 0, freqs[0]
)

spect_trend = 150.0 / (1.0**2.3 + freqs ** (2.3))



## === cell 26
if SHOW_SPECTRA_PLOTS:
    if NUM_CLUSTS < 10:
        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(3, 3, hspace=0, wspace=0)
        (ax0, ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8) = gs.subplots(
            sharex="col", sharey="row"
        )
        pltaxs = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
    else:
        fig = plt.figure(figsize=(10, 13.5))
        gs = fig.add_gridspec(4, 3, hspace=0, wspace=0)
        (ax0, ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8), (ax9, ax10, ax11) = (
            gs.subplots(sharex="col", sharey="row")
        )
        pltaxs = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11]
    all_medians = []
    all_means = []
    all_clusts = []
    print(
        "Plotting {} x 4 processed spectra".format(int(len(spectro_meta) / every_nth))
    )
    print_every_nth = max(
        [100, 100 * int(0.5 + (len(spectro_meta) / every_nth) / (100.0 * 15))]
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
                    this_spectro.iloc[loc_offset + 148, ibeg:iend]
                    + this_spectro.iloc[loc_offset + 149, ibeg:iend]
                    + this_spectro.iloc[loc_offset + 150, ibeg:iend]
                    + this_spectro.iloc[loc_offset + 151, ibeg:iend]
                ) / (4.0 * spect_trend)
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
                pltaxs[iorder_of_clust[this_clust]].plot(
                    (freqs), middle4s, c=kmclrs[this_clust], lw=3, alpha=0.01
                )
        if (ilocrow / every_nth + 1) % print_every_nth == 0:
            print("... {} done...".format(int(ilocrow / every_nth) + 1))

    for iax in range(NUM_CLUSTS):
        pltaxs[iax].text(8.5, 0.8, kmnames[iax], fontsize=16)
        pltaxs[iax].plot([0.0, 20.0], [0.0, 0.0], c="black", lw=3, alpha=0.2)
        pltaxs[iax].plot(downsel_freqs, len(downsel_freqs) * [0.0], ".k")
        pltaxs[iax].set_ylim(-1.0, 1.0)  # log already taken
        pltaxs[iax].set_xlim(0.0, 20.5)
        if (iax == 0) or (iax == 3) or (iax == 6) or (iax == 9):
            pltaxs[iax].set_ylabel("log10[ Spectrum /reference /mean" + " & smoothed]")
        if (iax == 6) or (iax == 7) or (iax == 8):
            pltaxs[iax].set_xlabel("Frequency (Hz)")

    fig.suptitle(
        "Spectra of middle 8s (color-coded by the {} clusters, smooth={})".format(
            num_clusts, smooth_width
        )
    )
    plt.savefig("middle8s_spectra.png")
    plt.show()
else:
    all_means, all_medians, all_clusts = [], [], []



## === cell 27
if SHOW_SPECTRA_PLOTS:
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



## === cell 28
preproc_train_meta_path = os.path.join(above_dir_preproc, "Xy_train_meta_v47.csv")
preproc_train_feats_path = os.path.join(above_dir_preproc, "Xy_train_feats_v47.csv")
preproc_valid_meta_path = os.path.join(above_dir_preproc, "Xy_valid_meta_v47.csv")
preproc_valid_feats_path = os.path.join(above_dir_preproc, "Xy_valid_feats_v47.csv")



## === cell 29
if USE_PREPROC and (not os.path.exists(above_dir_preproc)):
    print(
        "Preproc directory not found; falling back to on-the-fly feature assembly:",
        above_dir_preproc,
    )
    USE_PREPROC = False



## === cell 30
Xy_train_meta, Xy_train_feats, used_preproc_train = _load_or_build_xy(
    meta_df=train_meta[clust_rows_bool],
    meta_path=preproc_train_meta_path,
    feats_path=preproc_train_feats_path,
    traintest="train",
    downsel=TRAIN_DOWNSEL,
    smooth_width=SMOOTH_WIDTH,
    show_plot=False,
)
print("Used preprocessed train frames:", used_preproc_train)



## === cell 31
Xy_train_feats



## === cell 32
assert "clust_id" in Xy_train_feats.columns, "Training features missing clust_id."
assert len(Xy_train_feats) == len(Xy_train_meta), "Train meta/feats length mismatch."



## === cell 33
Xy_valid_meta, Xy_valid_feats, used_preproc_valid = _load_or_build_xy(
    meta_df=train_meta[valid_rows_bool],
    meta_path=preproc_valid_meta_path,
    feats_path=preproc_valid_feats_path,
    traintest="validation",
    downsel=VALID_DOWNSEL,
    smooth_width=SMOOTH_WIDTH,
    show_plot=False,
)
print("Used preprocessed valid frames:", used_preproc_valid)



## === cell 34
Xy_valid_feats



## === cell 35
assert "clust_id" in Xy_valid_feats.columns, "Validation features missing clust_id."
assert len(Xy_valid_feats) == len(Xy_valid_meta), "Valid meta/feats length mismatch."



## === cell 36
X = Xy_train_feats.drop(columns=["clust_id"])
y = Xy_train_feats.clust_id

Xlr = X.drop(columns=X.columns[-10:])

if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : int(len(Xlr.columns) / 2)]
    lrmodel1 = LogisticRegression(
        penalty="l1",
        C=LR1_C,
        solver="saga",
        max_iter=1500,
        multi_class="multinomial",  # ovr or multinomial
        n_jobs=-1,
        random_state=GLOBAL_SEED,  # stability
    ).fit(Xlr1, y)

    plt.figure(figsize=(8, 4))
    plt.plot(lrmodel1.coef_.T, "-", alpha=0.5)
    plt.plot(lrmodel1.coef_.T, ".", alpha=1.0)
    plt.title("Logistic Regression coefficients (colored by cluster)")
    plt.show()

    print("\nLR model score for X,y = {:.1f}%\n".format(100 * lrmodel1.score(Xlr1, y)))

if USE_LR2:
    Xlr2 = Xlr.iloc[:, int(len(Xlr.columns) / 2) :]
    lrmodel2 = LogisticRegression(
        penalty="l1",
        C=LR2_C,
        solver="saga",
        max_iter=1500,
        multi_class="multinomial",  # ovr or multinomial
        n_jobs=-1,
        random_state=GLOBAL_SEED,  # stability
    ).fit(Xlr2, y)

    plt.figure(figsize=(8, 4))
    plt.plot(lrmodel2.coef_.T, "-", alpha=0.5)
    plt.plot(lrmodel2.coef_.T, ".", alpha=1.0)
    plt.title("Logistic Regression coefficients (colored by cluster)")
    plt.show()

    print("\nLR model score for X,y = {:.1f}%\n".format(100 * lrmodel2.score(Xlr2, y)))



## === cell 37
Xy_train_wLRfeats = Xy_train_feats.copy()
X = Xy_train_feats.drop(columns=["clust_id"])
Xlr = X.drop(columns=X.columns[-10:])
if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : int(len(Xlr.columns) / 2)]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    for iadd in range(NUM_CLUSTS):
        Xy_train_wLRfeats["lr" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            rng.random(len(lrprobas)) - 0.5
        )
    plt.figure(figsize=(7, 2.5))
    plt.hist(np.max(lrmodel1.predict_proba(Xlr1), axis=1), bins=50)
    plt.xlim(-0.5 * LR_BLUR, 1.01)
    plt.title("Training max lrmodel1 proba values (pre-blur)")
    plt.show()
if USE_LR2:
    Xlr2 = Xlr.iloc[:, int(len(Xlr.columns) / 2) :]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    for iadd in range(NUM_CLUSTS):
        Xy_train_wLRfeats["rlr" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            rng.random(len(lrprobas)) - 0.5
        )
    plt.figure(figsize=(7, 2.5))
    plt.hist(np.max(lrmodel2.predict_proba(Xlr2), axis=1), bins=50)
    plt.xlim(-0.5 * LR_BLUR, 1.01)
    plt.title("Training max lrmodel2 proba values (pre-blur)")
    plt.show()

Xy_valid_wLRfeats = Xy_valid_feats.copy()
X = Xy_valid_feats.drop(columns=["clust_id"])
Xlr = X.drop(columns=X.columns[-10:])
if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : int(len(Xlr.columns) / 2)]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    for iadd in range(NUM_CLUSTS):
        Xy_valid_wLRfeats["lr" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            rng.random(len(lrprobas)) - 0.5
        )
    plt.figure(figsize=(7, 2.5))
    plt.hist(np.max(lrmodel1.predict_proba(Xlr1), axis=1), bins=50)
    plt.xlim(-0.5 * LR_BLUR, 1.01)
    plt.title("Validation max lrmodel1 proba values (pre-blur)")
    plt.show()
if USE_LR2:
    Xlr2 = Xlr.iloc[:, int(len(Xlr.columns) / 2) :]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    for iadd in range(NUM_CLUSTS):
        Xy_valid_wLRfeats["rlr" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            rng.random(len(lrprobas)) - 0.5
        )
    plt.figure(figsize=(7, 2.5))
    plt.hist(np.max(lrmodel2.predict_proba(Xlr2), axis=1), bins=50)
    plt.xlim(-0.5 * LR_BLUR, 1.01)
    plt.title("Validation max lrmodel2 proba values (pre-blur)")
    plt.show()



## === cell 38
if USE_LR1:
    assert "lrmodel1" in globals(), "lrmodel1 was not trained."
if USE_LR2:
    assert "lrmodel2" in globals(), "lrmodel2 was not trained."



## === cell 39
X = Xy_train_wLRfeats.drop(columns=["clust_id"])
y = Xy_train_wLRfeats.clust_id

rf_sample_weight = np.clip(
    Xy_train_meta["total_vote"].astype(float).to_numpy(), 1e-12, None
)

ave_oob = []
nfits = 3
for ifit in range(nfits):
    rfmodel = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=9,
        max_features=0.20,
        max_samples=0.8,
        oob_score=True,
        class_weight="balanced_subsample",
        n_jobs=-1,
        verbose=0,
        random_state=GLOBAL_SEED + ifit,
    ).fit(X, y, sample_weight=rf_sample_weight)
    ave_oob.append(rfmodel.oob_score_)

sort_inds = rfmodel.feature_importances_.argsort()
plt.figure(figsize=(4, 7))
plt.barh(rfmodel.feature_names_in_[sort_inds], rfmodel.feature_importances_[sort_inds])
plt.ylim(len(sort_inds) - 40, len(sort_inds) + 0.2)
plt.title("Feature Importances (top 40)")
plt.show()

print(
    "\nRF model ave OOB score = {:.1f}% +/- {:.1f}".format(
        100 * np.mean(ave_oob), 100 * np.std(ave_oob)
    )
)

print("\nRF model score for X,y = {:.1f}%\n".format(100 * rfmodel.score(X, y)))



## === cell 40
Xy_train_meta["pred_id"] = rfmodel.predict(Xy_train_wLRfeats.drop(columns=["clust_id"]))

maxprobs_train = np.max(rfmodel.predict_proba(X), axis=1)
plt.figure(figsize=(7, 2.5))
plt.hist(maxprobs_train, bins=50)
plt.xlim(0.0, 1.0)
plt.title("Histogram of max(proba) for model-training samples")
plt.show()

solution = Xy_train_meta[["eeg_id", "total_vote"] + HBA_votes]
for col_pre in HBA_names:
    solution.loc[:, col_pre + "_vote"] = Xy_train_meta[col_pre + "_prob"]

submission = solution[["eeg_id"] + HBA_votes].copy()
pred_ids = Xy_train_meta["pred_id"]  # putting clust_id gives the ideal value

mean_all_probs = _compute_mean_all_probs_from_meta(train_meta)

best_fracs, best_centers = find_best_tamed_kl(mean_all_probs=mean_all_probs)

for iprob in range(HBA_number):
    this_col_probs = best_centers[:, iprob]
    submission[HBA_votes[iprob]] = this_col_probs[pred_ids]
this_kl = kld_score(solution.drop(columns=["total_vote"]), submission)
print("Tamed fractions:\n", best_fracs, "\nand centers:\n", best_centers)
print("\nKL from tamed centers: {:.4f}".format(this_kl))



## === cell 41
assert "best_centers" in globals(), "best_centers not computed."



## === cell 42
Xy_valid_meta["pred_id"] = rfmodel.predict(Xy_valid_wLRfeats.drop(columns=["clust_id"]))

maxprobs_valid = np.max(
    rfmodel.predict_proba(Xy_valid_wLRfeats.drop(columns=["clust_id"])), axis=1
)
plt.figure(figsize=(7, 2.5))
plt.hist(maxprobs_valid, bins=50)
plt.xlim(0.0, 1.0)
plt.title("Histogram of max(proba) for validation samples")
plt.show()

solution = Xy_valid_meta[["eeg_id", "total_vote"] + HBA_votes]
for col_pre in HBA_names:
    solution.loc[:, col_pre + "_vote"] = Xy_valid_meta[col_pre + "_prob"]

submission = solution[["eeg_id"] + HBA_votes].copy()
pred_ids = Xy_valid_meta["pred_id"]

mean_all_probs = _compute_mean_all_probs_from_meta(train_meta)

best_fracs, best_centers_valid = find_best_tamed_kl(mean_all_probs=mean_all_probs)

for iprob in range(HBA_number):
    this_col_probs = best_centers_valid[:, iprob]
    submission[HBA_votes[iprob]] = this_col_probs[pred_ids]
this_kl = kld_score(solution.drop(columns=["total_vote"]), submission)
print("Tamed fractions:\n", best_fracs, "\nand centers:\n", best_centers_valid)
print("\nKL from tamed centers: {:.4f}".format(this_kl))




## === cell 43
def _postprocess_probs(df, prob_cols, eps=1e-15):
    arr = df[prob_cols].astype(float).to_numpy()
    arr = np.clip(arr, eps, 1.0)
    arr = arr / np.clip(arr.sum(axis=1, keepdims=True), eps, None)
    df.loc[:, prob_cols] = arr
    return df




## === cell 44
if False:
    X = Xy_train_wLRfeats.drop(columns=["clust_id"])
    y = Xy_train_wLRfeats.clust_id
    solution = Xy_valid_meta[["eeg_id"] + HBA_votes]
    for col_pre in HBA_names:
        solution.loc[:, col_pre + "_vote"] = Xy_valid_meta[col_pre + "_prob"]
    submission = solution.copy()

if False:  # for scan_this in [1,1,2,2,3,3,4,4,5,5,6,6]:
    rfmodel = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=3,
        max_features=0.20,
        max_samples=0.8,
        oob_score=True,
        class_weight="balanced_subsample",
        n_jobs=-1,
        verbose=0,
        random_state=GLOBAL_SEED,
    ).fit(X, y)
    print("\nRF model score for X,y = {:.1f}%".format(100 * rfmodel.score(X, y)))
    Xy_valid_meta["pred_id"] = rfmodel.predict(
        Xy_valid_wLRfeats.drop(columns=["clust_id"])
    )
    pred_ids = Xy_valid_meta["pred_id"]
    best_fracs, best_centers = find_best_tamed_kl(mean_all_probs=mean_all_probs)
    for iprob in range(HBA_number):
        this_col_probs = best_centers[:, iprob]
        submission[HBA_votes[iprob]] = this_col_probs[pred_ids]
    this_kl = kld_score(solution, submission)
    print("KL from tamed centers: {:.4f}".format(this_kl))



## === cell 45
assert hasattr(rfmodel, "predict"), "rfmodel is not trained."



## === cell 46
assert (
    "spectrogram_label_offset_seconds" in test_meta.columns
), "test_meta missing spectrogram_label_offset_seconds."



## === cell 47
final_centers = (
    best_centers_valid if "best_centers_valid" in globals() else best_centers
)

Xy_test_feats = assemble_features(
    test_meta, traintest="test", smooth_width=SMOOTH_WIDTH, SHOW_PLOT=False
)

Xy_test_wLRfeats = Xy_test_feats.copy()
Xlr = Xy_test_feats.drop(columns=Xy_test_feats.columns[-10:])
if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : int(len(Xlr.columns) / 2)]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    for iadd in range(NUM_CLUSTS):
        Xy_test_wLRfeats["lr" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            rng.random(len(lrprobas)) - 0.5
        )
if USE_LR2:
    Xlr2 = Xlr.iloc[:, int(len(Xlr.columns) / 2) :]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    for iadd in range(NUM_CLUSTS):
        Xy_test_wLRfeats["rlr" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            rng.random(len(lrprobas)) - 0.5
        )

pred_ids = rfmodel.predict(Xy_test_wLRfeats)

test_submit = test_meta[["eeg_id"]].copy()
for new_col in HBA_votes:
    test_submit[new_col] = 1 / HBA_number
for iprob in range(HBA_number):
    this_col_probs = final_centers[:, iprob]
    test_submit[HBA_votes[iprob]] = this_col_probs[pred_ids]

test_submit = _postprocess_probs(test_submit, HBA_votes, eps=1e-15)

print(test_submit.head())

test_submit.to_csv(
    "submission.csv", header=True, index=False, na_rep="", float_format="%.6f"
)

print("Wrote submission.csv with shape:", test_submit.shape)
print(
    "Row-sum min/max:",
    test_submit[HBA_votes].sum(axis=1).min(),
    test_submit[HBA_votes].sum(axis=1).max(),
)
