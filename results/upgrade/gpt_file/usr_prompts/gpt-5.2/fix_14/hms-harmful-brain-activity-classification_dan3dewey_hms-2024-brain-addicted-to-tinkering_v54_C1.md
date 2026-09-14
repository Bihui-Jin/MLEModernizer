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

1.055544758302554

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.3662) has done: 'Your code currently doesn’t yield a Kaggle score mainly because it won’t reliably run end-to-end in this environment: it references preprocessed CSVs that aren’t present and has a `cell 0` header that doesn’t match the required execution order. I make minimal execution-unblocking changes: (1) keep `USE_PREPROC=True` as intent but auto-fallback if files don’t exist, (2) ensure the feature assembly loop is robust under pandas 2.2 by using positional tuple access, and (3) guarantee the final `submission.csv` matches `sample_submission.csv` order, has all required columns, and each row sums to 1 (with clipping/renorm). These changes preserve your core modeling logic (KMeans -> RF -> tamed centers) while ensuring a valid submission is always produced so you can obtain a score and move toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import pyarrow
import pyarrow.parquet as pq
import pyarrow.dataset as pads



## === cell 1
NUM_CLUSTS = 9  # 6 to 10

SMOOTH_WIDTH = 5  # Odd>1: 3,5,7,9,...
TRAIN_DOWNSEL = 1  # set to 1 to output all.
VALID_DOWNSEL = 1  # set to 1 to output all.
USE_PREPROC = (
    True  # Read in saved meta and features frames (fallback added if files missing)
)

USE_LR1 = False
LR1_C = 0.10  # smaller --> fewer non-zero coeff.s
USE_LR2 = False
LR2_C = 0.10
LR_BLUR = 0.10

DEFAULT_ABOVE_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"
ALT_ABOVE_DIR = "/kaggle/data/hms-harmful-brain-activity-classification/"
above_dir = DEFAULT_ABOVE_DIR if os.path.exists(DEFAULT_ABOVE_DIR) else ALT_ABOVE_DIR

above_dir_preproc = "../input/hms-2024-brain-data/"

SEED = 1234
np.random.seed(SEED)



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
    Calculate the average KL divergence score.
    Assumes solution/submission have the same probability columns.

    Stability: clip away from 0 to avoid NaNs/infs.
    """
    sumsum = 0.0
    for prob_col in solution.columns.values:
        p = np.clip(solution[prob_col].to_numpy(dtype=float), eps, 1.0)
        q = np.clip(submission[prob_col].to_numpy(dtype=float), eps, 1.0)
        sumsum += np.nansum(-1.0 * p * np.log(q / p))
    return sumsum / (len(solution))




## === cell 4
def read_hms_meta():
    """
    Read in the train.csv and test.csv files.
    Add total_vote, _prob columns, and vote entropy to train_meta.
    Add extra cols to test to allow the same processing as train.
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

    print("\nHistogram of the total votes")
    plt.figure(figsize=(6, 3))
    plt.hist(train_meta["total_vote"], bins=55, log=True)
    plt.title("Histogram of Total Votes")
    plt.savefig("hist_total_votes.png")
    plt.close()

    for col_pre in HBA_names:
        train_meta[col_pre + "_prob"] = (
            train_meta[col_pre + "_vote"] / train_meta["total_vote"]
        )

    print("Calculating voting entropy values ...")

    p = np.clip(train_meta[HBA_probs].to_numpy(dtype=float), 1.0e-8, 1.0)
    train_meta["entropy"] = np.nansum(p * (-1.0) * np.log(p), axis=1)

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
    plt.savefig(f"scatter_{name1}_{name2}.png")
    plt.close()
    return kmclrs




## === cell 6
from collections import OrderedDict


def _rolling_center_mean_1d(x, width):
    """Center rolling mean with min_periods=width, center=True, like pandas.Series.rolling(...).mean()."""
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if width <= 1 or n < width:
        return out
    half = (width - 1) // 2
    csum = np.cumsum(np.insert(x.astype(np.float64, copy=False), 0, 0.0))
    wsum = csum[width:] - csum[:-width]
    centers = np.arange(half, n - half, dtype=np.int64)
    out[centers] = wsum / float(width)
    return out


def assemble_features(meta_frame, traintest="train", smooth_width=5, SHOW_PLOT=True):
    """
    Create a dataframe of spectrogram features from the meta_frame rows.
    Will include clust_id (i.e, the y) if it is in the input meta_frame.
    Assumes these are available: above_dir, num_clusts

    Runtime fix: use itertuples(..., name=None) and positional access for pandas 2.2+ robustness.
    """
    freqs = np.array(range(100)) * 0.19525 + 0.59
    spect_trend = 150.0 / (1.0**2.3 + freqs ** (2.3))
    freqs[0] = 0.0  # separate the (unsmoothed) first bin from others
    spect_trend4 = np.array(4 * list(spect_trend)).astype(np.float64)

    if SHOW_PLOT:
        plt.figure(figsize=(10, 8))

    nrows = len(meta_frame.index)
    print_every_nth = max([100, 100 * int(0.5 + nrows / (100.0 * 15))])

    half = int((smooth_width - 1) / 2)
    baseinds = np.insert(np.arange(half, 100, smooth_width), 0, 0)
    select_inds = np.concatenate(
        (baseinds, 100 + baseinds, 200 + baseinds, 300 + baseinds)
    ).astype(np.int64)

    middle_cols = [str(i) for i in range(select_inds.shape[0])]
    ratio_cols = ["r" + str(i) for i in range(select_inds.shape[0])]
    extra_cols = (
        ["Mean", "Median"]
        + [c + "mean" for c in the4chains]
        + [c + "median" for c in the4chains]
    )
    feat_cols = middle_cols + ratio_cols + extra_cols

    out = np.empty((nrows, len(feat_cols)), dtype=np.float64)

    def _spectro_path(spectro_id_str):
        if traintest != "test":
            return above_dir + "train_spectrograms/" + spectro_id_str + ".parquet"
        return above_dir + "test_spectrograms/" + spectro_id_str + ".parquet"

    meta_frame = meta_frame.reset_index(drop=True)
    meta_frame["_row_i__"] = np.arange(len(meta_frame), dtype=np.int64)

    col_to_i = {c: i for i, c in enumerate(meta_frame.columns)}
    need_cols = ["_row_i__", "spectrogram_label_offset_seconds"]
    for c in need_cols:
        if c not in col_to_i:
            raise KeyError(
                f"assemble_features requires column '{c}' but it is missing."
            )

    done = 0
    for spectrogram_id, grp in meta_frame.groupby("spectrogram_id", sort=False):
        spectro_id_str = str(int(spectrogram_id))
        spectro_file = _spectro_path(spectro_id_str)
        if not os.path.exists(spectro_file):
            raise FileNotFoundError(f"Missing spectrogram file: {spectro_file}")

        pf = pq.ParquetFile(spectro_file)
        colnames = pf.schema.names
        use_cols = colnames[1:]  # 400 freq-region columns
        tbl = pf.read(columns=use_cols)
        this_arr = np.ascontiguousarray(
            tbl.to_pandas(types_mapper=None).to_numpy(dtype=np.float64, copy=False)
        )

        for row in grp.itertuples(index=False, name=None):
            i = int(row[col_to_i["_row_i__"]])
            loc_offset = int(row[col_to_i["spectrogram_label_offset_seconds"]] / 2)

            a = this_arr[loc_offset + 148]
            b = this_arr[loc_offset + 149]
            c = this_arr[loc_offset + 150]
            d = this_arr[loc_offset + 151]
            s4 = a + b + c + d

            middle4s = s4 / (4.0 * spect_trend4)
            middle4s = np.clip(middle4s, 0.001, 1000.0)
            middle4s = np.where(np.isfinite(middle4s), middle4s, 0.001)

            denom = (
                this_arr[loc_offset + 149 - 56]
                + this_arr[loc_offset + 149 - 40]
                + this_arr[loc_offset + 149 - 24]
                + this_arr[loc_offset + 149 + 56]
                + this_arr[loc_offset + 149 + 40]
                + this_arr[loc_offset + 149 + 24]
            )
            ratio4s = s4 / denom
            ratio4s = np.clip((6.0 / 4.0) * ratio4s, 0.01, 100.0)
            ratio4s = np.where(np.isfinite(ratio4s), ratio4s, 1.0)

            ratio4spre = np.log10(ratio4s)

            spect_mean = float(np.mean(middle4s))
            spect_median = float(np.median(middle4s))

            segs = middle4s.reshape(4, 100)
            the4means = segs.mean(axis=1)
            the4medians = np.median(segs, axis=1)

            middle4spre = np.log10(middle4s / spect_mean)

            middle4s_sm = _rolling_center_mean_1d(middle4spre, smooth_width)
            ratio4s_sm = _rolling_center_mean_1d(ratio4spre, smooth_width)

            if half > 0:
                for ioff in (0, 100, 200, 300):
                    sl = slice(ioff, ioff + half)
                    middle4s_sm[sl] = middle4spre[sl]
                    ratio4s_sm[sl] = ratio4spre[sl]

            middle4sds = middle4s_sm[select_inds]
            ratio4sds = ratio4s_sm[select_inds]

            spect_mean_log = np.log10(spect_mean)
            spect_median_log = np.log10(spect_median)
            the4means_log = np.log10(the4means)
            the4medians_log = np.log10(the4medians)

            out[i, 0 : len(middle4sds)] = middle4sds
            out[i, len(middle4sds) : len(middle4sds) + len(ratio4sds)] = ratio4sds
            base = len(middle4sds) + len(ratio4sds)
            out[i, base + 0] = spect_mean_log
            out[i, base + 1] = spect_median_log
            out[i, base + 2 : base + 6] = the4means_log
            out[i, base + 6 : base + 10] = the4medians_log

            done += 1
            if done % print_every_nth == 0:
                print("... {} done...".format(done))

            if SHOW_PLOT:
                this_clr = "blue"
                title_start = "Middle-8s Feature Values of " + traintest
                plt.title(title_start + " (smooth={})".format(smooth_width))
                if "clust_id" in meta_frame.columns:
                    this_clust = int(row[col_to_i["clust_id"]])
                    this_clr = kmclrs[this_clust]
                    plt.title(
                        title_start
                        + " (color-coded by the {} clusters, smooth={})".format(
                            num_clusts, smooth_width
                        )
                    )
                the_alpha = np.clip(0.05 * 350 / len(meta_frame), 0.003, 0.5)
                freqs = np.array(range(100)) * 0.19525 + 0.59
                freqs[0] = 0.0
                freqs4 = np.array(4 * list(freqs))
                freqs4ds = freqs4[select_inds]
                plt.plot(
                    freqs4ds
                    + smooth_width * 0.15 * (np.random.rand(len(freqs4ds)) - 0.5),
                    middle4sds,
                    ".",
                    c=this_clr,
                    markersize=10,
                    alpha=the_alpha,
                )
                plt.plot(
                    -1.2 + smooth_width * 0.15 * (np.random.rand(4) - 0.5),
                    the4means_log + (0.0 * (np.random.rand(4) - 0.5)),
                    ".",
                    c=this_clr,
                    markersize=10,
                    alpha=the_alpha,
                )
                plt.ylim(-1.5, 1.5)
                plt.xlabel(
                    "<-- Means are band < 0"
                    + 20 * " "
                    + "Frequency Bands (Hz)"
                    + 50 * " "
                )
                plt.ylabel(
                    "Amplitude (/ref-spectrum, /mean, and smoothed n={})".format(
                        smooth_width
                    )
                )

    feats_frame = pd.DataFrame(out, columns=feat_cols)

    if "clust_id" in meta_frame.columns:
        feats_frame["clust_id"] = meta_frame["clust_id"].to_numpy()

    if SHOW_PLOT:
        plt.savefig("middle8s_" + traintest + "_features.png")
        plt.close()

    return feats_frame.reset_index().drop(columns=["index"])




## === cell 7
def find_best_tamed_kl(solution, submission, pred_ids, clust_centers, NUM_CLUSTS):
    """
    Adjust the taming fraction for each cluster center to optimize KL.

    Stability: clip/renormalize per-cluster centers used for scoring.
    """
    mean_all_probs = np.array(
        [0.208319, 0.132120, 0.128532, 0.138913, 0.179294, 0.212822]
    )
    tamed_fracs = 0.0 * np.ones(NUM_CLUSTS)
    tamed_centers = clust_centers.copy()
    for iclust in range(NUM_CLUSTS):
        tamed_centers[iclust, :] = mean_all_probs

    best_fracs = tamed_fracs.copy()
    best_centers = tamed_centers.copy()

    for iclust in range(NUM_CLUSTS):
        last_kl = 10.0
        for this_frac in np.arange(0.03, 1.00, 0.05):  # 0.03--0.98
            tamed_fracs[iclust] = this_frac
            this_cent = (
                tamed_fracs[iclust] * clust_centers[iclust, :]
                + (1.0 - tamed_fracs[iclust]) * mean_all_probs
            )
            this_cent = np.clip(this_cent, 1e-15, 1.0)
            this_cent = this_cent / np.sum(this_cent)
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




## === cell 8
def tune_global_shrink_alpha(
    solution_votes_df, pred_ids, centers, mean_probs, alphas=None
):
    if alphas is None:
        alphas = np.linspace(0.30, 1.00, 15)  # coarse grid; small compute; stable
    best_a = None
    best_kl = np.inf
    tmp = solution_votes_df.copy()
    for a in alphas:
        shrunk = a * centers + (1.0 - a) * mean_probs[None, :]
        shrunk = np.clip(shrunk, 1e-15, 1.0)
        shrunk = shrunk / shrunk.sum(axis=1, keepdims=True)
        for j in range(HBA_number):
            tmp[HBA_votes[j]] = shrunk[pred_ids, j]
        kl = kld_score(solution_votes_df[HBA_votes], tmp[HBA_votes])
        if kl < best_kl:
            best_kl = kl
            best_a = float(a)
    return best_a, best_kl




## === cell 9
train_meta, test_meta = read_hms_meta()



## === cell 10
plt.figure(figsize=(5, 2))
plt.hist(train_meta["spectrogram_sub_id"], bins=55, log=True)
plt.title("Histogram of spectrogram_sub_id")
plt.savefig("hist_spectrogram_sub_id.png")
plt.close()

plt.figure(figsize=(5, 2))
plt.hist(train_meta["eeg_sub_id"], bins=55, log=True)
plt.title("Histogram of eeg_sub_id")
plt.savefig("hist_eeg_sub_id.png")
plt.close()



## === cell 11
num_clusts = NUM_CLUSTS  # 6 to 10
clust_rows_bool = train_meta.eeg_sub_id < 200



## === cell 12
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
plt.savefig("hist_total_vote_clustered.png")
plt.close()

prob_array = np.array(prob_vectors)
kmeans = KMeans(
    n_clusters=num_clusts, init="k-means++", n_init=10, max_iter=300, random_state=SEED
)
kmeans.fit(prob_array)

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



## === cell 13
train_meta["clust_id"] = kmeans.predict(np.array(train_meta[HBA_probs]))

all_probs = train_meta[HBA_probs]
all_ids = train_meta["clust_id"]

kmclrs = prob_prob_scatter("Seizure", "GPD", all_probs, all_ids, iclust_of_order)
kmclrs = prob_prob_scatter("LPD", "GRDA", all_probs, all_ids, iclust_of_order)
kmclrs = prob_prob_scatter("LRDA", "Other", all_probs, all_ids, iclust_of_order)

clust_counts = train_meta.clust_id.value_counts()

iorder_of_clust = num_clusts * [-1]
for iord, iclust in enumerate(iclust_of_order):
    iorder_of_clust[iclust] = iord
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
    plt.savefig(f"cluster_center_{iord}_{iclust}.png")
    plt.close()



## === cell 14
solution_train = train_meta[["eeg_id"] + HBA_votes]
for col_pre in HBA_names:
    solution_train.loc[:, col_pre + "_vote"] = train_meta[col_pre + "_prob"]

submission_train = solution_train.copy()
clust_ids = train_meta["clust_id"]

for iprob in range(HBA_number):
    this_col_probs = clust_centers[:, iprob]
    submission_train[HBA_votes[iprob]] = this_col_probs[clust_ids]

print(
    "Score if HBA samples are correctly assigned cluster prob.s:",
    np.round(kld_score(solution_train[HBA_votes], submission_train[HBA_votes]), 4),
)



## === cell 15
train_rows_bool = (
    (train_meta.eeg_sub_id < 33 + 1) & (train_meta.eeg_sub_id % 5 == 3)
) | ((train_meta.eeg_sub_id == 0) & ((train_meta.eeg_id % 23) % 8 > 1))
print("Number of Training rows:", sum(train_rows_bool))

valid_rows_bool = (
    (train_meta.eeg_sub_id < 44 + 1) & (train_meta.eeg_sub_id % 19 == 6)
) | ((train_meta.eeg_sub_id == 0) & ((train_meta.eeg_id % 23) % 8 < 2))
print("Number of Validation rows:", sum(valid_rows_bool))




## === cell 16
def _maybe_read_preproc(meta_path, feats_path):
    if os.path.exists(meta_path) and os.path.exists(feats_path):
        return pd.read_csv(meta_path), pd.read_csv(feats_path)
    return None, None


def _resolve_clust_id_after_merge(df):
    if "clust_id" in df.columns:
        return df
    if "clust_id_x" in df.columns and "clust_id_y" in df.columns:
        df["clust_id"] = df["clust_id_y"].combine_first(df["clust_id_x"])
        df = df.drop(columns=["clust_id_x", "clust_id_y"])
        return df
    if "clust_id_y" in df.columns:
        df = df.rename(columns={"clust_id_y": "clust_id"})
        if "clust_id_x" in df.columns:
            df = df.drop(columns=["clust_id_x"])
        return df
    if "clust_id_x" in df.columns:
        df = df.rename(columns={"clust_id_x": "clust_id"})
        return df
    return df


if USE_PREPROC:
    meta_path = above_dir_preproc + "Xy_train_meta_v47.csv"
    feats_path = above_dir_preproc + "Xy_train_feats_v47.csv"
    Xy_train_meta, Xy_train_feats = _maybe_read_preproc(meta_path, feats_path)
    if Xy_train_meta is None:
        print(
            f"Preproc not found at {above_dir_preproc}; falling back to on-the-fly feature assembly for TRAIN."
        )
        USE_PREPROC = False  # for subsequent blocks

if not USE_PREPROC:
    Xy_train_meta = (train_meta[train_rows_bool])[::TRAIN_DOWNSEL].copy()
    Xy_train_meta = Xy_train_meta.reset_index().drop(columns=["index"])
    print("Number of samples used for training =", len(Xy_train_meta))

    Xy_train_feats = assemble_features(
        Xy_train_meta, traintest="train", smooth_width=SMOOTH_WIDTH, SHOW_PLOT=False
    )
    Xy_train_meta.to_csv(
        "Xy_train_meta.csv", header=True, index=False, float_format="%.6f"
    )
    Xy_train_feats.to_csv(
        "Xy_train_feats.csv", header=True, index=False, float_format="%.6f"
    )

Xy_train_meta = Xy_train_meta.merge(
    train_meta[["label_id", "clust_id"]],
    on="label_id",
    how="left",
    validate="many_to_one",
)
Xy_train_meta = _resolve_clust_id_after_merge(Xy_train_meta)
if "clust_id" not in Xy_train_meta.columns:
    raise ValueError("clust_id missing after merge/resolve; check label_id alignment.")
if Xy_train_meta["clust_id"].isna().any():
    raise ValueError("Missing clust_id after merging; check label_id alignment.")

Xy_train_feats["clust_id"] = Xy_train_meta["clust_id"].astype(int).to_numpy()

Xy_train_feats



## === cell 17
if USE_PREPROC:
    meta_path = above_dir_preproc + "Xy_valid_meta_v47.csv"
    feats_path = above_dir_preproc + "Xy_valid_feats_v47.csv"
    Xy_valid_meta, Xy_valid_feats = _maybe_read_preproc(meta_path, feats_path)
    if Xy_valid_meta is None:
        print(
            f"Preproc not found at {above_dir_preproc}; falling back to on-the-fly feature assembly for VALID."
        )
        USE_PREPROC = False

if not USE_PREPROC:
    Xy_valid_meta = (train_meta[valid_rows_bool])[::VALID_DOWNSEL].copy()
    Xy_valid_meta = Xy_valid_meta.reset_index().drop(columns=["index"])
    print("Number of samples used for Validation =", len(Xy_valid_meta))

    Xy_valid_feats = assemble_features(
        Xy_valid_meta,
        traintest="train",
        smooth_width=SMOOTH_WIDTH,
        SHOW_PLOT=False,
    )
    Xy_valid_meta.to_csv(
        "Xy_valid_meta.csv", header=True, index=False, float_format="%.6f"
    )
    Xy_valid_feats.to_csv(
        "Xy_valid_feats.csv", header=True, index=False, float_format="%.6f"
    )

Xy_valid_meta = Xy_valid_meta.merge(
    train_meta[["label_id", "clust_id"]],
    on="label_id",
    how="left",
    validate="many_to_one",
)
Xy_valid_meta = _resolve_clust_id_after_merge(Xy_valid_meta)
if "clust_id" not in Xy_valid_meta.columns:
    raise ValueError("clust_id missing after merge/resolve; check label_id alignment.")
if Xy_valid_meta["clust_id"].isna().any():
    raise ValueError("Missing clust_id after merging; check label_id alignment.")

Xy_valid_feats["clust_id"] = Xy_valid_meta["clust_id"].astype(int).to_numpy()

Xy_valid_feats



## === cell 18
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
        multi_class="multinomial",
        n_jobs=-1,
        random_state=SEED,
    ).fit(Xlr1, y)

    plt.figure(figsize=(8, 4))
    plt.plot(lrmodel1.coef_.T, "-", alpha=0.5)
    plt.plot(lrmodel1.coef_.T, ".", alpha=1.0)
    plt.title("LR1 coefficients, C=" + str(LR1_C) + " (colored by cluster)")
    plt.savefig("lr1_coeffs.png")
    plt.close()

    print("\nLR1 model score for X,y = {:.1f}%\n".format(100 * lrmodel1.score(Xlr1, y)))

if USE_LR2:
    Xlr2 = Xlr.iloc[:, int(len(Xlr.columns) / 2) :]
    lrmodel2 = LogisticRegression(
        penalty="l1",
        C=LR2_C,
        solver="saga",
        max_iter=1500,
        multi_class="multinomial",
        n_jobs=-1,
        random_state=SEED,
    ).fit(Xlr2, y)

    plt.figure(figsize=(8, 4))
    plt.plot(lrmodel2.coef_.T, "-", alpha=0.5)
    plt.plot(lrmodel2.coef_.T, ".", alpha=1.0)
    plt.title("LR2 coefficients, C=" + str(LR2_C) + " (colored by cluster)")
    plt.savefig("lr2_coeffs.png")
    plt.close()

    print("\nLR2 model score for X,y = {:.1f}%\n".format(100 * lrmodel2.score(Xlr2, y)))



## === cell 19
Xy_train_wLRfeats = Xy_train_feats.copy()
X = Xy_train_feats.drop(columns=["clust_id"])
Xlr = X.drop(columns=X.columns[-10:])
if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : int(len(Xlr.columns) / 2)]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    for iadd in range(NUM_CLUSTS):
        Xy_train_wLRfeats["lr" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            np.random.rand(len(lrprobas)) - 0.5
        )
    plt.figure(figsize=(7, 2.5))
    plt.hist(np.max(lrmodel1.predict_proba(Xlr1), axis=1), bins=50)
    plt.xlim(-0.5 * LR_BLUR, 1.01)
    plt.title("Training max LR1 proba values (pre-blur)")
    plt.savefig("hist_lr1_train_maxproba.png")
    plt.close()
if USE_LR2:
    Xlr2 = Xlr.iloc[:, int(len(Xlr.columns) / 2) :]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    for iadd in range(NUM_CLUSTS):
        Xy_train_wLRfeats["rlr" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            np.random.rand(len(lrprobas)) - 0.5
        )
    plt.figure(figsize=(7, 2.5))
    plt.hist(np.max(lrmodel2.predict_proba(Xlr2), axis=1), bins=50)
    plt.xlim(-0.5 * LR_BLUR, 1.01)
    plt.title("Training max LR2 proba values (pre-blur)")
    plt.savefig("hist_lr2_train_maxproba.png")
    plt.close()

Xy_valid_wLRfeats = Xy_valid_feats.copy()
X = Xy_valid_feats.drop(columns=["clust_id"])
Xlr = X.drop(columns=X.columns[-10:])
if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : int(len(Xlr.columns) / 2)]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    for iadd in range(NUM_CLUSTS):
        Xy_valid_wLRfeats["lr" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            np.random.rand(len(lrprobas)) - 0.5
        )
if USE_LR2:
    Xlr2 = Xlr.iloc[:, int(len(Xlr.columns) / 2) :]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    for iadd in range(NUM_CLUSTS):
        Xy_valid_wLRfeats["rlr" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            np.random.rand(len(lrprobas)) - 0.5
        )



## === cell 20
X = Xy_train_wLRfeats.drop(columns=["clust_id"])
y = Xy_train_wLRfeats.clust_id

ave_oob = []
nfits = 3
for ifit in range(nfits):
    rfmodel = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        max_features=0.2,
        max_samples=0.9,
        oob_score=True,
        class_weight="balanced_subsample",
        n_jobs=-1,
        verbose=0,
        random_state=SEED + ifit,
    ).fit(X, y)
    ave_oob.append(rfmodel.oob_score_)

sort_inds = rfmodel.feature_importances_.argsort()
plt.figure(figsize=(4, 7))
plt.barh(rfmodel.feature_names_in_[sort_inds], rfmodel.feature_importances_[sort_inds])
plt.ylim(len(sort_inds) - 40, len(sort_inds) + 0.2)
plt.title("Feature Importances (top 40)")
plt.savefig("rf_feature_importances.png")
plt.close()

print(
    "\nRF model ave OOB score = {:.1f}% +/- {:.1f}".format(
        100 * np.mean(ave_oob), 100 * np.std(ave_oob)
    )
)
print("\nRF model score for X,y = {:.1f}%\n".format(100 * rfmodel.score(X, y)))



## === cell 21
Xy_train_meta["pred_id"] = rfmodel.predict(Xy_train_wLRfeats.drop(columns=["clust_id"]))

maxprobs_train = np.max(rfmodel.predict_proba(X), axis=1)
plt.figure(figsize=(7, 2.5))
plt.hist(maxprobs_train, bins=50)
plt.xlim(0.0, 1.0)
plt.title("Histogram of max(proba) for model-training samples")
plt.savefig("hist_rf_train_maxproba.png")
plt.close()

solution = Xy_train_meta[["eeg_id"] + HBA_votes]
for col_pre in HBA_names:
    solution.loc[:, col_pre + "_vote"] = Xy_train_meta[col_pre + "_prob"]

submission = solution.copy()
pred_ids = Xy_train_meta["pred_id"].to_numpy()

best_fracs, best_centers = find_best_tamed_kl(
    solution=solution[HBA_votes],
    submission=submission[HBA_votes],
    pred_ids=pred_ids,
    clust_centers=clust_centers,
    NUM_CLUSTS=NUM_CLUSTS,
)

for iprob in range(HBA_number):
    this_col_probs = best_centers[:, iprob]
    submission[HBA_votes[iprob]] = this_col_probs[pred_ids]
this_kl = kld_score(solution[HBA_votes], submission[HBA_votes])
print("Tamed fractions:\n", best_fracs, "\nand centers:\n", best_centers)
print("\nKL from tamed centers: {:.4f}".format(this_kl))

best_centers_train = best_centers.copy()



## === cell 22
Xy_valid_meta["pred_id"] = rfmodel.predict(Xy_valid_wLRfeats.drop(columns=["clust_id"]))

maxprobs_valid = np.max(
    rfmodel.predict_proba(Xy_valid_wLRfeats.drop(columns=["clust_id"])), axis=1
)
plt.figure(figsize=(7, 2.5))
plt.hist(maxprobs_valid, bins=50)
plt.xlim(0.0, 1.0)
plt.title("Histogram of max(proba) for validation samples")
plt.savefig("hist_rf_valid_maxproba.png")
plt.close()

solution = Xy_valid_meta[["eeg_id"] + HBA_votes]
for col_pre in HBA_names:
    solution.loc[:, col_pre + "_vote"] = Xy_valid_meta[col_pre + "_prob"]

submission = solution.copy()
pred_ids = Xy_valid_meta["pred_id"].to_numpy()

best_fracs_valid, best_centers_valid = find_best_tamed_kl(
    solution=solution[HBA_votes],
    submission=submission[HBA_votes],
    pred_ids=pred_ids,
    clust_centers=clust_centers,
    NUM_CLUSTS=NUM_CLUSTS,
)

for iprob in range(HBA_number):
    this_col_probs = best_centers_valid[:, iprob]
    submission[HBA_votes[iprob]] = this_col_probs[pred_ids]
this_kl = kld_score(solution[HBA_votes], submission[HBA_votes])
print(
    "Tamed fractions (valid):\n",
    best_fracs_valid,
    "\nand centers:\n",
    best_centers_valid,
)
print("\nKL from tamed centers (valid): {:.4f}".format(this_kl))

mean_all_probs = np.array(
    [0.208319, 0.132120, 0.128532, 0.138913, 0.179294, 0.212822], dtype=np.float64
)
valid_pred_ids = Xy_valid_meta["pred_id"].to_numpy(dtype=int)
alpha_best, alpha_best_kl = tune_global_shrink_alpha(
    solution_votes_df=solution[HBA_votes],
    pred_ids=valid_pred_ids,
    centers=best_centers_train,
    mean_probs=mean_all_probs,
    alphas=np.linspace(0.20, 1.00, 17),
)
print(
    f"Best global shrink alpha on validation = {alpha_best:.3f} with KL = {alpha_best_kl:.4f}"
)



## === cell 23
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
            np.random.rand(len(lrprobas)) - 0.5
        )
if USE_LR2:
    Xlr2 = Xlr.iloc[:, int(len(Xlr.columns) / 2) :]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    for iadd in range(NUM_CLUSTS):
        Xy_test_wLRfeats["rlr" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            np.random.rand(len(lrprobas)) - 0.5
        )

X_test = Xy_test_wLRfeats.copy()
pred_ids = rfmodel.predict(X_test).astype(int)

test_pred_rows = test_meta[["eeg_id"]].copy()
for new_col in HBA_votes:
    test_pred_rows[new_col] = 1.0 / HBA_number

centers_for_test = best_centers_train.copy()

centers_for_test = (
    alpha_best * centers_for_test + (1.0 - alpha_best) * mean_all_probs[None, :]
)
centers_for_test = np.clip(centers_for_test, 1e-15, 1.0)
centers_for_test = centers_for_test / centers_for_test.sum(axis=1, keepdims=True)

for iprob in range(HBA_number):
    this_col_probs = centers_for_test[:, iprob]
    test_pred_rows[HBA_votes[iprob]] = this_col_probs[pred_ids]

test_submit = test_pred_rows.groupby("eeg_id", as_index=False)[HBA_votes].mean()

probs = test_submit[HBA_votes].to_numpy(dtype=float)
probs = np.clip(probs, 1e-8, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)
test_submit[HBA_votes] = probs

sample_path = above_dir + "sample_submission.csv"
sample_sub = pd.read_csv(sample_path)

test_submit = sample_sub[["eeg_id"]].merge(
    test_submit, on="eeg_id", how="left", sort=False
)

for c in HBA_votes:
    if c not in test_submit.columns:
        test_submit[c] = 1.0 / HBA_number
test_submit[HBA_votes] = test_submit[HBA_votes].fillna(1.0 / HBA_number).astype(float)

probs = test_submit[HBA_votes].to_numpy(dtype=float)
probs = np.where(np.isfinite(probs), probs, 1.0 / HBA_number)
probs = np.clip(probs, 1e-8, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)
test_submit[HBA_votes] = probs

test_submit = test_submit[["eeg_id"] + HBA_votes].copy()

print(test_submit.head())
print("Submission length:", len(test_submit), "Expected:", len(sample_sub))
print("Row-sum check (min/max):", probs.sum(axis=1).min(), probs.sum(axis=1).max())
print("Columns:", list(test_submit.columns))

test_submit.to_csv(
    "submission.csv", header=True, index=False, na_rep="", float_format="%.6f"
)
print("Wrote submission.csv with shape:", test_submit.shape)
