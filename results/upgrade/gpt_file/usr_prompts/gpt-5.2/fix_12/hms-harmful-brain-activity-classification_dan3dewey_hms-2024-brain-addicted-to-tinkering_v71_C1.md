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

0.801569848006421

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, warnings

warnings.filterwarnings("ignore")




## === cell 1
import matplotlib

matplotlib.use("Agg")




## === cell 2
NUM_CLUSTS = 9  # 6 to 10

USE_PREPROC = True  # Read-in saved meta and features frames
TRAIN_DOWNSEL = 1  # large values for code test; set to 1 to output all.
VALID_DOWNSEL = 1  #  "
SMOOTH_WIDTH = 5  # Odd>1: 3,5,7,9,...

LR_REGU = "l1"
USE_LR1 = False
LR1_C = 0.05  # 0.15     # smaller --> fewer non-zero coeff.s
USE_LR2 = False
LR2_C = 0.03  # 0.10
LR_BLUR = 0.10

USE_TAMED = False


def _resolve_above_dir():
    cands = [
        "/kaggle/input/hms-harmful-brain-activity-classification/",
        "/kaggle/data/hms-harmful-brain-activity-classification/",
        "../input/hms-harmful-brain-activity-classification/",
        "/kaggle/input/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification/",
        "/kaggle/data/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification/",
    ]
    for c in cands:
        c = c if c.endswith("/") else (c + "/")
        if (
            os.path.exists(os.path.join(c, "train.csv"))
            and os.path.exists(os.path.join(c, "test.csv"))
            and os.path.exists(os.path.join(c, "sample_submission.csv"))
        ):
            return c
    return "/kaggle/input/hms-harmful-brain-activity-classification/"


above_dir = _resolve_above_dir()

above_dir_preproc = "../input/hms-2024-brain-data/"




## === cell 3
def _pjoin(a, b):
    return os.path.join(a, b)


def _exists(path):
    try:
        return os.path.exists(path)
    except Exception:
        return False




## === cell 4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import pyarrow.parquet as pq

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression




## === cell 5
EPS = 1e-12




## === cell 6
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




## === cell 7
def kld_score(solution, submission):
    """
    Calculate the average KL divergence score (like the competition).
    To avoid NaNs / inf due to zero or non-normalized predictions, we:
      - clip submission probabilities
      - renormalize each row of submission to sum to 1
    """
    sol_vals = solution.to_numpy(dtype=np.float64, copy=True)
    sub_vals = submission.to_numpy(dtype=np.float64, copy=True)

    sub_vals = np.clip(sub_vals, 1e-15, 1.0)
    row_sums = sub_vals.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    sub_vals /= row_sums

    sol_vals = np.clip(sol_vals, 1e-15, 1.0)

    kld = -np.nansum(sol_vals * np.log(sub_vals / sol_vals))
    n = len(solution)
    return kld / (n if n else 1)




## === cell 8
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
    plt.savefig("hist_total_votes.png", bbox_inches="tight")
    plt.close()

    for col_pre in HBA_names:
        train_meta[col_pre + "_prob"] = (
            train_meta[col_pre + "_vote"] / train_meta["total_vote"]
        )

    print("Calculating voting entropy values ...")
    probs = np.clip(train_meta[HBA_probs].to_numpy(dtype=np.float64), 1.0e-8, 1.0)
    train_meta["entropy"] = np.nansum(probs * (-np.log(probs)), axis=1)

    return train_meta, test_meta




## === cell 9
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
    plt.savefig(f"scatter_{name1}_{name2}.png", bbox_inches="tight")
    plt.close()
    return kmclrs  # returns cluster colors appropriate for iclust




## === cell 10
from collections import OrderedDict

_SPECTRO_CACHE = OrderedDict()
_SPECTRO_CACHE_MAX = 4096  # larger cache reduces repeated disk/deserialize work


def _spectro_path(spectro_id, traintest):
    spectro_id_str = str(int(spectro_id))
    if traintest != "test":
        return above_dir + "train_spectrograms/" + spectro_id_str + ".parquet"
    else:
        return above_dir + "test_spectrograms/" + spectro_id_str + ".parquet"


def _table_to_numpy_float64(tbl):
    return tbl.to_pandas().to_numpy(dtype=np.float64, copy=False)


def _load_spectro_np(spectro_id, traintest):
    key = (traintest, int(spectro_id))
    arr = _SPECTRO_CACHE.get(key)
    if arr is not None:
        _SPECTRO_CACHE.move_to_end(key)
        return arr

    tbl = pq.read_table(_spectro_path(spectro_id, traintest), memory_map=True)
    mat = _table_to_numpy_float64(tbl)

    _SPECTRO_CACHE[key] = mat
    if len(_SPECTRO_CACHE) > _SPECTRO_CACHE_MAX:
        _SPECTRO_CACHE.popitem(last=False)
    return mat


def _rolling_mean_centered_same(x, kernel):
    return np.convolve(x, kernel, mode="same")


def _rolling_mean_simple2_inplace(x):
    c = np.cumsum(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    out[1:] = (c[1:] - c[:-1]) / 2.0
    return out


def assemble_features(meta_frame, traintest="train", smooth_width=5):
    """
    Create a dataframe of spectrogram features from the meta_frame rows.
    Will include clust_id (i.e, the y) if it is in the input meta_frame.
    Assumes these are available: above_dir, the4chains
    """
    if traintest == "validation":
        plt.figure(figsize=(9, 7))

    freqs = np.arange(100, dtype=np.float64) * 0.19525 + 0.59
    spect_trend = 150.0 / (1.0**2.3 + freqs ** (2.3))
    baseinds = np.insert(
        np.arange(int((smooth_width - 1) / 2), 100, smooth_width), 0, 0
    )
    spect_trend4 = np.tile(spect_trend, 4)

    n_fft_bins = 256
    freqbins = np.array([7, 15, 23, 60], dtype=np.int64)
    apod_wind = np.blackman(n_fft_bins).astype(np.float64, copy=False)
    fftfeatbins = np.array([4, 9, 16, 25, 36, 49], dtype=np.int64)

    halfw = int((smooth_width - 1) / 2)
    select_inds = np.concatenate(
        (baseinds, 100 + baseinds, 200 + baseinds, 300 + baseinds)
    ).astype(np.int64)
    n_mid = select_inds.size
    n_flare = 4 * len(freqbins) * len(fftfeatbins)
    n_extra = 2 + 4 * 2
    add_clust = int("clust_id" in meta_frame.columns)
    n_total = n_mid + n_flare + n_extra + add_clust

    mid_cols = [f"mid_{i}" for i in range(n_mid)]
    flarecols = []
    for ispec in range(4):
        for freqbin in freqbins.tolist():
            for fftfeatbin in fftfeatbins.tolist():
                flarecols.append(
                    "fft-"
                    + the4chains[ispec]
                    + "{:.1f}-".format(freqs[freqbin])
                    + str(fftfeatbin)
                )
    extra_cols = (
        ["Mean", "Median"]
        + [the4chains[i] + "mean" for i in range(4)]
        + [the4chains[i] + "median" for i in range(4)]
    )
    out_cols = mid_cols + flarecols + extra_cols + (["clust_id"] if add_clust else [])

    feats = np.empty((len(meta_frame), n_total), dtype=np.float64)

    kernel_sm = np.ones(smooth_width, dtype=np.float64) / smooth_width
    kernel5 = np.ones(5, dtype=np.float64) / 5.0

    sum_spect_trend = np.array(
        [float(np.sum(spect_trend[fb - 4 : fb + 5 : 2])) for fb in freqbins],
        dtype=np.float64,
    )

    print_every_nth = max([100, 100 * int(0.5 + len(meta_frame.index) / (100.0 * 15))])
    plot_every_nth = max([1, int(0.5 + len(meta_frame.index) / 100)])

    spectro_ids = meta_frame["spectrogram_id"].to_numpy(np.int64, copy=False)
    offsets_sec = meta_frame["spectrogram_label_offset_seconds"].to_numpy(
        np.float64, copy=False
    )
    if add_clust:
        clust_ids = meta_frame["clust_id"].to_numpy(np.int64, copy=False)

    col_centers = np.empty((4, len(freqbins)), dtype=np.int64)
    for ispec in range(4):
        base = ispec * 100
        for j, fb in enumerate(freqbins):
            col_centers[ispec, j] = int(fb + 1 + base)

    for i in range(len(meta_frame)):
        spectro_arr = _load_spectro_np(spectro_ids[i], traintest)

        max_loc = spectro_arr.shape[0] - 152
        loc_offset = int(offsets_sec[i] / 2)
        if loc_offset < 0:
            loc_offset = 0
        if max_loc < 0:
            max_loc = 0
        if loc_offset > max_loc:
            loc_offset = max_loc

        fftlocbeg = int(loc_offset + 149 - (n_fft_bins / 2 - 1))
        fftlocend = int(loc_offset + 150 + (n_fft_bins / 2 - 1))
        if fftlocbeg < 0:
            fftlocbeg = 0
        if fftlocend >= spectro_arr.shape[0]:
            fftlocend = spectro_arr.shape[0] - 1
        if fftlocend - fftlocbeg + 1 < n_fft_bins:
            need = n_fft_bins - (fftlocend - fftlocbeg + 1)
            fftlocbeg = max(0, fftlocbeg - need)
            fftlocend = min(spectro_arr.shape[0] - 1, fftlocbeg + n_fft_bins - 1)

        mid_raw = (
            spectro_arr[loc_offset + 148, 1:]
            + spectro_arr[loc_offset + 149, 1:]
            + spectro_arr[loc_offset + 150, 1:]
            + spectro_arr[loc_offset + 151, 1:]
        ) / (4.0 * spect_trend4)
        mid_raw = np.clip(mid_raw, 0.001, 1000.0)
        mid_raw = np.nan_to_num(mid_raw, nan=0.001, posinf=0.001, neginf=0.001)

        spect_mean = float(np.mean(mid_raw))
        spect_median = float(np.median(mid_raw))

        mid_raw_4x100 = mid_raw.reshape(4, 100)
        the4means = mid_raw_4x100.mean(axis=1)
        the4medians = np.median(mid_raw_4x100, axis=1)

        mid_pre = np.log10(mid_raw / spect_mean)

        mid_sm = _rolling_mean_centered_same(mid_pre, kernel_sm)
        if halfw > 0:
            mid_sm[0:halfw] = mid_pre[0:halfw]
            mid_sm[100 : 100 + halfw] = mid_pre[100 : 100 + halfw]
            mid_sm[200 : 200 + halfw] = mid_pre[200 : 200 + halfw]
            mid_sm[300 : 300 + halfw] = mid_pre[300 : 300 + halfw]

        feats[i, 0:n_mid] = mid_sm[select_inds]

        off_flare = n_mid
        k = 0
        for ispec in range(4):
            for j in range(len(freqbins)):
                cc = col_centers[ispec, j]
                s = (
                    spectro_arr[fftlocbeg : fftlocend + 1, cc - 4]
                    + spectro_arr[fftlocbeg : fftlocend + 1, cc - 2]
                    + spectro_arr[fftlocbeg : fftlocend + 1, cc]
                    + spectro_arr[fftlocbeg : fftlocend + 1, cc + 2]
                    + spectro_arr[fftlocbeg : fftlocend + 1, cc + 4]
                ) / sum_spect_trend[j]
                s = np.clip(s, 0.001, 1000.0)
                s = np.nan_to_num(s, nan=0.001, posinf=0.001, neginf=0.001)

                if s.shape[0] < n_fft_bins:
                    pad = n_fft_bins - s.shape[0]
                    s = np.pad(s, (0, pad), mode="edge")
                elif s.shape[0] > n_fft_bins:
                    s = s[:n_fft_bins]

                denom = s[127] + s[128]
                if denom == 0:
                    denom = 1.0
                s = 2.0 * s / denom
                s = apod_wind * np.clip(s, 0.0, 10.0)

                s = _rolling_mean_simple2_inplace(s)
                s = _rolling_mean_simple2_inplace(s)

                amplfft = np.abs(np.fft.fft(s))[0 : int(n_fft_bins / 2)]
                amplfft_sm = np.convolve(amplfft, kernel5, mode="same")
                amplfft_sm = np.log10(1.0 + amplfft_sm)

                feats[i, off_flare + k : off_flare + k + len(fftfeatbins)] = amplfft_sm[
                    fftfeatbins
                ]
                k += len(fftfeatbins)

                if (i % plot_every_nth == 0) and traintest == "validation":
                    plt.plot(
                        np.sqrt(np.arange(int(n_fft_bins / 2))),
                        amplfft_sm,
                        c=kmclrs[int(clust_ids[i])],
                        lw=2,
                        alpha=0.01,
                    )

        off = n_mid + n_flare
        feats[i, off + 0] = np.log10(spect_mean)
        feats[i, off + 1] = np.log10(spect_median)
        feats[i, off + 2 : off + 6] = np.log10(the4means)
        feats[i, off + 6 : off + 10] = np.log10(the4medians)

        if add_clust:
            feats[i, -1] = float(clust_ids[i])

        if (i + 1) % print_every_nth == 0:
            print("... {} done...".format(i + 1))

    if traintest == "validation":
        plt.ylim(-0.05, 2.25)
        plt.xlim(0.0, 11.5)
        plt.xlabel("sqrt[  FFT bin number ]")
        plt.ylabel("log10[1+ FFT amplitude ]")
        plt.title("Flare-Rate Spectra (colored by cluster)")
        plt.savefig("validation_flare_rate_spectra.png", bbox_inches="tight")
        plt.close()

    feats_frame = pd.DataFrame(feats, columns=out_cols)
    return feats_frame.reset_index(drop=True)




## === cell 11
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
            this_kl = kld_score(solution[HBA_votes], submission[HBA_votes])
            if this_kl < last_kl:
                best_fracs = tamed_fracs.copy()
                best_centers = tamed_centers.copy()
                last_kl = this_kl
            else:
                tamed_fracs[iclust] = best_fracs[iclust]
                tamed_centers[iclust, :] = best_centers[iclust, :]
                break
    return best_fracs, best_centers




## === cell 12
def normalize_rows(df, cols):
    arr = df[cols].values.astype(float)
    arr = np.clip(arr, 1e-15, 1.0)
    s = arr.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    arr = arr / s
    df.loc[:, cols] = arr
    return df




## === cell 13
train_meta, test_meta = read_hms_meta()




## === cell 14
plt.figure(figsize=(5, 2))
plt.hist(train_meta["spectrogram_sub_id"], bins=55, log=True)
plt.title("Histogram of spectrogram_sub_id")
plt.savefig("hist_spectrogram_sub_id.png", bbox_inches="tight")
plt.close()

plt.figure(figsize=(5, 2))
plt.hist(train_meta["eeg_sub_id"], bins=55, log=True)
plt.title("Histogram of eeg_sub_id")
plt.savefig("hist_eeg_sub_id.png", bbox_inches="tight")
plt.close()




## === cell 15
np.random.seed(0)




## === cell 16
num_clusts = NUM_CLUSTS  # 6 to 10
clust_rows_bool = train_meta.eeg_sub_id < 200




## === cell 17
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
    n_clusters=num_clusts, init="k-means++", n_init=10, max_iter=300, random_state=0
)
kmeans.fit(prob_array)

clust_centers = kmeans.cluster_centers_
for iclust in range(NUM_CLUSTS):
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




## === cell 18
train_meta["clust_id"] = kmeans.predict(np.array(train_meta[HBA_probs]))

all_probs = train_meta[HBA_probs]
all_ids = train_meta["clust_id"]

if True:
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
    plt.savefig(f"cluster_center_{iord}.png", bbox_inches="tight")
    plt.close()




## === cell 19
clust_centers = clust_centers / clust_centers.sum(axis=1, keepdims=True)




## === cell 20
train_rows_bool = None
valid_rows_bool = None




## === cell 21
solution_train = train_meta[["eeg_id"]].copy()
for col_pre in HBA_names:
    solution_train[col_pre + "_vote"] = train_meta[col_pre + "_prob"]

submission_train = solution_train.copy()
clust_ids = train_meta["clust_id"]
if False:
    clust_ids = np.random.choice(9, size=len(train_meta), replace=True, p=None)
for iprob in range(HBA_number):
    this_col_probs = clust_centers[:, iprob]
    submission_train[HBA_votes[iprob]] = this_col_probs[clust_ids]

solution_train_probs = solution_train[HBA_votes]
submission_train_probs = submission_train[HBA_votes]
submission_train = normalize_rows(submission_train, HBA_votes)

print(
    "Score if HBA samples are correctly assigned cluster prob.s:",
    np.round(kld_score(solution_train_probs, submission_train[HBA_votes]), 4),
)




## === cell 22
DO_SPECTRA_PLOTS = False




## === cell 23
smooth_width = SMOOTH_WIDTH  # Here smooth_width is just for looking,

spectro_meta = train_meta[clust_rows_bool].copy()
every_nth = 97

freqs = np.array(range(100)) * 0.19525 + 0.59
downsel_freqs = np.insert(
    freqs[int((smooth_width - 1) / 2) : 100 : smooth_width], 0, freqs[0]
)

spect_trend = 150.0 / (1.0**2.3 + freqs ** (2.3))

freqbins = [7, 15, 23, 60]
freqclrs = ["red", "blue", "green", "yellow"]
n_fft_bins = 256
apod_wind = np.blackman(n_fft_bins)




## === cell 24
if DO_SPECTRA_PLOTS:
    pass




## === cell 25
if DO_SPECTRA_PLOTS:
    pass




## === cell 26
SMOOTH_WIDTH = SMOOTH_WIDTH




## === cell 27
train_rows_bool = (
    (train_meta.eeg_sub_id < 50 + 1) & (train_meta.eeg_sub_id % 4 == 2)
) | (  # 2, 6, 10, ..., 50
    train_meta.eeg_sub_id == 0
) & (
    (train_meta.eeg_id % 23) % 8 > 1
)  # include odd and even eeg_ids
print("Number of Training rows:", sum(train_rows_bool))

valid_rows_bool = (
    (train_meta.eeg_sub_id < 51 + 1) & (train_meta.eeg_sub_id % 8 == 3)
) | (  # 3, 11, ..., 51
    train_meta.eeg_sub_id == 0
) & (
    (train_meta.eeg_id % 23) % 8 < 2
)  # include odd and even eeg_ids
print("Number of Validation rows:", sum(valid_rows_bool))




## === cell 28
preproc_train_meta_path = _pjoin(above_dir_preproc, "Xy_train_meta_v66.csv")
preproc_train_feats_path = _pjoin(above_dir_preproc, "Xy_train_feats_v66.csv")
preproc_valid_meta_path = _pjoin(above_dir_preproc, "Xy_valid_meta_v66.csv")
preproc_valid_feats_path = _pjoin(above_dir_preproc, "Xy_valid_feats_v66.csv")

CAN_USE_PREPROC = USE_PREPROC and all(
    map(
        _exists,
        [
            preproc_train_meta_path,
            preproc_train_feats_path,
            preproc_valid_meta_path,
            preproc_valid_feats_path,
        ],
    )
)

if USE_PREPROC and not CAN_USE_PREPROC:
    print(
        "Preprocessed feature files not found; falling back to on-the-fly feature assembly."
    )
    print(
        "Missing one of:",
        preproc_train_meta_path,
        preproc_train_feats_path,
        preproc_valid_meta_path,
        preproc_valid_feats_path,
    )




## === cell 29
if CAN_USE_PREPROC:
    Xy_train_meta = pd.read_csv(preproc_train_meta_path)
    Xy_train_feats = pd.read_csv(preproc_train_feats_path)
    Xy_train_meta["clust_id"] = kmeans.predict(np.array(Xy_train_meta[HBA_probs]))
    Xy_train_feats["clust_id"] = Xy_train_meta["clust_id"]
    SMOOTH_WIDTH = 5
else:
    Xy_train_meta = (train_meta[train_rows_bool])[::TRAIN_DOWNSEL].copy()
    Xy_train_meta = Xy_train_meta.reset_index().drop(columns=["index"])
    print("Number of samples used for training =", len(Xy_train_meta))

    Xy_train_feats = assemble_features(
        Xy_train_meta, traintest="train", smooth_width=SMOOTH_WIDTH
    )
    Xy_train_meta.to_csv(
        "Xy_train_meta.csv", header=True, index=False, float_format="%.6f"
    )
    Xy_train_feats.to_csv(
        "Xy_train_feats.csv", header=True, index=False, float_format="%.6f"
    )




## === cell 30
print("Xy_train_feats shape:", Xy_train_feats.shape)
Xy_train_feats.head(2)




## === cell 31
assert "clust_id" in Xy_train_feats.columns
assert len(Xy_train_feats) == len(Xy_train_meta)




## === cell 32
if CAN_USE_PREPROC:
    Xy_valid_meta = pd.read_csv(preproc_valid_meta_path)
    Xy_valid_feats = pd.read_csv(preproc_valid_feats_path)
    Xy_valid_meta["clust_id"] = kmeans.predict(np.array(Xy_valid_meta[HBA_probs]))
    Xy_valid_feats["clust_id"] = Xy_valid_meta["clust_id"]
else:
    Xy_valid_meta = (train_meta[valid_rows_bool])[::VALID_DOWNSEL].copy()
    Xy_valid_meta = Xy_valid_meta.reset_index().drop(columns=["index"])
    print("Number of samples used for Validation =", len(Xy_valid_meta))

    Xy_valid_feats = assemble_features(
        Xy_valid_meta, traintest="validation", smooth_width=SMOOTH_WIDTH
    )
    Xy_valid_meta.to_csv(
        "Xy_valid_meta.csv", header=True, index=False, float_format="%.6f"
    )
    Xy_valid_feats.to_csv(
        "Xy_valid_feats.csv", header=True, index=False, float_format="%.6f"
    )




## === cell 33
print(Xy_valid_feats.shape)
Xy_valid_feats.iloc[-5:, 80:90]




## === cell 34
Xy_train_feats.iloc[-5:, 80:90]




## === cell 35
assert "clust_id" in Xy_valid_feats.columns
assert len(Xy_valid_feats) == len(Xy_valid_meta)




## === cell 36
X = Xy_train_feats.drop(columns=["clust_id"])
y = Xy_train_feats.clust_id

Xlr = X.drop(columns=X.columns[-10:])

if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : 83 + 1]
    lrmodel1 = LogisticRegression(
        penalty=LR_REGU,
        C=LR1_C + 1.0,
        solver="saga",
        max_iter=1500,
        multi_class="multinomial",  # ovr or multinomial
        n_jobs=-1,
    ).fit(Xlr1, y)
    plt.figure(figsize=(8, 4))
    plt.plot(lrmodel1.coef_.T, "-", alpha=0.5)
    plt.plot(lrmodel1.coef_.T, ".", alpha=1.0)
    plt.title("LR1 coefficients, C=" + str(LR1_C) + " (colored by cluster)")
    plt.savefig("lr1_coefs.png", bbox_inches="tight")
    plt.close()

    thecoefs = lrmodel1.coef_.T.flatten()
    print("Non-zero coef.s fraction: {:.3f}".format(sum(thecoefs != 0) / len(thecoefs)))
    print("\nLR1 model score for X,y = {:.1f}%\n".format(100 * lrmodel1.score(Xlr1, y)))

if USE_LR2:
    Xlr2 = Xlr.iloc[:, 84 : 179 + 1]
    lrmodel2 = LogisticRegression(
        penalty=LR_REGU,
        C=LR2_C + 1.0,
        solver="saga",
        max_iter=1500,
        multi_class="multinomial",  # ovr or multinomial
        n_jobs=-1,
    ).fit(Xlr2, y)
    plt.figure(figsize=(8, 4))
    plt.plot(lrmodel2.coef_.T, "-", alpha=0.5)
    plt.plot(lrmodel2.coef_.T, ".", alpha=1.0)
    plt.title("LR2 coefficients, C=" + str(LR2_C) + " (colored by cluster)")
    plt.savefig("lr2_coefs.png", bbox_inches="tight")
    plt.close()

    thecoefs = lrmodel2.coef_.T.flatten()
    print("Non-zero coef.s fraction: {:.3f}".format(sum(thecoefs != 0) / len(thecoefs)))
    print("\nLR2 model score for X,y = {:.1f}%\n".format(100 * lrmodel2.score(Xlr2, y)))




## === cell 37
Xy_train_wLRfeats = Xy_train_feats.copy()
X = Xy_train_feats.drop(columns=["clust_id"])
Xlr = X.drop(columns=X.columns[-10:])
if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : 83 + 1]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    for iadd in range(NUM_CLUSTS):
        Xy_train_wLRfeats["lrMid" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            np.random.rand(len(lrprobas)) - 0.5
        )
    plt.figure(figsize=(7, 2.5))
    plt.hist(np.max(lrmodel1.predict_proba(Xlr1), axis=1), bins=50)
    plt.xlim(-0.5 * LR_BLUR, 1.01)
    plt.title("Training max LR1 proba values (pre-blur)")
    plt.savefig("hist_lr1_train_maxproba.png", bbox_inches="tight")
    plt.close()
if USE_LR2:
    Xlr2 = Xlr.iloc[:, 84 : 179 + 1]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    for iadd in range(NUM_CLUSTS):
        Xy_train_wLRfeats["lrFFT" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            np.random.rand(len(lrprobas)) - 0.5
        )
    plt.figure(figsize=(7, 2.5))
    plt.hist(np.max(lrmodel2.predict_proba(Xlr2), axis=1), bins=50)
    plt.xlim(-0.5 * LR_BLUR, 1.01)
    plt.title("Training max LR2 proba values (pre-blur)")
    plt.savefig("hist_lr2_train_maxproba.png", bbox_inches="tight")
    plt.close()

Xy_valid_wLRfeats = Xy_valid_feats.copy()
X = Xy_valid_feats.drop(columns=["clust_id"])
Xlr = X.drop(columns=X.columns[-10:])
if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : 83 + 1]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    for iadd in range(NUM_CLUSTS):
        Xy_valid_wLRfeats["lrMid" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            np.random.rand(len(lrprobas)) - 0.5
        )
    plt.figure(figsize=(7, 2.5))
    plt.hist(np.max(lrmodel1.predict_proba(Xlr1), axis=1), bins=50)
    plt.xlim(-0.5 * LR_BLUR, 1.01)
    plt.title("Validation max LR1 proba values (pre-blur)")
    plt.savefig("hist_lr1_valid_maxproba.png", bbox_inches="tight")
    plt.close()
if USE_LR2:
    Xlr2 = Xlr.iloc[:, 84 : 179 + 1]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    for iadd in range(NUM_CLUSTS):
        Xy_valid_wLRfeats["lrFFT" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            np.random.rand(len(lrprobas)) - 0.5
        )
    plt.figure(figsize=(7, 2.5))
    plt.hist(np.max(lrmodel2.predict_proba(Xlr2), axis=1), bins=50)
    plt.xlim(-0.5 * LR_BLUR, 1.01)
    plt.title("Validation max LR2 proba values (pre-blur)")
    plt.savefig("hist_lr2_valid_maxproba.png", bbox_inches="tight")
    plt.close()




## === cell 38
assert list(Xy_train_wLRfeats.drop(columns=["clust_id"]).columns) == list(
    Xy_valid_wLRfeats.drop(columns=["clust_id"]).columns
)




## === cell 39
X = Xy_train_wLRfeats.drop(columns=["clust_id"])
y = Xy_train_wLRfeats.clust_id

ave_oob = []
rfmodel = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=5,  # 7
    max_features=0.2,  # 0.3
    max_samples=0.9,  # 0.9
    oob_score=True,
    class_weight="balanced_subsample",
    n_jobs=-1,
    verbose=0,
    random_state=0,
).fit(X, y)
ave_oob.append(rfmodel.oob_score_)

sort_inds = rfmodel.feature_importances_.argsort()
plt.figure(figsize=(4, 7))
plt.barh(rfmodel.feature_names_in_[sort_inds], rfmodel.feature_importances_[sort_inds])
plt.ylim(len(sort_inds) - 40, len(sort_inds) + 0.2)
plt.title("Feature Importances (top 40)")
plt.savefig("rf_feature_importances.png", bbox_inches="tight")
plt.close()

print("\nRF model ave OOB score = {:.1f}%".format(100 * np.mean(ave_oob)))
print("\nRF model score for X,y = {:.1f}%\n".format(100 * rfmodel.score(X, y)))




## === cell 40
Xy_train_meta["pred_id"] = rfmodel.predict(Xy_train_wLRfeats.drop(columns=["clust_id"]))

probas_train = rfmodel.predict_proba(X)
maxprobs_train = np.max(probas_train, axis=1)
plt.figure(figsize=(7, 2.5))
plt.hist(maxprobs_train, bins=50)
plt.xlim(0.0, 1.0)
plt.title("Histogram of max(proba) for model-training samples")
plt.savefig("hist_train_maxproba.png", bbox_inches="tight")
plt.close()

solution = Xy_train_meta[["eeg_id"]].copy()
for col_pre in HBA_names:
    solution[col_pre + "_vote"] = Xy_train_meta[col_pre + "_prob"]

submission = solution.copy()

if USE_TAMED:
    pred_ids = Xy_train_meta["pred_id"]  # putting clust_id gives the ideal value
    best_fracs, best_centers = find_best_tamed_kl()
    for iprob in range(HBA_number):
        this_col_probs = best_centers[:, iprob]
        submission[HBA_votes[iprob]] = this_col_probs[pred_ids]
    print("Tamed fractions:\n", best_fracs, "\nand centers:\n", best_centers)

else:
    pred_probs = probas_train @ clust_centers  # matrix multiply
    for iprob in range(HBA_number):
        submission[HBA_votes[iprob]] = pred_probs[:, iprob]
    print("Predicted probabilites are proba-weighted cluster centers.")

submission = normalize_rows(submission, HBA_votes)
this_kl = kld_score(solution[HBA_votes], submission[HBA_votes])
print("\nKL from predicted centers: {:.4f}".format(this_kl))




## === cell 41
pd.crosstab(Xy_train_meta["pred_id"], Xy_train_meta["clust_id"])




## === cell 42
X_valid = Xy_valid_wLRfeats.drop(columns=["clust_id"])




## === cell 43
Xy_valid_meta["pred_id"] = rfmodel.predict(Xy_valid_wLRfeats.drop(columns=["clust_id"]))

probas_valid = rfmodel.predict_proba(X_valid)
maxprobs_valid = np.max(probas_valid, axis=1)
plt.figure(figsize=(7, 2.5))
plt.hist(maxprobs_valid, bins=50)
plt.xlim(0.0, 1.0)
plt.title("Histogram of max(proba) for validation samples")
plt.savefig("hist_valid_maxproba.png", bbox_inches="tight")
plt.close()

solution = Xy_valid_meta[["eeg_id"]].copy()
for col_pre in HBA_names:
    solution[col_pre + "_vote"] = Xy_valid_meta[col_pre + "_prob"]

submission = solution.copy()

if USE_TAMED:
    pred_ids = Xy_valid_meta["pred_id"]
    best_fracs, best_centers = find_best_tamed_kl()
    for iprob in range(HBA_number):
        this_col_probs = best_centers[:, iprob]
        submission[HBA_votes[iprob]] = this_col_probs[pred_ids]
    print("Tamed fractions:\n", best_fracs, "\nand centers:\n", best_centers)

else:
    pred_probs = probas_valid @ clust_centers  # matrix multiply
    for iprob in range(HBA_number):
        submission[HBA_votes[iprob]] = pred_probs[:, iprob]
    print("Predicted probabilites are proba-weighted cluster centers.")

submission = normalize_rows(submission, HBA_votes)
this_kl = kld_score(solution[HBA_votes], submission[HBA_votes])
print("\nKL from predicted centers: {:.4f}".format(this_kl))




## === cell 44
pd.crosstab(Xy_valid_meta["pred_id"], Xy_valid_meta["clust_id"])




## === cell 45
clust_centers = clust_centers / clust_centers.sum(axis=1, keepdims=True)




## === cell 46
assert "spectrogram_id" in test_meta.columns
assert "spectrogram_label_offset_seconds" in test_meta.columns




## === cell 47
import joblib

joblib.dump(rfmodel, "rfmodel.joblib")




## === cell 48
TRAIN_FEATURE_COLS = list(Xy_train_wLRfeats.drop(columns=["clust_id"]).columns)




## === cell 49
Xy_test_feats = assemble_features(
    test_meta, traintest="test", smooth_width=SMOOTH_WIDTH
)

Xy_test_wLRfeats = Xy_test_feats.copy()

if USE_LR1 or USE_LR2:
    Xlr = Xy_test_feats.iloc[:, :-10]  # same as drop last 10 columns, faster

if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : 83 + 1]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    for iadd in range(NUM_CLUSTS):
        Xy_test_wLRfeats["lrMid" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            np.random.rand(len(lrprobas)) - 0.5
        )
if USE_LR2:
    Xlr2 = Xlr.iloc[:, 84 : 179 + 1]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    for iadd in range(NUM_CLUSTS):
        Xy_test_wLRfeats["lrFFT" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            np.random.rand(len(lrprobas)) - 0.5
        )

X_test = Xy_test_wLRfeats
missing = [c for c in TRAIN_FEATURE_COLS if c not in X_test.columns]
if missing:
    raise RuntimeError(
        f"Test features missing columns seen in train: {missing[:5]} ... total {len(missing)}"
    )
X_test = X_test[TRAIN_FEATURE_COLS]

pred_df = test_meta[["eeg_id"]].copy()
pred_df["eeg_id"] = pred_df["eeg_id"].astype(np.int64)

if USE_TAMED:
    pred_ids = rfmodel.predict(X_test)
    for iprob in range(HBA_number):
        this_col_probs = best_centers[:, iprob]
        pred_df[HBA_votes[iprob]] = this_col_probs[pred_ids]
else:
    probas = rfmodel.predict_proba(X_test)
    pred_probs = probas @ clust_centers  # matrix multiply
    for iprob in range(HBA_number):
        pred_df[HBA_votes[iprob]] = pred_probs[:, iprob]

pred_df = normalize_rows(pred_df, HBA_votes)

pred_df = pred_df.groupby("eeg_id", as_index=False)[HBA_votes].mean()
pred_df = normalize_rows(pred_df, HBA_votes)

sample_path = above_dir + "sample_submission.csv"
sample_sub = pd.read_csv(sample_path)[["eeg_id"] + HBA_votes]
sample_sub["eeg_id"] = sample_sub["eeg_id"].astype(np.int64)

test_submit = sample_sub[["eeg_id"]].merge(pred_df, on="eeg_id", how="left", sort=False)

test_submit[HBA_votes] = test_submit[HBA_votes].fillna(1.0 / HBA_number)
test_submit = normalize_rows(test_submit, HBA_votes)
test_submit = test_submit[["eeg_id"] + HBA_votes]

assert "eeg_id" in test_submit.columns
assert list(test_submit.columns) == ["eeg_id"] + HBA_votes
assert len(test_submit) == len(sample_sub)
row_sums = test_submit[HBA_votes].sum(axis=1).to_numpy()
assert np.all(np.isfinite(row_sums))
assert (
    float(row_sums.min()) > 0.999999 - 1e-6 and float(row_sums.max()) < 1.000001 + 1e-6
)

out_path = "/kaggle/working/submission.csv"
test_submit.to_csv(out_path, header=True, index=False, na_rep="", float_format="%.6f")

print(test_submit.head())
print(
    "Row-sum stats:",
    test_submit[HBA_votes].sum(axis=1).min(),
    test_submit[HBA_votes].sum(axis=1).max(),
)
print("Wrote submission.csv with shape:", test_submit.shape, "to", out_path)
print("Using data root:", above_dir)
