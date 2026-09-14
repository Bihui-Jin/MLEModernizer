# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

1.015208464728662

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)

above_dir = "/kaggle/input/hms-harmful-brain-activity-classification/"
above_dir_preproc = "/kaggle/data/hms-2024-brain-data/"




## === cell 1
NUM_CLUSTS = 9  # 6 to 10

SMOOTH_WIDTH = 5  # Odd>1: 3,5,7,9,...

TRAIN_DOWNSEL = 1  # set to 1 to use all selected training rows
VALID_DOWNSEL = 1  # set to 1 to use all selected validation rows

USE_PREPROC = True  # Read in saved meta and features frames

USE_LR1 = True
LR1_C = 0.10  # smaller --> fewer non-zero coeff.s
USE_LR2 = True
LR2_C = 0.10
LR_BLUR = 0.10




## === cell 2
from sklearn.cluster import KMeans
import pyarrow.parquet as pq
import pyarrow as pa
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression




## === cell 3
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




## === cell 4
def kld_score(solution, submission, eps=1e-15):
    """
    Calculate the average KL divergence score.
    Assumes solution/submission have the same probability columns.
    Clips both to [eps, 1] and renormalizes per row for stability.
    """
    sol = solution.copy()
    sub = submission.copy()

    prob_cols = [c for c in sol.columns if c != "eeg_id"]

    sol_vals = sol[prob_cols].astype(float).to_numpy()
    sub_vals = sub[prob_cols].astype(float).to_numpy()

    sol_vals = np.clip(sol_vals, eps, 1.0)
    sub_vals = np.clip(sub_vals, eps, 1.0)

    sol_vals = sol_vals / sol_vals.sum(axis=1, keepdims=True)
    sub_vals = sub_vals / sub_vals.sum(axis=1, keepdims=True)

    kls = np.sum(sol_vals * np.log(sol_vals / sub_vals), axis=1)
    return float(np.mean(kls))




## === cell 5
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

    vote_arr = train_meta[HBA_votes].to_numpy(dtype=np.float64)
    total_vote = vote_arr.sum(axis=1)
    train_meta["total_vote"] = total_vote
    train_meta["max_vote"] = vote_arr.max(axis=1)

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

    probs = (vote_arr / total_vote[:, None]).astype(np.float64)
    for i, col_pre in enumerate(HBA_names):
        train_meta[col_pre + "_prob"] = probs[:, i]

    print("Calculating voting entropy values ...")
    p = np.clip(train_meta[HBA_probs].to_numpy(dtype=np.float64), 1.0e-8, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    train_meta["entropy"] = -np.sum(p * np.log(p), axis=1)

    return train_meta, test_meta




## === cell 6
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

    clstclrs = []
    for ilab in clust_ids:
        clstclrs.append(kmclrs[int(ilab)])

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




## === cell 7
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

_SPECTRO_NP_CACHE = OrderedDict()
_SPECTRO_NP_CACHE_MAX = 4096

_FEAT_NWORKERS = min(8, (os.cpu_count() or 4))


def _read_spectrogram_np(spectro_file: str) -> np.ndarray:
    """
    Convert parquet to numpy array (float64) with caching.
    Drops first column if it looks like time/index.
    """
    arr = _SPECTRO_NP_CACHE.get(spectro_file)
    if arr is not None:
        _SPECTRO_NP_CACHE.move_to_end(spectro_file)
        return arr

    tbl = pq.read_table(spectro_file)
    names = tbl.schema.names
    if len(names) > 1:
        first = names[0].lower()
        if (
            (first in ("time", "timestamp"))
            or (first.startswith("__index"))
            or (first == "index")
        ):
            tbl = tbl.select(names[1:])

    try:
        arr = tbl.to_pandas(self_destruct=True).to_numpy(copy=False)
    except TypeError:
        arr = tbl.to_pandas().to_numpy(copy=False)
    if arr.dtype != np.float64:
        arr = arr.astype(np.float64, copy=False)

    _SPECTRO_NP_CACHE[spectro_file] = arr
    if len(_SPECTRO_NP_CACHE) > _SPECTRO_NP_CACHE_MAX:
        _SPECTRO_NP_CACHE.popitem(last=False)
    return arr


def _rolling_mean_centered_vec(x: np.ndarray, w: int) -> np.ndarray:
    """
    x: (L,) float64
    returns: (L,) float64 with NaNs on edges, centered window mean elsewhere.
    """
    L = x.shape[0]
    if w <= 1:
        return x.astype(np.float64, copy=True)
    half = w // 2
    out = np.full(L, np.nan, dtype=np.float64)
    c = np.cumsum(np.insert(x.astype(np.float64, copy=False), 0, 0.0))
    valid = (c[w:] - c[:-w]) / w
    out[half : half + valid.shape[0]] = valid
    return out


def assemble_features(meta_frame, traintest="train", smooth_width=5, SHOW_PLOT=True):
    """
    Create a dataframe of spectrogram features from the meta_frame rows.
    Will include clust_id (i.e, the y) if it is in the input meta_frame.
    Assumes these are available: above_dir, num_clusts
    """
    freqs = np.array(range(100), dtype=np.float64) * 0.19525 + 0.59
    spect_trend = 150.0 / (1.0**2.3 + freqs ** (2.3))
    freqs[0] = 0.0
    freqs4 = np.array(4 * list(freqs), dtype=np.float64)
    spect_trend4 = np.array(4 * list(spect_trend), dtype=np.float64)

    if SHOW_PLOT:
        plt.figure(figsize=(10, 8))

    baseinds = np.insert(
        np.arange(int((smooth_width - 1) / 2), 100, smooth_width), 0, 0
    )
    select_inds = np.concatenate(
        (baseinds, 100 + baseinds, 200 + baseinds, 300 + baseinds)
    )
    freqs4ds = freqs4[select_inds]

    n_rows = len(meta_frame)
    n_sel = len(select_inds)
    n_features = (2 * n_sel) + 2 + 4 + 4
    feat_mat = np.empty((n_rows, n_features), dtype=np.float64)

    has_clust = "clust_id" in meta_frame.columns
    clust_ids_out = np.empty(n_rows, dtype=np.int32) if has_clust else None

    half = int((smooth_width - 1) / 2)

    print_every_nth = max([100, 100 * int(0.5 + n_rows / (100.0 * 15))])

    spectrogram_ids = meta_frame["spectrogram_id"].to_numpy()
    offsets = meta_frame["spectrogram_label_offset_seconds"].to_numpy()
    clust_in = meta_frame["clust_id"].to_numpy() if has_clust else None

    base_path = (
        (above_dir + "train_spectrograms/")
        if (traintest != "test")
        else (above_dir + "test_spectrograms/")
    )

    spectro_files = np.array(
        [base_path + str(sid) + ".parquet" for sid in spectrogram_ids], dtype=object
    )

    inv_trend4 = 1.0 / spect_trend4
    select_inds_local = select_inds  # local alias
    n_sel_local = n_sel
    n_features_local = n_features

    def _compute_one(i: int):
        spec_np = _read_spectrogram_np(spectro_files[i])
        loc_offset = int(offsets[i] / 2)

        a148 = spec_np[loc_offset + 148]
        a149 = spec_np[loc_offset + 149]
        a150 = spec_np[loc_offset + 150]
        a151 = spec_np[loc_offset + 151]
        mid_sum = a148 + a149 + a150 + a151

        middle4s = mid_sum * (0.25 * inv_trend4)
        middle4s = np.clip(middle4s, 0.001, 1000.0)
        middle4s = np.nan_to_num(middle4s, nan=0.001, posinf=0.001, neginf=0.001)

        den = (
            spec_np[loc_offset + 149 - 56]
            + spec_np[loc_offset + 149 - 40]
            + spec_np[loc_offset + 149 - 24]
            + spec_np[loc_offset + 149 + 56]
            + spec_np[loc_offset + 149 + 40]
            + spec_np[loc_offset + 149 + 24]
        )
        ratio4s = mid_sum / den
        ratio4s = np.clip((6.0 / 4.0) * ratio4s, 0.01, 100.0)
        ratio4s = np.nan_to_num(ratio4s, nan=1.0, posinf=1.0, neginf=1.0)
        ratio4spre = np.log10(ratio4s)

        spect_mean = float(np.mean(middle4s))
        spect_median = float(np.median(middle4s))

        m = middle4s.reshape(4, 100)
        the4means = m.mean(axis=1)
        the4medians = np.median(m, axis=1)

        middle4spre = np.log10(middle4s / spect_mean)

        middle4s_sm = _rolling_mean_centered_vec(middle4spre, smooth_width)
        ratio4s_sm = _rolling_mean_centered_vec(ratio4spre, smooth_width)

        if half > 0:
            for ioff in (0, 100, 200, 300):
                middle4s_sm[ioff : ioff + half] = middle4spre[ioff : ioff + half]
                ratio4s_sm[ioff : ioff + half] = ratio4spre[ioff : ioff + half]

        middle4sds = middle4s_sm[select_inds_local]
        ratio4sds = ratio4s_sm[select_inds_local]

        spect_mean_log = np.log10(spect_mean)
        spect_median_log = np.log10(spect_median)
        the4means_log = np.log10(the4means)
        the4medians_log = np.log10(the4medians)

        out_row = np.empty(n_features_local, dtype=np.float64)
        j = 0
        out_row[j : j + n_sel_local] = middle4sds
        j += n_sel_local
        out_row[j : j + n_sel_local] = ratio4sds
        j += n_sel_local
        out_row[j] = spect_mean_log
        out_row[j + 1] = spect_median_log
        j += 2
        out_row[j : j + 4] = the4means_log
        j += 4
        out_row[j : j + 4] = the4medians_log

        if has_clust:
            return i, out_row, int(clust_in[i])
        return i, out_row, None

    if SHOW_PLOT:
        last_file = None
        spec_np = None
        for i in range(n_rows):
            if spectro_files[i] != last_file:
                spec_np = _read_spectrogram_np(spectro_files[i])
                last_file = spectro_files[i]
            _, out_row, clid = _compute_one(i)
            feat_mat[i] = out_row
            if has_clust:
                clust_ids_out[i] = clid

            n_done = i + 1
            if (n_done % print_every_nth) == 0:
                print("... {} done...".format(n_done))

            this_clr = "blue"
            title_start = "Middle-8s Feature Values of " + traintest
            plt.title(title_start + " (smooth={})".format(smooth_width))
            if has_clust:
                this_clr = kmclrs[int(clid)]
                plt.title(
                    title_start
                    + " (color-coded by the "
                    + "{} clusters, smooth={})".format(num_clusts, smooth_width)
                )
            the_alpha = np.clip(0.05 * 350 / n_rows, 0.003, 0.5)
            middle4sds = out_row[:n_sel]
            the4means_log = out_row[2 * n_sel + 2 : 2 * n_sel + 2 + 4]
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
                the4means_log + (0.0 * (np.random.rand(4) - 0.5)),
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
    else:
        with ThreadPoolExecutor(max_workers=_FEAT_NWORKERS) as ex:
            for k, (i, out_row, clid) in enumerate(
                ex.map(_compute_one, range(n_rows), chunksize=128)
            ):
                feat_mat[i] = out_row
                if has_clust:
                    clust_ids_out[i] = int(clid)
                n_done = k + 1
                if (n_done % print_every_nth) == 0:
                    print("... {} done...".format(n_done))

    mid_cols = [f"m{i}" for i in range(n_sel)]
    ratio_cols = ["r" + str(c) for c in mid_cols]
    extra_cols = (
        ["Mean", "Median"]
        + [ch + "mean" for ch in the4chains]
        + [ch + "median" for ch in the4chains]
    )
    cols = mid_cols + ratio_cols + extra_cols

    feats_frame = pd.DataFrame(feat_mat, columns=cols)
    if has_clust:
        feats_frame["clust_id"] = clust_ids_out.astype(int)

    if SHOW_PLOT:
        plt.savefig("middle8s_" + traintest + "_features.png")
        plt.show()
    return feats_frame.reset_index().drop(columns=["index"])




## === cell 8
def _kld_np(sol_vals: np.ndarray, sub_vals: np.ndarray, eps: float = 1e-15) -> float:
    sol = np.clip(sol_vals, eps, 1.0)
    sub = np.clip(sub_vals, eps, 1.0)
    sol = sol / sol.sum(axis=1, keepdims=True)
    sub = sub / sub.sum(axis=1, keepdims=True)
    return float(np.mean(np.sum(sol * np.log(sol / sub), axis=1)))


def find_best_tamed_kl(solution, pred_ids, base_submission):
    """
    Adjust the taming fraction for each cluster center to optimize KL.
    NOTE: base_submission is unused (kept to preserve function signature / core logic).
    """
    mean_all_probs = np.array(
        [0.208319, 0.132120, 0.128532, 0.138913, 0.179294, 0.212822], dtype=np.float64
    )
    tamed_fracs = np.zeros(NUM_CLUSTS, dtype=np.float64)
    tamed_centers = np.tile(mean_all_probs[None, :], (NUM_CLUSTS, 1)).astype(
        np.float64, copy=True
    )

    best_fracs = tamed_fracs.copy()
    best_centers = tamed_centers.copy()

    pred_ids_np = np.asarray(pred_ids, dtype=int)

    prob_cols = [c for c in solution.columns if c != "eeg_id"]
    sol_vals = solution[prob_cols].astype(float).to_numpy()

    frac_grid = np.round(np.arange(0.01, 1.001, 0.02), 2)

    for iclust in range(NUM_CLUSTS):
        last_kl = 10.0
        for this_frac in frac_grid:
            tamed_fracs[iclust] = float(this_frac)
            this_cent = (
                tamed_fracs[iclust] * clust_centers[iclust, :]
                + (1.0 - tamed_fracs[iclust]) * mean_all_probs
            )
            this_cent = np.clip(this_cent, 1e-12, 1.0)
            this_cent = this_cent / np.sum(this_cent)
            tamed_centers[iclust, :] = this_cent

            sub_vals = tamed_centers[pred_ids_np]

            this_kl = _kld_np(sol_vals, sub_vals)
            if this_kl < last_kl:
                best_fracs = tamed_fracs.copy()
                best_centers = tamed_centers.copy()
                last_kl = this_kl
            else:
                tamed_fracs[iclust] = best_fracs[iclust]
                tamed_centers[iclust, :] = best_centers[iclust, :]
                break
    return best_fracs, best_centers




## === cell 9
train_meta, test_meta = read_hms_meta()




## === cell 10
SHOW_DIAGNOSTIC_PLOTS = False

if SHOW_DIAGNOSTIC_PLOTS:
    plt.figure(figsize=(5, 2))
    plt.hist(train_meta["spectrogram_sub_id"], bins=55, log=True)
    plt.title("Histogram of spectrogram_sub_id")
    plt.show()

    plt.figure(figsize=(5, 2))
    plt.hist(train_meta["eeg_sub_id"], bins=55, log=True)
    plt.title("Histogram of eeg_sub_id")
    plt.show()




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
if SHOW_DIAGNOSTIC_PLOTS:
    plt.figure(figsize=(6, 3))
    plt.hist(train_meta.loc[clust_rows_bool, "total_vote"], bins=55, log=True)
    plt.title("Histogram of total_vote in HBA samples clustered")
    plt.show()

prob_array = np.array(prob_vectors)
kmeans = KMeans(
    n_clusters=num_clusts, init="k-means++", n_init=10, max_iter=300, random_state=0
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

if SHOW_DIAGNOSTIC_PLOTS:
    all_probs = train_meta[HBA_probs]
    all_ids = train_meta["clust_id"]

    kmclrs = prob_prob_scatter("Seizure", "GPD", all_probs, all_ids, iclust_of_order)
    kmclrs = prob_prob_scatter("LPD", "GRDA", all_probs, all_ids, iclust_of_order)
    kmclrs = prob_prob_scatter("LRDA", "Other", all_probs, all_ids, iclust_of_order)

    clust_counts = train_meta.clust_id.value_counts()

    iorder_of_clust = num_clusts * [-1]
    for iord, iclust in enumerate(iclust_of_order):
        iorder_of_clust[int(iclust)] = iord
        if iord == 0:
            print("The close-to-unit-vector cluster centers:")
        if iord == 6:
            print("The Hybrid cluster centers:")
        plt.figure(figsize=(5, 1))
        plt.bar(
            HBA_expert_names, clust_centers[int(iclust), :], color=kmclrs[int(iclust)]
        )
        plt.ylim(-0.01, 1.01)
        plt.title(
            kmnames[iord]
            + "  kmclust={} has {} samples".format(
                int(iclust), clust_counts[int(iclust)]
            ),
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




## === cell 14
solution_train = train_meta[["eeg_id"] + HBA_votes].copy()
solution_train[HBA_votes] = train_meta[HBA_probs].astype(float).to_numpy()

submission_train = solution_train.copy()
clust_ids = train_meta["clust_id"].to_numpy()
for iprob in range(HBA_number):
    this_col_probs = clust_centers[:, iprob]
    submission_train[HBA_votes[iprob]] = this_col_probs[clust_ids]

print(
    "Score if HBA samples are correctly assigned cluster prob.s:",
    np.round(kld_score(solution_train, submission_train), 4),
)




## === cell 15
train_rows_bool = (
    (train_meta.eeg_sub_id < (33 + 1)) & (train_meta.eeg_sub_id % 5 == 3)
) | ((train_meta.eeg_sub_id == 0) & (((train_meta.eeg_id % 23) % 8) > 1))
print("Number of Training rows:", int(train_rows_bool.sum()))

valid_rows_bool = (
    (train_meta.eeg_sub_id < (44 + 1)) & (train_meta.eeg_sub_id % 19 == 6)
) | ((train_meta.eeg_sub_id == 0) & (((train_meta.eeg_id % 23) % 8) < 2))
print("Number of Validation rows:", int(valid_rows_bool.sum()))




## === cell 16
def _safe_read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


if USE_PREPROC and (not os.path.isdir(above_dir_preproc)):
    print(
        "Preprocessed directory not found; falling back to on-the-fly feature assembly."
    )
    USE_PREPROC = False

if USE_PREPROC:
    Xy_train_meta = _safe_read_csv(above_dir_preproc + "Xy_train_meta_v47.csv")
    Xy_train_feats = _safe_read_csv(above_dir_preproc + "Xy_train_feats_v47.csv")
    if (Xy_train_meta is None) or (Xy_train_feats is None):
        print(
            "Preprocessed train files not found; falling back to on-the-fly feature assembly."
        )
        USE_PREPROC = False

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
else:
    Xy_train_meta["clust_id"] = kmeans.predict(np.array(Xy_train_meta[HBA_probs]))
    Xy_train_feats["clust_id"] = Xy_train_meta["clust_id"]
    SMOOTH_WIDTH = 5




## === cell 17
if USE_PREPROC:
    Xy_valid_meta = _safe_read_csv(above_dir_preproc + "Xy_valid_meta_v47.csv")
    Xy_valid_feats = _safe_read_csv(above_dir_preproc + "Xy_valid_feats_v47.csv")
    if (Xy_valid_meta is None) or (Xy_valid_feats is None):
        print(
            "Preprocessed valid files not found; falling back to on-the-fly feature assembly."
        )
        USE_PREPROC = False

if not USE_PREPROC:
    Xy_valid_meta = (train_meta[valid_rows_bool])[::VALID_DOWNSEL].copy()
    Xy_valid_meta = Xy_valid_meta.reset_index().drop(columns=["index"])
    print("Number of samples used for Validation =", len(Xy_valid_meta))

    Xy_valid_feats = assemble_features(
        Xy_valid_meta,
        traintest="validation",
        smooth_width=SMOOTH_WIDTH,
        SHOW_PLOT=False,
    )
    Xy_valid_meta.to_csv(
        "Xy_valid_meta.csv", header=True, index=False, float_format="%.6f"
    )
    Xy_valid_feats.to_csv(
        "Xy_valid_feats.csv", header=True, index=False, float_format="%.6f"
    )
else:
    Xy_valid_meta["clust_id"] = kmeans.predict(np.array(Xy_valid_meta[HBA_probs]))
    Xy_valid_feats["clust_id"] = Xy_valid_meta["clust_id"]




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
        random_state=0,
    ).fit(Xlr1, y)

    if SHOW_DIAGNOSTIC_PLOTS:
        plt.figure(figsize=(8, 4))
        plt.plot(lrmodel1.coef_.T, "-", alpha=0.5)
        plt.plot(lrmodel1.coef_.T, ".", alpha=1.0)
        plt.title("LR1 coefficients, C=" + str(LR1_C) + " (colored by cluster)")
        plt.show()

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
        random_state=0,
    ).fit(Xlr2, y)

    if SHOW_DIAGNOSTIC_PLOTS:
        plt.figure(figsize=(8, 4))
        plt.plot(lrmodel2.coef_.T, "-", alpha=0.5)
        plt.plot(lrmodel2.coef_.T, ".", alpha=1.0)
        plt.title("LR2 coefficients, C=" + str(LR2_C) + " (colored by cluster)")
        plt.show()

    print("\nLR2 model score for X,y = {:.1f}%\n".format(100 * lrmodel2.score(Xlr2, y)))




## === cell 19
def _clip_and_renorm_rows(a: np.ndarray, lo: float = 1e-6) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    a = np.nan_to_num(a, nan=lo, posinf=lo, neginf=lo)
    a = np.clip(a, lo, 1.0)
    a = a / a.sum(axis=1, keepdims=True)
    return a


Xy_train_wLRfeats = Xy_train_feats.copy()
X = Xy_train_feats.drop(columns=["clust_id"])
Xlr = X.drop(columns=X.columns[-10:])

if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : int(len(Xlr.columns) / 2)]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    blur = LR_BLUR * (np.random.rand(*lrprobas.shape) - 0.5)
    lr_add = _clip_and_renorm_rows(lrprobas + blur)
    Xy_train_wLRfeats[[f"lr{i}" for i in range(NUM_CLUSTS)]] = lr_add

if USE_LR2:
    Xlr2 = Xlr.iloc[:, int(len(Xlr.columns) / 2) :]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    blur = LR_BLUR * (np.random.rand(*lrprobas.shape) - 0.5)
    lr_add = _clip_and_renorm_rows(lrprobas + blur)
    Xy_train_wLRfeats[[f"rlr{i}" for i in range(NUM_CLUSTS)]] = lr_add

Xy_valid_wLRfeats = Xy_valid_feats.copy()
X = Xy_valid_feats.drop(columns=["clust_id"])
Xlr = X.drop(columns=X.columns[-10:])

if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : int(len(Xlr.columns) / 2)]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    blur = LR_BLUR * (np.random.rand(*lrprobas.shape) - 0.5)
    lr_add = _clip_and_renorm_rows(lrprobas + blur)
    Xy_valid_wLRfeats[[f"lr{i}" for i in range(NUM_CLUSTS)]] = lr_add

if USE_LR2:
    Xlr2 = Xlr.iloc[:, int(len(Xlr.columns) / 2) :]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    blur = LR_BLUR * (np.random.rand(*lrprobas.shape) - 0.5)
    lr_add = _clip_and_renorm_rows(lrprobas + blur)
    Xy_valid_wLRfeats[[f"rlr{i}" for i in range(NUM_CLUSTS)]] = lr_add




## === cell 20
X = Xy_train_wLRfeats.drop(columns=["clust_id"])
y = Xy_train_wLRfeats.clust_id

rfmodel = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=5,
    max_features=0.2,
    max_samples=0.9,
    oob_score=True,
    class_weight="balanced_subsample",
    n_jobs=-1,
    verbose=0,
    random_state=2,
).fit(X, y)

if SHOW_DIAGNOSTIC_PLOTS:
    sort_inds = rfmodel.feature_importances_.argsort()
    plt.figure(figsize=(4, 7))
    plt.barh(
        rfmodel.feature_names_in_[sort_inds], rfmodel.feature_importances_[sort_inds]
    )
    plt.ylim(len(sort_inds) - 40, len(sort_inds) + 0.2)
    plt.title("Feature Importances (top 40)")
    plt.show()

print("\nRF model OOB score = {:.1f}%".format(100 * rfmodel.oob_score_))
print("\nRF model score for X,y = {:.1f}%\n".format(100 * rfmodel.score(X, y)))




## === cell 21
Xy_train_meta["pred_id"] = rfmodel.predict(Xy_train_wLRfeats.drop(columns=["clust_id"]))

solution = Xy_train_meta[["eeg_id"] + HBA_votes].copy()
solution[HBA_votes] = Xy_train_meta[HBA_probs].astype(float).to_numpy()

base_submission = solution.copy()
pred_ids = Xy_train_meta["pred_id"]

best_fracs_train, best_centers_train = find_best_tamed_kl(
    solution, pred_ids, base_submission
)

submission = base_submission.copy()
for iprob in range(HBA_number):
    this_col_probs = best_centers_train[:, iprob]
    submission[HBA_votes[iprob]] = this_col_probs[np.asarray(pred_ids, dtype=int)]

this_kl = kld_score(solution, submission)
print(
    "Tamed fractions (train-subsample):\n",
    best_fracs_train,
    "\nand centers:\n",
    best_centers_train,
)
print("\nKL from tamed centers (train-subsample): {:.4f}".format(this_kl))




## === cell 22
Xy_valid_meta["pred_id"] = rfmodel.predict(Xy_valid_wLRfeats.drop(columns=["clust_id"]))

solution = Xy_valid_meta[["eeg_id"] + HBA_votes].copy()
solution[HBA_votes] = Xy_valid_meta[HBA_probs].astype(float).to_numpy()

base_submission = solution.copy()
pred_ids = Xy_valid_meta["pred_id"]

best_fracs_valid, best_centers_valid = find_best_tamed_kl(
    solution, pred_ids, base_submission
)

submission = base_submission.copy()
for iprob in range(HBA_number):
    this_col_probs = best_centers_valid[:, iprob]
    submission[HBA_votes[iprob]] = this_col_probs[np.asarray(pred_ids, dtype=int)]

this_kl = kld_score(solution, submission)
print(
    "\nTamed fractions (valid-subsample):\n",
    best_fracs_valid,
    "\nand centers:\n",
    best_centers_valid,
)
print("\nKL from tamed centers (valid-subsample): {:.4f}".format(this_kl))




## === cell 23
Xy_test_feats = assemble_features(
    test_meta, traintest="test", smooth_width=SMOOTH_WIDTH, SHOW_PLOT=False
)

Xy_test_wLRfeats = Xy_test_feats.copy()
Xlr = Xy_test_feats.drop(columns=Xy_test_feats.columns[-10:])

if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : int(len(Xlr.columns) / 2)]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    blur = LR_BLUR * (np.random.rand(*lrprobas.shape) - 0.5)
    lr_add = _clip_and_renorm_rows(lrprobas + blur)
    Xy_test_wLRfeats[[f"lr{i}" for i in range(NUM_CLUSTS)]] = lr_add

if USE_LR2:
    Xlr2 = Xlr.iloc[:, int(len(Xlr.columns) / 2) :]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    blur = LR_BLUR * (np.random.rand(*lrprobas.shape) - 0.5)
    lr_add = _clip_and_renorm_rows(lrprobas + blur)
    Xy_test_wLRfeats[[f"rlr{i}" for i in range(NUM_CLUSTS)]] = lr_add

if "clust_id" in Xy_test_wLRfeats.columns:
    pred_ids = rfmodel.predict(Xy_test_wLRfeats.drop(columns=["clust_id"]))
else:
    pred_ids = rfmodel.predict(Xy_test_wLRfeats)

sample_sub = pd.read_csv(above_dir + "sample_submission.csv")[
    ["eeg_id"] + HBA_votes
].copy()

best_centers = best_centers_valid.copy()
pred_probs = best_centers[np.asarray(pred_ids, dtype=int)]
pred_probs = np.nan_to_num(
    pred_probs, nan=1.0 / HBA_number, posinf=1.0 / HBA_number, neginf=1.0 / HBA_number
)
pred_probs = np.clip(pred_probs, 1e-15, 1.0)
pred_probs = pred_probs / pred_probs.sum(axis=1, keepdims=True)

test_pred = pd.DataFrame(pred_probs, columns=HBA_votes)
test_pred.insert(0, "eeg_id", test_meta["eeg_id"].to_numpy())

test_submit = sample_sub[["eeg_id"]].merge(test_pred, on="eeg_id", how="left")

missing = test_submit[HBA_votes[0]].isna().to_numpy()
if missing.any():
    test_submit.loc[missing, HBA_votes] = 1.0 / HBA_number

out = test_submit[HBA_votes].to_numpy(dtype=np.float64, copy=False)
out = np.nan_to_num(
    out, nan=1.0 / HBA_number, posinf=1.0 / HBA_number, neginf=1.0 / HBA_number
)
out = np.clip(out, 1e-15, 1.0)
out = out / out.sum(axis=1, keepdims=True)
test_submit[HBA_votes] = out

assert len(test_submit) == len(sample_sub), "Submission row count mismatch."
assert (
    test_submit["eeg_id"].to_numpy() == sample_sub["eeg_id"].to_numpy()
).all(), "eeg_id order mismatch."

print(test_submit.head())

test_submit.to_csv(
    "submission.csv", header=True, index=False, na_rep="", float_format="%.6f"
)
print("\nWrote submission.csv with shape:", test_submit.shape)
print(
    "Row-sum check (min/max):",
    float(test_submit[HBA_votes].sum(axis=1).min()),
    float(test_submit[HBA_votes].sum(axis=1).max()),
)

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/1107977745.py in <cell line: 0>()
     58 test_submit[HBA_votes] = out
     59 
---> 60 assert len(test_submit) == len(sample_sub), "Submission row count mismatch."
     61 assert (
     62     test_submit["eeg_id"].to_numpy() == sample_sub["eeg_id"].to_numpy()

AssertionError: Submission row count mismatch.

## --- ERROR in outputing the csv:
Invalid submission: Submission must contain eeg_id column
