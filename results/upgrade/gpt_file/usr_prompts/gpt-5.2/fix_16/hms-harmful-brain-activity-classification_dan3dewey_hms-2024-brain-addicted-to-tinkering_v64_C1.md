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

1.0116395317562643

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")



## === cell 1
import numpy as np

np.random.seed(42)
_rng = np.random.RandomState(42)  # deterministic RNG for LR_BLUR noise (stability)



## === cell 2
pass



## === cell 3
NUM_CLUSTS = 6  # 6 to 10

TRAIN_DOWNSEL = 1  # large values for code test; set to 1 to output all.
VALID_DOWNSEL = 1  #  "
SMOOTH_WIDTH = 5  # Odd>1: 3,5,7,9,...
USE_PREPROC = True  # Read-in saved meta and features frames

USE_LR1 = True
LR1_C = 1.0  # smaller --> fewer non-zero coeff.s
USE_LR2 = True
LR2_C = 1.0
LR_BLUR = 0.10

_candidate_above_dirs = [
    "../input/hms-harmful-brain-activity-classification/",
    "/kaggle/input/hms-harmful-brain-activity-classification/",
    "/kaggle/data/hms-harmful-brain-activity-classification/",
    "/kaggle/data/input/hms-harmful-brain-activity-classification/",
]
above_dir = None
for _d in _candidate_above_dirs:
    if os.path.exists(os.path.join(_d, "train.csv")) and os.path.exists(
        os.path.join(_d, "test.csv")
    ):
        above_dir = _d
        break
if above_dir is None:
    above_dir = "../input/hms-harmful-brain-activity-classification/"

_candidate_preproc_dirs = [
    "../input/hms-2024-brain-data/",
    "/kaggle/input/hms-2024-brain-data/",
    os.path.join(above_dir, ""),  # allow above_dir root
    os.path.join(above_dir, "hms-2024-brain-data/"),
]
above_dir_preproc = None
for _d in _candidate_preproc_dirs:
    if _d and os.path.exists(_d):
        above_dir_preproc = _d if _d.endswith("/") else (_d + "/")
        break
if above_dir_preproc is None:
    above_dir_preproc = "../input/hms-2024-brain-data/"



## === cell 4
_preproc_needed = [
    os.path.join(above_dir_preproc, "Xy_train_meta_v62.csv"),
    os.path.join(above_dir_preproc, "Xy_train_feats_v62.csv"),
    os.path.join(above_dir_preproc, "Xy_valid_meta_v62.csv"),
    os.path.join(above_dir_preproc, "Xy_valid_feats_v62.csv"),
]

if USE_PREPROC and (not all(os.path.exists(p) for p in _preproc_needed)):
    found_dir = None
    for root, _, files in os.walk(above_dir):
        fnset = set(files)
        if {
            "Xy_train_meta_v62.csv",
            "Xy_train_feats_v62.csv",
            "Xy_valid_meta_v62.csv",
            "Xy_valid_feats_v62.csv",
        }.issubset(fnset):
            found_dir = root
            break
    if found_dir is not None:
        above_dir_preproc = found_dir if found_dir.endswith("/") else (found_dir + "/")
        _preproc_needed = [
            os.path.join(above_dir_preproc, "Xy_train_meta_v62.csv"),
            os.path.join(above_dir_preproc, "Xy_train_feats_v62.csv"),
            os.path.join(above_dir_preproc, "Xy_valid_meta_v62.csv"),
            os.path.join(above_dir_preproc, "Xy_valid_feats_v62.csv"),
        ]

if USE_PREPROC and (not all(os.path.exists(p) for p in _preproc_needed)):
    print(
        "Preprocessed files not found; switching USE_PREPROC=False (may be slow / may time out)."
    )
    USE_PREPROC = False
else:
    if USE_PREPROC:
        print("Using preprocessed feature files from:", above_dir_preproc)



## === cell 5
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
    print("sklearn-intelex patch enabled.")
except Exception as _e:
    print("sklearn-intelex patch not enabled:", repr(_e))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import pyarrow
import pyarrow.parquet as pq

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



## === cell 6
pass



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
def kld_score(solution, submission):
    """
    Calculate the average KL divergence score.
    Uses clipping for numerical safety (avoid log(0) or division by 0).
    Assumes both have identical probability columns.
    """
    eps = 1e-15
    sumsum = 0.0
    for prob_col in solution.columns.values:
        p = np.clip(solution[prob_col].to_numpy(dtype=float), eps, 1.0)
        q = np.clip(submission[prob_col].to_numpy(dtype=float), eps, 1.0)
        sumsum += np.nansum(-1.0 * p * np.log(q / p))
    return sumsum / (len(solution) if len(solution) else 1)




## === cell 9
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

    for col_pre in HBA_names:
        train_meta[col_pre + "_prob"] = (
            train_meta[col_pre + "_vote"] / train_meta["total_vote"]
        )

    print("Calculating voting entropy values ...")
    probs = np.clip(train_meta[HBA_probs].to_numpy(dtype=float), 1.0e-8, 1.0)
    train_meta["entropy"] = np.nansum(probs * (-np.log(probs)), axis=1)

    return train_meta, test_meta




## === cell 10
def prob_prob_scatter(name1, name2, probs2plot, clust_ids, iclust_order=[0]):
    """
    Make a prob1 vs prob2 scatter plot.
    (Kept for compatibility; not used in the scoring pipeline.)
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




## === cell 11
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

_spectro_cache = OrderedDict()
_SPECTRO_CACHE_MAX = 16

_spectro_cols_cache = {}  # file -> list[str] cols


def _spectro_columns_for_file(spectro_file: str):
    cols = _spectro_cols_cache.get(spectro_file)
    if cols is not None:
        return cols

    schema = pq.read_schema(spectro_file)
    names = list(schema.names)

    if "time" not in names:
        raise ValueError(f"'time' column not found in spectrogram file: {spectro_file}")

    chain_cols = []
    for ch in the4chains:
        pref = f"{ch}_"
        these = [n for n in names if n.startswith(pref)]

        def _freq(nm: str) -> float:
            try:
                return float(nm.split("_", 1)[1])
            except Exception:
                return float("inf")

        these = sorted(these, key=_freq)
        if len(these) < 100:
            raise ValueError(
                f"Expected >=100 columns for chain {ch} in {spectro_file}, found {len(these)}"
            )
        chain_cols.extend(these[:100])

    cols = ["time"] + chain_cols  # total 401 columns
    _spectro_cols_cache[spectro_file] = cols
    return cols


def _read_spectro_np(spectro_file: str) -> np.ndarray:
    arr = _spectro_cache.get(spectro_file)
    if arr is not None:
        _spectro_cache.move_to_end(spectro_file)
        return arr

    cols = _spectro_columns_for_file(spectro_file)

    tbl = pq.read_table(spectro_file, columns=cols)
    col_arrays = [
        tbl.column(i).combine_chunks().to_numpy(zero_copy_only=False)
        for i in range(tbl.num_columns)
    ]
    arr = np.column_stack(col_arrays).astype(np.float64, copy=False)

    _spectro_cache[spectro_file] = arr
    if len(_spectro_cache) > _SPECTRO_CACHE_MAX:
        _spectro_cache.popitem(last=False)
    return arr


def _rolling_mean_centered_np(x: np.ndarray, w: int) -> np.ndarray:
    """Centered rolling mean with min_periods=w, matching pandas.rolling(w, min_periods=w, center=True).mean()."""
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if w <= 1:
        out[:] = x
        return out
    if n < w:
        return out
    half = (w - 1) // 2
    csum = np.cumsum(np.insert(x, 0, 0.0))
    means = (csum[w:] - csum[:-w]) / w
    out[half : n - half] = means
    return out


def _rolling_mean_trailing_w2_np(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    out[1:] = 0.5 * (x[1:] + x[:-1])
    return out


def _compute_feats_for_rows(
    this_spectro_np,
    row_indices,
    spectro_offsets,
    include_clust,
    clust_vals,
    smooth_width,
    spect_trend4,
    select_inds,
    n_mid,
    n_flare,
    n_total,
    sum_spect_trend_by_bin,
    freqbins,
    fftfeatbins,
    apod_wind,
    n_fft_bins,
):
    half = int(n_fft_bins / 2)
    halfw = int((smooth_width - 1) / 2)

    feats_local = np.empty((len(row_indices), n_total), dtype=np.float64)

    the4means = np.empty(4, dtype=np.float64)
    the4medians = np.empty(4, dtype=np.float64)
    flarevals = np.empty(n_flare, dtype=np.float64)

    for j, i_idx in enumerate(row_indices):
        loc_offset = int(spectro_offsets[i_idx] / 2)
        fftlocbeg = int(loc_offset + 149 - (n_fft_bins / 2 - 1))
        fftlocend = int(loc_offset + 150 + (n_fft_bins / 2 - 1))

        mid_raw = (
            this_spectro_np[loc_offset + 148, 1:]
            + this_spectro_np[loc_offset + 149, 1:]
            + this_spectro_np[loc_offset + 150, 1:]
            + this_spectro_np[loc_offset + 151, 1:]
        ) / (4.0 * spect_trend4)

        mid_raw = np.clip(mid_raw, 0.001, 1000.0)
        mid_raw = np.where(np.isfinite(mid_raw), mid_raw, 0.001)

        spect_mean = float(np.mean(mid_raw))
        spect_median = float(np.median(mid_raw))

        for ispec, ibeg in enumerate((0, 100, 200, 300)):
            segm = mid_raw[ibeg : ibeg + 100]
            the4means[ispec] = np.mean(segm)
            the4medians[ispec] = np.median(segm)

        mid_pre = np.log10(mid_raw / spect_mean)
        mid_sm = _rolling_mean_centered_np(mid_pre, smooth_width)

        if halfw > 0:
            for ioff in (0, 100, 200, 300):
                mid_sm[ioff : ioff + halfw] = mid_pre[ioff : ioff + halfw]
        nanmask = ~np.isfinite(mid_sm)
        if np.any(nanmask):
            mid_sm[nanmask] = mid_pre[nanmask]

        feats_local[j, 0:n_mid] = mid_sm[select_inds]

        k = 0
        for ispec in range(4):
            for freqbin in freqbins:
                sum_spect_trend = sum_spect_trend_by_bin[freqbin]
                ifreqoff = freqbin + 1 + ispec * 100
                c = 1 + ifreqoff
                seg = (
                    this_spectro_np[fftlocbeg : fftlocend + 1, c - 4]
                    + this_spectro_np[fftlocbeg : fftlocend + 1, c - 2]
                    + this_spectro_np[fftlocbeg : fftlocend + 1, c]
                    + this_spectro_np[fftlocbeg : fftlocend + 1, c + 2]
                    + this_spectro_np[fftlocbeg : fftlocend + 1, c + 4]
                ) / sum_spect_trend

                seg = np.clip(seg, 0.001, 1000.0)
                seg = np.where(np.isfinite(seg), seg, 0.001)
                seg = 2.0 * seg / (seg[127] + seg[128])
                seg = apod_wind * np.clip(seg, 0.0, 10.0)

                seg = _rolling_mean_trailing_w2_np(seg)
                seg = _rolling_mean_trailing_w2_np(seg)

                amplfft = np.log10(1 + np.abs(np.fft.fft(seg))[0:half])
                for fftfeatbin in fftfeatbins:
                    flarevals[k] = amplfft[fftfeatbin]
                    k += 1

        feats_local[j, n_mid : n_mid + n_flare] = flarevals

        s0 = n_mid + n_flare
        feats_local[j, s0 + 0] = np.log10(spect_mean)
        feats_local[j, s0 + 1] = np.log10(spect_median)
        feats_local[j, s0 + 2 : s0 + 6] = np.log10(the4means)
        feats_local[j, s0 + 6 : s0 + 10] = np.log10(the4medians)

        if include_clust:
            feats_local[j, -1] = clust_vals[i_idx]

    return feats_local


def assemble_features(meta_frame, traintest="train", smooth_width=5):
    """
    Create a dataframe of spectrogram features from the meta_frame rows.
    Will include clust_id (i.e, the y) if it is in the input meta_frame.

    Change (execution): map any non-test to train_spectrograms/ (there is no validation_spectrograms/),
    preventing crashes when USE_PREPROC=False.
    """
    freqs = np.array(range(100)) * 0.19525 + 0.59
    spect_trend = 150.0 / (1.0**2.3 + freqs ** (2.3))
    baseinds = np.insert(
        np.arange(int((smooth_width - 1) / 2), 100, smooth_width), 0, 0
    )
    spect_trend4 = np.array(4 * list(spect_trend))

    n_fft_bins = 256
    freqbins = [7, 18, 60, 84]
    apod_wind = np.blackman(n_fft_bins)
    fftfeatbins = [4, 9, 16, 25, 36, 49]

    n_rows = len(meta_frame.index)
    print_every_nth = max([100, 100 * int(0.5 + n_rows / (100.0 * 15))])

    select_inds = np.concatenate(
        (baseinds, 100 + baseinds, 200 + baseinds, 300 + baseinds)
    )
    n_mid = len(select_inds)
    n_flare = 4 * len(freqbins) * len(fftfeatbins)
    n_scalar = 2 + 4 * 2
    include_clust = "clust_id" in meta_frame.columns
    n_total = n_mid + n_flare + n_scalar + (1 if include_clust else 0)

    mid_cols = [str(int(i)) for i in select_inds.tolist()]

    flarecols = []
    for ispec in range(4):
        for freqbin in freqbins:
            for fftfeatbin in fftfeatbins:
                flarecols.append(
                    "fft-"
                    + the4chains[ispec]
                    + "{:.1f}-".format(freqs[freqbin])
                    + str(fftfeatbin)
                )
    scalar_cols = (
        ["Mean", "Median"]
        + [the4chains[i] + "mean" for i in range(4)]
        + [the4chains[i] + "median" for i in range(4)]
    )
    cols = mid_cols + flarecols + scalar_cols + (["clust_id"] if include_clust else [])

    feats = np.empty((n_rows, n_total), dtype=np.float64)

    spectrogram_ids = meta_frame["spectrogram_id"].to_numpy()
    spectro_offsets = meta_frame["spectrogram_label_offset_seconds"].to_numpy()
    if include_clust:
        clust_vals = meta_frame["clust_id"].to_numpy(dtype=np.float64)
    else:
        clust_vals = None

    sum_spect_trend_by_bin = {
        freqbin: float(np.sum(spect_trend[freqbin - 4 : freqbin + 4 + 1 : 2]))
        for freqbin in freqbins
    }

    ids = spectrogram_ids.astype(np.int64)
    uniq_ids, inv = np.unique(ids, return_inverse=True)
    rows_by_uid = [[] for _ in range(len(uniq_ids))]
    for r, u in enumerate(inv):
        rows_by_uid[u].append(r)

    def _process_one(uidx: int):
        spectro_id_str = str(int(uniq_ids[uidx]))
        if traintest == "test":
            spectro_file = (
                above_dir + "test_spectrograms/" + spectro_id_str + ".parquet"
            )
        else:
            spectro_file = (
                above_dir + "train_spectrograms/" + spectro_id_str + ".parquet"
            )

        arr = _read_spectro_np(spectro_file)
        row_indices = rows_by_uid[uidx]

        feats_local = _compute_feats_for_rows(
            arr,
            row_indices,
            spectro_offsets,
            include_clust,
            clust_vals,
            smooth_width,
            spect_trend4,
            select_inds,
            n_mid,
            n_flare,
            n_total,
            sum_spect_trend_by_bin,
            freqbins,
            fftfeatbins,
            apod_wind,
            n_fft_bins,
        )
        return row_indices, feats_local

    max_workers = min(4, (os.cpu_count() or 2))
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_process_one, uidx) for uidx in range(len(uniq_ids))]
        for fut in as_completed(futs):
            row_indices, feats_local = fut.result()
            feats[np.array(row_indices, dtype=np.int64)] = feats_local
            done += len(row_indices)
            if done % print_every_nth == 0:
                print(f"... {done} done...")

    feats_frame = pd.DataFrame(feats, columns=cols)
    return feats_frame




## === cell 12
def find_best_tamed_kl():
    """
    Adjust the taming fraction for each cluster center to optimize KL
    Assumed inputs in environment:
        submission, pred_ids, solution
    Assumed useful values available:
        clust_centers, NUM_CLUSTS, HBA_number, HBA_votes
    """
    mean_all_probs = np.array(
        [0.208319, 0.132120, 0.128532, 0.138913, 0.179294, 0.212822],
        dtype=np.float64,
    )

    solP = solution[HBA_votes].to_numpy(dtype=np.float64)
    eps = 1e-15
    solP = np.clip(solP, eps, 1.0)

    pred = np.asarray(pred_ids, dtype=np.int64)
    base_centers = clust_centers.astype(np.float64, copy=True)

    tamed_fracs = np.zeros(NUM_CLUSTS, dtype=np.float64)
    tamed_centers = np.tile(mean_all_probs[None, :], (NUM_CLUSTS, 1)).astype(np.float64)

    best_fracs = tamed_fracs.copy()
    best_centers = tamed_centers.copy()

    def kl_from_centers(centers: np.ndarray) -> float:
        Q = centers[pred]
        Q = np.clip(Q, eps, 1.0)
        return float(np.sum(-solP * np.log(Q / solP)) / (len(solP) if len(solP) else 1))

    for iclust in range(NUM_CLUSTS):
        last_kl = 10.0
        for this_frac in np.arange(0.03, 1.00, 0.05):
            tamed_fracs[iclust] = this_frac
            this_cent = (
                this_frac * base_centers[iclust, :] + (1.0 - this_frac) * mean_all_probs
            )
            this_cent = np.clip(this_cent, 1e-12, 1.0)
            this_cent = this_cent / np.sum(this_cent)
            tamed_centers[iclust, :] = this_cent

            this_kl = kl_from_centers(tamed_centers)
            if this_kl < last_kl:
                best_fracs = tamed_fracs.copy()
                best_centers = tamed_centers.copy()
                last_kl = this_kl
            else:
                tamed_fracs[iclust] = best_fracs[iclust]
                tamed_centers[iclust, :] = best_centers[iclust, :]
                break
    return best_fracs, best_centers




## === cell 13
pass



## === cell 14
train_meta, test_meta = read_hms_meta()



## === cell 15
if False:
    plt.figure(figsize=(5, 2))
    plt.hist(train_meta["spectrogram_sub_id"], bins=55, log=True)
    plt.title("Histogram of spectrogram_sub_id")
    plt.show()

    plt.figure(figsize=(5, 2))
    plt.hist(train_meta["eeg_sub_id"], bins=55, log=True)
    plt.title("Histogram of eeg_sub_id")
    plt.show()



## === cell 16
pass



## === cell 17
num_clusts = NUM_CLUSTS  # 6 to 10
clust_rows_bool = train_meta.eeg_sub_id < 200



## === cell 18
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
    n_clusters=num_clusts, init="k-means++", n_init=10, max_iter=300, random_state=42
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



## === cell 19
train_meta["clust_id"] = kmeans.predict(np.array(train_meta[HBA_probs]))

all_probs = train_meta[HBA_probs]
all_ids = train_meta["clust_id"]

clust_counts = train_meta.clust_id.value_counts()

iorder_of_clust = num_clusts * [-1]
for iord, iclust in enumerate(iclust_of_order):
    iorder_of_clust[iclust] = iord



## === cell 20
pass



## === cell 21
pass



## === cell 22
solution_train = train_meta[["eeg_id"]].copy()
solution_train[HBA_votes] = train_meta[HBA_probs].to_numpy(dtype=np.float64)

submission_train = solution_train.copy()
clust_ids = train_meta["clust_id"].to_numpy(dtype=np.int64)
for iprob in range(HBA_number):
    this_col_probs = clust_centers[:, iprob]
    submission_train[HBA_votes[iprob]] = np.take(this_col_probs, clust_ids)

print(
    "Score if HBA samples are correctly assigned cluster prob.s:",
    np.round(kld_score(solution_train[HBA_votes], submission_train[HBA_votes]), 4),
)



## === cell 23
pass



## === cell 24
if False:
    smooth_width = SMOOTH_WIDTH  # Here smooth_width is just for looking,

    spectro_meta = train_meta[clust_rows_bool].copy()
    every_nth = 997

    freqs = np.array(range(100)) * 0.19525 + 0.59
    downsel_freqs = np.insert(
        freqs[int((smooth_width - 1) / 2) : 100 : smooth_width], 0, freqs[0]
    )

    spect_trend = 150.0 / (1.0**2.3 + freqs ** (2.3))

    freqbins = [7, 18, 60, 84]
    freqclrs = ["red", "blue", "green", "yellow"]
    n_fft_bins = 256
    apod_wind = np.blackman(n_fft_bins)



## === cell 25
if False:
    pass



## === cell 26
if False:
    pass



## === cell 27
pass



## === cell 28
train_rows_bool = (
    (train_meta.eeg_sub_id < 33 + 1) & (train_meta.eeg_sub_id % 5 == 3)
) | (  # id=3,8,13,18,23,28,33
    train_meta.eeg_sub_id == 0
) & (
    (train_meta.eeg_id % 23) % 8 > 1
)  # include odd and even eeg_ids
print("Number of Training rows:", sum(train_rows_bool))

valid_rows_bool = (
    (train_meta.eeg_sub_id < 44 + 1) & (train_meta.eeg_sub_id % 19 == 6)
) | (  # id=6,25,44
    train_meta.eeg_sub_id == 0
) & (
    (train_meta.eeg_id % 23) % 8 < 2
)  # include odd and even eeg_ids
print("Number of Validation rows:", sum(valid_rows_bool))



## === cell 29
pass



## === cell 30
if USE_PREPROC:
    Xy_train_meta = pd.read_csv(above_dir_preproc + "Xy_train_meta_v62.csv")
    Xy_train_feats = pd.read_csv(above_dir_preproc + "Xy_train_feats_v62.csv")
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



## === cell 31
Xy_train_feats



## === cell 32
pass



## === cell 33
if USE_PREPROC:
    Xy_valid_meta = pd.read_csv(above_dir_preproc + "Xy_valid_meta_v62.csv")
    Xy_valid_feats = pd.read_csv(above_dir_preproc + "Xy_valid_feats_v62.csv")
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



## === cell 34
print(Xy_valid_feats.shape)
Xy_valid_feats.iloc[-5:, 80:90]



## === cell 35
pass



## === cell 36
X = Xy_train_feats.drop(columns=["clust_id"])
y = Xy_train_feats.clust_id

Xlr = X.drop(columns=X.columns[-10:])

if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : 83 + 1]
    lrmodel1 = LogisticRegression(
        penalty="l2",
        C=LR1_C,
        solver="saga",
        max_iter=1500,
        multi_class="multinomial",
        n_jobs=-1,
        random_state=42,
    ).fit(Xlr1, y)

    print("\nLR1 model score for X,y = {:.1f}%\n".format(100 * lrmodel1.score(Xlr1, y)))

if USE_LR2:
    Xlr2 = Xlr.iloc[:, 84 : 179 + 1]
    lrmodel2 = LogisticRegression(
        penalty="l2",
        C=LR2_C,
        solver="saga",
        max_iter=1500,
        multi_class="multinomial",
        n_jobs=-1,
        random_state=42,
    ).fit(Xlr2, y)

    print("\nLR2 model score for X,y = {:.1f}%\n".format(100 * lrmodel2.score(Xlr2, y)))



## === cell 37
Xy_train_wLRfeats = Xy_train_feats.copy()
X_no_y = Xy_train_feats.drop(columns=["clust_id"])
Xlr = X_no_y.drop(columns=X_no_y.columns[-10:])

_extra_cols_train = []
if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : 83 + 1]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    lrprobas = lrprobas + LR_BLUR * (_rng.rand(len(lrprobas), NUM_CLUSTS) - 0.5)
    _extra_cols_train.append(
        pd.DataFrame(lrprobas, columns=[f"lrMid{i}" for i in range(NUM_CLUSTS)])
    )

if USE_LR2:
    Xlr2 = Xlr.iloc[:, 84 : 179 + 1]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    lrprobas = lrprobas + LR_BLUR * (_rng.rand(len(lrprobas), NUM_CLUSTS) - 0.5)
    _extra_cols_train.append(
        pd.DataFrame(lrprobas, columns=[f"lrFFT{i}" for i in range(NUM_CLUSTS)])
    )

if _extra_cols_train:
    Xy_train_wLRfeats = pd.concat([Xy_train_wLRfeats] + _extra_cols_train, axis=1)

Xy_valid_wLRfeats = Xy_valid_feats.copy()
X_no_y = Xy_valid_feats.drop(columns=["clust_id"])
Xlr = X_no_y.drop(columns=X_no_y.columns[-10:])

_extra_cols_valid = []
if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : 83 + 1]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    lrprobas = lrprobas + LR_BLUR * (_rng.rand(len(lrprobas), NUM_CLUSTS) - 0.5)
    _extra_cols_valid.append(
        pd.DataFrame(lrprobas, columns=[f"lrMid{i}" for i in range(NUM_CLUSTS)])
    )

if USE_LR2:
    Xlr2 = Xlr.iloc[:, 84 : 179 + 1]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    lrprobas = lrprobas + LR_BLUR * (_rng.rand(len(lrprobas), NUM_CLUSTS) - 0.5)
    _extra_cols_valid.append(
        pd.DataFrame(lrprobas, columns=[f"lrFFT{i}" for i in range(NUM_CLUSTS)])
    )

if _extra_cols_valid:
    Xy_valid_wLRfeats = pd.concat([Xy_valid_wLRfeats] + _extra_cols_valid, axis=1)



## === cell 38
pass



## === cell 39
X = Xy_train_wLRfeats.drop(columns=["clust_id"])
y = Xy_train_wLRfeats.clust_id

ave_oob = []
rfmodel = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=5,
    max_features=0.2,
    max_samples=0.9,
    oob_score=True,
    class_weight="balanced_subsample",
    n_jobs=-1,
    verbose=0,
    random_state=42,
).fit(X, y)
ave_oob.append(rfmodel.oob_score_)

print("\nRF model ave OOB score = {:.1f}%".format(100 * np.mean(ave_oob)))
print("\nRF model score for X,y = {:.1f}%\n".format(100 * rfmodel.score(X, y)))



## === cell 40
Xy_train_meta["pred_id"] = rfmodel.predict(X).astype(np.int64)

train_probas = rfmodel.predict_proba(X)
maxprobs_train = np.max(train_probas, axis=1)

solution = Xy_train_meta[["eeg_id"]].copy()
solution[HBA_votes] = Xy_train_meta[HBA_probs].to_numpy(dtype=np.float64)

submission = solution.copy()
pred_ids = Xy_train_meta["pred_id"].to_numpy(dtype=np.int64)

best_fracs, best_centers = find_best_tamed_kl()

for iprob in range(HBA_number):
    this_col_probs = best_centers[:, iprob]
    submission[HBA_votes[iprob]] = np.take(this_col_probs, pred_ids)

this_kl = kld_score(solution[HBA_votes], submission[HBA_votes])
print("Tamed fractions:\n", best_fracs, "\nand centers:\n", best_centers)
print("\nKL from tamed centers: {:.4f}".format(this_kl))



## === cell 41
pass



## === cell 42
X_valid = Xy_valid_wLRfeats.drop(columns=["clust_id"])
Xy_valid_meta["pred_id"] = rfmodel.predict(X_valid).astype(np.int64)

valid_probas = rfmodel.predict_proba(X_valid)
maxprobs_valid = np.max(valid_probas, axis=1)

solution = Xy_valid_meta[["eeg_id"]].copy()
solution[HBA_votes] = Xy_valid_meta[HBA_probs].to_numpy(dtype=np.float64)

submission = solution.copy()
pred_ids = Xy_valid_meta["pred_id"].to_numpy(dtype=np.int64)

best_fracs, best_centers = find_best_tamed_kl()

for iprob in range(HBA_number):
    this_col_probs = best_centers[:, iprob]
    submission[HBA_votes[iprob]] = np.take(this_col_probs, pred_ids)

this_kl = kld_score(solution[HBA_votes], submission[HBA_votes])
print("Tamed fractions:\n", best_fracs, "\nand centers:\n", best_centers)
print("\nKL from tamed centers: {:.4f}".format(this_kl))

best_centers_for_test = best_centers.copy()



## === cell 43
pass



## === cell 44
pass



## === cell 45
if False:
    X = Xy_train_wLRfeats.drop(columns=["clust_id"])
    y = Xy_train_wLRfeats.clust_id
    solution = Xy_valid_meta[["eeg_id"] + HBA_votes]
    for col_pre in HBA_names:
        solution.loc[:, col_pre + "_vote"] = Xy_valid_meta[col_pre + "_prob"]
    submission = solution.copy()

if False:
    pass



## === cell 46
pass



## === cell 47
pass



## === cell 48
test_preproc_path = os.path.join(above_dir_preproc, "Xy_test_feats_v62.csv")
if USE_PREPROC and os.path.exists(test_preproc_path):
    Xy_test_feats = pd.read_csv(test_preproc_path)
    print(
        "Loaded preprocessed test feats:",
        Xy_test_feats.shape,
        "from",
        test_preproc_path,
    )
else:
    Xy_test_feats = assemble_features(
        test_meta, traintest="test", smooth_width=SMOOTH_WIDTH
    )

train_feat_cols = Xy_train_feats.drop(columns=["clust_id"]).columns.tolist()
if list(Xy_test_feats.columns) != train_feat_cols:
    Xy_test_feats = Xy_test_feats.reindex(columns=train_feat_cols)
    Xy_test_feats = Xy_test_feats.fillna(0.0)

Xy_test_wLRfeats = Xy_test_feats.copy()
Xlr = Xy_test_feats.drop(columns=Xy_test_feats.columns[-10:])

_extra_cols_test = []
if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : 83 + 1]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    lrprobas = lrprobas + LR_BLUR * (_rng.rand(len(lrprobas), NUM_CLUSTS) - 0.5)
    _extra_cols_test.append(
        pd.DataFrame(lrprobas, columns=[f"lrMid{i}" for i in range(NUM_CLUSTS)])
    )

if USE_LR2:
    Xlr2 = Xlr.iloc[:, 84 : 179 + 1]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    lrprobas = lrprobas + LR_BLUR * (_rng.rand(len(lrprobas), NUM_CLUSTS) - 0.5)
    _extra_cols_test.append(
        pd.DataFrame(lrprobas, columns=[f"lrFFT{i}" for i in range(NUM_CLUSTS)])
    )

if _extra_cols_test:
    Xy_test_wLRfeats = pd.concat([Xy_test_wLRfeats] + _extra_cols_test, axis=1)

rf_train_cols = Xy_train_wLRfeats.drop(columns=["clust_id"]).columns.tolist()
X_test_rf = Xy_test_wLRfeats.reindex(columns=rf_train_cols).fillna(0.0)

pred_ids = rfmodel.predict(X_test_rf).astype(np.int64)

test_submit = test_meta[["eeg_id"]].copy()
for new_col in HBA_votes:
    test_submit[new_col] = 1 / HBA_number

for iprob in range(HBA_number):
    this_col_probs = best_centers_for_test[:, iprob]
    test_submit[HBA_votes[iprob]] = np.take(this_col_probs, pred_ids)

test_submit = test_submit.groupby("eeg_id", as_index=False)[HBA_votes].mean()

probs = test_submit[HBA_votes].to_numpy(dtype=float)
probs = np.clip(probs, 1e-15, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)
test_submit[HBA_votes] = probs
test_submit = test_submit[["eeg_id"] + HBA_votes]

sample_sub = pd.read_csv(above_dir + "sample_submission.csv")

final_sub = sample_sub[["eeg_id"]].merge(test_submit, on="eeg_id", how="left")
final_sub[HBA_votes] = final_sub[HBA_votes].fillna(1.0 / HBA_number)

p = final_sub[HBA_votes].to_numpy(dtype=float)
p = np.clip(p, 1e-15, 1.0)
p = p / p.sum(axis=1, keepdims=True)
final_sub[HBA_votes] = p
final_sub = final_sub[["eeg_id"] + HBA_votes]

assert list(final_sub.columns) == ["eeg_id"] + HBA_votes, "Submission columns mismatch."
assert len(final_sub) == len(sample_sub), "Row count mismatch vs sample_submission."
row_sums = final_sub[HBA_votes].sum(axis=1).to_numpy()
if not np.all(np.isfinite(row_sums)):
    raise ValueError("Non-finite row sums in submission probabilities.")
if np.max(np.abs(row_sums - 1.0)) > 1e-6:
    pp = final_sub[HBA_votes].to_numpy(dtype=np.float64)
    pp = np.clip(pp, 1e-15, 1.0)
    pp = pp / pp.sum(axis=1, keepdims=True)
    final_sub[HBA_votes] = pp

print(final_sub.head())
print("Final submission shape:", final_sub.shape)

final_sub.to_csv(
    "submission.csv", header=True, index=False, na_rep="", float_format="%.6f"
)
print("Wrote submission.csv")
