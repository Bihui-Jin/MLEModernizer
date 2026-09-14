# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.dataset as ds
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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

above_dir_candidates = [
    "/kaggle/input/hms-harmful-brain-activity-classification/",
    "../input/hms-harmful-brain-activity-classification/",
]
above_dir = None
for p in above_dir_candidates:
    if os.path.exists(p):
        above_dir = p
        break
if above_dir is None:
    above_dir = "../input/hms-harmful-brain-activity-classification/"

above_dir_preproc_candidates = [
    "/kaggle/input/hms-2024-brain-data/",
    "../input/hms-2024-brain-data/",
]
above_dir_preproc = None
for p in above_dir_preproc_candidates:
    if os.path.exists(p):
        above_dir_preproc = p
        break
if above_dir_preproc is None:
    above_dir_preproc = "../input/hms-2024-brain-data/"


def _ensure_dir_slash(p: str) -> str:
    if p is None:
        return ""
    return p if p.endswith("/") else (p + "/")


above_dir = _ensure_dir_slash(above_dir)
above_dir_preproc = _ensure_dir_slash(above_dir_preproc)

print("above_dir:", above_dir)
print("above_dir_preproc:", above_dir_preproc)




## === cell 1
def _try_read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None




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
def _safe_probs(arr, eps=1e-6):
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.clip(arr, eps, 1.0)
    s = arr.sum(axis=1, keepdims=True)
    s = np.where(s == 0.0, 1.0, s)
    return arr / s


def kld_score(solution, submission):
    """
    Calculate the average KL divergence score.

    Bugfix preserved:
    - Explicitly drop 'eeg_id' if present; clip to avoid log(0).
    """
    sol = solution.copy()
    sub = submission.copy()

    drop_cols = [c for c in ["eeg_id"] if c in sol.columns]
    if drop_cols:
        sol = sol.drop(columns=drop_cols)
    if drop_cols and all(c in sub.columns for c in drop_cols):
        sub = sub.drop(columns=drop_cols)

    eps = 1e-12
    solv = np.clip(sol.to_numpy(dtype=float), eps, 1.0)
    subv = np.clip(sub.to_numpy(dtype=float), eps, 1.0)

    kl = np.sum(solv * (np.log(solv) - np.log(subv)), axis=1)
    return float(np.mean(kl))




## === cell 4
def read_hms_meta():
    """
    Read train/test metadata and add derived columns (votes->probs, entropy).
    """
    if (not isinstance(above_dir, str)) or (above_dir.strip() == ""):
        raise ValueError("above_dir is not set to a valid dataset directory path.")

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
    probs = np.clip(train_meta[HBA_probs].to_numpy(dtype=float), 1.0e-8, 1.0)
    train_meta["entropy"] = np.nansum(probs * (-np.log(probs)), axis=1)

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
        kmclrs = hba_clrs

    clstclrs = np.take(np.array(kmclrs, dtype=object), np.asarray(clust_ids))

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
def assemble_features(
    meta_frame,
    traintest="train",
    smooth_width=5,
):
    """
    Create a dataframe of spectrogram features from the meta_frame rows.
    Will include clust_id (i.e, the y) if it is in the input meta_frame.

    Timeout fix (equivalence-preserving):
    - Fast parquet I/O: read only required columns with pq.read_table(columns=...),
      cache the ordered column list per file (avoids repeated schema scans).
    - Avoid pyarrow.dataset overhead and avoid repeated pandas object creation.
    - Vectorize per-file index computations; keep identical per-row feature math.
    """
    freqs = np.array(range(100)) * 0.19525 + 0.59
    spect_trend = 150.0 / (1.0**2.3 + freqs ** (2.3))
    baseinds = np.insert(
        np.arange(int((smooth_width - 1) / 2), 100, smooth_width), 0, 0
    )
    spect_trend4 = np.array(4 * list(spect_trend))

    n_fft_bins = 256
    freqbins = [7, 15, 23, 60]
    apod_wind = np.blackman(n_fft_bins).astype(np.float64, copy=False)
    fftfeatbins = [4, 9, 16, 25, 36, 49]

    select_inds = np.concatenate(
        (baseinds, 100 + baseinds, 200 + baseinds, 300 + baseinds)
    ).astype(np.int64, copy=False)

    half = (smooth_width - 1) // 2
    kernel_sw = np.ones(smooth_width, dtype=np.float64) / float(smooth_width)

    def centered_rolling_mean_full(arr_1d: np.ndarray) -> np.ndarray:
        out = np.full(arr_1d.shape, np.nan, dtype=np.float64)
        if arr_1d.shape[0] < smooth_width:
            return out
        valid = np.convolve(arr_1d, kernel_sw, mode="valid")
        out[half : half + valid.shape[0]] = valid
        return out

    kernel2 = np.array([0.5, 0.5], dtype=np.float64)

    def causal_roll2_twice(arr_1d: np.ndarray) -> np.ndarray:
        x = arr_1d
        y = np.empty_like(x, dtype=np.float64)
        y[0] = x[0]
        y[1:] = np.convolve(x, kernel2, mode="valid")
        z = np.empty_like(y, dtype=np.float64)
        z[0] = y[0]
        z[1:] = np.convolve(y, kernel2, mode="valid")
        return z

    kernel5 = np.ones(5, dtype=np.float64) / 5.0

    def centered_roll5_min1(arr_1d: np.ndarray) -> np.ndarray:
        n = arr_1d.shape[0]
        sums = np.convolve(arr_1d, kernel5, mode="same") * 5.0
        counts = np.convolve(np.ones(n, dtype=np.float64), kernel5, mode="same") * 5.0
        return sums / counts

    n_rows = len(meta_frame.index)
    n_mid = len(select_inds)
    n_flare = 4 * len(freqbins) * len(fftfeatbins)
    n_stat = 2 + 2 * 4
    n_total = n_mid + n_flare + n_stat
    has_clust = "clust_id" in meta_frame.columns
    n_total_out = n_total + (1 if has_clust else 0)

    mid_cols = [str(i) for i in range(n_mid)]
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
    statcols = (
        ["Mean", "Median"]
        + [c + "mean" for c in the4chains]
        + [c + "median" for c in the4chains]
    )
    out_cols = mid_cols + flarecols + statcols + (["clust_id"] if has_clust else [])

    feats_mat = np.empty((n_rows, n_total_out), dtype=np.float64)

    sum_spect_trend_cache = {
        freqbin: float(np.sum(spect_trend[freqbin - 4 : freqbin + 4 + 1 : 2]))
        for freqbin in freqbins
    }
    cols_offsets_by_freqbin = {
        freqbin: np.array([-4, -2, 0, 2, 4], dtype=np.int64) for freqbin in freqbins
    }

    def _ordered_spectro_columns_from_schema(parquet_path: str):
        schema_names = pq.read_schema(parquet_path).names
        ordered = ["time"]
        for ch in the4chains:
            expected = [f"{ch}_{f:.2f}" for f in freqs]
            if all(c in schema_names for c in expected):
                ordered.extend(expected)
            else:
                ch_cols = [c for c in schema_names if c.startswith(ch + "_")]
                if len(ch_cols) == 0:
                    raise ValueError(
                        f"Could not find any spectrogram columns for chain {ch} in {parquet_path}"
                    )

                def _freq_of(col):
                    try:
                        return float(col.split("_", 1)[1])
                    except Exception:
                        return np.inf

                ch_cols_sorted = sorted(ch_cols, key=_freq_of)
                if len(ch_cols_sorted) < 100:
                    raise ValueError(
                        f"Found only {len(ch_cols_sorted)} {ch}_* columns (need >=100) in {parquet_path}"
                    )
                ordered.extend(ch_cols_sorted[:100])
        return ordered

    meta_frame = meta_frame.reset_index(drop=True)
    offset_arr = meta_frame["spectrogram_label_offset_seconds"].to_numpy(
        dtype=np.float64, copy=False
    )
    clust_arr = (
        meta_frame["clust_id"].to_numpy(dtype=np.float64, copy=False)
        if has_clust
        else None
    )

    groups = meta_frame.groupby("spectrogram_id", sort=False).indices
    print_every_nth = max([200, 200 * int(0.5 + n_rows / (100.0 * 15))])

    spectro_cols_cache = {}

    def _parquet_spectro_to_numpy_needed(path: str) -> np.ndarray:
        cols = spectro_cols_cache.get(path)
        if cols is None:
            cols = _ordered_spectro_columns_from_schema(path)
            spectro_cols_cache[path] = cols
        table = pq.read_table(path, columns=cols)
        return table.to_pandas(self_destruct=True).to_numpy(
            dtype=np.float64, copy=False
        )

    done = 0
    for spectro_id, row_idxs in groups.items():
        spectro_id_int = int(spectro_id)
        if traintest != "test":
            spectro_file = (
                above_dir + "train_spectrograms/" + str(spectro_id_int) + ".parquet"
            )
        else:
            spectro_file = (
                above_dir + "test_spectrograms/" + str(spectro_id_int) + ".parquet"
            )

        this_spectro_np = _parquet_spectro_to_numpy_needed(spectro_file)

        row_idxs = np.asarray(row_idxs, dtype=np.int64)
        loc_offsets = (offset_arr[row_idxs] / 2.0).astype(np.int64, copy=False)
        fftlocbegs = (loc_offsets + 149 - (n_fft_bins / 2 - 1)).astype(
            np.int64, copy=False
        )
        fftlocends = (loc_offsets + 150 + (n_fft_bins / 2 - 1)).astype(
            np.int64, copy=False
        )

        for j, out_i in enumerate(row_idxs):
            loc_offset = int(loc_offsets[j])
            fftlocbeg = int(fftlocbegs[j])
            fftlocend = int(fftlocends[j])

            mid_raw = (
                this_spectro_np[loc_offset + 148, 1:]
                + this_spectro_np[loc_offset + 149, 1:]
                + this_spectro_np[loc_offset + 150, 1:]
                + this_spectro_np[loc_offset + 151, 1:]
            ) / (4.0 * spect_trend4)
            mid_raw = np.clip(mid_raw, 0.001, 1000.0)
            mid_raw = np.nan_to_num(mid_raw, nan=0.001, posinf=0.001, neginf=0.001)

            spect_mean = float(np.mean(mid_raw))
            spect_median = float(np.median(mid_raw))

            mid_4x100 = mid_raw.reshape(4, 100)
            the4means = mid_4x100.mean(axis=1)
            the4medians = np.median(mid_4x100, axis=1)

            mid_pre = np.log10(mid_raw / spect_mean)
            mid_sm = centered_rolling_mean_full(mid_pre)

            for ioff in (0, 100, 200, 300):
                mid_sm[ioff : ioff + half] = mid_pre[ioff : ioff + half]

            feats_mat[out_i, 0:n_mid] = mid_sm[select_inds]

            flare_off = n_mid
            fidx = 0
            for ispec in range(4):
                base_col = 1 + ispec * 100
                for freqbin in freqbins:
                    sum_spect_trend = sum_spect_trend_cache[freqbin]
                    ifreqoff = base_col + freqbin
                    cols = ifreqoff + cols_offsets_by_freqbin[freqbin]

                    ampl = (
                        np.sum(this_spectro_np[fftlocbeg : fftlocend + 1, cols], axis=1)
                        / sum_spect_trend
                    )
                    ampl = np.clip(ampl, 0.001, 1000.0)
                    ampl = np.nan_to_num(ampl, nan=0.001, posinf=0.001, neginf=0.001)
                    ampl = 2.0 * ampl / (ampl[127] + ampl[128])
                    ampl = apod_wind * np.clip(ampl, 0.0, 10.0)

                    ampl = causal_roll2_twice(ampl)
                    amplfft = np.abs(np.fft.fft(ampl))[0 : int(n_fft_bins / 2)]
                    amplfft = np.log10(1.0 + centered_roll5_min1(amplfft))

                    for fftfeatbin in fftfeatbins:
                        feats_mat[out_i, flare_off + fidx] = amplfft[fftfeatbin]
                        fidx += 1

            stat_off = n_mid + n_flare
            feats_mat[out_i, stat_off + 0] = np.log10(spect_mean)
            feats_mat[out_i, stat_off + 1] = np.log10(spect_median)
            feats_mat[out_i, stat_off + 2 : stat_off + 2 + 4] = np.log10(the4means)
            feats_mat[out_i, stat_off + 2 + 4 : stat_off + 2 + 8] = np.log10(
                the4medians
            )

            if has_clust:
                feats_mat[out_i, -1] = float(clust_arr[out_i])

            done += 1
            if done % print_every_nth == 0:
                print("... {} done...".format(done))

    feats_frame = pd.DataFrame(feats_mat, columns=out_cols)
    return feats_frame




## === cell 7
def find_best_tamed_kl():
    """
    Adjust the taming fraction for each cluster center to optimize KL
    Assumed inputs in environment:
        submission, pred_ids, solution
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
        for this_frac in np.arange(0.03, 1.00, 0.05):
            tamed_fracs[iclust] = this_frac
            this_cent = (
                tamed_fracs[iclust] * clust_centers[iclust, :]
                + (1.0 - tamed_fracs[iclust]) * mean_all_probs
            )
            tamed_centers[iclust, :] = this_cent

            for iprob in range(HBA_number):
                this_col_probs = tamed_centers[:, iprob]
                submission[HBA_votes[iprob]] = this_col_probs[pred_ids]
            submission.loc[:, HBA_votes] = _safe_probs(submission[HBA_votes].to_numpy())
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
pass



## === cell 9
train_meta, test_meta = read_hms_meta()



## === cell 10
if False:
    plt.figure(figsize=(5, 2))
    plt.hist(train_meta["spectrogram_sub_id"], bins=55, log=True)
    plt.title("Histogram of spectrogram_sub_id")
    plt.show()

    plt.figure(figsize=(5, 2))
    plt.hist(train_meta["eeg_sub_id"], bins=55, log=True)
    plt.title("Histogram of eeg_sub_id")
    plt.show()



## === cell 11
pass



## === cell 12
num_clusts = NUM_CLUSTS  # 6 to 10
clust_rows_bool = train_meta.eeg_sub_id < 200



## === cell 13
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
    random_state=RANDOM_STATE,
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



## === cell 14
train_meta["clust_id"] = kmeans.predict(np.array(train_meta[HBA_probs]))

all_probs = train_meta[HBA_probs]
all_ids = train_meta["clust_id"]

if False:
    kmclrs = prob_prob_scatter("Seizure", "GPD", all_probs, all_ids, iclust_of_order)
    kmclrs = prob_prob_scatter("LPD", "GRDA", all_probs, all_ids, iclust_of_order)
    kmclrs = prob_prob_scatter("LRDA", "Other", all_probs, all_ids, iclust_of_order)

if "kmclrs" not in globals():
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

clust_counts = train_meta.clust_id.value_counts()

iorder_of_clust = num_clusts * [-1]
for iord, iclust in enumerate(iclust_of_order):
    iorder_of_clust[iclust] = iord



## === cell 15
pass



## === cell 16
pass



## === cell 17
solution_train = train_meta[["eeg_id"] + HBA_votes].copy()
solution_train.loc[:, HBA_votes] = train_meta[HBA_probs].to_numpy(
    dtype=float, copy=False
)

submission_train = solution_train.copy()
clust_ids = train_meta["clust_id"].to_numpy()
pred_probs = clust_centers[clust_ids]
submission_train.loc[:, HBA_votes] = _safe_probs(pred_probs)

print(
    "Score if HBA samples are correctly assigned cluster prob.s:",
    np.round(kld_score(solution_train, submission_train), 4),
)



## === cell 18
pass



## === cell 19
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



## === cell 20
if False:
    pass



## === cell 21
if False:
    pass



## === cell 22
pass



## === cell 23
train_rows_bool = (
    (train_meta.eeg_sub_id < 50 + 1) & (train_meta.eeg_sub_id % 4 == 2)
) | (train_meta.eeg_sub_id == 0) & ((train_meta.eeg_id % 23) % 8 > 1)
print("Number of Training rows:", sum(train_rows_bool))

valid_rows_bool = (
    (train_meta.eeg_sub_id < 51 + 1) & (train_meta.eeg_sub_id % 8 == 3)
) | (train_meta.eeg_sub_id == 0) & ((train_meta.eeg_id % 23) % 8 < 2)
print("Number of Validation rows:", sum(valid_rows_bool))



## === cell 24
pass



## === cell 25
if USE_PREPROC and (not os.path.exists(above_dir_preproc)):
    print(
        "Warning: above_dir_preproc not found; falling back to above_dir for any preproc CSV lookups."
    )
    above_dir_preproc = above_dir



## === cell 26
Xy_train_meta = None
Xy_train_feats = None

if USE_PREPROC:
    meta_path = above_dir_preproc + "Xy_train_meta_v66.csv"
    feats_path = above_dir_preproc + "Xy_train_feats_v66.csv"
    Xy_train_meta = _try_read_csv(meta_path)
    Xy_train_feats = _try_read_csv(feats_path)

if (Xy_train_meta is None) or (Xy_train_feats is None):
    print(
        "Preprocessed train files not found; assembling train features from raw data..."
    )
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
else:
    Xy_train_meta["clust_id"] = kmeans.predict(np.array(Xy_train_meta[HBA_probs]))
    Xy_train_feats["clust_id"] = Xy_train_meta["clust_id"]
    SMOOTH_WIDTH = 5



## === cell 27
Xy_train_feats



## === cell 28
pass



## === cell 29
Xy_valid_meta = None
Xy_valid_feats = None

if USE_PREPROC:
    meta_path = above_dir_preproc + "Xy_valid_meta_v66.csv"
    feats_path = above_dir_preproc + "Xy_valid_feats_v66.csv"
    Xy_valid_meta = _try_read_csv(meta_path)
    Xy_valid_feats = _try_read_csv(feats_path)

if (Xy_valid_meta is None) or (Xy_valid_feats is None):
    print(
        "Preprocessed valid files not found; assembling validation features from raw data..."
    )
    Xy_valid_meta = (train_meta[valid_rows_bool])[::VALID_DOWNSEL].copy()
    Xy_valid_meta = Xy_valid_meta.reset_index().drop(columns=["index"])
    print("Number of samples used for Validation =", len(Xy_valid_meta))

    Xy_valid_feats = assemble_features(
        Xy_valid_meta, traintest="train", smooth_width=SMOOTH_WIDTH
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



## === cell 30
print(Xy_valid_feats.shape)
Xy_valid_feats.iloc[-5:, 80:90]



## === cell 31
Xy_train_feats



## === cell 32
pass



## === cell 33
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
        multi_class="multinomial",
        n_jobs=-1,
    ).fit(Xlr1, y)

if USE_LR2:
    Xlr2 = Xlr.iloc[:, 84 : 179 + 1]
    lrmodel2 = LogisticRegression(
        penalty=LR_REGU,
        C=LR2_C + 1.0,
        solver="saga",
        max_iter=1500,
        multi_class="multinomial",
        n_jobs=-1,
    ).fit(Xlr2, y)



## === cell 34
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
if USE_LR2:
    Xlr2 = Xlr.iloc[:, 84 : 179 + 1]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    for iadd in range(NUM_CLUSTS):
        Xy_train_wLRfeats["lrFFT" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            np.random.rand(len(lrprobas)) - 0.5
        )

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
if USE_LR2:
    Xlr2 = Xlr.iloc[:, 84 : 179 + 1]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    for iadd in range(NUM_CLUSTS):
        Xy_valid_wLRfeats["lrFFT" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            np.random.rand(len(lrprobas)) - 0.5
        )



## === cell 35
pass



## === cell 36
X = Xy_train_wLRfeats.drop(columns=["clust_id"])
y = Xy_train_wLRfeats.clust_id

ave_oob = []
rfmodel = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=7,
    max_features=0.3,
    max_samples=0.8,
    oob_score=True,
    class_weight="balanced_subsample",
    n_jobs=-1,
    verbose=0,
    random_state=RANDOM_STATE,
).fit(X, y)
ave_oob.append(rfmodel.oob_score_)

print("\nRF model ave OOB score = {:.1f}%".format(100 * np.mean(ave_oob)))
print("\nRF model score for X,y = {:.1f}%\n".format(100 * rfmodel.score(X, y)))



## === cell 37
Xy_train_meta["pred_id"] = rfmodel.predict(Xy_train_wLRfeats.drop(columns=["clust_id"]))

solution = Xy_train_meta[["eeg_id"] + HBA_votes].copy()
solution.loc[:, HBA_votes] = Xy_train_meta[HBA_probs].to_numpy(dtype=float, copy=False)
submission = solution.copy()

if USE_TAMED:
    pred_ids = Xy_train_meta["pred_id"].to_numpy()
    best_fracs, best_centers = find_best_tamed_kl()
    submission.loc[:, HBA_votes] = best_centers[pred_ids]
else:
    probas = rfmodel.predict_proba(Xy_train_wLRfeats.drop(columns=["clust_id"]))
    pred_probs = probas @ clust_centers
    submission.loc[:, HBA_votes] = _safe_probs(pred_probs)

this_kl = kld_score(solution, submission)
print("\nKL from predicted centers: {:.4f}".format(this_kl))



## === cell 38
pd.crosstab(Xy_train_meta["pred_id"], Xy_train_meta["clust_id"])



## === cell 39
pass



## === cell 40
Xy_valid_meta["pred_id"] = rfmodel.predict(Xy_valid_wLRfeats.drop(columns=["clust_id"]))

solution = Xy_valid_meta[["eeg_id"] + HBA_votes].copy()
solution.loc[:, HBA_votes] = Xy_valid_meta[HBA_probs].to_numpy(dtype=float, copy=False)
submission = solution.copy()

if USE_TAMED:
    pred_ids = Xy_valid_meta["pred_id"].to_numpy()
    best_fracs, best_centers = find_best_tamed_kl()
    submission.loc[:, HBA_votes] = best_centers[pred_ids]
else:
    probas = rfmodel.predict_proba(Xy_valid_wLRfeats.drop(columns=["clust_id"]))
    pred_probs = probas @ clust_centers
    submission.loc[:, HBA_votes] = _safe_probs(pred_probs)

this_kl = kld_score(solution, submission)
print("\nKL from predicted centers: {:.4f}".format(this_kl))



## === cell 41
pd.crosstab(Xy_valid_meta["pred_id"], Xy_valid_meta["clust_id"])



## === cell 42
pass



## === cell 43
pass



## === cell 44
pass



## === cell 45
pass



## === cell 46
pass



## === cell 47
pass



## === cell 48
pass



## === cell 49
if False:
    pass



## === cell 50
pass



## === cell 51
pass



## === cell 52
Xy_test_feats = assemble_features(
    test_meta, traintest="test", smooth_width=SMOOTH_WIDTH
)

Xy_test_wLRfeats = Xy_test_feats.copy()
Xlr = Xy_test_feats.drop(columns=Xy_test_feats.columns[-10:])
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

sample_sub_path = above_dir + "sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)
sample_sub["eeg_id"] = sample_sub["eeg_id"].astype(np.int64, copy=False)

X_test_rf = (
    Xy_test_wLRfeats
    if "clust_id" not in Xy_test_wLRfeats.columns
    else Xy_test_wLRfeats.drop(columns=["clust_id"])
)

if USE_TAMED:
    pred_ids = rfmodel.predict(X_test_rf)
    pred_mat = best_centers[pred_ids]
else:
    probas = rfmodel.predict_proba(X_test_rf)
    pred_mat = _safe_probs(probas @ clust_centers)

pred_df = pd.DataFrame(pred_mat, columns=HBA_votes)
pred_df.insert(0, "eeg_id", test_meta["eeg_id"].to_numpy(dtype=np.int64, copy=False))

test_submit = sample_sub[["eeg_id"]].merge(pred_df, on="eeg_id", how="left")

if test_submit[HBA_votes].isna().any().any():
    test_submit.loc[:, HBA_votes] = test_submit[HBA_votes].to_numpy(
        dtype=np.float64, copy=False
    )
    fill = np.full((len(test_submit), HBA_number), 1.0 / HBA_number, dtype=np.float64)
    mat = test_submit[HBA_votes].to_numpy(dtype=np.float64, copy=False)
    mask = np.isnan(mat)
    if mask.any():
        mat[mask] = fill[mask]
    mat = np.nan_to_num(
        mat, nan=1.0 / HBA_number, posinf=1.0 / HBA_number, neginf=1.0 / HBA_number
    )
    mat = _safe_probs(mat)
    test_submit.loc[:, HBA_votes] = mat

test_submit = test_submit[["eeg_id"] + HBA_votes]
assert len(test_submit) == len(sample_sub), (len(test_submit), len(sample_sub))

print(test_submit.head())
print(
    "Row sums (min/mean/max):",
    test_submit[HBA_votes].sum(axis=1).min(),
    test_submit[HBA_votes].sum(axis=1).mean(),
    test_submit[HBA_votes].sum(axis=1).max(),
)

test_submit.to_csv(
    "submission.csv", header=True, index=False, na_rep="", float_format="%.6f"
)
print("Wrote submission.csv with shape:", test_submit.shape)
print("Saved to:", os.path.abspath("submission.csv"))

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers must have the same length
