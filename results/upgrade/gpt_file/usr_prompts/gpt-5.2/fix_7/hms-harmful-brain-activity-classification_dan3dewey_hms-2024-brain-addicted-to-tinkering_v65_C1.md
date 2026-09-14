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

1.013965030697664

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import pyarrow
import pyarrow.parquet as pq
import pyarrow.dataset as pads

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

np.set_printoptions(precision=6, suppress=True)

SEED = 42
np.random.seed(SEED)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")




## === cell 1
NUM_CLUSTS = 9  # 6 to 10

USE_PREPROC = True  # Read-in saved meta and features frames
TRAIN_DOWNSEL = 1  # large values for code test; set to 1 to output all.
VALID_DOWNSEL = 1  #  "
SMOOTH_WIDTH = 5  # Odd>1: 3,5,7,9,...

LR_REGU = "l1"
USE_LR1 = True
LR1_C = 0.05  # 0.15     # smaller --> fewer non-zero coeff.s
USE_LR2 = True
LR2_C = 0.03  # 0.10
LR_BLUR = 0.10

above_dir = "../input/hms-harmful-brain-activity-classification/"
above_dir_preproc = "../input/hms-2024-brain-data/"




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




## === cell 3
def kld_score(solution, submission):
    """
    Average KL divergence score, assuming both frames share same prob columns.
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
    Read train.csv and test.csv, add *_prob columns and entropy to train_meta.
    Add extra cols to test to allow same downstream processing as train.
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

    probs = np.clip(train_meta[HBA_probs].to_numpy(dtype=np.float64), 1.0e-8, 1.0)
    train_meta["entropy"] = np.nansum(probs * (-np.log(probs)), axis=1)

    return train_meta, test_meta




## === cell 5
def prob_prob_scatter(name1, name2, probs2plot, clust_ids, iclust_order=[0]):
    """
    Scatter plot helper (debug/EDA).
    External: HBA_probs, iHBA_of_expert, clust_centers
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
    kmclrs = hba_clrs.copy()
    if len(iclust_order) > 2:
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
_DS_CACHE = {}  # path -> (dataset, column_names_without_time_index)

_FEAT_CACHE = {}


def _get_ds_and_cols(parquet_path: str):
    v = _DS_CACHE.get(parquet_path)
    if v is not None:
        return v
    ds = pads.dataset(parquet_path, format="parquet")
    cols = ds.schema.names[1:]  # drop time index col
    _DS_CACHE[parquet_path] = (ds, cols)
    return ds, cols


def _rolling_center_mean_1d_fullwindows(x: np.ndarray, width: int) -> np.ndarray:
    half = (width - 1) // 2
    kernel = np.ones(width, dtype=np.float64) / width
    y_valid = np.convolve(x, kernel, mode="valid")  # length n-width+1
    y = np.full(x.shape[0], np.nan, dtype=np.float64)
    y[half : len(x) - half] = y_valid
    return y


def _read_spectro_window_as_numpy(
    parquet_path: str, row_start: int, row_end_inclusive: int
) -> np.ndarray:
    ds, cols = _get_ds_and_cols(parquet_path)
    nrows = row_end_inclusive - row_start + 1

    table = ds.to_table(columns=cols, use_threads=True, limit=row_end_inclusive + 1)
    table = table.slice(row_start, nrows)

    table = table.combine_chunks()
    arr = table.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    if arr.shape != (nrows, len(cols)):
        arr = arr.reshape(nrows, len(cols))
    return arr




## === cell 7
def assemble_features(meta_frame, traintest="train", smooth_width=5):
    """
    Create a dataframe of spectrogram features from the meta_frame rows.
    Includes clust_id if present in input meta_frame.
    Assumes these are available: above_dir, the4chains
    """
    do_plot = False  # preserve computations; only skips plotting side-effects.

    freqs = np.array(range(100)) * 0.19525 + 0.59
    spect_trend = 150.0 / (1.0**2.3 + freqs ** (2.3))
    baseinds = np.insert(
        np.arange(int((smooth_width - 1) / 2), 100, smooth_width), 0, 0
    )
    spect_trend4 = np.repeat(spect_trend, 4)

    n_fft_bins = 256
    freqbins = [7, 18, 60, 84]
    apod_wind = np.blackman(n_fft_bins).astype(np.float64, copy=False)
    fftfeatbins = [4, 9, 16, 25, 36, 49]

    select_inds = np.concatenate(
        (baseinds, 100 + baseinds, 200 + baseinds, 300 + baseinds)
    ).astype(np.int64, copy=False)

    n_mid = len(select_inds)

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
    n_flare = len(flarecols)

    extra_cols = (
        ["Mean", "Median"]
        + [c + "mean" for c in the4chains]
        + [c + "median" for c in the4chains]
    )
    has_clust = "clust_id" in meta_frame.columns
    all_cols = (
        [f"m{i}" for i in range(n_mid)]
        + flarecols
        + extra_cols
        + (["clust_id"] if has_clust else [])
    )
    n_rows = len(meta_frame.index)
    out = np.empty((n_rows, len(all_cols)), dtype=np.float64)

    print_every_nth = max([100, 100 * int(0.5 + n_rows / (100.0 * 15))])

    sum_trend_by_freqbin = {
        fb: float(np.sum(spect_trend[fb - 4 : fb + 5 : 2])) for fb in freqbins
    }

    half = int((smooth_width - 1) / 2)

    the4means = np.empty(4, dtype=np.float64)
    the4medians = np.empty(4, dtype=np.float64)

    for i_out, r in enumerate(meta_frame.itertuples(index=False)):
        spectro_id_str = str(int(r.spectrogram_id))
        if traintest != "test":
            parquet_path = (
                above_dir + "train_spectrograms/" + spectro_id_str + ".parquet"
            )
        else:
            parquet_path = (
                above_dir + "test_spectrograms/" + spectro_id_str + ".parquet"
            )

        loc_offset = int(r.spectrogram_label_offset_seconds / 2)
        mid_rows_start = loc_offset + 148
        fftlocbeg = int(loc_offset + 149 - (n_fft_bins / 2 - 1))
        fftlocend = int(loc_offset + 150 + (n_fft_bins / 2 - 1))

        cache_key = (parquet_path, fftlocbeg, fftlocend, smooth_width)
        cached = _FEAT_CACHE.get(cache_key)
        if cached is not None:
            row = out[i_out, :]
            row[: len(cached)] = cached
            if has_clust:
                row[-1] = float(getattr(r, "clust_id"))
            if (i_out + 1) % print_every_nth == 0:
                print("... {} done...".format(i_out + 1))
            continue

        Xwin = _read_spectro_window_as_numpy(
            parquet_path, fftlocbeg, fftlocend
        )  # (256, 400)
        mid_rel0 = mid_rows_start - fftlocbeg
        mid_block = Xwin[mid_rel0 : mid_rel0 + 4, :]  # (4, 400)

        middle8s = (mid_block[0] + mid_block[1] + mid_block[2] + mid_block[3]) / (
            4.0 * spect_trend4
        )
        middle8s = np.clip(middle8s, 0.001, 1000.0)
        bad = ~np.isfinite(middle8s)
        if bad.any():
            middle8s[bad] = 0.001

        spect_mean = float(np.mean(middle8s))
        spect_median = float(np.median(middle8s))

        for ispec in range(4):
            ibeg = ispec * 100
            iend = ibeg + 100
            seg = middle8s[ibeg:iend]
            the4means[ispec] = np.mean(seg)
            the4medians[ispec] = np.median(seg)

        middle8spre = np.log10(middle8s / spect_mean)

        middle8s_sm = _rolling_center_mean_1d_fullwindows(middle8spre, smooth_width)
        for ioff in (0, 100, 200, 300):
            middle8s_sm[ioff : ioff + half] = middle8spre[ioff : ioff + half]

        middle_feats = middle8s_sm[select_inds]

        flarevals = np.empty(n_flare, dtype=np.float64)
        k = 0
        for ispec in range(4):
            col_base = ispec * 100
            for freqbin in freqbins:
                sum_spect_trend = sum_trend_by_freqbin[freqbin]
                ifreqoff = freqbin + 1 + col_base

                amplvstime = (
                    Xwin[:, ifreqoff - 4]
                    + Xwin[:, ifreqoff - 2]
                    + Xwin[:, ifreqoff]
                    + Xwin[:, ifreqoff + 2]
                    + Xwin[:, ifreqoff + 4]
                ) / sum_spect_trend
                amplvstime = np.clip(amplvstime, 0.001, 1000.0)
                bad2 = ~np.isfinite(amplvstime)
                if bad2.any():
                    amplvstime[bad2] = 0.001

                denom = amplvstime[127] + amplvstime[128]
                amplvstime = 2.0 * amplvstime / denom
                amplvstime = apod_wind * np.clip(amplvstime, 0.0, 10.0)

                for _ in range(2):
                    amplvstime[1:] = 0.5 * (amplvstime[:-1] + amplvstime[1:])

                amplfft = np.log10(
                    1.0 + np.abs(np.fft.fft(amplvstime))[: n_fft_bins // 2]
                )

                for fftfeatbin in fftfeatbins:
                    flarevals[k] = amplfft[fftfeatbin]
                    k += 1

        row = out[i_out, :]
        row[:n_mid] = middle_feats
        row[n_mid : n_mid + n_flare] = flarevals

        idx = n_mid + n_flare
        row[idx] = np.log10(spect_mean)
        idx += 1
        row[idx] = np.log10(spect_median)
        idx += 1
        row[idx : idx + 4] = np.log10(the4means)
        idx += 4
        row[idx : idx + 4] = np.log10(the4medians)
        idx += 4

        if has_clust:
            row[idx] = float(getattr(r, "clust_id"))

        _FEAT_CACHE[cache_key] = row[: (len(all_cols) - (1 if has_clust else 0))].copy()

        if (i_out + 1) % print_every_nth == 0:
            print("... {} done...".format(i_out + 1))

    feats_frame = pd.DataFrame(out, columns=all_cols)

    rename_map = {f"m{i}": int(select_inds[i]) for i in range(n_mid)}
    feats_frame = feats_frame.rename(columns=rename_map)

    return feats_frame.reset_index().drop(columns=["index"])




## === cell 8
def _kld_numpy(solution_probs: np.ndarray, submission_probs: np.ndarray) -> float:
    return float(
        np.nansum(-solution_probs * np.log(submission_probs / solution_probs))
        / solution_probs.shape[0]
    )


def find_best_tamed_kl():
    """
    Adjust taming fraction per cluster center to optimize KL.
    Assumed in env: pred_ids, solution, clust_centers, NUM_CLUSTS, HBA_number, HBA_votes
    """
    mean_all_probs = np.array(
        [0.208319, 0.132120, 0.128532, 0.138913, 0.179294, 0.212822],
        dtype=np.float64,
    )
    tamed_fracs = 0.0 * np.ones(NUM_CLUSTS, dtype=np.float64)
    tamed_centers = clust_centers.copy()
    for iclust in range(NUM_CLUSTS):
        tamed_centers[iclust, :] = mean_all_probs

    sol_np = solution[HBA_votes].to_numpy(dtype=np.float64)

    for iclust in range(NUM_CLUSTS):
        last_kl = 10.0
        best_fracs = tamed_fracs.copy()
        best_centers = tamed_centers.copy()
        for this_frac in np.arange(0.03, 1.00, 0.05):
            tamed_fracs[iclust] = this_frac
            this_cent = (
                tamed_fracs[iclust] * clust_centers[iclust, :]
                + (1.0 - tamed_fracs[iclust]) * mean_all_probs
            )
            tamed_centers[iclust, :] = this_cent

            sub_np = tamed_centers[pred_ids, :]
            this_kl = _kld_numpy(sol_np, sub_np)
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
num_clusts = NUM_CLUSTS
clust_rows_bool = train_meta.eeg_sub_id < 200

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
    n_clusters=num_clusts, init="k-means++", n_init=10, max_iter=300, random_state=SEED
)
kmeans.fit(prob_array)

clust_centers = kmeans.cluster_centers_
for iclust in range(NUM_CLUSTS):
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

train_meta["clust_id"] = kmeans.predict(np.array(train_meta[HBA_probs]))




## === cell 11
solution_train = train_meta[["eeg_id"] + HBA_votes].copy()
for vcol, pcol in zip(HBA_votes, HBA_probs):
    solution_train[vcol] = train_meta[pcol].astype(float)

submission_train = solution_train.copy()
clust_ids = train_meta["clust_id"].values
for iprob in range(HBA_number):
    this_col_probs = clust_centers[:, iprob]
    submission_train[HBA_votes[iprob]] = this_col_probs[clust_ids]

print(
    "Score if HBA samples are assigned cluster prob.s:",
    np.round(kld_score(solution_train[HBA_votes], submission_train[HBA_votes]), 4),
)




## === cell 12
train_rows_bool = (
    (train_meta.eeg_sub_id < 33 + 1) & (train_meta.eeg_sub_id % 5 == 3)
) | ((train_meta.eeg_sub_id == 0) & (((train_meta.eeg_id % 23) % 8) > 1))
print("Number of Training rows:", int(train_rows_bool.sum()))

valid_rows_bool = (
    (train_meta.eeg_sub_id < 44 + 1) & (train_meta.eeg_sub_id % 19 == 6)
) | ((train_meta.eeg_sub_id == 0) & (((train_meta.eeg_id % 23) % 8) < 2))
print("Number of Validation rows:", int(valid_rows_bool.sum()))




## === cell 13
def _maybe_read_preproc(prefix_path):
    files = {
        "Xy_train_meta": os.path.join(prefix_path, "Xy_train_meta_v62.csv"),
        "Xy_train_feats": os.path.join(prefix_path, "Xy_train_feats_v62.csv"),
        "Xy_valid_meta": os.path.join(prefix_path, "Xy_valid_meta_v62.csv"),
        "Xy_valid_feats": os.path.join(prefix_path, "Xy_valid_feats_v62.csv"),
    }
    return files, all(os.path.exists(p) for p in files.values())


preproc_files, have_preproc = _maybe_read_preproc(above_dir_preproc)
if USE_PREPROC and not have_preproc:
    print("Preprocessed v62 CSVs not found at:", above_dir_preproc)
    print(
        "Falling back to feature assembly from parquet spectrograms (this is slower but runs end-to-end)."
    )
    USE_PREPROC = False

if USE_PREPROC:
    Xy_train_meta = pd.read_csv(preproc_files["Xy_train_meta"])
    Xy_train_feats = pd.read_csv(preproc_files["Xy_train_feats"])
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




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3599144126.py in <cell line: 0>()
     27     Xy_train_meta = Xy_train_meta.reset_index().drop(columns=["index"])
     28     print("Number of samples used for training =", len(Xy_train_meta))
---> 29     Xy_train_feats = assemble_features(
     30         Xy_train_meta, traintest="train", smooth_width=SMOOTH_WIDTH
     31     )

/tmp/ipykernel_11/2506223953.py in assemble_features(meta_frame, traintest, smooth_width)
     92             continue
     93 
---> 94         Xwin = _read_spectro_window_as_numpy(
     95             parquet_path, fftlocbeg, fftlocend
     96         )  # (256, 400)

/tmp/ipykernel_11/3987119564.py in _read_spectro_window_as_numpy(parquet_path, row_start, row_end_inclusive)
     37     # Read only needed rows: take first (row_end+1) rows, then tail nrows.
     38     # This is equivalent to slicing [row_start:row_end+1] but avoids reading the entire file.
---> 39     table = ds.to_table(columns=cols, use_threads=True, limit=row_end_inclusive + 1)
     40     table = table.slice(row_start, nrows)
     41 

/usr/local/lib/python3.11/dist-packages/pyarrow/_dataset.pyx in pyarrow._dataset.Dataset.to_table()

TypeError: to_table() got an unexpected keyword argument 'limit'

## === cell 14
if USE_PREPROC:
    Xy_valid_meta = pd.read_csv(preproc_files["Xy_valid_meta"])
    Xy_valid_feats = pd.read_csv(preproc_files["Xy_valid_feats"])
    Xy_valid_meta["clust_id"] = kmeans.predict(np.array(Xy_valid_meta[HBA_probs]))
    Xy_valid_feats["clust_id"] = Xy_valid_meta["clust_id"]
else:
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

print(
    "Train feats shape:",
    Xy_train_feats.shape,
    "Valid feats shape:",
    Xy_valid_feats.shape,
)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2534500217.py in <cell line: 0>()
      8     Xy_valid_meta = Xy_valid_meta.reset_index().drop(columns=["index"])
      9     print("Number of samples used for Validation =", len(Xy_valid_meta))
---> 10     Xy_valid_feats = assemble_features(
     11         Xy_valid_meta, traintest="train", smooth_width=SMOOTH_WIDTH
     12     )

/tmp/ipykernel_11/2506223953.py in assemble_features(meta_frame, traintest, smooth_width)
     92             continue
     93 
---> 94         Xwin = _read_spectro_window_as_numpy(
     95             parquet_path, fftlocbeg, fftlocend
     96         )  # (256, 400)

/tmp/ipykernel_11/3987119564.py in _read_spectro_window_as_numpy(parquet_path, row_start, row_end_inclusive)
     37     # Read only needed rows: take first (row_end+1) rows, then tail nrows.
     38     # This is equivalent to slicing [row_start:row_end+1] but avoids reading the entire file.
---> 39     table = ds.to_table(columns=cols, use_threads=True, limit=row_end_inclusive + 1)
     40     table = table.slice(row_start, nrows)
     41 

/usr/local/lib/python3.11/dist-packages/pyarrow/_dataset.pyx in pyarrow._dataset.Dataset.to_table()

TypeError: to_table() got an unexpected keyword argument 'limit'

## === cell 15
X = Xy_train_feats.drop(columns=["clust_id"])
y = Xy_train_feats.clust_id
Xlr = X.drop(columns=X.columns[-10:])

if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : 83 + 1]
    lrmodel1 = LogisticRegression(
        penalty=LR_REGU,
        C=LR1_C,
        solver="saga",
        max_iter=1500,
        multi_class="multinomial",
        n_jobs=-1,
        random_state=SEED,
    ).fit(Xlr1, y)

if USE_LR2:
    Xlr2 = Xlr.iloc[:, 84 : 179 + 1]
    lrmodel2 = LogisticRegression(
        penalty=LR_REGU,
        C=LR2_C,
        solver="saga",
        max_iter=1500,
        multi_class="multinomial",
        n_jobs=-1,
        random_state=SEED,
    ).fit(Xlr2, y)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3424914972.py in <cell line: 0>()
----> 1 X = Xy_train_feats.drop(columns=["clust_id"])
      2 y = Xy_train_feats.clust_id
      3 Xlr = X.drop(columns=X.columns[-10:])
      4 
      5 if USE_LR1:

NameError: name 'Xy_train_feats' is not defined

## === cell 16
Xy_train_wLRfeats = Xy_train_feats.copy()
X = Xy_train_feats.drop(columns=["clust_id"])
Xlr = X.drop(columns=X.columns[-10:])

if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : 83 + 1]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    jitter = LR_BLUR * (np.random.rand(len(lrprobas), 1) - 0.5)
    lr_mid = np.clip(lrprobas + jitter, 0.0, 1.0)
    lr_mid_df = pd.DataFrame(lr_mid, columns=[f"lrMid{i}" for i in range(NUM_CLUSTS)])
    Xy_train_wLRfeats = pd.concat([Xy_train_wLRfeats, lr_mid_df], axis=1)

if USE_LR2:
    Xlr2 = Xlr.iloc[:, 84 : 179 + 1]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    jitter = LR_BLUR * (np.random.rand(len(lrprobas), 1) - 0.5)
    lr_fft = np.clip(lrprobas + jitter, 0.0, 1.0)
    lr_fft_df = pd.DataFrame(lr_fft, columns=[f"lrFFT{i}" for i in range(NUM_CLUSTS)])
    Xy_train_wLRfeats = pd.concat([Xy_train_wLRfeats, lr_fft_df], axis=1)

Xy_valid_wLRfeats = Xy_valid_feats.copy()
X = Xy_valid_feats.drop(columns=["clust_id"])
Xlr = X.drop(columns=X.columns[-10:])

if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : 83 + 1]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    jitter = LR_BLUR * (np.random.rand(len(lrprobas), 1) - 0.5)
    lr_mid = np.clip(lrprobas + jitter, 0.0, 1.0)
    lr_mid_df = pd.DataFrame(lr_mid, columns=[f"lrMid{i}" for i in range(NUM_CLUSTS)])
    Xy_valid_wLRfeats = pd.concat([Xy_valid_wLRfeats, lr_mid_df], axis=1)

if USE_LR2:
    Xlr2 = Xlr.iloc[:, 84 : 179 + 1]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    jitter = LR_BLUR * (np.random.rand(len(lrprobas), 1) - 0.5)
    lr_fft = np.clip(lrprobas + jitter, 0.0, 1.0)
    lr_fft_df = pd.DataFrame(lr_fft, columns=[f"lrFFT{i}" for i in range(NUM_CLUSTS)])
    Xy_valid_wLRfeats = pd.concat([Xy_valid_wLRfeats, lr_fft_df], axis=1)




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2079747383.py in <cell line: 0>()
----> 1 Xy_train_wLRfeats = Xy_train_feats.copy()
      2 X = Xy_train_feats.drop(columns=["clust_id"])
      3 Xlr = X.drop(columns=X.columns[-10:])
      4 
      5 if USE_LR1:

NameError: name 'Xy_train_feats' is not defined

## === cell 17
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
    random_state=SEED,
).fit(X, y)

print("\nRF model OOB score = {:.1f}%".format(100 * rfmodel.oob_score_))
print("RF model train score = {:.1f}%".format(100 * rfmodel.score(X, y)))




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2190209967.py in <cell line: 0>()
----> 1 X = Xy_train_wLRfeats.drop(columns=["clust_id"])
      2 y = Xy_train_wLRfeats.clust_id
      3 
      4 rfmodel = RandomForestClassifier(
      5     n_estimators=300,

NameError: name 'Xy_train_wLRfeats' is not defined

## === cell 18
Xy_valid_meta["pred_id"] = rfmodel.predict(Xy_valid_wLRfeats.drop(columns=["clust_id"]))

solution = Xy_valid_meta[["eeg_id"] + HBA_votes].copy()
for vcol, pcol in zip(HBA_votes, HBA_probs):
    solution[vcol] = Xy_valid_meta[pcol].astype(float)

submission = solution.copy()
pred_ids = Xy_valid_meta["pred_id"].values

best_fracs, best_centers = find_best_tamed_kl()
for iprob in range(HBA_number):
    this_col_probs = best_centers[:, iprob]
    submission[HBA_votes[iprob]] = this_col_probs[pred_ids]

this_kl = kld_score(solution[HBA_votes], submission[HBA_votes])
print("\nValidation KL from tamed centers: {:.4f}".format(this_kl))




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1290540425.py in <cell line: 0>()
----> 1 Xy_valid_meta["pred_id"] = rfmodel.predict(Xy_valid_wLRfeats.drop(columns=["clust_id"]))
      2 
      3 solution = Xy_valid_meta[["eeg_id"] + HBA_votes].copy()
      4 for vcol, pcol in zip(HBA_votes, HBA_probs):
      5     solution[vcol] = Xy_valid_meta[pcol].astype(float)

NameError: name 'rfmodel' is not defined

## === cell 19
Xy_test_feats = assemble_features(
    test_meta, traintest="test", smooth_width=SMOOTH_WIDTH
)

Xy_test_wLRfeats = Xy_test_feats.copy()
Xlr = Xy_test_feats.drop(columns=Xy_test_feats.columns[-10:])

if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : 83 + 1]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    jitter = LR_BLUR * (np.random.rand(len(lrprobas), 1) - 0.5)
    lr_mid = np.clip(lrprobas + jitter, 0.0, 1.0)
    lr_mid_df = pd.DataFrame(lr_mid, columns=[f"lrMid{i}" for i in range(NUM_CLUSTS)])
    Xy_test_wLRfeats = pd.concat([Xy_test_wLRfeats, lr_mid_df], axis=1)

if USE_LR2:
    Xlr2 = Xlr.iloc[:, 84 : 179 + 1]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    jitter = LR_BLUR * (np.random.rand(len(lrprobas), 1) - 0.5)
    lr_fft = np.clip(lrprobas + jitter, 0.0, 1.0)
    lr_fft_df = pd.DataFrame(lr_fft, columns=[f"lrFFT{i}" for i in range(NUM_CLUSTS)])
    Xy_test_wLRfeats = pd.concat([Xy_test_wLRfeats, lr_fft_df], axis=1)

pred_ids = rfmodel.predict(Xy_test_wLRfeats)

test_submit = test_meta[["eeg_id"]].copy()
for new_col in HBA_votes:
    test_submit[new_col] = 1 / HBA_number

for iprob in range(HBA_number):
    this_col_probs = best_centers[:, iprob]
    test_submit[HBA_votes[iprob]] = this_col_probs[pred_ids]

probs = test_submit[HBA_votes].to_numpy(dtype=np.float64)
probs = np.clip(probs, 1e-8, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)
test_submit[HBA_votes] = probs

sample_sub = pd.read_csv(above_dir + "sample_submission.csv")
test_submit = test_submit.reindex(columns=sample_sub.columns)

print(test_submit.head())

test_submit.to_csv(
    "submission.csv", header=True, index=False, na_rep="", float_format="%.6f"
)
print("\nWrote submission.csv with shape:", test_submit.shape)

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1103999606.py in <cell line: 0>()
----> 1 Xy_test_feats = assemble_features(
      2     test_meta, traintest="test", smooth_width=SMOOTH_WIDTH
      3 )
      4 
      5 Xy_test_wLRfeats = Xy_test_feats.copy()

/tmp/ipykernel_11/2506223953.py in assemble_features(meta_frame, traintest, smooth_width)
     92             continue
     93 
---> 94         Xwin = _read_spectro_window_as_numpy(
     95             parquet_path, fftlocbeg, fftlocend
     96         )  # (256, 400)

/tmp/ipykernel_11/3987119564.py in _read_spectro_window_as_numpy(parquet_path, row_start, row_end_inclusive)
     37     # Read only needed rows: take first (row_end+1) rows, then tail nrows.
     38     # This is equivalent to slicing [row_start:row_end+1] but avoids reading the entire file.
---> 39     table = ds.to_table(columns=cols, use_threads=True, limit=row_end_inclusive + 1)
     40     table = table.slice(row_start, nrows)
     41 

/usr/local/lib/python3.11/dist-packages/pyarrow/_dataset.pyx in pyarrow._dataset.Dataset.to_table()

TypeError: to_table() got an unexpected keyword argument 'limit'
