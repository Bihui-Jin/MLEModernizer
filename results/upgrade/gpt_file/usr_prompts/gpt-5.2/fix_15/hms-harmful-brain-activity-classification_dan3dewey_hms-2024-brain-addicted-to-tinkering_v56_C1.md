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

1.0230225952446363

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
def _pick_existing_base(candidates):
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _ensure_trailing_sep(p: str) -> str:
    if p is None:
        return None
    return p if p.endswith(os.sep) else (p + os.sep)


def _pjoin(*parts) -> str:
    return os.path.join(*[str(x) for x in parts])




## === cell 2
import numpy as np

np.random.seed(42)



## === cell 3
NUM_CLUSTS = 9  # 6 to 10

SMOOTH_WIDTH = 5  # Odd>1: 3,5,7,9,...

TRAIN_DOWNSEL = 1
VALID_DOWNSEL = 1

USE_PREPROC = True  # Read in saved meta and features frames

USE_LR1 = True
LR1_C = 1.0  # smaller --> fewer non-zero coeff.s
USE_LR2 = True
LR2_C = 1.0

LR_BLUR = 0.0

above_dir = _pick_existing_base(
    [
        "/kaggle/input/hms-harmful-brain-activity-classification/",
        "/kaggle/data/hms-harmful-brain-activity-classification/",
        "../input/hms-harmful-brain-activity-classification/",
    ]
)
if above_dir is None:
    raise FileNotFoundError(
        "Could not locate hms-harmful-brain-activity-classification dataset directory."
    )

above_dir = _ensure_trailing_sep(above_dir)

above_dir_preproc = _pick_existing_base(
    [
        "/kaggle/input/hms-2024-brain-data/",
        "/kaggle/data/hms-2024-brain-data/",
        "../input/hms-2024-brain-data/",
    ]
)
above_dir_preproc = _ensure_trailing_sep(above_dir_preproc)

DO_PLOTS = False

MAX_TRAIN_ROWS_NO_PREPROC = 3000
MAX_VALID_ROWS_NO_PREPROC = 1000



## === cell 4
preproc_needed = []
if above_dir_preproc is not None:
    preproc_needed = [
        os.path.join(above_dir_preproc, "Xy_train_meta_v47.csv"),
        os.path.join(above_dir_preproc, "Xy_train_feats_v47.csv"),
        os.path.join(above_dir_preproc, "Xy_valid_meta_v47.csv"),
        os.path.join(above_dir_preproc, "Xy_valid_feats_v47.csv"),
    ]

if USE_PREPROC:
    if (
        (above_dir_preproc is None)
        or (not preproc_needed)
        or (not all(os.path.exists(p) for p in preproc_needed))
    ):
        print(
            "Preproc CSVs not found at above_dir_preproc; forcing USE_PREPROC=False (compute minimal features from parquet)."
        )
        USE_PREPROC = False



## === cell 5
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
    print("sklearnex patch applied.")
except Exception as e:
    print("sklearnex patch not applied:", repr(e))

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import pyarrow.parquet as pq

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



## === cell 6
SK_SEED = 42



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
    Calculate the average KL divergence score.
    (Numerically safe) clips to avoid log(0) and division-by-0.
    Expects solution/submission to contain only the 6 probability columns.
    """
    sol = solution.to_numpy(dtype=float, copy=False)
    sub = submission.to_numpy(dtype=float, copy=False)
    sol = np.clip(sol, eps, 1.0)
    sub = np.clip(sub, eps, 1.0)
    kl = np.sum(sol * (np.log(sol) - np.log(sub)), axis=1)
    return float(np.mean(kl))




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
    test_meta["spectrogram_label_offset_seconds"] = 300.0
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
    probs = train_meta[HBA_probs].to_numpy(dtype=float)
    probs = np.clip(probs, 1.0e-8, 1.0)
    train_meta["entropy"] = np.nansum(probs * (-np.log(probs)), axis=1)

    return train_meta, test_meta




## === cell 10
from collections import OrderedDict

_SPECTRO_CACHE = OrderedDict()
_SPECTRO_CACHE_MAX = 64


def _read_spectrogram_df(spectro_file: str) -> pd.DataFrame:
    df = _SPECTRO_CACHE.get(spectro_file)
    if df is not None:
        _SPECTRO_CACHE.move_to_end(spectro_file)
        return df
    if not os.path.exists(spectro_file):
        raise FileNotFoundError(f"Spectrogram parquet not found: {spectro_file}")
    table = pq.read_table(spectro_file, memory_map=True)
    df = table.to_pandas()
    _SPECTRO_CACHE[spectro_file] = df
    if len(_SPECTRO_CACHE) > _SPECTRO_CACHE_MAX:
        _SPECTRO_CACHE.popitem(last=False)
    return df


def _centered_moving_average_same(x: np.ndarray, width: int) -> np.ndarray:
    k = np.ones(width, dtype=np.float64) / float(width)
    y = np.convolve(x, k, mode="same")
    half = (width - 1) // 2
    y[:half] = np.nan
    y[-half:] = np.nan
    return y


def assemble_features(meta_frame, traintest="train", smooth_width=5, SHOW_PLOT=True):
    """
    Create a dataframe of spectrogram features from the meta_frame rows.
    Will include clust_id (i.e, the y) if it is in the input meta_frame.
    Assumes these are available: above_dir, num_clusts
    """
    freqs = np.array(range(100)) * 0.19525 + 0.59
    spect_trend = 150.0 / (1.0**2.3 + freqs ** (2.3))
    freqs[0] = 0.0  # separate the (unsmoothed) first bin from others

    freqs4 = np.repeat(freqs, 4)  # 400
    spect_trend4 = np.repeat(spect_trend, 4)  # 400

    half = int((smooth_width - 1) / 2)
    baseinds = np.insert(np.arange(half, 100, smooth_width), 0, 0)
    select_inds = np.concatenate(
        (baseinds, 100 + baseinds, 200 + baseinds, 300 + baseinds)
    )
    n_ds = len(select_inds)

    mid_cols = [str(i) for i in range(n_ds)]
    rat_cols = ["r" + c for c in mid_cols]
    stat_cols = (
        ["Mean", "Median"]
        + [c + "mean" for c in the4chains]
        + [c + "median" for c in the4chains]
    )
    out_cols = mid_cols + rat_cols + stat_cols
    if "clust_id" in meta_frame.columns:
        out_cols = out_cols + ["clust_id"]

    n_rows = len(meta_frame.index)
    out = np.empty((n_rows, len(out_cols)), dtype=np.float64)

    if SHOW_PLOT:
        plt.figure(figsize=(10, 8))

    last_spectro_id_str = None
    this_spectro_np = None
    nrows_spect = None
    print_every_nth = max([100, 100 * int(0.5 + len(meta_frame.index) / (100.0 * 15))])

    for iout, irow in enumerate(meta_frame.index):
        this_row = meta_frame.loc[irow]
        spectro_id_str = str(int(this_row.spectrogram_id))  # make sure int

        if spectro_id_str != last_spectro_id_str:
            if traintest != "test":
                spectro_file = _pjoin(
                    above_dir, "train_spectrograms", f"{spectro_id_str}.parquet"
                )
            else:
                spectro_file = _pjoin(
                    above_dir, "test_spectrograms", f"{spectro_id_str}.parquet"
                )

            this_spectro = _read_spectrogram_df(spectro_file)
            this_spectro_np = this_spectro.iloc[:, 1:].to_numpy(
                dtype=np.float64, copy=False
            )
            nrows_spect = this_spectro_np.shape[0]
            last_spectro_id_str = spectro_id_str

        loc_offset = int(this_row.spectrogram_label_offset_seconds / 2)
        loc_offset = int(np.clip(loc_offset, 0, max(0, nrows_spect - 1)))
        center_base = int(np.clip(loc_offset + 149, 0, max(0, nrows_spect - 1)))

        i148 = int(np.clip(loc_offset + 148, 0, nrows_spect - 1))
        i149 = int(np.clip(loc_offset + 149, 0, nrows_spect - 1))
        i150 = int(np.clip(loc_offset + 150, 0, nrows_spect - 1))
        i151 = int(np.clip(loc_offset + 151, 0, nrows_spect - 1))

        middle_sum = (
            this_spectro_np[i148]
            + this_spectro_np[i149]
            + this_spectro_np[i150]
            + this_spectro_np[i151]
        )
        middle4s = middle_sum / (4.0 * spect_trend4)
        middle4s = np.clip(middle4s, 0.001, 1000.0)
        bad = ~np.isfinite(middle4s)
        if bad.any():
            middle4s[bad] = 0.001

        j56m = int(np.clip(center_base - 56, 0, nrows_spect - 1))
        j40m = int(np.clip(center_base - 40, 0, nrows_spect - 1))
        j24m = int(np.clip(center_base - 24, 0, nrows_spect - 1))
        j56p = int(np.clip(center_base + 56, 0, nrows_spect - 1))
        j40p = int(np.clip(center_base + 40, 0, nrows_spect - 1))
        j24p = int(np.clip(center_base + 24, 0, nrows_spect - 1))

        denom = (
            this_spectro_np[j56m]
            + this_spectro_np[j40m]
            + this_spectro_np[j24m]
            + this_spectro_np[j56p]
            + this_spectro_np[j40p]
            + this_spectro_np[j24p]
        )
        ratio4s = middle_sum / denom
        ratio4s = np.clip((6.0 / 4.0) * ratio4s, 0.01, 100.0)
        bad = ~np.isfinite(ratio4s)
        if bad.any():
            ratio4s[bad] = 1.0
        ratio4spre = np.log10(ratio4s)

        spect_mean_lin = float(np.mean(middle4s))
        spect_median_lin = float(np.median(middle4s))

        the4means_lin = np.empty(4, dtype=np.float64)
        the4medians_lin = np.empty(4, dtype=np.float64)
        for ispec in range(4):
            ibeg = (0, 100, 200, 300)[ispec]
            iend = ibeg + 100
            seg = middle4s[ibeg:iend]
            the4means_lin[ispec] = np.mean(seg)
            the4medians_lin[ispec] = np.median(seg)

        middle4spre = np.log10(middle4s / spect_mean_lin)

        middle4s_sm = _centered_moving_average_same(middle4spre, smooth_width)
        ratio4s_sm = _centered_moving_average_same(ratio4spre, smooth_width)

        if half > 0:
            for ioff in (0, 100, 200, 300):
                middle4s_sm[ioff : ioff + half] = middle4spre[ioff : ioff + half]
                ratio4s_sm[ioff : ioff + half] = ratio4spre[ioff : ioff + half]

        middle4sds = middle4s_sm[select_inds]
        ratio4sds = ratio4s_sm[select_inds]

        col = 0
        out[iout, col : col + n_ds] = middle4sds
        col += n_ds
        out[iout, col : col + n_ds] = ratio4sds
        col += n_ds

        out[iout, col] = np.log10(spect_mean_lin)
        col += 1
        out[iout, col] = np.log10(spect_median_lin)
        col += 1

        the4means = np.log10(the4means_lin)
        the4medians = np.log10(the4medians_lin)
        out[iout, col : col + 4] = the4means
        col += 4
        out[iout, col : col + 4] = the4medians
        col += 4

        if "clust_id" in meta_frame.columns:
            out[iout, col] = float(this_row.clust_id)
            col += 1

        if (iout + 1) % print_every_nth == 0:
            print("... {} done...".format(iout + 1))

    feats_frame = pd.DataFrame(out, columns=out_cols)
    if SHOW_PLOT:
        plt.savefig("middle8s_" + traintest + "_features.png")
        plt.show()
    return feats_frame


def assemble_features_grouped(
    meta_frame, traintest="test", smooth_width=5, SHOW_PLOT=False
):
    freqs = np.array(range(100)) * 0.19525 + 0.59
    spect_trend = 150.0 / (1.0**2.3 + freqs ** (2.3))
    freqs[0] = 0.0
    freqs4 = np.repeat(freqs, 4)
    spect_trend4 = np.repeat(spect_trend, 4)

    half = int((smooth_width - 1) / 2)
    baseinds = np.insert(np.arange(half, 100, smooth_width), 0, 0)
    select_inds = np.concatenate(
        (baseinds, 100 + baseinds, 200 + baseinds, 300 + baseinds)
    )
    n_ds = len(select_inds)

    mid_cols = [str(i) for i in range(n_ds)]
    rat_cols = ["r" + c for c in mid_cols]
    stat_cols = (
        ["Mean", "Median"]
        + [c + "mean" for c in the4chains]
        + [c + "median" for c in the4chains]
    )
    out_cols = mid_cols + rat_cols + stat_cols
    if "clust_id" in meta_frame.columns:
        out_cols = out_cols + ["clust_id"]

    n_rows = len(meta_frame)
    out = np.empty((n_rows, len(out_cols)), dtype=np.float64)

    meta_frame = meta_frame.reset_index(drop=True)
    groups = meta_frame.groupby("spectrogram_id", sort=False)

    print_every = max(50, int(len(groups) / 10))
    for ig, (spectro_id, g) in enumerate(groups):
        spectro_id_str = str(int(spectro_id))
        if traintest != "test":
            spectro_file = _pjoin(
                above_dir, "train_spectrograms", f"{spectro_id_str}.parquet"
            )
        else:
            spectro_file = _pjoin(
                above_dir, "test_spectrograms", f"{spectro_id_str}.parquet"
            )

        this_spectro = _read_spectrogram_df(spectro_file)
        this_spectro_np = this_spectro.iloc[:, 1:].to_numpy(
            dtype=np.float64, copy=False
        )
        nrows_spect = this_spectro_np.shape[0]

        for iout in g.index.to_numpy():
            this_row = meta_frame.loc[iout]

            loc_offset = int(this_row.spectrogram_label_offset_seconds / 2)
            loc_offset = int(np.clip(loc_offset, 0, max(0, nrows_spect - 1)))
            center_base = int(np.clip(loc_offset + 149, 0, max(0, nrows_spect - 1)))

            i148 = int(np.clip(loc_offset + 148, 0, nrows_spect - 1))
            i149 = int(np.clip(loc_offset + 149, 0, nrows_spect - 1))
            i150 = int(np.clip(loc_offset + 150, 0, nrows_spect - 1))
            i151 = int(np.clip(loc_offset + 151, 0, nrows_spect - 1))

            middle_sum = (
                this_spectro_np[i148]
                + this_spectro_np[i149]
                + this_spectro_np[i150]
                + this_spectro_np[i151]
            )
            middle4s = middle_sum / (4.0 * spect_trend4)
            middle4s = np.clip(middle4s, 0.001, 1000.0)
            bad = ~np.isfinite(middle4s)
            if bad.any():
                middle4s[bad] = 0.001

            j56m = int(np.clip(center_base - 56, 0, nrows_spect - 1))
            j40m = int(np.clip(center_base - 40, 0, nrows_spect - 1))
            j24m = int(np.clip(center_base - 24, 0, nrows_spect - 1))
            j56p = int(np.clip(center_base + 56, 0, nrows_spect - 1))
            j40p = int(np.clip(center_base + 40, 0, nrows_spect - 1))
            j24p = int(np.clip(center_base + 24, 0, nrows_spect - 1))

            denom = (
                this_spectro_np[j56m]
                + this_spectro_np[j40m]
                + this_spectro_np[j24m]
                + this_spectro_np[j56p]
                + this_spectro_np[j40p]
                + this_spectro_np[j24p]
            )
            ratio4s = middle_sum / denom
            ratio4s = np.clip((6.0 / 4.0) * ratio4s, 0.01, 100.0)
            bad = ~np.isfinite(ratio4s)
            if bad.any():
                ratio4s[bad] = 1.0
            ratio4spre = np.log10(ratio4s)

            spect_mean_lin = float(np.mean(middle4s))
            spect_median_lin = float(np.median(middle4s))

            the4means_lin = np.empty(4, dtype=np.float64)
            the4medians_lin = np.empty(4, dtype=np.float64)
            for ispec in range(4):
                ibeg = (0, 100, 200, 300)[ispec]
                iend = ibeg + 100
                seg = middle4s[ibeg:iend]
                the4means_lin[ispec] = np.mean(seg)
                the4medians_lin[ispec] = np.median(seg)

            middle4spre = np.log10(middle4s / spect_mean_lin)

            middle4s_sm = _centered_moving_average_same(middle4spre, smooth_width)
            ratio4s_sm = _centered_moving_average_same(ratio4spre, smooth_width)

            if half > 0:
                for ioff in (0, 100, 200, 300):
                    middle4s_sm[ioff : ioff + half] = middle4spre[ioff : ioff + half]
                    ratio4s_sm[ioff : ioff + half] = ratio4spre[ioff : ioff + half]

            middle4sds = middle4s_sm[select_inds]
            ratio4sds = ratio4s_sm[select_inds]

            col = 0
            out[iout, col : col + n_ds] = middle4sds
            col += n_ds
            out[iout, col : col + n_ds] = ratio4sds
            col += n_ds

            out[iout, col] = np.log10(spect_mean_lin)
            col += 1
            out[iout, col] = np.log10(spect_median_lin)
            col += 1

            the4means = np.log10(the4means_lin)
            the4medians = np.log10(the4medians_lin)
            out[iout, col : col + 4] = the4means
            col += 4
            out[iout, col : col + 4] = the4medians
            col += 4

            if "clust_id" in meta_frame.columns:
                out[iout, col] = float(this_row.clust_id)
                col += 1

        if (ig + 1) % print_every == 0:
            print(f"... spectrograms processed: {ig+1}/{len(groups)}")

    feats_frame = pd.DataFrame(out, columns=out_cols)
    return feats_frame




## === cell 11
def find_best_tamed_kl(solution, submission, pred_ids):
    """
    Adjust the taming fraction for each cluster center to optimize KL.

    Bugfix: pred_ids may be a pandas Series; convert to a 1D integer numpy array
    so it can index into numpy arrays (avoids IndexError).
    """
    pred_ids = np.asarray(pred_ids, dtype=np.int64)

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




## === cell 12
def normalize_submission_probs(df, prob_cols, eps=1e-6):
    vals = df[prob_cols].to_numpy(dtype=float)
    vals = np.nan_to_num(vals, nan=eps, posinf=1.0, neginf=eps)
    vals = np.clip(vals, eps, 1.0)
    vals = vals / vals.sum(axis=1, keepdims=True)
    df.loc[:, prob_cols] = vals
    return df




## === cell 13
train_meta, test_meta = read_hms_meta()



## === cell 14
if DO_PLOTS:
    plt.figure(figsize=(5, 2))
    plt.hist(train_meta["spectrogram_sub_id"], bins=55, log=True)
    plt.title("Histogram of spectrogram_sub_id")
    plt.show()

    plt.figure(figsize=(5, 2))
    plt.hist(train_meta["eeg_sub_id"], bins=55, log=True)
    plt.title("Histogram of eeg_sub_id")
    plt.show()



## === cell 15
pass



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
    n_clusters=num_clusts,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=SK_SEED,
)
kmeans.fit(prob_array)

clust_centers = kmeans.cluster_centers_
clust_centers = clust_centers / clust_centers.sum(axis=1, keepdims=True)

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

if DO_PLOTS:
    kmclrs = ["blue"] * max(NUM_CLUSTS, 10)
else:
    kmclrs = ["blue"] * max(NUM_CLUSTS, 10)

clust_counts = train_meta.clust_id.value_counts()

iclust_order_of_clust = num_clusts * [-1]
for iord, iclust in enumerate(iclust_of_order):
    iclust_order_of_clust[iclust] = iord



## === cell 19
pass



## === cell 20
solution_train = train_meta[["eeg_id"] + HBA_probs].copy()
solution_train.columns = [
    "eeg_id"
] + HBA_votes  # treat these as the probability targets
submission_train = solution_train.copy()

clust_ids = train_meta["clust_id"].to_numpy(dtype=np.int64, copy=False)
for iprob in range(HBA_number):
    this_col_probs = clust_centers[:, iprob]
    submission_train[HBA_votes[iprob]] = this_col_probs[clust_ids]

print(
    "Score if HBA samples are correctly assigned cluster prob.s:",
    np.round(
        kld_score(
            solution_train.drop(columns=["eeg_id"]),
            submission_train.drop(columns=["eeg_id"]),
        ),
        4,
    ),
)



## === cell 21
pass



## === cell 22
pass



## === cell 23
pass



## === cell 24
pass



## === cell 25
pass



## === cell 26
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



## === cell 27
pass



## === cell 28
if USE_PREPROC:
    Xy_train_meta = pd.read_csv(above_dir_preproc + "Xy_train_meta_v47.csv")
    Xy_train_feats = pd.read_csv(above_dir_preproc + "Xy_train_feats_v47.csv")
    Xy_train_meta["clust_id"] = kmeans.predict(np.array(Xy_train_meta[HBA_probs]))
    Xy_train_feats["clust_id"] = Xy_train_meta["clust_id"]
    SMOOTH_WIDTH = 5
else:
    Xy_train_meta = (train_meta[train_rows_bool])[::TRAIN_DOWNSEL].copy()
    Xy_train_meta = Xy_train_meta.reset_index().drop(columns=["index"])

    if len(Xy_train_meta) > MAX_TRAIN_ROWS_NO_PREPROC:
        Xy_train_meta = Xy_train_meta.iloc[:MAX_TRAIN_ROWS_NO_PREPROC].copy()

    print("Number of samples used for training =", len(Xy_train_meta))

    Xy_train_feats = assemble_features_grouped(
        Xy_train_meta, traintest="train", smooth_width=SMOOTH_WIDTH, SHOW_PLOT=False
    )
    Xy_train_meta.to_csv(
        "Xy_train_meta.csv", header=True, index=False, float_format="%.6f"
    )
    Xy_train_feats.to_csv(
        "Xy_train_feats.csv", header=True, index=False, float_format="%.6f"
    )



## === cell 29
Xy_train_feats



## === cell 30
pass



## === cell 31
if USE_PREPROC:
    Xy_valid_meta = pd.read_csv(above_dir_preproc + "Xy_valid_meta_v47.csv")
    Xy_valid_feats = pd.read_csv(above_dir_preproc + "Xy_valid_feats_v47.csv")
    Xy_valid_meta["clust_id"] = kmeans.predict(np.array(Xy_valid_meta[HBA_probs]))
    Xy_valid_feats["clust_id"] = Xy_valid_meta["clust_id"]
else:
    Xy_valid_meta = (train_meta[valid_rows_bool])[::VALID_DOWNSEL].copy()
    Xy_valid_meta = Xy_valid_meta.reset_index().drop(columns=["index"])

    if len(Xy_valid_meta) > MAX_VALID_ROWS_NO_PREPROC:
        Xy_valid_meta = Xy_valid_meta.iloc[:MAX_VALID_ROWS_NO_PREPROC].copy()

    print("Number of samples used for Validation =", len(Xy_valid_meta))

    Xy_valid_feats = assemble_features_grouped(
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



## === cell 32
Xy_valid_feats



## === cell 33
pass



## === cell 34
X = Xy_train_feats.drop(columns=["clust_id"])
y = Xy_train_feats.clust_id

Xlr = X.drop(columns=X.columns[-10:])

if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : int(len(Xlr.columns) / 4)]
    lrmodel1 = LogisticRegression(
        penalty="l2",
        C=LR1_C,
        solver="saga",
        max_iter=1500,
        multi_class="multinomial",
        n_jobs=-1,
        random_state=SK_SEED,
    ).fit(Xlr1, y)

    print("\nLR1 model score for X,y = {:.1f}%\n".format(100 * lrmodel1.score(Xlr1, y)))

if USE_LR2:
    Xlr2 = Xlr.iloc[:, int(len(Xlr.columns) / 4) : int(len(Xlr.columns) / 2)]
    lrmodel2 = LogisticRegression(
        penalty="l2",
        C=LR2_C,
        solver="saga",
        max_iter=1500,
        multi_class="multinomial",
        n_jobs=-1,
        random_state=SK_SEED,
    ).fit(Xlr2, y)

    print("\nLR2 model score for X,y = {:.1f}%\n".format(100 * lrmodel2.score(Xlr2, y)))




## === cell 35
def _sanitize_lr_features(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=eps, posinf=1.0, neginf=eps)
    return np.clip(arr, eps, 1.0)


_lr_blur_rng = np.random.default_rng(SK_SEED)

Xy_train_wLRfeats = Xy_train_feats.copy()
X = Xy_train_feats.drop(columns=["clust_id"])
Xlr = X.drop(columns=X.columns[-10:])
if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : int(len(Xlr.columns) / 4)]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    blur = LR_BLUR * (_lr_blur_rng.random(lrprobas.shape) - 0.5)
    lrprobas_blur = _sanitize_lr_features(lrprobas + blur)
    for iadd in range(NUM_CLUSTS):
        Xy_train_wLRfeats["lr" + str(iadd)] = lrprobas_blur[:, iadd]
if USE_LR2:
    Xlr2 = Xlr.iloc[:, int(len(Xlr.columns) / 4) : int(len(Xlr.columns) / 2)]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    blur = LR_BLUR * (_lr_blur_rng.random(lrprobas.shape) - 0.5)
    lrprobas_blur = _sanitize_lr_features(lrprobas + blur)
    for iadd in range(NUM_CLUSTS):
        Xy_train_wLRfeats["rlr" + str(iadd)] = lrprobas_blur[:, iadd]


Xy_valid_wLRfeats = Xy_valid_feats.copy()
X = Xy_valid_feats.drop(columns=["clust_id"])
Xlr = X.drop(columns=X.columns[-10:])
if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : int(len(Xlr.columns) / 4)]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    blur = LR_BLUR * (_lr_blur_rng.random(lrprobas.shape) - 0.5)
    lrprobas_blur = _sanitize_lr_features(lrprobas + blur)
    for iadd in range(NUM_CLUSTS):
        Xy_valid_wLRfeats["lr" + str(iadd)] = lrprobas_blur[:, iadd]
if USE_LR2:
    Xlr2 = Xlr.iloc[:, int(len(Xlr.columns) / 4) : int(len(Xlr.columns) / 2)]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    blur = LR_BLUR * (_lr_blur_rng.random(lrprobas.shape) - 0.5)
    lrprobas_blur = _sanitize_lr_features(lrprobas + blur)
    for iadd in range(NUM_CLUSTS):
        Xy_valid_wLRfeats["rlr" + str(iadd)] = lrprobas_blur[:, iadd]



## === cell 36
pass



## === cell 37
X = Xy_train_wLRfeats.drop(columns=["clust_id"])
y = Xy_train_wLRfeats.clust_id

ave_oob = []
nfits = 3
for ifit in range(nfits):
    rfmodel = RandomForestClassifier(
        n_estimators=300,  # was 100
        min_samples_leaf=5,  # was 9
        max_features=0.2,
        max_samples=0.9,  # was 0.8
        oob_score=True,
        class_weight="balanced_subsample",
        n_jobs=-1,
        verbose=0,
        random_state=SK_SEED + ifit,
    ).fit(X, y)
    ave_oob.append(rfmodel.oob_score_)

print(
    "\nRF model ave OOB score = {:.1f}% +/- {:.1f}".format(
        100 * np.mean(ave_oob), 100 * np.std(ave_oob)
    )
)

print("\nRF model score for X,y = {:.1f}%\n".format(100 * rfmodel.score(X, y)))



## === cell 38
Xy_train_meta["pred_id"] = rfmodel.predict(Xy_train_wLRfeats.drop(columns=["clust_id"]))

solution = Xy_train_meta[["eeg_id"] + HBA_probs].copy()
solution.columns = ["eeg_id"] + HBA_votes
submission = solution.copy()
pred_ids = Xy_train_meta["pred_id"].to_numpy(dtype=np.int64, copy=False)

best_fracs_train, best_centers_train = find_best_tamed_kl(
    solution=solution.drop(columns=["eeg_id"]),
    submission=submission.drop(columns=["eeg_id"]).copy(),
    pred_ids=pred_ids,
)

for iprob in range(HBA_number):
    this_col_probs = best_centers_train[:, iprob]
    submission[HBA_votes[iprob]] = this_col_probs[pred_ids]
this_kl = kld_score(
    solution.drop(columns=["eeg_id"]), submission.drop(columns=["eeg_id"])
)
print(
    "Tamed fractions (train-fit):\n",
    best_fracs_train,
    "\nand centers:\n",
    best_centers_train,
)
print("\nKL from tamed centers (train-fit): {:.4f}".format(this_kl))



## === cell 39
pass



## === cell 40
Xy_valid_meta["pred_id"] = rfmodel.predict(Xy_valid_wLRfeats.drop(columns=["clust_id"]))

solution = Xy_valid_meta[["eeg_id"] + HBA_probs].copy()
solution.columns = ["eeg_id"] + HBA_votes
submission = solution.copy()
pred_ids = Xy_valid_meta["pred_id"].to_numpy(dtype=np.int64, copy=False)

best_fracs_valid, best_centers_valid = find_best_tamed_kl(
    solution=solution.drop(columns=["eeg_id"]),
    submission=submission.drop(columns=["eeg_id"]).copy(),
    pred_ids=pred_ids,
)

for iprob in range(HBA_number):
    this_col_probs = best_centers_valid[:, iprob]
    submission[HBA_votes[iprob]] = this_col_probs[pred_ids]
this_kl = kld_score(
    solution.drop(columns=["eeg_id"]), submission.drop(columns=["eeg_id"])
)
print(
    "Tamed fractions (valid-tuned):\n",
    best_fracs_valid,
    "\nand centers:\n",
    best_centers_valid,
)
print("\nKL from tamed centers (valid-tuned): {:.4f}".format(this_kl))

best_centers = best_centers_valid.copy()



## === cell 41
pass



## === cell 42
pass



## === cell 43
pass



## === cell 44
pass



## === cell 45
pass



## === cell 46
SUB_EPS = 1e-5

test_meta_all = test_meta.copy().reset_index(drop=True)
print("Test rows:", len(test_meta_all))

Xy_test_feats = assemble_features_grouped(
    test_meta_all, traintest="test", smooth_width=SMOOTH_WIDTH, SHOW_PLOT=False
)

Xy_test_wLRfeats = Xy_test_feats.copy()
Xlr = Xy_test_feats.drop(columns=Xy_test_feats.columns[-10:])
if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : int(len(Xlr.columns) / 4)]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    lrprobas = _sanitize_lr_features(lrprobas, eps=SUB_EPS)
    for iadd in range(NUM_CLUSTS):
        Xy_test_wLRfeats["lr" + str(iadd)] = lrprobas[:, iadd]
if USE_LR2:
    Xlr2 = Xlr.iloc[:, int(len(Xlr.columns) / 4) : int(len(Xlr.columns) / 2)]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    lrprobas = _sanitize_lr_features(lrprobas, eps=SUB_EPS)
    for iadd in range(NUM_CLUSTS):
        Xy_test_wLRfeats["rlr" + str(iadd)] = lrprobas[:, iadd]

X_test_for_rf = Xy_test_wLRfeats.loc[:, rfmodel.feature_names_in_]
pred_ids = rfmodel.predict(X_test_for_rf).astype(np.int64, copy=False)

test_full = test_meta_all[["eeg_id"]].copy()
for new_col in HBA_votes:
    test_full[new_col] = 1.0 / HBA_number
for iprob in range(HBA_number):
    this_col_probs = best_centers[:, iprob]
    test_full[HBA_votes[iprob]] = this_col_probs[pred_ids]

test_full = normalize_submission_probs(test_full, HBA_votes, eps=SUB_EPS)

sample_sub = pd.read_csv(above_dir + "sample_submission.csv")
test_full = sample_sub[["eeg_id"]].merge(test_full, on="eeg_id", how="left")
test_full[HBA_votes] = test_full[HBA_votes].fillna(1.0 / HBA_number)
test_full = normalize_submission_probs(test_full, HBA_votes, eps=SUB_EPS)
test_full = test_full[["eeg_id"] + HBA_votes].copy()

print(test_full.head())

test_full.to_csv(
    "submission.csv", header=True, index=False, na_rep="", float_format="%.6f"
)
print("\nWrote submission.csv with shape:", test_full.shape)
print(
    "Row-sum stats (min/mean/max):",
    float(test_full[HBA_votes].sum(axis=1).min()),
    float(test_full[HBA_votes].sum(axis=1).mean()),
    float(test_full[HBA_votes].sum(axis=1).max()),
)
print("Submission length matches sample_submission:", len(test_full) == len(sample_sub))
print(
    "Submission columns match sample_submission:",
    list(test_full.columns) == list(sample_sub.columns),
)
