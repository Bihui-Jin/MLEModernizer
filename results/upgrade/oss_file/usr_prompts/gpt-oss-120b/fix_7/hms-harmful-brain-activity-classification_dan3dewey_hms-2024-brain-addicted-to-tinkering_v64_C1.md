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

1.0116395317562643

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
NUM_CLUSTS = 6  # 6 to 10

TRAIN_DOWNSEL = 1  # large values for code test; set to 1 to output all.
VALID_DOWNSEL = 1  #  "
SMOOTH_WIDTH = 5  # Odd>1: 3,5,7,9,...
USE_PREPROC = (
    False  # Read-in saved meta and features frames (set False to generate on the fly)
)

USE_LR1 = True
LR1_C = 1.0  # smaller --> fewer non-zero coeff.s
USE_LR2 = True
LR2_C = 1.0
LR_BLUR = 0.10

PLOT_SPECTRA = False  # set True only if visualisation is needed

above_dir = "../input/hms-harmful-brain-activity-classification/"
above_dir_preproc = "../input/hms-2024-brain-data/"




## === cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()
np.random.seed(42)  # deterministic randomness

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import pyarrow
import pyarrow.parquet as pq
import pyarrow.dataset as pads

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from joblib import Parallel, delayed
import os




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

    plt.figure(figsize=(6, 3))
    plt.hist(train_meta["total_vote"], bins=55, log=True)
    plt.title("Histogram of Total Votes")
    plt.close()

    for col_pre in HBA_names:
        train_meta[col_pre + "_prob"] = (
            train_meta[col_pre + "_vote"] / train_meta["total_vote"]
        )

    print("Calculating voting entropy values ...")

    def calc_entropy(row):
        the_probs = np.clip(row[16 : 21 + 1].values.astype(float), 1.0e-8, 1.0)
        return np.nansum(the_probs * -1 * np.log(the_probs))

    train_meta["entropy"] = train_meta.apply(calc_entropy, axis=1)

    return train_meta, test_meta




## === cell 5
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
    plt.close()
    return kmclrs  # returns cluster colors appropriate for iclust




## === cell 6
def _process_single_row(
    this_row,
    spectro_cache,
    freqs,
    spect_trend,
    baseinds,
    freqs4,
    n_fft_bins,
    apod_wind,
    fftfeatbins,
    the4chains,
    smooth_width,
):
    """
    Helper for parallel row processing – returns a DataFrame with the features for one row.
    All calculations are identical to the original loop.
    """
    this_spectro = spectro_cache[this_row.spectrogram_id]

    loc_offset = int(this_row.spectrogram_label_offset_seconds / 2)
    fftlocbeg = int(loc_offset + 149 - (n_fft_bins / 2 - 1))
    fftlocend = int(loc_offset + 150 + (n_fft_bins / 2 - 1))

    middle8s = (
        this_spectro.iloc[loc_offset + 148, 1:]
        + this_spectro.iloc[loc_offset + 149, 1:]
        + this_spectro.iloc[loc_offset + 150, 1:]
        + this_spectro.iloc[loc_offset + 151, 1:]
    ) / (4.0 * spect_trend4)

    middle8s = np.clip(middle8s, 0.001, 1000.0)
    middle8s = middle8s.replace([np.nan, -np.inf, np.inf], 0.001)
    spect_mean = np.mean(middle8s)
    spect_median = np.median(middle8s)

    the4means = []
    the4medians = []
    for ispec in range(4):
        ibeg = ([0, 100, 200, 300])[ispec]
        iend = ibeg + 100
        the4means.append(np.mean(middle8s[ibeg:iend]))
        the4medians.append(np.median(middle8s[ibeg:iend]))

    middle8spre = np.log10(middle8s / spect_mean)
    middle8s = middle8spre.rolling(
        smooth_width, min_periods=smooth_width, center=True, closed=None
    ).mean()
    for ioff in range(0, 399, 100):
        for ibin in range(int((smooth_width - 1) / 2)):
            middle8s[ibin + ioff] = middle8spre[ibin + ioff]

    select_inds = np.concatenate(
        (baseinds, 100 + baseinds, 200 + baseinds, 300 + baseinds)
    )
    freqs4ds = freqs4[select_inds]
    middle8sds = middle8s[select_inds]
    middle_feats = middle8sds.to_frame().T

    flarecols = []
    flarevals = []
    for ispec in range(4):
        for ifreq, freqbin in enumerate(freqbins):
            sum_spect_trend = sum(spect_trend[freqbin - 4 : freqbin + 4 + 1 : 2])
            ifreqoff = freqbin + 1 + ispec * 100
            amplvstime = (
                this_spectro.iloc[fftlocbeg : fftlocend + 1, ifreqoff - 4]
                + this_spectro.iloc[fftlocbeg : fftlocend + 1, ifreqoff - 2]
                + this_spectro.iloc[fftlocbeg : fftlocend + 1, ifreqoff]
                + this_spectro.iloc[fftlocbeg : fftlocend + 1, ifreqoff + 2]
                + this_spectro.iloc[fftlocbeg : fftlocend + 1, ifreqoff + 4]
            ) / sum_spect_trend
            amplvstime = np.clip(amplvstime, 0.001, 1000.0)
            amplvstime = amplvstime.replace([np.nan, -np.inf, np.inf], 0.001)
            amplvstime = (
                2.0
                * amplvstime
                / (amplvstime[fftlocbeg + 127] + amplvstime[fftlocbeg + 128])
            )
            amplvstime = apod_wind * np.clip(amplvstime, 0.0, 10.0)
            for iroll in range(2):
                amplvstime = amplvstime.rolling(
                    2, min_periods=1, center=False, closed=None
                ).mean()
            amplfft = np.log10(
                1 + np.abs(np.fft.fft(amplvstime))[0 : int(n_fft_bins / 2)]
            )
            for fftfeatbin in fftfeatbins:
                flarecols.append(
                    "fft-"
                    + the4chains[ispec]
                    + "{:.1f}-".format(freqs[freqbin])
                    + str(fftfeatbin)
                )
                flarevals.append(amplfft[fftfeatbin])

    flare_feats = pd.DataFrame([flarevals], columns=flarecols)
    these_feats = pd.concat([middle_feats, flare_feats], axis=1)

    spect_mean = np.log10(spect_mean)
    spect_median = np.log10(spect_median)
    the4means = np.log10(the4means)
    the4medians = np.log10(the4medians)

    these_feats["Mean"] = spect_mean
    these_feats["Median"] = spect_median
    for ispec in range(4):
        these_feats[the4chains[ispec] + "mean"] = the4means[ispec]
        these_feats[the4chains[ispec] + "median"] = the4medians[ispec]
    if "clust_id" in this_row.index:
        these_feats["clust_id"] = this_row.clust_id
    return these_feats


def assemble_features(
    meta_frame,
    traintest="train",
    smooth_width=5,
):
    """
    Create a dataframe of spectrogram features from the meta_frame rows.
    Will include clust_id (i.e, the y) if it is in the input meta_frame.
    Assumes these are available: above_dir, the4chains
    """
    if traintest == "validation":
        plt.figure(figsize=(9, 7))

    freqs = np.array(range(100)) * 0.19525 + 0.59
    spect_trend = 150.0 / (1.0**2.3 + freqs ** (2.3))
    baseinds = np.insert(
        np.arange(int((smooth_width - 1) / 2), 100, smooth_width), 0, 0
    )
    freqs4 = np.array(4 * list(freqs))
    spect_trend4 = np.array(4 * list(spect_trend))

    n_fft_bins = 256
    freqbins = [7, 18, 60, 84]
    apod_wind = np.blackman(n_fft_bins)
    fftfeatbins = [4, 9, 16, 25, 36, 49]

    unique_ids = meta_frame["spectrogram_id"].unique()
    spectro_cache = {}
    for sid in unique_ids:
        sid_str = str(int(sid))
        if traintest != "test":
            spectro_file = above_dir + "train_spectrograms/" + sid_str + ".parquet"
        else:
            spectro_file = above_dir + "test_spectrograms/" + sid_str + ".parquet"
        spectro_cache[sid] = pq.read_table(spectro_file).to_pandas()

    kmclrs_local = globals().get("kmclrs", None)

    n_jobs = max(1, os.cpu_count() - 1)
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_process_single_row)(
            meta_frame.loc[irow],
            spectro_cache,
            freqs,
            spect_trend,
            baseinds,
            freqs4,
            n_fft_bins,
            apod_wind,
            fftfeatbins,
            the4chains,
            smooth_width,
        )
        for irow in meta_frame.index
    )

    feats_frame = pd.concat(results, ignore_index=True)

    if traintest == "validation":
        plt.ylim(-0.2, 2.0)
        plt.xlim(0.0, 11.5)
        plt.xlabel("sqrt[  FFT bin number ]")
        plt.ylabel("log10[1+ FFT amplitude ]")
        plt.title("Flare-Rate Spectra (colored by cluster)")
        plt.close()

    return feats_frame.reset_index().drop(columns=["index"])




## === cell 7
solution_train = None  # placeholder to keep cell numbering consistent




## === cell 8
train_meta, test_meta = read_hms_meta()




## === cell 9
plt.figure(figsize=(5, 2))
plt.hist(train_meta["spectrogram_sub_id"], bins=55, log=True)
plt.title("Histogram of spectrogram_sub_id")
plt.close()

plt.figure(figsize=(5, 2))
plt.hist(train_meta["eeg_sub_id"], bins=55, log=True)
plt.title("Histogram of eeg_sub_id")
plt.close()




## === cell 10
num_clusts = NUM_CLUSTS  # 6 to 10

clust_rows_bool = train_meta.eeg_sub_id < 200




## === cell 11
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
plt.close()
prob_array = np.array(prob_vectors)
kmeans = KMeans(
    n_clusters=num_clusts, init="k-means++", n_init=10, max_iter=300, random_state=None
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




## === cell 12
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
    plt.close()




## === cell 13
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




## === cell 14
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




## === cell 15
if PLOT_SPECTRA:
    all_medians = []
    all_means = []
    all_clusts = []

    fig = plt.figure(figsize=(10, 3.5 * NUM_CLUSTS))
    gs = fig.add_gridspec(NUM_CLUSTS, 2, hspace=0.05, wspace=0.25)
    pltaxs = gs.subplots(sharex="col")

    chainclrs = ["red", "blue", "green", "yellow"]

    print_every_nth = max(
        [100, 100 * int(0.5 + (len(spectro_meta) / every_nth) / (100.0 * 15))]
    )

    print(
        "Plotting {} x 4 processed spectra".format(int(len(spectro_meta) / every_nth))
    )
    iloc_rows = range(0, len(spectro_meta), every_nth)

    plt_alpha = min([0.25 * (200 / len(iloc_rows)), 1.0])
    for ilocrow in iloc_rows:
        this_row = spectro_meta.iloc[ilocrow]
        this_clust = this_row.clust_id  # the k-means iclust
        this_std_clust = iorder_of_clust[this_row.clust_id]
        if True:
            spectro_id_str = str(this_row.spectrogram_id)
            spectro_file = (
                above_dir + "train_spectrograms/" + spectro_id_str + ".parquet"
            )
            pads_spectro = pads.dataset(spectro_file)
            this_spectro = pads_spectro.to_table().to_pandas()
            loc_offset = int(this_row.spectrogram_label_offset_seconds / 2)  # 2s bins
            locbeg = int(loc_offset + 149 - (n_fft_bins / 2 - 1))
            locend = int(loc_offset + 150 + (n_fft_bins / 2 - 1))
            for ispec in range(4):

                ibeg = ([1, 101, 201, 301])[ispec]
                iend = ibeg + 100
                middle8s = (
                    this_spectro.iloc[loc_offset + 148, ibeg:iend]
                    + this_spectro.iloc[loc_offset + 149, ibeg:iend]
                    + this_spectro.iloc[loc_offset + 150, ibeg:iend]
                    + this_spectro.iloc[loc_offset + 151, ibeg:iend]
                ) / (4.0 * spect_trend)
                middle8s = np.clip(middle8s, 0.001, 1000.0)
                middle8s = middle8s.replace([np.nan, -np.inf, np.inf], 0.001)
                spect_mean = np.log10(np.mean(middle8s))
                all_means.append(spect_mean)
                spect_median = np.log10(np.median(middle8s))
                all_medians.append(spect_median)
                all_clusts.append(this_clust)
                middle8s = np.log10(middle8s)
                middle8spre = middle8s - spect_mean
                middle8s = middle8spre.rolling(
                    smooth_width, min_periods=smooth_width, center=True, closed=None
                ).mean()
                for ibin in range(int((smooth_width - 1) / 2)):
                    middle8s[ibin] = middle8spre[ibin]
                pltaxs[this_std_clust, 0].plot(
                    (freqs), middle8s, c=chainclrs[ispec], lw=2, alpha=plt_alpha
                )

                for ifreq, freqbin in enumerate(freqbins):
                    sum_spect_trend = sum(
                        spect_trend[freqbin - 4 : freqbin + 4 + 1 : 2]
                    )
                    ifreqoff = freqbin + 1 + ispec * 100
                    amplvstime = (
                        this_spectro.iloc[locbeg : locend + 1, ifreqoff - 4]
                        + this_spectro.iloc[locbeg : locend + 1, ifreqoff - 2]
                        + this_spectro.iloc[locbeg : locend + 1, ifreqoff]
                        + this_spectro.iloc[locbeg : locend + 1, ifreqoff + 2]
                        + this_spectro.iloc[locbeg : locend + 1, ifreqoff + 4]
                    ) / (sum_spect_trend)
                    amplvstime = np.clip(amplvstime, 0.001, 1000.0)
                    amplvstime = amplvstime.replace([np.nan, -np.inf, np.inf], 0.001)
                    amplvstime = (
                        2.0
                        * amplvstime
                        / (amplvstime[locbeg + 127] + amplvstime[locbeg + 128])
                    )
                    amplvstime = apod_wind * np.clip(amplvstime, 0.0, 10.0)
                    for iroll in range(2):
                        amplvstime = amplvstime.rolling(
                            2, min_periods=1, center=False, closed=None
                        ).mean()
                    amplfft = np.log10(
                        1 + np.abs(np.fft.fft(amplvstime))[0 : int(n_fft_bins / 2)]
                    )
                    pltaxs[this_std_clust, 1].plot(
                        np.sqrt(range(int(n_fft_bins / 2))),
                        amplfft,
                        c=freqclrs[ifreq],
                        lw=2,
                        alpha=plt_alpha / 7.0,
                    )

        if (ilocrow / every_nth + 1) % print_every_nth == 0:
            print("... {} done...".format(int(ilocrow / every_nth) + 1))

    print(
        "\n\n Colors are: LL-red, RL-blue, LP-green, RP-yellow "
        + "         f_Hz-color: 2.0-red, 4.1-blue, 12.3-green, 17.0-yell."
    )

    for iax in range(NUM_CLUSTS):
        if True:
            pltaxs[iax, 0].text(12.5, 0.8, kmnames[iax], fontsize=14)
            pltaxs[iax, 0].plot([0.0, 20.0], [0.0, 0.0], c="black", lw=3, alpha=0.2)
            pltaxs[iax, 0].plot(
                downsel_freqs, len(downsel_freqs) * [0.0], ".", c="gray"
            )
            pltaxs[iax, 0].set_ylim(-1.5, 1.0)
            pltaxs[iax, 0].set_xlim(0.0, 20.5)
            pltaxs[iax, 1].plot(
                [0.0, len(amplvstime)], [0.0, 0.0], c="black", lw=3, alpha=0.2
            )
            pltaxs[iax, 1].set_ylim(-0.2, 2.0)
            pltaxs[iax, 1].set_xlim(0.0, 11.5)

    pltaxs[NUM_CLUSTS - 1, 0].set_ylabel(
        25 * " " + "log10[ Spectra /RefSpectrum /Mean" + " & smoothed]"
    )
    pltaxs[NUM_CLUSTS - 1, 0].set_xlabel("Frequency (Hz)")
    pltaxs[NUM_CLUSTS - 1, 1].set_ylabel("log10[1+ FFT amplitude ]")
    pltaxs[NUM_CLUSTS - 1, 1].set_xlabel("sqrt[  FFT bin number ]")
    pltaxs[0, 0].set_title(
        "Middle 4x2s Spectra (n_sm={}, colored by LL,RL,LP,RP)".format(smooth_width),
        fontsize=11,
    )
    pltaxs[0, 1].set_title(
        "Flare-Rate Spectra (colored by freq. FFT'ed)".format(freqs[ifreq]), fontsize=11
    )
    plt.savefig("spectra_flare-rate_by_clust.png")
    plt.close()




## === cell 16
print("\nMedian of the Means(below): {:.4f}".format(np.median(all_means)))
plt.figure(figsize=(6, 2))
plt.hist(np.clip(all_means, -2.0, 2.0), bins=100)
plt.xlim(-2.05, 2.05)
plt.xlabel("Mean")
plt.title("Histogram of the Means of the log(Spectra/Ref-spect)")
plt.close()
print("\nMedian of the Medians(below): {:.4f}".format(np.median(all_medians)))
plt.figure(figsize=(6, 2))
plt.hist(np.clip(all_medians, -2.0, 2.0), bins=100)
plt.xlim(-2.05, 2.05)
plt.xlabel("log10( Median )")
plt.title("Histogram of the Medians of the log(Spectra/Ref-spect)")
plt.close()
mmclrs = []
for ilab in all_clusts:
    mmclrs.append(kmclrs[ilab])
plt.figure(figsize=(3, 3))
plt.scatter(all_medians, all_means, s=2, c=mmclrs, alpha=0.3)
plt.xlim(-1.0, 1)
plt.ylim(-1.0, 1)
plt.xlabel("Median")
plt.ylabel("Mean")
plt.close()




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/420616233.py in <cell line: 0>()
----> 1 print("\nMedian of the Means(below): {:.4f}".format(np.median(all_means)))
      2 plt.figure(figsize=(6, 2))
      3 plt.hist(np.clip(all_means, -2.0, 2.0), bins=100)
      4 plt.xlim(-2.05, 2.05)
      5 plt.xlabel("Mean")

NameError: name 'all_means' is not defined

## === cell 17
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




## === cell 18
if USE_PREPROC:
    try:
        Xy_train_meta = pd.read_csv(above_dir_preproc + "Xy_train_meta_v62.csv")
        Xy_train_feats = pd.read_csv(above_dir_preproc + "Xy_train_feats_v62.csv")
        Xy_train_meta["clust_id"] = kmeans.predict(np.array(Xy_train_meta[HBA_probs]))
        Xy_train_feats["clust_id"] = Xy_train_meta["clust_id"]
    except FileNotFoundError:
        USE_PREPROC = False  # fallback
if not USE_PREPROC:
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




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/joblib/_utils.py", line 72, in __call__
    return self.func(**kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in __call__
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in <listcomp>
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
            ^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/2144142938.py", line 29, in _process_single_row
    ) / (4.0 * spect_trend4)
               ^^^^^^^^^^^^
NameError: name 'spect_trend4' is not defined
"""

The above exception was the direct cause of the following exception:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2282260301.py in <cell line: 0>()
     11     Xy_train_meta = Xy_train_meta.reset_index().drop(columns=["index"])
     12     print("Number of samples used for training =", len(Xy_train_meta))
---> 13     Xy_train_feats = assemble_features(
     14         Xy_train_meta, traintest="train", smooth_width=SMOOTH_WIDTH
     15     )

/tmp/ipykernel_11/2144142938.py in assemble_features(meta_frame, traintest, smooth_width)
    152     # Use threading backend to avoid copying large cache to child processes
    153     n_jobs = max(1, os.cpu_count() - 1)
--> 154     results = Parallel(n_jobs=n_jobs, backend="threading")(
    155         delayed(_process_single_row)(
    156             meta_frame.loc[irow],

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in __call__(self, iterable)
   2070         next(output)
   2071 
-> 2072         return output if self.return_generator else list(output)
   2073 
   2074     def __repr__(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _get_outputs(self, iterator, pre_dispatch)
   1680 
   1681             with self._backend.retrieval_context():
-> 1682                 yield from self._retrieve()
   1683 
   1684         except GeneratorExit:

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _retrieve(self)
   1782             # worker traceback.
   1783             if self._aborting:
-> 1784                 self._raise_error_fast()
   1785                 break
   1786 

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _raise_error_fast(self)
   1857         # called directly or if the generator is gc'ed.
   1858         if error_job is not None:
-> 1859             error_job.get_result(self.timeout)
   1860 
   1861     def _warn_exit_early(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in get_result(self, timeout)
    756             # callback thread, and is stored internally. It's just waiting to
    757             # be returned.
--> 758             return self._return_or_raise()
    759 
    760         # For other backends, the main thread needs to run the retrieval step.

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _return_or_raise(self)
    771         try:
    772             if self.status == TASK_ERROR:
--> 773                 raise self._result
    774             return self._result
    775         finally:

NameError: name 'spect_trend4' is not defined

## === cell 19
if USE_PREPROC:
    try:
        Xy_valid_meta = pd.read_csv(above_dir_preproc + "Xy_valid_meta_v62.csv")
        Xy_valid_feats = pd.read_csv(above_dir_preproc + "Xy_valid_feats_v62.csv")
        Xy_valid_meta["clust_id"] = kmeans.predict(np.array(Xy_valid_meta[HBA_probs]))
        Xy_valid_feats["clust_id"] = Xy_valid_meta["clust_id"]
    except FileNotFoundError:
        USE_PREPROC = False  # fallback
if not USE_PREPROC:
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




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/joblib/_utils.py", line 72, in __call__
    return self.func(**kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in __call__
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in <listcomp>
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
            ^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/2144142938.py", line 29, in _process_single_row
    ) / (4.0 * spect_trend4)
               ^^^^^^^^^^^^
NameError: name 'spect_trend4' is not defined
"""

The above exception was the direct cause of the following exception:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3732320911.py in <cell line: 0>()
     11     Xy_valid_meta = Xy_valid_meta.reset_index().drop(columns=["index"])
     12     print("Number of samples used for Validation =", len(Xy_valid_meta))
---> 13     Xy_valid_feats = assemble_features(
     14         Xy_valid_meta, traintest="validation", smooth_width=SMOOTH_WIDTH
     15     )

/tmp/ipykernel_11/2144142938.py in assemble_features(meta_frame, traintest, smooth_width)
    152     # Use threading backend to avoid copying large cache to child processes
    153     n_jobs = max(1, os.cpu_count() - 1)
--> 154     results = Parallel(n_jobs=n_jobs, backend="threading")(
    155         delayed(_process_single_row)(
    156             meta_frame.loc[irow],

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in __call__(self, iterable)
   2070         next(output)
   2071 
-> 2072         return output if self.return_generator else list(output)
   2073 
   2074     def __repr__(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _get_outputs(self, iterator, pre_dispatch)
   1680 
   1681             with self._backend.retrieval_context():
-> 1682                 yield from self._retrieve()
   1683 
   1684         except GeneratorExit:

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _retrieve(self)
   1782             # worker traceback.
   1783             if self._aborting:
-> 1784                 self._raise_error_fast()
   1785                 break
   1786 

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _raise_error_fast(self)
   1857         # called directly or if the generator is gc'ed.
   1858         if error_job is not None:
-> 1859             error_job.get_result(self.timeout)
   1860 
   1861     def _warn_exit_early(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in get_result(self, timeout)
    756             # callback thread, and is stored internally. It's just waiting to
    757             # be returned.
--> 758             return self._return_or_raise()
    759 
    760         # For other backends, the main thread needs to run the retrieval step.

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _return_or_raise(self)
    771         try:
    772             if self.status == TASK_ERROR:
--> 773                 raise self._result
    774             return self._result
    775         finally:

NameError: name 'spect_trend4' is not defined

## === cell 20
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
    ).fit(Xlr1, y)

    plt.figure(figsize=(8, 4))
    plt.plot(lrmodel1.coef_.T, "-", alpha=0.5)
    plt.plot(lrmodel1.coef_.T, ".", alpha=1.0)
    plt.title("LR1 coefficients, C=" + str(LR1_C) + " (colored by cluster)")
    plt.close()

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
    ).fit(Xlr2, y)

    plt.figure(figsize=(8, 4))
    plt.plot(lrmodel2.coef_.T, "-", alpha=0.5)
    plt.plot(lrmodel2.coef_.T, ".", alpha=1.0)
    plt.title("LR2 coefficients, C=" + str(LR2_C) + " (colored by cluster)")
    plt.close()

    print("\nLR2 model score for X,y = {:.1f}%\n".format(100 * lrmodel2.score(Xlr2, y)))




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3899858804.py in <cell line: 0>()
----> 1 X = Xy_train_feats.drop(columns=["clust_id"])
      2 y = Xy_train_feats.clust_id
      3 
      4 Xlr = X.drop(columns=X.columns[-10:])
      5 

NameError: name 'Xy_train_feats' is not defined

## === cell 21
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
    plt.close()




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2233590270.py in <cell line: 0>()
----> 1 Xy_train_wLRfeats = Xy_train_feats.copy()
      2 X = Xy_train_feats.drop(columns=["clust_id"])
      3 Xlr = X.drop(columns=X.columns[-10:])
      4 if USE_LR1:
      5     Xlr1 = Xlr.iloc[:, 0 : 83 + 1]

NameError: name 'Xy_train_feats' is not defined

## === cell 22
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
    random_state=None,
).fit(X, y)
ave_oob.append(rfmodel.oob_score_)

sort_inds = rfmodel.feature_importances_.argsort()
plt.figure(figsize=(4, 7))
plt.barh(rfmodel.feature_names_in_[sort_inds], rfmodel.feature_importances_[sort_inds])
plt.ylim(len(sort_inds) - 40, len(sort_inds) + 0.2)
plt.title("Feature Importances (top 40)")
plt.close()

print("\nRF model ave OOB score = {:.1f}%".format(100 * np.mean(ave_oob)))
print("\nRF model score for X,y = {:.1f}%\n".format(100 * rfmodel.score(X, y)))




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3084875143.py in <cell line: 0>()
----> 1 X = Xy_train_wLRfeats.drop(columns=["clust_id"])
      2 y = Xy_train_wLRfeats.clust_id
      3 
      4 ave_oob = []
      5 rfmodel = RandomForestClassifier(

NameError: name 'Xy_train_wLRfeats' is not defined

## === cell 23
Xy_train_meta["pred_id"] = rfmodel.predict(Xy_train_wLRfeats.drop(columns=["clust_id"]))

maxprobs_train = np.max(rfmodel.predict_proba(X), axis=1)
plt.figure(figsize=(7, 2.5))
plt.hist(maxprobs_train, bins=50)
plt.xlim(0.0, 1.0)
plt.title("Histogram of max(proba) for model-training samples")
plt.close()

solution = Xy_train_meta[["eeg_id"] + HBA_votes]
for col_pre in HBA_names:
    solution.loc[:, col_pre + "_vote"] = Xy_train_meta[col_pre + "_prob"]

submission = solution.copy()
pred_ids = Xy_train_meta["pred_id"]  # putting clust_id gives the ideal value


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


best_fracs, best_centers = find_best_tamed_kl()

for iprob in range(HBA_number):
    this_col_probs = best_centers[:, iprob]
    submission[HBA_votes[iprob]] = this_col_probs[pred_ids]
this_kl = kld_score(solution, submission)
print("Tamed fractions:\n", best_fracs, "\nand centers:\n", best_centers)
print("\nKL from tamed centers: {:.4f}".format(this_kl))




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3540653284.py in <cell line: 0>()
----> 1 Xy_train_meta["pred_id"] = rfmodel.predict(Xy_train_wLRfeats.drop(columns=["clust_id"]))
      2 
      3 maxprobs_train = np.max(rfmodel.predict_proba(X), axis=1)
      4 plt.figure(figsize=(7, 2.5))
      5 plt.hist(maxprobs_train, bins=50)

NameError: name 'rfmodel' is not defined

## === cell 24
Xy_valid_meta["pred_id"] = rfmodel.predict(Xy_valid_wLRfeats.drop(columns=["clust_id"]))

maxprobs_valid = np.max(
    rfmodel.predict_proba(Xy_valid_wLRfeats.drop(columns=["clust_id"])), axis=1
)
plt.figure(figsize=(7, 2.5))
plt.hist(maxprobs_valid, bins=50)
plt.xlim(0.0, 1.0)
plt.title("Histogram of max(proba) for validation samples")
plt.close()

solution = Xy_valid_meta[["eeg_id"] + HBA_votes]
for col_pre in HBA_names:
    solution.loc[:, col_pre + "_vote"] = Xy_valid_meta[col_pre + "_prob"]

submission = solution.copy()
pred_ids = Xy_valid_meta["pred_id"]

best_fracs, best_centers = find_best_tamed_kl()

for iprob in range(HBA_number):
    this_col_probs = best_centers[:, iprob]
    submission[HBA_votes[iprob]] = this_col_probs[pred_ids]
this_kl = kld_score(solution, submission)
print("Tamed fractions:\n", best_fracs, "\nand centers:\n", best_centers)
print("\nKL from tamed centers: {:.4f}".format(this_kl))




## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2504730475.py in <cell line: 0>()
----> 1 Xy_valid_meta["pred_id"] = rfmodel.predict(Xy_valid_wLRfeats.drop(columns=["clust_id"]))
      2 
      3 maxprobs_valid = np.max(
      4     rfmodel.predict_proba(Xy_valid_wLRfeats.drop(columns=["clust_id"])), axis=1
      5 )

NameError: name 'rfmodel' is not defined

## === cell 25
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

pred_ids = rfmodel.predict(Xy_test_wLRfeats)

test_submit = test_meta[["eeg_id"]].copy()
for new_col in HBA_votes:
    test_submit[new_col] = 1 / HBA_number
for iprob in range(HBA_number):
    this_col_probs = best_centers[:, iprob]
    test_submit[HBA_votes[iprob]] = this_col_probs[pred_ids]

print(test_submit.head())

test_submit.to_csv(
    "submission.csv", header=True, index=False, na_rep="", float_format="%.6f"
)

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/joblib/_utils.py", line 72, in __call__
    return self.func(**kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in __call__
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in <listcomp>
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
            ^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/2144142938.py", line 29, in _process_single_row
    ) / (4.0 * spect_trend4)
               ^^^^^^^^^^^^
NameError: name 'spect_trend4' is not defined
"""

The above exception was the direct cause of the following exception:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1730834523.py in <cell line: 0>()
----> 1 Xy_test_feats = assemble_features(
      2     test_meta, traintest="test", smooth_width=SMOOTH_WIDTH
      3 )
      4 
      5 Xy_test_wLRfeats = Xy_test_feats.copy()

/tmp/ipykernel_11/2144142938.py in assemble_features(meta_frame, traintest, smooth_width)
    152     # Use threading backend to avoid copying large cache to child processes
    153     n_jobs = max(1, os.cpu_count() - 1)
--> 154     results = Parallel(n_jobs=n_jobs, backend="threading")(
    155         delayed(_process_single_row)(
    156             meta_frame.loc[irow],

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in __call__(self, iterable)
   2070         next(output)
   2071 
-> 2072         return output if self.return_generator else list(output)
   2073 
   2074     def __repr__(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _get_outputs(self, iterator, pre_dispatch)
   1680 
   1681             with self._backend.retrieval_context():
-> 1682                 yield from self._retrieve()
   1683 
   1684         except GeneratorExit:

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _retrieve(self)
   1782             # worker traceback.
   1783             if self._aborting:
-> 1784                 self._raise_error_fast()
   1785                 break
   1786 

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _raise_error_fast(self)
   1857         # called directly or if the generator is gc'ed.
   1858         if error_job is not None:
-> 1859             error_job.get_result(self.timeout)
   1860 
   1861     def _warn_exit_early(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in get_result(self, timeout)
    756             # callback thread, and is stored internally. It's just waiting to
    757             # be returned.
--> 758             return self._return_or_raise()
    759 
    760         # For other backends, the main thread needs to run the retrieval step.

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _return_or_raise(self)
    771         try:
    772             if self.status == TASK_ERROR:
--> 773                 raise self._result
    774             return self._result
    775         finally:

NameError: name 'spect_trend4' is not defined
