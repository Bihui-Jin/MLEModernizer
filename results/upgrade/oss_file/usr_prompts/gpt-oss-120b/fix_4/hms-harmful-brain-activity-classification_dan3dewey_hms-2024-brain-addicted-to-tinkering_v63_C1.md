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

1.022695284356648

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
NUM_CLUSTS = 9  # 6 to 10
TRAIN_DOWNSEL = 1  # large values for code test; set to 1 to output all.
VALID_DOWNSEL = 1  #  "
SMOOTH_WIDTH = 5  # Odd>1: 3,5,7,9,...
USE_PREPROC = True  # Load pre‑computed meta/feature CSVs when they exist
USE_LR1 = False
LR1_C = 1.0  # smaller --> fewer non‑zero coeff.s
USE_LR2 = False
LR2_C = 1.0
LR_BLUR = 0.10

above_dir = "/kaggle/input/hms-harmful-brain-activity-classification/"
above_dir_preproc = "/kaggle/input/hms-2024-brain-data/"



## === cell 1
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import pyarrow.parquet as pq
import pyarrow.dataset as pads



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

    vote_cols = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    train_meta["total_vote"] = train_meta[vote_cols].sum(axis=1)
    train_meta["max_vote"] = train_meta[vote_cols].max(axis=1)

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

    prob_array = train_meta[HBA_probs].clip(lower=1e-8)
    train_meta["entropy"] = -(prob_array * np.log(prob_array)).sum(axis=1)

    return train_meta, test_meta




## === cell 5
def prob_prob_scatter(name1, name2, probs2plot, clust_ids, iclust_order=[0]):
    """
    Make a prob1 vs prob2 scatter plot.
    - name1, name2 are 2 of the 6 HBA expert_consensus labels.
    - probs2plot is 6‑column dataframe, e.g., train_meta[HBA_probs]
    - clust_ids is an array of cluster id integers, e.g., train_meta["clust_id"]
    Include an x at the cluster centers in the chosen axes.
    Use sqrt scaling to emphasize lower values.
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
    if "clust_centers" in globals():
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
def assemble_features(meta_frame, traintest="train", smooth_width=5):
    """
    Create a dataframe of spectrogram features from the meta_frame rows.
    Will include clust_id (i.e, the y) if it is in the input meta_frame.
    Assumes these are available: above_dir, the4chains
    """
    if traintest == "validation":
        plt.figure(figsize=(9, 7))

    freqs = np.arange(100) * 0.19525 + 0.59
    spect_trend = 150.0 / (1.0**2.3 + freqs ** (2.3))
    baseinds = np.insert(
        np.arange(int((smooth_width - 1) / 2), 100, smooth_width), 0, 0
    )  # bin 0 put at front
    freqs4 = np.array(4 * list(freqs))
    spect_trend4 = np.array(4 * list(spect_trend))

    n_fft_bins = 256
    freqbins = [7, 18, 60, 84]
    fftfeatbins = [4, 9, 16, 25, 36, 49]

    feats_list = []  # accumulate per‑row feature DataFrames
    last_spectro_id_str = None
    spectro_cache = {}  # simple cache in case files repeat non‑consecutively
    print_every_nth = max([100, 100 * int(0.5 + len(meta_frame) / (100.0 * 15))])
    plot_every_nth = max([1, int(0.5 + len(meta_frame) / 100)])

    for row_counter, irow in enumerate(meta_frame.index, start=1):
        this_row = meta_frame.loc[irow]
        spectro_id_str = str(int(this_row.spectrogram_id))

        if spectro_id_str != last_spectro_id_str:
            if spectro_id_str in spectro_cache:
                this_spectro_np = spectro_cache[spectro_id_str]
            else:
                if traintest != "test":
                    spectro_file = (
                        above_dir + "train_spectrograms/" + spectro_id_str + ".parquet"
                    )
                else:
                    spectro_file = (
                        above_dir + "test_spectrograms/" + spectro_id_str + ".parquet"
                    )
                this_spectro_df = pd.read_parquet(spectro_file)
                this_spectro_np = this_spectro_df.values  # shape (time, freq_bins+1)
                spectro_cache[spectro_id_str] = this_spectro_np
            last_spectro_id_str = spectro_id_str

        loc_offset = int(this_row.spectrogram_label_offset_seconds / 2)
        fftlocbeg = loc_offset + 149 - (n_fft_bins // 2 - 1)
        fftlocend = loc_offset + 150 + (n_fft_bins // 2 - 1)

        middle_idxs = np.arange(loc_offset + 148, loc_offset + 152)  # 148‑151 inclusive
        middle_vals = this_spectro_np[middle_idxs, 1:]  # drop time column
        middle8s = middle_vals.sum(axis=0) / (4.0 * spect_trend4)
        middle8s = np.clip(middle8s, 0.001, 1000.0)
        middle8s = np.nan_to_num(middle8s, nan=0.001, posinf=1000.0, neginf=0.001)
        spect_mean = middle8s.mean()
        spect_median = np.median(middle8s)

        the4means = []
        the4medians = []
        for ispec in range(4):
            ibeg = ispec * 100
            iend = ibeg + 100
            seg = middle8s[ibeg:iend]
            the4means.append(seg.mean())
            the4medians.append(np.median(seg))

        middle8spre = np.log10(middle8s / spect_mean)
        middle_series = pd.Series(middle8s)
        middle_smoothed = (
            middle_series.rolling(smooth_width, min_periods=smooth_width, center=True)
            .mean()
            .values
        )
        for ioff in range(0, 399, 100):
            middle_smoothed[ioff : ioff + smooth_width // 2] = middle8spre[
                ioff : ioff + smooth_width // 2
            ]
        middle8sds = middle_smoothed[
            np.concatenate((baseinds, 100 + baseinds, 200 + baseinds, 300 + baseinds))
        ]
        middle_feats = pd.DataFrame(
            [middle8sds], columns=[f"mid_{i}" for i in range(len(middle8sds))]
        )

        flarecols = []
        flarevals = []
        for ispec in range(4):
            for freqbin in freqbins:
                sum_spect_trend = spect_trend[freqbin - 4 : freqbin + 5 : 2].sum()
                ifreqoff = freqbin + 1 + ispec * 100
                cols = [
                    ifreqoff - 4,
                    ifreqoff - 2,
                    ifreqoff,
                    ifreqoff + 2,
                    ifreqoff + 4,
                ]
                amplvstime = (
                    this_spectro_np[fftlocbeg : fftlocend + 1][:, cols].sum(axis=1)
                    / sum_spect_trend
                )
                amplvstime = np.clip(amplvstime, 0.001, 1000.0)
                amplvstime = np.nan_to_num(
                    amplvstime, nan=0.001, posinf=1000.0, neginf=0.001
                )
                centre = amplvstime[fftlocbeg + 127] + amplvstime[fftlocbeg + 128]
                amplvstime = 2.0 * amplvstime / centre
                apod_wind = np.blackman(n_fft_bins)
                amplvstime = apod_wind * np.clip(amplvstime, 0.0, 10.0)
                amplvstime = (
                    pd.Series(amplvstime).rolling(2, min_periods=1).mean().values
                )
                amplvstime = (
                    pd.Series(amplvstime).rolling(2, min_periods=1).mean().values
                )
                amplfft = np.log10(
                    1 + np.abs(np.fft.fft(amplvstime))[: n_fft_bins // 2]
                )

                for fftfeatbin in fftfeatbins:
                    flarecols.append(
                        f"fft-{the4chains[ispec]}{freqs[freqbin]:.1f}-{fftfeatbin}"
                    )
                    flarevals.append(amplfft[fftfeatbin])

                if (row_counter % plot_every_nth == 0) and traintest == "validation":
                    plt.plot(
                        np.sqrt(np.arange(n_fft_bins // 2)),
                        amplfft,
                        c=kmclrs[this_row["clust_id"]],
                        lw=2,
                        alpha=0.01,
                    )

        flare_feats = pd.DataFrame([flarevals], columns=flarecols)
        these_feats = pd.concat([middle_feats, flare_feats], axis=1)

        spect_mean = np.log10(spect_mean)
        spect_median = np.log10(spect_median)
        the4means = np.log10(the4means)
        the4medians = np.log10(the4medians)

        these_feats["Mean"] = spect_mean
        these_feats["Median"] = spect_median
        for ispec in range(4):
            these_feats[f"{the4chains[ispec]}mean"] = the4means[ispec]
            these_feats[f"{the4chains[ispec]}median"] = the4medians[ispec]

        if "clust_id" in meta_frame.columns:
            these_feats["clust_id"] = this_row.clust_id

        feats_list.append(these_feats)

        if row_counter % print_every_nth == 0:
            print(f"... {row_counter} done...")

    feats_frame = pd.concat(feats_list, ignore_index=True)

    if traintest == "validation":
        plt.ylim(-0.2, 2.0)
        plt.xlim(0.0, 11.5)
        plt.xlabel("sqrt[  FFT bin number ]")
        plt.ylabel("log10[1+ FFT amplitude ]")
        plt.title("Flare-Rate Spectra (colored by cluster)")

    return feats_frame.reset_index(drop=True)




## === cell 7
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
    tamed_fracs = np.zeros(NUM_CLUSTS)
    tamed_centers = clust_centers.copy()
    for iclust in range(NUM_CLUSTS):
        tamed_centers[iclust, :] = mean_all_probs
    best_fracs = tamed_fracs.copy()
    best_centers = tamed_centers.copy()
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
                submission[HBA_votes[iprob]] = tamed_centers[:, iprob][pred_ids]
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
train_meta, test_meta = read_hms_meta()



## === cell 9
plt.figure(figsize=(5, 2))
plt.hist(train_meta["spectrogram_sub_id"], bins=55, log=True)
plt.title("Histogram of spectrogram_sub_id")
plt.show()

plt.figure(figsize=(5, 2))
plt.hist(train_meta["eeg_sub_id"], bins=55, log=True)
plt.title("Histogram of eeg_sub_id")
plt.show()



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
plt.show()

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

if False:
    kmclrs = prob_prob_scatter("Seizure", "GPD", all_probs, all_ids, iclust_of_order)
    kmclrs = prob_prob_scatter("LPD", "GRDA", all_probs, all_ids, iclust_of_order)
    kmclrs = prob_prob_scatter("LRDA", "Other", all_probs, all_ids, iclust_of_order)
else:
    kmclrs = ["black"] * NUM_CLUSTS

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
    plt.show()



## === cell 13
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
    np.round(kld_score(solution_train, submission_train), 4),
)



## === cell 14
train_rows_bool = (
    (train_meta.eeg_sub_id < 33 + 1) & (train_meta.eeg_sub_id % 5 == 3)
) | (  # id=3,8,13,18,23,28,33
    train_meta.eeg_sub_id == 0
) & (
    (train_meta.eeg_id % 23) % 8 > 1
)

valid_rows_bool = (
    (train_meta.eeg_sub_id < 44 + 1) & (train_meta.eeg_sub_id % 19 == 6)
) | (  # id=6,25,44
    train_meta.eeg_sub_id == 0
) & (
    (train_meta.eeg_id % 23) % 8 < 2
)



## === cell 15
train_meta_path = "Xy_train_meta.csv"
train_feats_path = "Xy_train_feats.csv"
if USE_PREPROC and os.path.exists(train_meta_path) and os.path.exists(train_feats_path):
    Xy_train_meta = pd.read_csv(train_meta_path)
    Xy_train_feats = pd.read_csv(train_feats_path)
    Xy_train_meta["clust_id"] = kmeans.predict(np.array(Xy_train_meta[HBA_probs]))
    Xy_train_feats["clust_id"] = Xy_train_meta["clust_id"]
else:
    Xy_train_meta = (train_meta[train_rows_bool])[::TRAIN_DOWNSEL].copy()
    Xy_train_meta = Xy_train_meta.reset_index(drop=True)
    print("Number of samples used for training =", len(Xy_train_meta))
    Xy_train_feats = assemble_features(
        Xy_train_meta, traintest="train", smooth_width=SMOOTH_WIDTH
    )
    Xy_train_meta.to_csv(train_meta_path, header=True, index=False, float_format="%.6f")
    Xy_train_feats.to_csv(
        train_feats_path, header=True, index=False, float_format="%.6f"
    )



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/2349128068.py in <cell line: 0>()
     10     Xy_train_meta = Xy_train_meta.reset_index(drop=True)
     11     print("Number of samples used for training =", len(Xy_train_meta))
---> 12     Xy_train_feats = assemble_features(
     13         Xy_train_meta, traintest="train", smooth_width=SMOOTH_WIDTH
     14     )

/tmp/ipykernel_11/1251624897.py in assemble_features(meta_frame, traintest, smooth_width)
    117                 )
    118                 # normalise by centre of FFT window
--> 119                 centre = amplvstime[fftlocbeg + 127] + amplvstime[fftlocbeg + 128]
    120                 amplvstime = 2.0 * amplvstime / centre
    121                 # apodisation and smoothing

IndexError: index 405 is out of bounds for axis 0 with size 256

## === cell 16
valid_meta_path = "Xy_valid_meta.csv"
valid_feats_path = "Xy_valid_feats.csv"
if USE_PREPROC and os.path.exists(valid_meta_path) and os.path.exists(valid_feats_path):
    Xy_valid_meta = pd.read_csv(valid_meta_path)
    Xy_valid_feats = pd.read_csv(valid_feats_path)
    Xy_valid_meta["clust_id"] = kmeans.predict(np.array(Xy_valid_meta[HBA_probs]))
    Xy_valid_feats["clust_id"] = Xy_valid_meta["clust_id"]
else:
    Xy_valid_meta = (train_meta[valid_rows_bool])[::VALID_DOWNSEL].copy()
    Xy_valid_meta = Xy_valid_meta.reset_index(drop=True)
    print("Number of samples used for Validation =", len(Xy_valid_meta))
    Xy_valid_feats = assemble_features(
        Xy_valid_meta, traintest="validation", smooth_width=SMOOTH_WIDTH
    )
    Xy_valid_meta.to_csv(valid_meta_path, header=True, index=False, float_format="%.6f")
    Xy_valid_feats.to_csv(
        valid_feats_path, header=True, index=False, float_format="%.6f"
    )



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/3910707663.py in <cell line: 0>()
     10     Xy_valid_meta = Xy_valid_meta.reset_index(drop=True)
     11     print("Number of samples used for Validation =", len(Xy_valid_meta))
---> 12     Xy_valid_feats = assemble_features(
     13         Xy_valid_meta, traintest="validation", smooth_width=SMOOTH_WIDTH
     14     )

/tmp/ipykernel_11/1251624897.py in assemble_features(meta_frame, traintest, smooth_width)
    117                 )
    118                 # normalise by centre of FFT window
--> 119                 centre = amplvstime[fftlocbeg + 127] + amplvstime[fftlocbeg + 128]
    120                 amplvstime = 2.0 * amplvstime / centre
    121                 # apodisation and smoothing

IndexError: index 284 is out of bounds for axis 0 with size 256

## === cell 17
X = Xy_train_feats.drop(columns=["clust_id"])
y = Xy_train_feats.clust_id



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1281424111.py in <cell line: 0>()
----> 1 X = Xy_train_feats.drop(columns=["clust_id"])
      2 y = Xy_train_feats.clust_id
      3 

NameError: name 'Xy_train_feats' is not defined

## === cell 18
Xy_train_wLRfeats = Xy_train_feats.copy()
Xy_valid_wLRfeats = Xy_valid_feats.copy()



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2262459894.py in <cell line: 0>()
----> 1 Xy_train_wLRfeats = Xy_train_feats.copy()
      2 Xy_valid_wLRfeats = Xy_valid_feats.copy()
      3 

NameError: name 'Xy_train_feats' is not defined

## === cell 19
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

ave_oob = [rfmodel.oob_score_]
print("\nRF model ave OOB score = {:.1f}%".format(100 * np.mean(ave_oob)))
print("\nRF model score for X,y = {:.1f}%\n".format(100 * rfmodel.score(X, y)))



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3394410773.py in <cell line: 0>()
      9     verbose=0,
     10     random_state=None,
---> 11 ).fit(X, y)
     12 
     13 ave_oob = [rfmodel.oob_score_]

NameError: name 'X' is not defined

## === cell 20
Xy_train_meta["pred_id"] = rfmodel.predict(Xy_train_wLRfeats.drop(columns=["clust_id"]))

maxprobs_train = np.max(rfmodel.predict_proba(X), axis=1)
plt.figure(figsize=(7, 2.5))
plt.hist(maxprobs_train, bins=50)
plt.xlim(0.0, 1.0)
plt.title("Histogram of max(proba) for model‑training samples")
plt.show()

solution = Xy_train_meta[["eeg_id"] + HBA_votes]
for col_pre in HBA_names:
    solution.loc[:, col_pre + "_vote"] = Xy_train_meta[col_pre + "_prob"]

submission = solution.copy()
pred_ids = Xy_train_meta["pred_id"]

best_fracs, best_centers = find_best_tamed_kl()

for iprob in range(HBA_number):
    this_col_probs = best_centers[:, iprob]
    submission[HBA_votes[iprob]] = this_col_probs[pred_ids]
this_kl = kld_score(solution, submission)
print("Tamed fractions:\n", best_fracs, "\nand centers:\n", best_centers)
print("\nKL from tamed centers: {:.4f}".format(this_kl))



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2634943031.py in <cell line: 0>()
----> 1 Xy_train_meta["pred_id"] = rfmodel.predict(Xy_train_wLRfeats.drop(columns=["clust_id"]))
      2 
      3 maxprobs_train = np.max(rfmodel.predict_proba(X), axis=1)
      4 plt.figure(figsize=(7, 2.5))
      5 plt.hist(maxprobs_train, bins=50)

NameError: name 'rfmodel' is not defined

## === cell 21
Xy_valid_meta["pred_id"] = rfmodel.predict(Xy_valid_wLRfeats.drop(columns=["clust_id"]))

maxprobs_valid = np.max(
    rfmodel.predict_proba(Xy_valid_wLRfeats.drop(columns=["clust_id"])), axis=1
)
plt.figure(figsize=(7, 2.5))
plt.hist(maxprobs_valid, bins=50)
plt.xlim(0.0, 1.0)
plt.title("Histogram of max(proba) for validation samples")
plt.show()

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



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1534533251.py in <cell line: 0>()
----> 1 Xy_valid_meta["pred_id"] = rfmodel.predict(Xy_valid_wLRfeats.drop(columns=["clust_id"]))
      2 
      3 maxprobs_valid = np.max(
      4     rfmodel.predict_proba(Xy_valid_wLRfeats.drop(columns=["clust_id"])), axis=1
      5 )

NameError: name 'rfmodel' is not defined

## === cell 22
Xy_test_feats = assemble_features(
    test_meta, traintest="test", smooth_width=SMOOTH_WIDTH
)

Xy_test_wLRfeats = Xy_test_feats.copy()

pred_ids = rfmodel.predict(Xy_test_wLRfeats.drop(columns=["clust_id"]))

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

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2870619174.py in <cell line: 0>()
      5 Xy_test_wLRfeats = Xy_test_feats.copy()
      6 
----> 7 pred_ids = rfmodel.predict(Xy_test_wLRfeats.drop(columns=["clust_id"]))
      8 
      9 test_submit = test_meta[["eeg_id"]].copy()

NameError: name 'rfmodel' is not defined
