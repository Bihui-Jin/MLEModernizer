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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import pyarrow
import pyarrow.parquet as pq
import pyarrow.dataset as pads

np.random.seed(0)



## === cell 1
above_dir = "../input/hms-harmful-brain-activity-classification/"
if not os.path.exists(above_dir):
    above_dir = "/kaggle/input/hms-harmful-brain-activity-classification/"

above_dir_preproc = "../input/hms-2024-brain-data/"
if not os.path.exists(above_dir_preproc):
    above_dir_preproc = "/kaggle/input/hms-2024-brain-data/"

print("above_dir:", above_dir)
print(
    "above_dir_preproc:",
    above_dir_preproc,
    "(exists:",
    os.path.exists(above_dir_preproc),
    ")",
)



## === cell 2
NUM_CLUSTS = 9  # 6 to 10

TRAIN_DOWNSEL = 1  # large values for code test; set to 1 to output all.
VALID_DOWNSEL = 1  #  "
SMOOTH_WIDTH = 5  # Odd>1: 3,5,7,9,...
USE_PREPROC = True  # Read-in saved meta and features frames

USE_LR1 = True
LR1_C = 1.0  # smaller --> fewer non-zero coeff.s
USE_LR2 = True
LR2_C = 1.0
LR_BLUR = 0.10



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
def kld_score(solution, submission):
    """
    Calculate the average KL divergence score.
    Fix: ignore id columns and avoid log(0) by clipping + renormalizing submission rows.
    """
    sol = solution.copy()
    sub = submission.copy()

    prob_cols = [c for c in sol.columns if c in HBA_votes]
    sol = sol[prob_cols].astype(float)
    sub = sub[prob_cols].astype(float)

    sub = sub.clip(1e-15, 1.0)
    sub = sub.div(sub.sum(axis=1), axis=0)

    sol = sol.clip(1e-15, 1.0)
    sol = sol.div(sol.sum(axis=1), axis=0)

    kls = (sol * (np.log(sol) - np.log(sub))).sum(axis=1)
    return float(np.mean(kls))




## === cell 5
def read_hms_meta():
    """
    Read in the train.csv and test.csv files.
    Add total_vote, _prob columns, and vote entropy to train_meta.
    Add extra cols to test to allow the same processing as train:
        eeg[spectro]_sub_id, eeg[spectro]_label_offset_seconds, label_id
    """
    test_meta = pd.read_csv(os.path.join(above_dir, "test.csv"))
    test_meta_len = len(test_meta)
    print("Test has length", test_meta_len)
    test_meta["eeg_sub_id"] = 0
    test_meta["eeg_label_offset_seconds"] = 0.0
    test_meta["spectrogram_sub_id"] = 0
    test_meta["spectrogram_label_offset_seconds"] = 0.0
    test_meta["label_id"] = test_meta.eeg_id

    train_meta = pd.read_csv(os.path.join(above_dir, "train.csv"))
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

    def calc_entropy(row):
        the_probs = np.clip(row[HBA_probs].values.astype(float), 1.0e-8, 1.0)
        return np.nansum(the_probs * -1 * np.log(the_probs))

    train_meta["entropy"] = train_meta.apply(calc_entropy, axis=1)

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
    kmclrs = hba_clrs.copy()
    if len(iclust_order) > 2:
        for iord, iclust in enumerate(iclust_order):
            kmclrs[iclust] = hba_clrs[iord]

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
    plt.show()
    return kmclrs




## === cell 7
def assemble_features(meta_frame, traintest="train", smooth_width=5):
    """
    Create a dataframe of spectrogram features from the meta_frame rows.
    Will include clust_id (i.e, the y) if it is in the input meta_frame.
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

    feats_frame = []
    last_spectro_id_str = "starting"
    print_every_nth = max([100, 100 * int(0.5 + len(meta_frame.index) / (100.0 * 15))])
    plot_every_nth = max([1, int(0.5 + len(meta_frame.index) / 100)])

    for irow in meta_frame.index:
        this_row = meta_frame.loc[irow]
        spectro_id_str = str(int(this_row.spectrogram_id))
        if spectro_id_str != last_spectro_id_str:
            if traintest != "test":
                spectro_file = os.path.join(
                    above_dir, "train_spectrograms", spectro_id_str + ".parquet"
                )
            else:
                spectro_file = os.path.join(
                    above_dir, "test_spectrograms", spectro_id_str + ".parquet"
                )
            pads_spectro = pads.dataset(spectro_file)
            this_spectro = pads_spectro.to_table().to_pandas()
        last_spectro_id_str = spectro_id_str

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
        the4means, the4medians = [], []
        for ispec in range(4):
            ibeg = ([0, 100, 200, 300])[ispec]
            iend = ibeg + 100
            the4means.append(np.mean(middle8s[ibeg:iend]))
            the4medians.append(np.median(middle8s[ibeg:iend]))

        middle8spre = np.log10(middle8s / spect_mean)
        middle8s_sm = middle8spre.rolling(
            smooth_width, min_periods=smooth_width, center=True
        ).mean()

        for ioff in range(0, 399, 100):
            for ibin in range(int((smooth_width - 1) / 2)):
                middle8s_sm[ibin + ioff] = middle8spre[ibin + ioff]

        select_inds = np.concatenate(
            (baseinds, 100 + baseinds, 200 + baseinds, 300 + baseinds)
        )
        middle8sds = middle8s_sm[select_inds]
        middle_feats = middle8sds.to_frame().T

        flarecols, flarevals = [], []
        for ispec in range(4):
            for freqbin in freqbins:
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
                for _ in range(2):
                    amplvstime = amplvstime.rolling(2, min_periods=1).mean()

                amplfft = np.log10(
                    1 + np.abs(np.fft.fft(amplvstime))[0 : int(n_fft_bins / 2)]
                )
                for fftfeatbin in fftfeatbins:
                    flarecols.append(
                        f"fft-{the4chains[ispec]}{freqs[freqbin]:.1f}-{fftfeatbin}"
                    )
                    flarevals.append(amplfft[fftfeatbin])

                if (
                    len(feats_frame) % plot_every_nth == 0
                ) and traintest == "validation":
                    plt.plot(
                        np.sqrt(range(int(n_fft_bins / 2))), amplfft, lw=2, alpha=0.01
                    )

        flare_feats = pd.DataFrame([flarevals], columns=flarecols)
        these_feats = pd.concat([middle_feats, flare_feats], axis=1)

        these_feats["Mean"] = np.log10(spect_mean)
        these_feats["Median"] = np.log10(spect_median)
        the4means = np.log10(the4means)
        the4medians = np.log10(the4medians)
        for ispec in range(4):
            these_feats[the4chains[ispec] + "mean"] = the4means[ispec]
            these_feats[the4chains[ispec] + "median"] = the4medians[ispec]

        if "clust_id" in meta_frame.columns:
            these_feats["clust_id"] = this_row.clust_id

        if len(feats_frame) == 0:
            feats_frame = these_feats.copy()
        else:
            feats_frame = pd.concat([feats_frame, these_feats], ignore_index=True)

        if len(feats_frame) % print_every_nth == 0:
            print("... {} done...".format(len(feats_frame)))

    return feats_frame.reset_index(drop=True)




## === cell 8
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
                this_col_probs = tamed_centers[:, iprob]
                submission[HBA_votes[iprob]] = this_col_probs[
                    np.asarray(pred_ids, dtype=int)
                ]

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




## === cell 9
train_meta, test_meta = read_hms_meta()



## === cell 10
num_clusts = NUM_CLUSTS
clust_rows_bool = train_meta.eeg_sub_id < 200

prob_vectors = train_meta.loc[clust_rows_bool, HBA_probs]
print("\nUsing {} HBA samples for clustering.".format(len(prob_vectors)))

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
print("cluster centers:\n", clust_centers)

iclust_of_order = []
for icol in range(HBA_number):
    iclust_of_order.append(int(np.argmax(clust_centers[:, icol])))
clust_by_max = np.argsort(-1 * np.max(clust_centers, axis=1))
for iord in range(HBA_number, num_clusts):
    iclust_of_order.append(int(clust_by_max[iord]))

kmnames = HBA_expert_names.copy()
for ihyb in range(1, (num_clusts - HBA_number) + 1):
    kmnames.append("Hybrid-" + str(ihyb))



## === cell 11
train_meta["clust_id"] = kmeans.predict(np.array(train_meta[HBA_probs]))

clust_counts = train_meta.clust_id.value_counts()
iorder_of_clust = num_clusts * [-1]
for iord, iclust in enumerate(iclust_of_order):
    iorder_of_clust[int(iclust)] = int(iord)

DO_PLOTS = False
if DO_PLOTS:
    all_probs = train_meta[HBA_probs]
    all_ids = train_meta["clust_id"]
    kmclrs = prob_prob_scatter("Seizure", "GPD", all_probs, all_ids, iclust_of_order)



## === cell 12
solution_train = train_meta[["eeg_id"] + HBA_votes].copy()
for col_pre in HBA_names:
    solution_train.loc[:, col_pre + "_vote"] = train_meta[col_pre + "_prob"].values

submission_train = solution_train.copy()
clust_ids = train_meta["clust_id"].values.astype(int)
for iprob in range(HBA_number):
    this_col_probs = clust_centers[:, iprob]
    submission_train[HBA_votes[iprob]] = this_col_probs[clust_ids]

print(
    "Score if HBA samples are correctly assigned cluster prob.s:",
    np.round(kld_score(solution_train, submission_train), 4),
)



## === cell 13
train_rows_bool = (
    (train_meta.eeg_sub_id < 33 + 1) & (train_meta.eeg_sub_id % 5 == 3)
) | (train_meta.eeg_sub_id == 0) & ((train_meta.eeg_id % 23) % 8 > 1)
print("Number of Training rows:", int(sum(train_rows_bool)))

valid_rows_bool = (
    (train_meta.eeg_sub_id < 44 + 1) & (train_meta.eeg_sub_id % 19 == 6)
) | (train_meta.eeg_sub_id == 0) & ((train_meta.eeg_id % 23) % 8 < 2)
print("Number of Validation rows:", int(sum(valid_rows_bool)))




## === cell 14
def _try_load_preproc(prefix):
    meta_path = os.path.join(above_dir_preproc, f"Xy_{prefix}_meta_v62.csv")
    feat_path = os.path.join(above_dir_preproc, f"Xy_{prefix}_feats_v62.csv")
    if os.path.exists(meta_path) and os.path.exists(feat_path):
        meta = pd.read_csv(meta_path)
        feats = pd.read_csv(feat_path)
        return meta, feats
    return None, None


if USE_PREPROC:
    Xy_train_meta, Xy_train_feats = _try_load_preproc("train")
    if Xy_train_meta is None or Xy_train_feats is None:
        print("Preproc train files not found; assembling features on-the-fly.")
        USE_PREPROC = False  # switch off for both train/valid consistently

if not USE_PREPROC:
    Xy_train_meta = (
        (train_meta[train_rows_bool])[::TRAIN_DOWNSEL].copy().reset_index(drop=True)
    )
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

if "clust_id" not in Xy_train_meta.columns:
    Xy_train_meta["clust_id"] = kmeans.predict(np.array(Xy_train_meta[HBA_probs]))
Xy_train_feats["clust_id"] = Xy_train_meta["clust_id"].values



## === cell 15
if USE_PREPROC:
    Xy_valid_meta, Xy_valid_feats = _try_load_preproc("valid")
    if Xy_valid_meta is None or Xy_valid_feats is None:
        print("Preproc valid files not found; assembling features on-the-fly.")
        Xy_valid_meta = (
            (train_meta[valid_rows_bool])[::VALID_DOWNSEL].copy().reset_index(drop=True)
        )
        print("Number of samples used for Validation =", len(Xy_valid_meta))
        Xy_valid_feats = assemble_features(
            Xy_valid_meta, traintest="validation", smooth_width=SMOOTH_WIDTH
        )
else:
    Xy_valid_meta = (
        (train_meta[valid_rows_bool])[::VALID_DOWNSEL].copy().reset_index(drop=True)
    )
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

if "clust_id" not in Xy_valid_meta.columns:
    Xy_valid_meta["clust_id"] = kmeans.predict(np.array(Xy_valid_meta[HBA_probs]))
Xy_valid_feats["clust_id"] = Xy_valid_meta["clust_id"].values

print(
    "Train feats shape:",
    Xy_train_feats.shape,
    "Valid feats shape:",
    Xy_valid_feats.shape,
)



## === cell 16
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
    print("\nLR2 model score for X,y = {:.1f}%\n".format(100 * lrmodel2.score(Xlr2, y)))



## === cell 17
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
Xv = Xy_valid_feats.drop(columns=["clust_id"])
Xvlr = Xv.drop(columns=Xv.columns[-10:])

if USE_LR1:
    Xlr1 = Xvlr.iloc[:, 0 : 83 + 1]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    for iadd in range(NUM_CLUSTS):
        Xy_valid_wLRfeats["lrMid" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            np.random.rand(len(lrprobas)) - 0.5
        )

if USE_LR2:
    Xlr2 = Xvlr.iloc[:, 84 : 179 + 1]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    for iadd in range(NUM_CLUSTS):
        Xy_valid_wLRfeats["lrFFT" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            np.random.rand(len(lrprobas)) - 0.5
        )



## === cell 18
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
    random_state=0,
).fit(X, y)

print("\nRF model OOB score = {:.1f}%".format(100 * rfmodel.oob_score_))
print("RF model train score = {:.1f}%\n".format(100 * rfmodel.score(X, y)))



## === cell 19
Xy_train_meta = Xy_train_meta.copy()
Xy_train_meta["pred_id"] = rfmodel.predict(Xy_train_wLRfeats.drop(columns=["clust_id"]))
pred_ids = Xy_train_meta["pred_id"].values.astype(int)

solution = Xy_train_meta[["eeg_id"] + HBA_votes].copy()
for col_pre in HBA_names:
    solution.loc[:, col_pre + "_vote"] = Xy_train_meta[col_pre + "_prob"].values

submission = solution.copy()
best_fracs, best_centers = find_best_tamed_kl()

for iprob in range(HBA_number):
    this_col_probs = best_centers[:, iprob]
    submission[HBA_votes[iprob]] = this_col_probs[pred_ids]

this_kl = kld_score(solution, submission)
print("KL from tamed centers (train): {:.4f}".format(this_kl))



## === cell 20
Xy_valid_meta = Xy_valid_meta.copy()
Xy_valid_meta["pred_id"] = rfmodel.predict(Xy_valid_wLRfeats.drop(columns=["clust_id"]))
pred_ids_v = Xy_valid_meta["pred_id"].values.astype(int)

solution_v = Xy_valid_meta[["eeg_id"] + HBA_votes].copy()
for col_pre in HBA_names:
    solution_v.loc[:, col_pre + "_vote"] = Xy_valid_meta[col_pre + "_prob"].values

submission_v = solution_v.copy()
for iprob in range(HBA_number):
    this_col_probs = best_centers[:, iprob]
    submission_v[HBA_votes[iprob]] = this_col_probs[pred_ids_v]

this_kl_v = kld_score(solution_v, submission_v)
print("KL from tamed centers (valid): {:.4f}".format(this_kl_v))



## === cell 21
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

pred_ids_test = rfmodel.predict(Xy_test_wLRfeats)

test_submit = test_meta[["eeg_id"]].copy()
for new_col in HBA_votes:
    test_submit[new_col] = 1.0 / HBA_number

for iprob in range(HBA_number):
    this_col_probs = best_centers[:, iprob]
    test_submit[HBA_votes[iprob]] = this_col_probs[np.asarray(pred_ids_test, dtype=int)]

probs = test_submit[HBA_votes].astype(float).clip(1e-15, 1.0)
probs = probs.div(probs.sum(axis=1), axis=0)
test_submit[HBA_votes] = probs

test_submit.to_csv(
    "submission.csv", header=True, index=False, na_rep="", float_format="%.6f"
)
print("Wrote submission.csv with shape:", test_submit.shape)
print(test_submit.head())
