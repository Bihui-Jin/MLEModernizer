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

1.0283213069014434

# 6. Current score

1.2565

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39105) has done: 'I fix the immediate runtime blocker by disabling the missing preprocessed CSV dependency and instead generating features directly from the provided spectrogram parquet files. To keep the original modeling logic intact, the clustering, LR feature-augmentation, RF training, and “tamed centers” KL calibration are left as-is; the only logic changes are safe guards for missing files/columns and ensuring the submission probabilities are valid (finite, clipped, and row-normalized to sum to 1). I also add minimal determinism (fixed random seeds) to avoid occasional solver/convergence randomness causing unstable runs. Finally, the script always write `submission.csv` with the exact required columns from `sample_submission.csv` order.'
- What this solution (achieved 1.38825) has done: 'Your current score (1.39105) is worse than the target (1.0283) for a lower-is-better KL metric, so we need a small, low-risk improvement. The biggest issue is that the test-time pipeline mistakenly feeds the RandomForest with an extra non-feature column (it includes `clust_id` during training but not at test), causing a feature mismatch and degraded predictions; we align the feature columns exactly between train/valid/test without changing the model itself. Second, your “tamed centers” are tuned on train/valid but you always use the *train* centers for test; we instead use the validation-tuned centers for test to better match generalization (minimal semantic change, still the same calibration method). Finally, we remove unintended randomness in LR feature augmentation at inference by using a fixed RNG stream (still the same LR_BLUR behavior, just deterministic), improving stability and typically slightly improving KL.'
- What this solution (achieved 1.20785) has done: 'Your current KL (1.38825, lower-is-better) is worse than the target (1.0283), so we need a small but real generalization improvement without changing the model structure. The biggest low-risk gain here is to make the LR feature augmentation consistent: right now the same RNG stream is reused across train and valid, so the random “blur” noise depends on how many train rows you had and can unintentionally worsen validation calibration; we use independent, fixed RNGs for train/valid/test so the noise is deterministic and comparable. Next, we ensure the taming optimization does not accidentally mutate global `submission/pred_ids/solution` from a previous call by making it explicitly operate on passed-in arrays (same algorithm/semantics, just avoids cross-talk), which stabilizes the chosen `best_centers_v` and usually improves KL slightly. Finally, we keep using the validation-tuned centers for test (already good) and keep strict row-normalization for submission validity.'
- What this solution (achieved 1.20695) has done: 'To move your KL (lower-is-better) from 1.20785 toward the 1.0283 target with minimal risk, I keep your clustering + LR-aug + RF + “tamed centers” pipeline unchanged and only tighten two calibration-related details that directly affect KL. First, I convert the “tamed centers” from raw vote-prob space into an *optimizable* space by adding a very small Dirichlet/Laplace smoothing to both the validation solution probabilities and the candidate centers inside `find_best_tamed_kl`; this prevents overconfident zeros and typically reduces KL on this competition. Second, I broaden the taming search grid slightly (still the same algorithm) so the best fraction can be found more precisely, while keeping runtime small. The submission writing, paths, and row-normalization are preserved to ensure a valid `submission.csv`.'
- What this solution (achieved 1.2565) has done: 'Your current KL (1.20695, lower-is-better) is still well above the target (1.0283), so we make a small calibration-focused improvement without changing the model/feature logic: instead of mapping each test sample to a single cluster center (hard RF class), we use the RF’s predicted class probabilities to form a weighted mixture of the already validation-tuned tamed centers (soft assignment). This keeps the same clustering + LR-aug + RF + “tamed centers” approach, but typically reduces KL by removing overconfident discrete jumps. We also apply the same tiny smoothing/normalization safety checks after mixing to guarantee valid probabilities summing to 1. No training loop, architecture, or feature extraction is changed; only the final mapping from RF output to class-probabilities is made probabilistic.'
- What this solution (achieved 1.2565) has done: 'Your current KL (1.2565, lower-is-better) is still far from the target (1.0283), so we need a small, low-risk calibration improvement without changing your feature extraction, clustering, LR-augmentation, RF training, or taming-search core logic. The biggest likely issue hurting KL is that RF `predict_proba()` column order can differ when some cluster labels are missing in the downselected training subset; mixing centers with mis-ordered probabilities produces wrong class mixtures. I fix this by explicitly aligning the RF probability columns to the full set of cluster ids `[0..NUM_CLUSTS-1]` for train/valid/test before using them, while keeping the same RF and the same “soft centers” idea. I also apply the same soft-mixture mapping for train/valid KL diagnostics (not just test) so the taming/search is tuned against the inference behavior you actually use, which typically improves generalization KL with minimal semantic change. Submission writing/normalization stays intact.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import pyarrow.dataset as pads



## === cell 1
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)



## === cell 2
NUM_CLUSTS = 8  # 6 to 10

USE_PREPROC = False

SMOOTH_WIDTH = 5  # Odd>1: 3,5,7,9,...

TRAIN_DOWNSEL = 23  # small values for code test; set to 1 to output all.
VALID_DOWNSEL = 47  #  "

USE_LR1 = True
LR1_C = 0.0005  # smaller --> fewer non-zero coeff.s
USE_LR2 = False
LR2_C = 0.9
LR_BLUR = 0.10

above_dir = "../input/hms-harmful-brain-activity-classification/"
above_dir_preproc = "../input/hms-2024-brain-data/"



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
    Assumes all columns in `solution` are probability columns.
    """
    eps = 1e-15
    sumsum = 0.0
    for prob_col in solution.columns.values:
        p = np.clip(solution[prob_col].values.astype(float), eps, 1.0)
        q = np.clip(submission[prob_col].values.astype(float), eps, 1.0)
        sumsum += np.nansum(-1.0 * p * np.log(q / p))
    return sumsum / len(solution)




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
        the_probs = np.array([row[c] for c in HBA_probs], dtype=float)
        the_probs = np.clip(the_probs, 1e-8, 1.0)
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
    ]
    if len(iclust_order) > 2:
        kmclrs = hba_clrs.copy()
        for iord, iclust in enumerate(iclust_order):
            kmclrs[iclust] = hba_clrs[iord]
    else:
        kmclrs = hba_clrs

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
def assemble_features(meta_frame, traintest="train", smooth_width=5, SHOW_PLOT=True):
    """
    Create a dataframe of spectrogram features from the meta_frame rows.
    Will include clust_id (i.e, the y) if it is in the input meta_frame.
    Assumes these are available: above_dir, num_clusts
    """
    freqs = np.array(range(100)) * 0.19525 + 0.59
    spect_trend = 150.0 / (1.0**2.3 + freqs ** (2.3))
    freqs[0] = 0.0
    freqs4 = np.array(4 * list(freqs))
    spect_trend4 = np.array(4 * list(spect_trend))

    if SHOW_PLOT:
        plt.figure(figsize=(10, 8))

    feats_frame = []
    last_spectro_id_str = "starting"
    print_every_nth = max([100, 100 * int(0.5 + len(meta_frame.index) / (100.0 * 15))])

    for irow in meta_frame.index:
        this_row = meta_frame.loc[irow]
        spectro_id_str = str(int(this_row.spectrogram_id))
        if spectro_id_str != last_spectro_id_str:
            if traintest != "test":
                spectro_file = (
                    above_dir + "train_spectrograms/" + spectro_id_str + ".parquet"
                )
            else:
                spectro_file = (
                    above_dir + "test_spectrograms/" + spectro_id_str + ".parquet"
                )

            if not os.path.exists(spectro_file):
                raise FileNotFoundError(f"Missing spectrogram parquet: {spectro_file}")

            pads_spectro = pads.dataset(spectro_file)
            this_spectro = pads_spectro.to_table().to_pandas()

        last_spectro_id_str = spectro_id_str

        loc_offset = int(this_row.spectrogram_label_offset_seconds / 2)

        middle4s_raw = (
            this_spectro.iloc[loc_offset + 148, 1:]
            + this_spectro.iloc[loc_offset + 149, 1:]
            + this_spectro.iloc[loc_offset + 150, 1:]
            + this_spectro.iloc[loc_offset + 151, 1:]
        )

        middle4s = middle4s_raw / (4.0 * spect_trend4)
        middle4s = np.clip(middle4s, 0.001, 1000.0)
        middle4s = middle4s.replace([np.nan, -np.inf, np.inf], 0.001)

        denom = (
            this_spectro.iloc[loc_offset + 149 - 56, 1:]
            + this_spectro.iloc[loc_offset + 149 - 40, 1:]
            + this_spectro.iloc[loc_offset + 149 - 24, 1:]
            + this_spectro.iloc[loc_offset + 149 + 56, 1:]
            + this_spectro.iloc[loc_offset + 149 + 40, 1:]
            + this_spectro.iloc[loc_offset + 149 + 24, 1:]
        )

        ratio4s = middle4s_raw / denom
        ratio4s = np.clip((6.0 / 4.0) * ratio4s, 0.01, 100.0)
        ratio4s = ratio4s.replace([np.nan, -np.inf, np.inf], 1.0)
        ratio4spre = np.log10(ratio4s)

        spect_mean = np.mean(middle4s)
        spect_median = np.median(middle4s)

        the4means = []
        the4medians = []
        for ispec in range(4):
            ibeg = ([0, 100, 200, 300])[ispec]
            iend = ibeg + 100
            the4means.append(np.mean(middle4s[ibeg:iend]))
            the4medians.append(np.median(middle4s[ibeg:iend]))

        middle4spre = np.log10(middle4s / spect_mean)

        middle4s_smooth = middle4spre.rolling(
            smooth_width, min_periods=smooth_width, center=True
        ).mean()
        ratio4s_smooth = ratio4spre.rolling(
            smooth_width, min_periods=smooth_width, center=True
        ).mean()

        for ioff in range(0, 400, 100):
            for ibin in range(int((smooth_width - 1) / 2)):
                middle4s_smooth.iloc[ibin + ioff] = middle4spre.iloc[ibin + ioff]
                ratio4s_smooth.iloc[ibin + ioff] = ratio4spre.iloc[ibin + ioff]

        baseinds = np.insert(
            np.arange(int((smooth_width - 1) / 2), 100, smooth_width), 0, 0
        )
        select_inds = np.concatenate(
            (baseinds, 100 + baseinds, 200 + baseinds, 300 + baseinds)
        )

        middle4sds = middle4s_smooth.iloc[select_inds]
        ratio4sds = ratio4s_smooth.iloc[select_inds]

        middle_feats = middle4sds.to_frame().T
        ratio_feats = ratio4sds.to_frame().T
        ratio_feats.columns = ["r" + str(c) for c in ratio_feats.columns]

        these_feats = pd.concat([middle_feats, ratio_feats], axis=1)

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
            feats_frame = pd.concat([feats_frame, these_feats], axis=0)

        if len(feats_frame) % print_every_nth == 0:
            print(f"... {len(feats_frame)} done...")

        if SHOW_PLOT:
            this_clr = "blue"
            title_start = "Middle-8s Feature Values of " + traintest
            plt.title(title_start + f" (smooth={smooth_width})")
            if "clust_id" in meta_frame.columns:
                this_clust = int(this_row.clust_id)
                this_clr = kmclrs[this_clust]
                plt.title(
                    title_start
                    + " (color-coded by the {} clusters, smooth={})".format(
                        num_clusts, smooth_width
                    )
                )
            the_alpha = np.clip(0.05 * 350 / len(meta_frame), 0.003, 0.5)
            plt.plot(
                (
                    freqs4[select_inds]
                    + smooth_width * 0.15 * (np.random.rand(len(select_inds)) - 0.5)
                ),
                middle4sds.values,
                ".",
                c=this_clr,
                markersize=10,
                alpha=the_alpha,
            )
            plt.plot(
                -1.2 + smooth_width * 0.15 * (np.random.rand(4) - 0.5),
                the4means + (0.0 * (np.random.rand(4) - 0.5)),
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
                f"Amplitude (/ref-spectrum, /mean, and smoothed n={smooth_width})"
            )

    if SHOW_PLOT:
        plt.savefig("middle8s_" + traintest + "_features.png")
        plt.show()

    return feats_frame.reset_index().drop(columns=["index"])




## === cell 8
def find_best_tamed_kl(solution_probs_df, pred_ids_series, clust_centers_in):
    """
    Same taming search as before, but made explicit/pure to avoid any unintended
    cross-talk between train/valid calls (improves stability/generalization).

    Change for score-improvement toward target:
    - Add tiny Dirichlet/Laplace smoothing to BOTH solution probabilities and centers
      before KL computation so that overconfident near-zeros do not dominate KL.
      This preserves semantics (probabilities) but typically reduces KL on this task.
    - Use a slightly finer taming fraction grid to find a better calibration point.
    """
    mean_all_probs = np.array(
        [0.208319, 0.132120, 0.128532, 0.138913, 0.179294, 0.212822]
    )

    clust_centers_local = np.array(clust_centers_in, dtype=float, copy=True)

    alpha = 5e-4  # chosen to be small; stabilizes KL when any component ~0

    def _smooth_probs(arr):
        arr = np.array(arr, dtype=float, copy=True)
        arr = np.nan_to_num(
            arr, nan=1.0 / HBA_number, posinf=1.0 / HBA_number, neginf=1.0 / HBA_number
        )
        arr = np.clip(arr, 0.0, 1.0)
        arr = arr + alpha
        arr = arr / arr.sum(axis=1, keepdims=True)
        return arr

    tamed_fracs = 0.0 * np.ones(NUM_CLUSTS)
    tamed_centers = clust_centers_local.copy()
    for iclust in range(NUM_CLUSTS):
        tamed_centers[iclust, :] = mean_all_probs

    best_fracs = tamed_fracs.copy()
    best_centers = tamed_centers.copy()

    sol_raw = solution_probs_df[HBA_votes].values.astype(float)
    sol = _smooth_probs(sol_raw)
    pred_ids = pred_ids_series.values.astype(int)

    clust_centers_sm = _smooth_probs(clust_centers_local)
    mean_all_probs_sm = _smooth_probs(mean_all_probs.reshape(1, -1))[0]

    last_kl_by_clust = 10.0 * np.ones(NUM_CLUSTS, dtype=float)

    frac_grid = np.round(np.arange(0.02, 1.001, 0.03), 4)

    for iclust in range(NUM_CLUSTS):
        for this_frac in frac_grid:
            tamed_fracs[iclust] = float(this_frac)
            this_cent = (
                tamed_fracs[iclust] * clust_centers_sm[iclust, :]
                + (1.0 - tamed_fracs[iclust]) * mean_all_probs_sm
            )
            this_cent = np.clip(this_cent, 1e-12, 1.0)
            this_cent = this_cent / np.sum(this_cent)
            tamed_centers[iclust, :] = this_cent

            sub = tamed_centers[pred_ids, :]
            sub = _smooth_probs(sub)

            eps = 1e-15
            p = np.clip(sol, eps, 1.0)
            q = np.clip(sub, eps, 1.0)
            this_kl = np.nansum(-1.0 * p * np.log(q / p)) / len(sol)

            if this_kl < last_kl_by_clust[iclust]:
                best_fracs = tamed_fracs.copy()
                best_centers = tamed_centers.copy()
                last_kl_by_clust[iclust] = this_kl
            else:
                tamed_fracs[iclust] = best_fracs[iclust]
                tamed_centers[iclust, :] = best_centers[iclust, :]
                break

    return best_fracs, best_centers




## === cell 9
def aligned_predict_proba(rf_model, X_df, num_classes):
    proba = rf_model.predict_proba(X_df).astype(float)
    aligned = np.zeros((len(X_df), num_classes), dtype=float)
    for j, cls in enumerate(rf_model.classes_):
        if 0 <= int(cls) < num_classes:
            aligned[:, int(cls)] = proba[:, j]
    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    aligned = aligned / row_sums
    return aligned




## === cell 10
train_meta, test_meta = read_hms_meta()



## === cell 11
num_clusts = NUM_CLUSTS

clust_rows_bool = (
    (train_meta.eeg_sub_id < 33 + 1) & (train_meta.eeg_sub_id % 5 == 3)
) | ((train_meta.eeg_sub_id == 0) & (((train_meta.eeg_id % 23) % 8) > 1))
print("Cluster rows selected:", int(clust_rows_bool.sum()))

valid_rows_bool = (
    (train_meta.eeg_sub_id < 44 + 1) & (train_meta.eeg_sub_id % 19 == 6)
) | ((train_meta.eeg_sub_id == 0) & (((train_meta.eeg_id % 23) % 8) < 2))
print("Valid rows selected:", int(valid_rows_bool.sum()))



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

prob_array = np.array(prob_vectors)
kmeans = KMeans(
    n_clusters=num_clusts,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=GLOBAL_SEED,
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

PLOT_CLUSTER_SCATTERS = False
if PLOT_CLUSTER_SCATTERS:
    kmclrs = prob_prob_scatter("Seizure", "GPD", all_probs, all_ids, iclust_of_order)
    kmclrs = prob_prob_scatter("LPD", "GRDA", all_probs, all_ids, iclust_of_order)
    kmclrs = prob_prob_scatter("LRDA", "Other", all_probs, all_ids, iclust_of_order)
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
    ][:num_clusts]

clust_counts = train_meta.clust_id.value_counts()

iorder_of_clust = num_clusts * [-1]
for iord, iclust in enumerate(iclust_of_order):
    iorder_of_clust[int(iclust)] = iord



## === cell 14
solution_train = train_meta[["eeg_id"] + HBA_votes].copy()
for col_pre in HBA_names:
    solution_train.loc[:, col_pre + "_vote"] = train_meta[col_pre + "_prob"]

submission_train = solution_train.copy()
clust_ids = train_meta["clust_id"].astype(int).values
for iprob in range(HBA_number):
    submission_train[HBA_votes[iprob]] = clust_centers[:, iprob][clust_ids]

print(
    "Score if samples are assigned cluster center prob.s:",
    np.round(kld_score(solution_train[HBA_votes], submission_train[HBA_votes]), 4),
)



## === cell 15
if USE_PREPROC:
    Xy_train_meta = pd.read_csv(above_dir_preproc + "Xy_train_meta_v47.csv")
    Xy_train_feats = pd.read_csv(above_dir_preproc + "Xy_train_feats_v47.csv")
    Xy_train_meta["clust_id"] = kmeans.predict(np.array(Xy_train_meta[HBA_probs]))
    Xy_train_feats["clust_id"] = Xy_train_meta["clust_id"]
    SMOOTH_WIDTH = 5
else:
    Xy_train_meta = (train_meta[clust_rows_bool])[::TRAIN_DOWNSEL].copy()
    Xy_train_meta = Xy_train_meta.reset_index().drop(columns=["index"])
    print("Number of samples used for training =", len(Xy_train_meta))

    Xy_train_feats = assemble_features(
        Xy_train_meta, traintest="train", smooth_width=SMOOTH_WIDTH, SHOW_PLOT=False
    )



## === cell 16
if USE_PREPROC:
    Xy_valid_meta = pd.read_csv(above_dir_preproc + "Xy_valid_meta_v47.csv")
    Xy_valid_feats = pd.read_csv(above_dir_preproc + "Xy_valid_feats_v47.csv")
    Xy_valid_meta["clust_id"] = kmeans.predict(np.array(Xy_valid_meta[HBA_probs]))
    Xy_valid_feats["clust_id"] = Xy_valid_meta["clust_id"]
else:
    Xy_valid_meta = (train_meta[valid_rows_bool])[::VALID_DOWNSEL].copy()
    Xy_valid_meta = Xy_valid_meta.reset_index().drop(columns=["index"])
    print("Number of samples used for Validation =", len(Xy_valid_meta))

    Xy_valid_feats = assemble_features(
        Xy_valid_meta,
        traintest="validation",
        smooth_width=SMOOTH_WIDTH,
        SHOW_PLOT=False,
    )



## === cell 17
X = Xy_train_feats.drop(columns=["clust_id"])
y = Xy_train_feats.clust_id.astype(int)

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
        random_state=GLOBAL_SEED,
    ).fit(Xlr1, y)

    print("\nLR model score for X,y = {:.1f}%\n".format(100 * lrmodel1.score(Xlr1, y)))

if USE_LR2:
    Xlr2 = Xlr.iloc[:, int(len(Xlr.columns) / 2) :]
    lrmodel2 = LogisticRegression(
        penalty="l1",
        C=LR2_C,
        solver="saga",
        max_iter=1500,
        multi_class="multinomial",
        n_jobs=-1,
        random_state=GLOBAL_SEED,
    ).fit(Xlr2, y)

    print("\nLR model score for X,y = {:.1f}%\n".format(100 * lrmodel2.score(Xlr2, y)))



## === cell 18
_rng_lr_train = np.random.RandomState(GLOBAL_SEED + 123)
_rng_lr_valid = np.random.RandomState(GLOBAL_SEED + 234)

Xy_train_wLRfeats = Xy_train_feats.copy()
X = Xy_train_feats.drop(columns=["clust_id"])
Xlr = X.drop(columns=X.columns[-10:])
if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : int(len(Xlr.columns) / 2)]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    for iadd in range(NUM_CLUSTS):
        Xy_train_wLRfeats["lr" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            _rng_lr_train.rand(len(lrprobas)) - 0.5
        )
if USE_LR2:
    Xlr2 = Xlr.iloc[:, int(len(Xlr.columns) / 2) :]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    for iadd in range(NUM_CLUSTS):
        Xy_train_wLRfeats["rlr" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            _rng_lr_train.rand(len(lrprobas)) - 0.5
        )

Xy_valid_wLRfeats = Xy_valid_feats.copy()
X = Xy_valid_feats.drop(columns=["clust_id"])
Xlr = X.drop(columns=X.columns[-10:])
if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : int(len(Xlr.columns) / 2)]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    for iadd in range(NUM_CLUSTS):
        Xy_valid_wLRfeats["lr" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            _rng_lr_valid.rand(len(lrprobas)) - 0.5
        )
if USE_LR2:
    Xlr2 = Xlr.iloc[:, int(len(Xlr.columns) / 2) :]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    for iadd in range(NUM_CLUSTS):
        Xy_valid_wLRfeats["rlr" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            _rng_lr_valid.rand(len(lrprobas)) - 0.5
        )



## === cell 19
rf_feature_cols = [c for c in Xy_train_wLRfeats.columns if c != "clust_id"]

X = Xy_train_wLRfeats[rf_feature_cols]
y = Xy_train_wLRfeats.clust_id.astype(int)

ave_oob = []
nfits = 3
rfmodel = None
for ifit in range(nfits):
    rfmodel = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=9,
        max_features=0.20,
        max_samples=0.8,
        oob_score=True,
        class_weight="balanced_subsample",
        n_jobs=-1,
        verbose=0,
        random_state=GLOBAL_SEED + ifit,
    ).fit(X, y)
    ave_oob.append(rfmodel.oob_score_)

print(
    "\nRF model ave OOB score = {:.1f}% +/- {:.1f}".format(
        100 * np.mean(ave_oob), 100 * np.std(ave_oob)
    )
)
print("\nRF model score for X,y = {:.1f}%\n".format(100 * rfmodel.score(X, y)))



## === cell 20
Xy_train_meta = Xy_train_meta.copy()
rf_proba_train = aligned_predict_proba(
    rfmodel, Xy_train_wLRfeats[rf_feature_cols], NUM_CLUSTS
)

solution = Xy_train_meta[["eeg_id"] + HBA_votes].copy()
for col_pre in HBA_names:
    solution.loc[:, col_pre + "_vote"] = Xy_train_meta[col_pre + "_prob"]

pred_ids_train_hard = pd.Series(
    np.argmax(rf_proba_train, axis=1), index=Xy_train_meta.index
).astype(int)
best_fracs, best_centers = find_best_tamed_kl(
    solution, pred_ids_train_hard, clust_centers
)

centers_for_train = np.array(best_centers, dtype=float, copy=True)
train_vals = rf_proba_train @ centers_for_train
train_vals = np.nan_to_num(
    train_vals, nan=1.0 / HBA_number, posinf=1.0 / HBA_number, neginf=1.0 / HBA_number
)
train_vals = np.clip(train_vals, 1e-8, 1.0)
train_vals = train_vals / train_vals.sum(axis=1, keepdims=True)

submission = solution.copy()
submission[HBA_votes] = train_vals

this_kl = kld_score(solution[HBA_votes], submission[HBA_votes])
print("KL from tamed centers + soft mixture (train): {:.4f}".format(this_kl))



## === cell 21
Xy_valid_meta = Xy_valid_meta.copy()
rf_proba_valid = aligned_predict_proba(
    rfmodel, Xy_valid_wLRfeats[rf_feature_cols], NUM_CLUSTS
)

solution_v = Xy_valid_meta[["eeg_id"] + HBA_votes].copy()
for col_pre in HBA_names:
    solution_v.loc[:, col_pre + "_vote"] = Xy_valid_meta[col_pre + "_prob"]

pred_ids_v_hard = pd.Series(
    np.argmax(rf_proba_valid, axis=1), index=Xy_valid_meta.index
).astype(int)
best_fracs_v, best_centers_v = find_best_tamed_kl(
    solution_v, pred_ids_v_hard, clust_centers
)

centers_for_valid = np.array(best_centers_v, dtype=float, copy=True)
valid_vals = rf_proba_valid @ centers_for_valid
valid_vals = np.nan_to_num(
    valid_vals, nan=1.0 / HBA_number, posinf=1.0 / HBA_number, neginf=1.0 / HBA_number
)
valid_vals = np.clip(valid_vals, 1e-8, 1.0)
valid_vals = valid_vals / valid_vals.sum(axis=1, keepdims=True)

submission_v = solution_v.copy()
submission_v[HBA_votes] = valid_vals

this_kl_v = kld_score(solution_v[HBA_votes], submission_v[HBA_votes])
print("KL from tamed centers + soft mixture (valid): {:.4f}".format(this_kl_v))



## === cell 22
Xy_test_feats = assemble_features(
    test_meta, traintest="test", smooth_width=SMOOTH_WIDTH, SHOW_PLOT=False
)

_rng_lr_test = np.random.RandomState(GLOBAL_SEED + 456)

Xy_test_wLRfeats = Xy_test_feats.copy()
Xlr = Xy_test_feats.drop(columns=Xy_test_feats.columns[-10:])
if USE_LR1:
    Xlr1 = Xlr.iloc[:, 0 : int(len(Xlr.columns) / 2)]
    lrprobas = lrmodel1.predict_proba(Xlr1)
    for iadd in range(NUM_CLUSTS):
        Xy_test_wLRfeats["lr" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            _rng_lr_test.rand(len(lrprobas)) - 0.5
        )
if USE_LR2:
    Xlr2 = Xlr.iloc[:, int(len(Xlr.columns) / 2) :]
    lrprobas = lrmodel2.predict_proba(Xlr2)
    for iadd in range(NUM_CLUSTS):
        Xy_test_wLRfeats["rlr" + str(iadd)] = lrprobas[:, iadd] + LR_BLUR * (
            _rng_lr_test.rand(len(lrprobas)) - 0.5
        )

for c in rf_feature_cols:
    if c not in Xy_test_wLRfeats.columns:
        Xy_test_wLRfeats[c] = 0.0
Xy_test_wLRfeats = Xy_test_wLRfeats[rf_feature_cols]

rf_proba_test = aligned_predict_proba(rfmodel, Xy_test_wLRfeats, NUM_CLUSTS)

sample_sub = pd.read_csv(above_dir + "sample_submission.csv")
test_submit = sample_sub[["eeg_id"]].copy()

centers_for_test = np.array(best_centers_v, dtype=float, copy=True)  # (NUM_CLUSTS, 6)

test_vals = rf_proba_test @ centers_for_test

test_vals = np.nan_to_num(
    test_vals, nan=1.0 / HBA_number, posinf=1.0 / HBA_number, neginf=1.0 / HBA_number
)
test_vals = np.clip(test_vals, 1e-8, 1.0)
test_vals = test_vals / test_vals.sum(axis=1, keepdims=True)

test_submit[HBA_votes] = test_vals
test_submit = test_submit[sample_sub.columns.tolist()]

print(test_submit.head())
print(
    "Row-sum check (min/max):",
    test_submit[HBA_votes].sum(axis=1).min(),
    test_submit[HBA_votes].sum(axis=1).max(),
)

test_submit.to_csv(
    "submission.csv", header=True, index=False, na_rep="", float_format="%.6f"
)
print("Wrote submission.csv with shape:", test_submit.shape)
