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

No external packages required in the script and installed.

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

0.3463126379953835

# 6. Current score

0.78827

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I remove the invalid pip‑install commands, guard the imports of TensorFlow, EfficientNet and CuPy so they don’t crash when unavailable, compute class‑wise prior probabilities from the training metadata, and fall back to these priors whenever the pre‑trained model files cannot be loaded. This ensures the script runs end‑to‑end and writes a valid submission.csv with probabilities that sum to 1.'
- What this solution (achieved 1.41937) has done: 'I remove the heavy modality loading by setting `DATATYPE` to an empty list, compute per‑eeg ID vote distributions from the training metadata, and use those distributions (falling back to the global class priors) to generate the test predictions. This keeps the original flow but replaces the placeholder prior‑only prediction with a more informative, still lightweight, heuristic, improving the KL‑divergence score while ensuring a valid CSV is written.'
- What this solution (achieved 1.68479) has done: 'I avoid importing TensorFlow, EfficientNet and CuPy (they cause protobuf errors) by conditioning those imports on NEEDTRAIN, and I add a simple patient‑level fallback: if a test eeg_id is unseen, use the averaged vote distribution of its patient_id from the training data before finally falling back to the global class priors. This keeps the original lightweight heuristic while giving more informative predictions, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.77767) has done: 'I add a small smoothing step that mixes each predicted probability vector with the global class prior (10 % prior, 90 % specific) before normalising. This keeps the original heuristic (eeg‑level → patient‑level → prior) but makes the predictions less extreme, which usually lowers KL‑divergence and moves the score from 1.68 closer to the target 0.346. No other logic is changed.'
- What this solution (achieved 0.82785) has done: 'I keep the overall workflow unchanged but replace the single‑fallback prediction with a small blend of EEG‑level, patient‑level and global‑prior probabilities. When an EEG‑specific distribution exists we mix it with its patient’s distribution (giving the patient a modest influence) and a tiny amount of the overall class prior; if the EEG is unseen we fall back to patient + prior. This richer yet still lightweight heuristic usually yields probabilities that are better calibrated to the true label distribution, lowering the KL‑divergence and moving the score closer to the target while preserving the original script’s structure and output format.'
- What this solution (achieved 0.78827) has done: 'I add a lightweight spectrogram‑level distribution and adjust the blending weights to give a bit more influence to the specific EEG and spectrogram information while reducing the prior contribution. This keeps the original simple heuristic but makes the predictions more calibrated, which should lower the KL‑divergence toward the target score.'

# 9. Code solution

## === cell 0
import os, io, random, itertools
import pandas as pd, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PLATFORM = "kaggle"  # keep as original
NEEDTRAIN = False
DATATYPE = []  # original was ["eeg", "spe", "img"]
STAGETRAIN = [2, 3]
STAGETEST = 3
threshold = 0.2
EEG_LENGTH = 30  # s
SFREQ = 100
HIGH, LENGTH = 128, 128
IMG_HIGH, IMG_WIDE = 64, 256
SEED = 2024
NSPLIT = 5
BATCHSIZE = 16
READ_EXTRA_SPEC_FILES = READ_EXTRA_EEG_FILES = READ_EXTRA_IMG_FILES = (
    READ_EXTRA_STFT_FILES
) = True

LOAD_MODELS_FROM = "models2024040403"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

filter_range = [0.5, 40]
BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

if NEEDTRAIN:
    try:
        import tensorflow as tf
    except Exception as e:
        print("TensorFlow import failed:", e)
        tf = None

    try:
        import efficientnet.tfkeras as efn
    except Exception as e:
        print("EfficientNet import failed:", e)
        efn = None

    try:
        import cupy as cp
    except Exception as e:
        print("CuPy import failed (not needed for inference):", e)
        cp = None
else:
    tf = None
    efn = None
    cp = None

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
elif PLATFORM == "kaggle":
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )
TARGETS = df.columns[-6:]  # ['seizure_vote', ..., 'other_vote']
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))

train_votes_sum = df[TARGETS].sum(axis=0).astype(float)
prior_probs = (train_votes_sum / train_votes_sum.sum()).values  # shape (6,)
print("Class prior probabilities:", prior_probs)

epsilon = 1e-6
eeg_counts = df.groupby("eeg_id")[list(TARGETS)].sum() + epsilon
eeg_prob_df = eeg_counts.div(eeg_counts.sum(axis=1), axis=0)

patient_counts = df.groupby("patient_id")[list(TARGETS)].sum() + epsilon
patient_prob_df = patient_counts.div(patient_counts.sum(axis=1), axis=0)

spectrogram_counts = df.groupby("spectrogram_id")[list(TARGETS)].sum() + epsilon
spectrogram_prob_df = spectrogram_counts.div(spectrogram_counts.sum(axis=1), axis=0)



## === cell 1
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
    elif PLATFORM == "kaggle":
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
    print("Test shape", test.shape)
    test.head()

    w_eeg = 0.60  # EEG‑specific
    w_patient = 0.20  # patient‑level
    w_spec = 0.15  # spectrogram‑level (new)
    w_prior = 0.05  # global prior

    probs_list = []
    for idx, row in test.iterrows():
        eid = row["eeg_id"]
        pid = row["patient_id"]
        sid = row["spectrogram_id"]

        probs = np.zeros(len(TARGETS), dtype=float)

        if eid in eeg_prob_df.index:
            probs += w_eeg * eeg_prob_df.loc[eid].values

        if pid in patient_prob_df.index:
            probs += w_patient * patient_prob_df.loc[pid].values

        if sid in spectrogram_prob_df.index:
            probs += w_spec * spectrogram_prob_df.loc[sid].values

        probs += w_prior * prior_probs

        probs = probs / probs.sum()
        probs_list.append(probs)

    pred = np.vstack(probs_list)

    sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
    sub[TARGETS] = pred
    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print("First rows of submission:")
    print(sub.head())
