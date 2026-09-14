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

3.13

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

0.3128683855148949

# 6. Current score

0.77818

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The fix removes the problematic external EfficientNet import (which crashes due to a protobuf incompatibility) and replaces it with TensorFlow’s built‑in EfficientNetB0. It also skips loading those ImageNet weights and, for inference, uses a simple prior‑based prediction (the average class distribution from the training set) to guarantee a valid CSV submission without heavy computation. These minimal changes resolve the runtime error and produce a proper submission while keeping the original workflow intact.'
- What this solution (achieved 1.41937) has done: 'The fix prevents the TensorFlow import (which fails due to a protobuf incompatibility) from crashing the script by wrapping it in a safe try‑except block and only loading it when training is required. Since `NEEDTRAIN` is set to 0, the script now skips all TensorFlow‑related code and directly creates a prior‑based submission, ensuring a valid `submission.csv` is written without runtime errors.'
- What this solution (achieved 1.41937) has done: 'I replace the naïve global‑prior prediction with a hierarchy of priors: first use the class distribution observed for each eeg_id in the training set, then fall back to the patient‑level distribution, and finally to the overall class prior. This keeps the script purely inference‑only (no training, no external libraries) while giving much more specific probability estimates, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.90499) has done: 'I fixed the dimensionality issue when creating the probability matrix. The masks for EEG‑ and patient‑level priors were computed per‑row (as Series) instead of per‑cell, preventing an unintended extra dimension that caused a `(9850, 9850, 6)` shape. Consequently, the combined probabilities now have the correct `(n_samples, 6)` shape, allowing the DataFrame assignment and yielding a valid submission CSV.'
- What this solution (achieved 0.82785) has done: 'I keep the overall workflow unchanged but give more weight to the more specific priors (eeg‑level then patient‑level) when combining them with the global prior. This small weighting tweak should produce probability estimates that are closer to the true distribution, moving the KL‑divergence score down toward the target while still writing a valid `submission.csv`.'
- What this solution (achieved 0.78226) has done: 'I keep the overall workflow unchanged but increase the influence of the more specific priors (eeg‑level and patient‑level) by raising their weights. This makes the combined probabilities rely more on information that is closer to each test sample, which should lower the KL‑divergence (move the score toward the lower target). The only code change is the adjustment of the three weight constants and a brief comment explaining the intention.'
- What this solution (achieved 0.77818) has done: 'I keep the overall workflow unchanged but increase the influence of the more specific priors (eeg‑level and patient‑level) by raising their weights. This should push the predictions closer to the observed distributions for each sample, which is expected to lower the KL‑divergence and move the score toward the target 0.3128. The only modification is the adjustment of the three weight constants and a brief comment.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri
email: syuri@tju.edu.cn
"""

import os
import warnings
import pandas as pd, numpy as np

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

NEEDTRAIN = 0  # keep existing setting (0 = inference only)

if NEEDTRAIN:
    try:
        import tensorflow as tf
        from tensorflow.keras import optimizers
        from tensorflow.keras.models import clone_model
        from tensorflow.python.framework.ops import reset_default_graph
        from tensorflow.keras.applications import EfficientNetB0 as EfficientNetB0_tf
    except Exception as e:
        tf = None
        EfficientNetB0_tf = None
        print("TensorFlow import failed:", e)
        raise RuntimeError(
            "Training requested but TensorFlow could not be imported."
        ) from e
else:
    tf = None
    EfficientNetB0_tf = None

import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
import time
import gc

MIX = True
if MIX:
    if tf is not None:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print("Mixed precision enabled")
    else:
        print("Mixed precision requested but TensorFlow not available.")
else:
    print("Using full precision")

PLATFORM = "kaggle"  # *** local kaggle *** local training or online testing
DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print("DATATYPE:", DATATYPE)

LOAD_MODELS_FROM = "models20241125c"  # the path of trained model weights for testing

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]  # seizure_vote … other_vote
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

if not NEEDTRAIN:
    class_counts = df[TARGETS].sum()
    class_prior = class_counts / class_counts.sum()  # Series length 6

    eeg_group = df.groupby("eeg_id")[TARGETS].sum()
    eeg_totals = eeg_group.sum(axis=1)
    eeg_prior = eeg_group.div(eeg_totals, axis=0).replace([np.inf, -np.inf], np.nan)

    patient_group = df.groupby("patient_id")[TARGETS].sum()
    patient_totals = patient_group.sum(axis=1)
    patient_prior = patient_group.div(patient_totals, axis=0).replace(
        [np.inf, -np.inf], np.nan
    )

    test_df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))

    sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})

    eeg_probs = eeg_prior.reindex(test_df["eeg_id"]).reset_index(drop=True)
    patient_probs = patient_prior.reindex(test_df["patient_id"]).reset_index(drop=True)

    eeg_mask = (~eeg_probs.isnull()).any(axis=1).astype(int)  # 1 if EEG prior exists
    patient_mask = (
        (~patient_probs.isnull()).any(axis=1).astype(int)
    )  # 1 if patient prior exists

    eeg_filled = eeg_probs.fillna(0)
    patient_filled = patient_probs.fillna(0)

    w_eeg = 20  # original 10
    w_patient = 10  # original 5
    w_global = 1  # unchanged
    print(f"Using weights -> EEG: {w_eeg}, Patient: {w_patient}, Global: {w_global}")

    weight_counts = (
        eeg_mask * w_eeg + patient_mask * w_patient + w_global
    )  # total weight per row

    global_prior_matrix = pd.DataFrame(
        np.tile(class_prior.values, (len(test_df), 1)),
        columns=TARGETS,
    )

    eeg_arr = eeg_filled.to_numpy()  # (n, 6)
    patient_arr = patient_filled.to_numpy()  # (n, 6)
    global_arr = global_prior_matrix.to_numpy()  # (n, 6)

    eeg_mask_arr = eeg_mask.values[:, None]  # (n, 1)
    patient_mask_arr = patient_mask.values[:, None]  # (n, 1)
    weight_arr = weight_counts.values[:, None]  # (n, 1)

    combined_arr = (
        eeg_arr * (eeg_mask_arr * w_eeg)
        + patient_arr * (patient_mask_arr * w_patient)
        + global_arr * w_global
    ) / weight_arr

    row_sums = combined_arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs = combined_arr / row_sums

    sub[TARGETS] = probs
    sub[TARGETS] = sub[TARGETS].fillna(class_prior)  # safety fallback

    submission_path = "submission.csv"
    sub.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")
    print("Submission shape:", sub.shape)
    print(sub.head())
