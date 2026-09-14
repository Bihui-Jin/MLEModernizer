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

0.3090758002577844

# 6. Current score

0.76105

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'Implemented a lightweight fallback that bypasses the heavy parquet‑based preprocessing which was causing a protobuf `MessageFactory` error.  
The new logic simply computes overall class vote proportions from the training metadata and assigns these probabilities to every test record, guaranteeing a valid `submission.csv` with correctly summed rows. This fixes the runtime failure while providing a reasonable baseline prediction for the KL‑divergence metric.'
- What this solution (achieved 1.68479) has done: 'Implemented a lightweight fix for the protobuf MessageFactory error by setting the protobuf implementation before importing TensorFlow.  
Replaced the naïve global‑class baseline with a patient‑aware prior: compute per‑patient vote distributions from the training data and use them for test predictions, falling back to the overall class distribution when a patient is unseen. This retains the original “no‑training” workflow while providing a more informative probability estimate, keeps rows normalized, and writes a proper `submission.csv`.'
- What this solution (achieved 1.68479) has done: 'Implemented a lightweight fix for the protobuf MessageFactory error by forcing TensorFlow to be skipped (setting `KAGGLE_SKIP_TF=1`) and guarding all TensorFlow‑dependent classes (`DataGenerator`, `CosineAnnealingLRScheduler`, `build_model`) behind a check that `tf` is available. This prevents import‑time crashes while keeping the original inference‑only logic intact, ensuring a valid `submission.csv` is written with correctly normalized probabilities.'
- What this solution (achieved 1.68479) has done: 'I guard the TensorFlow‑related imports so they are only attempted when TensorFlow can be loaded; otherwise dummy placeholders are set. This prevents the protobuf `MessageFactory` error that occurs during the unconditional imports, while keeping the rest of the logic untouched (the model isn’t needed because `NEEDTRAIN=False`). The rest of the pipeline (computing per‑patient and global vote distributions and writing a normalized `submission.csv`) remains unchanged.'
- What this solution (achieved 1.68479) has done: 'I guard the TensorFlow‑related imports so they are only attempted when TensorFlow is actually loaded; this prevents the protobuf `MessageFactory` error that aborts the script. The rest of the logic (per‑patient priors, normalization, and CSV output) remains unchanged, ensuring a valid `submission.csv` and moving the KL‑divergence score toward the target.'
- What this solution (achieved 0.76139) has done: 'The fix adds a per‑eeg‑id prior and makes the patient‑weight smoothing stronger (smaller ALPHA). For each test record we now use the specific eeg_id distribution when it exists, otherwise we fall back to the blended patient‑global prior, and finally normalise the probabilities. This modest change keeps the original workflow but gives more informative predictions, moving the KL‑divergence score closer to the target.'
- What this solution (achieved 0.80881) has done: 'I increase the smoothing α so the patient‑level prior is weighted less heavily (reducing over‑fitting to small patient groups) and drop the per‑eeg override, letting every record use the blended patient / global probabilities. This modest change keeps the overall workflow intact while making the predictions more conservative, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.76139) has done: 'I lower the smoothing factor (ALPHA) so patient‑level priors have more influence and add a per‑eeg‑id fallback: when an eeg_id exists in the training data we directly use its class distribution, otherwise we keep the blended patient/global probabilities. This keeps the overall workflow unchanged while providing more specific predictions, which should reduce the KL‑divergence toward the target score.'
- What this solution (achieved 0.71277) has done: 'I replace the way class probabilities are computed: instead of summing raw vote counts, I first convert each row’s votes to a probability distribution and then average these per‑patient and per‑eeg. This yields better‑calibrated priors while keeping the same blending logic, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.713) has done: 'I slightly adjust the smoothing of the patient‑level blend (reduce ALPHA to give a bit more weight to patient priors) and add a tiny uniform smoothing term before the final normalisation. These small changes keep the original workflow intact while making the predicted distributions a little less extreme, which should reduce the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.76105) has done: 'We give the patient‑level prior more influence (lower ALPHA) and compute it with vote‑count weighting, then smooth the final probabilities a bit more. This keeps the original workflow but should produce less extreme predictions and lower KL divergence, moving the score toward the target.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

import os

os.environ["KAGGLE_SKIP_TF"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import io, gc, time, itertools
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
import importlib
import warnings

warnings.filterwarnings("ignore")

tf = None
if not os.getenv("KAGGLE_SKIP_TF"):
    try:
        tf = importlib.import_module("tensorflow")
    except Exception:
        tf = None

if tf is not None:
    try:
        from tensorflow.keras import optimizers
        from tensorflow.keras.models import clone_model
        from tensorflow.python.framework.ops import reset_default_graph
        from tensorflow.keras.applications import EfficientNetB0
    except Exception:
        optimizers = None
        clone_model = None
        reset_default_graph = None
        EfficientNetB0 = None
else:
    optimizers = None
    clone_model = None
    reset_default_graph = None
    EfficientNetB0 = None

PLATFORM = "kaggle"
NEEDTRAIN = False

DATATYPE = ["eeg"]
print(DATATYPE)

if PLATFORM == "local":
    LOAD_MODELS_FROM = "./input/models20241124b"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = "/kaggle/input/models20241124b"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

SEED = 2024
np.random.seed(SEED)

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

if not NEEDTRAIN:
    vote_counts = df[TARGETS].sum(axis=1)

    row_probs = df[TARGETS].div(vote_counts, axis=0)

    global_probs = row_probs.mean()

    weighted_sum = (
        row_probs.multiply(vote_counts, axis=0).groupby(df["patient_id"]).sum()
    )

    patient_vote_counts = vote_counts.groupby(df["patient_id"]).sum()

    patient_probs = weighted_sum.div(patient_vote_counts, axis=0)

    ALPHA = 0.5  # reduced from 2.0 to give patient priors more impact
    patient_weights = patient_vote_counts / (patient_vote_counts + ALPHA)

    eeg_probs = row_probs.groupby(df["eeg_id"]).mean()

    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    sub = pd.DataFrame({"eeg_id": test["eeg_id"]})

    merged_pat = test[["eeg_id", "patient_id"]].merge(
        patient_probs, left_on="patient_id", right_index=True, how="left"
    )

    patient_w_arr = patient_weights.reindex(merged_pat["patient_id"]).fillna(0).values

    for col in TARGETS:
        patient_col = merged_pat[col].fillna(0).values
        blended = patient_w_arr * patient_col + (1 - patient_w_arr) * global_probs[col]
        merged_pat[col] = blended

    sub[TARGETS] = merged_pat[TARGETS].values

    eeg_present = sub["eeg_id"].isin(eeg_probs.index)
    if eeg_present.any():
        eeg_idx = sub.loc[eeg_present, "eeg_id"]
        sub.loc[eeg_present, TARGETS] = eeg_probs.loc[eeg_idx].values

    SMOOTH_EPS = 1e-2  # increased smoothing (was 1e-3)
    sub[TARGETS] = sub[TARGETS] + SMOOTH_EPS

    EPS = 1e-4  # clip very small values before final normalisation
    sub[TARGETS] = sub[TARGETS].clip(lower=EPS)

    sub[TARGETS] = sub[TARGETS].div(sub[TARGETS].sum(axis=1), axis=0)

    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(sub.head())
