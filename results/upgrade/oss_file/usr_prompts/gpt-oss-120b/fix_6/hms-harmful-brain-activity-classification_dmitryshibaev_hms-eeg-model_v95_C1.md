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

# 5. Code solution

## === cell 0
import pandas as pd
import numpy as np
import os
import psutil
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from pathlib import Path
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from functools import lru_cache
from pympler import asizeof

import typing
from dataclasses import dataclass
from collections.abc import Callable
from collections.abc import Sequence

import itertools

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)




## === cell 1
SUBMISSION = True
SUBMISSION = False
VALIDATION_FRAC = None
VALIDATION_FRAC = 0.05
USE_GPU = True  # enable GPU by default
USE_TPU = False  # = True not tested yet




## === cell 2
ROOT_DIR = "/kaggle/input/hms-harmful-brain-activity-classification"
TRAIN_EEGS = os.path.join(ROOT_DIR, "train_eegs")
TRAIN_SPECTR = os.path.join(ROOT_DIR, "train_spectrograms")
TEST_EEGS = os.path.join(ROOT_DIR, "test_eegs")
TEST_SPECTR = os.path.join(ROOT_DIR, "test_spectrograms")




## === cell 3
ADAM_LEARNING_RATE = 0.0001




## === cell 4
DEBUG = False  # use full training data
if SUBMISSION:
    DEBUG = False

DEBUG_TRAIN_SIZE = 512  # kept for possible manual debugging, not used when DEBUG=False
SKIP_ASSERT = SUBMISSION or not DEBUG




## === cell 5
NUM_EPOCHS = 12
NUM_SUB_EPOCHS = 5 if not DEBUG else 2
BATCH_SIZE = 128




## === cell 6
print("SUBMISSION =", SUBMISSION)
print("USE_TPU =", USE_TPU)
print("USE_GPU =", USE_GPU)
print("DEBUG = ", DEBUG)




## === cell 7
EEG_FRAME_PER_SECOND = 200
EEG_FRAME = 50 * EEG_FRAME_PER_SECOND
SPECTR_FRAME = 10 * 60 // 2




## === cell 8
EEG_FILTER_PERIOD = 7
EEG_FILTER_BASE_PERIOD = EEG_FILTER_PERIOD * 41

EEG_WINDOW_IN_SEC = 24  # in seconds
EEG_WINDOW = EEG_WINDOW_IN_SEC * EEG_FRAME_PER_SECOND  # in ticks
EEG_MODEL_WINDOW = EEG_WINDOW // 3




## === cell 9
def size_2_str(value):
    if value < 5 * 1024:
        return str(value) + " bytes"
    if value < 5 * 1024 * 1024:
        return str(value // 1024) + " KB"
    return str(value // (1024 * 1024)) + " MB"


def get_mem_usage():
    pid = os.getpid()
    py = psutil.Process(pid)
    return py.memory_info()[0] // 2**20




## === cell 10
EEG_COLUMNS = [  # to assert columns order is the same
    "Fp1",
    "F3",
    "C3",
    "P3",
    "F7",
    "T3",
    "T5",
    "O1",
    "Fz",
    "Cz",
    "Pz",
    "Fp2",
    "F4",
    "C4",
    "P4",
    "F8",
    "T4",
    "T6",
    "O2",
    "EKG",
]




## === cell 11
EEG_FEATURES = EEG_COLUMNS  # [ 'EKG' ]




## === cell 12
SPECTR_COLUMNS = [  # to assert columns order is the same
    "time",
    "LL_0.59",
    "LL_0.78",
    "LL_0.98",
    "LL_1.17",
    "LL_1.37",
    "LL_1.56",
    "LL_1.76",
    "LL_1.95",
    "LL_2.15",
    "LL_2.34",
    "LL_2.54",
    "LL_2.73",
    "LL_2.93",
    "LL_3.13",
    "LL_3.32",
    "LL_3.52",
    "LL_3.71",
    "LL_3.91",
    "LL_4.1",
    "LL_4.3",
    "LL_4.49",
    "LL_4.69",
    "LL_4.88",
    "LL_5.08",
    "LL_5.27",
    "LL_5.47",
    "LL_5.66",
    "LL_5.86",
    "LL_6.05",
    "LL_6.25",
    "LL_6.45",
    "LL_6.64",
    "LL_6.84",
    "LL_7.03",
    "LL_7.23",
    "LL_7.42",
    "LL_7.62",
    "LL_7.81",
    "LL_8.01",
    "LL_8.2",
    "LL_8.4",
    "LL_8.59",
    "LL_8.79",
    "LL_8.98",
    "LL_9.18",
    "LL_9.38",
    "LL_9.57",
    "LL_9.77",
    "LL_9.96",
    "LL_10.16",
    "LL_10.35",
    "LL_10.55",
    "LL_10.74",
    "LL_10.94",
    "LL_11.13",
    "LL_11.33",
    "LL_11.52",
    "LL_11.72",
    "LL_11.91",
    "LL_12.11",
    "LL_12.3",
    "LL_12.5",
    "LL_12.7",
    "LL_12.89",
    "LL_13.09",
    "LL_13.28",
    "LL_13.48",
    "LL_13.67",
    "LL_13.87",
    "LL_14.06",
    "LL_14.26",
    "LL_14.45",
    "LL_14.65",
    "LL_14.84",
    "LL_15.04",
    "LL_15.23",
    "LL_15.43",
    "LL_15.63",
    "LL_15.82",
    "LL_16.02",
    "LL_16.21",
    "LL_16.41",
    "LL_16.6",
    "LL_16.8",
    "LL_16.99",
    "LL_17.19",
    "LL_17.38",
    "LL_17.58",
    "LL_17.77",
    "LL_17.97",
    "LL_18.16",
    "LL_18.36",
    "LL_18.55",
    "LL_18.75",
    "LL_18.95",
    "LL_19.14",
    "LL_19.34",
    "LL_19.53",
    "LL_19.73",
    "LL_19.92",
    "RL_0.59",
    "RL_0.78",
    "RL_0.98",
    "RL_1.17",
    "RL_1.37",
    "RL_1.56",
    "RL_1.76",
    "RL_1.95",
    "RL_2.15",
    "RL_2.34",
    "RL_2.54",
    "RL_2.73",
    "RL_2.93",
    "RL_3.13",
    "RL_3.32",
    "RL_3.52",
    "RL_3.71",
    "RL_3.91",
    "RL_4.1",
    "RL_4.3",
    "RL_4.49",
    "RL_4.69",
    "RL_4.88",
    "RL_5.08",
    "RL_5.27",
    "RL_5.47",
    "RL_5.66",
    "RL_5.86",
    "RL_6.05",
    "RL_6.25",
    "RL_6.45",
    "RL_6.64",
    "RL_6.84",
    "RL_7.03",
    "RL_7.23",
    "RL_7.42",
    "RL_7.62",
    "RL_7.81",
    "RL_8.01",
    "RL_8.2",
    "RL_8.4",
    "RL_8.59",
    "RL_8.79",
    "RL_8.98",
    "RL_9.18",
    "RL_9.38",
    "RL_9.57",
    "RL_9.77",
    "RL_9.96",
    "RL_10.16",
    "RL_10.35",
    "RL_10.55",
    "RL_10.74",
    "RL_10.94",
    "RL_11.13",
    "RL_11.33",
    "RL_11.52",
    "RL_11.72",
    "RL_11.91",
    "RL_12.11",
    "RL_12.3",
    "RL_12.5",
    "RL_12.7",
    "RL_12.89",
    "RL_13.09",
    "RL_13.28",
    "RL_13.48",
    "RL_13.67",
    "RL_13.87",
    "RL_14.06",
    "RL_14.26",
    "RL_14.45",
    "RL_14.65",
    "RL_14.84",
    "RL_15.04",
    "RL_15.23",
    "RL_15.43",
    "RL_15.63",
    "RL_15.82",
    "RL_16.02",
    "RL_16.21",
    "RL_16.41",
    "RL_16.6",
    "RL_16.8",
    "RL_16.99",
    "RL_17.19",
    "RL_17.38",
    "RL_17.58",
    "RL_17.77",
    "RL_17.97",
    "RL_18.16",
    "RL_18.36",
    "RL_18.55",
    "RL_18.75",
    "RL_18.95",
    "RL_19.14",
    "RL_19.34",
    "RL_19.53",
    "RL_19.73",
    "RL_19.92",
    "LP_0.59",
    "LP_0.78",
    "LP_0.98",
    "LP_1.17",
    "LP_1.37",
    "LP_1.56",
    "LP_1.76",
    "LP_1.95",
    "LP_2.15",
    "LP_2.34",
    "LP_2.54",
    "LP_2.73",
    "LP_2.93",
    "LP_3.13",
    "LP_3.32",
    "LP_3.52",
    "LP_3.71",
    "LP_3.91",
    "LP_4.1",
    "LP_4.3",
    "LP_4.49",
    "LP_4.69",
    "LP_4.88",
    "LP_5.08",
    "LP_5.27",
    "LP_5.47",
    "LP_5.66",
    "LP_5.86",
    "LP_6.05",
    "LP_6.25",
    "LP_6.45",
    "LP_6.64",
    "LP_6.84",
    "LP_7.03",
    "LP_7.23",
    "LP_7.42",
    "LP_7.62",
    "LP_7.81",
    "LP_8.01",
    "LP_8.2",
    "LP_8.4",
    "LP_8.59",
    "LP_8.79",
    "LP_8.98",
    "LP_9.18",
    "LP_9.38",
    "LP_9.57",
    "LP_9.77",
    "LP_9.96",
    "LP_10.16",
    "LP_10.35",
    "LP_10.55",
    "LP_10.74",
    "LP_10.94",
    "LP_11.13",
    "LP_11.33",
    "LP_11.52",
    "LP_11.72",
    "LP_11.91",
    "LP_12.11",
    "LP_12.3",
    "LP_12.5",
    "LP_12.7",
    "LP_12.89",
    "LP_13.09",
    "LP_13.28",
    "LP_13.48",
    "LP_13.67",
    "LP_13.87",
    "LP_14.06",
    "LP_14.26",
    "LP_14.45",
    "LP_14.65",
    "LP_14.84",
    "LP_15.04",
    "LP_15.23",
    "LP_15.43",
    "LP_15.63",
    "LP_15.82",
    "LP_16.02",
    "LP_16.21",
    "LP_16.41",
    "LP_16.6",
    "LP_16.8",
    "LP_16.99",
    "LP_17.19",
    "LP_17.38",
    "LP_17.58",
    "LP_17.77",
    "LP_17.97",
    "LP_18.16",
    "LP_18.36",
    "LP_18.55",
    "LP_18.75",
    "LP_18.95",
    "LP_19.14",
    "LP_19.34",
    "LP_19.53",
    "LP_19.72",
    "LP_19.92",
    "RP_0.59",
    "RP_0.78",
    "RP_0.98",
    "RP_1.17",
    "RP_1.37",
    "RP_1.56",
    "RP_1.76",
    "RP_1.95",
    "RP_2.15",
    "RP_2.34",
    "RP_2.54",
    "RP_2.73",
    "RP_2.93",
    "RP_3.13",
    "RP_3.32",
    "RP_3.52",
    "RP_3.71",
    "RP_3.91",
    "RP_4.1",
    "RP_4.3",
    "RP_4.49",
    "RP_4.69",
    "RP_4.88",
    "RP_5.08",
    "RP_5.27",
    "RP_5.47",
    "RP_5.66",
    "RP_5.86",
    "RP_6.05",
    "RP_6.25",
    "RP_6.45",
    "RP_6.64",
    "RP_6.84",
    "RP_7.03",
    "RP_7.23",
    "RP_7.42",
    "RP_7.62",
    "RP_7.81",
    "RP_8.01",
    "RP_8.2",
    "RP_8.4",
    "RP_8.59",
    "RP_8.79",
    "RP_8.98",
    "RP_9.18",
    "RP_9.38",
    "RP_9.57",
    "RP_9.77",
    "RP_9.96",
    "RP_10.16",
    "RP_10.35",
    "RP_10.55",
    "RP_10.74",
    "RP_10.94",
    "RP_11.13",
    "RP_11.33",
    "RP_11.52",
    "RP_11.72",
    "RP_11.91",
    "RP_12.11",
    "RP_12.3",
    "RP_12.5",
    "RP_12.7",
    "RP_12.89",
    "RP_13.09",
    "RP_13.28",
    "RP_13.48",
    "RP_13.67",
    "RP_13.87",
    "RP_14.06",
    "RP_14.26",
    "RP_14.45",
    "RP_14.65",
    "RP_14.84",
    "RP_15.04",
    "RP_15.23",
    "RP_15.43",
    "RP_15.63",
    "RP_15.82",
    "RP_16.02",
    "RP_16.21",
    "RP_16.41",
    "RP_16.6",
    "RP_16.8",
    "RP_16.99",
    "RP_17.19",
    "RP_17.38",
    "RP_17.58",
    "RP_17.77",
    "RP_17.97",
    "RP_18.16",
    "RP_18.36",
    "RP_18.55",
    "RP_18.75",
    "RP_18.95",
    "RP_19.14",
    "RP_19.34",
    "RP_19.53",
    "RP_19.73",
    "RP_19.92",
]




## === cell 13
VOTE_COLUMNS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]




## === cell 14
TARGET_FEATURES = VOTE_COLUMNS




## === cell 15
SPECTR_FEATURES = list([x for x in SPECTR_COLUMNS if x != "time"])




## === cell 16
def scale_probs(probs):
    s = np.sum(probs, axis=-1, keepdims=True)
    return probs / s




## === cell 17
def _moving_mean(arr, window, axis):
    """Vectorised moving average using cumulative sum (O(N) per axis)."""
    if window <= 1:
        return arr.astype(float)
    pad = window // 2
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(arr, pad_width, mode="edge")
    cumsum = np.cumsum(padded, axis=axis, dtype=float)
    slc_start = [slice(None)] * arr.ndim
    slc_end = [slice(None)] * arr.ndim
    slc_start[axis] = slice(0, -window)
    slc_end[axis] = slice(window, None)
    sum_window = cumsum[tuple(slc_end)] - cumsum[tuple(slc_start)]
    return sum_window / window


def filter_eeg_signals(data):  # shape = time_index, eeg_chanal
    data = data[
        len(data) // 2
        - EEG_WINDOW // 2
        - EEG_FILTER_BASE_PERIOD : len(data) // 2
        + EEG_WINDOW // 2
        + EEG_FILTER_BASE_PERIOD
    ]
    data = np.nan_to_num(data, nan=0, copy=False)
    base_mean = _moving_mean(data, window=EEG_FILTER_BASE_PERIOD, axis=0)
    if base_mean.shape[0] != data.shape[0]:
        data = data[: base_mean.shape[0]]
    data = data - base_mean
    data = _moving_mean(data, window=EEG_FILTER_PERIOD, axis=0)
    total_max = np.max(np.abs(data), axis=0)
    data = data[:: EEG_WINDOW // EEG_MODEL_WINDOW]
    total_max = total_max.reshape(1, 20)
    total_max[total_max == 0] = 1
    data = data[
        len(data) // 2 - EEG_MODEL_WINDOW // 2 : len(data) // 2 + EEG_MODEL_WINDOW // 2
    ]
    data = data / total_max
    return data




## === cell 18
def decrease_int_type(column):
    try:
        new_column = column.astype("int8")
        if new_column.astype(column.dtype).equals(column):
            return new_column
    except Exception:
        pass
    try:
        new_column = column.astype("int16")
        if new_column.astype(column.dtype).equals(column):
            return new_column
    except Exception:
        pass
    try:
        new_column = column.astype("int32")
        if new_column.astype(column.dtype).equals(column):
            return new_column
    except Exception:
        pass
    return column




## === cell 19
@lru_cache(maxsize=128)
def load_train_eeg_frame(id):
    data = pd.read_parquet(
        os.path.join(TRAIN_EEGS, str(id) + ".parquet"), engine="pyarrow"
    )
    if not SKIP_ASSERT:
        assert list(data.columns) == EEG_COLUMNS, "EEG columns order is not the same!"
    return data


@lru_cache(maxsize=128)
def load_test_eeg_frame(id):
    data = pd.read_parquet(
        os.path.join(TEST_EEGS, str(id) + ".parquet"), engine="pyarrow"
    )
    if not SKIP_ASSERT:
        assert list(data.columns) == EEG_COLUMNS, "EEG columns order is not the same!"
    return data


def load_train_eeg_frame_wrapper(id):
    return load_train_eeg_frame(id)


def load_test_eeg_frame_wrapper(id):
    return load_test_eeg_frame(id)


class TrainTargetLoader(Callable):
    def __init__(self, test_info):
        self.data = scale_probs(test_info[TARGET_FEATURES].to_numpy())

    def __call__(self, index):
        return self.data[index]




## === cell 20
class TrainEegLoader(Callable):
    def __init__(self, train_info):
        self.data = pd.DataFrame(
            {
                c: decrease_int_type(train_info[c])
                for c in ["eeg_id", "eeg_label_offset_seconds"]
            }
        )

    @lru_cache(maxsize=None)
    def __call__(self, index):
        eeg_id, start = self.data.iloc[index]
        start = start * EEG_FRAME_PER_SECOND
        end = start + EEG_FRAME
        data = load_train_eeg_frame(eeg_id)
        if not SKIP_ASSERT:
            assert start >= 0 and start <= len(
                data
            ), "inlvalid start = {}, len = {}".format(start, len(data))
            assert end <= len(data) and end >= 0, "invalid end = {}, len = {}".format(
                end, len(data)
            )
        data = filter_eeg_signals(data[EEG_FEATURES].iloc[start:end].to_numpy())
        return data.astype(dtype=np.float32)




## === cell 21
def load_train():
    train_info = pd.read_csv(os.path.join(ROOT_DIR, "train.csv")).drop(
        columns=[
            "expert_consensus",
            "eeg_sub_id",
            "spectrogram_sub_id",
            "patient_id",
            "label_id",
        ]
    )
    if DEBUG:
        train_info = train_info.sample(DEBUG_TRAIN_SIZE)
    return (len(train_info), TrainEegLoader(train_info), TrainTargetLoader(train_info))




## === cell 22
train_size, train_eeg_loader, train_target_loader = load_train()




## === cell 23
class TestEegLoader(Callable):
    def __init__(self, test_info):
        self.data = test_info

    def __call__(self, index):
        eeg_id = self.data["eeg_id"].iloc[index]
        data = load_test_eeg_frame(eeg_id)
        return filter_eeg_signals(data.to_numpy())




## === cell 24
class TestEegIdLoader(Callable):
    def __init__(self, test_info):
        self.data = test_info

    def __call__(self, index):
        eeg_id = self.data["eeg_id"].iloc[index]
        return eeg_id




## === cell 25
def load_test():
    test_info = pd.read_csv(os.path.join(ROOT_DIR, "test.csv"))

    test_eeg_loader = TestEegLoader(test_info)
    test_eeg_id_loader = TestEegIdLoader(test_info)
    return len(test_info), test_eeg_loader, test_eeg_id_loader




## === cell 26
class BatchSlicer:
    def __init__(self, length: int, batch_size: int = None, num_batches: int = None):
        assert length >= 0, "invalid length, must be > 0"
        assert (batch_size is None) != (
            num_batches is None
        ), "one and only one from batch_size and num_batches must be setted"
        assert batch_size is None or batch_size > 0, "invalid batch_size, must be > 0"
        assert (
            num_batches is None or num_batches > 0
        ), "invalid num_batches, must be > 0"
        self._length = length
        if num_batches is None:
            self._batch_size = batch_size
            self._num_batches = (
                self._length + self._batch_size - 1
            ) // self._batch_size
        else:
            self._num_batches = num_batches
            self._batch_size = (
                self._length + self._num_batches - 1
            ) // self._num_batches
        self._real_batch_size = self._length // self._num_batches
        self._num_full_batches = (
            self._length - self._num_batches * self._real_batch_size
        )
        self._num_short_batches = self._num_batches - self._num_full_batches

    def __getitem__(self, index: int):  # return -> range(begin, end)
        if not SKIP_ASSERT:
            assert (
                index >= 0 and index < self._num_batches
            ), "invalid batch index ={}, num_batches={}".format(
                index, self._num_batches
            )
        if index < self._num_full_batches:
            begin = index * (self._real_batch_size + 1)
            end = begin + (self._real_batch_size + 1)
        else:
            begin = self._num_full_batches * (self._real_batch_size + 1) + (
                index - self._num_full_batches
            ) * (self._real_batch_size)
            end = begin + self._real_batch_size
        return range(begin, end)

    def __len__(self):
        return self._num_batches




## === cell 27
@dataclass(frozen=True)
class SubSequence(Sequence):
    source: Sequence
    start: int
    stop: int

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, index):
        if index < 0 or index >= self.stop - self.start:
            raise IndexError
        return self.source[self.start + index]




## === cell 28
class BatchedSequence(Sequence):
    def __init__(self, source: Sequence, batch_size: int, item_dtype, item_shape):
        self.source = source
        self.item_dtype = item_dtype
        self.item_shape = item_shape
        self.batch_slicer = BatchSlicer(len(self.source), batch_size=batch_size)

    def __len__(self):
        return len(self.batch_slicer)

    def __getitem__(self, index):
        rng = self.batch_slicer[index]
        return np.fromiter(
            [self.source[i] for i in rng],
            count=rng.stop - rng.start,
            dtype=(self.item_dtype, self.item_shape),
        )




## === cell 29
def assert_no_nan(x):
    assert not np.isnan(x).any(), "has NAN!!!"
    return x




## === cell 30
@dataclass(frozen=True)
class TransformSequence(Sequence):
    source: Sequence
    transform: Callable

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return self.transform(self.source[index])




## === cell 31
@dataclass(frozen=True)
class JoinSequence(Sequence):
    a: Sequence
    b: Sequence

    def __len__(self):
        return len(self.a)

    def __getitem__(self, index):
        return (self.a[index], self.b[index])




## === cell 32
class AsKerasSequence(keras.utils.Sequence):
    def __init__(
        self, source: Sequence, on_epoch_end_callback: Callable = lambda: True
    ):
        self.source = source
        self.on_epoch_end_callback = on_epoch_end_callback

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return self.source[index]

    def on_epoch_end(self):
        self.on_epoch_end_callback()




## === cell 33
class SplitSubEpoches(keras.utils.Sequence):
    def __init__(self, source, num_sub_epoches):
        self.source = source
        self.num_sub_epoches = num_sub_epoches
        self.current_sub_epoch = 0
        self.slicer = BatchSlicer(len(self.source), num_batches=self.num_sub_epoches)

    def __len__(self):
        rng = self.slicer[self.current_sub_epoch]
        return rng.stop - rng.start

    def __getitem__(self, index):
        rng = self.slicer[self.current_sub_epoch]
        if index < 0 or index >= rng.stop - rng.start:
            raise IndexError()
        return self.source[rng.start + index]

    def on_epoch_end(self):
        self.current_sub_epoch += 1
        if self.current_sub_epoch == self.num_sub_epoches:
            self.source.on_epoch_end()
            self.current_sub_epoch = 0




## === cell 34
def make_decision(probs):
    decision = np.zeros_like(probs)
    decision[np.arange(len(probs)), probs.argmax(1)] = 1
    return decision




## === cell 35
def scale_features(data):
    data = np.transpose(data, axes=(0, 2, 1))
    mean = np.nanmean(data, axis=(2), keepdims=True)
    std = np.nanstd(data, axis=(2), keepdims=True)
    std[std == 0] = 1
    data = np.nan_to_num((data - mean) / std, 0, 0, 0)
    data = np.transpose(data, axes=(0, 2, 1))
    return data




## === cell 36
def create_eeg_model():
    input = keras.layers.Input(
        shape=(EEG_MODEL_WINDOW, len(EEG_FEATURES)), name="eeg.input"
    )
    model = keras.layers.Conv1D(
        filters=29,
        kernel_size=5,
        padding="valid",
        name="eeg.1",
        data_format="channels_last",
    )(input)
    model = keras.layers.MaxPooling1D(pool_size=7, strides=3, name="eeg.1.max")(model)
    model = keras.layers.Conv1D(
        filters=29,
        kernel_size=5,
        padding="valid",
        name="eeg.2",
        data_format="channels_last",
        activation="tanh",
    )(model)
    model = keras.layers.MaxPooling1D(pool_size=7, strides=3, name="eeg.2.max")(model)
    model = keras.layers.Conv1D(
        filters=31, kernel_size=5, padding="valid", name="eeg.3"
    )(model)
    model = keras.layers.MaxPooling1D(pool_size=7, strides=3, name="eeg.3.max")(model)
    model = keras.layers.Conv1D(
        filters=41, kernel_size=5, padding="valid", name="eeg.4"
    )(model)
    model = keras.layers.MaxPooling1D(pool_size=7, strides=3, name="eeg.4.max")(model)
    model = keras.layers.Flatten(name="eeg.flatten")(model)
    model = keras.layers.Dense(units=20, activation="relu", name="eeg.dense.1")(model)
    model = keras.layers.Dense(units=20, activation="relu", name="eeg.dense.2")(model)
    model = keras.layers.Dense(
        units=len(VOTE_COLUMNS), activation="softmax", name="eeg.output"
    )(model)
    model = keras.models.Model(inputs=input, outputs=model)
    return model




## === cell 37
if USE_TPU:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tpu_strategy = tf.distribute.TPUStrategy(tpu)




## === cell 38
def create_and_compile_model():
    model = create_eeg_model()
    optimizer = Adam(learning_rate=ADAM_LEARNING_RATE)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["acc"])
    return model




## === cell 39
if USE_TPU:
    with tpu_strategy.scope():
        model = create_and_compile_model()
else:
    model = create_and_compile_model()

model.summary()
keras.utils.plot_model(model, "model.png", show_shapes=True)




## === cell 40
def build_train_data(ids):
    eeg = TransformSequence(ids, train_eeg_loader)
    target = TransformSequence(ids, train_target_loader)

    eeg = BatchedSequence(
        eeg,
        BATCH_SIZE,
        item_dtype=np.float32,
        item_shape=(EEG_MODEL_WINDOW, len(EEG_FEATURES)),
    )
    target = BatchedSequence(
        target, BATCH_SIZE, item_dtype=np.float64, item_shape=(len(TARGET_FEATURES),)
    )

    features_seq = eeg
    features_target_seq = JoinSequence(features_seq, target)
    return AsKerasSequence(features_target_seq, lambda: random.shuffle(ids))




## === cell 41
train_ids = list([x for x in range(train_size)])
if VALIDATION_FRAC > 0:
    train_ids, valid_ids = train_test_split(train_ids, test_size=VALIDATION_FRAC)


train_data = build_train_data(train_ids)
valid_data = build_train_data(valid_ids) if VALIDATION_FRAC > 0 else None

train_data = SplitSubEpoches(train_data, NUM_SUB_EPOCHS)




## === cell 42
device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
print("Found GPU at: {}".format(device_name))




## === cell 43
def fit_model(model, train_data, epochs, validation_data):
    if USE_GPU:
        with tf.device("/gpu:0"):
            return model.fit(train_data, epochs=epochs, validation_data=validation_data)
    else:
        return model.fit(train_data, epochs=epochs, validation_data=validation_data)


history = fit_model(model, train_data, NUM_EPOCHS * NUM_SUB_EPOCHS, valid_data)




## === cell 44
plt.plot(history.history["loss"], "r", label="Training loss")
if VALIDATION_FRAC > 0:
    plt.plot(history.history["val_loss"], "g", label="Validation loss")
plt.title("Training VS Validation loss")
plt.xlabel("No. of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()




## === cell 45
plt.plot(history.history["acc"], "r", label="Training accuracy")
if VALIDATION_FRAC > 0:
    plt.plot(history.history["val_acc"], "g", label="Validation accuracy")
plt.title("Training Vs Validation Accuracy")
plt.xlabel("No. of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()




## === cell 46
print("--------------- model fitted ----------------------")




## === cell 47
def check_validation():
    y_true = []
    y_predict = []
    for index in range(len(valid_data)):
        train_batch, target_batch = valid_data[index]
        predict = model.predict(train_batch)
        decision = make_decision(predict)
        target_decision = make_decision(target_batch)
        for i in range(len(decision)):
            y_predict.append(decision[i])
            y_true.append(target_decision[i])
    print("accuracy =", accuracy_score(y_true, y_predict))




## === cell 48
if VALIDATION_FRAC > 0:
    if USE_GPU:
        with tf.device("/gpu:0"):
            check_validation()
    else:
        check_validation()




## === cell 49
print("mem usage =", size_2_str(get_mem_usage()))
del train_eeg_loader, train_target_loader




## === cell 50
def build_test_data(test_size, test_eeg_loader, test_eeg_id_loader):
    indexes = list([x for x in range(test_size)])
    eeg = TransformSequence(indexes, test_eeg_loader)
    eeg_id = TransformSequence(indexes, test_eeg_id_loader)

    eeg = BatchedSequence(
        eeg,
        BATCH_SIZE,
        item_dtype=np.float32,
        item_shape=(EEG_MODEL_WINDOW, len(EEG_FEATURES)),
    )
    eeg_id = BatchedSequence(eeg_id, BATCH_SIZE, item_dtype=np.int64, item_shape=(1,))

    features_seq = eeg
    features_eeg_id_seq = JoinSequence(features_seq, eeg_id)
    return AsKerasSequence(features_eeg_id_seq)




## === cell 51
test_size, test_eeg_loader, test_eeg_id_loader = load_test()
test_data = build_test_data(test_size, test_eeg_loader, test_eeg_id_loader)




## === cell 52
output = []
for index in range(len(test_data)):
    batch, eeg_ids = test_data[index]
    predict = model.predict(batch)
    for i in range(len(eeg_ids)):
        res = [*eeg_ids[i]]
        res.extend(predict[i].tolist())
        output.append(res)
output = pd.DataFrame(data=output, columns=["eeg_id"] + VOTE_COLUMNS)
output.to_csv("submission.csv", index=False)




## === cell 53
model.save("eeg_model.keras")




## === cell 54
print("--------------- submission done ----------------------")
