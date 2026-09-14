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

1.126431773362965

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40075) has done: 'I remove the problematic `bottleneck` import and replace its moving‑average calls with a NumPy implementation, and I make the `AsKerasSequence` class mutable (remove `frozen=True`) so Keras can assign internal attributes like `_workers`. These minimal fixes unblock execution, allow model training, and keep the original architecture and data handling unchanged, which should bring the validation score closer to the target.'
- What this solution (achieved 1.42553) has done: 'The update adds unlimited caching for raw EEG parquet files to avoid repeated disk reads, defines a small pool of worker processes, and runs model fitting (and validation) with parallel data loading. These changes keep the exact model architecture, training schedule, and data handling logic while dramatically reducing I/O‑bound time, allowing the whole pipeline to finish well within the 600‑second limit.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd
import psutil
import random
from functools import lru_cache
from dataclasses import dataclass
from typing import Callable, Sequence
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

ROOT_DIR = "/kaggle/input/hms-harmful-brain-activity-classification"
TRAIN_EEGS = os.path.join(ROOT_DIR, "train_eegs")
TRAIN_SPECTR = os.path.join(ROOT_DIR, "train_spectrograms")
TEST_EEGS = os.path.join(ROOT_DIR, "test_eegs")
TEST_SPECTR = os.path.join(ROOT_DIR, "test_spectrograms")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
SUBMISSION = True
VALIDATION_FRAC = 0.05
USE_GPU = True  # enable GPU by default
USE_TPU = False  # = True not tested yet



## === cell 2
DEBUG = False  # use full training data
if SUBMISSION:
    DEBUG = True  # when submitting, work on a small debug set for speed

DEBUG_TRAIN_SIZE = 512  # kept for possible manual debugging, not used when DEBUG=False
SKIP_ASSERT = SUBMISSION or not DEBUG



## === cell 3
print("SUBMISSION =", SUBMISSION)
print("USE_TPU =", USE_TPU)
print("USE_GPU =", USE_GPU)
print("DEBUG = ", DEBUG)
NUM_WORKERS = min(4, os.cpu_count() or 1)
print("NUM_WORKERS =", NUM_WORKERS)




## === cell 4
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




## === cell 5
@lru_cache(maxsize=None)
def load_train_eeg_frame(id):
    data = pd.read_parquet(
        os.path.join(TRAIN_EEGS, str(id) + ".parquet"), engine="pyarrow"
    )
    if not SKIP_ASSERT:
        assert list(data.columns) == EEG_COLUMNS, "EEG columns order is not the same!"
    return data


@lru_cache(maxsize=None)
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
    def __init__(self, train_info):
        self.data = scale_probs(train_info[VOTE_COLUMNS].to_numpy())

    def __call__(self, index):
        return self.data[index]




## === cell 6
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
            ), f"inlvalid start = {start}, len = {len(data)}"
            assert (
                end <= len(data) and end >= 0
            ), f"invalid end = {end}, len = {len(data)}"
        data = filter_eeg_signals(
            data[EEG_FEATURES].iloc[int(start) : int(end)].to_numpy()
        )
        return data.astype(dtype=np.float32)




## === cell 7
class TestEegLoader(Callable):
    def __init__(self, test_info):
        self.data = test_info

    def __call__(self, index):
        eeg_id = self.data["eeg_id"].iloc[index]
        data = load_test_eeg_frame(eeg_id)
        return filter_eeg_signals(data.to_numpy())




## === cell 8
class TestEegIdLoader(Callable):
    def __init__(self, test_info):
        self.data = test_info

    def __call__(self, index):
        eeg_id = self.data["eeg_id"].iloc[index]
        return eeg_id




## === cell 9
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




## === cell 10
@dataclass(frozen=True)
class TransformSequence(Sequence):
    source: Sequence
    transform: Callable

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return self.transform(self.source[index])




## === cell 11
@dataclass(frozen=True)
class JoinSequence(Sequence):
    a: Sequence
    b: Sequence

    def __len__(self):
        return len(self.a)

    def __getitem__(self, index):
        return (self.a[index], self.b[index])




## === cell 12
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




## === cell 13
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




## === cell 14
def create_and_compile_model():
    model = create_eeg_model()
    optimizer = Adam(learning_rate=ADAM_LEARNING_RATE)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["acc"])
    return model


try:
    if USE_TPU:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.tpu.experimental.initialize_tpu_system(tpu)
        tpu_strategy = tf.distribute.TPUStrategy(tpu)
        with tpu_strategy.scope():
            model = create_and_compile_model()
    else:
        model = create_and_compile_model()
    MODEL_AVAILABLE = True
except Exception as e:
    print(f"TensorFlow import failed ({e}) – using fallback dummy model.")

    class DummyModel:
        def __init__(self, constant_probs):
            self.constant_probs = constant_probs

        def fit(self, *args, **kwargs):
            class History:
                history = {"loss": [0.0], "acc": [0.0]}

            return History()

        def predict(self, batch):
            batch_size = batch.shape[0]
            return np.tile(self.constant_probs, (batch_size, 1))

        def save(self, path):
            pass

    overall_mean = np.mean(train_target_loader.data, axis=0)
    model = DummyModel(overall_mean)
    MODEL_AVAILABLE = False

model.summary()
if hasattr(keras.utils, "plot_model"):
    keras.utils.plot_model(model, "model.png", show_shapes=True)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2411598719.py in <cell line: 0>()
     15     else:
---> 16         model = create_and_compile_model()
     17     MODEL_AVAILABLE = True

/tmp/ipykernel_55/2411598719.py in create_and_compile_model()
      1 def create_and_compile_model():
----> 2     model = create_eeg_model()
      3     optimizer = Adam(learning_rate=ADAM_LEARNING_RATE)

NameError: name 'create_eeg_model' is not defined

During handling of the above exception, another exception occurred:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2411598719.py in <cell line: 0>()
     36             pass
     37 
---> 38     overall_mean = np.mean(train_target_loader.data, axis=0)
     39     model = DummyModel(overall_mean)
     40     MODEL_AVAILABLE = False

NameError: name 'train_target_loader' is not defined

## === cell 15
train_ids = list(range(train_size))
if VALIDATION_FRAC > 0:
    train_ids, valid_ids = train_test_split(
        train_ids, test_size=VALIDATION_FRAC, random_state=42
    )

train_data = build_train_data(train_ids)
valid_data = build_train_data(valid_ids) if VALIDATION_FRAC > 0 else None

train_data = SplitSubEpoches(train_data, NUM_SUB_EPOCHS)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2952240417.py in <cell line: 0>()
----> 1 train_ids = list(range(train_size))
      2 if VALIDATION_FRAC > 0:
      3     train_ids, valid_ids = train_test_split(
      4         train_ids, test_size=VALIDATION_FRAC, random_state=42
      5     )

NameError: name 'train_size' is not defined

## === cell 16
device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
print("Found GPU at:", device_name)




## === cell 17
def fit_model(model, train_data, epochs, validation_data):
    fit_kwargs = {
        "epochs": epochs,
        "validation_data": validation_data,
    }
    if USE_GPU:
        with tf.device("/gpu:0"):
            return model.fit(train_data, **fit_kwargs)
    else:
        return model.fit(train_data, **fit_kwargs)


history = fit_model(model, train_data, NUM_EPOCHS * NUM_SUB_EPOCHS, valid_data)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4159739249.py in <cell line: 0>()
     11 
     12 
---> 13 history = fit_model(model, train_data, NUM_EPOCHS * NUM_SUB_EPOCHS, valid_data)
     14 

NameError: name 'model' is not defined

## === cell 18
print("mem usage =", size_2_str(get_mem_usage()))
del train_eeg_loader, train_target_loader



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1785290615.py in <cell line: 0>()
      1 print("mem usage =", size_2_str(get_mem_usage()))
----> 2 del train_eeg_loader, train_target_loader
      3 

NameError: name 'train_eeg_loader' is not defined

## === cell 19
output = []
for index in range(len(test_data)):
    batch, eeg_ids = test_data[index]
    predict = model.predict(batch)
    for i in range(len(eeg_ids)):
        eeg_id_val = int(eeg_ids[i][0])
        res = [eeg_id_val]
        res.extend(predict[i].tolist())
        output.append(res)

output = pd.DataFrame(data=output, columns=["eeg_id"] + VOTE_COLUMNS)
output[VOTE_COLUMNS] = output[VOTE_COLUMNS].apply(
    lambda row: scale_probs(row.values), axis=1
)
output.to_csv("submission.csv", index=False)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1255434244.py in <cell line: 0>()
      1 output = []
----> 2 for index in range(len(test_data)):
      3     batch, eeg_ids = test_data[index]
      4     predict = model.predict(batch)
      5     for i in range(len(eeg_ids)):

NameError: name 'test_data' is not defined

## === cell 20
print("--------------- submission done ----------------------")
