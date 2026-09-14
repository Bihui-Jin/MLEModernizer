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

0.4575461801753707

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the two blockers that prevent the notebook from running end-to-end: the TensorFlow/protobuf `MessageFactory.GetPrototype` crash (by forcing the C++ protobuf backend and importing TensorFlow after that), and the `MergeError` in submission building (by removing the too-strict `validate="one_to_one"` and ensuring we keep exactly the `sample_submission` row order). I also make the cell numbering start at 1 so it matches the expected “cells” format, while keeping your model/data logic unchanged. Finally, I keep the existing weight-loading behavior; if weights are present you get real predictions (better than uniform), otherwise it still produce a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by forcing TensorFlow to use the pure-Python protobuf backend (and setting it before importing TensorFlow), which resolves the `MessageFactory.GetPrototype` error in Kaggle’s Py3.12 environment. Then I make the inference path robust and faster by reading only the spectrogram parquet files actually needed for the 9850 test rows (instead of loading the entire directory), without changing any model/data semantics. Finally, I keep the submission building aligned to `sample_submission.csv` row order, enforce probability normalization, and ensure a valid `submission.csv` is always written.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` *and* `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` before importing TensorFlow (this is the most reliable workaround for the `MessageFactory.GetPrototype` error in Kaggle Py3.12). Then I fix an inference ordering bug: the generator currently follows `sample_submission` order but `DataGenerator.__data_generation` incorrectly indexes `self.data.iloc[i]` instead of `iloc[indexes[i]]`, causing a mismatch between the requested batch indices and the rows actually used. This bug can severely degrade score even with correct weights, so correcting it should move KL divergence down toward your target without changing the model/feature logic. Finally, I keep submission alignment to `sample_submission.csv`, enforce probability normalization, and ensure `submission.csv` is always written.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf implementation environment variables *before* any TensorFlow-related import and by importing `google.protobuf` first (this ordering is what prevents the `MessageFactory.GetPrototype` error in Kaggle Py3.12). Then I keep your model and feature logic unchanged but ensure inference uses the correct batch indices inside `DataGenerator.__data_generation` (this was a silent indexing bug that can severely corrupt predictions and inflate KL). Finally, I make the submission writing path always produce a valid `submission.csv` in the exact `sample_submission.csv` row order with row-wise probability normalization (score-neutral correctness).'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
LOAD_MODELS_FROM = "models202402093"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 20.48  # s
SFREQ = 200

HIGH = 64  # 128
LENGTH = 256  # 256

READ_SPEC_FILES = False
READ_EEG_FILES = False

filter_range = [0.5, 40]

BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy import signal

try:
    from tensorflow.python.framework.ops import reset_default_graph
except Exception:

    def reset_default_graph():
        return None


print("TensorFlow version =", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")
if len(gpus) <= 1:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print(f"Using {len(gpus)} GPU")
else:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using {len(gpus)} GPUs")

VER = 1

MIX = True
if MIX:
    try:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print("Mixed precision enabled")
    except Exception as e:
        print("Mixed precision could not be enabled; continuing. Reason:", repr(e))
else:
    print("Using full precision")

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
elif PLATFORM == "kaggle":
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )

TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/3702057600.py in <cell line: 0>()
     40 import numpy as np
     41 import pandas as pd
---> 42 import tensorflow as tf
     43 import matplotlib.pyplot as plt
     44 

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     47 _tf2.enable()
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__
     51 from tensorflow._api.v2 import audio

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator
     10 from tensorflow._api.v2.__internal__ import dispatch

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     19 import threading
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export
     23 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     15 """Utility module that contains APIs usable in the generated code."""
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors
     19 from tensorflow.python.autograph.utils.tensor_list import dynamic_list_append

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     17 import contextlib
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops
     21 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     31 
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2
     35 from tensorflow.core.framework import function_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      3 # source: tensorflow/core/framework/attr_value.proto
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor
      7 from google.protobuf import descriptor_pool as _descriptor_pool

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     16 
     17 from google.protobuf.internal import enum_type_wrapper
---> 18 from google.protobuf.internal import python_message
     19 from google.protobuf import message as _message
     20 from google.protobuf import reflection as _reflection

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     36 import weakref
     37 
---> 38 from google.protobuf import descriptor as descriptor_mod
     39 from google.protobuf import message as message_mod
     40 from google.protobuf import text_format

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     27   # TODO: Remove this import after fix api_implementation
     28   if _message is None:
---> 29     from google.protobuf.pyext import _message
     30   _USE_C_DESCRIPTORS = True
     31 

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

## === cell 1
train = df.groupby("eeg_id")[
    ["spectrogram_id", "spectrogram_label_offset_seconds", "eeg_label_offset_seconds"]
].agg(
    {
        "spectrogram_id": "first",
        "spectrogram_label_offset_seconds": "min",
        "eeg_label_offset_seconds": "median",
    }
)
train.columns = ["spec_id", "min", "eeg_median"]

tmp = df.groupby("eeg_id")[["spectrogram_id", "spectrogram_label_offset_seconds"]].agg(
    {"spectrogram_label_offset_seconds": "max"}
)
train["max"] = tmp

tmp = df.groupby("eeg_id")[["patient_id"]].agg("first")
train["patient_id"] = tmp

tmp = df.groupby("eeg_id")[TARGETS].agg("sum")
for t in TARGETS:
    train[t] = tmp[t].values

y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1, keepdims=True)
train[TARGETS] = y_data

tmp = df.groupby("eeg_id")[["expert_consensus"]].agg("first")
train["target"] = tmp

train = train.reset_index()
print("Train non-overlapp eeg_id shape:", train.shape)
train.head()



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4166120209.py in <cell line: 0>()
----> 1 train = df.groupby("eeg_id")[
      2     ["spectrogram_id", "spectrogram_label_offset_seconds", "eeg_label_offset_seconds"]
      3 ].agg(
      4     {
      5         "spectrogram_id": "first",

NameError: name 'df' is not defined

## === cell 2
if NEEDTRAIN:
    if PLATFORM == "local":
        PATH = "./input/hms-harmful-brain-activity-classification/train_spectrograms/"
    elif PLATFORM == "kaggle":
        PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    files = os.listdir(PATH)
    print(f"There are {len(files)} spectrogram parquets")

    if READ_SPEC_FILES:
        spectrograms = {}
        for i, f in enumerate(files):
            if i % 100 == 0:
                print(i, ", ", end="")
            tmp = pd.read_parquet(f"{PATH}{f}")
            name = int(f.split(".")[0])
            spectrograms[name] = tmp.iloc[:, 1:].values
        if not os.path.exists("./input/brain-spectrograms"):
            os.makedirs("./input/brain-spectrograms")
        np.save("./input/brain-spectrograms/specs.npy", spectrograms, allow_pickle=True)
    else:
        if PLATFORM == "local":
            spectrograms = np.load(
                "./input/brain-spectrograms/specs.npy", allow_pickle=True
            ).item()
        elif PLATFORM == "kaggle":
            spectrograms = np.load(
                "/kaggle/input/brain-spectrograms/specs.npy", allow_pickle=True
            ).item()

if NEEDTRAIN:
    if PLATFORM == "local":
        PATH = "./input/hms-harmful-brain-activity-classification/train_eegs/"
    elif PLATFORM == "kaggle":
        PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
    files = os.listdir(PATH)
    print(f"There are {len(files)} eeg parquets")

    if READ_EEG_FILES:
        eegs = {}
        if len(filter_range) == 1:
            if filter_range[0] > 5:
                b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "lowpass")
            else:
                b, a = signal.butter(
                    3, np.float32(filter_range) * 2 / SFREQ, "highpass"
                )
        else:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

        for i, f in enumerate(files):
            if i % 100 == 0:
                print(i, ", ", end="")
            raw_eeg = pd.read_parquet(f"{PATH}{f}")
            name = int(f.split(".")[0])

            if len(train[train.eeg_id == name]) > 0:
                time_temp = train[train.eeg_id == name].eeg_median.iloc[-1]
                time_start = round(time_temp * 200 + (50 - EEG_LENGTH) / 2 * 200)
                time_stop = round(time_temp * 200 + (50 + EEG_LENGTH) / 2 * 200)

                eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(
                    drop=True
                )

                list_eeg = list()
                for region in BRAIN.keys():
                    eeg = np.zeros(
                        (len(BRAIN[region]), eeg_default.shape[0]), dtype=np.float32
                    )
                    for chan_i, chan in enumerate(BRAIN[region]):
                        eeg[chan_i, :] = (
                            eeg_default.loc[:, chan.split("-")[0]]
                            - eeg_default.loc[:, chan.split("-")[1]]
                        ).values

                    eeg[np.isnan(eeg)] = 0

                    if 200 != SFREQ:
                        eeg = signal.resample_poly(eeg, SFREQ, 200, axis=1)

                    eeg = signal.filtfilt(b, a, eeg, axis=1)

                    list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

                list_eeg = np.concatenate(list_eeg, 2)
                eegs[name] = list_eeg

        if not os.path.exists("./input/brain-eegs"):
            os.makedirs("./input/brain-eegs")
        np.save("./input/brain-eegs/eegs.npy", eegs, allow_pickle=True)
    else:
        if PLATFORM == "local":
            eegs = np.load("./input/brain-eegs/eegs.npy", allow_pickle=True).item()
        elif PLATFORM == "kaggle":
            eegs = np.load(
                "/kaggle/input/brain-eegs/eegs.npy", allow_pickle=True
            ).item()



## === cell 3
try:
    import albumentations as albu
except Exception:
    albu = None

TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
TARS2 = {x: y for y, x in TARS.items()}


def _stft_db_tf(x_1d: np.ndarray, length: int = LENGTH) -> np.ndarray:
    """Compute log-power spectrogram in dB-like scale using TF; returns float32 [freq, time]."""
    x = tf.convert_to_tensor(x_1d, dtype=tf.float32)
    hop = max(int(len(x_1d) // length), 1)
    stft = tf.signal.stft(
        x,
        frame_length=128,
        frame_step=hop,
        fft_length=256,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )
    power = tf.abs(stft) ** 2  # [time, freq]
    power = tf.transpose(power)  # [freq, time]
    power = power[:48, :length]
    ref = tf.maximum(tf.reduce_median(power), tf.constant(1e-10, tf.float32))
    db = (
        10.0
        * (tf.math.log(tf.maximum(power, 1e-10)) - tf.math.log(ref))
        / tf.math.log(10.0)
    )
    return db.numpy().astype(np.float32)


class DataGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        data,
        batch_size=32,
        shuffle=False,
        augment=False,
        mode="train",
        specs=None,
        eegs=None,
    ):

        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.mode = mode
        self.specs = specs
        self.eegs = eegs
        self.on_epoch_end()

    def __len__(self):
        ct = int(np.ceil(len(self.data) / self.batch_size))
        return ct

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X, X_mel, X_eeg, y = self.__data_generation(indexes)
        if self.augment:
            X = self.__augment_batch(X)
        return [X, X_mel, X_eeg], y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        X = np.zeros((len(indexes), HIGH, LENGTH, 4), dtype="float32")
        X_mel = np.zeros((len(indexes), HIGH, LENGTH, 4), dtype="float32")
        X_eeg = np.zeros(
            (len(indexes), 6, round(EEG_LENGTH * SFREQ), 4), dtype="float32"
        )
        y = np.zeros((len(indexes), 6), dtype="float32")

        for j, idx in enumerate(indexes):
            row = self.data.iloc[idx]

            if self.mode == "test":
                r = 0
            elif self.mode == "valid":
                r = int((row["min"] + row["max"]) // 4)
            else:
                r = np.random.randint(row["min"], row["max"] + 1) // 2

            for k in range(4):
                img = self.specs[row.spec_id][r : r + 300, k * 100 : (k + 1) * 100].T
                img_eeg = self.eegs[row.eeg_id][:, :, k]

                img = np.clip(img, np.exp(-6), np.exp(8))
                img = np.log(img)
                img = np.nan_to_num(img, nan=0.0)

                img = img[
                    :,
                    max(round((600 / 2 - LENGTH) / 2), 0) : min(
                        (round((600 / 2 - LENGTH) / 2) + LENGTH), img.shape[1]
                    ),
                ]
                if HIGH != 100:
                    img = np.array(
                        tf.image.resize(
                            np.reshape(img, (img.shape[0], img.shape[1], 1)),
                            ((HIGH - 16), LENGTH),
                        ),
                        dtype=np.float32,
                    )[:, :, 0]
                    X[
                        j,
                        round((HIGH - img.shape[0]) / 2) : round(
                            (HIGH + img.shape[0]) / 2
                        ),
                        :,
                        k,
                    ] = img
                else:
                    X[j, :, :, k] = img

                X[j, :, :, k] = (X[j, :, :, k] - np.mean(X[j, :, :, k])) / (
                    np.std(X[j, :, :, k]) + 1e-6
                )

                mel_spec_db = None
                for ii_x in range(img_eeg.shape[0]):
                    x = img_eeg[ii_x, :]
                    spec_db = _stft_db_tf(x, length=LENGTH)
                    if mel_spec_db is None:
                        mel_spec_db = spec_db
                    else:
                        mel_spec_db = mel_spec_db + spec_db
                mel_spec_db = mel_spec_db / img_eeg.shape[0]

                if HIGH != 100:
                    mel_spec_db = np.array(
                        tf.image.resize(
                            np.reshape(
                                mel_spec_db,
                                (mel_spec_db.shape[0], mel_spec_db.shape[1], 1),
                            ),
                            ((HIGH - 16), LENGTH),
                        ),
                        dtype=np.float32,
                    )[:, :, 0]
                    X_mel[
                        j,
                        round((HIGH - mel_spec_db.shape[0]) / 2) : round(
                            (HIGH + mel_spec_db.shape[0]) / 2
                        ),
                        :,
                        k,
                    ] = mel_spec_db
                else:
                    X_mel[j, :, :, k] = mel_spec_db
                X_mel[j, :, :, k] = (X_mel[j, :, :, k] - np.mean(X_mel[j, :, :, k])) / (
                    np.std(X_mel[j, :, :, k]) + 1e-6
                )

                X_eeg[j, 1:5, :, k] = img_eeg
                X_eeg[j, :, :, k] = (
                    X_eeg[j, :, :, k] - np.mean(X_eeg[j, :, :, k], 1, keepdims=True)
                ) / (np.std(X_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

            if self.mode != "test":
                y[j] = row[TARGETS].values

        return X, X_mel, X_eeg, y

    def __random_transform(self, img):
        if albu is None:
            return img
        composition = albu.Compose(
            [
                albu.HorizontalFlip(p=0.5),
                albu.CoarseDropout(
                    max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5
                ),
            ]
        )
        return composition(image=img)["image"]

    def __augment_batch(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i,] = self.__random_transform(img_batch[i,])
        return img_batch




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3698691337.py in <cell line: 0>()
     32 
     33 
---> 34 class DataGenerator(tf.keras.utils.Sequence):
     35     "Generates data for Keras"
     36 

NameError: name 'tf' is not defined

## === cell 4
def wave_block(x, filters, kernel_size, n):
    dilation_rates = [2**i for i in range(n)]
    x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding="same")(x)
    res_x = x
    for dilation_rate in dilation_rates:
        tanh_out = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation="tanh",
            dilation_rate=dilation_rate,
        )(x)
        sigm_out = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation="sigmoid",
            dilation_rate=dilation_rate,
        )(x)
        x = tf.keras.layers.Multiply()([tanh_out, sigm_out])
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding="same")(x)
        res_x = tf.keras.layers.Add()([res_x, x])
    return res_x


def build_model():
    inp = tf.keras.Input(shape=(HIGH, LENGTH, 4))
    inp_mel = tf.keras.Input(shape=(HIGH, LENGTH, 4))
    inp_eeg = tf.keras.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4))

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_tensor=None,
        name="spectrogram_extractor_efficientnetb0",
    )

    x0 = inp[:, :, :, :1]
    x1 = inp[:, :, :, 1:2]
    x2 = inp[:, :, :, 2:3]
    x3 = inp[:, :, :, 3:4]
    x01 = tf.keras.layers.Concatenate(axis=1)([x0, x1, x2, x3])

    x_mel0 = inp_mel[:, :, :, :1]
    x_mel1 = inp_mel[:, :, :, 1:2]
    x_mel2 = inp_mel[:, :, :, 2:3]
    x_mel3 = inp_mel[:, :, :, 3:4]
    x_mel = tf.keras.layers.Concatenate(axis=1)([x_mel0, x_mel1, x_mel2, x_mel3])

    x01 = tf.keras.layers.Concatenate(axis=2)([x01, x_mel])
    x = tf.keras.layers.Concatenate(axis=3)([x01, x01, x01])

    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x)

    base_model_eeg = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_tensor=None,
        name="eeg_extractor_efficientnetb0",
    )

    x0_eeg = inp_eeg[:, :, :, :1]
    x1_eeg = inp_eeg[:, :, :, 1:2]
    x2_eeg = inp_eeg[:, :, :, 2:3]
    x3_eeg = inp_eeg[:, :, :, 3:4]
    x_eeg = tf.keras.layers.Concatenate(axis=1)([x0_eeg, x1_eeg, x2_eeg, x3_eeg])
    x_eeg = tf.keras.layers.Concatenate(axis=3)([x_eeg, x_eeg, x_eeg])

    x_eeg = base_model_eeg(x_eeg)
    x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
    x_eeg = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x_eeg)

    x = tf.keras.layers.Concatenate(axis=1)([x, x_eeg])
    x = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs=[inp, inp_mel, inp_eeg], outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.KLDivergence()
    model.compile(loss=loss, optimizer=opt)
    return model




## === cell 5
if not NEEDTRAIN:
    if PLATFORM == "local":
        test_meta = pd.read_csv(
            "./input/hms-harmful-brain-activity-classification/test.csv"
        )
        sample_sub = pd.read_csv(
            "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )
    elif PLATFORM == "kaggle":
        test_meta = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
        sample_sub = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )

    print("Test meta shape", test_meta.shape, "| sample_sub shape", sample_sub.shape)

    test_meta = test_meta.rename({"spectrogram_id": "spec_id"}, axis=1)
    test_meta = test_meta.sort_values(["eeg_id", "spec_id"]).drop_duplicates(
        subset=["eeg_id"], keep="first"
    )

    gen_df = sample_sub[["eeg_id"]].merge(
        test_meta[["eeg_id", "spec_id"]], on="eeg_id", how="left"
    )
    gen_df["min"] = 0
    gen_df["max"] = 0

    if gen_df["spec_id"].isna().any():
        nmiss = int(gen_df["spec_id"].isna().sum())
        raise ValueError(
            f"{nmiss} eeg_id in sample_submission missing spec_id after merge."
        )

    if PLATFORM == "local":
        PATH_SPEC = (
            "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
    else:
        PATH_SPEC = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )

    needed_spec_ids = set(gen_df["spec_id"].astype(int).tolist())
    files_spec = [f"{sid}.parquet" for sid in needed_spec_ids]
    print(f"There are {len(files_spec)} required test spectrogram parquets")

    spectrograms2 = {}
    for i, f in enumerate(files_spec):
        if i % 250 == 0:
            print(i, ", ", end="")
        p = os.path.join(PATH_SPEC, f)
        tmp = pd.read_parquet(p)
        name = int(f.split(".")[0])
        spectrograms2[name] = tmp.iloc[:, 1:].values

    if PLATFORM == "local":
        PATH_EEG = "./input/hms-harmful-brain-activity-classification/test_eegs/"
    else:
        PATH_EEG = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

    needed_eeg_ids = set(gen_df["eeg_id"].astype(int).tolist())
    print(f"\nThere are {len(needed_eeg_ids)} required test eeg parquets")

    eegs2 = {}
    if len(filter_range) == 1:
        if filter_range[0] > 5:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "lowpass")
        else:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "highpass")
    else:
        b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

    files_eeg = [f"{eid}.parquet" for eid in needed_eeg_ids]
    for i, f in enumerate(files_eeg):
        if i % 250 == 0:
            print(i, ", ", end="")
        name = int(f.split(".")[0])
        p = os.path.join(PATH_EEG, f)
        raw_eeg = pd.read_parquet(p)

        time_temp = 0
        time_start = round(time_temp * 200 + (50 - EEG_LENGTH) / 2 * 200)
        time_stop = round(time_temp * 200 + (50 + EEG_LENGTH) / 2 * 200)

        eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(
            drop=True
        )

        list_eeg = []
        for region in BRAIN.keys():
            eeg = np.zeros((len(BRAIN[region]), eeg_default.shape[0]), dtype=np.float32)
            for chan_i, chan in enumerate(BRAIN[region]):
                eeg[chan_i, :] = (
                    eeg_default.loc[:, chan.split("-")[0]]
                    - eeg_default.loc[:, chan.split("-")[1]]
                ).values

            eeg[np.isnan(eeg)] = 0

            if 200 != SFREQ:
                eeg = signal.resample_poly(eeg, SFREQ, 200, axis=1)

            eeg = signal.filtfilt(b, a, eeg, axis=1)
            list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

        list_eeg = np.concatenate(list_eeg, 2)
        eegs2[name] = list_eeg

    missing_eeg = [eid for eid in needed_eeg_ids if eid not in eegs2]
    if len(missing_eeg) > 0:
        raise FileNotFoundError(
            f"Missing {len(missing_eeg)} eeg parquet(s) for required eeg_id. Example: {missing_eeg[:5]}"
        )

    with strategy.scope():
        model = build_model()

    test_gen = DataGenerator(
        gen_df,
        shuffle=False,
        batch_size=32,
        mode="test",
        specs=spectrograms2,
        eegs=eegs2,
    )

    def _find_weight_path_anywhere(base: str) -> str:
        candidates = [base + ".h5", base + ".weights.h5"]
        for fn in candidates:
            pth = os.path.join(LOAD_MODELS_FROM, fn)
            if os.path.exists(pth):
                return pth
        if PLATFORM == "kaggle":
            root = "/kaggle/input"
            for dirpath, dirnames, filenames in os.walk(root):
                bn = os.path.basename(dirpath)
                if bn in {
                    "train_eegs",
                    "test_eegs",
                    "train_spectrograms",
                    "test_spectrograms",
                    "hms-harmful-brain-activity-classification",
                }:
                    continue
                for fn in candidates:
                    if fn in filenames:
                        return os.path.join(dirpath, fn)
        raise FileNotFoundError(
            f"Could not find weights for base={base} under {LOAD_MODELS_FROM} or /kaggle/input"
        )

    preds = []
    any_loaded = False
    for i in range(5):
        print(f"\nFold {i + 1}")
        try:
            wpath = _find_weight_path_anywhere(f"EB2_v{VER}_f{i}")
            print("Loading:", wpath)
            model.load_weights(wpath)
            pred_i = model.predict(test_gen, verbose=1)
            preds.append(pred_i)
            any_loaded = True
        except FileNotFoundError as e:
            print("WARNING:", str(e))
            print("Skipping this fold.")

    if any_loaded:
        pred = np.mean(preds, axis=0)
        print("\nTest preds shape", pred.shape)
    else:
        print(
            "WARNING: No weights found for any fold. Falling back to uniform predictions."
        )
        pred = np.full(
            (len(gen_df), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32
        )

    if pred.shape[0] != len(sample_sub):
        raise ValueError(
            f"Prediction length mismatch: pred has {pred.shape[0]} rows but sample_submission has {len(sample_sub)} rows."
        )

    sub = sample_sub.copy()
    sub[list(TARGETS)] = pred.astype(np.float32)

    p = sub[list(TARGETS)].to_numpy(dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    sub[list(TARGETS)] = p.astype(np.float32)

    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(
        "Row-sum check (min/mean/max):",
        float(sub[list(TARGETS)].sum(axis=1).min()),
        float(sub[list(TARGETS)].sum(axis=1).mean()),
        float(sub[list(TARGETS)].sum(axis=1).max()),
    )
    sub.head()

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3556595801.py in <cell line: 0>()
     71             b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "highpass")
     72     else:
---> 73         b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")
     74 
     75     files_eeg = [f"{eid}.parquet" for eid in needed_eeg_ids]

NameError: name 'signal' is not defined
