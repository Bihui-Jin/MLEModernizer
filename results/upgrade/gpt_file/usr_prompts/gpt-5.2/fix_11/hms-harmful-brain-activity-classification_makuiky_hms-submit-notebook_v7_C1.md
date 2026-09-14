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
pillow==11.3.0
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.9948561058052816

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.4778) has done: 'I fix the protobuf/TensorFlow startup crash by removing the environment override that forces the pure-Python protobuf implementation (it’s incompatible with protobuf 6.x and causes the `MessageFactory.GetPrototype` error). Then I fix the batching loop bug caused by `.loc[eeg_id]` returning a DataFrame (duplicate `eeg_id`s), by selecting a single deterministic row per `eeg_id` before calling `preprocess_data`. Finally, I ensure `pdsubmit` is always defined and the script always writes a valid `submission.csv` with probabilities clipped and renormalized to sum to 1 per row.'
- What this solution (achieved 1.56406) has done: 'I fix the TensorFlow import crash by removing the incompatible protobuf environment override that forces the missing C++ `_message` module, allowing `import tensorflow as tf` to succeed in this Kaggle image. Then I make sure `tf`/`keras` are always defined before any later cells use them, which removes the downstream `NameError`s and ensures the model and preprocessing run. Finally, I guard the submission-building step so `pdsubmit` is always created and a valid `submission.csv` is always written with clipped/renormalized probabilities that sum to 1 per row (as required by the KL metric checker).'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf startup crash by forcing protobuf to use the C++ implementation (the default) and explicitly preventing the pure-Python fallback that triggers the `MessageFactory.GetPrototype` error in this Kaggle image. Then I keep the existing inference pipeline and submission-writing logic intact, only adding a small, metric-aligned post-processing calibration (blend with uniform) to reduce extreme/confident probabilities that can hurt KL-divergence. Finally, I ensure the script runs end-to-end under Python 3.12/TensorFlow 2.18 and always writes a valid `submission.csv` with rows summing to 1.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import sys
import gc
import io
import subprocess
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass
try:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass

plt.ioff()



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/3910471064.py in <cell line: 0>()
     23 from PIL import Image
     24 
---> 25 import tensorflow as tf
     26 from tensorflow import keras
     27 

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
EEG_TEST_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
SPEC_TEST_PATH = (
    "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
)

EEG_IMG_TEST_PATH = "/kaggle/working/test_eegs_img/"
SPEC_IMG_TEST_PATH = "/kaggle/working/test_spec_img/"
EEG_IMG_TRAIN_PATH = "/kaggle/working/train_eegs_img/"
SPEC_IMG_TRAIN_PATH = "/kaggle/working/train_spec_img/"

for p in [
    EEG_IMG_TEST_PATH,
    SPEC_IMG_TEST_PATH,
    EEG_IMG_TRAIN_PATH,
    SPEC_IMG_TRAIN_PATH,
]:
    os.makedirs(p, exist_ok=True)

META_TEST = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
SAMPLE_SUB = (
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)



## === cell 2
TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



## === cell 3
eeg_zone = {
    "Cz-Pz": ["Cz", "Pz"],
    "Fz-Cz": ["Fz", "Cz"],
    "P4-O2": ["P4", "O2"],
    "C4-P4": ["C4", "P4"],
    "F4-C4": ["F4", "C4"],
    "Fp2-F4": ["Fp2", "F4"],
    "P3-O1": ["P3", "O1"],
    "C3-P3": ["C3", "P3"],
    "F3-C3": ["F3", "C3"],
    "Fp1-F3": ["Fp1", "F3"],
    "T6-O2": ["T6", "O2"],
    "T4-T6": ["T4", "T6"],
    "F8-T4": ["F8", "T4"],
    "Fp2-F8": ["Fp2", "F8"],
    "T5-O1": ["T5", "O1"],
    "T3-T5": ["T3", "T5"],
    "F7-T3": ["F7", "T3"],
    "Fp1-F7": ["Fp1", "F7"],
}



## === cell 4
_EEG_H = 300
_EEG_W = 300


def _draw_polyline(img, xs, ys, thickness=1, color=0):
    """Very fast raster line draw by stamping points (no antialiasing). Deterministic."""
    h, w = img.shape
    xs = xs.astype(np.int32, copy=False)
    ys = ys.astype(np.int32, copy=False)
    m = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs = xs[m]
    ys = ys[m]
    if xs.size == 0:
        return
    if thickness <= 1:
        img[ys, xs] = color
        return
    r = thickness // 2
    for dy in range(-r, r + 1):
        y2 = ys + dy
        m2 = (y2 >= 0) & (y2 < h)
        img[y2[m2], xs[m2]] = color


def generate_eeg(
    eegid,
    input_path,
    eeg_zone=eeg_zone,
    linewidth=0.2,
    eeg_out=EEG_IMG_TEST_PATH,
    return_pil=True,
):
    cols = []
    for a, b in eeg_zone.values():
        cols.append(a)
        cols.append(b)
    cols = list(dict.fromkeys(cols))  # preserve order, unique

    eeg = pd.read_parquet(f"{input_path}{eegid}.parquet", columns=cols)
    eeg_np = eeg.to_numpy(dtype=np.float32, copy=False)
    n = eeg_np.shape[0]
    if n <= 0:
        rgb = np.zeros((_EEG_H, _EEG_W, 3), dtype=np.uint8)
        return Image.fromarray(rgb, mode="RGB") if return_pil else None

    xpix = np.round(np.linspace(0, _EEG_W - 1, n)).astype(np.int32)

    col2i = {c: i for i, c in enumerate(eeg.columns)}
    canvas = np.full((_EEG_H, _EEG_W), 255, dtype=np.uint8)

    relpos = 0.0
    cycles = 0

    y_scale = 2.0  # uV per pixel
    y_center = _EEG_H // 2

    thickness = 1  # linewidth=0.2 effectively 1px at this resolution

    for _, (a, b) in eeg_zone.items():
        y = eeg_np[:, col2i[a]] - eeg_np[:, col2i[b]] + relpos
        ypix = (y_center - (y / y_scale)).astype(np.int32, copy=False)
        _draw_polyline(canvas, xpix, ypix, thickness=thickness, color=0)

        if cycles == 1 or cycles == 5 or cycles == 9 or cycles == 13:
            relpos += 200.0
        else:
            relpos += 40.0
        cycles += 1

    rgb = np.stack([canvas, canvas, canvas], axis=-1)

    if return_pil:
        return Image.fromarray(rgb, mode="RGB")
    else:
        save_path = f"{eeg_out}{eegid}.jpeg"
        Image.fromarray(rgb, mode="RGB").save(save_path, format="JPEG", quality=95)
        return save_path




## === cell 5
spec_zones = ["LL", "RL", "LP", "RP"]

_SPEC_H = 300
_SPEC_W = 300

_turbo = (plt.get_cmap("turbo")(np.linspace(0, 1, 256))[:, :3] * 255.0).astype(np.uint8)


def _resize_nearest(img, out_h, out_w):
    """Nearest-neighbor resize for speed (deterministic)."""
    h, w = img.shape[:2]
    if h == out_h and w == out_w:
        return img
    y_idx = (np.linspace(0, h - 1, out_h)).astype(np.int32)
    x_idx = (np.linspace(0, w - 1, out_w)).astype(np.int32)
    return img[y_idx[:, None], x_idx[None, :]]


def _apply_turbo(data_2d, vmax):
    if data_2d.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    x = data_2d.astype(np.float32, copy=False)
    x = np.clip(x / float(vmax), 0.0, 1.0)
    idx = (x * 255.0 + 0.5).astype(np.int32)
    return _turbo[idx]


def generate_spectrogram(
    specid,
    input_path,
    spec_out=SPEC_IMG_TEST_PATH,
    spec_zones=spec_zones,
    output_filter=0.5,
    return_pil=True,
):
    spec = pd.read_parquet(f"{input_path}{specid}.parquet")
    spec = spec.fillna(0)

    cols = [c for c in spec.columns if c != "time"]

    zones = []
    freqs = []
    for c in cols:
        z, f = c.split("_", 1)
        zones.append(z)
        freqs.append(float(f))
    zones = np.array(zones, dtype=object)
    freqs = np.array(freqs, dtype=np.float32)

    values = spec[cols].to_numpy(dtype=np.float32, copy=False)  # shape (T, C)

    row_imgs = []
    for zone in spec_zones:
        m = zones == zone
        if not np.any(m):
            row_imgs.append(np.zeros((_SPEC_H // 4, _SPEC_W, 3), dtype=np.uint8))
            continue

        v = values[:, m]  # (T, Fcols)
        f = freqs[m]
        order = np.argsort(f, kind="mergesort")  # stable/deterministic
        v = v[:, order]
        data = v.T  # (F, T)

        vmax = float(np.max(data)) * float(output_filter) if data.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0

        rgb = _apply_turbo(data, vmax)  # (F, T, 3)
        rgb = _resize_nearest(rgb, _SPEC_H // 4, _SPEC_W)
        row_imgs.append(rgb.astype(np.uint8, copy=False))

    full = np.vstack(row_imgs)
    full = _resize_nearest(full, _SPEC_H, _SPEC_W)

    if return_pil:
        return Image.fromarray(full, mode="RGB")
    else:
        save_path = f"{spec_out}{specid}.jpeg"
        Image.fromarray(full, mode="RGB").save(save_path, format="JPEG", quality=95)
        return save_path




## === cell 6
metadata = pd.read_csv(META_TEST)
train_metadata = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
)
sample_sub = pd.read_csv(SAMPLE_SUB)

metadata["eeg_id"] = metadata["eeg_id"].astype(np.int64)
sample_sub["eeg_id"] = sample_sub["eeg_id"].astype(np.int64)




## === cell 7
def drop_images(paths):
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass




## === cell 8
def _img_to_tensor_uint8(img_pil):
    arr = np.array(img_pil)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return tf.convert_to_tensor(arr, dtype=tf.uint8)


_EEG_CACHE = {}
_SPEC_CACHE = {}


def preprocess_data(metadata, row, train=False):
    if train:
        eegid = int(metadata.loc[row].eeg_id)
        specid = int(metadata.loc[row].spectrogram_id)
        eeg_input = (
            "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
        )
        spec_input = "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
        eeg_out = EEG_IMG_TRAIN_PATH
        spec_out = SPEC_IMG_TRAIN_PATH
        output_filter = 0.7
    else:
        eegid = int(metadata.loc[row].eeg_id)
        specid = int(metadata.loc[row].spectrogram_id)
        eeg_input = EEG_TEST_PATH
        spec_input = SPEC_TEST_PATH
        eeg_out = EEG_IMG_TEST_PATH
        spec_out = SPEC_IMG_TEST_PATH
        output_filter = 0.7

    eeg_img_tr = _EEG_CACHE.get(eegid)
    if eeg_img_tr is None:
        eeg_img = generate_eeg(
            eegid=eegid,
            input_path=eeg_input,
            eeg_out=eeg_out,
            return_pil=True,
        )
        eeg_img_tr = tf.expand_dims(_img_to_tensor_uint8(eeg_img), axis=0)
        _EEG_CACHE[eegid] = eeg_img_tr

    spec_img_tr = _SPEC_CACHE.get(specid)
    if spec_img_tr is None:
        spec_img = generate_spectrogram(
            specid=specid,
            input_path=spec_input,
            spec_out=spec_out,
            output_filter=output_filter,
            return_pil=True,
        )
        spec_img_tr = tf.expand_dims(_img_to_tensor_uint8(spec_img), axis=0)
        _SPEC_CACHE[specid] = spec_img_tr

    return eeg_img_tr, spec_img_tr, eegid




## === cell 9
def build_fallback_model():
    eeg_in = keras.Input(shape=(None, None, 3), dtype=tf.uint8, name="eeg_img")
    spec_in = keras.Input(shape=(None, None, 3), dtype=tf.uint8, name="spec_img")

    def to_float_and_pool(x):
        x = tf.cast(x, tf.float32) / 255.0
        x = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=False)  # (batch,)
        x = tf.expand_dims(x, axis=-1)  # (batch, 1)
        return x

    eeg_feat = keras.layers.Lambda(to_float_and_pool, name="eeg_pool")(eeg_in)
    spec_feat = keras.layers.Lambda(to_float_and_pool, name="spec_pool")(spec_in)

    x = keras.layers.Concatenate(name="concat_feats")([eeg_feat, spec_feat])  # (b,2)

    def logits_from_feats(z):
        eeg_m = z[:, 0:1]
        spec_m = z[:, 1:2]

        s1 = 1.0 - eeg_m  # more ink in EEG trace -> lower mean -> higher s1
        s2 = 1.0 - spec_m  # darker spec -> higher s2
        s3 = tf.abs(eeg_m - spec_m)  # disagreement/contrast
        s4 = eeg_m
        s5 = spec_m
        s6 = 0.5 * (eeg_m + spec_m)

        feats6 = tf.concat([s1, s2, s3, s4, s5, s6], axis=1)  # (batch,6)

        scale = tf.constant([2.0, 1.8, 1.6, 1.3, 1.3, 1.2], dtype=tf.float32)[None, :]
        bias = tf.constant([0.10, 0.05, 0.02, 0.00, 0.00, 0.00], dtype=tf.float32)[
            None, :
        ]
        logits = feats6 * scale + bias
        return logits

    logits = keras.layers.Lambda(logits_from_feats, name="fixed_logits")(x)
    out = keras.layers.Softmax(name="softmax_out")(logits)

    model = keras.Model(inputs=[eeg_in, spec_in], outputs=out)
    return model


model = build_fallback_model()
model.compile(run_eagerly=False)
_ = model.predict(
    [tf.zeros((1, 8, 8, 3), tf.uint8), tf.zeros((1, 8, 8, 3), tf.uint8)], verbose=0
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/103690902.py in <cell line: 0>()
     41 
     42 
---> 43 model = build_fallback_model()
     44 model.compile(run_eagerly=False)
     45 _ = model.predict(

/tmp/ipykernel_55/103690902.py in build_fallback_model()
      1 def build_fallback_model():
----> 2     eeg_in = keras.Input(shape=(None, None, 3), dtype=tf.uint8, name="eeg_img")
      3     spec_in = keras.Input(shape=(None, None, 3), dtype=tf.uint8, name="spec_img")
      4 
      5     def to_float_and_pool(x):

NameError: name 'keras' is not defined

## === cell 10
submission = {
    "eeg_id": [],
    "seizure_vote": [],
    "lpd_vote": [],
    "gpd_vote": [],
    "lrda_vote": [],
    "grda_vote": [],
    "other_vote": [],
}
iter_dict = TARGET_COLS

BATCH = 64

eeg_batch = []
spec_batch = []
id_batch = []


def _flush_batch():
    if not id_batch:
        return
    eeg_x = tf.concat(eeg_batch, axis=0)
    spec_x = tf.concat(spec_batch, axis=0)
    preds = model.predict([eeg_x, spec_x], verbose=0).astype(np.float64)

    alpha = 0.25  # small shrinkage toward uniform
    uniform = np.full_like(preds, 1.0 / preds.shape[1], dtype=np.float64)
    preds = (1.0 - alpha) * preds + alpha * uniform

    preds = np.clip(preds, 1e-15, 1.0)
    preds = preds / preds.sum(axis=1, keepdims=True)

    submission["eeg_id"].extend([int(x) for x in id_batch])
    for i, illness in enumerate(iter_dict):
        submission[illness].extend(preds[:, i].astype(np.float64, copy=False).tolist())

    eeg_batch.clear()
    spec_batch.clear()
    id_batch.clear()


meta_one = metadata.drop_duplicates(subset=["eeg_id"], keep="first").set_index(
    "eeg_id", drop=False
)

for eeg_id in sample_sub["eeg_id"].tolist():
    eeg_id = int(eeg_id)
    if eeg_id not in meta_one.index:
        eeg_img = tf.zeros((1, _EEG_H, _EEG_W, 3), tf.uint8)
        spec_img = tf.zeros((1, _SPEC_H, _SPEC_W, 3), tf.uint8)
        eeg_batch.append(eeg_img)
        spec_batch.append(spec_img)
        id_batch.append(eeg_id)
    else:
        tmp = meta_one.loc[[eeg_id]].reset_index(drop=True)
        eeg_img, spec_img, eeg_id2 = preprocess_data(tmp, 0, train=False)

        eeg_batch.append(eeg_img)
        spec_batch.append(spec_img)
        id_batch.append(int(eeg_id2))

    if len(id_batch) >= BATCH:
        _flush_batch()

_flush_batch()

pdsubmit = pd.DataFrame(submission)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1459557341.py in <cell line: 0>()
     56     else:
     57         tmp = meta_one.loc[[eeg_id]].reset_index(drop=True)
---> 58         eeg_img, spec_img, eeg_id2 = preprocess_data(tmp, 0, train=False)
     59 
     60         eeg_batch.append(eeg_img)

/tmp/ipykernel_55/2694013644.py in preprocess_data(metadata, row, train)
     40             return_pil=True,
     41         )
---> 42         eeg_img_tr = tf.expand_dims(_img_to_tensor_uint8(eeg_img), axis=0)
     43         _EEG_CACHE[eegid] = eeg_img_tr
     44 

NameError: name 'tf' is not defined

## === cell 11
if "pdsubmit" not in globals() or pdsubmit is None or len(pdsubmit) == 0:
    pdsubmit = sample_sub.copy()
    pdsubmit[TARGET_COLS] = 1.0 / len(TARGET_COLS)

pdsubmit = pdsubmit.groupby("eeg_id", as_index=False)[TARGET_COLS].mean()

pdsubmit = sample_sub[["eeg_id"]].merge(pdsubmit, on="eeg_id", how="left")

for c in TARGET_COLS:
    if c not in pdsubmit.columns:
        pdsubmit[c] = np.nan
pdsubmit[TARGET_COLS] = pdsubmit[TARGET_COLS].fillna(1.0 / len(TARGET_COLS))

vals = pdsubmit[TARGET_COLS].to_numpy(dtype=np.float64)
vals = np.clip(vals, 1e-15, 1.0)
vals = vals / vals.sum(axis=1, keepdims=True)
pdsubmit[TARGET_COLS] = vals

assert len(pdsubmit) == len(sample_sub), (len(pdsubmit), len(sample_sub))
assert (
    pdsubmit["eeg_id"].to_numpy().tolist() == sample_sub["eeg_id"].to_numpy().tolist()
)

pdsubmit.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", pdsubmit.shape)
print(pdsubmit.head())
print("Row-sum stats:", float(vals.sum(axis=1).min()), float(vals.sum(axis=1).max()))
