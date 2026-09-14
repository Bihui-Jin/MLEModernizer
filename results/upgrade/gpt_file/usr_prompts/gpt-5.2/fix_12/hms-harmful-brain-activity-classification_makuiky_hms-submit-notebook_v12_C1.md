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

0.9209651751737288

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import sys
import gc
import io
import math
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tensorflow as tf
from tensorflow import keras

np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
EEG_TEST_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
SPEC_TEST_PATH = (
    "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
)

if not os.path.exists("/kaggle/working/test_eegs_img/"):
    os.makedirs("/kaggle/working/test_eegs_img/")
EEG_IMG_TEST_PATH = "/kaggle/working/test_eegs_img/"

if not os.path.exists("/kaggle/working/test_spec_img/"):
    os.makedirs("/kaggle/working/test_spec_img/")
SPEC_IMG_TEST_PATH = "/kaggle/working/test_spec_img/"

META_TEST = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
SAMPLE_SUB = (
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)



## === cell 2
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



## === cell 3
import pyarrow.parquet as pq


def _read_parquet_df(path, columns=None):
    return pq.read_table(path, columns=columns).to_pandas(types_mapper=None)


def _draw_polyline_uint8(img, xs, ys, color=(0, 0, 0), linewidth=1):
    h, w, _ = img.shape
    xs = np.asarray(xs, dtype=np.int32)
    ys = np.asarray(ys, dtype=np.int32)
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    if xs.size < 2:
        return

    x0 = xs[:-1]
    y0 = ys[:-1]
    x1 = xs[1:]
    y1 = ys[1:]

    dx = np.abs(x1 - x0)
    dy = np.abs(y1 - y0)
    n = np.maximum(np.maximum(dx, dy), 1).astype(np.int32)

    counts = n + 1
    total = int(counts.sum())
    if total <= 0:
        return

    seg_ids = np.repeat(np.arange(n.size, dtype=np.int32), counts)

    start = np.cumsum(counts) - counts
    t_idx = (np.arange(total, dtype=np.int32) - start[seg_ids]).astype(np.float32)
    denom = n[seg_ids].astype(np.float32)
    t = t_idx / denom

    xi = (
        x0[seg_ids].astype(np.float32)
        + (x1[seg_ids] - x0[seg_ids]).astype(np.float32) * t
    ).astype(np.int32)
    yi = (
        y0[seg_ids].astype(np.float32)
        + (y1[seg_ids] - y0[seg_ids]).astype(np.float32) * t
    ).astype(np.int32)

    xi = np.clip(xi, 0, w - 1)
    yi = np.clip(yi, 0, h - 1)

    if linewidth <= 1:
        img[yi, xi, :] = color
    else:
        r = int(linewidth // 2)
        for offy in range(-r, r + 1):
            yj = np.clip(yi + offy, 0, h - 1)
            img[yj, xi, :] = color
        for offx in range(-r, r + 1):
            xj = np.clip(xi + offx, 0, w - 1)
            img[yi, xj, :] = color


from collections import OrderedDict


class _LRU(OrderedDict):
    def __init__(self, maxsize=256):
        super().__init__()
        self.maxsize = int(maxsize)

    def get(self, key, default=None):
        if key in self:
            self.move_to_end(key)
            return super().get(key)
        return default

    def put(self, key, value):
        self[key] = value
        self.move_to_end(key)
        if len(self) > self.maxsize:
            self.popitem(last=False)


_EEG_CACHE = _LRU(maxsize=4096)
_SPEC_PARSED_CACHE = _LRU(maxsize=4096)

_XS_CACHE = {}  # keyed by n, tiny
_EEG_ZONE_ITEMS = list(eeg_zone.items())
_EEG_COLS_NEEDED = sorted({ch for pair in eeg_zone.values() for ch in pair})


def generate_eeg(
    eegid,
    input_path=EEG_TEST_PATH,
    eeg_zone=eeg_zone,
    linewidth=0.2,
    eeg_out=EEG_IMG_TEST_PATH,
):
    key = (input_path, int(eegid))
    eeg = _EEG_CACHE.get(key)
    if eeg is None:
        eeg = _read_parquet_df(
            f"{input_path}{int(eegid)}.parquet", columns=_EEG_COLS_NEEDED
        )
        _EEG_CACHE.put(key, eeg)

    H = W = 300
    img = np.full((H, W, 3), 255, dtype=np.uint8)

    n = len(eeg)
    xs = _XS_CACHE.get(n)
    if xs is None:
        xs = (np.arange(n, dtype=np.float32) / (200.0 * 50.0) * (W - 1)).astype(
            np.int32
        )
        _XS_CACHE[n] = xs

    cicles = 0
    relpos = 0.0

    _items = _EEG_ZONE_ITEMS
    for _, (a, b) in _items:
        ya = eeg[a].to_numpy(dtype=np.float32, copy=False)
        yb = eeg[b].to_numpy(dtype=np.float32, copy=False)
        y = (ya - yb) + relpos

        y_scaled = (H * 0.5 - (y * 0.25)).astype(np.int32)
        _draw_polyline_uint8(img, xs, y_scaled, color=(0, 0, 0), linewidth=1)

        if cicles == 1 or cicles == 5 or cicles == 9 or cicles == 13:
            relpos += 200.0
        else:
            relpos += 40.0
        cicles += 1

    return img




## === cell 4
spec_zones = ["LL", "RL", "LP", "RP"]

_TURBO_LUT = np.array(
    [
        [48, 18, 59],
        [50, 21, 67],
        [51, 24, 74],
        [52, 27, 81],
        [53, 30, 88],
        [54, 33, 95],
        [55, 36, 102],
        [56, 39, 109],
        [57, 42, 115],
        [58, 45, 122],
        [59, 48, 129],
        [60, 51, 136],
        [61, 54, 142],
        [62, 57, 149],
        [63, 60, 156],
        [64, 63, 162],
        [65, 66, 169],
        [66, 69, 175],
        [67, 72, 182],
        [68, 75, 188],
        [69, 78, 195],
        [70, 81, 201],
        [71, 84, 208],
        [72, 87, 214],
        [73, 90, 221],
        [74, 93, 227],
        [75, 96, 234],
        [76, 99, 240],
        [77, 102, 247],
        [78, 105, 253],
        [79, 108, 255],
        [80, 111, 255],
        [81, 114, 255],
        [82, 117, 255],
        [83, 120, 255],
        [84, 123, 255],
        [85, 126, 255],
        [86, 129, 255],
        [87, 132, 255],
        [88, 135, 255],
        [89, 138, 255],
        [90, 141, 255],
        [91, 144, 255],
        [92, 147, 255],
        [93, 150, 255],
        [94, 153, 255],
        [95, 156, 255],
        [96, 159, 255],
        [97, 162, 255],
        [98, 165, 255],
        [99, 168, 255],
        [100, 171, 255],
        [101, 174, 255],
        [102, 177, 255],
        [103, 180, 255],
        [104, 183, 255],
        [105, 186, 255],
        [106, 189, 255],
        [107, 192, 255],
        [108, 195, 255],
        [109, 198, 255],
        [110, 201, 255],
        [111, 204, 255],
        [112, 207, 255],
        [113, 210, 255],
        [114, 213, 255],
        [115, 216, 255],
        [116, 219, 255],
        [117, 222, 255],
        [118, 225, 255],
        [119, 228, 255],
        [120, 231, 255],
        [121, 234, 255],
        [122, 237, 255],
        [123, 240, 255],
        [124, 243, 255],
        [125, 246, 255],
        [126, 249, 255],
        [127, 252, 255],
        [128, 255, 255],
        [130, 255, 253],
        [132, 255, 250],
        [134, 255, 248],
        [136, 255, 245],
        [138, 255, 243],
        [140, 255, 240],
        [142, 255, 238],
        [144, 255, 235],
        [146, 255, 233],
        [148, 255, 230],
        [150, 255, 228],
        [152, 255, 225],
        [154, 255, 223],
        [156, 255, 220],
        [158, 255, 218],
        [160, 255, 215],
        [162, 255, 213],
        [164, 255, 210],
        [166, 255, 208],
        [168, 255, 205],
        [170, 255, 203],
        [172, 255, 200],
        [174, 255, 198],
        [176, 255, 195],
        [178, 255, 193],
        [180, 255, 190],
        [182, 255, 188],
        [184, 255, 185],
        [186, 255, 183],
        [188, 255, 180],
        [190, 255, 178],
        [192, 255, 175],
        [194, 255, 173],
        [196, 255, 170],
        [198, 255, 168],
        [200, 255, 165],
        [202, 255, 163],
        [204, 255, 160],
        [206, 255, 158],
        [208, 255, 155],
        [210, 255, 153],
        [212, 255, 150],
        [214, 255, 148],
        [216, 255, 145],
        [218, 255, 143],
        [220, 255, 140],
        [222, 255, 138],
        [224, 255, 135],
        [226, 255, 133],
        [228, 255, 130],
        [230, 255, 128],
        [232, 255, 125],
        [234, 255, 123],
        [236, 255, 120],
        [238, 255, 118],
        [240, 255, 115],
        [242, 255, 113],
        [244, 255, 110],
        [246, 255, 108],
        [248, 255, 105],
        [250, 255, 103],
        [252, 255, 100],
        [254, 255, 98],
        [255, 254, 95],
        [255, 252, 93],
        [255, 250, 90],
        [255, 248, 88],
        [255, 246, 85],
        [255, 244, 83],
        [255, 242, 80],
        [255, 240, 78],
        [255, 238, 75],
        [255, 236, 73],
        [255, 234, 70],
        [255, 232, 68],
        [255, 230, 65],
        [255, 228, 63],
        [255, 226, 60],
        [255, 224, 58],
        [255, 222, 55],
        [255, 220, 53],
        [255, 218, 50],
        [255, 216, 48],
        [255, 214, 45],
        [255, 212, 43],
        [255, 210, 40],
        [255, 208, 38],
        [255, 206, 35],
        [255, 204, 33],
        [255, 202, 30],
        [255, 200, 28],
        [255, 198, 25],
        [255, 196, 23],
        [255, 194, 20],
        [255, 192, 18],
        [255, 190, 15],
        [255, 188, 13],
        [255, 186, 10],
        [255, 184, 8],
        [255, 182, 5],
        [255, 180, 3],
        [255, 178, 0],
        [255, 176, 0],
        [255, 174, 0],
        [255, 172, 0],
        [255, 170, 0],
        [255, 168, 0],
        [255, 166, 0],
        [255, 164, 0],
        [255, 162, 0],
        [255, 160, 0],
        [255, 158, 0],
        [255, 156, 0],
        [255, 154, 0],
        [255, 152, 0],
        [255, 150, 0],
        [255, 148, 0],
        [255, 146, 0],
        [255, 144, 0],
        [255, 142, 0],
        [255, 140, 0],
        [255, 138, 0],
        [255, 136, 0],
        [255, 134, 0],
        [255, 132, 0],
        [255, 130, 0],
        [255, 128, 0],
        [255, 126, 0],
        [255, 124, 0],
        [255, 122, 0],
        [255, 120, 0],
        [255, 118, 0],
        [255, 116, 0],
        [255, 114, 0],
        [255, 112, 0],
        [255, 110, 0],
        [255, 108, 0],
        [255, 106, 0],
        [255, 104, 0],
        [255, 102, 0],
        [255, 100, 0],
        [255, 98, 0],
        [255, 96, 0],
        [255, 94, 0],
        [255, 92, 0],
        [255, 90, 0],
        [255, 88, 0],
        [255, 86, 0],
        [255, 84, 0],
        [255, 82, 0],
        [255, 80, 0],
        [255, 78, 0],
        [255, 76, 0],
        [255, 74, 0],
        [255, 72, 0],
        [255, 70, 0],
        [255, 68, 0],
        [255, 66, 0],
        [255, 64, 0],
        [255, 62, 0],
        [255, 60, 0],
        [255, 58, 0],
        [255, 56, 0],
        [255, 54, 0],
        [255, 52, 0],
        [255, 50, 0],
        [255, 48, 0],
        [255, 46, 0],
        [255, 44, 0],
        [255, 42, 0],
        [255, 40, 0],
        [255, 38, 0],
        [255, 36, 0],
        [255, 34, 0],
        [255, 32, 0],
        [255, 30, 0],
        [255, 28, 0],
        [255, 26, 0],
        [255, 24, 0],
        [255, 22, 0],
        [255, 20, 0],
        [255, 18, 0],
        [255, 16, 0],
        [255, 14, 0],
        [255, 12, 0],
        [255, 10, 0],
        [255, 8, 0],
        [255, 6, 0],
        [255, 4, 0],
        [255, 2, 0],
        [255, 0, 0],
        [253, 0, 0],
    ],
    dtype=np.uint8,
)
if _TURBO_LUT.shape[0] != 256:
    _TURBO_LUT = np.vstack(
        [_TURBO_LUT, np.tile(_TURBO_LUT[-1:], (256 - _TURBO_LUT.shape[0], 1))]
    )[:256]


def _apply_colormap_uint8(gray_uint8):
    return _TURBO_LUT[gray_uint8]


_RESIZE_INDEX_CACHE = {}


def _get_resize_indices(src_h, src_w, dst_h, dst_w):
    key = (int(src_h), int(src_w), int(dst_h), int(dst_w))
    out = _RESIZE_INDEX_CACHE.get(key)
    if out is None:
        yy = (np.linspace(0, src_h - 1, dst_h)).astype(np.int32)
        xx = (np.linspace(0, src_w - 1, dst_w)).astype(np.int32)
        out = (yy, xx)
        _RESIZE_INDEX_CACHE[key] = out
    return out


_SPEC_COLS_CACHE = {}


def _parse_spectrogram_df(spec_df):
    cols_key = tuple(spec_df.columns)
    cached = _SPEC_COLS_CACHE.get(cols_key)
    if cached is None:
        feat_cols = [c for c in spec_df.columns if c != "time"]
        cols = np.asarray(feat_cols, dtype=object)
        zones = np.char.partition(cols.astype(str), "_")[:, 0].astype(object)
        zone_indices = {}
        for zone in spec_zones:
            zone_indices[zone] = np.where(zones == zone)[0]
        cached = (feat_cols, zone_indices)
        _SPEC_COLS_CACHE[cols_key] = cached
    else:
        feat_cols, zone_indices = cached

    m = spec_df[feat_cols].to_numpy(dtype=np.float32, copy=False).T
    if np.isnan(m).any():
        m = np.nan_to_num(m, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

    out = {}
    for zone in spec_zones:
        idx = zone_indices[zone]
        out[zone] = None if idx.size == 0 else m[idx, :]
    return out


def generate_spectrogram(
    specid,
    input_path=SPEC_TEST_PATH,
    spec_out=SPEC_IMG_TEST_PATH,
    spec_zones=spec_zones,
    output_filter=0.5,
):
    key = (input_path, int(specid))
    parsed = _SPEC_PARSED_CACHE.get(key)
    if parsed is None:
        spec = _read_parquet_df(f"{input_path}{int(specid)}.parquet", columns=None)
        parsed = _parse_spectrogram_df(spec)
        _SPEC_PARSED_CACHE.put(key, parsed)

    H = W = 300
    nrows = len(spec_zones)
    panel_h = H // nrows
    img = np.full((panel_h * nrows, W, 3), 255, dtype=np.uint8)

    for i, zone in enumerate(spec_zones):
        m = parsed.get(zone, None)
        if m is None or m.size == 0:
            continue

        vmax = float(np.max(m)) * float(output_filter)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0

        gray = (np.clip(m / vmax, 0.0, 1.0) * 255.0).astype(np.uint8)

        src_h, src_w = gray.shape
        yy, xx = _get_resize_indices(src_h, src_w, panel_h, W)
        resized = gray[yy[:, None], xx[None, :]]

        rgb = _apply_colormap_uint8(resized)
        y0 = i * panel_h
        img[y0 : y0 + panel_h, :, :] = rgb

    return img




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2417495190.py in <cell line: 0>()
    283 if _TURBO_LUT.shape[0] != 256:
    284     _TURBO_LUT = np.vstack(
--> 285         [_TURBO_LUT, np.tile(_TURBO_LUT[-1:], (256 - _TURBO_LUT.shape[0], 1))]
    286     )[:256]
    287 

/usr/local/lib/python3.11/dist-packages/numpy/lib/shape_base.py in tile(A, reps)
   1270         for dim_in, nrep in zip(c.shape, tup):
   1271             if nrep != 1:
-> 1272                 c = c.reshape(-1, n).repeat(nrep, 0)
   1273             n //= dim_in
   1274     return c.reshape(shape_out)

ValueError: negative dimensions are not allowed

## === cell 5
metadata = pd.read_csv(META_TEST)
train_metadata = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
)

sample_sub = pd.read_csv(SAMPLE_SUB)
TARGET_COLS = [c for c in sample_sub.columns if c != "eeg_id"]
assert TARGET_COLS == [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

_meta_eeg_ids = metadata["eeg_id"].to_numpy()
_meta_spec_ids = metadata["spectrogram_id"].to_numpy()

_train_eeg_ids = train_metadata["eeg_id"].to_numpy()
_train_spec_ids = train_metadata["spectrogram_id"].to_numpy()




## === cell 6
def drop_images(paths):
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass




## === cell 7
def preprocess_data(row, train=False):
    if train:
        eeg_id = int(_train_eeg_ids[row])
        spec_id = int(_train_spec_ids[row])
        eeg_arr = generate_eeg(
            eegid=eeg_id,
            input_path="/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/",
        )
        spec_arr = generate_spectrogram(
            specid=spec_id,
            output_filter=0.7,
            input_path="/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/",
        )
    else:
        eeg_id = int(_meta_eeg_ids[row])
        spec_id = int(_meta_spec_ids[row])
        eeg_arr = generate_eeg(eegid=eeg_id)
        spec_arr = generate_spectrogram(specid=spec_id, output_filter=0.7)

    return eeg_arr, spec_arr, eeg_id


@tf.function(reduce_retracing=True)
def _resize_and_scale(eeg_np, spec_np):
    eeg = tf.cast(eeg_np, tf.float32) / 255.0
    spec = tf.cast(spec_np, tf.float32) / 255.0
    eeg = tf.image.resize(eeg, [224, 224])
    spec = tf.image.resize(spec, [240, 240])
    return eeg, spec


def _batch_to_model_inputs(eeg_uint8_list, spec_uint8_list):
    eeg_np = np.stack(eeg_uint8_list, axis=0).astype(np.uint8, copy=False)
    spec_np = np.stack(spec_uint8_list, axis=0).astype(np.uint8, copy=False)
    eeg, spec = _resize_and_scale(eeg_np, spec_np)
    return eeg, spec




## === cell 8
class UniformModel:
    def predict(self, inputs, verbose=0):
        batch = int(inputs[0].shape[0])
        return np.full((batch, 6), 1.0 / 6.0, dtype=np.float32)


MODEL_PATH = "/kaggle/input/hms-models/best_model_st_6_ft.keras"

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
else:
    print(f"[WARN] Model not found at {MODEL_PATH}. Using UniformModel fallback.")
    model = UniformModel()



## === cell 9
from concurrent.futures import ThreadPoolExecutor, as_completed


def _preprocess_row(r):
    return preprocess_data(r, train=False)


MAX_WORKERS = min(8, (os.cpu_count() or 4))

submission = {"eeg_id": []}
for c in TARGET_COLS:
    submission[c] = []

BATCH_SIZE = 32

n = len(metadata)
row = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    while row < n:
        end = min(row + BATCH_SIZE, n)
        rows = list(range(row, end))
        k = len(rows)

        eeg_batch_u8 = [None] * k
        spec_batch_u8 = [None] * k
        eeg_ids_batch = [None] * k

        future_to_idx = {ex.submit(_preprocess_row, r): (r - row) for r in rows}
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            eeg_arr, spec_arr, eeg_id = fut.result()
            eeg_batch_u8[idx] = eeg_arr
            spec_batch_u8[idx] = spec_arr
            eeg_ids_batch[idx] = eeg_id

        eeg_batch, spec_batch = _batch_to_model_inputs(eeg_batch_u8, spec_batch_u8)

        prediction = model.predict([eeg_batch, spec_batch], verbose=0)
        prediction = np.asarray(prediction)

        prediction = np.clip(prediction, 1e-8, None)
        prediction = prediction / prediction.sum(axis=1, keepdims=True)

        submission["eeg_id"].extend(eeg_ids_batch)
        for j, col in enumerate(TARGET_COLS):
            submission[col].extend(prediction[:, j].astype(np.float64).tolist())

        del (
            eeg_batch,
            spec_batch,
            prediction,
            eeg_batch_u8,
            spec_batch_u8,
            eeg_ids_batch,
            future_to_idx,
        )

        if (row // BATCH_SIZE) % 200 == 0:
            gc.collect()

        row = end

pdsubmit = pd.DataFrame(submission)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3783108123.py in <cell line: 0>()
     32         for fut in as_completed(future_to_idx):
     33             idx = future_to_idx[fut]
---> 34             eeg_arr, spec_arr, eeg_id = fut.result()
     35             eeg_batch_u8[idx] = eeg_arr
     36             spec_batch_u8[idx] = spec_arr

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    447                     raise CancelledError()
    448                 elif self._state == FINISHED:
--> 449                     return self.__get_result()
    450 
    451                 self._condition.wait(timeout)

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/concurrent/futures/thread.py in run(self)
     56 
     57         try:
---> 58             result = self.fn(*self.args, **self.kwargs)
     59         except BaseException as exc:
     60             self.future.set_exception(exc)

/tmp/ipykernel_55/3783108123.py in _preprocess_row(r)
      5 
      6 def _preprocess_row(r):
----> 7     return preprocess_data(r, train=False)
      8 
      9 

/tmp/ipykernel_55/4051317025.py in preprocess_data(row, train)
     16         spec_id = int(_meta_spec_ids[row])
     17         eeg_arr = generate_eeg(eegid=eeg_id)
---> 18         spec_arr = generate_spectrogram(specid=spec_id, output_filter=0.7)
     19 
     20     return eeg_arr, spec_arr, eeg_id

NameError: name 'generate_spectrogram' is not defined

## === cell 10
pdsubmit = sample_sub[["eeg_id"]].merge(pdsubmit, on="eeg_id", how="left")

for col in TARGET_COLS:
    if col not in pdsubmit.columns:
        pdsubmit[col] = 1.0 / 6.0
    pdsubmit[col] = pdsubmit[col].fillna(1.0 / 6.0)

probs = pdsubmit[TARGET_COLS].to_numpy(dtype=np.float64)
probs = np.clip(probs, 1e-12, None)
probs = probs / probs.sum(axis=1, keepdims=True)
pdsubmit.loc[:, TARGET_COLS] = probs

pdsubmit



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/496825118.py in <cell line: 0>()
----> 1 pdsubmit = sample_sub[["eeg_id"]].merge(pdsubmit, on="eeg_id", how="left")
      2 
      3 for col in TARGET_COLS:
      4     if col not in pdsubmit.columns:
      5         pdsubmit[col] = 1.0 / 6.0

NameError: name 'pdsubmit' is not defined

## === cell 11
pdsubmit.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", pdsubmit.shape)
print(pdsubmit.head())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/813838049.py in <cell line: 0>()
----> 1 pdsubmit.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", pdsubmit.shape)
      3 print(pdsubmit.head())

NameError: name 'pdsubmit' is not defined
