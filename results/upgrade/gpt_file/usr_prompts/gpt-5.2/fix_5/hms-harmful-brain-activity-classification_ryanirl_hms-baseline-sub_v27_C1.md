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

albumentations==2.0.8
cudf-polars-cu12==25.6.0
geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
polars==1.25.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scipy==1.15.3
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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
from tqdm.auto import tqdm
import pandas as pd
import polars as pl
import numpy as np
import argparse
import os
import glob
import sys
import math
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Optional, Tuple, List, Dict, Any, Union



## === cell 1
DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"
SPEC_DIR = os.path.join(DATA_DIR, "test_spectrograms/")
EEG_DIR = os.path.join(DATA_DIR, "test_eegs/")

df_train = pl.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test = pl.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_submission = pl.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

df_test.head()



## === cell 2
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True




## === cell 3
def load_model(path: str, model: nn.Module) -> nn.Module:
    ckpt = torch.load(path, map_location=torch.device("cpu"))
    state = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    model.load_state_dict(state)
    return model




## === cell 4
HMS_MODELS_DIR = "/kaggle/input/hms-models/"
HAS_HMS_MODELS = os.path.isdir(HMS_MODELS_DIR)

MultimodalModel = None
if HAS_HMS_MODELS:
    sys.path.append(HMS_MODELS_DIR)
    try:
        from eeg_cnn_rnn_att import EegModel as EegModel  # noqa: F401
        from spc_cnn_att import SpectrogramCnnModel as SpcModel  # noqa: F401
        from comb_model import MultimodalModel as _MM

        MultimodalModel = _MM
        print("Imported external HMS models.")
    except Exception as e:
        print(
            "Could not import external HMS models, will use fallback. Error:", repr(e)
        )
        MultimodalModel = None
else:
    print(
        "No /kaggle/input/hms-models/ found; using fallback model to generate a valid submission."
    )


class FallbackMultimodalModel(nn.Module):
    def __init__(self, n_classes: int = 6):
        super().__init__()
        self.n_classes = n_classes
        self.register_buffer("logp", torch.log(torch.ones(n_classes) / n_classes))

    def forward(self, eeg, eeg_spec, kspec):
        b = eeg.shape[0]
        return self.logp.unsqueeze(0).repeat(b, 1)


if MultimodalModel is None:
    MultimodalModel = FallbackMultimodalModel



## === cell 5
models_multimodal: List[nn.Module] = []

if HAS_HMS_MODELS:
    for model_path in sorted(
        glob.glob(
            "/kaggle/input/hms-models/final_multimodal_sanity/final_multimodal_sanity/*"
        )
    ):
        try:
            model = (
                MultimodalModel(None, None, None)
                if "Fallback" not in MultimodalModel.__name__
                else MultimodalModel()
            )
            print("Loading:", model_path)
            model = load_model(model_path, model)
            model = model.to(device)
            model.eval()
            models_multimodal.append(model)
        except Exception:
            print("Corrupted/unloadable:", model_path)

    for model_path in sorted(glob.glob("/kaggle/input/hms-models/stage_6/stage_6/*")):
        try:
            model = (
                MultimodalModel(None, None, None)
                if "Fallback" not in MultimodalModel.__name__
                else MultimodalModel()
            )
            print("Loading:", model_path)
            model = load_model(model_path, model)
            model = model.to(device)
            model.eval()
            models_multimodal.append(model)
        except Exception:
            print("Corrupted/unloadable:", model_path)

if len(models_multimodal) == 0:
    model = MultimodalModel().to(device).eval()
    models_multimodal = [model]

print("Number of models:", len(models_multimodal))




## === cell 6
def MAD(signal, axis=-1):
    """Compute the robust standard deviation (MAD) of a signal."""
    median = np.median(signal, axis=axis, keepdims=True)
    absolute_deviations = np.abs(signal - median)
    median_absolute_deviation = np.median(absolute_deviations, axis=axis, keepdims=True)
    scale_factor = 1.4826
    robust_std = median_absolute_deviation * scale_factor
    return robust_std


from scipy.signal import butter, sosfiltfilt

_FS = 200
_BUTTER_SOS: Dict[Tuple[int, str, Tuple[float, ...]], np.ndarray] = {}


def _get_butter_sos(fs: int, cutoff_freq, order: int, btype: str) -> np.ndarray:
    if np.isscalar(cutoff_freq):
        key = (order, btype, (float(cutoff_freq), float(fs)))
    else:
        cf = tuple(float(x) for x in np.asarray(cutoff_freq).ravel().tolist())
        key = (order, btype, cf + (float(fs),))
    sos = _BUTTER_SOS.get(key)
    if sos is None:
        sos = butter(
            N=order,
            Wn=np.asarray(cutoff_freq, dtype=np.float64) / (0.5 * fs),
            btype=btype,
            analog=False,
            output="sos",
        )
        _BUTTER_SOS[key] = sos
    return sos


def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4, btype="lowpass"):
    sos = _get_butter_sos(fs=fs, cutoff_freq=cutoff_freq, order=order, btype=btype)
    return sosfiltfilt(sos, eeg_data)




## === cell 7
import torchaudio
from torchaudio.transforms import Spectrogram as _Spectrogram

n_fft = 800
win_length = 256
hop_length = 44

spectrogram = _Spectrogram(
    n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=None
)

_SPEC_DEVICE = device if device.type == "cuda" else torch.device("cpu")
spectrogram = spectrogram.to(_SPEC_DEVICE)


@torch.inference_mode()
def compute_spec(chain):
    chain_t = torch.as_tensor(chain, dtype=torch.float32, device=_SPEC_DEVICE)
    chain_t = spectrogram(chain_t)
    chain_t = chain_t[:, :, 2:98]
    chain_t = torch.abs(chain_t) / 15
    chain_t = torch.log(chain_t.clamp(min=math.exp(-4), max=math.exp(7)))
    chain_t = chain_t.mean(axis=1)
    return chain_t.to("cpu").numpy()


def compute_spec_eeg(a, b) -> np.ndarray:
    return butter_filter(
        a - b, cutoff_freq=np.array([0.25, 40.0]), order=5, btype="bandpass"
    )


def compute_spec_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    cols = [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "F7",
        "F8",
        "C3",
        "C4",
        "P3",
        "P4",
        "T3",
        "T4",
        "T5",
        "T6",
        "O1",
        "O2",
    ]
    arr = df_eeg.select(cols).to_numpy()
    (Fp1, Fp2, F3, F4, F7, F8, C3, C4, P3, P4, T3, T4, T5, T6, O1, O2) = [
        arr[:, i] for i in range(arr.shape[1])
    ]

    ll = np.stack(
        [
            (
                compute_spec_eeg(Fp1, F7),
                compute_spec_eeg(F7, T3),
                compute_spec_eeg(T3, T5),
                compute_spec_eeg(T5, O1),
            )
        ]
    )
    lp = np.stack(
        [
            (
                compute_spec_eeg(Fp1, F3),
                compute_spec_eeg(F3, C3),
                compute_spec_eeg(C3, P3),
                compute_spec_eeg(P3, O1),
            )
        ]
    )
    rp = np.stack(
        [
            (
                compute_spec_eeg(Fp2, F4),
                compute_spec_eeg(F4, C4),
                compute_spec_eeg(C4, P4),
                compute_spec_eeg(P4, O2),
            )
        ]
    )
    rl = np.stack(
        [
            (
                compute_spec_eeg(Fp2, F8),
                compute_spec_eeg(F8, T4),
                compute_spec_eeg(T4, T6),
                compute_spec_eeg(T6, O2),
            )
        ]
    )
    chain = np.stack([ll, lp, rp, rl])[:, 0]

    mads = MAD(chain, axis=-1)
    mads = np.median(mads.reshape(-1))
    chain = chain / (mads + 1e-5)

    chain = compute_spec(chain)
    return chain


def compute_spec_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_spec_chain(df_eeg)
    return chain




## === cell 8
def process_spec(spec: np.ndarray) -> np.ndarray:
    spec = spec[:, 1:]
    spec = np.stack(
        [
            spec[:, 0:100].T,
            spec[:, 100:200].T,
            spec[:, 200:300].T,
            spec[:, 300:400].T,
        ]
    )
    return spec


def compute_kaggle_spec_from_file(filepath: str) -> np.ndarray:
    spec = pl.read_parquet(filepath).to_numpy().astype(np.float32)
    spec = process_spec(spec)
    return spec




## === cell 9
def bin_array(
    array,
    bin_size,
    axis=-1,
    pad_dir="symmetric",
    mode="edge",
    return_padding=False,
    **padding_kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    if axis == -1:
        axis = array.ndim - 1

    curr_len = array.shape[axis]
    n_bins = math.ceil(curr_len / bin_size)
    new_len = n_bins * bin_size

    new_shape = list(array.shape)
    new_shape[axis] = n_bins
    new_shape.insert(axis + 1, bin_size)

    padding = [(0, 0)] * array.ndim
    if curr_len != new_len:
        if pad_dir == "left":
            pad_l = new_len - curr_len
            pad_r = 0
        elif pad_dir == "right":
            pad_l = 0
            pad_r = new_len - curr_len
        else:
            pad_l = (new_len - curr_len) // 2
            pad_r = (new_len - curr_len) - pad_l

        padding[axis] = (pad_l, pad_r)
        array = np.pad(array, padding, mode=mode, **padding_kwargs)

    array = array.reshape(new_shape)

    if return_padding:
        return array, padding
    return array




## === cell 10
def compute_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = butter_filter(
        eeg, cutoff_freq=np.array([0.25, 50]), btype="bandpass", order=4
    )
    eeg = bin_array(eeg, bin_size=4, mode="reflect").mean(axis=-1)
    return eeg


def compute_eeg_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    cols = [
        "Fp1",
        "Fp2",
        "Fz",
        "Cz",
        "Pz",
        "F3",
        "F4",
        "F7",
        "F8",
        "C3",
        "C4",
        "P3",
        "P4",
        "T3",
        "T4",
        "T5",
        "T6",
        "O1",
        "O2",
        "EKG",
    ]
    arr = df_eeg.select(cols).to_numpy()
    (
        Fp1,
        Fp2,
        Fz,
        Cz,
        Pz,
        F3,
        F4,
        F7,
        F8,
        C3,
        C4,
        P3,
        P4,
        T3,
        T4,
        T5,
        T6,
        O1,
        O2,
        ekg,
    ) = [arr[:, i] for i in range(arr.shape[1])]

    ekg = butter_filter(
        ekg, cutoff_freq=np.array([0.50, 20.0]), btype="bandpass", order=4
    )
    ekg = bin_array(ekg, bin_size=4, mode="reflect").mean(axis=-1)
    ekg = ekg.reshape(1, -1)

    ll = np.stack(
        [
            (
                compute_eeg(Fp1 - F7),
                compute_eeg(F7 - T3),
                compute_eeg(T3 - T5),
                compute_eeg(T5 - O1),
            )
        ]
    )
    lp = np.stack(
        [
            (
                compute_eeg(Fp1 - F3),
                compute_eeg(F3 - C3),
                compute_eeg(C3 - P3),
                compute_eeg(P3 - O1),
            )
        ]
    )
    rp = np.stack(
        [
            (
                compute_eeg(Fp2 - F4),
                compute_eeg(F4 - C4),
                compute_eeg(C4 - P4),
                compute_eeg(P4 - O2),
            )
        ]
    )
    rl = np.stack(
        [
            (
                compute_eeg(Fp2 - F8),
                compute_eeg(F8 - T4),
                compute_eeg(T4 - T6),
                compute_eeg(T6 - O2),
            )
        ]
    )
    mid = np.stack([compute_eeg(Fz - Cz), compute_eeg(Cz - Pz)])

    chain = np.stack([ll, lp, rp, rl])[:, 0]
    return chain, mid, ekg


def compute_eeg_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    return compute_eeg_chain(df_eeg)




## === cell 11
LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

np.set_printoptions(formatter={"all": lambda x: f"{x:0.3f}"})

import albumentations as A
import cv2

spec_transforms = A.Compose(
    [
        A.Resize(
            height=96,
            width=224,
            interpolation=cv2.INTER_CUBIC,
            always_apply=True,
            p=1.0,
        ),
    ]
)


def proc_kspec(x):
    x = x.copy()
    x = x[:, 2:98]
    x[np.isnan(x) | np.isinf(x)] = 0
    x = x.clip(np.exp(-4), np.exp(7))
    x = np.log(x)

    x = x - x.mean(axis=(1, 2), keepdims=True)
    x = x / (x.std(axis=(1, 2), keepdims=True) + 1e-5)

    x = x.transpose(1, 2, 0)  # (freq, time, 4)
    x = spec_transforms(image=x)["image"]  # -> (96, 224, 4)
    x = x.transpose(2, 0, 1)  # (4, 96, 224)
    x = x.reshape(4, 96, 224)
    return x


def _resize_or_pad_time_to_224(x2d: np.ndarray, target_t: int = 224) -> np.ndarray:
    F, T = x2d.shape
    if T == target_t:
        return x2d
    x_img = x2d.astype(np.float32)  # (F, T)
    x_img = cv2.resize(x_img, (target_t, F), interpolation=cv2.INTER_CUBIC)
    return x_img


def proc_eeg_spec(x):
    x = x.copy()  # expected (4, time, freq~96)
    x = x[:, :, 2:98]  # -> (4, time, 96)
    x[np.isnan(x) | np.isinf(x)] = 0
    x = x + 1

    x = x.transpose(0, 2, 1).astype(np.float32)
    out = np.empty((4, 96, 224), dtype=np.float32)
    for c in range(4):
        out[c] = _resize_or_pad_time_to_224(x[c], target_t=224)
    return out


def proc_eeg(eeg, mid, ekg):
    eeg = eeg.copy()
    mid = mid.copy()
    ekg = ekg.copy()

    eeg[np.isnan(eeg) | np.isinf(eeg)] = 0
    ekg[np.isnan(ekg) | np.isinf(ekg)] = 0
    mid[np.isnan(mid) | np.isinf(mid)] = 0

    eeg = eeg - eeg.mean(axis=-1, keepdims=True)
    mid = mid - mid.mean(axis=-1, keepdims=True)

    mad_std = MAD(eeg, axis=-1).reshape(-1)
    mad_std = np.median(mad_std) + 1e-5

    eeg = eeg / mad_std
    eeg = eeg.clip(-10, 10)

    mid = mid / mad_std
    mid = mid.clip(-10, 10)

    ekg = ekg / (MAD(ekg, axis=-1).reshape(-1) + 1e-5)

    eeg = eeg.reshape(16, -1)
    eeg = np.concatenate([eeg, mid, ekg], axis=0)
    eeg = eeg.reshape(19, 2_500)
    return eeg


_EEG_DF_CACHE: Dict[int, pl.DataFrame] = {}

_EEG_FEATURE_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = (
    {}
)  # eeg_id -> (eeg_proc, eeg_spec_proc)
_KSPEC_CACHE: Dict[int, np.ndarray] = {}  # spectrogram_id -> kspec_proc


def _get_eeg_df(eeg_id: int) -> pl.DataFrame:
    df = _EEG_DF_CACHE.get(eeg_id)
    if df is None:
        eeg_filepath = os.path.join(EEG_DIR, f"{eeg_id}.parquet")
        df = pl.read_parquet(eeg_filepath).fill_null(0)
        _EEG_DF_CACHE[eeg_id] = df
    return df


def _compute_eeg_features_for_eeg_id(eeg_id: int) -> Tuple[np.ndarray, np.ndarray]:
    cached = _EEG_FEATURE_CACHE.get(eeg_id)
    if cached is not None:
        return cached

    df_eeg = _get_eeg_df(eeg_id)

    eeg_spec = compute_spec_chain(df_eeg)
    eeg_spec = proc_eeg_spec(eeg_spec)

    eeg, mid, ekg = compute_eeg_chain(df_eeg)
    eeg[np.isnan(eeg) | np.isinf(eeg)] = 0
    eeg = proc_eeg(eeg, mid, ekg)

    _EEG_FEATURE_CACHE[eeg_id] = (eeg, eeg_spec)
    return eeg, eeg_spec


def _compute_kspec_for_spc_id(spc_id: int) -> np.ndarray:
    cached = _KSPEC_CACHE.get(spc_id)
    if cached is not None:
        return cached
    spc_filepath = os.path.join(SPEC_DIR, f"{spc_id}.parquet")
    kspec = compute_kaggle_spec_from_file(spc_filepath)
    kspec = proc_kspec(kspec)
    _KSPEC_CACHE[spc_id] = kspec
    return kspec


def _compute_all_features_for_ids(eeg_id: int, spc_id: int):
    eeg, eeg_spec = _compute_eeg_features_for_eeg_id(eeg_id)
    kspec = _compute_kspec_for_spc_id(spc_id)
    return eeg, eeg_spec, kspec




## === cell 12
eeg_ids = df_test["eeg_id"].to_numpy()
spc_ids = df_test["spectrogram_id"].to_numpy()

uniq_eeg_ids = np.unique(eeg_ids)
uniq_spc_ids = np.unique(spc_ids)

print(
    "Unique eeg_ids:", len(uniq_eeg_ids), "Unique spectrogram_ids:", len(uniq_spc_ids)
)

for eid in tqdm(uniq_eeg_ids, desc="Precomputing EEG features"):
    try:
        _compute_eeg_features_for_eeg_id(int(eid))
    except Exception:
        pass

for sid in tqdm(uniq_spc_ids, desc="Precomputing spectrogram features"):
    try:
        _compute_kspec_for_spc_id(int(sid))
    except Exception:
        pass


@torch.inference_mode()
def gen_ensemble_pred_batch(
    eeg_ids: np.ndarray, spc_ids: np.ndarray, batch_size: int = 16
) -> np.ndarray:
    n = len(eeg_ids)
    out_preds = np.empty((n, 6), dtype=np.float32)

    pin_memory = device.type == "cuda"
    for start in tqdm(range(0, n, batch_size), desc="Model inference"):
        end = min(n, start + batch_size)
        bs = end - start

        eeg_list = [None] * bs
        eeg_spec_list = [None] * bs
        kspec_list = [None] * bs
        valid_mask = np.ones(bs, dtype=bool)

        for j in range(bs):
            eeg_id = int(eeg_ids[start + j])
            spc_id = int(spc_ids[start + j])
            try:
                eeg, eeg_spec, kspec = _compute_all_features_for_ids(eeg_id, spc_id)
                eeg_list[j] = eeg
                eeg_spec_list[j] = eeg_spec
                kspec_list[j] = kspec
            except Exception:
                valid_mask[j] = False

        if not valid_mask.all():
            out_preds[start:end][~valid_mask] = np.ones(6, dtype=np.float32) / 6.0

        if valid_mask.any():
            idx = np.flatnonzero(valid_mask)
            eeg_np = np.stack([eeg_list[j] for j in idx], axis=0).astype(
                np.float32, copy=False
            )
            eeg_spec_np = np.stack([eeg_spec_list[j] for j in idx], axis=0).astype(
                np.float32, copy=False
            )
            kspec_np = np.stack([kspec_list[j] for j in idx], axis=0).astype(
                np.float32, copy=False
            )

            eeg_t = torch.from_numpy(eeg_np)
            eeg_spec_t = torch.from_numpy(eeg_spec_np)
            kspec_t = torch.from_numpy(kspec_np)
            if pin_memory:
                eeg_t = eeg_t.pin_memory()
                eeg_spec_t = eeg_spec_t.pin_memory()
                kspec_t = kspec_t.pin_memory()

            eeg_t = eeg_t.to(device, non_blocking=True)
            eeg_spec_t = eeg_spec_t.to(device, non_blocking=True)
            kspec_t = kspec_t.to(device, non_blocking=True)

            preds_sum = None
            for model in models_multimodal:
                out = model(eeg_t, eeg_spec_t, kspec_t)  # log-probs
                p = out.exp().float().cpu().numpy()
                if preds_sum is None:
                    preds_sum = p
                else:
                    preds_sum += p

            preds = preds_sum / float(len(models_multimodal))  # (n_valid, 6)
            preds = np.clip(preds, 1e-12, None)
            preds = preds / preds.sum(axis=1, keepdims=True)
            out_preds[start:end][valid_mask] = preds.astype(np.float32)

    return out_preds


preds_final = gen_ensemble_pred_batch(eeg_ids, spc_ids, batch_size=16)
print(
    "preds_final shape:",
    preds_final.shape,
    "row_sum min/max:",
    preds_final.sum(1).min(),
    preds_final.sum(1).max(),
)



## === cell 13
df_sub = pd.DataFrame({"eeg_id": df_test["eeg_id"].to_list()})
for j, col in enumerate(LABELS):
    df_sub[col] = preds_final[:, j]

probs = df_sub[LABELS].to_numpy(dtype=np.float64)
probs = np.clip(probs, 1e-12, None)
probs = probs / probs.sum(axis=1, keepdims=True)
df_sub[LABELS] = probs

df_sub.to_csv("submission.csv", index=False)
print(df_sub.head())
print("Wrote submission.csv with shape:", df_sub.shape)
