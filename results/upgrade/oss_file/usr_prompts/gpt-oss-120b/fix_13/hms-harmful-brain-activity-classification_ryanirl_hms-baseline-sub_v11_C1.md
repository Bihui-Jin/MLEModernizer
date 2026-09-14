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

cudf-polars-cu12==25.6.0
geopandas==0.14.4
librosa==0.11.0
numpy==1.26.4
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

# 5. Target score

0.3642930579177063

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I pad or truncate each EEG feature array to a fixed length (2500) so all feature tensors have identical shapes, allowing the stacking operation to succeed. This resolves the ValueError in cell 9 and enables the subsequent cells to run, producing a valid `submission.csv` file. The change is limited to the feature‑loading logic and does not alter model architecture or training.'
- What this solution (achieved 1.40711) has done: 'We blend the model‑averaged predictions with the overall class prior from the training data (using a small weight α≈0.1). This regularises over‑confident predictions, which often improves KL‑divergence, moving the score closer to the target while keeping the original model architecture and training untouched.'
- What this solution (achieved 1.40414) has done: 'I increase the regularisation of the model outputs by applying temperature scaling to smooth the predicted probabilities and raise the prior blending weight (α) so the predictions are pulled more toward the overall class distribution. These lightweight post‑processing tweaks keep the core model unchanged while expectedly lowering the KL‑divergence score.'
- What this solution (achieved 1.41138) has done: 'I adjust the post‑processing to smooth the model outputs more aggressively and blend them stronger toward the overall class prior, which is known to lower KL‑divergence when the raw predictions are over‑confident. In cell 9 I increase the temperature scaling from 2.0 to 5.0 and raise the prior‑blending weight α from 0.4 to 0.8. These minimal changes keep the core model unchanged while moving the score closer to the target lower‑is‑better metric.'
- What this solution (achieved 1.40427) has done: 'I reduce the aggressive smoothing and prior‑blending that were hurting the KL‑divergence. In cell 9 I set a milder temperature (1.5) and a smaller prior‑mix weight (α = 0.3), which moves the predictions closer to the original model outputs while still providing a small regularisation – this should lower the score toward the target.'
- What this solution (achieved 1.40711) has done: 'I keep the overall architecture unchanged but adjust the post‑processing that smooths the model outputs. Increasing the temperature scaling (more smoothing) and reducing the prior‑mix weight pull overly‑confident predictions toward a softer distribution, which should lower the KL‑divergence and move the score closer to the target. The changes are limited to the temperature and α values in cell 9.'
- What this solution (achieved 1.40524) has done: 'I slightly increase the temperature scaling to smooth the model probabilities further and raise the prior‑blending weight a bit. This keeps the core model untouched while applying a gentler post‑processing that is expected to lower the KL‑divergence (moving the score closer to the target).'
- What this solution (achieved 1.40841) has done: 'I keep the whole pipeline unchanged but adjust the post‑processing that smooths the model outputs. The temperature is reduced from 10.0 to 2.0 to avoid overly uniform predictions, and the prior‑blending weight is lowered from 0.2 to 0.05 so the model’s own probabilities dominate while still keeping a tiny regularisation. These minimal tweaks should move the KL‑divergence lower toward the target without altering the core model or data handling.'

# 9. Code solution

## === cell 0
import os
import polars as pl

DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"
SPEC_DIR = os.path.join(DATA_DIR, "test_spectrograms/")
EEG_DIR = os.path.join(DATA_DIR, "test_eegs/")

df_train = pl.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test = pl.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_submission = pl.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

df_test.head()



## === cell 1
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()




## === cell 2
def load_model(path: str, model: nn.Module) -> nn.Module:
    model.load_state_dict(
        torch.load(path, map_location=torch.device("cpu"))["model_state_dict"]
    )
    return model




## === cell 3
import sys

sys.path.append("/kaggle/input/hms-models/")

try:
    from eeg_cnn_rnn_w1 import EegModel as EegModel0
except ModuleNotFoundError:
    EegModel0 = None

try:
    from eeg_cnn_rnn import EegModel as EegModel1
except ModuleNotFoundError:
    EegModel1 = None



## === cell 4
import glob
import os

models_0 = []
if EegModel0 is not None:
    for fold_path in glob.glob("/kaggle/input/hms-models/new_fold/send_kaggle/*"):
        model = EegModel0()
        model = load_model(fold_path, model)
        model = model.to(device).eval()
        models_0.append(model)

models_1 = []
if EegModel1 is not None:
    for fold_path in glob.glob("/kaggle/input/hms-models/baseline_eeg_diff/*"):
        model_path = os.path.join(fold_path, "model_best_val_g10.pt")
        model = EegModel1()
        model = load_model(model_path, model)
        model = model.to(device).eval()
        models_1.append(model)

print("Loaded models:", len(models_0) + len(models_1))



## === cell 5
import polars as pl
import librosa
import numpy as np

from scipy.ndimage import convolve

KERNEL = np.array([-1, -1, -1, 0, 1, 1, 1])


def compute_spec(eeg: np.ndarray) -> np.ndarray:
    spectrogram = librosa.stft(
        eeg,
        n_fft=1024,
        hop_length=39,
        win_length=256,
        window="hann",
        center=True,
        pad_mode="constant",
    )
    spectrogram = librosa.power_to_db(np.abs(spectrogram) ** 2, ref=np.max).astype(
        np.float32
    )
    spectrogram = (spectrogram + 80) / 80
    spectrogram = spectrogram**2
    return spectrogram[:256][::2, ::2]


def spec(eeg: np.ndarray) -> np.ndarray:
    spec = compute_spec(eeg)
    eeg = convolve(eeg, KERNEL)
    spec = spec + compute_spec(eeg)
    return spec / 2


def compute_chain(df_eeg: pl.DataFrame) -> np.ndarray:
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
    vals = {c: df_eeg[c].to_numpy() for c in cols}
    ll = np.stack(
        [
            spec(vals["Fp1"] - vals["F7"]),
            spec(vals["F7"] - vals["T3"]),
            spec(vals["T3"] - vals["T5"]),
            spec(vals["T5"] - vals["O1"]),
        ]
    )
    lp = np.stack(
        [
            spec(vals["Fp1"] - vals["F3"]),
            spec(vals["F3"] - vals["C3"]),
            spec(vals["C3"] - vals["P3"]),
            spec(vals["P3"] - vals["O1"]),
        ]
    )
    rp = np.stack(
        [
            spec(vals["Fp2"] - vals["F4"]),
            spec(vals["F4"] - vals["C4"]),
            spec(vals["C4"] - vals["P4"]),
            spec(vals["P4"] - vals["O2"]),
        ]
    )
    rl = np.stack(
        [
            spec(vals["Fp2"] - vals["F8"]),
            spec(vals["F8"] - vals["T4"]),
            spec(vals["T4"] - vals["T6"]),
            spec(vals["T6"] - vals["O2"]),
        ]
    )

    chain = np.stack([ll, lp, rp, rl])[:, 0]
    return chain


def compute_spec_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_chain(df_eeg)
    chain = preprocess_chain(chain) if "preprocess_chain" in globals() else chain
    return chain




## === cell 6
import numpy as np
import math
from typing import Union, Tuple, List


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




## === cell 7
from scipy.signal import butter, filtfilt

_FS = 200
_BUTTER_B, _BUTTER_A = butter(
    N=4,
    Wn=np.array([0.25, 50]) / (0.5 * _FS),
    btype="bandpass",
    analog=False,
)


def butter_filter(eeg_data):
    """Apply the pre‑computed band‑pass filter."""
    return filtfilt(_BUTTER_B, _BUTTER_A, eeg_data)


def compute_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = butter_filter(eeg)
    eeg = bin_array(eeg, bin_size=4, mode="reflect").mean(axis=-1)
    return eeg


def compute_eeg_chain(df_eeg: pl.DataFrame) -> np.ndarray:
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
    vals = {c: df_eeg[c].to_numpy() for c in cols}
    ll = np.stack(
        [
            compute_eeg(vals["Fp1"] - vals["F7"]),
            compute_eeg(vals["F7"] - vals["T3"]),
            compute_eeg(vals["T3"] - vals["T5"]),
            compute_eeg(vals["T5"] - vals["O1"]),
        ]
    )
    lp = np.stack(
        [
            compute_eeg(vals["Fp1"] - vals["F3"]),
            compute_eeg(vals["F3"] - vals["C3"]),
            compute_eeg(vals["C3"] - vals["P3"]),
            compute_eeg(vals["P3"] - vals["O1"]),
        ]
    )
    rp = np.stack(
        [
            compute_eeg(vals["Fp2"] - vals["F4"]),
            compute_eeg(vals["F4"] - vals["C4"]),
            compute_eeg(vals["C4"] - vals["P4"]),
            compute_eeg(vals["P4"] - vals["O2"]),
        ]
    )
    rl = np.stack(
        [
            compute_eeg(vals["Fp2"] - vals["F8"]),
            compute_eeg(vals["F8"] - vals["T4"]),
            compute_eeg(vals["T4"] - vals["T6"]),
            compute_eeg(vals["T6"] - vals["O2"]),
        ]
    )

    chain = np.stack([ll, lp, rp, rl])[:, 0]
    return chain


def compute_eeg_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    return compute_eeg_chain(df_eeg)




## === cell 8
np.set_printoptions(formatter={"all": lambda x: f"{x:0.3f}"})


def proc_0(x):
    x = x.copy()
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / (np.mean(x.std(axis[-1])) + 1e-5)
    total = x.size
    dim = total // (4 * 4)
    x = x.reshape(4, 4, dim)
    return x


def proc_1(x):
    x = x.copy()
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / (x.std(axis=-1, keepdims=True) + 1e-5)
    total = x.size
    dim = total // (16 * 1)
    x = x.reshape(16, 1, dim)
    return x




## === cell 9
from tqdm.auto import tqdm
import concurrent.futures

LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

eeg_ids = df_test["eeg_id"].to_list()

TARGET_LEN = 2500


def _pad_or_truncate(arr: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
    """Ensure `arr` has shape (4, target_len) by padding with zeros or truncating."""
    if arr.shape[1] == target_len:
        return arr
    if arr.shape[1] < target_len:
        pad_width = target_len - arr.shape[1]
        return np.pad(arr, ((0, 0), (0, pad_width)), mode="constant")
    return arr[:, :target_len]


def load_feature(eeg_id):
    path = os.path.join(EEG_DIR, f"{eeg_id}.parquet")
    try:
        arr = compute_eeg_from_file(path)  # shape (4, L)
    except Exception as e:
        print(f"Warning: failed to process {path}: {e}")
        arr = np.zeros((4, TARGET_LEN), dtype=np.float32)
    else:
        arr = _pad_or_truncate(arr, TARGET_LEN)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    all_features = list(
        tqdm(
            ex.map(load_feature, eeg_ids), total=len(eeg_ids), desc="Load & preprocess"
        )
    )

features_np = np.stack(all_features, axis=0)  # shape (N, 4, TARGET_LEN)

proc0_batch = np.stack([proc_0(f) for f in features_np], axis=0)  # (N,4,4,dim)
proc1_batch = np.stack([proc_1(f) for f in features_np], axis=0)  # (N,16,1,dim)

x0_batch = torch.tensor(proc0_batch, dtype=torch.float32, device=device)
x1_batch = torch.tensor(proc1_batch, dtype=torch.float32, device=device)

preds_acc = []  # list of (N, 6) arrays from each model

for model in models_0:
    with torch.no_grad():
        batch_pred = model(x0_batch).exp()  # (N, 6)
    preds_acc.append(batch_pred.cpu().numpy())

for model in models_1:
    with torch.no_grad():
        batch_pred = model(x1_batch).exp()
    preds_acc.append(batch_pred.cpu().numpy())

if preds_acc:
    preds_mean = np.mean(np.stack(preds_acc, axis=0), axis=0)  # (N,6)
    preds_mean /= preds_mean.sum(axis=1, keepdims=True)
else:
    preds_mean = np.ones((len(eeg_ids), len(LABELS)), dtype=np.float32) / len(LABELS)

temperature = 3.0  # higher temperature → softer probabilities
if temperature != 1.0:
    preds_mean = np.power(preds_mean, 1.0 / temperature)
    preds_mean /= preds_mean.sum(axis=1, keepdims=True)

train_votes = df_train.select(LABELS).sum()
prior = train_votes.to_numpy().astype(np.float32)
prior /= prior.sum()  # shape (6,)

alpha = 0.2  # stronger prior blending
preds_blend = alpha * prior + (1 - alpha) * preds_mean
preds_blend /= preds_blend.sum(axis=1, keepdims=True)  # re‑normalize

preds_final = preds_blend.tolist()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3250648032.py in <cell line: 0>()
     48 features_np = np.stack(all_features, axis=0)  # shape (N, 4, TARGET_LEN)
     49 
---> 50 proc0_batch = np.stack([proc_0(f) for f in features_np], axis=0)  # (N,4,4,dim)
     51 proc1_batch = np.stack([proc_1(f) for f in features_np], axis=0)  # (N,16,1,dim)
     52 

/tmp/ipykernel_55/3250648032.py in <listcomp>(.0)
     48 features_np = np.stack(all_features, axis=0)  # shape (N, 4, TARGET_LEN)
     49 
---> 50 proc0_batch = np.stack([proc_0(f) for f in features_np], axis=0)  # (N,4,4,dim)
     51 proc1_batch = np.stack([proc_1(f) for f in features_np], axis=0)  # (N,16,1,dim)
     52 

/tmp/ipykernel_55/3049135350.py in proc_0(x)
      5     x = x.copy()
      6     x = x - x.mean(axis=-1, keepdims=True)
----> 7     x = x / (np.mean(x.std(axis[-1])) + 1e-5)
      8     total = x.size
      9     dim = total // (4 * 4)

NameError: name 'axis' is not defined

## === cell 10
import pandas as pd

df_sub = pd.DataFrame({"eeg_id": df_test["eeg_id"].to_list()})
df_sub[LABELS] = preds_final
df_sub.to_csv("submission.csv", index=False)
df_sub

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1655438312.py in <cell line: 0>()
      2 
      3 df_sub = pd.DataFrame({"eeg_id": df_test["eeg_id"].to_list()})
----> 4 df_sub[LABELS] = preds_final
      5 df_sub.to_csv("submission.csv", index=False)
      6 df_sub

NameError: name 'preds_final' is not defined
