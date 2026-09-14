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

# 5. Target score

0.3936035101604908

# 6. Current score

1.16647

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I remove the hard dependency on the missing `/kaggle/input/hms-models/` package by adding a safe fallback that still produces valid probability predictions when the pretrained models can’t be imported/loaded. I fix the EEG-spectrogram reshape crash by resizing/cropping the computed EEG spectrogram into the expected `(4, 96, 224)` shape deterministically. I also fix the Polars row indexing issue (`df_test[i]` returns a Series) by iterating as dictionaries/rows, ensuring `eeg_id`/`spectrogram_id` are read correctly. Finally, I ensure the submission is written with the exact required columns, float probabilities, and per-row normalization to sum to 1.'
- What this solution (achieved 1.43453) has done: 'Your current score is far worse than the target (lower is better), and the main reason is that you’re effectively submitting uniform probabilities because the external pretrained models aren’t available. To move the score toward the target with minimal core-logic disruption, I keep your existing feature extraction intact but replace the uniform fallback with a lightweight, legitimate fallback predictor learned from `train.csv` only: the per-`patient_id` mean label distribution (and global mean as a backup). This preserves evaluation semantics (proper probability vectors summing to 1) and should substantially reduce KL versus uniform without changing your model/feature pipeline when external models exist. I also ensure deterministic normalization and that the submission row order matches `test.csv`.'
- What this solution (achieved 0.97064) has done: 'Your current score is much worse than the target (lower is better), and the main issue is that without the external pretrained models you’re predicting only from a coarse per-`patient_id` prior, which is weak for this dataset. To move the score toward the target while keeping the same overall pipeline, I keep your existing fallback but strengthen it with a second, still-legit signal available at inference: the per-`spectrogram_id` mean label distribution computed from `train.csv` (and combine it with the patient prior). I also make the fallback more robust by smoothing priors slightly (Dirichlet/additive smoothing) to avoid overconfident zeros that can hurt KL. Everything else (feature extraction, model path logic, prediction normalization, submission schema) stays the same, and it still write a valid `submission.csv`.'
- What this solution (achieved 1.16647) has done: 'Your current score (0.97064, lower is better) is still far from the target (0.3936), and since external pretrained models are unavailable your solution relies entirely on metadata priors. To move the score downward toward the target with minimal logic change, I strengthen the fallback by learning a consolidated per-`eeg_id` prior from `train.csv` (aggregating overlapping subsamples), which is a much closer match to the test unit (`eeg_id`) than `patient_id`/`spectrogram_id` priors alone. I then combine `eeg_id`, `spectrogram_id`, and `patient_id` priors with conservative weights and keep the same smoothing/normalization to avoid KL blow-ups. Everything else (feature extraction, model path logic, prediction normalization, submission schema and filename) stays the same.'

# 9. Code solution

## === cell 0
from tqdm.auto import tqdm
import pandas as pd
import polars as pl
import numpy as np
import os
import glob
import math

import torch
import torch.nn as nn

from typing import Union, Tuple, List



## === cell 1
DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"
SPEC_DIR = os.path.join(DATA_DIR, "test_spectrograms/")
EEG_DIR = os.path.join(DATA_DIR, "test_eegs/")

df_train = pl.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test = pl.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_submission = pl.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

df_test.head()



## === cell 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## === cell 3
def load_model(path: str, model: nn.Module) -> nn.Module:
    ckpt = torch.load(path, map_location=torch.device("cpu"))
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    return model




## === cell 4
models_multimodal = []
HAVE_EXTERNAL_MODELS = False

try:
    import sys

    sys.path.append("/kaggle/input/hms-models/")
    from comb_model import MultimodalModel  # type: ignore

    HAVE_EXTERNAL_MODELS = True
except Exception as e:
    HAVE_EXTERNAL_MODELS = False
    MultimodalModel = None  # type: ignore
    print("WARNING: External models not available, using fallback predictions.")
    print("Import error:", repr(e))

if HAVE_EXTERNAL_MODELS:
    for model_path in sorted(
        glob.glob("/kaggle/input/hms-models/final_multimodal_s1/final_multimodal_s1/*")
    ):
        try:
            model = MultimodalModel(None, None, None)
            print("Loading:", model_path)
            model = load_model(model_path, model)
            model = model.to(device)
            model.eval()
            models_multimodal.append(model)
        except Exception as e:
            print("Corrupted/failed:", model_path, "err:", repr(e))

print("Loaded multimodal models:", len(models_multimodal))




## === cell 5
def MAD(signal, axis=-1):
    """Compute the robust standard deviation (MAD) of a signal."""
    median = np.median(signal, axis=axis, keepdims=True)
    absolute_deviations = np.abs(signal - median)
    median_absolute_deviation = np.median(absolute_deviations, axis=axis, keepdims=True)
    scale_factor = 1.4826
    robust_std = median_absolute_deviation * scale_factor
    return robust_std


from scipy.signal import butter, filtfilt


def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4, btype="lowpass"):
    b, a = butter(N=order, Wn=cutoff_freq / (0.5 * fs), btype=btype, analog=False)
    return filtfilt(b, a, eeg_data)




## === cell 6
import torchaudio
from torchaudio.transforms import Spectrogram as _Spectrogram

n_fft = 800
win_length = 256
hop_length = 44

spectrogram = _Spectrogram(
    n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=None
)


@torch.no_grad()
def compute_spec(chain):
    chain = torch.tensor(chain, dtype=torch.float32)
    chain = spectrogram(chain)  # (..., freq, time) complex
    chain = chain[:, :, 2:98]
    chain = torch.abs(chain) / 15.0
    chain = torch.log(chain.clamp(min=math.exp(-4), max=math.exp(7)))
    chain = chain.mean(axis=1)
    return chain.cpu().numpy()


def compute_spec_eeg(a, b) -> np.ndarray:
    return butter_filter(
        a - b, cutoff_freq=np.array([0.25, 40.0]), order=5, btype="bandpass"
    )


def compute_spec_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    Fp1 = df_eeg["Fp1"].to_numpy()
    Fp2 = df_eeg["Fp2"].to_numpy()
    Fz = df_eeg["Fz"].to_numpy()
    Cz = df_eeg["Cz"].to_numpy()
    Pz = df_eeg["Pz"].to_numpy()
    F3 = df_eeg["F3"].to_numpy()
    F4 = df_eeg["F4"].to_numpy()
    F7 = df_eeg["F7"].to_numpy()
    F8 = df_eeg["F8"].to_numpy()
    C3 = df_eeg["C3"].to_numpy()
    C4 = df_eeg["C4"].to_numpy()
    P3 = df_eeg["P3"].to_numpy()
    P4 = df_eeg["P4"].to_numpy()
    T3 = df_eeg["T3"].to_numpy()
    T4 = df_eeg["T4"].to_numpy()
    T5 = df_eeg["T5"].to_numpy()
    T6 = df_eeg["T6"].to_numpy()
    O1 = df_eeg["O1"].to_numpy()
    O2 = df_eeg["O2"].to_numpy()

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




## === cell 7
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




## === cell 8
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




## === cell 9
def compute_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = butter_filter(eeg, cutoff_freq=np.array([0.25, 50]), btype="bandpass")
    eeg = bin_array(eeg, bin_size=4, mode="reflect").mean(axis=-1)
    return eeg


def compute_eeg_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    Fp1 = df_eeg["Fp1"].to_numpy()
    Fp2 = df_eeg["Fp2"].to_numpy()
    Fz = df_eeg["Fz"].to_numpy()
    Cz = df_eeg["Cz"].to_numpy()
    Pz = df_eeg["Pz"].to_numpy()
    F3 = df_eeg["F3"].to_numpy()
    F4 = df_eeg["F4"].to_numpy()
    F7 = df_eeg["F7"].to_numpy()
    F8 = df_eeg["F8"].to_numpy()
    C3 = df_eeg["C3"].to_numpy()
    C4 = df_eeg["C4"].to_numpy()
    P3 = df_eeg["P3"].to_numpy()
    P4 = df_eeg["P4"].to_numpy()
    T3 = df_eeg["T3"].to_numpy()
    T4 = df_eeg["T4"].to_numpy()
    T5 = df_eeg["T5"].to_numpy()
    T6 = df_eeg["T6"].to_numpy()
    O1 = df_eeg["O1"].to_numpy()
    O2 = df_eeg["O2"].to_numpy()

    ekg = df_eeg["O2"].to_numpy()
    ekg = butter_filter(ekg, cutoff_freq=np.array([0.50, 20.0]), btype="bandpass")
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




## === cell 10
LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



## === cell 11
train_pd = df_train.select(
    ["eeg_id", "patient_id", "spectrogram_id"] + LABELS
).to_pandas()

y = train_pd[LABELS].values.astype(np.float64)
y_sum = y.sum(axis=1, keepdims=True)
y_prob = y / np.clip(y_sum, 1e-12, None)

SMOOTH_EPS = (
    1e-3  # small enough to be negligible when priors are strong, but avoids zeros
)

global_prior = y_prob.mean(axis=0)
global_prior = global_prior + SMOOTH_EPS
global_prior = np.clip(global_prior, 1e-12, None)
global_prior = global_prior / global_prior.sum()

patient_df = pd.DataFrame(y_prob, columns=LABELS)
patient_df["patient_id"] = train_pd["patient_id"].values
patient_prior = patient_df.groupby("patient_id")[LABELS].mean()

spec_df = pd.DataFrame(y_prob, columns=LABELS)
spec_df["spectrogram_id"] = train_pd["spectrogram_id"].values
spec_prior = spec_df.groupby("spectrogram_id")[LABELS].mean()

eeg_df = pd.DataFrame(y_prob, columns=LABELS)
eeg_df["eeg_id"] = train_pd["eeg_id"].values
eeg_prior = eeg_df.groupby("eeg_id")[LABELS].mean()


def _normalize_prob(p: np.ndarray) -> np.ndarray:
    p = p.astype(np.float64, copy=False)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = p + SMOOTH_EPS
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return p


def get_patient_prior(pid: int) -> np.ndarray:
    if pid in patient_prior.index:
        return _normalize_prob(patient_prior.loc[pid].values)
    return global_prior.copy()


def get_spec_prior(sid: int) -> np.ndarray:
    if sid in spec_prior.index:
        return _normalize_prob(spec_prior.loc[sid].values)
    return global_prior.copy()


def get_eeg_prior(eid: int) -> np.ndarray:
    if eid in eeg_prior.index:
        return _normalize_prob(eeg_prior.loc[eid].values)
    return global_prior.copy()


def get_fallback_prior(eeg_id: int, patient_id: int, spectrogram_id: int) -> np.ndarray:
    p_eeg = get_eeg_prior(int(eeg_id))
    p_spc = get_spec_prior(int(spectrogram_id))
    p_pat = get_patient_prior(int(patient_id))
    p = 0.55 * p_eeg + 0.30 * p_spc + 0.15 * p_pat
    return _normalize_prob(p)


print("Computed priors. Global prior:", global_prior)



## === cell 12
np.set_printoptions(formatter={"all": lambda x: f"{x:0.3f}"})

import albumentations as A
import cv2

spec_transforms = A.Compose(
    [
        A.Resize(height=96, width=224, interpolation=cv2.INTER_CUBIC, p=1.0),
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

    x = x.transpose(1, 2, 0)
    x = spec_transforms(image=x)["image"]
    x = x.transpose(2, 0, 1)

    x = x.reshape(4, 96, 224)
    return x


def _resize_2d_timefreq(arr2d: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    arr2d = arr2d.astype(np.float32, copy=False)
    return cv2.resize(arr2d, (out_w, out_h), interpolation=cv2.INTER_CUBIC)


def proc_eeg_spec(x):
    x = x.copy()
    x[np.isnan(x) | np.isinf(x)] = 0

    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[0] != 4 and x.shape[1] == 4:
        x = x.T
    if x.shape[0] != 4:
        if x.shape[0] < 4:
            x = np.pad(x, ((0, 4 - x.shape[0]), (0, 0)), mode="edge")
        else:
            x = x[:4]

    x = x + 1.0

    img = _resize_2d_timefreq(x, out_h=4 * 96, out_w=224)  # (384,224)
    img = img.reshape(4, 96, 224)
    return img


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


@torch.no_grad()
def gen_ensemble_pred(eeg_id: int, spc_id: int, patient_id: int) -> np.ndarray:
    eeg_filepath = os.path.join(EEG_DIR, f"{eeg_id}.parquet")
    spc_filepath = os.path.join(SPEC_DIR, f"{spc_id}.parquet")

    if (not HAVE_EXTERNAL_MODELS) or (len(models_multimodal) == 0):
        return get_fallback_prior(int(eeg_id), int(patient_id), int(spc_id)).astype(
            np.float32
        )

    preds = []

    kspec = compute_kaggle_spec_from_file(spc_filepath)
    kspec = proc_kspec(kspec)
    kspec = torch.tensor(kspec, dtype=torch.float32, device=device).unsqueeze(0)

    eeg_spec = compute_spec_from_file(eeg_filepath)
    eeg_spec = proc_eeg_spec(eeg_spec)
    eeg_spec = torch.tensor(eeg_spec, dtype=torch.float32, device=device).unsqueeze(0)

    eeg, mid, ekg = compute_eeg_from_file(eeg_filepath)
    eeg[np.isnan(eeg) | np.isinf(eeg)] = 0
    eeg = proc_eeg(eeg, mid, ekg)
    eeg = torch.tensor(eeg, dtype=torch.float32, device=device).unsqueeze(0)

    for model in models_multimodal:
        model.eval()
        out = model(eeg, eeg_spec, kspec)
        p = out.exp().detach().cpu().numpy().reshape(-1).astype(np.float64)
        preds.append(p)

    preds = np.mean(preds, axis=0)
    preds = np.clip(preds, 1e-12, np.inf)
    preds = preds / preds.sum()
    return preds.astype(np.float32)




## === cell 13
preds_final = np.zeros((len(df_test), 6), dtype=np.float32)

eeg_ids = df_test["eeg_id"].to_list()
spc_ids = df_test["spectrogram_id"].to_list()
patient_ids_test = df_test["patient_id"].to_list()

for i in tqdm(range(len(df_test))):
    pred = gen_ensemble_pred(int(eeg_ids[i]), int(spc_ids[i]), int(patient_ids_test[i]))
    pred = np.clip(pred, 1e-12, np.inf)
    pred = pred / pred.sum()
    preds_final[i] = pred

preds_final[:3], preds_final.shape



## === cell 14
df_sub = pd.DataFrame({"eeg_id": eeg_ids})
for j, col in enumerate(LABELS):
    df_sub[col] = preds_final[:, j].astype(np.float64)

row_sums = df_sub[LABELS].sum(axis=1).values
df_sub[LABELS] = (
    df_sub[LABELS].values / np.clip(row_sums[:, None], 1e-12, None)
).astype(np.float64)

out_path = "submission.csv"
df_sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", df_sub.shape)
df_sub.head()
