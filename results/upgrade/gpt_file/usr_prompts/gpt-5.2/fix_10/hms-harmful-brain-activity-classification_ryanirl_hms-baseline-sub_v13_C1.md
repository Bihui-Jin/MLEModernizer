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

0.3589514306667826

# 6. Current score

1.41927

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39683) has done: 'I fix the missing external model import by defining fallback `EegModel0/EegModel1` classes that match the expected forward usage (returning log-probabilities so `.exp()` works), so the notebook runs even when `/kaggle/input/hms-models/` isn’t available. I also fix the reshape crash by ensuring `compute_eeg()` always returns exactly 2500 time steps (downsample + pad/crop), which preserves the intended preprocessing semantics while making shapes consistent. Finally, I remove the per-sample printing (which can time out) and guarantee the submission is valid by clipping/renormalizing probabilities and writing `submission.csv` with the required columns.'
- What this solution (achieved 1.3955) has done: 'Your score is far worse than the target (lower-is-better), and the biggest reason is that you’re running fallback/random weights because `/kaggle/input/hms-models/` isn’t available, so predictions are essentially untrained. To move toward the target with minimal change and without altering the model/training logic, I add a deterministic “empirical prior” computed from `train.csv` vote proportions and blend it with the model ensemble outputs (small mixture) to dramatically reduce KL compared to random while keeping the rest of your pipeline intact. I also make the ensemble numerically consistent by averaging in log-space (geometric mean) before converting to probabilities, which is better aligned with KL for probabilistic models and is a minimal change in post-processing. The code still produce a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 1.39765) has done: 'Your current score is far worse than the target (lower-is-better), and because the external pretrained models are not available you’re effectively submitting near-random predictions with only a small prior blend. To move the score closer to the target with minimal risk and without changing the model/feature core logic, I increase the empirical-prior blending when running on fallback models (so predictions become close to the dataset’s average label distribution, which is much better than random under KL). I also make the ensemble averaging explicitly log-space stable via `logsumexp` (same semantics as geometric mean, but numerically safer) and keep the final probability clipping/renorm to satisfy Kaggle’s submission constraints. These changes are localized to post-processing and should substantially reduce KL while preserving your pipeline.'
- What this solution (achieved 1.48685) has done: 'Your current score is much worse than the target (lower-is-better), and the main reason is that without the external pretrained models you’re effectively predicting near-random, so the safest way to move toward the target is to lean more heavily on a strong empirical label prior. I keep your model/feature pipeline identical, but improve the prior itself by computing it at the *eeg_id* level (averaging normalized votes per eeg_id) to better match the test unit, and I slightly strengthen the prior mixing when external models are missing. I also make the ensemble averaging exactly match a numerically-stable `logsumexp - log(n)` form (same semantics as your current code, just more stable). The script still run end-to-end and write a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 1.4849) has done: 'I fix the Polars aggregation bug that prevents computing the empirical prior (the `pl.sum(pl.col(expr))` misuse), which currently stops execution and cascades into the later `NameError`. Then I ensure the prior is computed correctly at the `eeg_id` level using `group_by().agg()` with expression `.sum()` so the pipeline runs end-to-end. Finally, I keep your existing model inference and prior-mixing logic intact (score-neutral vs your intended behavior), and ensure a valid `submission.csv` is always written with probabilities clipped and row-normalized to sum to 1.'
- What this solution (achieved 1.40714) has done: 'Your current score (1.4849, lower-is-better) is far from the target (0.35895), and because external pretrained models aren’t available your fallback models produce essentially untrained predictions; the safest minimal way to move toward the target is to rely more on a strong label prior. I keep your full preprocessing + model inference intact, but (1) compute a slightly more robust prior by smoothing the vote distribution with a small Dirichlet/Laplace term and (2) increase the prior mixing weight when running fallback models (and reduce overconfidence via a tiny uniform mix), which typically improves KL a lot versus near-random/overconfident outputs. I also make the ensemble aggregation exactly the geometric-mean of probabilities (average of log-probs), which is the intended “log-space averaging” and avoids the current arithmetic-mean-in-probability-space effect. The submission writing and probability normalization remain the same to ensure a valid `submission.csv`.'
- What this solution (achieved 0.75929) has done: 'Your score is much worse than the target (lower-is-better), and the main driver is that without external pretrained weights your network outputs are effectively untrained; the safest minimal improvement is to make the submission closer to the true label distribution. I keep your preprocessing + model inference unchanged, but (1) compute a stronger, test-aligned prior at the patient level (averaging normalized votes per `patient_id`), and (2) use a patient-aware prior for each test row when possible, falling back to the global prior otherwise. I also slightly reduce the prior mix from 0.997 to 0.990 to avoid being “too prior-only” (which can hurt KL if test distribution shifts), while keeping your uniform mix and probability safety/renorm intact. These are localized post-processing changes that should move KL substantially toward your target without changing the core model logic.'
- What this solution (achieved 1.41927) has done: 'Your current score is still far from the target (lower-is-better), and the biggest lever without changing your core model/feature logic is the post-processing mixture that turns essentially-untrained fallback model outputs into reasonable probabilities. I keep your entire preprocessing and inference intact, but replace the patient-level prior (which can mismatch test patients and overfit noise) with a more stable EEG-level prior computed from `train.csv`, then blend it with a stronger weight when external pretrained models are missing. I also compute the global prior from total votes (not “mean of patient priors”) to better match the overall label distribution under the KL metric. These are minimal, localized changes that should move the KL substantially downward toward your target while preserving valid probability constraints.'

# 9. Code solution

## === cell 0
from tqdm.auto import tqdm
import pandas as pd
import polars as pl
import numpy as np
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Optional, Tuple, List, Dict, Any



## === cell 1
DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"
SPEC_DIR = os.path.join(DATA_DIR, "test_spectrograms/")
EEG_DIR = os.path.join(DATA_DIR, "test_eegs/")

df_train = pl.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test = pl.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_submission = pl.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

df_test.head()



## === cell 2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()




## === cell 3
def load_model(path: str, model: nn.Module) -> nn.Module:
    obj = torch.load(path, map_location=torch.device("cpu"))
    if isinstance(obj, dict) and "model_state_dict" in obj:
        state = obj["model_state_dict"]
    else:
        state = obj
    model.load_state_dict(state, strict=False)
    return model




## === cell 4
import sys

sys.path.append("/kaggle/input/hms-models/")

NUM_CLASSES = 6

try:
    from eeg_cnn_rnn_w1 import EegModel as EegModel0  # type: ignore
    from eeg_cnn_rnn import EegModel as EegModel1  # type: ignore

    _HAVE_EXTERNAL_MODELS = True
except Exception:
    _HAVE_EXTERNAL_MODELS = False

    class _FallbackEegModel(nn.Module):
        def __init__(self, in_channels: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(16, NUM_CLASSES),
                nn.LogSoftmax(dim=-1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class EegModel0(_FallbackEegModel):
        def __init__(self):
            super().__init__(in_channels=4)

    class EegModel1(_FallbackEegModel):
        def __init__(self):
            super().__init__(in_channels=16)


print("External models available:", _HAVE_EXTERNAL_MODELS)



## === cell 5
import glob

model_0_files = glob.glob(
    "/kaggle/input/hms-models/new_distil/send_kaggle/*"
) + glob.glob("/kaggle/input/hms-models/new_fold/send_kaggle/*")

models_0 = []
if len(model_0_files) > 0:
    for fold_path in model_0_files:
        model = EegModel0()
        model = load_model(fold_path, model)
        model = model.to(device)
        models_0.append(model)
else:
    model = EegModel0().to(device)
    models_0.append(model)

models_1 = []
model_1_dirs = glob.glob("/kaggle/input/hms-models/baseline_eeg_diff/*")
if len(model_1_dirs) > 0:
    for fold_path in model_1_dirs:
        model_path = os.path.join(fold_path, "model_best_val_g10.pt")
        model = EegModel1()
        model = load_model(model_path, model)
        model = model.to(device)
        models_1.append(model)
else:
    model = EegModel1().to(device)
    models_1.append(model)

len(models_0) + len(models_1)



## === cell 6
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
        out=None,
        dtype=None,
    )
    spectrogram = librosa.power_to_db(np.abs(spectrogram) ** 2, ref=np.max).astype(
        np.float32
    )
    spectrogram = (spectrogram + 80) / 80
    spectrogram = spectrogram**2
    return spectrogram[:256][::2, ::2]


def spec(eeg: np.ndarray) -> np.ndarray:
    s = compute_spec(eeg)
    eeg2 = convolve(eeg, KERNEL)
    s = s + compute_spec(eeg2)
    return s / 2


def compute_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    Fp1 = df_eeg["Fp1"].to_numpy()
    Fp2 = df_eeg["Fp2"].to_numpy()
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

    ll = np.stack([(spec(Fp1 - F7), spec(F7 - T3), spec(T3 - T5), spec(T5 - O1))])
    lp = np.stack([(spec(Fp1 - F3), spec(F3 - C3), spec(C3 - P3), spec(P3 - O1))])
    rp = np.stack([(spec(Fp2 - F4), spec(F4 - C4), spec(C4 - P4), spec(P4 - O2))])
    rl = np.stack([(spec(Fp2 - F8), spec(F8 - T4), spec(T4 - T6), spec(T6 - O2))])

    chain = np.stack([ll, lp, rp, rl])[:, 0]
    return chain


def preprocess_chain(chain: np.ndarray) -> np.ndarray:
    return chain


def compute_spec_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_chain(df_eeg)
    chain = preprocess_chain(chain)  # kept as-is even if unused elsewhere
    return chain




## === cell 7
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




## === cell 8
from scipy.signal import welch
from scipy.stats import linregress
from scipy.signal import butter, filtfilt


def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4, btype="lowpass"):
    b, a = butter(N=order, Wn=cutoff_freq / (0.5 * fs), btype=btype, analog=False)
    return filtfilt(b, a, eeg_data)


def _fix_len_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.shape[-1] == target_len:
        return x
    if x.shape[-1] > target_len:
        return x[..., :target_len]
    pad = target_len - x.shape[-1]
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)], mode="edge")


def compute_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = butter_filter(eeg, cutoff_freq=np.array([0.25, 50]), btype="bandpass")
    eeg = bin_array(eeg, bin_size=4, mode="reflect").mean(axis=-1)
    eeg = _fix_len_1d(eeg, 2500)
    return eeg.astype(np.float32)


def compute_eeg_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    Fp1 = df_eeg["Fp1"].to_numpy()
    Fp2 = df_eeg["Fp2"].to_numpy()
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

    chain = np.stack([ll, lp, rp, rl])[:, 0]
    return chain


def compute_eeg_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_eeg_chain(df_eeg)
    return chain




## === cell 9
np.set_printoptions(formatter={"all": lambda x: f"{x:0.3f}"})


def proc_0(x):
    x = x.copy()
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / (np.mean(x.std(axis=-1)) + 1e-5)
    x = x.reshape(4, 4, 2_500)
    return x


def proc_1(x):
    x = x.copy()
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / (x.std(axis=-1, keepdims=True) + 1e-5)
    x = x.reshape(16, 1, 2_500)
    return x


def _safe_probs(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, eps, None)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        p = np.ones_like(p, dtype=np.float64) / p.size
    else:
        p = p / s
    return p.astype(np.float32)


LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

_SMOOTH_ALPHA = 1.0  # keep smoothing for robustness against low-count rows

train_eeg_prior = (
    df_train.select(["eeg_id"] + LABELS)
    .with_columns(pl.sum_horizontal([pl.col(c) for c in LABELS]).alias("_sum_votes"))
    .group_by("eeg_id")
    .agg(
        [pl.col(c).sum().alias(c) for c in LABELS]
        + [pl.col("_sum_votes").sum().alias("_w")]
    )
    .with_columns(
        [
            (
                (pl.col(c) + _SMOOTH_ALPHA)
                / (pl.sum_horizontal([pl.col(cc) + _SMOOTH_ALPHA for cc in LABELS]))
            ).alias(c)
            for c in LABELS
        ]
    )
)

global_vote_totals = (
    df_train.select([pl.col(c).sum().alias(c) for c in LABELS]).to_numpy().reshape(-1)
)
PRIOR_GLOBAL = _safe_probs(global_vote_totals + _SMOOTH_ALPHA, eps=1e-6)

_eeg_ids = train_eeg_prior["eeg_id"].to_list()
_eeg_priors = train_eeg_prior.select(LABELS).to_numpy().astype(np.float32)
EEG_PRIOR_MAP: Dict[int, np.ndarray] = {
    int(eid): _safe_probs(_eeg_priors[i], eps=1e-6) for i, eid in enumerate(_eeg_ids)
}

PRIOR_MIX = 0.997 if not _HAVE_EXTERNAL_MODELS else 0.10

UNIFORM_MIX = 0.01 if not _HAVE_EXTERNAL_MODELS else 0.0
UNIFORM = np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES


@torch.no_grad()
def gen_ensemble_pred(models_0, models_1, eeg_id: int, patient_id: int) -> np.ndarray:
    filepath = os.path.join(EEG_DIR, f"{eeg_id}.parquet")

    x = compute_eeg_from_file(filepath)
    x[np.isnan(x) | np.isinf(x)] = 0

    x0 = proc_0(x)
    x0 = torch.tensor(x0, dtype=torch.float32, device=device).unsqueeze(0)

    x1 = proc_1(x)
    x1 = torch.tensor(x1, dtype=torch.float32, device=device).unsqueeze(0)

    logps = []

    for model in models_0:
        model.eval()
        logp = model(x0)
        logps.append(logp.detach().cpu().numpy().reshape(-1))

    for model in models_1:
        model.eval()
        logp = model(x1)
        logps.append(logp.detach().cpu().numpy().reshape(-1))

    logps = np.stack(logps, axis=0).astype(np.float64)
    logps_mean = logps.mean(axis=0)

    preds = np.exp(logps_mean)
    preds = _safe_probs(preds)

    if UNIFORM_MIX > 0:
        preds = _safe_probs((1.0 - UNIFORM_MIX) * preds + UNIFORM_MIX * UNIFORM)

    prior = EEG_PRIOR_MAP.get(int(eeg_id), PRIOR_GLOBAL)
    preds = _safe_probs((1.0 - PRIOR_MIX) * preds + PRIOR_MIX * prior)
    return preds




## === cell 10
preds_final = []

test_eeg_ids = df_test["eeg_id"].to_list()
test_patient_ids = df_test["patient_id"].to_list()

for eeg_id, patient_id in tqdm(
    list(zip(test_eeg_ids, test_patient_ids)), total=df_test.height
):
    pred = gen_ensemble_pred(models_0, models_1, int(eeg_id), int(patient_id))
    preds_final.append(pred)

preds_final = np.asarray(preds_final, dtype=np.float32)
preds_final.shape



## === cell 11
if preds_final.ndim != 2 or preds_final.shape[1] != len(LABELS):
    raise ValueError(
        f"preds_final has wrong shape: {preds_final.shape}, expected (n,{len(LABELS)})"
    )

preds_final = np.clip(preds_final, 1e-6, None)
preds_final = preds_final / preds_final.sum(axis=1, keepdims=True)

df_sub = pd.DataFrame({"eeg_id": df_test["eeg_id"].to_list()})
df_sub[LABELS] = preds_final
df_sub.to_csv("submission.csv", index=False)

df_sub.head()
