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

0.3701096170379047

# 6. Current score

0.93006

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I remove the hard dependency on the missing `/kaggle/input/hms-models/` dataset by adding a safe fallback `EegModel` definition that matches the expected inference contract (returns log-probabilities over 6 classes) so the notebook runs end-to-end. I also fix the EEG chain construction bug that returned shape `(4, 4, 128, 40)` instead of `(16, 2500)`, which caused the reshape failure in `proc_0`. Finally, I ensure predictions are always finite, strictly positive, and row-normalized to sum to 1 so the submission is valid and the pipeline always writes `submission.csv`.'
- What this solution (achieved 0.77245) has done: 'Your current score is far worse than the target (lower is better), and the main reason is that you’re effectively submitting near-uniform predictions because no real model checkpoints are being loaded from `/kaggle/input/hms-models/`. The smallest safe improvement toward the target is to add a deterministic, competition-legal fallback that uses only `train.csv` label priors (and patient-specific priors when available) to produce non-uniform probabilities; this preserves your inference-only pipeline and avoids changing the model/feature logic. I keep your EEG feature extraction and model inference path intact when checkpoints exist, but when they don’t, predictions come from smoothed empirical vote distributions (global or per-patient) and still be strictly positive and row-normalized. This should reduce KL substantially versus uniform while remaining minimal and stable.'
- What this solution (achieved 0.77245) has done: 'The crash is from enforcing a `validate="one_to_one"` merge even though `sample_submission` (and/or your constructed `pred_df`) can contain duplicated `eeg_id` keys in this environment, so pandas correctly raises `MergeError`. I make the submission assembly robust by (1) de-duplicating predictions by `eeg_id` (mean over duplicates) and (2) merging without strict `validate`, then re-aligning to the sample_submission order and filling any missing predictions with the global prior. This keeps your modeling/inference logic unchanged and only fixes the submission writing path, while still guaranteeing strictly positive, row-normalized probabilities. The pipeline then always write a valid `submission.csv`.'
- What this solution (achieved 0.82495) has done: 'Your current score (0.77245, lower-is-better) is still far above the target (0.3701), and the biggest lever without changing your model/feature core is improving the fallback (no-checkpoint) predictions so they better match the test label distribution. I keep your existing “use real checkpoints if available, else use priors” structure, but make the fallback prior more informative by (1) weighting patient prior + global prior + a mild expert-consensus-derived prior and (2) calibrating with a temperature (sharpening) tuned to reduce KL versus overly-flat distributions. I also fix a small bug in the diagnostics print (“PATIENT_PRIOR size” was incorrectly printing EEG_PRIOR size) and ensure the final submission is aligned to sample_submission order (stable) while remaining strictly positive and row-normalized.'
- What this solution (achieved 0.93006) has done: 'Your current score is much worse than the target (lower is better), and since no checkpoints are available you’re entirely in the “fallback priors” regime; the biggest safe lever is making those priors closer to the expected test distribution without changing your model/feature pipeline. I keep your existing fallback structure, but (1) compute patient/eeg posteriors using a proper Dirichlet-multinomial smoothing with strength proportional to total votes (instead of a near-negligible alpha), and (2) mix in a small patient→global shrinkage based on how many votes that patient has (more data → trust patient more). I also soften the sharpening temperature slightly to avoid overconfident wrong predictions (often increases KL), and keep the same strict positivity + row-normalization so the submission stays valid. These are minimal changes localized to the fallback prior computation and should move the score down toward your target.'
- What this solution (achieved 0.93006) has done: 'Your score is far above the target (lower is better), and since no checkpoints are being loaded you’re entirely in the “fallback priors” regime; the cleanest way to move KL down toward the target is to make the fallback distribution more informative while staying legal (train-only) and keeping your overall pipeline unchanged. I keep your existing global/patient/eeg prior machinery, but fix the fallback mixing bug where the “global weight” currently cancels out (it adds `g` twice and never uses a non-global base), and replace it with a proper convex mix of patient/global/consensus. I also add a tiny per-row floor (epsilon) before normalization to reduce extreme KL penalties from near-zero probabilities, without changing semantics or architecture. Everything else (EEG preprocessing, model inference path when checkpoints exist, submission alignment/normalization) stays the same.'

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
from torch.utils.data import DataLoader, Dataset

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




## === cell 3
def load_model(path: str, model: nn.Module) -> nn.Module:
    ckpt = torch.load(path, map_location=torch.device("cpu"))
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)
    return model




## === cell 4
import sys
import glob
import importlib.util
from types import ModuleType

MODELS_DIR = "/kaggle/input/hms-models/"
sys.path.append(MODELS_DIR)


def _try_load_module(py_path: str) -> Optional[ModuleType]:
    name = os.path.splitext(os.path.basename(py_path))[0]
    try:
        spec = importlib.util.spec_from_file_location(name, py_path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


EegModel0 = None
EegModel1 = None

py_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.py")))
loaded = []
for p in py_files:
    mod = _try_load_module(p)
    if mod is None:
        continue
    if hasattr(mod, "EegModel"):
        loaded.append((p, mod))

for p, mod in loaded:
    bn = os.path.basename(p)
    if EegModel0 is None and ("w1" in bn or "eeg_cnn_rnn_w1" in bn):
        EegModel0 = getattr(mod, "EegModel")
    if EegModel1 is None and ("eeg_cnn_rnn" in bn and "w1" not in bn):
        EegModel1 = getattr(mod, "EegModel")

if EegModel0 is None and len(loaded) > 0:
    EegModel0 = getattr(loaded[0][1], "EegModel")

if EegModel0 is None:

    class EegModel(nn.Module):
        """
        Fallback model used only when external model code/weights are unavailable.
        Keeps the same interface: forward(x) -> log-probs with 6 classes.
        """

        def __init__(self, n_classes: int = 6):
            super().__init__()
            self.n_classes = n_classes

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b = x.shape[0]
            logits = torch.zeros((b, self.n_classes), device=x.device, dtype=x.dtype)
            return torch.log_softmax(logits, dim=1)

    EegModel0 = EegModel
    EegModel1 = None

print(
    "Loaded external model definition:",
    "yes" if loaded else "no (using fallback EegModel0)",
)



## === cell 5
import glob

models_0 = []
weight_glob = "/kaggle/input/hms-models/new_distil/send_kaggle/*"
weight_paths = sorted(glob.glob(weight_glob))

if len(weight_paths) == 0:
    weight_paths = sorted(glob.glob("/kaggle/input/hms-models/**/*", recursive=True))
    weight_paths = [
        p
        for p in weight_paths
        if os.path.isfile(p)
        and (p.endswith(".pt") or p.endswith(".pth") or p.endswith(".bin"))
    ]

if len(weight_paths) == 0:
    model = EegModel0()
    model = model.to(device)
    model.eval()
    models_0.append(model)
    print("No checkpoint files found; using single fallback model.")
else:
    for fold_path in weight_paths:
        model = EegModel0()
        model = load_model(fold_path, model)
        model = model.to(device)
        model.eval()
        models_0.append(model)
    print(f"Loaded {len(models_0)} checkpoints.")

len(models_0)



## === cell 6
import librosa
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
    eeg_f = convolve(eeg, KERNEL)
    s = s + compute_spec(eeg_f)
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

    ll = np.stack([spec(Fp1 - F7), spec(F7 - T3), spec(T3 - T5), spec(T5 - O1)])
    lp = np.stack([spec(Fp1 - F3), spec(F3 - C3), spec(C3 - P3), spec(P3 - O1)])
    rp = np.stack([spec(Fp2 - F4), spec(F4 - C4), spec(C4 - P4), spec(P4 - O2)])
    rl = np.stack([spec(Fp2 - F8), spec(F8 - T4), spec(T4 - T6), spec(T6 - O2)])

    chain = np.concatenate([ll, lp, rp, rl], axis=0)  # (16, F, T)
    return chain




## === cell 7
import math
from typing import Union


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
from scipy.signal import butter, filtfilt

_BUTTER_CACHE: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}


def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4, btype="lowpass"):
    key = (
        fs,
        order,
        btype,
        (
            tuple(np.asarray(cutoff_freq).ravel().tolist())
            if isinstance(cutoff_freq, (list, tuple, np.ndarray))
            else float(cutoff_freq)
        ),
    )
    ba = _BUTTER_CACHE.get(key)
    if ba is None:
        b, a = butter(
            N=order,
            Wn=(
                np.array(cutoff_freq) / (0.5 * fs)
                if isinstance(cutoff_freq, (list, tuple, np.ndarray))
                else cutoff_freq / (0.5 * fs)
            ),
            btype=btype,
            analog=False,
        )
        _BUTTER_CACHE[key] = (b, a)
    else:
        b, a = ba
    return filtfilt(b, a, eeg_data)


def compute_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = np.asarray(eeg, dtype=np.float32)
    target_len = 10_000
    if eeg.shape[0] < target_len:
        eeg = np.pad(eeg, (0, target_len - eeg.shape[0]), mode="edge")
    elif eeg.shape[0] > target_len:
        eeg = eeg[:target_len]

    eeg = butter_filter(eeg, cutoff_freq=np.array([0.25, 50]), btype="bandpass")
    eeg = bin_array(eeg, bin_size=4, mode="reflect").mean(axis=-1)

    if eeg.shape[0] < 2_500:
        eeg = np.pad(eeg, (0, 2_500 - eeg.shape[0]), mode="edge")
    elif eeg.shape[0] > 2_500:
        eeg = eeg[:2_500]

    return eeg.astype(np.float32)


_EEG_COLS = [
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
_COL_TO_IDX = {c: i for i, c in enumerate(_EEG_COLS)}
_CHAIN_PAIRS = [
    ("Fp1", "F7"),
    ("F7", "T3"),
    ("T3", "T5"),
    ("T5", "O1"),  # ll
    ("Fp1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),  # lp
    ("Fp2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),  # rp
    ("Fp2", "F8"),
    ("F8", "T4"),
    ("T4", "T6"),
    ("T6", "O2"),  # rl
]
_CHAIN_A = np.array([_COL_TO_IDX[a] for a, b in _CHAIN_PAIRS], dtype=np.int64)
_CHAIN_B = np.array([_COL_TO_IDX[b] for a, b in _CHAIN_PAIRS], dtype=np.int64)


def compute_eeg_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    X = df_eeg.select(_EEG_COLS).to_numpy()
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    X = X.T
    diffs = X[_CHAIN_A] - X[_CHAIN_B]  # (16, T)
    out = np.empty((16, 2500), dtype=np.float32)
    for i in range(16):
        out[i] = compute_eeg(diffs[i])
    return out


def compute_eeg_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_eeg_chain(df_eeg)
    return chain




## === cell 9
np.set_printoptions(formatter={"all": lambda x: f"{x:0.3f}"})


def proc_0(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / (np.mean(x.std(axis=-1)) + 1e-5)
    x = x.reshape(4, 4, 2_500)
    return x


def proc_1(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / (x.std(axis=-1, keepdims=True) + 1e-5)
    x = x.reshape(16, 1, 2_500)
    return x


LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]




## === cell 10
def _make_global_prior_from_train(
    df_train: pl.DataFrame, labels: List[str], tau: float = 30.0
) -> np.ndarray:
    vote_sums = df_train.select([pl.col(c).sum().alias(c) for c in labels]).to_dicts()[
        0
    ]
    counts = np.array([vote_sums[c] for c in labels], dtype=np.float64)
    p = (counts + tau / len(labels)) / (counts.sum() + tau)
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return p.astype(np.float32)


def _make_patient_posteriors_from_train(
    df_train: pl.DataFrame,
    labels: List[str],
    global_prior: np.ndarray,
    tau_patient: float = 60.0,
) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
    df_pat = df_train.group_by("patient_id").agg(
        [pl.col(c).sum().alias(c) for c in labels]
    )
    pat_probs: Dict[int, np.ndarray] = {}
    pat_total_votes: Dict[int, float] = {}
    g = global_prior.astype(np.float64)
    for row in df_pat.iter_rows(named=True):
        counts = np.array([row[c] for c in labels], dtype=np.float64)
        total = float(counts.sum())
        probs = (counts + tau_patient * g) / (total + tau_patient)
        probs = np.clip(probs, 1e-12, None)
        probs = probs / probs.sum()
        pid = int(row["patient_id"])
        pat_probs[pid] = probs.astype(np.float32)
        pat_total_votes[pid] = total
    return pat_probs, pat_total_votes


def _make_eeg_posteriors_from_train(
    df_train: pl.DataFrame,
    labels: List[str],
    global_prior: np.ndarray,
    tau_eeg: float = 40.0,
) -> Dict[int, np.ndarray]:
    df_eeg = df_train.group_by("eeg_id").agg([pl.col(c).sum().alias(c) for c in labels])
    eeg_probs: Dict[int, np.ndarray] = {}
    g = global_prior.astype(np.float64)
    for row in df_eeg.iter_rows(named=True):
        counts = np.array([row[c] for c in labels], dtype=np.float64)
        total = float(counts.sum())
        probs = (counts + tau_eeg * g) / (total + tau_eeg)
        probs = np.clip(probs, 1e-12, None)
        probs = probs / probs.sum()
        eeg_probs[int(row["eeg_id"])] = probs.astype(np.float32)
    return eeg_probs


def _make_expert_consensus_prior(
    df_train: pl.DataFrame, labels: List[str], tau: float = 20.0
) -> np.ndarray:
    mapping = {
        "Seizure": "seizure_vote",
        "LPD": "lpd_vote",
        "GPD": "gpd_vote",
        "LRDA": "lrda_vote",
        "GRDA": "grda_vote",
        "Other": "other_vote",
    }
    vc = df_train.select(pl.col("expert_consensus")).to_series().value_counts()
    counts = np.zeros(len(labels), dtype=np.float64)
    for row in vc.iter_rows():
        key = row[0]
        n = float(row[1])
        col = mapping.get(key, None)
        if col is None:
            continue
        counts[labels.index(col)] += n
    p = (counts + tau / len(labels)) / (counts.sum() + tau)
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return p.astype(np.float32)


GLOBAL_PRIOR = _make_global_prior_from_train(df_train, LABELS, tau=30.0)
PATIENT_PRIOR, PATIENT_VOTES = _make_patient_posteriors_from_train(
    df_train, LABELS, GLOBAL_PRIOR, tau_patient=60.0
)
EEG_PRIOR = _make_eeg_posteriors_from_train(
    df_train, LABELS, GLOBAL_PRIOR, tau_eeg=40.0
)
CONSENSUS_PRIOR = _make_expert_consensus_prior(df_train, LABELS, tau=20.0)

print(
    "Global prior:",
    dict(zip(LABELS, GLOBAL_PRIOR.tolist())),
    "sum=",
    float(GLOBAL_PRIOR.sum()),
)
print(
    "Consensus prior:",
    dict(zip(LABELS, CONSENSUS_PRIOR.tolist())),
    "sum=",
    float(CONSENSUS_PRIOR.sum()),
)
print("EEG_PRIOR size:", len(EEG_PRIOR), "PATIENT_PRIOR size:", len(PATIENT_PRIOR))


def _is_fallback_uniform_model(models: List[nn.Module]) -> bool:
    return len(weight_paths) == 0


_FALLBACK_W_GLOBAL = 0.30
_FALLBACK_W_CONSENSUS = 0.10
_FALLBACK_TEMPERATURE = 1.05  # keep same intent (slight softening)


def _apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    if not np.isfinite(t) or t <= 0:
        return p.astype(np.float32)
    logp = np.log(p)
    logp = logp / t
    logp = logp - logp.max()
    q = np.exp(logp)
    q = np.clip(q, 1e-12, None)
    q = q / q.sum()
    return q.astype(np.float32)


def _fallback_prior_for(pid: Optional[int], eeg_id: Optional[int] = None) -> np.ndarray:
    if eeg_id is not None and int(eeg_id) in EEG_PRIOR:
        p = EEG_PRIOR[int(eeg_id)].astype(np.float32)
        return _apply_temperature(p, _FALLBACK_TEMPERATURE)

    g = GLOBAL_PRIOR.astype(np.float64)
    c = CONSENSUS_PRIOR.astype(np.float64)

    p_pat = PATIENT_PRIOR.get(pid, None)
    if p_pat is None:
        w_g = float(_FALLBACK_W_GLOBAL)
        w_c = float(_FALLBACK_W_CONSENSUS)
        w_u = max(
            0.0, 1.0 - w_g - w_c
        )  # remaining weight (uninformed) -> global (kept minimal change)
        mix = (w_u + w_g) * g + w_c * c
    else:
        total = float(PATIENT_VOTES.get(pid, 0.0))
        w_pat = total / (total + 200.0)  # 0..1, saturates with more patient data
        w_pat = float(np.clip(w_pat, 0.0, 0.85))
        w_rem = 1.0 - w_pat
        w_g = float(_FALLBACK_W_GLOBAL)
        w_c = float(_FALLBACK_W_CONSENSUS)
        w_base = max(0.0, 1.0 - w_g - w_c)
        base = w_base * g + w_g * g + w_c * c
        mix = w_pat * p_pat.astype(np.float64) + w_rem * base

    mix = np.clip(mix, 1e-12, None)
    mix = mix / mix.sum()
    return _apply_temperature(mix, _FALLBACK_TEMPERATURE)




## === cell 11
@torch.no_grad()
def gen_ensemble_pred(
    models_0: List[nn.Module], models_1, df_row: pl.DataFrame
) -> np.ndarray:
    if _is_fallback_uniform_model(models_0):
        eeg_id = int(df_row["eeg_id"].item())
        pid = (
            int(df_row["patient_id"].item()) if "patient_id" in df_row.columns else None
        )
        p = _fallback_prior_for(pid=pid, eeg_id=eeg_id)
        p = np.clip(p, 1e-12, None)
        p = p / p.sum()
        return p.astype(np.float32)

    eeg_id = df_row["eeg_id"].item()
    filepath = os.path.join(EEG_DIR, f"{eeg_id}.parquet")

    x = compute_eeg_from_file(filepath)
    x[np.isnan(x) | np.isinf(x)] = 0

    x0 = proc_0(x)
    x0 = torch.tensor(x0, dtype=torch.float32, device=device).unsqueeze(0)

    preds = []
    for model in models_0:
        pred = model(x0).exp()  # model outputs log-probs
        preds.append(pred.detach().cpu().numpy().reshape(-1))

    preds = np.mean(preds, axis=0)
    preds = np.clip(preds, 1e-12, None)
    s = float(preds.sum())
    if not np.isfinite(s) or s <= 0:
        preds = np.ones_like(preds, dtype=np.float32) / len(preds)
    else:
        preds = preds / s
    return preds.astype(np.float32)




## === cell 12
class TestEEGDataset(Dataset):
    def __init__(self, df: pl.DataFrame):
        self.eeg_ids = df["eeg_id"].to_list()
        self.patient_ids = (
            df["patient_id"].to_list()
            if "patient_id" in df.columns
            else [None] * len(self.eeg_ids)
        )

    def __len__(self) -> int:
        return len(self.eeg_ids)

    def __getitem__(self, idx: int):
        eeg_id = int(self.eeg_ids[idx])
        pid = None if self.patient_ids[idx] is None else int(self.patient_ids[idx])
        if _is_fallback_uniform_model(models_0):
            return eeg_id, pid, None

        filepath = os.path.join(EEG_DIR, f"{eeg_id}.parquet")
        x = compute_eeg_from_file(filepath)
        x[np.isnan(x) | np.isinf(x)] = 0
        x0 = proc_0(x).astype(np.float32, copy=False)  # (4,4,2500)
        return eeg_id, pid, x0


def _collate(batch):
    eeg_ids, pids, xs = zip(*batch)
    eeg_ids = np.asarray(eeg_ids, dtype=np.int64)
    pids = np.asarray([(-1 if p is None else p) for p in pids], dtype=np.int64)
    if xs[0] is None:
        return eeg_ids, pids, None
    x = np.stack(xs, axis=0)  # (B,4,4,2500)
    return eeg_ids, pids, x


torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

for m in models_0:
    m.eval()

ds = TestEEGDataset(df_test)

num_workers = min(4, max(1, (os.cpu_count() or 2) // 2))
loader = DataLoader(
    ds,
    batch_size=1 if _is_fallback_uniform_model(models_0) else 8,
    shuffle=False,
    num_workers=0 if _is_fallback_uniform_model(models_0) else num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
    collate_fn=_collate,
)

preds_final = np.zeros((len(df_test), len(LABELS)), dtype=np.float32)

row = 0
with torch.no_grad():
    for eeg_ids, pids, xb in tqdm(loader, total=len(loader)):
        bsz = len(eeg_ids)
        if _is_fallback_uniform_model(models_0):
            for j in range(bsz):
                eeg_id = int(eeg_ids[j])
                pid = None if int(pids[j]) == -1 else int(pids[j])
                preds_final[row + j] = _fallback_prior_for(pid=pid, eeg_id=eeg_id)
        else:
            x0 = torch.from_numpy(xb).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            preds = None
            for model in models_0:
                p = model(x0).exp()  # (B,6)
                preds = p if preds is None else (preds + p)
            preds = (
                (preds / len(models_0)).detach().cpu().numpy().astype(np.float32)
            )  # (B,6)
            preds = np.clip(preds, 1e-12, None)
            row_sums = preds.sum(axis=1, keepdims=True)
            bad = (~np.isfinite(row_sums)) | (row_sums <= 0)
            if np.any(bad):
                preds[bad[:, 0]] = 1.0 / preds.shape[1]
                row_sums = preds.sum(axis=1, keepdims=True)
            preds = preds / row_sums
            preds_final[row : row + bsz] = preds
        row += bsz

preds_final.shape



## === cell 13
sub = sample_submission.to_pandas()
test_eeg_ids = df_test["eeg_id"].to_list()

pred_df = pd.DataFrame({"eeg_id": test_eeg_ids})
pred_df[LABELS] = preds_final

pred_df = pred_df.groupby("eeg_id", as_index=False)[LABELS].mean()

sub = sub[["eeg_id"]].merge(pred_df, on="eeg_id", how="left")
sub = (
    sub.set_index("eeg_id").reindex(sample_submission["eeg_id"].to_list()).reset_index()
)

for i, c in enumerate(LABELS):
    if c not in sub.columns:
        sub[c] = np.nan
    sub[c] = sub[c].astype("float64")
    sub[c] = sub[c].fillna(float(GLOBAL_PRIOR[i]))

probs = sub[LABELS].to_numpy(dtype=np.float64)
probs[~np.isfinite(probs)] = 0.0

eps = 1e-6
probs = np.clip(probs, eps, None)

row_sums = probs.sum(axis=1, keepdims=True)
bad = (~np.isfinite(row_sums)) | (row_sums <= 0)
if np.any(bad):
    probs[bad[:, 0]] = 1.0 / probs.shape[1]
    row_sums = probs.sum(axis=1, keepdims=True)
probs = probs / row_sums
sub[LABELS] = probs.astype(np.float32)

assert len(sub) == len(sample_submission), (len(sub), len(sample_submission))

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print("Row-sum stats:", float(probs.sum(axis=1).min()), float(probs.sum(axis=1).max()))
print("Any NaNs?", bool(np.isnan(probs).any()))
sub.head()
