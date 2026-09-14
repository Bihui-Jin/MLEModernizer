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

0.3389334953593508

# 6. Current score

1.39906

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the EEG reshape bug by making `proc_eeg()` robust to slight length mismatches after filtering/binning (it currently assumes exactly 2500 time steps, but some files end up with a different length). The fix crop or pad the concatenated EEG/mid/EKG signals to exactly 2500 samples so the downstream model input shape stays unchanged. This is a correctness/stability fix (not a modeling change) and allow feature precomputation, inference, and submission writing to complete. I also add a small safety guard so the fallback model always returns the correct shape log-probabilities even if the input is missing.'
- What this solution (achieved 1.40741) has done: 'Your current score (1.40995, lower is better) is far worse than the target (0.33893), so we should make the smallest changes that legitimately improve the KL metric without changing the model architecture or training. The biggest likely issue is a mismatch between what the external models expect (input scaling/clipping) and what this inference pipeline feeds them; I keep your feature extraction intact but add a very light “probability sharpening/softening” temperature on the *final* ensemble probabilities (metric-aligned calibration) plus an optional blend with the class prior from `train.csv` to reduce extreme KL penalties when the model is miscalibrated. Both changes preserve semantics (still valid probabilities summing to 1) and are easy to tune toward the target; by default I choose conservative values that typically improve KL rather than chase the best score. If the external models aren’t available and you’re using the uniform fallback, these changes won’t magically reach the target, but they still keep the submission valid.'
- What this solution (achieved 1.40618) has done: 'Your score (1.40741, lower-is-better) is far worse than the target (0.33893), so we should make the smallest legitimate changes that improve KL without touching the model or feature extraction. The most likely culprit is overconfident/miscalibrated probabilities; KL heavily penalizes putting near-zero probability on the true class, so we slightly increase smoothing toward the train prior and soften the distribution a bit more via temperature. I keep your existing calibration block but adjust `PRIOR_BLEND` and `TEMP` to be more conservative (reduce extremes), while preserving valid probabilities summing to 1 and the same end-to-end submission workflow. No architecture/training/feature logic is changed.'
- What this solution (achieved 1.40913) has done: 'Your current score (1.40618, lower-is-better) is far from the target (0.33893), so we should improve KL in the smallest way without touching the model/feature logic. The biggest KL failure mode is overconfident near-zero probabilities; instead of pushing harder toward the global prior, we do metric-aligned smoothing with a per-row “floor” (Dirichlet-style/additive smoothing) and a slightly softer temperature, which reduces catastrophic penalties while keeping probabilities valid. We also make the train prior more representative by computing it per `eeg_id` (to avoid overweighting heavily-overlapped recordings), which is a small correctness/calibration improvement. Everything else (feature extraction, model inference, ensembling) stays identical and it still writes a valid `submission.csv`.'
- What this solution (achieved 1.40758) has done: 'Your score is much worse than the target (lower-is-better), so the smallest legitimate improvement is to reduce KL’s harsh penalties from near-zero probabilities by doing a bit more conservative probability smoothing and softening at the very end (post-ensemble), without touching model inference or feature extraction. I keep your current “train prior per eeg_id” calibration block but increase the per-row probability floor slightly (Dirichlet-style pseudo-count via the prior) and soften a bit more via temperature, which typically improves KL when predictions are miscalibrated/overconfident. I also add a tiny uniform floor mix to guarantee no class ever goes extremely small even if the prior is skewed, while still keeping probabilities valid and summing to 1. Everything else (data, feature pipeline, model loading, ensembling, submission writing) remains unchanged.'
- What this solution (achieved 1.40147) has done: 'Your current score (1.40758, lower-is-better) is far above the target (0.33893), so we should make the smallest change that can reduce KL without touching feature extraction or model inference. The KL metric is especially punishing when any class probability is near-zero, so I increase the final-stage Dirichlet-style smoothing toward a robust train prior and also slightly increase the uniform floor to prevent catastrophic zeros. To avoid washing everything out, I simultaneously reduce the temperature softening a bit (less flattening), keeping probabilities better-shaped while still safe. These are purely post-processing calibration changes; the model/feature pipeline stays identical and the submission remains valid.'
- What this solution (achieved 1.39906) has done: 'Your score is far worse than the target (lower-is-better), so the smallest legitimate improvement is to make the final probability calibration more KL-friendly without touching feature extraction or model inference. KL is harsh when any predicted class is near-zero, so I increase the final Dirichlet-style smoothing slightly and add a tiny post-temperature probability floor to further prevent extreme minima. I keep your existing “per-eeg_id prior” logic and temperature scaling intact (same semantics: post-processing only), and only adjust the calibration constants plus a final floor-and-renormalize for extra numerical safety. This should move the score downward toward the target while preserving the core pipeline and producing the same submission format.'

# 9. Code solution

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

pl.Config.set_tbl_rows(20)
try:
    pl.Config.set_engine("pyarrow")
except Exception:
    pass

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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
torch.set_num_threads(1)

if device.type == "cuda":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(False)




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
        if eeg is None:
            b = 1
        else:
            b = int(eeg.shape[0])
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


_EEG_SPEC_COLS = [
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


def _pl_to_numpy_fast(df: pl.DataFrame, cols: List[str]) -> np.ndarray:
    try:
        return df.select(cols).to_numpy(zero_copy_only=True)
    except Exception:
        return df.select(cols).to_numpy()


def compute_spec_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    arr = _pl_to_numpy_fast(df_eeg, _EEG_SPEC_COLS)
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


_EEG_EEG_COLS = [
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


def compute_eeg_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    arr = _pl_to_numpy_fast(df_eeg, _EEG_EEG_COLS)
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

import cv2

_RESIZE_H = 96
_RESIZE_W = 224
_RESIZE_INTERP = cv2.INTER_CUBIC


def proc_kspec(x):
    x = x.copy()
    x = x[:, 2:98]
    bad = ~np.isfinite(x)
    if bad.any():
        x[bad] = 0
    x = x.clip(np.exp(-4), np.exp(7))
    x = np.log(x)

    x = x - x.mean(axis=(1, 2), keepdims=True)
    x = x / (x.std(axis=(1, 2), keepdims=True) + 1e-5)

    x = x.transpose(1, 2, 0)  # (freq, time, 4)
    x = cv2.resize(
        x.astype(np.float32, copy=False),
        (_RESIZE_W, _RESIZE_H),
        interpolation=_RESIZE_INTERP,
    )
    x = x.transpose(2, 0, 1)  # (4, 96, 224)
    x = x.reshape(4, 96, 224)
    return x


def _resize_or_pad_time_to_224(x2d: np.ndarray, target_t: int = 224) -> np.ndarray:
    F, T = x2d.shape
    if T == target_t:
        return x2d
    x_img = x2d.astype(np.float32, copy=False)  # (F, T)
    x_img = cv2.resize(x_img, (target_t, F), interpolation=_RESIZE_INTERP)
    return x_img


def proc_eeg_spec(x):
    x = x.copy()  # expected (4, time, freq~96)
    x = x[:, :, 2:98]  # -> (4, time, 96)
    bad = ~np.isfinite(x)
    if bad.any():
        x[bad] = 0
    x = x + 1

    x = x.transpose(0, 2, 1).astype(np.float32, copy=False)
    out = np.empty((4, 96, 224), dtype=np.float32)
    for c in range(4):
        out[c] = _resize_or_pad_time_to_224(x[c], target_t=224)
    return out


def _fix_len_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    """Bugfix: ensure exact length for concatenation/reshape (crop or edge-pad)."""
    x = np.asarray(x)
    if x.shape[-1] == target_len:
        return x
    if x.shape[-1] > target_len:
        return x[..., :target_len]
    pad = target_len - x.shape[-1]
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)], mode="edge")


def proc_eeg(eeg, mid, ekg, target_len: int = 2500):
    eeg = eeg.copy()
    mid = mid.copy()
    ekg = ekg.copy()

    bad = ~np.isfinite(eeg)
    if bad.any():
        eeg[bad] = 0
    bad = ~np.isfinite(ekg)
    if bad.any():
        ekg[bad] = 0
    bad = ~np.isfinite(mid)
    if bad.any():
        mid[bad] = 0

    eeg = eeg - eeg.mean(axis=-1, keepdims=True)
    mid = mid - mid.mean(axis=-1, keepdims=True)

    mad_std = MAD(eeg, axis=-1).reshape(-1)
    mad_std = np.median(mad_std) + 1e-5

    eeg = eeg / mad_std
    eeg = eeg.clip(-10, 10)

    mid = mid / mad_std
    mid = mid.clip(-10, 10)

    ekg = ekg / (MAD(ekg, axis=-1).reshape(-1) + 1e-5)

    eeg = _fix_len_1d(eeg.reshape(16, -1), target_len)
    mid = _fix_len_1d(mid.reshape(2, -1), target_len)
    ekg = _fix_len_1d(ekg.reshape(1, -1), target_len)

    eeg = np.concatenate([eeg, mid, ekg], axis=0)
    eeg = eeg.reshape(19, target_len)
    return eeg


from collections import OrderedDict


class LRUCache:
    def __init__(self, maxsize: int):
        self.maxsize = int(maxsize)
        self._d = OrderedDict()

    def get(self, k):
        if k in self._d:
            self._d.move_to_end(k)
            return self._d[k]
        return None

    def set(self, k, v):
        self._d[k] = v
        self._d.move_to_end(k)
        if len(self._d) > self.maxsize:
            self._d.popitem(last=False)


_EEG_DF_CACHE = LRUCache(maxsize=64)  # eeg_id -> polars df
_EEG_FEATURE_CACHE = LRUCache(maxsize=512)  # eeg_id -> (eeg_proc, eeg_spec_proc)
_KSPEC_CACHE = LRUCache(maxsize=1024)  # spectrogram_id -> kspec_proc


def _get_eeg_df(eeg_id: int) -> pl.DataFrame:
    df = _EEG_DF_CACHE.get(eeg_id)
    if df is None:
        eeg_filepath = os.path.join(EEG_DIR, f"{eeg_id}.parquet")
        df = pl.read_parquet(eeg_filepath).fill_null(0)
        _EEG_DF_CACHE.set(eeg_id, df)
    return df


def _compute_eeg_features_for_eeg_id(eeg_id: int) -> Tuple[np.ndarray, np.ndarray]:
    cached = _EEG_FEATURE_CACHE.get(eeg_id)
    if cached is not None:
        return cached

    df_eeg = _get_eeg_df(eeg_id)

    eeg_spec = compute_spec_chain(df_eeg)
    eeg_spec = proc_eeg_spec(eeg_spec)

    eeg, mid, ekg = compute_eeg_chain(df_eeg)
    bad = ~np.isfinite(eeg)
    if bad.any():
        eeg[bad] = 0
    eeg = proc_eeg(eeg, mid, ekg)

    _EEG_FEATURE_CACHE.set(eeg_id, (eeg, eeg_spec))
    return eeg, eeg_spec


def _compute_kspec_for_spc_id(spc_id: int) -> np.ndarray:
    cached = _KSPEC_CACHE.get(spc_id)
    if cached is not None:
        return cached
    spc_filepath = os.path.join(SPEC_DIR, f"{spc_id}.parquet")
    kspec = compute_kaggle_spec_from_file(spc_filepath)
    kspec = proc_kspec(kspec)
    _KSPEC_CACHE.set(spc_id, kspec)
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

EEG_FEATS: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
KSPEC_FEATS: Dict[int, np.ndarray] = {}

for eeg_id in tqdm(uniq_eeg_ids.tolist(), desc="Precompute EEG features"):
    eid = int(eeg_id)
    EEG_FEATS[eid] = _compute_eeg_features_for_eeg_id(eid)

for spc_id in tqdm(uniq_spc_ids.tolist(), desc="Precompute KSpec features"):
    sid = int(spc_id)
    KSPEC_FEATS[sid] = _compute_kspec_for_spc_id(sid)


@torch.inference_mode()
def gen_ensemble_pred_batch(
    eeg_ids: np.ndarray, spc_ids: np.ndarray, batch_size: int = 16
) -> np.ndarray:
    n = len(eeg_ids)
    out_preds = np.empty((n, 6), dtype=np.float32)

    pin_memory = device.type == "cuda"
    n_models = float(len(models_multimodal))

    for start in tqdm(range(0, n, batch_size), desc="Model inference"):
        end = min(n, start + batch_size)
        bs = end - start

        eeg_np = np.empty((bs, 19, 2500), dtype=np.float32)
        eeg_spec_np = np.empty((bs, 4, 96, 224), dtype=np.float32)
        kspec_np = np.empty((bs, 4, 96, 224), dtype=np.float32)

        for j in range(bs):
            eeg_id = int(eeg_ids[start + j])
            spc_id = int(spc_ids[start + j])
            eeg, eeg_spec = EEG_FEATS[eeg_id]
            kspec = KSPEC_FEATS[spc_id]
            eeg_np[j] = eeg
            eeg_spec_np[j] = eeg_spec
            kspec_np[j] = kspec

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

        preds = preds_sum / n_models
        preds = np.clip(preds, 1e-12, None)
        preds = preds / preds.sum(axis=1, keepdims=True)
        out_preds[start:end] = preds.astype(np.float32)

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
train_group = (
    df_train.select(["eeg_id"] + LABELS)
    .group_by("eeg_id")
    .agg([pl.col(c).mean().alias(c) for c in LABELS])
)
train_votes = train_group.select(LABELS).to_numpy().astype(np.float64)
train_votes_sum = train_votes.sum(axis=1, keepdims=True)
train_votes_sum = np.clip(train_votes_sum, 1.0, None)
train_probs = train_votes / train_votes_sum
prior = train_probs.mean(axis=0)
prior = np.clip(prior, 1e-12, None)
prior = prior / prior.sum()
print("Train prior (per-eeg_id mean):", prior)

EPS_PRIOR = 0.28  # was 0.18; stronger prior smoothing tends to reduce KL when model is miscalibrated
EPS_UNI = (
    0.030  # was 0.015; slightly larger uniform floor further avoids extreme minima
)
TEMP = 1.35  # was 1.45; a bit less softening to keep some model signal while still calibrated
FINAL_FLOOR = (
    2e-3  # new: tiny post-temp additive floor to prevent any class becoming too small
)

p = preds_final.astype(np.float64, copy=True)
p = np.clip(p, 1e-12, None)
p = p / p.sum(axis=1, keepdims=True)

p = (p + EPS_PRIOR * prior.reshape(1, -1)) / (1.0 + EPS_PRIOR)
p = (p + EPS_UNI * (1.0 / p.shape[1])) / (1.0 + EPS_UNI)

logp = np.log(np.clip(p, 1e-12, None)) / TEMP
logp = logp - logp.max(axis=1, keepdims=True)
p = np.exp(logp)
p = np.clip(p, 1e-12, None)
p = p / p.sum(axis=1, keepdims=True)

p = p + FINAL_FLOOR
p = p / p.sum(axis=1, keepdims=True)

preds_final = p.astype(np.float32)
print(
    "After calibration row_sum min/max:",
    preds_final.sum(1).min(),
    preds_final.sum(1).max(),
)



## === cell 14
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
assert df_sub.shape[0] == df_test.shape[0], "Row count mismatch vs test.csv"
assert list(df_sub.columns) == ["eeg_id"] + LABELS, "Submission columns mismatch"
row_sums = df_sub[LABELS].sum(axis=1).to_numpy()
assert np.all(np.isfinite(row_sums)), "Non-finite probabilities"
assert np.max(np.abs(row_sums - 1.0)) < 1e-5, "Probabilities do not sum to 1"
print("Submission sanity checks passed.")
