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

cupy-cuda12x==13.6.0
fastai==2.8.5
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
import os
import random
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from fastai.vision.all import *
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import gc
import shutil
from collections import OrderedDict

import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

PARQUET_ENGINE = "pyarrow"
try:
    import pyarrow  # noqa: F401
except Exception:
    PARQUET_ENGINE = "auto"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


seed_everything(2024)


def _seed_worker(worker_id):
    base = 2024
    s = base + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


_G = torch.Generator()
_G.manual_seed(2024)



## === cell 1
PATH = {
    "test": "/kaggle/input/hms-harmful-brain-activity-classification/test_",
    "train": "/kaggle/input/hms-harmful-brain-activity-classification/train_",
}
BS = 512
models = {
    "eeg": [
        "/kaggle/input/eegs-cv-kaggle/EEGS_CNN_0",
        "/kaggle/input/eegs-cv-kaggle/EEGS_CNN_1",
        "/kaggle/input/eegs-cv-kaggle/EEGS_CNN_2",
        "/kaggle/input/eegs-cv-kaggle/EEGS_CNN_3",
        "/kaggle/input/eegs-cv-kaggle/EEGS_CNN_4",
    ],
    "spec": [
        "/kaggle/input/specs-cv-kaggle/SPECS_CNN_0",
        "/kaggle/input/specs-cv-kaggle/SPECS_CNN_1",
        "/kaggle/input/specs-cv-kaggle/SPECS_CNN_2",
        "/kaggle/input/specs-cv-kaggle/SPECS_CNN_3",
        "/kaggle/input/specs-cv-kaggle/SPECS_CNN_4",
    ],
    "spec_fixed": [
        "/kaggle/input/specs-cv-kaggle-fixed/SPECS_CNN_0",
        "/kaggle/input/specs-cv-kaggle-fixed/SPECS_CNN_1",
        "/kaggle/input/specs-cv-kaggle-fixed/SPECS_CNN_2",
        "/kaggle/input/specs-cv-kaggle-fixed/SPECS_CNN_3",
        "/kaggle/input/specs-cv-kaggle-fixed/SPECS_CNN_4",
    ],
    "custom": [
        "/kaggle/input/custom-cv-kaggle/custom_SPECS_CNN_0",
        "/kaggle/input/custom-cv-kaggle/custom_SPECS_CNN_1",
        "/kaggle/input/custom-cv-kaggle/custom_SPECS_CNN_2",
        "/kaggle/input/custom-cv-kaggle/custom_SPECS_CNN_3",
        "/kaggle/input/custom-cv-kaggle/custom_SPECS_CNN_4",
    ],
    "custom_fixed": [
        "/kaggle/input/custom-cv-kaggle-fixed/custom_SPECS_CNN_0",
        "/kaggle/input/custom-cv-kaggle-fixed/custom_SPECS_CNN_1",
        "/kaggle/input/custom-cv-kaggle-fixed/custom_SPECS_CNN_2",
        "/kaggle/input/custom-cv-kaggle-fixed/custom_SPECS_CNN_3",
        "/kaggle/input/custom-cv-kaggle-fixed/custom_SPECS_CNN_4",
    ],
    "HMS": [
        "/kaggle/input/hms-xtra/HMSmodel_0",
        "/kaggle/input/hms-xtra/HMSmodel_1",
        "/kaggle/input/hms-xtra/HMSmodel_2",
        "/kaggle/input/hms-xtra/HMSmodel_3",
        "/kaggle/input/hms-xtra/HMSmodel_4",
    ],
}
DEBUG = False



## === cell 2
TH = 5  # Percentile of pseudolabels to take
batch_size = 64
min_votes = 0
LR = 1e-4
EPOCHS = 1
start_p = 0.5
end_p = 1
start_min_v = 1
end_min_v = 1
label_aug = False
N_FOLDS = 5
FOLDS = [0, 1, 2, 3, 4]
HMS_DROP = 0.8
CV = "eeg_id"
GB = "votation"




## === cell 3
def normalize(spec, epsilon=1e-6, NATURAL=False):
    if NATURAL:
        spec = np.clip(spec, np.exp(-4), np.exp(8))
        spec = np.log(spec)
    else:
        spec = np.clip(spec, np.exp(-4), np.exp(6))
        spec = np.log10(spec)

    mask = ~np.isnan(spec)
    mean = np.mean(spec[mask])
    std = np.std(spec[mask])
    spec[mask] = spec[mask] - mean
    if std > 0:
        spec[mask] /= std + epsilon

    return spec


def _normalize_inplace_log10(x: np.ndarray, epsilon: float = 1e-6):
    np.clip(x, np.exp(-4), np.exp(6), out=x)
    np.log10(x, out=x)
    mask = ~np.isnan(x)
    if mask.all():
        mean = float(x.mean())
        std = float(x.std())
        x -= mean
        if std > 0:
            x /= std + epsilon
    else:
        if mask.any():
            xv = x[mask]
            mean = float(xv.mean())
            std = float(xv.std())
            x[mask] = xv - mean
            if std > 0:
                x[mask] /= std + epsilon
    return x




## === cell 4
from scipy.signal import butter, sosfiltfilt

_SOS_CACHE = {}


def butter_lowpass_filter(data, cutoff_freq=20, sampling_rate=200, order=4):
    key = (cutoff_freq, sampling_rate, order)
    sos = _SOS_CACHE.get(key, None)
    if sos is None:
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        sos = butter(order, normal_cutoff, btype="low", analog=False, output="sos")
        _SOS_CACHE[key] = sos
    return sosfiltfilt(sos, data, axis=-1)




## === cell 5
class HMSmodel(nn.Module):

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.FC = nn.Linear(1280 * len(models), 6).to(device)

        for model in self.models:
            for p in model.parameters():
                p.requires_grad = False

        weights = []
        bias = 0
        for model in self.models:
            model.features[8][0].weight.requires_grad = True
            model.classifier[0] = nn.Dropout(HMS_DROP)
            weights.append(model.classifier[1].weight)
            bias += model.classifier[1].bias
            model.classifier[1] = nn.Identity()

        self.FC.weight = nn.Parameter(torch.cat(weights, 1).to(device))
        self.FC.bias = nn.Parameter(bias.to(device))

    def forward(self, X):
        eeg, spec, custom_spec = X
        X = torch.cat([self.models[i](X[i]) for i in range(len(self.models))], 1)
        OUT = self.FC(X)
        return OUT




## === cell 6
submission = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
submission.head()



## === cell 7
votes = [c for c in submission.columns if "_vote" in c]
votes



## === cell 8
test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
test.head()



## === cell 9
if DEBUG:
    test = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )[:1000]
    PATH["test"] = PATH["train"]
    test.head()



## === cell 10
i = np.random.randint(len(test))
row = test[i : i + 1]
row



## === cell 11
from scipy.signal import (
    iirnotch,
    filtfilt as scipy_filtfilt,
    butter as scipy_butter,
    spectrogram as scipy_spectrogram,
)
from scipy.ndimage import gaussian_filter


def create_spectrogram_cpu(
    eeg_data,
    low_cut_freq=0.7,
    high_cut_freq=20,
    order_band=5,
    nperseg=500,
    noverlap=200,
    nfft=1024,
    sigma_gaussian=0.7,
    mean_montage_names=4,
    fs=200,
):
    electrode_pair_name_locations = {
        "LL": ["Fp1", "F7", "T3", "T5", "O1"],
        "RL": ["Fp2", "F8", "T4", "T6", "O2"],
        "LP": ["Fp1", "F3", "C3", "P3", "O1"],
        "RP": ["Fp2", "F4", "C4", "P4", "O2"],
    }

    nyquist_freq = 0.5 * fs
    low_cut_freq_normalized = low_cut_freq / nyquist_freq
    high_cut_freq_normalized = high_cut_freq / nyquist_freq

    notch_ba = iirnotch(w0=60, Q=30, fs=fs)
    band_ba = scipy_butter(
        order_band, [low_cut_freq_normalized, high_cut_freq_normalized], btype="band"
    )

    spectrogram_accum = None
    processed_eeg = {}

    for i, (montage_name, locs) in enumerate(electrode_pair_name_locations.items()):
        montage_sum = np.zeros(len(eeg_data), dtype=np.float32)

        for j in range(4):
            x = (
                eeg_data[locs[j]].to_numpy() - eeg_data[locs[j + 1]].to_numpy()
            ).astype(np.float32)
            mean_x = np.nanmean(x) if np.isnan(x).mean() < 1 else 0.0
            x = np.nan_to_num(x, nan=mean_x)

            x = scipy_filtfilt(*notch_ba, x)
            x = scipy_filtfilt(*band_ba, x)

            f, t, Sxx = scipy_spectrogram(
                x,
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                scaling="density",
                mode="psd",
            )
            valid = (f >= 0.59) & (f <= 20)
            Sxx = Sxx[valid, :].astype(np.float32)

            Sxx = np.clip(Sxx, np.exp(-4), np.exp(6))
            Sxx = np.log10(Sxx)

            mean = Sxx.mean()
            std = Sxx.std()
            if std > 0:
                Sxx = (Sxx - mean) / (std + 1e-6)
            else:
                Sxx = Sxx - mean

            if spectrogram_accum is None:
                spectrogram_accum = np.zeros(
                    (Sxx.shape[0], Sxx.shape[1], 4), dtype=np.float32
                )

            spectrogram_accum[:, :, i] += Sxx
            processed_eeg[f"{locs[j]}_{locs[j+1]}"] = x
            montage_sum += x

        if mean_montage_names > 0:
            spectrogram_accum[:, :, i] /= mean_montage_names
        processed_eeg[montage_name] = montage_sum

    if sigma_gaussian and sigma_gaussian > 0:
        spectrogram_accum = gaussian_filter(
            spectrogram_accum, sigma=(sigma_gaussian, sigma_gaussian, 0)
        )

    if "EKG" in eeg_data.columns:
        ekg = eeg_data["EKG"].to_numpy().astype(np.float32)
        mean_ekg = np.nanmean(ekg) if np.isnan(ekg).mean() < 1 else 0.0
        ekg = np.nan_to_num(ekg, nan=mean_ekg)
        ekg = scipy_filtfilt(*notch_ba, ekg)
        ekg = scipy_filtfilt(*band_ba, ekg)
        processed_eeg["EKG"] = ekg

    return spectrogram_accum, processed_eeg




## === cell 12
custom_dir = "/kaggle/working/custom"
os.makedirs(custom_dir, exist_ok=True)

custom_train_dir = "/kaggle/working/custom_train"
os.makedirs(custom_train_dir, exist_ok=True)


def _find_precomp_custom_dirs():
    candidates = [
        "/kaggle/input/custom-specs/custom",
        "/kaggle/input/custom-specs/custom_train",
        "/kaggle/input/custom-specs",
        "/kaggle/input/hms-harmful-brain-activity-classification/custom",
    ]
    try:
        for d in os.listdir("/kaggle/input"):
            p = os.path.join("/kaggle/input", d)
            if os.path.isdir(os.path.join(p, "custom")):
                candidates.append(os.path.join(p, "custom"))
            if os.path.isdir(os.path.join(p, "custom_specs")):
                candidates.append(os.path.join(p, "custom_specs"))
    except Exception:
        pass
    out = []
    for p in candidates:
        if os.path.isdir(p):
            out.append(p)
    seen = set()
    out2 = []
    for p in out:
        if p not in seen:
            out2.append(p)
            seen.add(p)
    return out2


CUSTOM_PRECOMP_DIRS = _find_precomp_custom_dirs()
print("Found precomputed custom spec dirs:", CUSTOM_PRECOMP_DIRS)

gc.collect()



## === cell 13
_EXISTS_CACHE = {}


def _exists_cached(path: str) -> bool:
    v = _EXISTS_CACHE.get(path)
    if v is None:
        v = os.path.exists(path)
        _EXISTS_CACHE[path] = v
    return v


def _find_custom_npy_anywhere(eeg_id: int):
    fname = f"{eeg_id}.npy"
    for d in CUSTOM_PRECOMP_DIRS:
        p = os.path.join(d, fname)
        if _exists_cached(p):
            return p
    return None


def get_custom_spec_path(origin: str, eeg_id: int) -> str:
    p = _find_custom_npy_anywhere(eeg_id)
    if p is not None:
        return p
    if origin == "train":
        return os.path.join(custom_train_dir, f"{eeg_id}.npy")
    else:
        return os.path.join(custom_dir, f"{eeg_id}.npy")


def ensure_custom_spec_exists(origin: str, eeg_id: int):
    p = _find_custom_npy_anywhere(eeg_id)
    if p is not None:
        return p
    return get_custom_spec_path(origin, eeg_id)




## === cell 14
gc.collect()

_TEST_EEG_NUMPY_CACHE = {}
_TEST_SPEC_NUMPY_CACHE = {}
_CUSTOM_NPY_CACHE = {}
_SPEC_COLIDX_CACHE = {}  # spectrogram_id -> (LP_i, LL_i, RP_i, RL_i)
_SPEC_COLIDX_BY_SIG = {}  # tuple(columns) -> (LP_i, LL_i, RP_i, RL_i)

_TEST_EEG_FEAT_CACHE = {}


def _load_custom_npy_cached(path: str):
    arr = _CUSTOM_NPY_CACHE.get(path)
    if arr is None:
        arr = np.load(path)
        _CUSTOM_NPY_CACHE[path] = arr
    return arr


_EEG_COLS_8 = ["Fp1", "T3", "O1", "C3", "Fp2", "C4", "O2", "T4"]


def _read_eeg_np_from_parquet(path: str) -> np.ndarray:
    df = pd.read_parquet(path, engine=PARQUET_ENGINE, columns=_EEG_COLS_8)
    return df.to_numpy(dtype=np.float32, copy=False)  # [T,8]


def _spec_build_colidx_from_cols(cols):
    LP, LL, RP, RL = [], [], [], []
    for i, c in enumerate(cols):
        if "LP" in c:
            LP.append(i)
        elif "LL" in c:
            LL.append(i)
        elif "RP" in c:
            RP.append(i)
        elif "RL" in c:
            RL.append(i)
    return (
        np.asarray(LP, dtype=np.int32),
        np.asarray(LL, dtype=np.int32),
        np.asarray(RP, dtype=np.int32),
        np.asarray(RL, dtype=np.int32),
    )


def _generate_custom_spec_from_eeg_id(origin: str, eeg_id: int):
    out_path = get_custom_spec_path(origin, eeg_id)
    if _exists_cached(out_path):
        return out_path
    eeg_df = pd.read_parquet(
        PATH[origin] + "eegs/" + str(eeg_id) + ".parquet", engine=PARQUET_ENGINE
    )
    spec_accum, _ = create_spectrogram_cpu(eeg_df)
    np.save(out_path, spec_accum.astype(np.float32, copy=False))
    _EXISTS_CACHE[out_path] = True
    return out_path


def _resize2d_linear(x2d: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    out_h, out_w = out_hw
    return cv2.resize(x2d, (out_w, out_h), interpolation=cv2.INTER_LINEAR).astype(
        np.float32, copy=False
    )


class HMS_DS(torch.utils.data.Dataset):
    """Test dataset"""

    def __init__(self, df):
        self.START = (10000 - 2048) // 2
        self.data = df.reset_index(drop=True)

        self._eeg_ids = self.data["eeg_id"].to_numpy(dtype=np.int64, copy=False)
        self._spec_ids = self.data["spectrogram_id"].to_numpy(
            dtype=np.int64, copy=False
        )

        self.eeg = np.zeros((8, 2048), dtype=np.float32)
        self.spec = np.zeros((1, 256, 256), dtype=np.float32)
        self.custom_placeholder = np.zeros((256, 256), dtype=np.float32)

    def __len__(self):
        return len(self._eeg_ids)

    def _read_test_eeg_np(self, eeg_id: int):
        eeg_np = _TEST_EEG_NUMPY_CACHE.get(eeg_id)
        if eeg_np is None:
            eeg_np = _read_eeg_np_from_parquet(
                PATH["test"] + "eegs/" + str(eeg_id) + ".parquet"
            )
            _TEST_EEG_NUMPY_CACHE[eeg_id] = eeg_np
        return eeg_np

    def _read_test_spec_np_and_idx(self, spectrogram_id: int):
        v = _TEST_SPEC_NUMPY_CACHE.get(spectrogram_id)
        if v is None:
            spec_df = pd.read_parquet(
                PATH["test"] + "spectrograms/" + str(spectrogram_id) + ".parquet",
                engine=PARQUET_ENGINE,
            )
            spec_np = spec_df.to_numpy(dtype=np.float32, copy=False)
            idx = _SPEC_COLIDX_CACHE.get(spectrogram_id)
            if idx is None:
                cols = tuple(spec_df.columns)
                idx = _SPEC_COLIDX_BY_SIG.get(cols)
                if idx is None:
                    idx = _spec_build_colidx_from_cols(cols)
                    _SPEC_COLIDX_BY_SIG[cols] = idx
                _SPEC_COLIDX_CACHE[spectrogram_id] = idx
            v = (spec_np, idx)
            _TEST_SPEC_NUMPY_CACHE[spectrogram_id] = v
        return v

    def _get_or_build_eeg_feat(self, eeg_id: int) -> np.ndarray:
        feat = _TEST_EEG_FEAT_CACHE.get(eeg_id)
        if feat is not None:
            return feat
        eeg_np = self._read_test_eeg_np(eeg_id)  # [T,8]
        seg = eeg_np[self.START : self.START + 2048]  # [2048,8]

        fp1 = seg[:, 0]
        mask = ~np.isnan(fp1)

        eeg = np.zeros((8, 2048), dtype=np.float32)
        eeg[0, mask] = seg[mask, 0] - seg[mask, 1]
        eeg[1, mask] = seg[mask, 1] - seg[mask, 2]
        eeg[2, mask] = seg[mask, 0] - seg[mask, 3]
        eeg[3, mask] = seg[mask, 3] - seg[mask, 2]
        eeg[4, mask] = seg[mask, 4] - seg[mask, 5]
        eeg[5, mask] = seg[mask, 5] - seg[mask, 6]
        eeg[6, mask] = seg[mask, 4] - seg[mask, 7]
        eeg[7, mask] = seg[mask, 7] - seg[mask, 6]

        eeg[:, :] = np.clip(eeg, -1024, 1024) / 32.0
        eeg[:, :] = butter_lowpass_filter(eeg).astype(np.float32, copy=False)

        _TEST_EEG_FEAT_CACHE[eeg_id] = eeg
        return eeg

    def __getitem__(self, idx):
        eeg_id = int(self._eeg_ids[idx])
        spectrogram_id = int(self._spec_ids[idx])

        eeg_feat = self._get_or_build_eeg_feat(eeg_id)
        eeg_t = torch.from_numpy(eeg_feat).float()

        spec_np, (LP_i, LL_i, RP_i, RL_i) = self._read_test_spec_np_and_idx(
            spectrogram_id
        )
        t0 = 22
        block = spec_np[t0 : t0 + 256]  # [256, all_cols]

        block_r = _resize2d_linear(block, (256, 300))  # [256,300]
        a = block_r[:, LP_i].copy()
        b = block_r[:, LL_i].copy()
        c = block_r[:, RP_i].copy()
        d = block_r[:, RL_i].copy()
        _normalize_inplace_log10(a)
        _normalize_inplace_log10(b)
        _normalize_inplace_log10(c)
        _normalize_inplace_log10(d)

        self.spec[0, :, :64] = a
        self.spec[0, :, 64:128] = b
        self.spec[0, :, 128:-64] = c
        self.spec[0, :, -64:] = d

        mask2 = np.isnan(self.spec)
        if mask2.any():
            self.spec[mask2] = 0.0
        spec_t = torch.from_numpy(self.spec.copy()).float()

        custom_path = _find_custom_npy_anywhere(eeg_id)
        if custom_path is None:
            self.spec[0] = self.custom_placeholder
            custom_t = torch.from_numpy(self.spec.copy()).float()
        else:
            custom_spec = _load_custom_npy_cached(custom_path)

            custom_spec = np.concatenate(
                (
                    custom_spec[:, :, 3],
                    custom_spec[:, :, 2],
                    custom_spec[:, :, 1],
                    custom_spec[:, :, 0],
                )
            )
            custom_spec = _resize2d_linear(custom_spec, (256, 300))
            self.spec[0] = custom_spec[:, t0 : t0 + 256]
            mask3 = np.isnan(self.spec)
            if mask3.any():
                self.spec[mask3] = 0.0
            custom_t = torch.from_numpy(self.spec.copy()).float()

        return eeg_id, spectrogram_id, eeg_t, spec_t, custom_t




## === cell 15
def safe_torch_load(path, map_location="cpu"):
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location=map_location)
    except TypeError:
        return torch.load(path)


def load_ensemble(paths):
    ens = []
    for p in paths:
        m = safe_torch_load(p, map_location=device)
        if m is None:
            continue
        m.eval()
        ens.append(m)
    return ens


def ensemble_predict_proba(
    eeg_ensemble,
    spec_ensemble,
    custom_ensemble,
    hms_ensemble,
    eegs_b,
    specs_b,
    custom_specs_b,
):
    preds_list = []
    for model in eeg_ensemble:
        preds_list.append(torch.softmax(model(eegs_b), -1))
    for model in spec_ensemble:
        preds_list.append(torch.softmax(model(specs_b), -1))
    for model in custom_ensemble:
        preds_list.append(torch.softmax(model(custom_specs_b), -1))
    for model in hms_ensemble:
        preds_list.append(torch.softmax(model([eegs_b, specs_b, custom_specs_b]), -1))

    if len(preds_list) == 0:
        return None

    probs = torch.stack(preds_list, dim=1).mean(1)
    probs = probs / probs.sum(-1, keepdim=True)
    return probs


def ensemble_sigma_weighted(preds_stack, probs_mean, eps=1e-8):
    std = preds_stack.std(1)
    denom = probs_mean.sum(-1).clamp_min(eps)
    sigma = (probs_mean * std).sum(-1) / denom
    sigma = torch.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    return sigma




## === cell 16
def _suggest_num_workers():
    try:
        cpu = os.cpu_count() or 2
    except Exception:
        cpu = 2
    return min(8, max(2, cpu // 2))


INFER_WORKERS = _suggest_num_workers()



## === cell 17
gc.collect()



## === cell 18
submission_pred_rows = []
submission_pred = pd.DataFrame(columns=["eeg_id", "spectrogram_id"] + votes + ["sigma"])

if len(test) > 0:
    ds = HMS_DS(test)
    dl = DataLoader(
        ds,
        batch_size=BS,
        shuffle=False,
        num_workers=INFER_WORKERS,
        pin_memory=(device == "cuda"),
        persistent_workers=(INFER_WORKERS > 0),
        prefetch_factor=4 if INFER_WORKERS > 0 else None,
        worker_init_fn=_seed_worker,
        generator=_G,
    )

    eeg_ensemble = load_ensemble(models["eeg"])
    spec_ensemble = load_ensemble(models["spec"])
    custom_ensemble = load_ensemble(models["custom"])
    HMS_ensemble = load_ensemble(models["HMS"])

    total_models = (
        len(eeg_ensemble)
        + len(spec_ensemble)
        + len(custom_ensemble)
        + len(HMS_ensemble)
    )
    print(
        "Loaded models:",
        {
            "eeg": len(eeg_ensemble),
            "spec": len(spec_ensemble),
            "custom": len(custom_ensemble),
            "HMS": len(HMS_ensemble),
            "total": total_models,
        },
    )

    with torch.inference_mode():
        PREDS = (
            torch.empty((BS, total_models, 6), device=device)
            if total_models > 0
            else None
        )

        for eeg_id, spectrogram_id, eegs, specs, custom_specs in tqdm.tqdm(
            dl, desc="Inference"
        ):
            eegs = eegs.to(device, non_blocking=True)
            specs = specs.to(device, non_blocking=True)
            custom_specs = custom_specs.to(device, non_blocking=True)
            N = eegs.shape[0]

            if total_models == 0:
                mu = torch.full((N, 6), 1 / 6, device="cpu", dtype=torch.float32)
                sigma = torch.zeros((N,), device="cpu", dtype=torch.float32)
            else:
                i = 0
                for model in eeg_ensemble:
                    PREDS[:N, i] = torch.softmax(model(eegs), -1)
                    i += 1
                for model in spec_ensemble:
                    PREDS[:N, i] = torch.softmax(model(specs), -1)
                    i += 1
                for model in custom_ensemble:
                    PREDS[:N, i] = torch.softmax(model(custom_specs), -1)
                    i += 1
                for model in HMS_ensemble:
                    PREDS[:N, i] = torch.softmax(model([eegs, specs, custom_specs]), -1)
                    i += 1

                probs_mean = PREDS[:N].mean(1)
                probs_mean = probs_mean / probs_mean.sum(-1, keepdim=True)
                mu = probs_mean.cpu()
                sigma = ensemble_sigma_weighted(PREDS[:N], probs_mean).cpu()

            eeg_id_np = eeg_id.numpy().astype(np.int64, copy=False)
            spectrogram_id_np = spectrogram_id.numpy().astype(np.int64, copy=False)
            mu_np = mu.numpy()
            sigma_np = sigma.numpy()

            batch_df = pd.DataFrame(
                {
                    "eeg_id": eeg_id_np,
                    "spectrogram_id": spectrogram_id_np,
                    "sigma": sigma_np.astype(np.float32, copy=False),
                }
            )
            for k in range(6):
                batch_df[votes[k]] = mu_np[:, k].astype(np.float32, copy=False)
            submission_pred_rows.append(batch_df)

submission_pred = (
    pd.concat(submission_pred_rows, ignore_index=True)
    if len(submission_pred_rows)
    else pd.DataFrame(columns=["eeg_id", "spectrogram_id"] + votes + ["sigma"])
)
submission_pred = submission_pred[["eeg_id", "spectrogram_id"] + votes + ["sigma"]]
print(submission_pred.head())



## === cell 19
if "sigma" not in submission_pred.columns:
    submission_pred["sigma"] = 0.0
for c in votes:
    if c not in submission_pred.columns:
        submission_pred[c] = np.nan

sigma_vals = submission_pred["sigma"].to_numpy(dtype=np.float64)
sigma_vals = np.nan_to_num(sigma_vals, nan=0.0, posinf=0.0, neginf=0.0)
submission_pred["sigma"] = sigma_vals

if len(sigma_vals) == 0:
    TH_val = 0.0
elif np.allclose(sigma_vals, sigma_vals[0]):
    TH_val = float(sigma_vals[0])
else:
    TH_val = float(np.percentile(sigma_vals, TH))

pseudo_labels = submission_pred.loc[submission_pred["sigma"] <= TH_val].copy()
if len(pseudo_labels) > 0:
    pseudo_labels[votes] = pseudo_labels[votes].astype(np.float32).fillna(1.0 / 6.0)
    pseudo_labels[votes] *= 10.0
pseudo_labels.head()



## === cell 20
df = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")
df = df[df[votes].sum(1) >= 10]
df["origin"] = "train"
df.head()



## === cell 21
m = 0.55
df.expert_consensus = "T"
S = "df.loc[(df.seizure_vote > m*(df.seizure_vote + df.lpd_vote))*(df.seizure_vote > m*(df.seizure_vote + df.gpd_vote))*(df.seizure_vote > m*(df.seizure_vote + df.lrda_vote))*(df.seizure_vote > m*(df.seizure_vote + df.grda_vote))*(df.seizure_vote > m*(df.seizure_vote + df.other_vote)),'expert_consensus'] = 'seizure'"

exec(S)
for a in ["lpd", "gpd", "lrda", "grda", "other"]:
    SWAP = S.replace(a, "SWAP")
    SWAP = SWAP.replace("seizure", a)
    exec(SWAP.replace("SWAP", "seizure"))

consensus = ["seizure", "lpd", "gpd", "lrda", "grda", "other", "T"]



## === cell 22
if len(pseudo_labels) == 0:
    pseudo_labels = pd.DataFrame(
        columns=["eeg_id", "spectrogram_id"] + votes + ["sigma"]
    )
for c in votes:
    if c not in pseudo_labels.columns:
        pseudo_labels[c] = 10.0 / 6.0

pseudo_labels["expert_consensus"] = "T"
S_p = S.replace("df", "pseudo_labels")

exec(S_p)
for a in ["lpd", "gpd", "lrda", "grda", "other"]:
    SWAP = S_p.replace(a, "SWAP")
    SWAP = SWAP.replace("seizure", a)
    exec(SWAP.replace("SWAP", "seizure"))

pseudo_labels["origin"] = "test"
pseudo_labels["spectrogram_label_offset_seconds"] = 0
pseudo_labels["eeg_label_offset_seconds"] = 0
pseudo_labels.head()




## === cell 23
class _LRUCache:
    def __init__(self, maxsize=256):
        self.maxsize = int(maxsize)
        self._d = OrderedDict()

    def get(self, key):
        v = self._d.get(key, None)
        if v is not None:
            self._d.move_to_end(key)
        return v

    def set(self, key, value):
        self._d[key] = value
        self._d.move_to_end(key)
        if len(self._d) > self.maxsize:
            self._d.popitem(last=False)


_TRAIN_EEG_CACHE = _LRUCache(maxsize=256)
_TEST_EEG_CACHE_TRAINLOOP = _LRUCache(maxsize=256)
_SPEC_CACHE = _LRUCache(maxsize=256)


def _read_eeg(origin: str, eeg_id: int):
    if origin == "train":
        cached = _TRAIN_EEG_CACHE.get(eeg_id)
        if cached is not None:
            return cached
        eeg = pd.read_parquet(
            PATH["train"] + "eegs/" + str(eeg_id) + ".parquet", engine=PARQUET_ENGINE
        )
        _TRAIN_EEG_CACHE.set(eeg_id, eeg)
        return eeg
    else:
        cached = _TEST_EEG_CACHE_TRAINLOOP.get(eeg_id)
        if cached is not None:
            return cached
        eeg = pd.read_parquet(
            PATH["test"] + "eegs/" + str(eeg_id) + ".parquet", engine=PARQUET_ENGINE
        )
        _TEST_EEG_CACHE_TRAINLOOP.set(eeg_id, eeg)
        return eeg


def _read_spec(origin: str, spectrogram_id: int):
    key = (origin, spectrogram_id)
    cached = _SPEC_CACHE.get(key)
    if cached is not None:
        return cached
    spec = pd.read_parquet(
        PATH[origin] + "spectrograms/" + str(spectrogram_id) + ".parquet",
        engine=PARQUET_ENGINE,
    )
    _SPEC_CACHE.set(key, spec)
    return spec




## === cell 24
_SPEC_COLS_CACHE_TRAIN = {}
_CUSTOM_NPY_CACHE_TRAIN = {}


def _get_spec_cols_train(spec_df, origin: str, spectrogram_id: int):
    key = (origin, spectrogram_id)
    cols = _SPEC_COLS_CACHE_TRAIN.get(key)
    if cols is not None:
        return cols
    c = spec_df.columns
    LL = [x for x in c if "LL" in x]
    RL = [x for x in c if "RL" in x]
    LP = [x for x in c if "LP" in x]
    RP = [x for x in c if "RP" in x]
    cols = (LL, RL, LP, RP)
    _SPEC_COLS_CACHE_TRAIN[key] = cols
    return cols


def _load_custom_npy_cached_train(path: str):
    arr = _CUSTOM_NPY_CACHE_TRAIN.get(path)
    if arr is None:
        arr = np.load(path)
        _CUSTOM_NPY_CACHE_TRAIN[path] = arr
    return arr


def _ensure_or_generate_custom_train(origin: str, eeg_id: int):
    p = ensure_custom_spec_exists(origin, eeg_id)
    if _exists_cached(p):
        return p
    return _generate_custom_spec_from_eeg_id(origin, eeg_id)


class HMS_TRAIN_DS(torch.utils.data.Dataset):
    """ """

    def __init__(self, df, W=1024, VALID=False):
        self.W = W
        self.VALID = VALID
        self.START = (10000 - 2048) // 2
        self.data = np.array(
            df[
                [
                    "spectrogram_id",
                    "eeg_id",
                    "expert_consensus",
                    "spectrogram_label_offset_seconds",
                    "eeg_label_offset_seconds",
                    "origin",
                ]
                + votes
            ].groupby(["eeg_id", "expert_consensus"]),
            dtype=object,
        )
        if VALID:
            self.eeg = np.zeros((8, 2048), dtype=np.float32)
        else:
            self.eeg = np.zeros((8, 1024), dtype=np.float32)
        self.spec = np.zeros((1, 256, 256), dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        df_ = self.data[idx][1]
        if self.VALID:
            row = df_.iloc[len(df_) // 2]
        else:
            row = df_.iloc[np.random.randint(len(df_))]

        eeg = _read_eeg(row.origin, int(row.eeg_id))
        START = 200 * int(row.eeg_label_offset_seconds) + self.START
        if not self.VALID:
            if self.W < 2048:
                START += np.random.randint(2048 - self.W)
                eeg = eeg.iloc[START : START + self.W]
        else:
            eeg = eeg.iloc[START : START + 2048]

        mask = ~np.isnan(eeg["Fp1"])
        self.eeg[:, :] = 0

        self.eeg[0][mask] = eeg["Fp1"][mask] - eeg["T3"][mask]
        self.eeg[1][mask] = eeg["T3"][mask] - eeg["O1"][mask]

        self.eeg[2][mask] = eeg["Fp1"][mask] - eeg["C3"][mask]
        self.eeg[3][mask] = eeg["C3"][mask] - eeg["O1"][mask]

        self.eeg[4][mask] = eeg["Fp2"][mask] - eeg["C4"][mask]
        self.eeg[5][mask] = eeg["C4"][mask] - eeg["O2"][mask]

        self.eeg[6][mask] = eeg["Fp2"][mask] - eeg["T4"][mask]
        self.eeg[7][mask] = eeg["T4"][mask] - eeg["O2"][mask]

        self.eeg[:, :] = np.clip(self.eeg, -1024, 1024) / 32.0
        self.eeg[:, :] = butter_lowpass_filter(self.eeg)
        eeg_t = torch.from_numpy(self.eeg.copy()).float()

        origin = row["origin"]
        spectrogram_id = int(row.spectrogram_id)
        spec = _read_spec(origin, spectrogram_id)
        LL, RL, LP, RP = _get_spec_cols_train(spec, origin, spectrogram_id)

        spec_offset = int(row.spectrogram_label_offset_seconds)
        if "time" in spec.columns:
            spec_ = spec[LL + RL + LP + RP].loc[
                (spec.time >= spec_offset) & (spec.time < spec_offset + 600)
            ]
        else:
            spec_ = spec[LL + RL + LP + RP]

        t = 22
        if not self.VALID:
            t += int(np.clip(np.random.normal(0, 11), -22, 22))
        spec_ = spec_[LL + RL + LP + RP][t : t + 256]

        self.spec[0, :, :64] = normalize(_resize2d_linear(spec_[LP].values, (256, 64)))
        self.spec[0, :, 64:128] = normalize(
            _resize2d_linear(spec_[LL].values, (256, 64))
        )
        self.spec[0, :, 128:-64] = normalize(
            _resize2d_linear(spec_[RP].values, (256, 64))
        )
        self.spec[0, :, -64:] = normalize(_resize2d_linear(spec_[RL].values, (256, 64)))

        mask = np.isnan(self.spec)
        if not self.VALID:
            self.spec *= np.random.normal(1, 0.0001)
            self.spec += np.random.normal(0, 0.0001)
        self.spec[mask] = 0
        if not self.VALID:
            if np.random.rand(1) > mask.sum() / (256 * 256):
                w = int(np.clip(np.random.normal(256 / 5, 256 / 20), 0, 512 / 5))
                t0 = (256 - w) / 2
                t0 = int(np.clip(np.random.normal(0, t0 / 2), -2 * t0, 2 * t0))
                if t0 < 0:
                    self.spec[:, t0 - w : t0] = 0
                else:
                    self.spec[:, t0 : t0 + w] = 0

        spec_t = torch.from_numpy(self.spec.copy()).float()

        custom_spec_path = _ensure_or_generate_custom_train(origin, int(row.eeg_id))
        custom_spec = _load_custom_npy_cached_train(custom_spec_path)

        spec_offset = int(2 * row.eeg_label_offset_seconds / 3)
        custom_spec = custom_spec[:, spec_offset : spec_offset + 32, :]
        custom_spec = np.concatenate(
            (
                custom_spec[:, :, 3],
                custom_spec[:, :, 2],
                custom_spec[:, :, 1],
                custom_spec[:, :, 0],
            )
        )
        custom_spec = _resize2d_linear(custom_spec, (256, 300))

        t = 22
        if not self.VALID:
            t += int(np.clip(np.random.normal(0, 11), -22, 22))

        self.spec[0] = custom_spec[:, t : t + 256]

        mask = np.isnan(self.spec)
        if not self.VALID:
            self.spec *= np.random.normal(1, 0.0001)
            self.spec += np.random.normal(0, 0.0001)
        self.spec[mask] = 0
        if not self.VALID:
            if np.random.rand(1) > mask.sum() / (256 * 256):
                w = int(np.clip(np.random.normal(256 / 5, 256 / 20), 0, 512 / 5))
                t0 = (256 - w) / 2
                t0 = int(np.clip(np.random.normal(0, t0 / 2), -2 * t0, 2 * t0))
                if t0 < 0:
                    self.spec[:, :, t0 - w : t0] = 0
                else:
                    self.spec[:, :, t0 : t0 + w] = 0

        custom_t = torch.from_numpy(self.spec.copy()).float()
        labels = row[votes].values.astype(np.float32)
        labels_t = torch.from_numpy(labels)
        return [eeg_t, spec_t, custom_t], labels_t




## === cell 25
class SemisupervisedKLDiv(nn.KLDivLoss):
    def __init__(self, min_v=1):
        super().__init__(reduce=False)
        self.min_v = torch.tensor(start_min_v, dtype=torch.float).to(device)
        self.p = torch.tensor(start_p, dtype=torch.float).to(device)
        self.step = 0

    def __steps__(self, steps):
        self.steps = steps

    def set_min_v(self):
        self.step += 1
        if self.step <= self.steps:
            self.min_v = nt(start_min_v, end_min_v, self.step, self.steps)

    def set_p(self):
        self.step += 1
        if self.step <= self.steps:
            self.p = nt(start_p, end_p, self.step, self.steps)

    def forward(self, y, t):
        v = t.sum(-1, keepdim=True)
        mask = v[:, 0] < self.min_v
        t = t.clone()
        t[mask] += (self.min_v - v[mask]) * (
            self.p * torch.softmax(y[mask].detach(), -1)
            + (1 - self.p) * torch.softmax(t[mask], -1)
        )
        v[mask] = self.min_v
        t /= v
        y = nn.functional.log_softmax(y, dim=1)
        loss = super().forward(y, t).sum(-1, keepdim=True)
        loss = loss * v
        loss = loss.sum() / v.sum()
        return loss




## === cell 26
def nt(nmin, nmax, tcur, tmax):
    return nmax - 0.5 * (nmax - nmin) * (1 + np.cos(tcur * np.pi / tmax))




## === cell 27
def cb(self):
    learn.loss_func.set_min_v()
    learn.loss_func.set_p()


min_v_cb = Callback(before_step=cb)



## === cell 28
seed_everything(2024)
splits = {}
for c in consensus:
    ID = df.loc[df.expert_consensus == c][CV].unique()
    splits[c] = []
    for t_idx, v_idx in KFold(N_FOLDS).split(ID):
        splits[c].append([ID[t_idx], ID[v_idx]])



## === cell 29
seed_everything(2024)
pseudo_labels_splits = {}
for c in consensus:
    if len(pseudo_labels) == 0:
        continue
    if CV not in pseudo_labels.columns:
        continue
    ID = pseudo_labels.loc[pseudo_labels["expert_consensus"] == c][CV].unique()
    if len(ID) > N_FOLDS:
        pseudo_labels_splits[c] = []
        for t_idx, v_idx in KFold(N_FOLDS).split(ID):
            pseudo_labels_splits[c].append([ID[t_idx], ID[v_idx]])



## === cell 30
base_hms_available = all(os.path.exists(p) for p in models["HMS"])
print("Base HMS fold checkpoints available:", base_hms_available)

NEED_TRAIN_HMS = not base_hms_available

if len(test) > 0 and NEED_TRAIN_HMS:
    TRAIN_WORKERS = _suggest_num_workers()
    if device == "cuda":
        TRAIN_WORKERS = 0

    for f in FOLDS:
        seed_everything(2024)
        print("FOLD: ", f)
        t = pd.DataFrame()
        v = pd.DataFrame()
        for c in splits:
            c_df = df.loc[df.expert_consensus == c]
            t = pd.concat([t, c_df[c_df[CV].isin(splits[c][f][0])]])
            v = pd.concat([v, c_df[c_df[CV].isin(splits[c][f][1])]])

        for c in pseudo_labels_splits:
            if f >= len(pseudo_labels_splits[c]):
                continue
            c_df = pseudo_labels.loc[pseudo_labels["expert_consensus"] == c]
            t = pd.concat([t, c_df[c_df[CV].isin(pseudo_labels_splits[c][f][0])]])
            v = pd.concat([v, c_df[c_df[CV].isin(pseudo_labels_splits[c][f][1])]])

        if len(t) == 0 or len(v) == 0:
            print("  Skipping fold due to empty train/valid after splits.")
            continue

        tds = HMS_TRAIN_DS(t)
        vds = HMS_TRAIN_DS(v, VALID=True)

        tdl = DataLoader(
            tds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=TRAIN_WORKERS,
            persistent_workers=False,
            pin_memory=(device == "cuda"),
            worker_init_fn=_seed_worker if TRAIN_WORKERS > 0 else None,
            generator=_G,
        )
        vdl = DataLoader(
            vds,
            batch_size=2 * batch_size,
            num_workers=TRAIN_WORKERS,
            persistent_workers=False,
            pin_memory=(device == "cuda"),
            worker_init_fn=_seed_worker if TRAIN_WORKERS > 0 else None,
            generator=_G,
        )
        dls = DataLoaders(tdl, vdl)

        base_path = models["HMS"][f]
        model = safe_torch_load(base_path, map_location=device)

        if model is None:

            class SimpleFallbackHMS(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.eeg_fc = nn.Sequential(
                        nn.Flatten(), nn.Linear(8 * 1024, 64), nn.ReLU()
                    )
                    self.spec_fc = nn.Sequential(
                        nn.Flatten(), nn.Linear(1 * 256 * 256, 64), nn.ReLU()
                    )
                    self.custom_fc = nn.Sequential(
                        nn.Flatten(), nn.Linear(1 * 256 * 256, 64), nn.ReLU()
                    )
                    self.head = nn.Linear(64 * 3, 6)

                def forward(self, X):
                    eeg, spec, custom = X
                    if eeg.shape[-1] == 2048:
                        eeg = eeg[..., 512:1536]
                    z = torch.cat(
                        [self.eeg_fc(eeg), self.spec_fc(spec), self.custom_fc(custom)],
                        dim=1,
                    )
                    return self.head(z)

            model = SimpleFallbackHMS().to(device)
            print(f"  Using fallback HMS model for fold {f} (no checkpoint found).")
        else:
            model = model.to(device)
            print(f"  Loaded base HMS checkpoint for fold {f}.")

        learn = Learner(
            dls,
            model,
            lr=LR,
            loss_func=SemisupervisedKLDiv(),
            cbs=[
                GradientClip(3.0),
                SaveModelCallback(every_epoch=True),
                ShowGraphCallback(),
                min_v_cb,
            ],
        )
        learn.loss_func.__steps__(
            (learn.dls.train.dataset.__len__() // batch_size) * EPOCHS
        )
        learn.fit_one_cycle(EPOCHS)

        torch.save(model, "HMSmodel_" + str(f))
        del tdl, vdl, tds, vds, dls, learn, model
        gc.collect()
else:
    print(
        "Skipping semisupervised training (base HMS checkpoints available or no test rows)."
    )



## === cell 31
if len(submission_pred) == 0:
    submission = pd.DataFrame({"eeg_id": test["eeg_id"].values})
    for c in votes:
        submission[c] = 1.0 / 6.0
else:
    submission = submission_pred[["eeg_id"] + votes].copy()

vals = submission[votes].to_numpy(dtype=np.float64)
vals = np.clip(vals, 1e-12, 1.0)
vals = vals / vals.sum(axis=1, keepdims=True)
submission[votes] = vals.astype(np.float32)

print(submission.head())
print(
    "Row-sum unique (rounded):",
    np.unique(np.round(submission[votes].sum(1).values, 6))[:10],
)

submission.to_csv("submission.csv", index=False)

if os.path.exists("/kaggle/working/custom"):
    shutil.rmtree("/kaggle/working/custom", ignore_errors=True)



## === cell 32
assert os.path.exists("submission.csv")
out = pd.read_csv("submission.csv")
assert list(out.columns) == ["eeg_id"] + votes
assert np.allclose(out[votes].sum(1).values, 1.0, atol=1e-5)
out.head()
