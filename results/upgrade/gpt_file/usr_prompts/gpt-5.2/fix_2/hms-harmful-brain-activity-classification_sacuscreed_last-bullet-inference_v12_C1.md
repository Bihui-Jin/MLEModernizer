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

# 5. Target score

0.4668129151277632

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1.40995) has done: 'I fix the two blockers preventing any valid submission: (1) CuPy failing due to an incompatible CUDA driver by replacing the CuPy spectrogram generation with a CPU SciPy/Numpy equivalent, and (2) missing external pretrained model files by falling back to a safe uniform-probability submission when those checkpoints aren’t available. I also correct the PATH bug (`test_`/`train_` suffix) so parquet reads resolve to the provided dataset folders. Finally, I ensure the written `submission.csv` has the exact required columns and each row sums to 1, so Kaggle accepts it end-to-end.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.utils.data import Dataset, DataLoader
from fastai.vision.all import *
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import gc
import shutil

device = "cuda" if torch.cuda.is_available() else "cpu"



## === cell 1
PATH = {
    "test": "/kaggle/input/hms-harmful-brain-activity-classification/test_",
    "train": "/kaggle/input/hms-harmful-brain-activity-classification/train_",
}
if not os.path.exists(PATH["test"] + "eegs"):
    PATH = {
        "test": "/kaggle/input/hms-harmful-brain-activity-classification/test_",
        "train": "/kaggle/input/hms-harmful-brain-activity-classification/train_",
    }

if not os.path.exists(PATH["test"] + "eegs"):
    PATH = {
        "test": "/kaggle/input/hms-harmful-brain-activity-classification/test_",
        "train": "/kaggle/input/hms-harmful-brain-activity-classification/train_",
    }
if not os.path.exists(PATH["test"] + "eegs"):
    PATH = {
        "test": "/kaggle/input/hms-harmful-brain-activity-classification/test_".replace(
            "test_", ""
        ),
        "train": "/kaggle/input/hms-harmful-brain-activity-classification/train_".replace(
            "train_", ""
        ),
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
TH = 5
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




## === cell 4
from scipy.signal import butter, lfilter


def butter_lowpass_filter(data, cutoff_freq=20, sampling_rate=200, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data




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
spec = pd.read_parquet(
    os.path.join(
        PATH["test"], "spectrograms", f"{row.spectrogram_id.values[0]}.parquet"
    )
)
LL = [c for c in spec.columns if "LL" in c]
RL = [c for c in spec.columns if "RL" in c]
LP = [c for c in spec.columns if "LP" in c]
RP = [c for c in spec.columns if "RP" in c]



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1521297849.py in <cell line: 0>()
----> 1 spec = pd.read_parquet(
      2     os.path.join(
      3         PATH["test"], "spectrograms", f"{row.spectrogram_id.values[0]}.parquet"
      4     )
      5 )

/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py in read_parquet(path, engine, columns, storage_options, use_nullable_dtypes, dtype_backend, filesystem, filters, **kwargs)
    665     check_dtype_backend(dtype_backend)
    666 
--> 667     return impl.read(
    668         path,
    669         columns=columns,

/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py in read(self, path, columns, filters, use_nullable_dtypes, dtype_backend, storage_options, filesystem, **kwargs)
    265             to_pandas_kwargs["split_blocks"] = True  # type: ignore[assignment]
    266 
--> 267         path_or_handle, handles, filesystem = _get_path_or_handle(
    268             path,
    269             filesystem,

/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py in _get_path_or_handle(path, fs, storage_options, mode, is_dir)
    138         # fsspec resources can also point to directories
    139         # this branch is used for example when reading from non-fsspec URLs
--> 140         handles = get_handle(
    141             path_or_handle, mode, is_text=False, storage_options=storage_options
    142         )

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    880         else:
    881             # Binary mode
--> 882             handle = open(handle, ioargs.mode)
    883         handles.append(handle)
    884 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/hms-harmful-brain-activity-classification/test_/spectrograms/577118473.parquet'

## === cell 12
from scipy.ndimage import gaussian_filter
from scipy.signal import (
    iirnotch,
    filtfilt,
    butter as scipy_butter,
    spectrogram as scipy_spectrogram,
)


def create_spectrogram_cpu(
    eeg_data,
    eeg_id,
    low_cut_freq=0.7,
    high_cut_freq=20,
    order_band=5,
    nperseg=500,
    noverlap=200,
    nfft=1024,
    sigma_gaussian=0.7,
    mean_montage_names=4,
):
    electrode_pair_name_locations = {
        "LL": ["Fp1", "F7", "T3", "T5", "O1"],
        "RL": ["Fp2", "F8", "T4", "T6", "O2"],
        "LP": ["Fp1", "F3", "C3", "P3", "O1"],
        "RP": ["Fp2", "F4", "C4", "P4", "O2"],
    }
    fs = 200
    nyquist_freq = 0.5 * fs
    low = low_cut_freq / nyquist_freq
    high = high_cut_freq / nyquist_freq

    b_notch, a_notch = iirnotch(w0=60, Q=30, fs=fs)
    b_bp, a_bp = scipy_butter(order_band, [low, high], btype="band")

    spectrogram_out = None
    processed_eeg = {}

    for i, (montage, locs) in enumerate(electrode_pair_name_locations.items()):
        for j in range(4):
            s = eeg_data[locs[j]].values - eeg_data[locs[j + 1]].values
            if np.isnan(s).mean() < 1:
                m = np.nanmean(s)
                s = np.nan_to_num(s, nan=m)
            else:
                s = np.zeros_like(s)

            try:
                s = filtfilt(b_notch, a_notch, s)
            except Exception:
                pass
            try:
                s = filtfilt(b_bp, a_bp, s)
            except Exception:
                pass

            f, t, Sxx = scipy_spectrogram(
                s,
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
            Sxx = (Sxx - mean) / (std + 1e-6)

            if spectrogram_out is None:
                spectrogram_out = np.zeros(
                    (Sxx.shape[0], Sxx.shape[1], 4), dtype=np.float32
                )
            spectrogram_out[:, :, i] += Sxx

            processed_eeg[f"{locs[j]}_{locs[j + 1]}"] = s
            processed_eeg[montage] = processed_eeg.get(montage, 0) + s

        spectrogram_out[:, :, i] /= float(mean_montage_names)

    if sigma_gaussian and sigma_gaussian > 0:
        spectrogram_out = gaussian_filter(
            spectrogram_out, sigma=(sigma_gaussian, sigma_gaussian, 0)
        )

    ekg = eeg_data["EKG"].values
    if np.isnan(ekg).mean() < 1:
        ekg = np.nan_to_num(ekg, nan=np.nanmean(ekg))
    else:
        ekg = np.zeros_like(ekg)
    try:
        ekg = filtfilt(b_notch, a_notch, ekg)
        ekg = filtfilt(b_bp, a_bp, ekg)
    except Exception:
        pass
    processed_eeg["EKG"] = ekg

    return spectrogram_out, processed_eeg




## === cell 13
custom_dir = "/kaggle/working/custom"
os.makedirs(custom_dir, exist_ok=True)

for eeg_id in tqdm.tqdm(test.eeg_id.unique(), desc="Building custom specs (CPU)"):
    out_path = os.path.join(custom_dir, f"{eeg_id}.npy")
    if os.path.exists(out_path):
        continue
    eeg = pd.read_parquet(os.path.join(PATH["test"], "eegs", f"{eeg_id}.parquet"))
    mask = np.isnan(eeg.values).max(1)
    spec_cpu, _ = create_spectrogram_cpu(
        eeg,
        eeg_id,
        low_cut_freq=0.7,
        high_cut_freq=20,
        order_band=5,
        nperseg=500,
        noverlap=200,
        nfft=1024,
        sigma_gaussian=0.7,
        mean_montage_names=4,
    )
    _, w, _ = spec_cpu.shape
    mask_r = resize(
        mask.astype(np.float32), (w, 1), preserve_range=True, anti_aliasing=False
    )
    spec_cpu[:, mask_r[:, 0] > 0.5, :] = 0
    np.save(out_path, spec_cpu.astype(np.float32))
    del eeg, mask, mask_r, spec_cpu

gc.collect()




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1894721378.py in <cell line: 0>()
      8     if os.path.exists(out_path):
      9         continue
---> 10     eeg = pd.read_parquet(os.path.join(PATH["test"], "eegs", f"{eeg_id}.parquet"))
     11     mask = np.isnan(eeg.values).max(1)
     12     spec_cpu, _ = create_spectrogram_cpu(

/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py in read_parquet(path, engine, columns, storage_options, use_nullable_dtypes, dtype_backend, filesystem, filters, **kwargs)
    665     check_dtype_backend(dtype_backend)
    666 
--> 667     return impl.read(
    668         path,
    669         columns=columns,

/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py in read(self, path, columns, filters, use_nullable_dtypes, dtype_backend, storage_options, filesystem, **kwargs)
    265             to_pandas_kwargs["split_blocks"] = True  # type: ignore[assignment]
    266 
--> 267         path_or_handle, handles, filesystem = _get_path_or_handle(
    268             path,
    269             filesystem,

/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py in _get_path_or_handle(path, fs, storage_options, mode, is_dir)
    138         # fsspec resources can also point to directories
    139         # this branch is used for example when reading from non-fsspec URLs
--> 140         handles = get_handle(
    141             path_or_handle, mode, is_text=False, storage_options=storage_options
    142         )

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    880         else:
    881             # Binary mode
--> 882             handle = open(handle, ioargs.mode)
    883         handles.append(handle)
    884 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/hms-harmful-brain-activity-classification/test_/eegs/2578018731.parquet'

## === cell 14
class HMS_DS(torch.utils.data.Dataset):
    def __init__(self, df):
        self.START = (10000 - 2048) // 2
        self.data = df.reset_index(drop=True)
        self.eeg = np.zeros((8, 2048), dtype=np.float32)
        self.spec = np.zeros((1, 256, 256), dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        eeg = pd.read_parquet(
            os.path.join(PATH["test"], "eegs", f"{row.eeg_id}.parquet")
        )
        eeg = eeg.iloc[self.START : self.START + 2048]

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

        eeg_np = np.clip(self.eeg, -1024, 1024) / 32.0
        eeg_np = butter_lowpass_filter(eeg_np)
        eeg_t = torch.from_numpy(eeg_np).float()

        spec_df = pd.read_parquet(
            os.path.join(PATH["test"], "spectrograms", f"{row.spectrogram_id}.parquet")
        )
        spec_df = spec_df[LL + RL + LP + RP]
        t = 22
        spec_df = spec_df[t : t + 256]

        self.spec[0, :, :64] = normalize(resize(spec_df[LP].values, (256, 64)))
        self.spec[0, :, 64:128] = normalize(resize(spec_df[LL].values, (256, 64)))
        self.spec[0, :, 128:-64] = normalize(resize(spec_df[RP].values, (256, 64)))
        self.spec[0, :, -64:] = normalize(resize(spec_df[RL].values, (256, 64)))

        self.spec[np.isnan(self.spec)] = 0
        spec_t = torch.from_numpy(self.spec.copy()).float()

        custom_path = os.path.join("/kaggle/working/custom", f"{row.eeg_id}.npy")
        custom_spec = np.load(custom_path)
        custom_spec = np.concatenate(
            (
                custom_spec[:, :, 3],
                custom_spec[:, :, 2],
                custom_spec[:, :, 1],
                custom_spec[:, :, 0],
            ),
            axis=0,
        )
        custom_spec = resize(custom_spec, (256, 300))
        t = 22
        self.spec[0] = custom_spec[:, t : t + 256]
        self.spec[np.isnan(self.spec)] = 0
        custom_t = torch.from_numpy(self.spec.copy()).float()

        return (
            int(row.eeg_id),
            int(row.spectrogram_id),
            eeg_t.to(device),
            spec_t.to(device),
            custom_t.to(device),
        )




## === cell 15
def _all_paths_exist(paths):
    return all(os.path.exists(p) for p in paths)


have_any_models = (
    _all_paths_exist(models["eeg"])
    and _all_paths_exist(models["spec"])
    and _all_paths_exist(models["custom"])
    and _all_paths_exist(models["HMS"])
)

if not have_any_models:
    sub = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
    )
    for c in votes:
        sub[c] = 1.0 / len(votes)
    sub[votes] = sub[votes].div(sub[votes].sum(axis=1), axis=0)
    sub.to_csv("submission.csv", index=False)
    print("External checkpoints not found. Wrote uniform-probability submission.csv")
else:
    print(
        "All expected checkpoints found; proceeding with model inference/training pipeline."
    )



## === cell 16
pass



## === cell 17
pass
