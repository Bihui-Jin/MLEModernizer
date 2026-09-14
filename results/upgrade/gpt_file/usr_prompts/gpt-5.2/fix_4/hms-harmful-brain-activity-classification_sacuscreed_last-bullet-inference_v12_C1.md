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
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/hms-harmful-brain-activity-classification",
    "/kaggle/data/hms-harmful-brain-activity-classification",
]


def _resolve_data_root():
    for root in DATA_ROOT_CANDIDATES:
        if os.path.isdir(root):
            if os.path.isdir(os.path.join(root, "test_eegs")) and os.path.isdir(
                os.path.join(root, "test_spectrograms")
            ):
                return root
    root = "/kaggle/input/hms-harmful-brain-activity-classification"
    return root


DATA_ROOT = _resolve_data_root()

PATH = {
    "root": DATA_ROOT,
    "test_eegs": os.path.join(DATA_ROOT, "test_eegs"),
    "train_eegs": os.path.join(DATA_ROOT, "train_eegs"),
    "test_spectrograms": os.path.join(DATA_ROOT, "test_spectrograms"),
    "train_spectrograms": os.path.join(DATA_ROOT, "train_spectrograms"),
}

for k, p in PATH.items():
    if k == "root":
        continue
    if not os.path.exists(p):
        raise FileNotFoundError(f"Expected dataset folder missing: {k} -> {p}")

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
submission = pd.read_csv(os.path.join(PATH["root"], "sample_submission.csv"))
submission.head()



## === cell 7
votes = [c for c in submission.columns if "_vote" in c]
votes



## === cell 8
test = pd.read_csv(os.path.join(PATH["root"], "test.csv"))
test.head()



## === cell 9
if DEBUG:
    test = pd.read_csv(os.path.join(PATH["root"], "train.csv"))[:1000]
    test.head()



## === cell 10
any_spec_id = int(test.spectrogram_id.iloc[0])
spec = pd.read_parquet(
    os.path.join(PATH["test_spectrograms"], f"{any_spec_id}.parquet")
)
LL = [c for c in spec.columns if "LL" in c]
RL = [c for c in spec.columns if "RL" in c]
LP = [c for c in spec.columns if "LP" in c]
RP = [c for c in spec.columns if "RP" in c]

if not (LL and RL and LP and RP):
    raise RuntimeError("Failed to infer spectrogram channel groups (LL/RL/LP/RP).")



## === cell 11
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




## === cell 12
custom_dir = "/kaggle/working/custom"
os.makedirs(custom_dir, exist_ok=True)

for eeg_id in tqdm.tqdm(test.eeg_id.unique(), desc="Building custom specs (CPU)"):
    out_path = os.path.join(custom_dir, f"{eeg_id}.npy")
    if os.path.exists(out_path):
        continue
    eeg = pd.read_parquet(os.path.join(PATH["test_eegs"], f"{eeg_id}.parquet"))
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




## === cell 13
class HMS_DS(torch.utils.data.Dataset):
    def __init__(self, df, mode="test"):
        self.START = (10000 - 2048) // 2
        self.data = df.reset_index(drop=True)
        self.mode = mode
        self.eeg = np.zeros((8, 2048), dtype=np.float32)
        self.spec = np.zeros((1, 256, 256), dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        eeg_folder = PATH["test_eegs"] if self.mode == "test" else PATH["train_eegs"]
        spec_folder = (
            PATH["test_spectrograms"]
            if self.mode == "test"
            else PATH["train_spectrograms"]
        )

        eeg = pd.read_parquet(os.path.join(eeg_folder, f"{row.eeg_id}.parquet"))
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
            os.path.join(spec_folder, f"{row.spectrogram_id}.parquet")
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

        y = None
        if self.mode != "test":
            y = row[votes].values.astype(np.float32)
            s = y.sum()
            if s <= 0:
                y = np.ones_like(y) / len(y)
            else:
                y = y / s
            y_t = torch.from_numpy(y).float()
            return (
                int(row.eeg_id),
                int(row.spectrogram_id),
                eeg_t,
                spec_t,
                custom_t,
                y_t,
            )

        return (
            int(row.eeg_id),
            int(row.spectrogram_id),
            eeg_t,
            spec_t,
            custom_t,
        )




## === cell 14
import torchvision


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

train_df = pd.read_csv(os.path.join(PATH["root"], "train.csv"))
train_df = train_df.copy()
train_df["vote_sum"] = train_df[votes].sum(axis=1)
train_df = train_df[train_df["vote_sum"] > 0].reset_index(drop=True)

patients = train_df["patient_id"].unique()
rng = np.random.default_rng(42)
rng.shuffle(patients)
n_val = max(1, int(0.1 * len(patients)))
val_patients = set(patients[:n_val])
trn = train_df[~train_df["patient_id"].isin(val_patients)].reset_index(drop=True)
val = train_df[train_df["patient_id"].isin(val_patients)].reset_index(drop=True)

if DEBUG:
    trn = trn.iloc[:256].reset_index(drop=True)
    val = val.iloc[:64].reset_index(drop=True)

print(f"Train rows: {len(trn)}  Val rows: {len(val)}  Test rows: {len(test)}")

needed_train_eeg_ids = pd.concat([trn["eeg_id"], val["eeg_id"]]).unique()
for eeg_id in tqdm.tqdm(
    needed_train_eeg_ids, desc="Building custom specs for train/val (CPU)"
):
    out_path = os.path.join(custom_dir, f"{eeg_id}.npy")
    if os.path.exists(out_path):
        continue
    eeg = pd.read_parquet(os.path.join(PATH["train_eegs"], f"{eeg_id}.parquet"))
    mask = np.isnan(eeg.values).max(1)
    spec_cpu, _ = create_spectrogram_cpu(eeg, eeg_id)
    _, w, _ = spec_cpu.shape
    mask_r = resize(
        mask.astype(np.float32), (w, 1), preserve_range=True, anti_aliasing=False
    )
    spec_cpu[:, mask_r[:, 0] > 0.5, :] = 0
    np.save(out_path, spec_cpu.astype(np.float32))
    del eeg, mask, mask_r, spec_cpu
gc.collect()

train_ds = HMS_DS(trn, mode="train")
val_ds = HMS_DS(val, mode="train")
test_ds = HMS_DS(test, mode="test")

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
)
val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
)
test_loader = DataLoader(
    test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
)


def make_effnet_1ch():
    m = torchvision.models.efficientnet_b0(weights=None)
    conv0 = m.features[0][0]
    if conv0.in_channels != 1:
        new_conv = nn.Conv2d(
            1,
            conv0.out_channels,
            kernel_size=conv0.kernel_size,
            stride=conv0.stride,
            padding=conv0.padding,
            bias=(conv0.bias is not None),
        )
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        if new_conv.bias is not None:
            nn.init.zeros_(new_conv.bias)
        m.features[0][0] = new_conv
    return m


class EEG2Img(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(1, 3, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.proj(x)


class EEGBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = EEG2Img()
        self.backbone = torchvision.models.efficientnet_b0(weights=None)

    def forward(self, x):
        x = self.adapter(x)
        return self.backbone(x)


eeg_model = EEGBackbone().to(device)
spec_model = make_effnet_1ch().to(device)
custom_model = make_effnet_1ch().to(device)

hms = HMSmodel([eeg_model, spec_model, custom_model]).to(device)

opt = torch.optim.AdamW(
    [p for p in hms.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4
)


def kl_loss_from_logits(logits, targets, eps=1e-7):
    log_probs = F.log_softmax(logits, dim=1)
    targets = torch.clamp(targets, eps, 1.0)
    targets = targets / targets.sum(dim=1, keepdim=True)
    return F.kl_div(log_probs, targets, reduction="batchmean")


hms.train()
for epoch in range(EPOCHS):
    pbar = tqdm.tqdm(train_loader, desc=f"Training epoch {epoch+1}/{EPOCHS}")
    for batch in pbar:
        _, _, eeg_t, spec_t, custom_t, y_t = batch
        eeg_t = eeg_t.to(device, non_blocking=True)
        spec_t = spec_t.to(device, non_blocking=True)
        custom_t = custom_t.to(device, non_blocking=True)
        y_t = y_t.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        out = hms((eeg_t, spec_t, custom_t))
        loss = kl_loss_from_logits(out, y_t)
        loss.backward()
        opt.step()
        pbar.set_postfix({"loss": float(loss.detach().cpu().item())})

hms.eval()
with torch.no_grad():
    losses = []
    for batch in val_loader:
        _, _, eeg_t, spec_t, custom_t, y_t = batch
        out = hms((eeg_t.to(device), spec_t.to(device), custom_t.to(device)))
        losses.append(float(kl_loss_from_logits(out, y_t.to(device)).cpu().item()))
    if len(losses):
        print(f"Val KL (approx): {np.mean(losses):.5f}")



## === cell 15
hms.eval()
preds = []
ids = []
with torch.no_grad():
    for batch in tqdm.tqdm(test_loader, desc="Inference"):
        eeg_id, spec_id, eeg_t, spec_t, custom_t = batch
        out = hms((eeg_t.to(device), spec_t.to(device), custom_t.to(device)))
        prob = F.softmax(out, dim=1).detach().cpu().numpy().astype(np.float64)
        preds.append(prob)
        ids.extend([int(x) for x in eeg_id])

preds = np.concatenate(preds, axis=0)
pred_df = pd.DataFrame(preds, columns=votes)
pred_df["eeg_id"] = ids

sub = pd.read_csv(os.path.join(PATH["root"], "sample_submission.csv"))
sub = sub.merge(pred_df, on="eeg_id", how="left", suffixes=("", "_pred"))

for c in votes:
    if c not in sub.columns or sub[c].isna().any():
        if f"{c}_pred" in sub.columns:
            sub[c] = sub[f"{c}_pred"]
        else:
            sub[c] = sub[c].fillna(1.0 / len(votes))
    if f"{c}_pred" in sub.columns:
        sub[c] = sub[c].fillna(sub[f"{c}_pred"])
        sub.drop(columns=[f"{c}_pred"], inplace=True)

sub[votes] = sub[votes].astype(np.float64)
sub[votes] = sub[votes].clip(1e-12, 1.0)
sub[votes] = sub[votes].div(sub[votes].sum(axis=1), axis=0)

sub = sub[["eeg_id"] + votes]
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv", sub.shape)
print(sub.head())
