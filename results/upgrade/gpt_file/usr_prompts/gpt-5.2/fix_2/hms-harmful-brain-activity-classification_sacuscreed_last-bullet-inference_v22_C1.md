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

0.4280755309867892

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.41937) has done: 'I remove the CuPy/GPU-dependent custom-spectrogram generation (it fails due to insufficient CUDA driver) and instead load precomputed custom spectrogram `.npy` files from a Kaggle input path when available, with a safe fallback to zeros if not. I also make the inference path robust to missing external model checkpoints by falling back to a simple, valid probability baseline derived from the training label distribution, so a submission is always produced. Additionally, I fix undefined variables (LL/RL/LP/RP), incorrect `DataFrame` indexing (`test[i:i+1]`), missing `PREDS` deletes, and ensure probabilities sum to 1 and the saved file is `submission.csv`. These changes are execution-unblocking and score-neutral-to-reasonable given the circumstances (no provided checkpoints), while preserving the existing model and dataset logic where it can run.'

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
TH = 1  # Percentile of pseudolabels to take
pseudo_votes = 10  # The votes assigned to pseudo_labels
batch_size = 128
min_votes = 0
LR = 1e-3
EPOCHS = 2
start_p = 0.5
end_p = 1
start_min_v = 1
end_min_v = 1
label_aug = False
N_FOLDS = 5
FOLDS = [0, 1, 2, 3, 4]
HMS_DROP = 0.9
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
submission_tmpl = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
votes = [c for c in submission_tmpl.columns if "_vote" in c]

test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
if DEBUG:
    test = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )[:1000]
    PATH["test"] = PATH["train"]



## === cell 7
row = test.iloc[[0]]
spec0 = pd.read_parquet(
    PATH["test"] + "spectrograms/" + str(row.spectrogram_id.values[0]) + ".parquet"
)
LL = [c for c in spec0.columns if "LL" in c]
RL = [c for c in spec0.columns if "RL" in c]
LP = [c for c in spec0.columns if "LP" in c]
RP = [c for c in spec0.columns if "RP" in c]
del spec0



## === cell 8
CUSTOM_TEST_DIR_CANDIDATES = [
    "/kaggle/input/custom-specs/custom",  # user code already expects train custom specs here
    "/kaggle/input/custom-specs-test/custom",
    "/kaggle/input/custom-spectrograms/custom",
]
CUSTOM_TEST_DIR = None
for d in CUSTOM_TEST_DIR_CANDIDATES:
    if os.path.isdir(d):
        CUSTOM_TEST_DIR = d
        break


def load_custom_spec_npy(eeg_id: int):
    """
    Returns a numpy array suitable for downstream processing.
    Expected original shape from prior code: (freq?, time?, 4) or (h,w,4) depending on producer.
    We'll handle (H,W,4) and (A,B,C,4) by slicing/reshaping similarly to original logic.
    If missing/unreadable, return None.
    """
    if CUSTOM_TEST_DIR is None:
        return None
    fp = os.path.join(CUSTOM_TEST_DIR, f"{int(eeg_id)}.npy")
    if not os.path.isfile(fp):
        return None
    try:
        arr = np.load(fp)
        return arr
    except Exception:
        return None




## === cell 9
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
            PATH["test"] + "eegs/" + str(int(row.eeg_id)) + ".parquet"
        )
        eeg = eeg.iloc[self.START : self.START + 2048]

        mask = ~np.isnan(eeg["Fp1"].values)
        self.eeg[:, :] = 0

        self.eeg[0][mask] = eeg["Fp1"].values[mask] - eeg["T3"].values[mask]
        self.eeg[1][mask] = eeg["T3"].values[mask] - eeg["O1"].values[mask]

        self.eeg[2][mask] = eeg["Fp1"].values[mask] - eeg["C3"].values[mask]
        self.eeg[3][mask] = eeg["C3"].values[mask] - eeg["O1"].values[mask]

        self.eeg[4][mask] = eeg["Fp2"].values[mask] - eeg["C4"].values[mask]
        self.eeg[5][mask] = eeg["C4"].values[mask] - eeg["O2"].values[mask]

        self.eeg[6][mask] = eeg["Fp2"].values[mask] - eeg["T4"].values[mask]
        self.eeg[7][mask] = eeg["T4"].values[mask] - eeg["O2"].values[mask]

        self.eeg = np.clip(self.eeg, -1024, 1024) / 32.0
        self.eeg = butter_lowpass_filter(self.eeg)
        eeg_t = torch.from_numpy(self.eeg).float()

        spec = pd.read_parquet(
            PATH["test"] + "spectrograms/" + str(int(row.spectrogram_id)) + ".parquet"
        )
        t0 = 22
        spec = spec[LL + RL + LP + RP].iloc[t0 : t0 + 256]

        self.spec[0, :, :64] = normalize(resize(spec[LP].values, (256, 64)))
        self.spec[0, :, 64:128] = normalize(resize(spec[LL].values, (256, 64)))
        self.spec[0, :, 128:-64] = normalize(resize(spec[RP].values, (256, 64)))
        self.spec[0, :, -64:] = normalize(resize(spec[RL].values, (256, 64)))

        m = np.isnan(self.spec)
        self.spec[m] = 0
        spec_t = torch.from_numpy(self.spec).float()

        custom_arr = load_custom_spec_npy(int(row.eeg_id))
        if custom_arr is None:
            custom_img = np.zeros((256, 256), dtype=np.float32)
        else:
            arr = custom_arr
            if arr.ndim == 4 and arr.shape[-1] == 4:
                time_len = arr.shape[1]
                start = max(0, min((time_len - 32) // 2, time_len - 32))
                arr = arr[:, start : start + 32, :, :]
                arr = np.nanmean(arr, axis=1)
            if arr.ndim == 3 and arr.shape[-1] == 4:
                custom_spec = np.concatenate(
                    (arr[:, :, 3], arr[:, :, 2], arr[:, :, 1], arr[:, :, 0]), axis=0
                )
                custom_spec = resize(
                    custom_spec, (256, 300), preserve_range=True, anti_aliasing=True
                ).astype(np.float32)
                t0 = 22
                custom_img = custom_spec[:, t0 : t0 + 256]
            else:
                custom_img = np.zeros((256, 256), dtype=np.float32)

        self.spec[0] = custom_img
        m = np.isnan(self.spec)
        self.spec[m] = 0
        custom_t = torch.from_numpy(self.spec).float()

        return (
            int(row.eeg_id),
            int(row.spectrogram_id),
            eeg_t.to(device),
            spec_t.to(device),
            custom_t.to(device),
        )




## === cell 10
def try_load_torch_model(path: str):
    if not os.path.exists(path):
        return None
    try:
        m = torch.load(path, map_location=device)
        m.eval()
        return m
    except Exception:
        return None


def available_model_list(paths):
    out = []
    for p in paths:
        m = try_load_torch_model(p)
        if m is not None:
            out.append(m)
    return out




## === cell 11
train_df = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
)
prior = train_df[votes].sum(0).values.astype(np.float64)
prior = prior / prior.sum()
prior = np.clip(prior, 1e-8, 1.0)
prior = prior / prior.sum()

ds = HMS_DS(test)
dl = DataLoader(ds, batch_size=BS, shuffle=False, drop_last=False)

eeg_ensemble = available_model_list(models["eeg"])
spec_ensemble = available_model_list(models["spec"])
custom_ensemble = available_model_list(models["custom"])
HMS_ensemble = available_model_list(models["HMS"])

use_models = (
    len(eeg_ensemble) + len(spec_ensemble) + len(custom_ensemble) + len(HMS_ensemble)
) > 0

submission = []
with torch.no_grad():
    for eeg_id, spectrogram_id, eegs, specs, custom_specs in dl:
        N = eegs.shape[0]
        if use_models:
            preds_list = []
            for m in eeg_ensemble:
                preds_list.append(torch.softmax(m(eegs), -1))
            for m in spec_ensemble:
                preds_list.append(torch.softmax(m(specs), -1))
            for m in custom_ensemble:
                preds_list.append(torch.softmax(m(custom_specs), -1))
            for m in HMS_ensemble:
                preds_list.append(torch.softmax(m([eegs, specs, custom_specs]), -1))

            P = torch.stack(preds_list, dim=1).mean(dim=1)  # (N,6)
            P = P / P.sum(dim=1, keepdim=True)
            P = P.clamp_min(1e-8)
            P = P / P.sum(dim=1, keepdim=True)
            P_np = P.detach().cpu().numpy()
        else:
            P_np = np.tile(prior[None, :], (N, 1))

        for i in range(N):
            row = {"eeg_id": int(eeg_id[i].cpu().item())}
            for j, col in enumerate(votes):
                row[col] = float(P_np[i, j])
            submission.append(row)

submission = pd.DataFrame(submission)
submission = submission[["eeg_id"] + votes].sort_values("eeg_id").reset_index(drop=True)

p = submission[votes].values.astype(np.float64)
p = np.clip(p, 1e-12, 1.0)
p = p / p.sum(axis=1, keepdims=True)
submission[votes] = p

submission.to_csv("submission.csv", index=False)
print(submission.head())
print(
    "Row sums (min/max):",
    submission[votes].sum(1).min(),
    submission[votes].sum(1).max(),
)
print("Wrote submission.csv with shape:", submission.shape)
