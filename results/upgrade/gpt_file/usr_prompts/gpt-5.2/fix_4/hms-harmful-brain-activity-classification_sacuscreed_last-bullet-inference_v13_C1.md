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

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I remove the CuPy/GPU spectrogram generation path that currently fails due to an insufficient CUDA driver, and instead generate the required `custom` spectrogram `.npy` files on CPU using SciPy/skimage with the same overall preprocessing intent. I also make model loading robust: if the external `/kaggle/input/*-cv-kaggle/*` checkpoints are missing, the code fall back to a safe, valid probability baseline so a submission CSV is always produced end-to-end. Finally, I ensure all tensors are moved to the correct device inside the inference loop (not inside the Dataset) to avoid DataLoader CUDA issues, and I always write `submission.csv` with the required columns and row-wise probability sums of 1. These changes are primarily bug fixes to unblock execution and produce a valid submission; any score improvement is only possible if the specified checkpoints exist in the environment.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.4668), so we should improve performance while keeping the same overall pipeline. The biggest likely issue is a mismatch between the pseudo-label uncertainty heuristic and the ensemble averaging: you currently take `softmax(mean(probabilities))`, which over-smooths and can hurt KL, and you compute sigma on probabilities but then transform them again; we change this to the standard “average probabilities” (mean of softmax outputs) consistently in both pseudo-label generation and final inference. We also make the percentile selection robust when `sigma` is all-zeros (common when no/one model is loaded), preventing degenerate pseudo-label behavior that can harm training. These changes preserve model architectures/training loops/feature extraction and only adjust aggregation + a safety guard, which should move the score toward the target.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is much worse than the target (0.4668), so we should improve score with minimal, low-risk changes that keep the same models/training. The biggest likely performance drag is that your semisupervised stage is being skipped when base HMS checkpoints are missing, and the final inference only uses HMS models, ignoring the stronger multi-branch ensemble you already computed earlier. I (1) always run semisupervised training by initializing from the available HMS models if present, otherwise falling back to a frozen, randomly-initialized HMS wrapper trained for the same EPOCHS (so the pipeline still works), and (2) in final inference, ensemble across all available models (eeg/spec/custom/HMS) by averaging probabilities (not softmax of averaged logits) and then renormalizing—this preserves evaluation semantics and should reduce KL toward the target. I also keep your pseudo-label selection guard but make the sigma computation numerically safer.'

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
print("device:", device)



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
    PATH["test"] + "spectrograms/" + str(row.spectrogram_id.values[0]) + ".parquet"
)
LL = [c for c in spec.columns if "LL" in c]
RL = [c for c in spec.columns if "RL" in c]
LP = [c for c in spec.columns if "LP" in c]
RP = [c for c in spec.columns if "RP" in c]
print(len(LL), len(RL), len(LP), len(RP))



## === cell 12
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




## === cell 13
custom_dir = "/kaggle/working/custom"
os.makedirs(custom_dir, exist_ok=True)

for eeg_id in tqdm.tqdm(test.eeg_id.unique(), desc="Creating custom specs (CPU)"):
    out_path = os.path.join(custom_dir, f"{eeg_id}.npy")
    if os.path.exists(out_path):
        continue
    eeg = pd.read_parquet(PATH["test"] + "eegs/" + str(eeg_id) + ".parquet")
    mask = np.isnan(eeg.values).max(1).astype(np.float32)

    spec_cpu, _ = create_spectrogram_cpu(
        eeg,
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
    mask_w = resize(mask, (w, 1), preserve_range=True, anti_aliasing=False).astype(bool)
    spec_cpu[:, mask_w[:, 0], :] = 0.0

    np.save(out_path, spec_cpu.astype(np.float32))
    del eeg, mask, spec_cpu, mask_w
gc.collect()




## === cell 14
class HMS_DS(torch.utils.data.Dataset):
    """ """

    def __init__(self, df):
        self.START = (10000 - 2048) // 2
        self.data = df.reset_index(drop=True)
        self.eeg = np.zeros((8, 2048), dtype=np.float32)
        self.spec = np.zeros((1, 256, 256), dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        eeg = pd.read_parquet(PATH["test"] + "eegs/" + str(row.eeg_id) + ".parquet")
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

        self.eeg = np.clip(self.eeg, -1024, 1024) / 32.0
        self.eeg = butter_lowpass_filter(self.eeg)
        eeg_t = torch.from_numpy(self.eeg).float()  # keep on CPU; move in loop

        spec = pd.read_parquet(
            PATH["test"] + "spectrograms/" + str(row.spectrogram_id) + ".parquet"
        )
        t = 22
        spec = spec[LL + RL + LP + RP][t : t + 256]

        self.spec[0, :, :64] = normalize(resize(spec[LP].values, (256, 64)))
        self.spec[0, :, 64:128] = normalize(resize(spec[LL].values, (256, 64)))
        self.spec[0, :, 128:-64] = normalize(resize(spec[RP].values, (256, 64)))
        self.spec[0, :, -64:] = normalize(resize(spec[RL].values, (256, 64)))

        mask2 = np.isnan(self.spec)
        self.spec[mask2] = 0.0
        spec_t = torch.from_numpy(self.spec.copy()).float()  # copy because reused

        custom_spec = np.load("/kaggle/working/custom/" + str(row.eeg_id) + ".npy")
        custom_spec = np.concatenate(
            (
                custom_spec[:, :, 3],
                custom_spec[:, :, 2],
                custom_spec[:, :, 1],
                custom_spec[:, :, 0],
            )
        )
        custom_spec = resize(custom_spec, (256, 300))

        t = 22
        self.spec[0] = custom_spec[:, t : t + 256]
        mask3 = np.isnan(self.spec)
        self.spec[mask3] = 0.0
        custom_t = torch.from_numpy(self.spec.copy()).float()

        return int(row.eeg_id), int(row.spectrogram_id), eeg_t, spec_t, custom_t




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
        return None  # handled upstream

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
if len(test) > 1:
    submission_pred = pd.DataFrame()

    ds = HMS_DS(test)
    dl = DataLoader(
        ds, batch_size=BS, shuffle=False, num_workers=0, pin_memory=(device == "cuda")
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

    with torch.no_grad():
        if total_models > 0:
            PREDS = torch.zeros((BS, total_models, 6), device=device)
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

            dfb = {
                "eeg_id": eeg_id.cpu().numpy().astype(np.int64),
                "spectrogram_id": spectrogram_id.cpu().numpy().astype(np.int64),
            }
            for k in range(6):
                dfb[votes[k]] = mu[:, k].numpy()
            dfb["sigma"] = sigma.numpy()
            submission_pred = pd.concat(
                [submission_pred, pd.DataFrame(dfb)], ignore_index=True
            )

print(submission_pred.head())



## === cell 17
if "sigma" not in submission_pred.columns:
    submission_pred["sigma"] = 0.0

sigma_vals = submission_pred["sigma"].to_numpy()
sigma_vals = np.nan_to_num(sigma_vals, nan=0.0, posinf=0.0, neginf=0.0)
submission_pred["sigma"] = sigma_vals

if len(sigma_vals) == 0:
    TH_val = 0.0
elif np.allclose(sigma_vals, sigma_vals[0]):
    TH_val = sigma_vals[0]
else:
    TH_val = np.percentile(sigma_vals, TH)

pseudo_labels = submission_pred[submission_pred["sigma"] <= TH_val].copy()
pseudo_labels[votes] *= 10
pseudo_labels.head()



## === cell 18
df = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")
df = df[df[votes].sum(1) >= 10]
df["origin"] = "train"
df.head()




## === cell 19
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(2024)



## === cell 20
m = 0.55
df.expert_consensus = "T"
S = "df.loc[(df.seizure_vote > m*(df.seizure_vote + df.lpd_vote))*(df.seizure_vote > m*(df.seizure_vote + df.gpd_vote))*(df.seizure_vote > m*(df.seizure_vote + df.lrda_vote))*(df.seizure_vote > m*(df.seizure_vote + df.grda_vote))*(df.seizure_vote > m*(df.seizure_vote + df.other_vote)),'expert_consensus'] = 'seizure'"

exec(S)
for a in ["lpd", "gpd", "lrda", "grda", "other"]:
    SWAP = S.replace(a, "SWAP")
    SWAP = SWAP.replace("seizure", a)
    exec(SWAP.replace("SWAP", "seizure"))

consensus = ["seizure", "lpd", "gpd", "lrda", "grda", "other", "T"]



## === cell 21
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




## === cell 22
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
        eeg = eegs[row.origin][row.eeg_id]
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

        self.eeg = np.clip(self.eeg, -1024, 1024) / 32.0
        self.eeg = butter_lowpass_filter(self.eeg)
        eeg_t = torch.from_numpy(self.eeg).float().to(device)

        spec = pd.read_parquet(
            PATH[row["origin"]] + "spectrograms/" + str(row.spectrogram_id) + ".parquet"
        )

        spec_offset = int(row.spectrogram_label_offset_seconds)
        if "time" in spec.columns:
            spec = spec[LL + RL + LP + RP].loc[
                (spec.time >= spec_offset) & (spec.time < spec_offset + 600)
            ]
        else:
            spec = spec[LL + RL + LP + RP]

        t = 22
        if not self.VALID:
            t += int(np.clip(np.random.normal(0, 11), -22, 22))
        spec = spec[LL + RL + LP + RP][t : t + 256]

        self.spec[0, :, :64] = normalize(resize(spec[LP].values, (256, 64)))
        self.spec[0, :, 64:128] = normalize(resize(spec[LL].values, (256, 64)))
        self.spec[0, :, 128:-64] = normalize(resize(spec[RP].values, (256, 64)))
        self.spec[0, :, -64:] = normalize(resize(spec[RL].values, (256, 64)))

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

        spec_t = torch.from_numpy(self.spec.copy()).float().to(device)

        if row["origin"] == "train":
            custom_spec = np.load(
                "/kaggle/input/custom-specs/custom/" + str(row.eeg_id) + ".npy"
            )
        else:
            custom_spec = np.load("/kaggle/working/custom/" + str(row.eeg_id) + ".npy")
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
        custom_spec = resize(custom_spec, (256, 300))

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

        custom_t = torch.from_numpy(self.spec.copy()).float().to(device)
        labels = row[votes].values.astype(np.float32)

        labels_t = torch.from_numpy(labels).to(device)
        return [eeg_t, spec_t, custom_t], labels_t




## === cell 23
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




## === cell 24
def nt(nmin, nmax, tcur, tmax):
    return nmax - 0.5 * (nmax - nmin) * (1 + np.cos(tcur * np.pi / tmax))




## === cell 25
def cb(self):
    learn.loss_func.set_min_v()
    learn.loss_func.set_p()


min_v_cb = Callback(before_step=cb)



## === cell 26
seed_everything(2024)
splits = {}
for c in consensus:
    ID = df.loc[df.expert_consensus == c][CV].unique()
    splits[c] = []
    for t_idx, v_idx in KFold(N_FOLDS).split(ID):
        splits[c].append([ID[t_idx], ID[v_idx]])



## === cell 27
seed_everything(2024)
pseudo_labels_splits = {}
for c in consensus:
    ID = pseudo_labels.loc[pseudo_labels["expert_consensus"] == c][CV].unique()
    if len(ID) > N_FOLDS:
        pseudo_labels_splits[c] = []
        for t_idx, v_idx in KFold(N_FOLDS).split(ID):
            pseudo_labels_splits[c].append([ID[t_idx], ID[v_idx]])



## === cell 28
base_hms_available = all(os.path.exists(p) for p in models["HMS"])
print("Base HMS fold checkpoints available:", base_hms_available)

if len(test) > 1:
    eegs = {"train": {}, "test": {}}
    for eeg_id in tqdm.tqdm(df.eeg_id.unique(), desc="Loading train EEGs"):
        eegs["train"][eeg_id] = pd.read_parquet(
            PATH["train"] + "eegs/" + str(eeg_id) + ".parquet"
        )

    for eeg_id in tqdm.tqdm(
        pseudo_labels.eeg_id.unique(), desc="Loading pseudo-label EEGs"
    ):
        eegs["test"][eeg_id] = pd.read_parquet(
            PATH["test"] + "eegs/" + str(eeg_id) + ".parquet"
        )

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
            c_df = pseudo_labels.loc[pseudo_labels["expert_consensus"] == c]
            t = pd.concat([t, c_df[c_df[CV].isin(pseudo_labels_splits[c][f][0])]])
            v = pd.concat([v, c_df[c_df[CV].isin(pseudo_labels_splits[c][f][1])]])

        tds = HMS_TRAIN_DS(t)
        vds = HMS_TRAIN_DS(v, VALID=True)

        tdl = DataLoader(
            tds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0
        )
        vdl = DataLoader(vds, batch_size=2 * batch_size, num_workers=0)
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
        learn.loss_func.__steps__((learn.dls.dataset.__len__() // batch_size) * EPOCHS)
        learn.fit_one_cycle(EPOCHS)

        torch.save(model, "HMSmodel_" + str(f))
        del tdl, vdl, tds, vds, dls, learn, model
        gc.collect()
else:
    print("Skipping semisupervised training (no test rows).")



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3976965889.py in <cell line: 0>()
     93         )
     94         learn.loss_func.__steps__((learn.dls.dataset.__len__() // batch_size) * EPOCHS)
---> 95         learn.fit_one_cycle(EPOCHS)
     96 
     97         torch.save(model, "HMSmodel_" + str(f))

/usr/local/lib/python3.11/dist-packages/fastai/callback/schedule.py in fit_one_cycle(self, n_epoch, lr_max, div, div_final, pct_start, wd, moms, cbs, reset_opt, start_epoch)
    119     scheds = {'lr': combined_cos(pct_start, lr_max/div, lr_max, lr_max/div_final),
    120               'mom': combined_cos(pct_start, *(self.moms if moms is None else moms))}
--> 121     self.fit(n_epoch, cbs=ParamScheduler(scheds)+L(cbs), reset_opt=reset_opt, wd=wd, start_epoch=start_epoch)
    122 
    123 # %% ../../nbs/14_callback.schedule.ipynb 50

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in fit(self, n_epoch, lr, wd, cbs, reset_opt, start_epoch)
    270             self.opt.set_hypers(lr=self.lr if lr is None else lr)
    271             self.n_epoch = n_epoch
--> 272             self._with_events(self._do_fit, 'fit', CancelFitException, self._end_cleanup)
    273 
    274     def _end_cleanup(self): self.dl,self.xb,self.yb,self.pred,self.loss = None,(None,),(None,),None,None

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_fit(self)
    259         for epoch in range(self.n_epoch):
    260             self.epoch=epoch
--> 261             self._with_events(self._do_epoch, 'epoch', CancelEpochException)
    262 
    263     def fit(self, n_epoch, lr=None, wd=None, cbs=None, reset_opt=False, start_epoch=0):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_epoch(self)
    253 
    254     def _do_epoch(self):
--> 255         self._do_epoch_train()
    256         self._do_epoch_validate()
    257 

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_epoch_train(self)
    245     def _do_epoch_train(self):
    246         self.dl = self.dls.train
--> 247         self._with_events(self.all_batches, 'train', CancelTrainException)
    248 
    249     def _do_epoch_validate(self, ds_idx=1, dl=None):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in all_batches(self)
    211     def all_batches(self):
    212         self.n_iter = len(self.dl)
--> 213         for o in enumerate(self.dl): self.one_batch(*o)
    214 
    215     def _backward(self): self.loss_grad.backward()

/usr/local/lib/python3.11/dist-packages/fastai/data/load.py in __iter__(self)
    127         self.before_iter()
    128         self.__idxs=self.get_idxs() # called in context of main process (not workers/subprocesses)
--> 129         for b in _loaders[self.fake_l.num_workers==0](self.fake_l):
    130             # pin_memory causes tuples to be converted to lists, so convert them back to tuples
    131             if self.pin_memory and type(b) == list: b = tuple(b)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     40                 raise StopIteration
     41         else:
---> 42             data = next(self.dataset_iter)
     43         return self.collate_fn(data)
     44 

/usr/local/lib/python3.11/dist-packages/fastai/data/load.py in create_batches(self, samps)
    138         if self.dataset is not None: self.it = iter(self.dataset)
    139         res = filter(lambda o:o is not None, map(self.do_item, samps))
--> 140         yield from map(self.do_batch, self.chunkify(res))
    141 
    142     def new(self, dataset=None, cls=None, **kwargs):

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in chunked(it, chunk_sz, drop_last, n_chunks, pad, pad_val)
    263     if not isinstance(it, Iterator): it = iter(it)
    264     while True:
--> 265         res = list(itertools.islice(it, chunk_sz))
    266         if res and (len(res)==chunk_sz or not drop_last):
    267             if pad: yield res + [pad_val]*(chunk_sz-len(res))

/usr/local/lib/python3.11/dist-packages/fastai/data/load.py in do_item(self, s)
    168     def prebatched(self): return self.bs is None
    169     def do_item(self, s):
--> 170         try: return self.after_item(self.create_item(s))
    171         except SkipItemException: return None
    172     def chunkify(self, b): return b if self.prebatched else chunked(b, self.bs, self.drop_last)

/usr/local/lib/python3.11/dist-packages/fastai/data/load.py in create_item(self, s)
    175     def retain(self, res, b):  return retain_types(res, b[0] if is_listy(b) else b)
    176     def create_item(self, s):
--> 177         if self.indexed: return self.dataset[s or 0]
    178         elif s is None:  return next(self.it)
    179         else: raise IndexError("Cannot index an iterable dataset numerically - must use `None`.")

/tmp/ipykernel_55/4149656365.py in __getitem__(self, idx)
    103 
    104         if row["origin"] == "train":
--> 105             custom_spec = np.load(
    106                 "/kaggle/input/custom-specs/custom/" + str(row.eeg_id) + ".npy"
    107             )

/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py in load(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size)
    425             own_fid = False
    426         else:
--> 427             fid = stack.enter_context(open(os_fspath(file), "rb"))
    428             own_fid = True
    429 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/custom-specs/custom/2025500940.npy'

## === cell 29
submission = pd.DataFrame()

ds = HMS_DS(test)
dl = DataLoader(
    ds, batch_size=BS, shuffle=False, num_workers=0, pin_memory=(device == "cuda")
)

Last_Bullet = [
    "/kaggle/working/HMSmodel_0",
    "/kaggle/working/HMSmodel_1",
    "/kaggle/working/HMSmodel_2",
    "/kaggle/working/HMSmodel_3",
    "/kaggle/working/HMSmodel_4",
]

eeg_ensemble_final = load_ensemble(models["eeg"])
spec_ensemble_final = load_ensemble(models["spec"])
custom_ensemble_final = load_ensemble(models["custom"])
HMS_ensemble_final = load_ensemble(Last_Bullet)
if len(HMS_ensemble_final) == 0:
    HMS_ensemble_final = load_ensemble(models["HMS"])

print(
    "Final ensemble sizes:",
    {
        "eeg": len(eeg_ensemble_final),
        "spec": len(spec_ensemble_final),
        "custom": len(custom_ensemble_final),
        "HMS": len(HMS_ensemble_final),
        "total": len(eeg_ensemble_final)
        + len(spec_ensemble_final)
        + len(custom_ensemble_final)
        + len(HMS_ensemble_final),
    },
)

with torch.no_grad():
    for eeg_id, spectrogram_id, eegs_b, specs_b, custom_specs_b in tqdm.tqdm(
        dl, desc="Final inference"
    ):
        eegs_b = eegs_b.to(device, non_blocking=True)
        specs_b = specs_b.to(device, non_blocking=True)
        custom_specs_b = custom_specs_b.to(device, non_blocking=True)
        N = eegs_b.shape[0]

        probs_gpu = ensemble_predict_proba(
            eeg_ensemble_final,
            spec_ensemble_final,
            custom_ensemble_final,
            HMS_ensemble_final,
            eegs_b,
            specs_b,
            custom_specs_b,
        )
        if probs_gpu is None:
            preds = torch.full((N, 6), 1 / 6, device="cpu", dtype=torch.float32)
        else:
            preds = probs_gpu.cpu()

        dfb = {"eeg_id": eeg_id.cpu().numpy().astype(np.int64)}
        for i, col in enumerate(votes):
            dfb[col] = preds[:, i].numpy()
        submission = pd.concat([submission, pd.DataFrame(dfb)], ignore_index=True)

submission = submission[["eeg_id"] + votes]
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



## === cell 30
assert os.path.exists("submission.csv")
out = pd.read_csv("submission.csv")
assert list(out.columns) == ["eeg_id"] + votes
assert np.allclose(out[votes].sum(1).values, 1.0, atol=1e-5)
out.head()
