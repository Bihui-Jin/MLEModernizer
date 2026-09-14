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
geopandas==0.14.4
librosa==0.11.0
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
sklearn-pandas==2.2.0
timm==1.0.19
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

0.6538911800803626

# 6. Current score

0.9237

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I fix the immediate runtime blocker by removing the dependency on a missing external model directory and instead generate a valid, score-safe baseline prediction from `train.csv` vote proportions. I also fix logic bugs that would crash test inference anyway (undefined `targets`, invalid use of `min`/`max` columns, and excessive `print` in `__getitem__`), but keep the existing model/dataset code intact and unused to preserve core logic while ensuring end-to-end execution. Finally, I enforce the submission requirements by normalizing rows to sum to 1 and writing `submission.csv` with the exact required columns.'
- What this solution (achieved 1.03903) has done: 'Your current submission is a global class-prior baseline; to move the KL score down toward the target, the smallest legitimate improvement is to make the prior patient-conditional (since `patient_id` exists in both train and test) and to fall back to the global prior for unseen patients. This keeps the same “vote-proportion prior” core approach (no model training/inference changes), but uses available metadata to better match label distributions per patient, which typically reduces KL noticeably versus a single global prior. I also compute the prior from *aggregated-by-(eeg_id, patient_id)* vote proportions (to reduce duplicate/overlap bias from multiple `eeg_sub_id` rows) while preserving the same evaluation semantics (probability simplex, clipped, normalized). The code still run end-to-end quickly and write a valid `submission.csv` with correct columns and row-wise sum-to-1.'
- What this solution (achieved 0.80093) has done: 'Your current approach is a patient-conditional class-prior; the smallest reliable way to reduce KL further (lower-is-better) without changing the core logic is to make the prior more specific than `patient_id` when possible. I compute a hierarchical prior: first try an `eeg_id`-specific prior learned from train (some `eeg_id` can repeat across consolidated rows), else fall back to `patient_id`, else fall back to the global prior—this remains a pure “vote-proportion prior” baseline. I also apply very light additive smoothing (Dirichlet/Laplace) when forming priors to avoid overconfident near-zero probabilities, which typically improves KL. The rest of your code (dataset/model/inference) stays intact and unused; the script still writes a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 1.22847) has done: 'Your current score (0.80093, lower-is-better) is still worse than the target (0.65389), so we should cautiously improve calibration without changing the core “hierarchical prior from vote proportions” logic. The smallest reliable improvement for KL is to mix each specific prior with the global prior (shrinkage), which reduces overconfident wrong priors and typically lowers KL. I also make the Dirichlet/Laplace smoothing scale with the number of aggregated samples (heavier smoothing when an eeg/patient has few observations), which is still the same vote-proportion prior approach but better calibrated. The rest of the model/dataset code remains intact and unused; the script still writes a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 0.9237) has done: 'Your current score is worse than the target (lower-is-better), and the most likely reason (given your prior score history) is that the last change accidentally made calibration much worse by averaging per-(eeg_id, patient_id) probabilities instead of aggregating counts, which can overweight small/noisy groups and miscalibrate KL. I keep the same core “hierarchical prior from vote proportions” logic, but compute EEG- and patient-specific priors from properly aggregated vote counts (Dirichlet-smoothed) and use shrinkage weights based on total vote mass (annotator count) rather than number of groups. This is a minimal semantic fix (still a prior baseline; no model training/inference), but it should move KL back down toward your target. I also keep the strict row-wise normalization and clipping to ensure valid submissions.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os



## === cell 1
import librosa
import albumentations as A
import gc
import matplotlib.pyplot as plt
import math
import multiprocessing
import random
import time
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from albumentations.pytorch import ToTensorV2
from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, List

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")




## === cell 2
class config:
    model = "resnet18d"
    epoch = 10
    lr = 1e-3
    batchsize = 32
    splits = 5
    momentum = 0.9
    MAX_GRAD_NORM = 1e7
    WEIGHT_DECAY = 0.01
    device = "cpu"
    FOLDS = 5
    AMP = True


class paths:
    preloadedeeg = "/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy"
    train_eeg_dir = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs"
    train_spec_dir = (
        "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms"
    )
    train_csv = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    test_csv = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    test_eeg = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs"
    test_spec = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms"
    )
    out = "/kaggle/working/"




## === cell 3
USE_WAVELET = None

NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]


def spectrogram_from_eeg(parquet_path, display=False):
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle : middle + 10_000]

    img = np.zeros((128, 256, 4), dtype="float32")

    if display:
        plt.figure(figsize=(10, 7))
    signals = []
    for k in range(4):
        COLS = FEATS[k]

        for kk in range(4):
            x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values

            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

            signals.append(x)

            mel_spec = librosa.feature.melspectrogram(
                y=x,
                sr=200,
                hop_length=len(x) // 256,
                n_fft=1024,
                n_mels=128,
                fmin=0,
                fmax=20,
                win_length=128,
            )

            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[
                :, :width
            ]

            mel_spec_db = (mel_spec_db + 40) / 40
            img[:, :, k] += mel_spec_db

        img[:, :, k] /= 4.0

        if display:
            plt.subplot(2, 2, k + 1)
            plt.imshow(img[:, :, k], aspect="auto", origin="lower")
            plt.title(f"Spectrogram {NAMES[k]}")

    if display:
        plt.show()
        plt.figure(figsize=(10, 5))
        offset = 0
        for k in range(4):
            if k > 0:
                offset -= signals[3 - k].min()
            plt.plot(range(10_000), signals[k] + offset, label=NAMES[3 - k])
            offset += signals[3 - k].max()
        plt.legend()
        plt.title("EEG Signals")
        plt.show()
        print()
        print("#" * 25)
        print()

    return img




## === cell 4
test_df = pd.read_csv(paths.test_csv)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()



## === cell 5
all_eegs = {}
all_spectrograms = {}



## === cell 6
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
targets = TARGETS  # backward-compatible alias used by CustomDataset




## === cell 7
class CustomDataset:
    def __init__(
        self,
        traindf: pd.DataFrame,
        config,
        mode: str = "train",
        specs: dict[int, np.ndarray] = None,
        eegs: dict[int, np.ndarray] = None,
    ):
        self.traindf = traindf
        self.specs = specs if specs is not None else {}
        self.eeg = eegs if eegs is not None else {}
        self.mode = mode

    def __len__(self):
        return len(self.traindf)

    def __getitem__(self, idx):
        X = np.zeros((128, 256, 8), dtype="float32")
        y = np.zeros(6, dtype="float32")

        row = self.traindf.iloc[idx]

        r = 0

        if len(self.specs) == 0 or len(self.eeg) == 0:
            x = torch.zeros((3, 512, 512), dtype=torch.float32)
            return {"data": x, "target": y}

        for region in range(4):
            img = self.specs[row.spectrogram_id][
                r : r + 300, region * 100 : (region + 1) * 100
            ].T
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            ep = 1e-6
            mu = np.nanmean(img.flatten())
            std = np.nanstd(img.flatten())
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)
            X[14:-14, :, region] = img[:, 22:-22] / 2.0

        img_eeg = self.eeg[row.eeg_id]
        X[:, :, 4:] = img_eeg

        X = torch.tensor(X)
        spectograms = [X[:, :, i : i + 1] for i in range(4)]
        spectograms = torch.cat(spectograms, dim=0)

        eegs = [X[:, :, i : i + 1] for i in range(4, 8)]
        eegs = torch.cat(eegs, dim=0)

        x = torch.cat([spectograms, eegs], dim=1)
        x = torch.cat([x, x, x], dim=2)
        x = x.permute(2, 0, 1)

        if self.mode != "test":
            y = row[targets].values.astype(np.float32)

        return {"data": x, "target": y}




## === cell 8
class Custommodel(nn.Module):
    def __init__(self, config, numclass: int = 6):
        super(Custommodel, self).__init__()
        self.model = timm.create_model(
            config.model,
            pretrained=False,
            drop_rate=0.1,
            drop_path_rate=0.2,
        )
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.customlayer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, numclass),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.customlayer(x)
        return x




## === cell 9
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    for batch in test_loader:
        x = batch["data"].to(device)
        with torch.no_grad():
            ypred = model(x)
        ypred = softmax(ypred)
        preds.append(ypred.detach().cpu().numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 10
train_df = pd.read_csv(paths.train_csv)

BASE_ALPHA = 2.0  # global Dirichlet prior mass
BASE_ALPHA_GROUP = 0.5  # small per-group smoothing to avoid zeros in group priors

MIX_VOTES_EEG = (
    200.0  # higher => more shrinkage to global when few votes for that eeg_id
)
MIX_VOTES_PATIENT = (
    400.0  # higher => more shrinkage to global when few votes for that patient_id
)

grp_cols = ["eeg_id", "patient_id"]
agg_ep = train_df.groupby(grp_cols, as_index=False)[TARGETS].sum()

global_counts = agg_ep[TARGETS].sum(axis=0).to_numpy(dtype=np.float64) + BASE_ALPHA
global_prior = global_counts / global_counts.sum()
global_prior = np.clip(global_prior, 1e-12, None)
global_prior = global_prior / global_prior.sum()

patient_counts_df = agg_ep.groupby("patient_id", as_index=False)[TARGETS].sum()
patient_vote_mass = patient_counts_df[TARGETS].sum(axis=1).to_numpy(dtype=np.float64)
patient_probs = patient_counts_df[TARGETS].to_numpy(dtype=np.float64) + (
    BASE_ALPHA_GROUP / len(TARGETS)
)
patient_probs = patient_probs / patient_probs.sum(axis=1, keepdims=True)
patient_prior = {
    int(pid): {t: float(p[i]) for i, t in enumerate(TARGETS)}
    for pid, p in zip(patient_counts_df["patient_id"].to_numpy(), patient_probs)
}
patient_vote_mass_map = {
    int(pid): float(vm)
    for pid, vm in zip(patient_counts_df["patient_id"].to_numpy(), patient_vote_mass)
}

eeg_counts_df = agg_ep.groupby("eeg_id", as_index=False)[TARGETS].sum()
eeg_vote_mass = eeg_counts_df[TARGETS].sum(axis=1).to_numpy(dtype=np.float64)
eeg_probs = eeg_counts_df[TARGETS].to_numpy(dtype=np.float64) + (
    BASE_ALPHA_GROUP / len(TARGETS)
)
eeg_probs = eeg_probs / eeg_probs.sum(axis=1, keepdims=True)
eeg_prior = {
    int(eid): {t: float(p[i]) for i, t in enumerate(TARGETS)}
    for eid, p in zip(eeg_counts_df["eeg_id"].to_numpy(), eeg_probs)
}
eeg_vote_mass_map = {
    int(eid): float(vm)
    for eid, vm in zip(eeg_counts_df["eeg_id"].to_numpy(), eeg_vote_mass)
}

print("Global prior probs:", dict(zip(TARGETS, global_prior.round(6))))
print("Num patient-specific priors:", len(patient_prior))
print("Num eeg-specific priors:", len(eeg_prior))

predictions = np.zeros((len(test_df), len(TARGETS)), dtype=np.float32)

test_eegs = test_df["eeg_id"].to_numpy()
test_patients = test_df["patient_id"].to_numpy()

for i, (eid, pid) in enumerate(zip(test_eegs, test_patients)):
    p_eeg = eeg_prior.get(int(eid), None)
    if p_eeg is not None:
        v_spec = np.array([p_eeg[t] for t in TARGETS], dtype=np.float64)
        vm = float(eeg_vote_mass_map.get(int(eid), 0.0))
        w = vm / (vm + MIX_VOTES_EEG)
        v = w * v_spec + (1.0 - w) * global_prior
    else:
        p_pat = patient_prior.get(int(pid), None)
        if p_pat is not None:
            v_spec = np.array([p_pat[t] for t in TARGETS], dtype=np.float64)
            vm = float(patient_vote_mass_map.get(int(pid), 0.0))
            w = vm / (vm + MIX_VOTES_PATIENT)
            v = w * v_spec + (1.0 - w) * global_prior
        else:
            v = global_prior.astype(np.float64)

    v = np.clip(v, 1e-12, None)
    v = v / v.sum()
    predictions[i] = v.astype(np.float32)



## === cell 11
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions

vals = sub[TARGETS].to_numpy(dtype=np.float64)
vals = np.clip(vals, 1e-12, None)
vals = vals / vals.sum(axis=1, keepdims=True)
sub[TARGETS] = vals.astype(np.float32)

out_path = os.path.join(paths.out, "submission.csv")
sub.to_csv(out_path, index=False)
print(f"Submission shape: {sub.shape}")
print("Saved to:", out_path)
sub.head()
