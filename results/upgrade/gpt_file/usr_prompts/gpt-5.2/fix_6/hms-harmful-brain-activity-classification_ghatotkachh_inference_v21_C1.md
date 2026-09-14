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

0.5752877302313822

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the crash caused by missing external model weights by falling back to a deterministic, valid probability baseline when `/kaggle/input/resnet5010ep2` is not present. I also fix multiple runtime/logic errors in the dataset pipeline (undefined `targets`, debug `print` spam, incorrect spectrogram slicing/indexing, and unnecessary heavy preloading that would likely time out). Finally, I ensure the produced submission has the exact required columns, correct row alignment with `test.csv`, and per-row probabilities that sum to 1 so Kaggle accepts the file.'
- What this solution (achieved 1.53235) has done: 'Your current score is much worse than the target (lower-is-better), and the biggest cause is that you’re falling back to a constant class-prior prediction (or using mismatched external weights), which is typically very high KL. To move the score toward the target with minimal change and without altering the model, we generate out-of-fold predictions on the provided training data using the exact same dataset/model/inference pipeline and then use patient-aware KNN over those OOF embeddings/probabilities to predict test rows (no label leakage, and it preserves the existing core logic). We also ensure we aggregate train rows correctly to the competition’s per-`eeg_id` target distribution (normalize votes) and keep submission normalization/clipping identical. This should substantially reduce KL versus a constant prior while staying within Kaggle constraints and keeping architecture/training logic unchanged (we’re only adding a lightweight, no-training “retrieval” post-process when external weights aren’t available).'
- What this solution (achieved 1.08823) has done: 'Your current score (1.53235, lower-is-better) is far from the target (0.5753), so we need a meaningful but still minimal change in the fallback path (when external weights are absent) to reduce KL. I keep your existing KNN-on-metadata core fallback, but make it more label-consistent by (1) using patient-aware neighbor restriction (same patient when possible) and (2) mixing in a global class prior with a small weight to avoid overconfident/erratic per-patient predictions that KL heavily penalizes. I also ensure we predict at the required per-`eeg_id` level explicitly (unique test `eeg_id` order), which avoids any accidental misalignment risk. The model/architecture/inference path with checkpoints remains unchanged.'

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
    model = "resnet50d"
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


TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



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
    for k in range(4):
        COLS = FEATS[k]
        for kk in range(4):
            x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values
            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

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
    return img




## === cell 4
test_df = pd.read_csv(paths.test_csv)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()



## === cell 5
all_eegs = None
all_spectrograms = None




## === cell 6
class CustomDataset(Dataset):
    def __init__(
        self,
        traindf: pd.DataFrame,
        config,
        mode: str = "train",
        specs: dict | None = None,
        eegs: dict | None = None,
        spec_dir: str = paths.test_spec,
        eeg_dir: str = paths.test_eeg,
    ):
        self.traindf = traindf.reset_index(drop=True)
        self.specs = specs
        self.eeg = eegs
        self.mode = mode
        self.spec_dir = spec_dir
        self.eeg_dir = eeg_dir

    def __len__(self):
        return len(self.traindf)

    def _load_spec(self, spectrogram_id: int) -> np.ndarray:
        if self.specs is not None and spectrogram_id in self.specs:
            return self.specs[spectrogram_id]
        fp = os.path.join(self.spec_dir, f"{int(spectrogram_id)}.parquet")
        return pd.read_parquet(fp).to_numpy()

    def _load_eeg_img(self, eeg_id: int) -> np.ndarray:
        if self.eeg is not None and eeg_id in self.eeg:
            return self.eeg[eeg_id]
        fp = os.path.join(self.eeg_dir, f"{int(eeg_id)}.parquet")
        return spectrogram_from_eeg(fp, display=False)

    def __getitem__(self, idx):
        X = np.zeros((128, 256, 8), dtype="float32")
        y = np.zeros(6, dtype="float32")
        row = self.traindf.iloc[idx]

        spec = self._load_spec(int(row.spectrogram_id))
        r = 0
        if spec.shape[0] >= 300:
            r = max(0, (spec.shape[0] - 300) // 2)

        for region in range(4):
            c0, c1 = region * 100, (region + 1) * 100
            if spec.shape[1] < c1:
                region_slice = np.zeros((300, 100), dtype=np.float32)
                take = spec[
                    r : r + min(300, spec.shape[0] - r), c0 : min(c1, spec.shape[1])
                ]
                region_slice[: take.shape[0], : take.shape[1]] = take.astype(np.float32)
            else:
                region_slice = spec[r : r + 300, c0:c1].astype(np.float32)

            img = region_slice.T  # (100, 300)
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            ep = 1e-6
            mu = np.nanmean(img)
            std = np.nanstd(img)
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)

            if img.shape[1] >= 256:
                img_w = img[
                    :, (img.shape[1] - 256) // 2 : (img.shape[1] - 256) // 2 + 256
                ]
            else:
                img_w = np.zeros((img.shape[0], 256), dtype=np.float32)
                img_w[:, : img.shape[1]] = img

            v0 = (128 - img.shape[0]) // 2
            X[v0 : v0 + img.shape[0], :, region] = img_w / 2.0

        eeg_img = self._load_eeg_img(int(row.eeg_id))  # (128,256,4)
        X[:, :, 4:] = eeg_img.astype(np.float32)

        X = torch.tensor(X)
        spectograms = torch.cat([X[:, :, i : i + 1] for i in range(4)], dim=0)
        eegs = torch.cat([X[:, :, i : i + 1] for i in range(4, 8)], dim=0)

        x = torch.cat([spectograms, eegs], dim=1)  # (512,512,1)
        x = torch.cat([x, x, x], dim=2)  # (512,512,3)
        x = x.permute(2, 0, 1).contiguous()  # (3,512,512)

        if self.mode != "test":
            y = row[TARGETS].values.astype(np.float32)

        return {"data": x, "target": torch.tensor(y)}




## === cell 7
customdataset = CustomDataset(
    test_df, config, mode="test", spec_dir=paths.test_spec, eeg_dir=paths.test_eeg
)
print("Single item data shape:", customdataset[0]["data"].shape)



## === cell 8
test_loader = DataLoader(
    customdataset,
    batch_size=config.batchsize,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)
batch = next(iter(test_loader))
print("Batch data shape:", batch["data"].shape)




## === cell 9
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




## === cell 10
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    for batch in tqdm(test_loader, desc="Infer", leave=False):
        x = batch["data"].to(device, non_blocking=True)
        with torch.no_grad():
            ypred = model(x)
            ypred = softmax(ypred)
        preds.append(ypred.detach().cpu().numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 11
def _make_eeg_level_targets(train_df: pd.DataFrame) -> pd.DataFrame:
    df = train_df[["eeg_id"] + TARGETS].copy()

    v = df[TARGETS].to_numpy(dtype=np.float64)
    v = np.clip(v, 0.0, None)
    df[TARGETS] = v

    g = df.groupby("eeg_id", as_index=False)[TARGETS].mean()

    y = g[TARGETS].to_numpy(dtype=np.float64)
    y = np.clip(y, 1e-12, None)
    y = y / y.sum(axis=1, keepdims=True)

    out = pd.DataFrame({"eeg_id": g["eeg_id"].values})
    out[TARGETS] = y.astype(np.float32)
    return out


def _make_meta_features(df: pd.DataFrame) -> np.ndarray:
    pid = df["patient_id"].astype(np.int64).values
    sid = df["spectrogram_id"].astype(np.int64).values
    eid = df["eeg_id"].astype(np.int64).values
    X = np.stack(
        [
            pid / 1e6,
            sid / 1e9,
            eid / 1e9,
        ],
        axis=1,
    ).astype(np.float32)
    return X


def _knn_predict(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    k: int = 50,
) -> np.ndarray:
    k = int(k)
    n_test = test_X.shape[0]
    preds = np.zeros((n_test, train_y.shape[1]), dtype=np.float64)

    train_norm = ((train_X * train_X).sum(axis=1, keepdims=True)).astype(np.float64)
    train_X64 = train_X.astype(np.float64)

    bs = 512
    for i in tqdm(range(0, n_test, bs), desc="KNN", leave=False):
        Xb = test_X[i : i + bs].astype(np.float64)
        test_norm = (Xb * Xb).sum(axis=1, keepdims=True)
        dist2 = test_norm + train_norm.T - 2.0 * (Xb @ train_X64.T)
        dist2 = np.maximum(dist2, 0.0)

        idx = np.argpartition(dist2, kth=min(k, dist2.shape[1] - 1), axis=1)[:, :k]
        d2 = np.take_along_axis(dist2, idx, axis=1)

        med = np.median(d2, axis=1, keepdims=True)
        bw2 = np.maximum(med, 1e-6)
        w = np.exp(-d2 / (2.0 * bw2))
        wsum = np.sum(w, axis=1, keepdims=True)
        w = w / np.maximum(wsum, 1e-12)

        yk = train_y[idx]
        pb = (w[:, :, None] * yk).sum(axis=1)
        preds[i : i + bs] = pb

    return preds.astype(np.float32)


def _knn_predict_patient_aware(
    train_rep: pd.DataFrame,
    test_rep: pd.DataFrame,
    k: int = 80,
    k_patient: int = 80,
) -> np.ndarray:
    train_X_all = _make_meta_features(train_rep)
    train_y_all = train_rep[TARGETS].to_numpy(dtype=np.float32)

    test_X = _make_meta_features(test_rep)

    preds_global = _knn_predict(train_X_all, train_y_all, test_X, k=k)

    preds = preds_global.copy()

    train_groups = {}
    for pid, idxs in train_rep.groupby("patient_id").indices.items():
        train_groups[int(pid)] = np.asarray(list(idxs), dtype=np.int64)

    for pid, test_idxs in test_rep.groupby("patient_id").indices.items():
        pid = int(pid)
        if pid not in train_groups:
            continue
        tr_idx = train_groups[pid]
        if tr_idx.size < 5:
            continue

        train_X = train_X_all[tr_idx]
        train_y = train_y_all[tr_idx]
        test_X_pid = test_X[np.asarray(list(test_idxs), dtype=np.int64)]
        preds_pid = _knn_predict(
            train_X, train_y, test_X_pid, k=min(k_patient, len(tr_idx))
        )
        preds[np.asarray(list(test_idxs), dtype=np.int64)] = preds_pid

    return preds


def _blend_with_prior(
    preds: np.ndarray, prior: np.ndarray, alpha: float = 0.15
) -> np.ndarray:
    alpha = float(alpha)
    p = preds.astype(np.float64)
    pr = prior.astype(np.float64)[None, :]
    out = (1.0 - alpha) * p + alpha * pr
    return out.astype(np.float32)


weights_dir = "/kaggle/input/resnet5010ep2"
train_path = paths.train_csv

if os.path.isdir(weights_dir) and len(os.listdir(weights_dir)) > 0:
    predictions_list = []
    for fn in sorted(os.listdir(weights_dir)):
        ckpt_path = os.path.join(weights_dir, fn)
        try:
            dd = torch.load(ckpt_path, map_location="cpu")
        except Exception:
            continue
        model = Custommodel(config)
        if isinstance(dd, dict) and "model" in dd:
            model.load_state_dict(dd["model"], strict=True)
        else:
            model.load_state_dict(dd, strict=True)

        model.to(device)
        out = inference_function(test_loader, model, device)
        predictions_list.append(out["predictions"])

    if len(predictions_list) == 0:
        raise RuntimeError(
            "No usable checkpoints found in weights_dir; cannot run model inference."
        )
    predictions = np.mean(np.stack(predictions_list, axis=0), axis=0)
else:
    train_df = pd.read_csv(train_path)

    train_eeg_targets = _make_eeg_level_targets(train_df)

    train_rep = (
        train_df.sort_values(["eeg_id", "spectrogram_id"])
        .drop_duplicates("eeg_id", keep="first")[
            ["eeg_id", "patient_id", "spectrogram_id"]
        ]
        .merge(train_eeg_targets, on="eeg_id", how="inner")
        .reset_index(drop=True)
    )

    test_rep = (
        test_df[["eeg_id", "patient_id", "spectrogram_id"]]
        .copy()
        .reset_index(drop=True)
    )

    prior = train_eeg_targets[TARGETS].to_numpy(dtype=np.float64).mean(axis=0)
    prior = np.clip(prior, 1e-12, None)
    prior = prior / prior.sum()

    base_preds = _knn_predict_patient_aware(train_rep, test_rep, k=80, k_patient=80)

    predictions = _blend_with_prior(base_preds, prior, alpha=0.25)

print("Predictions shape:", predictions.shape)



## === cell 12
predictions = np.clip(predictions, 1e-6, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

ss_path = (
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
sample_sub = pd.read_csv(ss_path)

sub = sample_sub[["eeg_id"]].copy()
pred_map = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
pred_map[TARGETS] = predictions
sub = sub.merge(pred_map, on="eeg_id", how="left")

miss = sub[TARGETS].isna().any(axis=1)
if miss.any():
    sub.loc[miss, TARGETS] = 1.0 / len(TARGETS)
sub[TARGETS] = np.clip(sub[TARGETS].to_numpy(dtype=np.float64), 1e-6, 1.0)
sub[TARGETS] = sub[TARGETS] / sub[TARGETS].sum(axis=1).values.reshape(-1, 1)

out_path = os.path.join(paths.out, "submission.csv")
sub.to_csv(out_path, index=False)

print(f"Submission shape: {sub.shape}")
print("Saved to:", out_path)
print(
    "Row sum check (min/max):",
    sub[TARGETS].sum(axis=1).min(),
    sub[TARGETS].sum(axis=1).max(),
)
sub.head()
