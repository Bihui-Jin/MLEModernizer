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
PyWavelets==1.8.0
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

# 5. Code solution

## === cell 0
import albumentations as A
import gc
import librosa
import matplotlib.pyplot as plt
import math
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pywt
import random
import time
import timm
import torch
import torch.nn as nn

from albumentations.pytorch import ToTensorV2
from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, List, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")




## === cell 1
class config:
    BATCH_SIZE = 64
    MODEL = "tf_efficientnet_b0"
    NUM_WORKERS = 0  # keep 0 for notebook stability
    PRINT_FREQ = 50
    SEED = 20
    VISUALIZE = False

    EPOCHS = 1
    LR = 2e-4
    WEIGHT_DECAY = 1e-4
    VAL_FRAC = 0.05  # small holdout; training still uses most data


class paths:
    MODEL_WEIGHTS = "/kaggle/input/hba-efficientnet-weights/efficient_net_weights.pt"
    OUTPUT_DIR = "/kaggle/working/"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TRAIN_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    TEST_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    TEST_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    TRAIN_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    )


model_weights = []
model_weights.extend(glob("/kaggle/input/hba-efficientnet-weights/*.pth"))
model_weights.extend(glob("/kaggle/input/hba-efficientnet-weights/*.pt"))
model_weights = sorted(list(set(model_weights)))

print(
    f"Found {len(model_weights)} model weight files under /kaggle/input/hba-efficientnet-weights/"
)
if len(model_weights) == 0:
    print(
        "[INFO] No external model weights found. Will do minimal local fine-tuning from timm pretrained to improve KL."
    )



## === cell 2
USE_WAVELET = None

NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]

label_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


def maddest(d, axis: int = None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x: np.ndarray, wavelet: str = "haar", level: int = 1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])
    output = pywt.waverec(coeff, wavelet, mode="per")
    return output


def spectrogram_from_eeg(parquet_path, eeg_id=None, display=False):
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

            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
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
            plt.title(f"EEG {eeg_id} - Spectrogram {NAMES[k]}")

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
        plt.title(f"EEG {eeg_id} Signals")
        plt.show()
        print()
        print("#" * 25)
        print()

    return img


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sep():
    print("-" * 100)


label_to_num = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
num_to_label = {v: k for k, v in label_to_num.items()}
seed_everything(config.SEED)



## === cell 3
test_df = pd.read_csv(paths.TEST_CSV)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()



## === cell 4
train_df = pd.read_csv(paths.TRAIN_CSV)
print("Train df:", train_df.shape)

votes = train_df[label_cols].values.astype(np.float64)
votes = np.clip(votes, 0.0, None)
s = votes.sum(axis=1, keepdims=True)
s = np.where(s <= 0, 1.0, s)
train_df["min"] = train_df["spectrogram_label_offset_seconds"].values.astype(np.float64)
train_df["max"] = train_df[
    "min"
]  # keep semantics expected by dataset (uses (min+max)//4)
for i, c in enumerate(label_cols):
    train_df[c] = (votes[:, i] / s[:, 0]).astype(np.float32)

rng = np.random.RandomState(config.SEED)
patients = train_df["patient_id"].unique()
rng.shuffle(patients)
n_val = max(1, int(len(patients) * config.VAL_FRAC))
val_patients = set(patients[:n_val])
trn_df = train_df[~train_df["patient_id"].isin(val_patients)].reset_index(drop=True)
val_df = train_df[train_df["patient_id"].isin(val_patients)].reset_index(drop=True)
print("Split:", trn_df.shape, val_df.shape)



## === cell 5
from functools import lru_cache


def _build_id_to_path_map(folder: str) -> Dict[int, str]:
    id2p = {}
    for fp in glob(os.path.join(folder, "*.parquet")):
        try:
            sid = int(os.path.basename(fp).split(".")[0])
        except Exception:
            continue
        id2p[sid] = fp
    return id2p


test_spec_id2path = _build_id_to_path_map(paths.TEST_SPECTROGRAMS)
train_spec_id2path = _build_id_to_path_map(paths.TRAIN_SPECTROGRAMS)
print(f"Test spectrogram files: {len(test_spec_id2path)}")
print(f"Train spectrogram files: {len(train_spec_id2path)}")


@lru_cache(maxsize=256)
def _load_spec_by_id(spec_id: int, is_train: bool) -> np.ndarray:
    fp = (train_spec_id2path if is_train else test_spec_id2path).get(int(spec_id), None)
    if fp is None:
        raise KeyError(
            f"Spectrogram id {spec_id} not found in {'train' if is_train else 'test'} paths map."
        )
    aux = pd.read_parquet(fp)
    arr = aux.iloc[:, 1:].values
    return arr




## === cell 6
test_eeg_id2path = {}
for fp in glob(os.path.join(paths.TEST_EEGS, "*.parquet")):
    try:
        eid = int(os.path.basename(fp).split(".")[0])
    except Exception:
        continue
    test_eeg_id2path[eid] = fp
print(f"Test EEG files: {len(test_eeg_id2path)}")


def _eeg_worker(args):
    eeg_id, fp = args
    img = spectrogram_from_eeg(fp, eeg_id=str(eeg_id), display=False)
    return int(eeg_id), img


def precompute_test_eeg_mels(test_df: pd.DataFrame, id2path: Dict[int, str], seed: int):
    eeg_ids = test_df["eeg_id"].astype(int).unique()
    items = [(int(eid), id2path[int(eid)]) for eid in eeg_ids]
    items.sort(key=lambda x: x[0])

    ncpu = os.cpu_count() or 2
    workers = max(1, min(8, ncpu // 2))

    t0 = time.time()
    out = {}

    if workers == 1:
        for it in tqdm(items, desc="Precompute EEG mels (serial)", unit="eeg"):
            k, v = _eeg_worker(it)
            out[k] = v
    else:
        ctx = mp.get_context("spawn")
        chunksize = 8
        with ctx.Pool(processes=workers, maxtasksperchild=50) as pool:
            for k, v in tqdm(
                pool.imap_unordered(_eeg_worker, items, chunksize=chunksize),
                total=len(items),
                desc=f"Precompute EEG mels ({workers} workers)",
                unit="eeg",
            ):
                out[k] = v

    dt = time.time() - t0
    print(f"Precomputed {len(out)}/{len(eeg_ids)} EEG mel tensors in {dt:.1f}s")
    return out


if config.VISUALIZE:
    first_id = int(next(iter(test_eeg_id2path.keys())))
    _ = spectrogram_from_eeg(
        test_eeg_id2path[first_id], eeg_id=str(first_id), display=True
    )




## === cell 7
class CustomModel(nn.Module):
    def __init__(self, config, num_classes: int = 6, use_timm_pretrained: bool = False):
        super(CustomModel, self).__init__()
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True

        self.model = timm.create_model(config.MODEL, pretrained=use_timm_pretrained)
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, num_classes),
        )

    def __reshape_input(self, x):
        spectrograms = [x[:, :, :, i : i + 1] for i in range(4)]
        spectrograms = torch.cat(spectrograms, dim=1)

        eegs = [x[:, :, :, i : i + 1] for i in range(4, 8)]
        eegs = torch.cat(eegs, dim=1)

        if self.USE_KAGGLE_SPECTROGRAMS & self.USE_EEG_SPECTROGRAMS:
            x = torch.cat([spectrograms, eegs], dim=2)
        elif self.USE_EEG_SPECTROGRAMS:
            x = eegs
        else:
            x = spectrograms

        x = torch.cat([x, x, x], dim=3)
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, x):
        x = self.__reshape_input(x)
        x = self.features(x)
        x = self.custom_layers(x)
        return x




## === cell 8
class CustomDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config,
        augment: bool = False,
        mode: str = "train",
        specs: Dict[int, np.ndarray] = None,
        eeg_specs: Dict[int, np.ndarray] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.batch_size = self.config.BATCH_SIZE
        self.augment = augment
        self.mode = mode
        self._is_train_specs = mode != "test"

        self._transform = A.Compose([A.HorizontalFlip(p=0.5)]) if augment else None

        self.eeg_specs = eeg_specs if eeg_specs is not None else {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X, y = self.__data_generation(index)
        if self._transform is not None:
            X = self._transform(image=X)["image"]
        return torch.from_numpy(X.astype(np.float32, copy=False)), torch.from_numpy(
            y.astype(np.float32, copy=False)
        )

    def __data_generation(self, index):
        X = np.zeros((128, 256, 8), dtype="float32")
        y = np.zeros(6, dtype="float32")
        row = self.df.iloc[index]

        if self.mode == "test":
            r = 0
        else:
            r = int((row["min"] + row["max"]) // 4)

        spec_id = int(row.spectrogram_id)
        spec = _load_spec_by_id(spec_id, is_train=self._is_train_specs)

        for region in range(4):
            img = spec[r : r + 300, region * 100 : (region + 1) * 100].T

            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            ep = 1e-6
            mu = np.nanmean(img)
            std = np.nanstd(img)
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)

            X[14:-14, :, region] = img[:, 22:-22] / 2.0

        if self.mode == "test":
            eeg_id = int(row.eeg_id)
            X[:, :, 4:] = self.eeg_specs[eeg_id]
        else:
            pass

        if self.mode != "test":
            y = row[label_cols].values.astype(np.float32)

        return X, y




## === cell 9
test_eeg_mels = precompute_test_eeg_mels(test_df, test_eeg_id2path, seed=config.SEED)

trn_dataset = CustomDataset(trn_df, config, mode="train", specs=None, eeg_specs=None)
val_dataset = CustomDataset(val_df, config, mode="train", specs=None, eeg_specs=None)

num_workers = config.NUM_WORKERS
persistent_workers = bool(num_workers and num_workers > 0)
g = torch.Generator()
g.manual_seed(config.SEED)

trn_loader = DataLoader(
    trn_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
    persistent_workers=persistent_workers,
    generator=g,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
    persistent_workers=persistent_workers,
)

test_dataset = CustomDataset(
    test_df, config, mode="test", specs=None, eeg_specs=test_eeg_mels
)
test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
    persistent_workers=persistent_workers,
)

X0, y0 = trn_dataset[0]
print(f"Train sample X shape: {X0.shape}, y shape: {y0.shape}")
Xt, yt = test_dataset[0]
print(f"Test sample X shape: {Xt.shape}, y shape: {yt.shape}")




## === cell 10
def _load_checkpoint_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
            new_sd = {}
            for k, v in sd.items():
                nk = k
                if nk.startswith("model."):
                    nk = nk[len("model.") :]
                if nk.startswith("module."):
                    nk = nk[len("module.") :]
                new_sd[nk] = v
            return new_sd
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        tensor_like = len(ckpt) > 0 and all(hasattr(v, "shape") for v in ckpt.values())
        if tensor_like:
            return ckpt
    raise ValueError("Unrecognized checkpoint format; cannot extract state_dict.")


def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for step, (X, y) in enumerate(tqdm_test_loader):
            X = X.to(device, non_blocking=True)
            with torch.no_grad():
                y_preds = model(X)
            y_preds = softmax(y_preds)
            preds.append(y_preds.detach().cpu().numpy())
    if len(preds) == 0:
        raise RuntimeError("Inference produced no batches; check DataLoader/dataset.")
    return np.concatenate(preds, axis=0)


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=1)
    loss = -(targets * logp).sum(dim=1).mean()
    return loss




## === cell 11
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    losses = []
    for X, y in tqdm(loader, desc="Train", unit="batch"):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = soft_cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().item())
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    losses = []
    for X, y in tqdm(loader, desc="Val", unit="batch"):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(X)
        loss = soft_cross_entropy(logits, y)
        losses.append(loss.detach().cpu().item())
    return float(np.mean(losses)) if losses else float("nan")


def _looks_like_meaningful_head(state_dict: dict) -> bool:
    for k, v in state_dict.items():
        if not hasattr(v, "shape"):
            continue
        if isinstance(k, str) and (k.endswith("weight") or k.endswith("bias")):
            if any(s in k for s in ["custom_layers.2", "classifier", "fc"]):
                try:
                    shape = tuple(v.shape)
                except Exception:
                    continue
                if len(shape) == 2 and shape[0] == 6:
                    return True
                if len(shape) == 1 and shape[0] == 6:
                    return True
    return False


predictions_list = []

if len(model_weights) > 0:
    any_used = False
    for model_weight in model_weights:
        model = CustomModel(config, use_timm_pretrained=False)
        checkpoint = torch.load(model_weight, map_location="cpu")
        state_dict = _load_checkpoint_state_dict(checkpoint)

        cleaned = {}
        for k, v in state_dict.items():
            nk = (
                k[len("module.") :]
                if isinstance(k, str) and k.startswith("module.")
                else k
            )
            cleaned[nk] = v

        if not _looks_like_meaningful_head(cleaned):
            print(
                f"[WARN] {os.path.basename(model_weight)} doesn't look 6-class; skipping."
            )
            del model, checkpoint, state_dict, cleaned
            gc.collect()
            continue

        model.load_state_dict(cleaned, strict=False)
        model.to(device)

        with torch.inference_mode():
            preds = inference_function(test_loader, model, device)
        predictions_list.append(preds)
        any_used = True

        del model, checkpoint, state_dict, cleaned, preds
        torch.cuda.empty_cache()
        gc.collect()

    if not any_used:
        model_weights = []  # fall through to training
        predictions_list = []

if len(model_weights) == 0:
    model = CustomModel(config, use_timm_pretrained=True)

    model.USE_KAGGLE_SPECTROGRAMS = True
    model.USE_EEG_SPECTROGRAMS = False

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )

    for ep in range(config.EPOCHS):
        tr_loss = train_one_epoch(model, trn_loader, optimizer, device)
        va_loss = eval_one_epoch(model, val_loader, device)
        print(
            f"Epoch {ep+1}/{config.EPOCHS} - train_loss={tr_loss:.5f} val_loss={va_loss:.5f}"
        )

    model.USE_KAGGLE_SPECTROGRAMS = True
    model.USE_EEG_SPECTROGRAMS = False

    with torch.inference_mode():
        preds = inference_function(test_loader, model, device)
    predictions_list = [preds]

    del model, optimizer, preds
    torch.cuda.empty_cache()
    gc.collect()

predictions = np.stack(predictions_list, axis=0).mean(axis=0)

if (
    predictions.ndim != 2
    or predictions.shape[0] != len(test_df)
    or predictions.shape[1] != 6
):
    raise ValueError(
        f"Bad predictions shape: {predictions.shape}, expected ({len(test_df)}, 6)"
    )

print("Predictions OK:", predictions.shape)



## === cell 12
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

eps = 1e-6
preds = np.asarray(predictions, dtype=np.float64)
preds = np.nan_to_num(preds, nan=1.0 / 6.0, posinf=1.0, neginf=0.0)
preds = np.clip(preds, eps, 1.0)
row_sums = preds.sum(axis=1, keepdims=True)
row_sums = np.where(row_sums <= 0, 1.0, row_sums)
preds = preds / row_sums

sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = preds.astype(np.float32)

if sub.shape[0] != test_df.shape[0]:
    raise RuntimeError("Submission row count mismatch with test.csv.")
if not np.array_equal(sub["eeg_id"].values, test_df["eeg_id"].values):
    raise RuntimeError(
        "eeg_id order mismatch; refusing to write potentially misaligned submission."
    )

out_path = os.path.join(paths.OUTPUT_DIR, "submission.csv")
sub.to_csv(out_path, index=False)
print(f"Wrote: {out_path}")
print(f"Submission shape: {sub.shape}")
print(
    "Row sums min/max:", sub[TARGETS].sum(axis=1).min(), sub[TARGETS].sum(axis=1).max()
)
sub.head()
