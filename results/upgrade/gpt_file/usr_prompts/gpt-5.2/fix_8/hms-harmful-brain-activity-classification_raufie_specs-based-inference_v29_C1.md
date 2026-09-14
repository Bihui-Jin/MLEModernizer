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
import multiprocessing
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

torch.set_num_threads(max(1, min(8, os.cpu_count() or 2)))
os.environ.setdefault("OMP_NUM_THREADS", str(torch.get_num_threads()))
os.environ.setdefault("MKL_NUM_THREADS", str(torch.get_num_threads()))




## === cell 1
class config:
    BATCH_SIZE = 32
    MODEL = "tf_efficientnet_b1"
    NUM_WORKERS = 0  # keep 0 for Kaggle notebook stability
    PRINT_FREQ = 50
    SEED = 20
    VISUALIZE = False

    EPOCHS = 1
    LR = 1e-3
    WEIGHT_DECAY = 1e-4


class paths:
    MODEL_WEIGHTS = (
        "/kaggle/input/hba-efficientnet-weights/tf_efficientnet_b1_epoch_8.pth"
    )
    OUTPUT_DIR = "/kaggle/working/"
    TRAIN_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TEST_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    TEST_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    TRAIN_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
    TRAIN_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    )


model_weights = []
if os.path.isfile(paths.MODEL_WEIGHTS):
    model_weights = [paths.MODEL_WEIGHTS]
else:
    model_weights = []  # if not available, we'll train a quick model from scratch below

print(f"Using {len(model_weights)} model weight file(s)")
for mw in model_weights[:10]:
    print(" -", mw)




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


_SR = 200
_NFFT = 1024
_NMELS = 128
_FMIN = 0
_FMAX = 20
_WIN_LENGTH = 128
_TARGET_FRAMES = 256
_HOP_LENGTH = 10000 // _TARGET_FRAMES  # same as original: len(x)//256

_mel_fb = librosa.filters.mel(
    sr=_SR, n_fft=_NFFT, n_mels=_NMELS, fmin=_FMIN, fmax=_FMAX, htk=False, norm="slaney"
).astype(np.float32)
_mel_fb_t = torch.from_numpy(_mel_fb)  # (n_mels, 1+n_fft//2)

_win = torch.hann_window(_WIN_LENGTH, periodic=True, dtype=torch.float32)

_TOP_DB = 80.0
_AMIN = 1e-10


def _power_to_db_np(S, ref_max=True):
    S = np.maximum(S, _AMIN)
    if ref_max:
        ref = np.max(S)
    else:
        ref = 1.0
    ref = max(ref, _AMIN)
    log_spec = 10.0 * np.log10(S)
    log_spec -= 10.0 * np.log10(ref)
    log_spec = np.maximum(log_spec, log_spec.max() - _TOP_DB)
    return log_spec


def _mel_spectrogram_fast(x: np.ndarray) -> np.ndarray:
    xt = torch.from_numpy(x.astype(np.float32, copy=False))
    X = torch.stft(
        xt,
        n_fft=_NFFT,
        hop_length=_HOP_LENGTH,
        win_length=_WIN_LENGTH,
        window=_win,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    P = X.real * X.real + X.imag * X.imag  # power spectrogram (freq, frames)
    mel = torch.matmul(_mel_fb_t, P)  # (n_mels, frames)
    mel_np = mel.cpu().numpy()
    return mel_np


def spectrogram_from_eeg(parquet_path, display=False):
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle : middle + 10_000]

    img = np.zeros((128, 256, 4), dtype="float32")

    if display:
        plt.figure(figsize=(10, 7))
    signals = []

    nanmean = np.nanmean
    nan_to_num = np.nan_to_num

    for k in range(4):
        COLS = FEATS[k]
        for kk in range(4):
            a = eeg[COLS[kk]].to_numpy()
            b = eeg[COLS[kk + 1]].to_numpy()
            x = a - b

            m = nanmean(x)
            if np.isnan(x).mean() < 1:
                x = nan_to_num(x, nan=m)
            else:
                x = x.copy()
                x[:] = 0

            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            mel_spec = _mel_spectrogram_fast(x)

            width = (mel_spec.shape[1] // 32) * 32
            mel_spec = mel_spec[:, :width]

            mel_spec_db = _power_to_db_np(mel_spec, ref_max=True).astype(np.float32)
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


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(config.SEED)




## === cell 3
test_df = pd.read_csv(paths.TEST_CSV)
sample_sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)

test_df = sample_sub[["eeg_id"]].merge(test_df, on="eeg_id", how="left")
print(f"Test dataframe shape is: {test_df.shape}")
print(test_df.head())




## === cell 4
class SpectrogramCache:
    def __init__(self, base_dir: str, max_items: int = 256):
        self.base_dir = base_dir
        self.max_items = max_items
        self._cache: Dict[int, np.ndarray] = {}
        self._lru: List[int] = []

    def get(self, sid: int) -> np.ndarray:
        sid = int(sid)
        arr = self._cache.get(sid)
        if arr is not None:
            return arr
        p = os.path.join(self.base_dir, f"{sid}.parquet")
        aux = pd.read_parquet(p)
        arr = aux.iloc[:, 1:].to_numpy()
        del aux
        self._cache[sid] = arr
        self._lru.append(sid)
        if len(self._lru) > self.max_items:
            old = self._lru.pop(0)
            if old in self._cache:
                del self._cache[old]
        return arr


test_spec_cache = SpectrogramCache(paths.TEST_SPECTROGRAMS, max_items=256)
print("Initialized lazy test spectrogram cache")




## === cell 5
def _eeg_worker(file_path: str):
    eeg_id = int(os.path.basename(file_path).split(".")[0])
    return eeg_id, spectrogram_from_eeg(file_path, display=False)


paths_eegs = glob(paths.TEST_EEGS + "*.parquet")
print(f"There are {len(paths_eegs)} EEG parquets (test)")
all_eegs = {}

n_proc = max(1, min(8, (os.cpu_count() or 2)))
ctx = multiprocessing.get_context("fork")

with ctx.Pool(processes=n_proc, maxtasksperchild=100) as pool:
    for eeg_id, eeg_spec in tqdm(
        pool.imap_unordered(_eeg_worker, paths_eegs, chunksize=16),
        total=len(paths_eegs),
        desc=f"Build test EEG spectrograms (proc={n_proc})",
    ):
        all_eegs[eeg_id] = eeg_spec




## === cell 6
class CustomModel(nn.Module):
    def __init__(self, config, num_classes: int = 6):
        super(CustomModel, self).__init__()
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True
        self.model = timm.create_model(config.MODEL, pretrained=False)
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, num_classes),
        )

    def __reshape_input(self, x):
        B, H, W, C = x.shape

        x5 = x.view(B, H, W, 4, 2)

        spectrograms = x5[..., 0].permute(0, 3, 1, 2).contiguous()
        eegs = x5[..., 1].permute(0, 3, 1, 2).contiguous()

        if self.USE_KAGGLE_SPECTROGRAMS & self.USE_EEG_SPECTROGRAMS:
            x2 = torch.cat([spectrograms, eegs], dim=2)  # height concat
        elif self.USE_EEG_SPECTROGRAMS:
            x2 = eegs
        else:
            x2 = spectrograms

        x2 = torch.cat([x2, x2, x2], dim=3)
        return x2

    def forward(self, x):
        x = self.__reshape_input(x)
        x = self.features(x)
        x = self.custom_layers(x)
        return x




## === cell 7
_HFLIP_TRANSFORM = A.Compose([A.HorizontalFlip(p=0.5)])


class CustomDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config,
        augment: bool = False,
        mode: str = "train",
        specs: Dict[int, np.ndarray] = None,
        eeg_specs: Dict[int, np.ndarray] = None,
        spec_cache: SpectrogramCache = None,
    ):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.batch_size = self.config.BATCH_SIZE
        self.augment = augment
        self.mode = mode
        self.spectrograms = specs if specs is not None else {}
        self.eeg_spectrograms = eeg_specs if eeg_specs is not None else {}
        self.spec_cache = spec_cache

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X, y = self.__data_generation(index)
        if self.augment:
            X = self.__transform(X)
        return torch.from_numpy(X), torch.from_numpy(y)

    def _get_spec(self, sid: int) -> np.ndarray:
        sid = int(sid)
        if sid in self.spectrograms:
            return self.spectrograms[sid]
        if self.spec_cache is not None:
            return self.spec_cache.get(sid)
        raise KeyError(f"Spectrogram id {sid} not found")

    def __data_generation(self, index):
        X = np.zeros((128, 256, 8), dtype="float32")
        y = np.zeros(6, dtype="float32")
        row = self.df.iloc[index]

        spec = self._get_spec(int(row.spectrogram_id))

        if self.mode == "test":
            max_r = max(0, spec.shape[0] - 300)
            r = max_r // 2
        else:
            if "min" in row and "max" in row:
                r = int((row["min"] + row["max"]) // 4)
            else:
                max_r = max(0, spec.shape[0] - 300)
                r = max_r // 2

        ep = 1e-6
        for region in range(4):
            img = spec[r : r + 300, region * 100 : (region + 1) * 100].T
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            mu = np.nanmean(img)
            std = np.nanstd(img)
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)
            X[14:-14, :, region] = img[:, 22:-22] / 2.0

        eeg_img = self.eeg_spectrograms[int(row.eeg_id)]
        X[:, :, 4:] = eeg_img

        if self.mode != "test":
            votes = row[label_cols].values.astype(np.float32)
            s = float(votes.sum())
            if s > 0:
                y = votes / s
            else:
                y[:] = 1.0 / 6.0

        return X, y

    def __transform(self, img):
        return _HFLIP_TRANSFORM(image=img)["image"]




## === cell 8
test_dataset = CustomDataset(
    test_df,
    config,
    mode="test",
    specs=None,
    eeg_specs=all_eegs,
    spec_cache=test_spec_cache,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True,
    drop_last=False,
)
X0, y0 = test_dataset[0]
print(f"X shape: {X0.shape}")
print(f"y shape: {y0.shape}")




## === cell 9
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        with torch.inference_mode():
            for step, (X, y) in enumerate(tqdm_test_loader):
                X = X.to(device, non_blocking=True)
                y_preds = model(X)
                y_preds = softmax(y_preds)
                preds.append(y_preds.detach().cpu().numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 10
def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["model", "state_dict", "model_state_dict", "net"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    if isinstance(ckpt, dict):
        return ckpt
    return ckpt


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def kl_divergence(
    p_true: torch.Tensor, p_pred: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    p_true = torch.clamp(p_true, eps, 1.0)
    p_pred = torch.clamp(p_pred, eps, 1.0)
    return torch.sum(p_true * torch.log(p_true / p_pred), dim=1).mean()


def make_train_val_split_by_patient(
    train_df: pd.DataFrame, val_frac: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    patients = train_df["patient_id"].dropna().unique()
    rng = np.random.RandomState(config.SEED)
    rng.shuffle(patients)
    n_val = max(1, int(len(patients) * val_frac))
    val_patients = set(patients[:n_val])
    is_val = train_df["patient_id"].isin(val_patients)
    return train_df.loc[~is_val].reset_index(drop=True), train_df.loc[
        is_val
    ].reset_index(drop=True)




## === cell 11
train_df = pd.read_csv(paths.TRAIN_CSV)

agg = train_df.groupby(["eeg_id", "spectrogram_id", "patient_id"], as_index=False)[
    label_cols
].mean()

train_part, val_part = make_train_val_split_by_patient(agg, val_frac=0.1)
print("Train rows:", len(train_part), "Val rows:", len(val_part))


def load_train_specs_for_ids(spec_ids: np.ndarray) -> Dict[int, np.ndarray]:
    out = {}
    unique_ids = np.unique(spec_ids.astype(int))
    for sid in tqdm(unique_ids, desc="Load train spectrograms"):
        p = os.path.join(paths.TRAIN_SPECTROGRAMS, f"{int(sid)}.parquet")
        aux = pd.read_parquet(p)
        out[int(sid)] = aux.iloc[:, 1:].to_numpy()
        del aux
    return out


def load_train_eegs_for_ids(eeg_ids: np.ndarray) -> Dict[int, np.ndarray]:
    out = {}
    unique_ids = np.unique(eeg_ids.astype(int))
    for eid in tqdm(unique_ids, desc="Build train EEG spectrograms"):
        p = os.path.join(paths.TRAIN_EEGS, f"{int(eid)}.parquet")
        out[int(eid)] = spectrogram_from_eeg(p, display=False)
    return out


MAX_TRAIN_ROWS = 2048
if len(train_part) > MAX_TRAIN_ROWS:
    train_part = train_part.iloc[:MAX_TRAIN_ROWS].reset_index(drop=True)
if len(val_part) > max(256, MAX_TRAIN_ROWS // 8):
    val_part = val_part.iloc[: max(256, MAX_TRAIN_ROWS // 8)].reset_index(drop=True)

train_specs = load_train_specs_for_ids(train_part["spectrogram_id"].values)
val_specs = load_train_specs_for_ids(val_part["spectrogram_id"].values)
train_eegs = load_train_eegs_for_ids(train_part["eeg_id"].values)
val_eegs = load_train_eegs_for_ids(val_part["eeg_id"].values)

train_dataset = CustomDataset(
    train_part,
    config,
    augment=True,
    mode="train",
    specs=train_specs,
    eeg_specs=train_eegs,
)
val_dataset = CustomDataset(
    val_part, config, augment=False, mode="train", specs=val_specs, eeg_specs=val_eegs
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
)




## === cell 12
trained_weight_path = os.path.join(paths.OUTPUT_DIR, "trained_model.pth")

if len(model_weights) == 0:
    print(
        "No external weights found; training a minimal model to improve score vs uniform submission."
    )
    model = CustomModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )

    for epoch in range(config.EPOCHS):
        model.train()
        tr_losses = []
        for X, y in tqdm(train_loader, desc=f"Train epoch {epoch+1}/{config.EPOCHS}"):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            p = torch.softmax(logits, dim=1)
            loss = kl_divergence(y, p)
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.inference_mode():
            for X, y in tqdm(val_loader, desc=f"Val epoch {epoch+1}/{config.EPOCHS}"):
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(X)
                p = torch.softmax(logits, dim=1)
                loss = kl_divergence(y, p)
                va_losses.append(loss.item())

        print(
            f"Epoch {epoch+1}: train_KL={np.mean(tr_losses):.5f} val_KL={np.mean(va_losses):.5f}"
        )

    torch.save({"model": model.state_dict()}, trained_weight_path)
    model_weights = [trained_weight_path]
    del model
    torch.cuda.empty_cache()
    gc.collect()
else:
    print("External weight provided; skipping training.")




## === cell 13
predictions_list = []

for model_weight in model_weights:
    model = CustomModel(config)
    checkpoint = torch.load(model_weight, map_location="cpu")
    state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"WARNING: strict=True load failed for {model_weight}: {repr(e)}")
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    prediction_dict = inference_function(test_loader, model, device)
    predictions_list.append(prediction_dict["predictions"])

    del model, checkpoint, state_dict, prediction_dict
    torch.cuda.empty_cache()
    gc.collect()

predictions = (
    np.mean(np.stack(predictions_list, axis=0), axis=0)
    if len(predictions_list) > 0
    else np.full((len(test_df), 6), 1.0 / 6.0, dtype=np.float32)
)
print("Raw predictions shape:", predictions.shape)




## === cell 14
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

pred = np.asarray(predictions, dtype=np.float64)
if pred.ndim != 2 or pred.shape[1] != 6 or pred.shape[0] != len(test_df):
    raise ValueError(
        f"Bad predictions shape {pred.shape}; expected ({len(test_df)}, 6)"
    )

pred = np.clip(pred, 1e-8, 1.0)
row_sums = pred.sum(axis=1, keepdims=True)
row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
pred = pred / row_sums

sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = pred.astype(np.float32)

out_path = os.path.join(paths.OUTPUT_DIR, "submission.csv")
sub.to_csv(out_path, index=False)
print(f"Submission shape: {sub.shape}")
print("Saved to:", out_path)
print(sub.head())
