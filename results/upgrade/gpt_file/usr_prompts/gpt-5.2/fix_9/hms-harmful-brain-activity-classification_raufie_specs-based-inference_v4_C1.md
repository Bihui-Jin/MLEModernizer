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

# 5. Target score

0.5422088534144383

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I first fix the empty-weights/empty-predictions failure that causes `predictions` to become a 0‑D array (hence the `AxisError`) by (1) using the correct weights path if present and (2) adding a safe fallback that produces a valid submission even when no weights are found. I also fix a small but real bug in `denoise()` where the thresholding result was a generator (not a list), which can break wavelet denoising if enabled. Finally, I keep the model/inference logic identical when weights exist, but ensure probabilities are always finite and row-normalized before writing `submission.csv` with the required columns.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower is better) is far worse than the target (0.5422), so the smallest change with the highest expected gain is to fix a likely weights-loading mismatch: your model’s head is a custom `Linear`, but many EfficientNet checkpoints store weights for the original classifier head, causing strict loading to fail or to not actually use the right weights. I keep the same architecture/inference pipeline, but make state-dict loading “head-flexible” (drop mismatched head keys and load the rest), and additionally set `model.eval()` + disable grads once per model for stable inference. Finally, I add a safe (but minimal) fallback to average multiple checkpoints and ensure predictions stay normalized and finite exactly as required for KL divergence submissions.'
- What this solution (achieved 1.40995) has done: 'Your current run doesn’t yield a Kaggle score because it likely can’t find/load any usable weights (the referenced dataset path may not exist), so it either crashes earlier or falls back to uniform predictions (which score very poorly). I keep your model/dataset/inference logic intact, but make the weights discovery robust by also searching common Kaggle input locations (including the competition dataset itself) and then load *all matching keys* (including the classifier head when present) to actually use the checkpoint. I also fix a subtle but important bug: in `CustomDataset.__data_generation`, the EEG spectrogram assignment is mistakenly inside the `for region in range(4)` loop (it should be done once), which can distort inputs; moving it out preserves intended semantics and should improve KL. Finally, I ensure we always write a valid `submission.csv` with correct ordering (matching `sample_submission.csv`) and normalized probabilities.'
- What this solution (achieved 1.40995) has done: 'Your current score is far worse than the target (KL lower is better), so we should make the smallest fixes that materially improve predictive signal without changing the model/inference core. The biggest likely issue is that you’re averaging *every* “weight-like” file found across `/kaggle/input`, which include many unrelated checkpoints and destroy predictions; we restrict to only checkpoints that actually match your model by requiring a minimum fraction of keys to load successfully. Next, we fix the incorrect `r` offset logic in `CustomDataset` for non-test mode (it references nonexistent `min/max` columns); while you’re in test mode here, keeping this correct avoids accidental misuse and keeps semantics aligned. Finally, we keep your exact preprocessing/model/softmax, but add a lightweight checkpoint validation step and only ensemble the validated ones, which should move KL substantially toward the target.'

# 9. Code solution

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
from typing import Dict, List

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")




## === cell 1
class config:
    BATCH_SIZE = 64
    MODEL = "tf_efficientnet_b0"
    NUM_WORKERS = 0  # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False

    UNIFORM_MIX_EPS = 0.02

    PARQUET_CACHE_SIZE = 64

    MIN_STATE_DICT_MATCH_FRAC = 0.15
    MAX_MODELS_TO_ENSEMBLE = 6  # keep runtime bounded and ensemble stable


class paths:
    MODEL_WEIGHTS = "/kaggle/input/hba-efficientnet-weights/efficient_net_weights.pt"
    OUTPUT_DIR = "/kaggle/working/"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TEST_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    TEST_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )


weight_roots = [
    "/kaggle/input/hba-efficientnet-weights/",
    "/kaggle/input/hms-harmful-brain-activity-classification/",
    "/kaggle/input/",  # last resort
]
model_weights = []
if os.path.isfile(paths.MODEL_WEIGHTS):
    model_weights.append(paths.MODEL_WEIGHTS)
for root in weight_roots:
    model_weights += glob(os.path.join(root, "**", "*.pth"), recursive=True)
    model_weights += glob(os.path.join(root, "**", "*.pt"), recursive=True)
    model_weights += glob(os.path.join(root, "**", "*.bin"), recursive=True)


def _looks_like_weight(p: str) -> bool:
    bn = os.path.basename(p).lower()
    return any(
        k in bn for k in ["eff", "efficient", "net", "model", "ckpt", "weight", "fold"]
    )


model_weights = sorted(list({p for p in model_weights if _looks_like_weight(p)}))

print("Found weights (pre-filter):", len(model_weights))
print("First few weights:", model_weights[:5])




## === cell 2
USE_WAVELET = None

NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]


def maddest(d, axis: int = None):
    """
    Denoise function.
    """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x: np.ndarray, wavelet: str = "haar", level: int = 1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = [pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:]]
    output = pywt.waverec(coeff, wavelet, mode="per")
    return output


def _read_parquet_robust(path: str) -> pd.DataFrame:
    """
    Try default first, then pyarrow/fastparquet explicitly.
    """
    try:
        return pd.read_parquet(path)
    except Exception:
        try:
            return pd.read_parquet(path, engine="pyarrow")
        except Exception:
            return pd.read_parquet(path, engine="fastparquet")


def spectrogram_from_eeg(parquet_path, display=False):
    eeg = _read_parquet_robust(parquet_path)
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


def plot_spectrogram(spectrogram_path: str):
    sample_spect = _read_parquet_robust(spectrogram_path)

    split_spect = {
        "LL": sample_spect.filter(regex="^LL", axis=1),
        "RL": sample_spect.filter(regex="^RL", axis=1),
        "RP": sample_spect.filter(regex="^RP", axis=1),
        "LP": sample_spect.filter(regex="^LP", axis=1),
    }

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    axes = axes.flatten()
    label_interval = 5
    for i, split_name in enumerate(split_spect.keys()):
        ax = axes[i]
        img = ax.imshow(
            np.log(split_spect[split_name]).T,
            cmap="viridis",
            aspect="auto",
            origin="lower",
        )
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label("Log(Value)")
        ax.set_title(split_name)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time")

        ax.set_yticks(np.arange(len(split_spect[split_name].columns)))
        ax.set_yticklabels(
            [column_name[3:] for column_name in split_spect[split_name].columns]
        )
        frequencies = [
            column_name[3:] for column_name in split_spect[split_name].columns
        ]
        ax.set_yticks(
            np.arange(0, len(split_spect[split_name].columns), label_interval)
        )
        ax.set_yticklabels(frequencies[::label_interval])
    plt.tight_layout()
    plt.show()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sep():
    print("-" * 100)


label_to_num = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
num_to_label = {v: k for k, v in label_to_num.items()}

label_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

seed_everything(config.SEED)




## === cell 3
test_df = pd.read_csv(paths.TEST_CSV)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()




## === cell 4
paths_spectrograms = glob(paths.TEST_SPECTROGRAMS + "*.parquet")
paths_eegs = glob(paths.TEST_EEGS + "*.parquet")
print(f"There are {len(paths_spectrograms)} spectrogram parquets")
print(f"There are {len(paths_eegs)} EEG parquets")

if config.VISUALIZE and len(paths_spectrograms) > 0:
    idx = np.random.randint(0, len(paths_spectrograms))
    plot_spectrogram(paths_spectrograms[idx])




## === cell 5
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
        """
        Reshapes input (128, 256, 8) -> (512, 512, 3) monotone image.
        """
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




## === cell 6
class _LRUCache:
    def __init__(self, maxsize: int = 64):
        self.maxsize = int(maxsize)
        self._data = {}
        self._order = []

    def get(self, key):
        if key in self._data:
            try:
                self._order.remove(key)
            except ValueError:
                pass
            self._order.append(key)
            return self._data[key]
        return None

    def put(self, key, value):
        if key in self._data:
            self._data[key] = value
            try:
                self._order.remove(key)
            except ValueError:
                pass
            self._order.append(key)
            return
        self._data[key] = value
        self._order.append(key)
        if len(self._order) > self.maxsize:
            old = self._order.pop(0)
            self._data.pop(old, None)


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
        self.df = df
        self.config = config
        self.batch_size = self.config.BATCH_SIZE
        self.augment = augment
        self.mode = mode

        self._spec_cache = _LRUCache(maxsize=getattr(config, "PARQUET_CACHE_SIZE", 64))
        self._eeg_cache = _LRUCache(maxsize=getattr(config, "PARQUET_CACHE_SIZE", 64))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X, y = self.__data_generation(index)
        if self.augment:
            X = self.__transform(X)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )

    def _load_spectrogram_values(self, spectrogram_id: int) -> np.ndarray:
        cached = self._spec_cache.get(spectrogram_id)
        if cached is not None:
            return cached
        path = os.path.join(paths.TEST_SPECTROGRAMS, f"{int(spectrogram_id)}.parquet")
        aux = _read_parquet_robust(path)
        arr = aux.iloc[:, 1:].values
        self._spec_cache.put(spectrogram_id, arr)
        return arr

    def _load_eeg_spectrogram(self, eeg_id: int) -> np.ndarray:
        cached = self._eeg_cache.get(eeg_id)
        if cached is not None:
            return cached
        path = os.path.join(paths.TEST_EEGS, f"{int(eeg_id)}.parquet")
        arr = spectrogram_from_eeg(path, display=False)
        self._eeg_cache.put(eeg_id, arr)
        return arr

    def __data_generation(self, index):
        X = np.zeros((128, 256, 8), dtype="float32")
        y = np.zeros(6, dtype="float32")
        row = self.df.iloc[index]

        if self.mode == "test":
            r = 150
        else:
            r = 150

        spec_vals = self._load_spectrogram_values(int(row.spectrogram_id))
        eeg_spec = self._load_eeg_spectrogram(int(row.eeg_id))

        r = int(np.clip(r, 0, max(0, spec_vals.shape[0] - 300)))

        X[:, :, 4:] = eeg_spec

        for region in range(4):
            img = spec_vals[r : r + 300, region * 100 : (region + 1) * 100].T

            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            ep = 1e-6
            mu = np.nanmean(img.flatten())
            std = np.nanstd(img.flatten())
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)
            X[14:-14, :, region] = img[:, 22:-22] / 2.0

            if self.mode != "test":
                y = row[label_cols].values.astype(np.float32)

        return X, y

    def __transform(self, img):
        transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
            ]
        )
        return transforms(image=img)["image"]




## === cell 7
test_dataset = CustomDataset(test_df, config, mode="test")
test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True,
    drop_last=False,
)
X, y = test_dataset[0]
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")




## === cell 8
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    with torch.inference_mode():
        with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
            for step, (X, y) in enumerate(tqdm_test_loader):
                X = X.to(device, non_blocking=True)
                y_preds = model(X)
                y_preds = softmax(y_preds)
                preds.append(y_preds.to("cpu").numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 9
def load_state_dict_flexible(model: nn.Module, checkpoint):
    """
    Load as much as possible from a checkpoint even if some head keys/shapes don't match.
    Returns match statistics so we can filter out unrelated checkpoints (score improvement).
    """
    if isinstance(checkpoint, dict):
        for key in ["model", "state_dict", "model_state_dict", "net", "weights"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                sd = checkpoint[key]
                break
        else:
            sd = checkpoint
    else:
        sd = checkpoint

    model_keys = set(model.state_dict().keys())
    sd_keys = set(sd.keys())
    if (
        len(sd_keys) > 0
        and all(k.startswith("module.") for k in sd_keys)
        and not any(k.startswith("module.") for k in model_keys)
    ):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    model_sd = model.state_dict()
    filtered = {}
    dropped = 0
    for k, v in sd.items():
        if k in model_sd and hasattr(v, "shape") and v.shape == model_sd[k].shape:
            filtered[k] = v
        else:
            dropped += 1

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    loaded = len(filtered)
    total = len(sd) if isinstance(sd, dict) else 0
    match_frac = loaded / max(1, len(model_sd))

    print(
        f"Loaded keys: {loaded}/{total} (dropped={dropped}), "
        f"missing={len(missing)}, unexpected={len(unexpected)}, match_frac={match_frac:.3f}"
    )
    return {
        "loaded": loaded,
        "total_in_ckpt": total,
        "model_keys": len(model_sd),
        "match_frac": match_frac,
    }




## === cell 10
predictions = []
used_weights = []

if len(model_weights) == 0:
    print("WARNING: No model weights found. Falling back to uniform probabilities.")
    predictions = np.full((len(test_df), 6), 1.0 / 6.0, dtype=np.float64)
else:
    for model_weight in model_weights:
        if len(used_weights) >= int(getattr(config, "MAX_MODELS_TO_ENSEMBLE", 6)):
            break

        try:
            checkpoint = torch.load(model_weight, map_location="cpu")
        except Exception as e:
            continue

        model = CustomModel(config)
        stats = load_state_dict_flexible(model, checkpoint)
        if stats["match_frac"] < float(
            getattr(config, "MIN_STATE_DICT_MATCH_FRAC", 0.15)
        ):
            del model, checkpoint
            gc.collect()
            continue

        model.to(device)
        model.eval()

        prediction_dict = inference_function(test_loader, model, device)
        predictions.append(prediction_dict["predictions"])
        used_weights.append(model_weight)

        del model, checkpoint
        torch.cuda.empty_cache()
        gc.collect()

    if len(predictions) == 0:
        print(
            "WARNING: No usable checkpoints after filtering. Falling back to uniform probabilities."
        )
        predictions = np.full((len(test_df), 6), 1.0 / 6.0, dtype=np.float64)
    else:
        print("Ensembling checkpoints:", len(used_weights))
        print("First few used:", used_weights[:5])
        predictions = np.mean(np.array(predictions), axis=0)

print("Predictions shape:", np.array(predictions).shape)




## === cell 11
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

pred = np.asarray(predictions, dtype=np.float64)
if pred.ndim != 2 or pred.shape[0] != len(test_df) or pred.shape[1] != 6:
    raise ValueError(
        f"Invalid prediction array shape: {pred.shape}, expected ({len(test_df)}, 6)"
    )

pred = np.nan_to_num(pred, nan=1.0 / 6.0, posinf=1.0 / 6.0, neginf=1.0 / 6.0)
pred = np.clip(pred, 1e-6, 1.0)
pred = pred / pred.sum(axis=1, keepdims=True)

eps = float(getattr(config, "UNIFORM_MIX_EPS", 0.0))
if eps > 0:
    pred = (1.0 - eps) * pred + eps * (1.0 / 6.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

sample_path = (
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
if not os.path.isfile(sample_path):
    sample_path = "/kaggle/input/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)

sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = pred.astype(np.float32)

sub = sub.groupby("eeg_id", as_index=False)[TARGETS].mean()

vals = sub[TARGETS].values.astype(np.float64)
vals = np.clip(vals, 1e-6, 1.0)
vals = vals / vals.sum(axis=1, keepdims=True)
sub[TARGETS] = vals.astype(np.float32)

sub = sample_sub[["eeg_id"]].merge(sub, on="eeg_id", how="left")
missing_mask = sub[TARGETS].isna().any(axis=1)
if missing_mask.any():
    sub.loc[missing_mask, TARGETS] = 1.0 / 6.0

vals = sub[TARGETS].values.astype(np.float64)
vals = np.clip(vals, 1e-6, 1.0)
vals = vals / vals.sum(axis=1, keepdims=True)
sub[TARGETS] = vals.astype(np.float32)

sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
print(
    "Row sums (min/mean/max):",
    sub[TARGETS].sum(1).min(),
    sub[TARGETS].sum(1).mean(),
    sub[TARGETS].sum(1).max(),
)
sub.head()
