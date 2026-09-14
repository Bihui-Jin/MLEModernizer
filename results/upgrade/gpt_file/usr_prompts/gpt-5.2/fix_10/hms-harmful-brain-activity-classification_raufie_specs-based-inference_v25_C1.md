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

0.5714586842162555

# 6. Current score

1.39771

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39771) has done: 'I make the smallest changes needed to ensure you always get a valid `submission.csv` and to move the expected KL score down toward your target by restoring inference when weights are present and preventing silent feature/row misalignment. Specifically: (1) fix the weight path resolution so it correctly finds the provided EfficientNet weights if they exist anywhere under `/kaggle/input`, (2) add a deterministic CPU-safe fallback that still respects the metric (smoothed label-prior) if weights truly aren’t available, and (3) harden submission alignment by enforcing test order, deduplicating on `eeg_id`, and guaranteeing probabilities sum to 1 with no NaNs/zeros. These are execution/validity and metric-relevant changes only; the model architecture, feature extraction, and inference logic remain the same.'
- What this solution (achieved 1.39771) has done: 'Your current score (1.39771, lower-is-better) is far worse than the target (0.57146), so we should improve (decrease) KL with minimal risk while preserving your model and feature logic. The biggest likely issue is a mismatch between how your spectrogram rows are selected (you currently use `r=0` for all test rows) and how the competition data is centered; this can severely hurt predictions even with good weights. I (1) compute `r` for test rows from each spectrogram’s true length to take a centered 300-row window (analogous to what your training metadata tries to do), (2) avoid redundant per-region dictionary lookups and keep everything deterministic, and (3) keep the same post-processing/normalization so the submission remains valid and metric-aligned.'
- What this solution (achieved 1.39771) has done: 'Your score is much worse than the target (1.39771 vs 0.57146, lower-is-better), so we should reduce KL with minimal, metric-aligned fixes while preserving your model and feature pipeline. The most likely high-impact issue is a mismatch between train-time spectrogram windowing and your test-time “center window” heuristic; I compute a test-time row offset that matches the dataset’s intended center (10 minutes centered on the labeled 10s) using spectrogram length and a 600s center, then clip safely. I also ensure the model input tensor is contiguous after permute (avoids subtle backend issues) and enforce deterministic ordering/alignment without changing architecture or training logic. These are small changes expected to meaningfully improve prediction quality (lower KL) without altering the core approach.'
- What this solution (achieved 1.39771) has done: 'We make two minimal, metric-aligned fixes that should reduce KL (lower is better) without changing your model or feature pipeline: (1) compute the test spectrogram window start `r` from each spectrogram’s actual length so the 300-row crop is truly centered (your current hard-coded `r=150` can be badly misaligned), and (2) apply the same centering logic even if some spectrograms are shorter, with safe clipping/padding to avoid NaNs/zeros. These changes preserve architecture, transforms, loss semantics (inference-only here), and submission formatting, but improve input alignment which is a common large KL driver in this competition. Everything else (weights loading, EfficientNet, EEG-derived spectrogram creation, softmax, normalization, and CSV writing) stays the same.'

# 9. Code solution

## === cell 0
import albumentations as A
import gc
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pywt
import random
import timm
import torch
import torch.nn as nn

from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 1
class config:
    BATCH_SIZE = 32
    MODEL = "tf_efficientnet_b0"
    NUM_WORKERS = 0
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False


class paths:
    MODEL_WEIGHTS = (
        "/kaggle/input/hba-efficientnet-weights/tf_efficientnet_b0_epoch_7.pth"
    )
    OUTPUT_DIR = "/kaggle/working/"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TRAIN_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    TEST_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    TEST_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    SAMPLE_SUB = (
        "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
    )


def _find_weight_file(preferred_path: str) -> str | None:
    """
    Change (score/execution relevant): robustly locate the expected weights file.
    If the originally referenced dataset isn't attached, search all of /kaggle/input.
    """
    if isinstance(preferred_path, str) and os.path.exists(preferred_path):
        return preferred_path

    fname = os.path.basename(preferred_path) if isinstance(preferred_path, str) else ""
    if not fname:
        return None

    candidates = glob(f"/kaggle/input/**/{fname}", recursive=True)
    if len(candidates) == 0:
        return None

    candidates = sorted(candidates, key=lambda p: (len(p), p))
    return candidates[0]


resolved_weight = _find_weight_file(paths.MODEL_WEIGHTS)
model_weights = [resolved_weight] if resolved_weight is not None else []

if resolved_weight is None:
    print(
        "WARNING: Could not find model weights. Will use a deterministic smoothed label-prior fallback."
    )
else:
    print("Found model weights at:", resolved_weight)



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
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x: np.ndarray, wavelet: str = "haar", level: int = 1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])
    output = pywt.waverec(coeff, wavelet, mode="per")
    return output


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
        plt.title("Signals")
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
test_df = test_df.sort_values("eeg_id").drop_duplicates("eeg_id").reset_index(drop=True)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()



## === cell 4
USE_MODEL = len(model_weights) > 0

all_spectrograms = {}
all_eegs = {}

if USE_MODEL:
    needed_spec_ids = set(test_df["spectrogram_id"].astype(int).unique().tolist())
    needed_eeg_ids = set(test_df["eeg_id"].astype(int).unique().tolist())

    paths_spectrograms = glob(paths.TEST_SPECTROGRAMS + "*.parquet")
    print(f"There are {len(paths_spectrograms)} spectrogram parquets on disk")
    kept = 0
    for file_path in tqdm(paths_spectrograms, desc="Loading test spectrograms"):
        name = int(os.path.basename(file_path).split(".")[0])
        if name not in needed_spec_ids:
            continue
        aux = pd.read_parquet(file_path)
        all_spectrograms[name] = aux.iloc[:, 1:].to_numpy(dtype=np.float32, copy=False)
        del aux
        kept += 1
    print(f"Loaded {kept} spectrograms used by test.csv")

    paths_eegs = glob(paths.TEST_EEGS + "*.parquet")
    print(f"There are {len(paths_eegs)} EEG parquets on disk")
    counter = 0
    kept = 0
    for file_path in tqdm(paths_eegs, desc="Computing EEG-derived spectrograms"):
        eeg_id = int(os.path.basename(file_path).split(".")[0])
        if eeg_id not in needed_eeg_ids:
            continue
        eeg_spectrogram = spectrogram_from_eeg(
            file_path, display=(config.VISUALIZE and counter < 1)
        )
        all_eegs[eeg_id] = eeg_spectrogram
        counter += 1
        kept += 1
    print(f"Computed {kept} EEG-derived spectrograms used by test.csv")
else:
    print(
        "Skipping spectrogram/EEG feature extraction because no weights were found (using prior fallback)."
    )




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
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x):
        x = self.__reshape_input(x)
        x = self.features(x)
        x = self.custom_layers(x)
        return x




## === cell 6
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
        self.spectrograms = specs if specs is not None else all_spectrograms
        self.eeg_spectrograms = eeg_specs if eeg_specs is not None else all_eegs

        self._test_r_by_specid = {}
        if self.mode == "test":
            spec_ids = self.df["spectrogram_id"].astype(int).unique().tolist()
            for sid in spec_ids:
                spec0 = self.spectrograms.get(int(sid), None)
                if spec0 is None:
                    self._test_r_by_specid[int(sid)] = 0
                    continue

                n_rows = int(spec0.shape[0])
                window = 300
                max_r = max(0, n_rows - window)
                r = int(np.clip((n_rows - window) // 2, 0, max_r))
                self._test_r_by_specid[int(sid)] = r

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X, y = self.__data_generation(index)
        if self.augment:
            X = self.__transform(X)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )

    def __data_generation(self, index):
        X = np.zeros((128, 256, 8), dtype="float32")
        y = np.zeros(6, dtype="float32")

        row = self.df.iloc[index]

        if self.mode == "test":
            r = int(self._test_r_by_specid.get(int(row.spectrogram_id), 0))
        else:
            if ("min" in row.index) and ("max" in row.index):
                r = int((row["min"] + row["max"]) // 4)
            else:
                r = 0

        eeg_img = self.eeg_spectrograms.get(int(row.eeg_id), None)
        if eeg_img is None:
            eeg_img = np.zeros((128, 256, 4), dtype="float32")

        spec = self.spectrograms.get(int(row.spectrogram_id), None)

        for region in range(4):
            if spec is None:
                img = np.zeros((128, 256), dtype="float32")
            else:
                r0 = int(np.clip(r, 0, max(0, spec.shape[0] - 1)))
                r1 = int(min(r0 + 300, spec.shape[0]))
                block = spec[r0:r1, region * 100 : (region + 1) * 100].T  # (100, <=300)

                if block.shape[1] < 300:
                    pad = 300 - block.shape[1]
                    block = np.pad(
                        block,
                        ((0, 0), (0, pad)),
                        mode="constant",
                        constant_values=np.exp(-4),
                    )

                img = np.clip(block, np.exp(-4), np.exp(8))
                img = np.log(img)

                ep = 1e-6
                mu = np.nanmean(img.flatten())
                std = np.nanstd(img.flatten())
                img = (img - mu) / (std + ep)
                img = np.nan_to_num(img, nan=0.0)

                img = img[:, 22:-22] / 2.0
            X[14:-14, :, region] = img

        X[:, :, 4:] = eeg_img

        if self.mode != "test":
            y = row[label_cols].values.astype(np.float32)

        return X, y

    def __transform(self, img):
        transforms = A.Compose([A.HorizontalFlip(p=0.5)])
        return transforms(image=img)["image"]




## === cell 7
if USE_MODEL:
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
else:
    print("Model not used; skipping dataset/loader sanity check.")




## === cell 8
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for step, (X, y) in enumerate(tqdm_test_loader):
            X = X.to(device)
            with torch.no_grad():
                y_preds = model(X)
            y_preds = softmax(y_preds)
            preds.append(y_preds.detach().to("cpu").numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 9
predictions = None

if USE_MODEL:
    fold_predictions = []
    for model_weight in model_weights:
        test_dataset = CustomDataset(test_df, config, mode="test", augment=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        model = CustomModel(config)
        checkpoint = torch.load(model_weight, map_location="cpu")
        state_dict = (
            checkpoint["model"]
            if isinstance(checkpoint, dict) and "model" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded weights from: {model_weight}")

        model.to(device)
        prediction_dict = inference_function(test_loader, model, device)
        fold_predictions.append(prediction_dict["predictions"])

        torch.cuda.empty_cache()
        gc.collect()

    predictions = np.array(fold_predictions)
    predictions = np.mean(predictions, axis=0)

    predictions = np.clip(predictions, 1e-8, 1.0)
    predictions = predictions / predictions.sum(axis=1, keepdims=True)
    print("Predictions shape:", predictions.shape)
else:
    train_df = pd.read_csv(paths.TRAIN_CSV, usecols=label_cols)
    votes = train_df[label_cols].to_numpy(dtype=np.float64)
    row_sums = votes.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    probs = votes / row_sums

    alpha = 1e-3  # minimal smoothing to reduce KL risk from near-zero predictions
    prior = probs.mean(axis=0) + alpha
    prior = np.clip(prior, 1e-8, None)
    prior = prior / prior.sum()

    predictions = np.tile(prior[None, :], (len(test_df), 1)).astype(np.float32)
    print(
        "Using smoothed mean-probability label-prior predictions. Shape:",
        predictions.shape,
    )



## === cell 10
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

sample_sub = pd.read_csv(paths.SAMPLE_SUB)

sub = sample_sub[["eeg_id"]].copy()

pred_df = pd.DataFrame(predictions, columns=TARGETS)
pred_df["eeg_id"] = test_df["eeg_id"].values
pred_df = pred_df.drop_duplicates("eeg_id")

sub = sub.merge(pred_df, on="eeg_id", how="left")

miss = sub[TARGETS].isna().any(axis=1)
if miss.any():
    sub.loc[miss, TARGETS] = 1.0 / len(TARGETS)

vals = sub[TARGETS].to_numpy(dtype=np.float64)
vals = np.nan_to_num(vals, nan=1.0 / len(TARGETS), posinf=1.0, neginf=0.0)
vals = np.clip(vals, 1e-8, 1.0)
vals = vals / vals.sum(axis=1, keepdims=True)
sub[TARGETS] = vals.astype(np.float32)

out_path = os.path.join(paths.OUTPUT_DIR, "submission.csv")
sub.to_csv(out_path, index=False)
print(f"Submission shape: {sub.shape}")
print("Saved to:", out_path)
sub.head()
