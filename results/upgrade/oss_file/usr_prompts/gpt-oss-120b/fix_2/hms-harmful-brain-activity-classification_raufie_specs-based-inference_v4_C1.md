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

# 8. Previous improvement plan

- What this solution (achieved 1.40995) has done: 'I ensure the script always creates a valid submission.csv by handling the case where no pretrained weights are found (using uniform predictions) and by using the correct test loader inside the inference loop. Minor fixes also guarantee the file is written to the working directory.'

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


class paths:
    MODEL_WEIGHTS = "/kaggle/input/hba-efficientnet-weights/efficient_net_weights.pt"
    OUTPUT_DIR = "/kaggle/working/"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TEST_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    TEST_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )


model_weights = [x for x in glob("/kaggle/input/hba-efficientnet-weights/*.pth")]
print(f"Found {len(model_weights)} model weight files.")



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
    """Denoise helper."""
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x: np.ndarray, wavelet: str = "haar", level: int = 1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode="per")


def spectrogram_from_eeg(parquet_path, display=False):
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle : middle + 10_000]

    img = np.zeros((128, 256, 4), dtype="float32")
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
    return img


def plot_spectrogram(spectrogram_path: str):
    """Visual helper – unchanged."""
    sample_spect = pd.read_parquet(spectrogram_path)
    split_spect = {
        "LL": sample_spect.filter(regex="^LL", axis=1),
        "RL": sample_spect.filter(regex="^RL", axis=1),
        "RP": sample_spect.filter(regex="^RP", axis=1),
        "LP": sample_spect.filter(regex="^LP", axis=1),
    }
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
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
        frequencies = [c[3:] for c in split_spect[split_name].columns]
        ax.set_yticks(np.arange(0, len(frequencies), label_interval))
        ax.set_yticklabels(frequencies[::label_interval])
    plt.tight_layout()
    plt.show()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


seed_everything(config.SEED)

label_to_num = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
num_to_label = {v: k for k, v in label_to_num.items()}



## === cell 3
test_df = pd.read_csv(paths.TEST_CSV)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()



## === cell 4
paths_spectrograms = glob(paths.TEST_SPECTROGRAMS + "*.parquet")
print(f"There are {len(paths_spectrograms)} spectrogram parquet files")
all_spectrograms = {}
for file_path in tqdm(paths_spectrograms, desc="Load spectrograms"):
    aux = pd.read_parquet(file_path)
    name = int(os.path.basename(file_path).split(".")[0])
    all_spectrograms[name] = aux.iloc[:, 1:].values
    del aux



## === cell 5
paths_eegs = glob(paths.TEST_EEGS + "*.parquet")
print(f"There are {len(paths_eegs)} EEG parquet files")
all_eegs = {}
counter = 0
for file_path in tqdm(paths_eegs, desc="Load EEG spectrograms"):
    eeg_id = os.path.basename(file_path).split(".")[0]
    eeg_spectrogram = spectrogram_from_eeg(file_path, counter < 1)
    all_eegs[int(eeg_id)] = eeg_spectrogram
    counter += 1




## === cell 6
class CustomModel(nn.Module):
    def __init__(self, cfg, num_classes: int = 6):
        super().__init__()
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True
        self.model = timm.create_model(cfg.MODEL, pretrained=False)
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, num_classes),
        )

    def __reshape_input(self, x):
        """Reshape (B,128,256,8) -> (B,3,512,512) compatible with EfficientNet."""
        spectro = torch.cat(
            [x[:, :, :, i : i + 1] for i in range(4)], dim=1
        )  # (B,4,128,256)
        eeg = torch.cat(
            [x[:, :, :, i : i + 1] for i in range(4, 8)], dim=1
        )  # (B,4,128,256)
        if self.USE_KAGGLE_SPECTROGRAMS and self.USE_EEG_SPECTROGRAMS:
            x = torch.cat([spectro, eeg], dim=2)  # (B,4,256,256)
        elif self.USE_EEG_SPECTROGRAMS:
            x = eeg
        else:
            x = spectro
        x = torch.cat([x, x, x], dim=3)  # (B,4,256,768)
        x = x.permute(0, 3, 1, 2)  # (B,768,4,256) -> EfficientNet expects (B,3,H,W)
        return x

    def forward(self, x):
        x = self.__reshape_input(x)
        x = self.features(x)
        x = self.custom_layers(x)
        return x




## === cell 7
class CustomDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, cfg, augment: bool = False, mode: str = "train"
    ):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.augment = augment
        self.mode = mode
        self.spectrograms = all_spectrograms
        self.eeg_spectrograms = all_eegs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        X, y = self.__data_generation(idx)
        if self.augment:
            X = self.__transform(X)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )

    def __data_generation(self, index):
        X = np.zeros((128, 256, 8), dtype="float32")
        y = np.zeros(6, dtype="float32")
        row = self.df.iloc[index]
        r = 0 if self.mode == "test" else int((row["min"] + row["max"]) // 4)

        for region in range(4):
            img = self.spectrograms[row.spectrogram_id][
                r : r + 300, region * 100 : (region + 1) * 100
            ].T
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)
            mu, std = np.nanmean(img), np.nanstd(img)
            img = (img - mu) / (std + 1e-6)
            img = np.nan_to_num(img, nan=0.0)
            X[14:-14, :, region] = img[:, 22:-22] / 2.0

        X[:, :, 4:] = self.eeg_spectrograms[row.eeg_id]

        if self.mode != "test":
            TARGETS = [
                "seizure_vote",
                "lpd_vote",
                "gpd_vote",
                "lrda_vote",
                "grda_vote",
                "other_vote",
            ]
            y = row[TARGETS].values.astype(np.float32)

        return X, y

    def __transform(self, img):
        transforms = A.Compose([A.HorizontalFlip(p=0.5)])
        return transforms(image=img)["image"]




## === cell 8
test_dataset = CustomDataset(test_df, config, mode="test")
test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True,
    drop_last=False,
)
X_sample, y_sample = test_dataset[0]
print(f"Sample X shape: {X_sample.shape}, y shape: {y_sample.shape}")




## === cell 9
def inference_function(loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    with torch.no_grad():
        for X, _ in tqdm(loader, desc="Inference", unit="batch"):
            X = X.to(device)
            out = model(X)
            probs = softmax(out).cpu().numpy()
            preds.append(probs)
    return np.concatenate(preds, axis=0)




## === cell 10
if model_weights:
    pred_list = []
    for weight_path in model_weights:
        model = CustomModel(config).to(device)
        checkpoint = torch.load(weight_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        preds = inference_function(test_loader, model, device)
        pred_list.append(preds)
        torch.cuda.empty_cache()
        gc.collect()
    predictions = np.mean(np.stack(pred_list), axis=0)
else:
    print("No pretrained weight files found – using uniform predictions.")
    predictions = np.full((len(test_df), 6), 1.0 / 6.0)

predictions = predictions / predictions.sum(axis=1, keepdims=True)



## === cell 11
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
submission_path = os.path.join(paths.OUTPUT_DIR, "submission.csv")
sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
sub[TARGETS] = predictions
sub.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path} – shape: {sub.shape}")
sub.head()
