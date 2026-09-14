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

0.72869

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I fix the missing model‑weight file error and replace the heavy model inference with a lightweight baseline that uses the class distribution from the training labels. I also simplify the data‑loading cells so they no longer try to read huge parquet files, preventing memory or file‑not‑found issues. The final cells compute per‑class probabilities from train.csv, repeat them for every test row, and write a correct submission.csv with probabilities that sum to 1.'
- What this solution (achieved 1.64506) has done: 'I replace the simple global‑mean baseline with a patient‑wise prior: for each patient present in the training data I compute the average normalized vote distribution and use it for the test rows sharing that patient_id, falling back to the global mean when a patient is unseen. This small change keeps the overall pipeline unchanged while providing a more informed probability estimate, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.71539) has done: 'Implemented missing imports (glob, os, random, numpy, pandas, torch, torch.nn, Dataset, albumentations, pywt, matplotlib, tqdm) and corrected the initial setup so all subsequent cells have access to required libraries. No changes to the core modeling or inference logic were made; the baseline patient‑wise prior computation now runs without errors and writes a valid `submission.csv` with probabilities that sum to 1. This fixes the runtime failures and enables generation of a proper submission file, moving the score toward the target.'
- What this solution (achieved 0.7224) has done: 'Implemented a smaller smoothing factor (ALPHA = 1.0) to let patient‑specific vote distributions dominate when available, and added a tiny epsilon‑smoothing step to avoid zero probabilities before the final normalization. This modest adjustment is expected to lower the KL‑divergence, moving the score closer to the target while preserving the original pipeline.'
- What this solution (achieved 0.7776) has done: 'Implemented a smaller smoothing factor (ALPHA = 0.1) so that patient‑specific vote distributions dominate more strongly when available, while still falling back to the global mean for unseen patients. This modest tweak keeps the overall pipeline unchanged but is expected to produce probability estimates closer to the true label distribution, thereby lowering the KL‑divergence toward the target score.'
- What this solution (achieved 0.71277) has done: 'I increase the smoothing factor (ALPHA) from 0.1 to 3.0 so that the patient‑specific priors are blended more strongly with the global class distribution. This larger global weight should produce more calibrated probabilities and lower the KL‑divergence, moving the score closer to the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.72545) has done: 'I replace the global prior (currently the mean of per‑row normalized votes) with the overall class distribution computed from all votes in the training set. This global distribution better reflects the true label frequencies, so blending patient‑specific priors with it should produce probabilities closer to the target KL‑divergence without altering any other part of the pipeline.'
- What this solution (achieved 0.72661) has done: 'Implemented a modest reduction of the smoothing factor `ALPHA` from 3.0 to 2.0, giving patient‑specific priors a slightly higher influence while still retaining global regularisation. This small tweak is expected to lower the KL‑divergence (move the score toward the target) without altering the overall pipeline or model architecture.'
- What this solution (achieved 0.73577) has done: 'I reduce the smoothing factor `ALPHA` so patient‑specific priors dominate more strongly and compute the global prior as the simple average of per‑row normalized vote distributions (instead of the vote‑count weighted frequency). This keeps the overall pipeline unchanged while giving a more calibrated blend that should lower the KL‑divergence, moving the score closer to the target.'
- What this solution (achieved 0.72545) has done: 'I adjust the baseline prior by (1) computing the global class distribution as the vote‑count‑weighted frequencies (which better reflects the true label mix) and (2) increasing the smoothing factor `ALPHA` to 3.0 so the global prior has a stronger influence. These tiny changes keep the overall pipeline unchanged but should produce better‑calibrated probabilities and lower the KL‑divergence, moving the score closer to the target.'
- What this solution (achieved 0.72869) has done: 'I increase the smoothing factor `ALPHA` from 3.0 to 5.0 so the global class distribution has a stronger influence when blending with patient‑specific priors. This small adjustment keeps the core logic unchanged while giving more weight to the overall label frequencies, which should lower the KL‑divergence and move the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import random
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import albumentations as A
import pywt
import matplotlib.pyplot as plt

from tqdm import tqdm


class config:
    BATCH_SIZE = 32
    MODEL = "tf_efficientnet_b0"
    NUM_WORKERS = 0  # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False


class paths:
    MODEL_WEIGHTS = (
        "/kaggle/input/hba-efficientnet-weights/tf_efficientnet_b0_epoch_7.pth"
    )
    OUTPUT_DIR = "/kaggle/working/"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TEST_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    TEST_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )


model_weights = [paths.MODEL_WEIGHTS]



## === cell 1
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
    coeff = pywt.wavedec(
        x, wavelet, mode="per"
    )  # multilevel 1D Discrete Wavelet Transform of data.
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
        plt.show()
        print()
        print("#" * 25)
        print()

    return img


def plot_spectrogram(spectrogram_path: str):
    """
    Source: https://www.kaggle.com/code/mvvppp/hms-eda-and-domain-journey
    Visualize spectrogram recordings from a parquet file.
    :param spectrogram_path: path to the spectrogram parquet.
    """
    sample_spect = pd.read_parquet(spectrogram_path)

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


def sep():
    print("-" * 100)


label_to_num = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
num_to_label = {v: k for k, v in label_to_num.items()}
seed_everything(config.SEED)



## === cell 2
test_df = pd.read_csv(paths.TEST_CSV)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()



## === cell 3
paths_spectrograms = glob(paths.TEST_SPECTROGRAMS + "*.parquet")
print(
    f"(Placeholder) Found {len(paths_spectrograms)} spectrogram files – not loading them for the baseline."
)
all_spectrograms = {}  # empty dict used by the dataset stub



## === cell 4
paths_eegs = glob(paths.TEST_EEGS + "*.parquet")
print(
    f"(Placeholder) Found {len(paths_eegs)} EEG files – not loading them for the baseline."
)
all_eegs = {}  # empty dict used by the dataset stub




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
class CustomDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config,
        augment: bool = False,
        mode: str = "train",
        specs: dict = None,
        eeg_specs: dict = None,
    ):
        self.df = df
        self.config = config
        self.batch_size = self.config.BATCH_SIZE
        self.augment = augment
        self.mode = mode
        self.spectrograms = specs if specs is not None else {}
        self.eeg_spectrograms = eeg_specs if eeg_specs is not None else {}

    def __len__(self):
        """Denotes the number of samples."""
        return len(self.df)

    def __getitem__(self, index):
        """Generate one sample."""
        X, y = self.__data_generation(index)
        if self.augment:
            X = self.__transform(X)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )

    def __data_generation(self, index):
        """Generates data containing a single sample."""
        X = np.zeros((128, 256, 8), dtype="float32")
        y = np.zeros(6, dtype="float32")
        img = np.ones((128, 256), dtype="float32")
        row = self.df.iloc[index]
        if self.mode == "test":
            r = 0
        else:
            r = int((row["min"] + row["max"]) // 4)

        for region in range(4):
            if row.spectrogram_id in self.spectrograms:
                img = self.spectrograms[row.spectrogram_id][
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
            if row.eeg_id in self.eeg_spectrograms:
                X[:, :, 4:] = self.eeg_spectrograms[row.eeg_id]

            if self.mode != "test":
                y = row[vote_cols].values.astype(np.float32)

        return X, y

    def __transform(self, img):
        transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
            ]
        )
        return transforms(image=img)["image"]




## === cell 7
test_loader = None
print("Placeholder test_loader created – not used by baseline inference.")



## === cell 8
train_path = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
train_df = pd.read_csv(train_path)

vote_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

row_probs = train_df[vote_cols].div(train_df[vote_cols].sum(axis=1), axis=0)

total_votes = train_df[vote_cols].sum()
global_distribution = (total_votes / total_votes.sum()).values.astype(np.float32)

patient_means = row_probs.groupby(train_df["patient_id"]).mean()
patient_counts = train_df.groupby("patient_id").size()

ALPHA = 5.0

test_patient_ids = test_df["patient_id"].values
pred_list = []
for pid in test_patient_ids:
    if pid in patient_means.index:
        n = patient_counts.get(pid, 0)
        blended = (n * patient_means.loc[pid].values + ALPHA * global_distribution) / (
            n + ALPHA
        )
        pred = blended.astype(np.float32)
    else:
        pred = global_distribution
    pred_list.append(pred)

predictions = np.vstack(pred_list)  # shape (len(test_df), 6)

epsilon = 1e-6
predictions = np.clip(predictions, epsilon, None)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

print(
    f"Predictions shape after patient‑wise prior with updated global distribution: {predictions.shape}"
)



## === cell 9
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
sub[TARGETS] = predictions
sub[TARGETS] = sub[TARGETS].div(sub[TARGETS].sum(axis=1), axis=0)
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
sub.head()
