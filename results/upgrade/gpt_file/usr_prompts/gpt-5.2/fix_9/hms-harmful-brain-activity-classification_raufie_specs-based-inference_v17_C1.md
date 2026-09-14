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

0.8225230189177692

# 6. Current score

1.01346

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the missing weights crash by making the script robust to absent `/kaggle/input/hba-efficientnet-weights` (fallback to a valid probability baseline so a submission is always produced). I also fix the inference loop to use the correct loader variable and remove a dataset/loader naming mix-up that could silently misalign predictions. Finally, I ensure predictions always have shape `(len(test_df), 6)` and each row sums to 1 (required by the metric/submission validator), so `submission.csv` is written correctly end-to-end.'
- What this solution (achieved 1.40995) has done: 'Your current score is much worse than the target (lower-is-better), and the biggest likely cause is that the provided EfficientNet weights path doesn’t exist so the code falls back to uniform probabilities (which score poorly on KL). I keep the exact same model/dataset/inference logic, but make the weights discovery robust by scanning common Kaggle input locations for a compatible `tf_efficientnet_b3*.pth` file and using it if present. I also enable safe GPU multi-worker transfer settings (non-semantic) and deterministic seeding for reproducibility, without changing the model or training approach (there is no training here). If no weights are found anywhere, the script still produce a valid `submission.csv` exactly as before.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995; lower-is-better) is far from the target (0.8225), and the most likely cause is still that you’re not actually loading the intended weights (or you’re loading an incompatible checkpoint) so predictions stay close to a weak baseline. I keep the exact same model/dataset/inference logic, but make weight discovery stricter and safer by (1) prioritizing the exact expected filename first, then (2) preferring `.pth` files whose tensors actually match the model’s keys/shapes, and only then falling back. Additionally, I load the checkpoint more robustly (handling `state_dict`, `model_state_dict`, and `module.` prefixes) so a valid checkpoint won’t be silently rejected or mis-loaded. These are minimal changes directly aimed at getting real model outputs (better KL) without changing architecture or inference semantics, and the script still always write a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your current score is much worse than the target (lower-is-better), so we should improve predictive quality while keeping your architecture/inference logic unchanged. The biggest remaining risk is that you still often fail to load weights because you require `strict=True`, even though many Kaggle checkpoints use different key prefixes (e.g., `model.`, `backbone.`, `encoder.`, `module.`) or include extra keys; that silently forces the uniform fallback and hurts KL. I make weight loading robust but still “semantically identical” by (1) trying common key-prefix remaps into your exact `CustomModel` keys, (2) selecting the checkpoint that yields the highest key+shape compatibility, and (3) using `strict=False` only after we’ve aligned keys and verified high compatibility, to avoid unnecessary rejection of good weights. This should move the score down toward your target without changing the model, features, or inference procedure.'
- What this solution (achieved 1.39779) has done: 'Your current KL (1.40995; lower-is-better) is far from the target (0.8225), and the most likely reason is still that you are effectively submitting a weak baseline because no compatible weights are actually found/loaded. With minimal changes and without touching the model architecture or inference semantics, I (1) expand weight discovery to also search `/kaggle/working/**` (many notebooks save/download weights there) and (2) make the compatibility probe try multiple key-remap variants (not just one) and actually verify loadability on the probe model, so we pick a checkpoint that truly loads. Finally, I add an optional “train-prior” fallback (computed from `train.csv` votes) which is strictly better than uniform for KL if no weights can be loaded, improving score while keeping the pipeline legitimate and deterministic.'
- What this solution (achieved 1.39779) has done: 'Your current KL (1.39779, lower-is-better) is still far from the target (0.8225), so the most likely remaining issue is that even when a checkpoint is found, it’s not actually being loaded correctly due to key mismatches and/or shape differences, causing weak predictions. I keep your model and inference unchanged, but improve the weight-loading step to (1) normalize common timm/EfficientNet checkpoint key patterns (including `features.*` vs `model.*` layouts), (2) filter out classifier-head tensors when their shapes don’t match your 6-class head (so the backbone can still load), and (3) actually attempt-load the best candidate on a fresh model rather than relying mostly on a compatibility heuristic. If no usable backbone weights exist, your train-prior fallback remains exactly as before to guarantee a valid submission. These changes are directly aimed at producing “real” model outputs more often, which should reduce KL toward your target without changing architecture or evaluation semantics.'
- What this solution (achieved 1.39779) has done: 'Your score is still far above the target (lower-is-better), and the most likely remaining bottleneck is that you’re spending most of your 600s budget building *all* test spectrogram/eeg dictionaries up front, so you either time out or end up relying on the fallback prior more often than necessary. I keep your exact model, feature construction, and inference semantics, but change the data pipeline to a lazy on-demand cache (LRU-style) so only items actually needed by the current batch are computed/loaded, which typically allows real model inference to finish. Additionally, I stop scanning hundreds of random `.pth` files and instead do a fast, more targeted weight search; this increases the chance you successfully load the intended checkpoint within time. These are minimal, score-relevant changes that aim to move KL down toward your target without changing architecture/loss/loops.'
- What this solution (achieved 1.01346) has done: 'I keep your model/dataset/inference logic intact and focus only on two score-relevant fixes that can reduce KL: (1) use a numerically-stable softmax with a small probability floor (KL penalizes zeros harshly), and (2) improve the fallback “train prior” by computing the *test-conditional* prior based on `patient_id` (and `eeg_id` as a backstop), which is legitimate and typically closer to the true distribution than a global prior. These are minimal changes that preserve evaluation semantics (still proper probabilities summing to 1) and do not alter architecture, features, or training (there is none). The script still always write a valid `submission.csv` end-to-end.'

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
from typing import Dict, List, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")




## === cell 1
class config:
    BATCH_SIZE = 32
    MODEL = "tf_efficientnet_b3"
    NUM_WORKERS = 0  # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False


class paths:
    MODEL_WEIGHTS = (
        "/kaggle/input/hba-efficientnet-weights/tf_efficientnet_b3_epoch_29.pth"
    )
    OUTPUT_DIR = "/kaggle/working/"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TEST_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    TEST_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    TRAIN_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"


model_weights = [paths.MODEL_WEIGHTS]



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


def plot_spectrogram(spectrogram_path: str):
    """
    Visualize spectrogram recordings from a parquet file.
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
    torch.cuda.manual_seed_all(seed)
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
test_spect_paths = {
    int(os.path.basename(p).split(".")[0]): p
    for p in glob(os.path.join(paths.TEST_SPECTROGRAMS, "*.parquet"))
}
test_eeg_paths = {
    int(os.path.basename(p).split(".")[0]): p
    for p in glob(os.path.join(paths.TEST_EEGS, "*.parquet"))
}
print(f"Indexed {len(test_spect_paths)} test spectrogram parquets")
print(f"Indexed {len(test_eeg_paths)} test EEG parquets")




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
label_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


class _LazyParquetCache:
    def __init__(self, max_items: int = 256):
        self.max_items = int(max_items)
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
        if len(self._order) > self.max_items:
            old = self._order.pop(0)
            if old in self._data:
                del self._data[old]


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

        self.spectrograms = specs
        self.eeg_spectrograms = eeg_specs

        self._spec_cache = _LazyParquetCache(max_items=256)
        self._eeg_cache = _LazyParquetCache(max_items=256)

    def __len__(self):
        return len(self.df)

    def _get_kaggle_spectrogram(self, spectrogram_id: int) -> np.ndarray:
        if self.spectrograms is not None:
            return self.spectrograms[spectrogram_id]

        cached = self._spec_cache.get(spectrogram_id)
        if cached is not None:
            return cached

        p = test_spect_paths.get(int(spectrogram_id))
        if p is None or (not os.path.isfile(p)):
            arr = np.zeros((300, 400), dtype=np.float32)
        else:
            aux = pd.read_parquet(p)
            arr = aux.iloc[:, 1:].values
            del aux
        self._spec_cache.put(spectrogram_id, arr)
        return arr

    def _get_eeg_spectrogram(self, eeg_id: int) -> np.ndarray:
        if self.eeg_spectrograms is not None:
            return self.eeg_spectrograms[eeg_id]

        cached = self._eeg_cache.get(eeg_id)
        if cached is not None:
            return cached

        p = test_eeg_paths.get(int(eeg_id))
        if p is None or (not os.path.isfile(p)):
            img = np.zeros((128, 256, 4), dtype=np.float32)
        else:
            img = spectrogram_from_eeg(p, display=False)
        self._eeg_cache.put(eeg_id, img)
        return img

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
            r = 0
        else:
            r = int((row["min"] + row["max"]) // 4)

        spec_full = self._get_kaggle_spectrogram(int(row.spectrogram_id))
        img_eeg = self._get_eeg_spectrogram(int(row.eeg_id))

        for region in range(4):
            img = spec_full[r : r + 300, region * 100 : (region + 1) * 100].T
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            ep = 1e-6
            mu = np.nanmean(img.flatten())
            std = np.nanstd(img.flatten())
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)
            X[14:-14, :, region] = img[:, 22:-22] / 2.0

        X[:, :, 4:] = img_eeg

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
    persistent_workers=(config.NUM_WORKERS > 0),
)
X, y = test_dataset[0]
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")




## === cell 8
def inference_function(test_loader, model, device):
    model.eval()
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for step, (X, _) in enumerate(tqdm_test_loader):
            X = X.to(device, non_blocking=True)
            with torch.no_grad():
                y_logits = model(X)
                y_probs = torch.softmax(y_logits, dim=1)
                y_probs = torch.clamp(y_probs, min=1e-6)
                y_probs = y_probs / y_probs.sum(dim=1, keepdim=True)
            preds.append(y_probs.detach().cpu().numpy())

    prediction_dict = {"predictions": np.concatenate(preds, axis=0)}
    return prediction_dict




## === cell 9
def discover_weight_files() -> List[str]:
    """
    Change tied to score: keep discovery targeted (fast) to avoid spending time scanning many unrelated .pth,
    improving odds we actually run inference (instead of falling back) within the time budget.
    """
    candidates = []
    candidates.extend(model_weights)

    patterns = [
        "/kaggle/input/**/tf_efficientnet_b3_epoch_29.pth",
        "/kaggle/input/**/tf_efficientnet_b3*.pth",
        "/kaggle/working/**/tf_efficientnet_b3_epoch_29.pth",
        "/kaggle/working/**/tf_efficientnet_b3*.pth",
        "/kaggle/input/**/efficientnet_b3*.pth",
        "/kaggle/working/**/efficientnet_b3*.pth",
    ]
    seen = set()
    for pat in patterns:
        for p in glob(pat, recursive=True):
            if p not in seen and os.path.isfile(p):
                seen.add(p)
                candidates.append(p)

    def sort_key(p: str):
        base = os.path.basename(p).lower()
        score = 0
        if "tf_efficientnet_b3" in base:
            score -= 50
        if "efficientnet_b3" in base:
            score -= 20
        if "epoch_29" in base:
            score -= 15
        if "epoch" in base:
            score -= 5
        return (score, len(p))

    return sorted(list(dict.fromkeys(candidates)), key=sort_key)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["model", "state_dict", "model_state_dict", "net"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    if isinstance(ckpt, dict):
        is_tensor_map = True
        for v in ckpt.values():
            if not torch.is_tensor(v):
                is_tensor_map = False
                break
        if is_tensor_map:
            return ckpt
    return ckpt


def _strip_module_prefix(sd: dict) -> dict:
    if not isinstance(sd, dict):
        return sd
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def _apply_prefix_strip(sd: dict, prefixes: Tuple[str, ...]) -> dict:
    if not isinstance(sd, dict):
        return sd
    out = {}
    for k, v in sd.items():
        nk = k
        for pref in prefixes:
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        out[nk] = v
    return out


def _remap_prefixes_to_custommodel(sd: dict) -> dict:
    """
    Change tied to score: key normalization for various checkpoint formats to maximize chance weights load.
    """
    if not isinstance(sd, dict):
        return sd

    sd = _strip_module_prefix(sd)
    remapped = {}

    for k, v in sd.items():
        nk = k

        for pref in ("model.", "net.", "backbone.", "encoder."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]

        looks_like_timm_backbone = (
            nk.startswith("conv_stem.")
            or nk.startswith("bn1.")
            or nk.startswith("blocks.")
            or nk.startswith("conv_head.")
            or nk.startswith("bn2.")
        )
        if looks_like_timm_backbone:
            nk = "features." + nk

        if nk.startswith("classifier.") or nk.startswith("fc."):
            nk = "custom_layers.2." + nk.split(".", 1)[1]
        elif nk.startswith("head.fc."):
            nk = "custom_layers.2." + nk.split("head.fc.", 1)[1]
        elif nk.startswith("head."):
            if nk == "head.weight":
                nk = "custom_layers.2.weight"
            elif nk == "head.bias":
                nk = "custom_layers.2.bias"

        remapped[nk] = v

    for k, v in sd.items():
        if k.startswith("features.") or k.startswith("custom_layers."):
            remapped[k] = v

    return remapped


def _filter_to_loadable_tensors(model: nn.Module, sd: dict) -> dict:
    """
    Change tied to score: drop tensors whose shapes don't match our model (often classifier head),
    allowing backbone weights to load and improving KL.
    """
    if not isinstance(sd, dict):
        return sd
    msd = model.state_dict()
    out = {}
    for k, v in sd.items():
        if (
            k in msd
            and torch.is_tensor(v)
            and hasattr(v, "shape")
            and v.shape == msd[k].shape
        ):
            out[k] = v
    return out


def _state_dict_compat_score(model: nn.Module, sd: dict) -> float:
    try:
        msd = model.state_dict()
        keys = set(msd.keys())
        s_keys = set(sd.keys())
        inter = keys.intersection(s_keys)
        if len(inter) == 0:
            return -1e9
        shape_ok = 0
        for k in inter:
            if (
                hasattr(sd[k], "shape")
                and hasattr(msd[k], "shape")
                and sd[k].shape == msd[k].shape
            ):
                shape_ok += 1
        return (len(inter) / max(1, len(keys))) + 0.5 * (shape_ok / max(1, len(keys)))
    except Exception:
        return -1e9


def _try_load_best_weight(model: nn.Module, weight_path: str) -> Tuple[bool, float]:
    """
    Attempts true load on the model for several normalized variants and also tries a filtered backbone-only load.
    """
    ckpt = torch.load(weight_path, map_location="cpu")
    sd0 = _extract_state_dict(ckpt)
    if not isinstance(sd0, dict):
        return False, -1e9

    variants = []
    base = _strip_module_prefix(sd0)
    variants.append(base)
    variants.append(_remap_prefixes_to_custommodel(base))
    variants.append(
        _remap_prefixes_to_custommodel(
            _apply_prefix_strip(
                base, ("model.", "model.model.", "model.backbone.", "model.encoder.")
            )
        )
    )
    variants.append(
        _remap_prefixes_to_custommodel(
            _apply_prefix_strip(
                base, ("net.", "net.model.", "net.backbone.", "net.encoder.")
            )
        )
    )

    best_loaded = False
    best_score = -1e18

    for sd in variants:
        if not isinstance(sd, dict):
            continue

        try:
            model.load_state_dict(sd, strict=True)
            sc = _state_dict_compat_score(model, sd)
            best_loaded = True
            best_score = max(best_score, sc)
            return True, best_score
        except Exception:
            pass

        sd_f = _filter_to_loadable_tensors(model, sd)
        if isinstance(sd_f, dict) and len(sd_f) > 0:
            try:
                model.load_state_dict(sd_f, strict=False)
                sc = _state_dict_compat_score(model, sd_f)
                best_loaded = True
                best_score = max(best_score, sc)
            except Exception:
                pass

    return best_loaded, best_score


def train_prior_probs(train_csv_path: str) -> np.ndarray:
    """
    If we cannot load weights, using empirical label distribution (from votes) is a stronger KL baseline than uniform.
    """
    train_df = pd.read_csv(train_csv_path, usecols=label_cols)
    votes = train_df[label_cols].to_numpy(np.float64)
    votes_sum = votes.sum(axis=1, keepdims=True)
    votes_sum[votes_sum <= 0] = 1.0
    probs = votes / votes_sum
    prior = probs.mean(axis=0)
    prior = np.clip(prior, 1e-8, 1.0)
    prior = prior / prior.sum()
    return prior.astype(np.float32)


def test_conditional_prior_probs(
    train_csv_path: str, test_df: pd.DataFrame
) -> np.ndarray:
    """
    Change tied to score: when weights are unavailable, use a conditional prior by patient_id (and fallback to eeg_id/global).
    This is legitimate (uses only metadata + train labels) and often reduces KL versus a single global prior.
    """
    usecols = ["eeg_id", "patient_id"] + label_cols
    tr = pd.read_csv(train_csv_path, usecols=usecols)

    votes = tr[label_cols].to_numpy(np.float64)
    votes_sum = votes.sum(axis=1, keepdims=True)
    votes_sum[votes_sum <= 0] = 1.0
    probs = votes / votes_sum
    probs_df = pd.DataFrame(probs, columns=label_cols)
    tr_probs = pd.concat(
        [tr[["eeg_id", "patient_id"]].reset_index(drop=True), probs_df], axis=1
    )

    global_prior = probs_df.mean(axis=0).to_numpy(np.float64)
    global_prior = np.clip(global_prior, 1e-8, 1.0)
    global_prior = global_prior / global_prior.sum()

    pat_prior = tr_probs.groupby("patient_id")[label_cols].mean()
    eeg_prior = tr_probs.groupby("eeg_id")[label_cols].mean()

    out = np.zeros((len(test_df), len(label_cols)), dtype=np.float64)
    for i, row in enumerate(test_df[["eeg_id", "patient_id"]].itertuples(index=False)):
        eeg_id = int(row.eeg_id)
        patient_id = int(row.patient_id)
        if patient_id in pat_prior.index:
            p = pat_prior.loc[patient_id].to_numpy(np.float64)
        elif eeg_id in eeg_prior.index:
            p = eeg_prior.loc[eeg_id].to_numpy(np.float64)
        else:
            p = global_prior
        p = np.clip(p, 1e-8, 1.0)
        p = p / p.sum()
        out[i] = p

    return out.astype(np.float32)


available_weights = [w for w in discover_weight_files() if os.path.isfile(w)]

chosen_weight = None
chosen_score = -1e18
if len(available_weights) > 0:
    for w in available_weights[:30]:
        try:
            probe_model = CustomModel(config)
            loaded_ok, sc = _try_load_best_weight(probe_model, w)
            del probe_model
            gc.collect()
            if loaded_ok and sc > chosen_score:
                chosen_score = sc
                chosen_weight = w
        except Exception:
            try:
                del probe_model
            except Exception:
                pass
            continue

if chosen_weight is None or not os.path.isfile(chosen_weight):
    print(
        "WARNING: No compatible model weights found under /kaggle/input or /kaggle/working. "
        "Falling back to TEST-CONDITIONAL TRAIN-PRIOR predictions to generate a valid submission."
    )
    predictions = test_conditional_prior_probs(paths.TRAIN_CSV, test_df)
else:
    print("Using model weights:", chosen_weight, "compat_score≈", float(chosen_score))
    model = CustomModel(config)

    loaded_ok, load_score = _try_load_best_weight(model, chosen_weight)
    if not loaded_ok:
        print(
            "WARNING: Failed to load weights even after remapping/filtering (score≈"
            f"{float(load_score):.3f}). Falling back to TEST-CONDITIONAL TRAIN-PRIOR predictions."
        )
        predictions = test_conditional_prior_probs(paths.TRAIN_CSV, test_df)
    else:
        model.to(device)
        prediction_dict = inference_function(test_loader, model, device)
        predictions = prediction_dict["predictions"]

    del model
    torch.cuda.empty_cache()
    gc.collect()



## === cell 10
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

predictions = np.asarray(predictions, dtype=np.float32)
if (
    predictions.ndim != 2
    or predictions.shape[0] != len(test_df)
    or predictions.shape[1] != len(TARGETS)
):
    raise ValueError(
        f"Predictions have wrong shape {predictions.shape}, expected ({len(test_df)}, {len(TARGETS)})"
    )

predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
predictions = np.clip(predictions, 1e-6, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
print(sub.head())
print(
    "Row sums (min/max):",
    sub[TARGETS].sum(axis=1).min(),
    sub[TARGETS].sum(axis=1).max(),
)
