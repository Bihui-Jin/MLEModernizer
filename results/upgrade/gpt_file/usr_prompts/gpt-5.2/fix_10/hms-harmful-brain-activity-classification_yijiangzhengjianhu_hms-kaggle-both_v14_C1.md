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

0.4283603173019717

# 6. Current score

1.41844

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the inference loop so it correctly iterates over the generator, batches/reshapes tensors into the model’s expected NCHW format, and never shadows the target variable `y`. I also make the model loading robust to CUDA/CPU environments and “state_dict vs full module” checkpoints without changing the architecture/training logic. Then I ensure `test_pred` is always a proper `(n_samples, 6)` float array, and I enforce valid probability rows (clip + renormalize) to avoid submission failures and stabilize KL divergence. Finally, I add the missing `offset`/`spec_id` columns for test rows (required by your `DataGenerator`) and write a valid `submission.csv` with the exact required columns.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.42836), so we should improve calibration/semantics without changing the core model or feature pipeline. The biggest likely issue is that your submission is producing probabilities per *test row* (9850) but your inference loop is effectively producing probabilities per *generator step* and relying on concatenation—this is fine only if each step corresponds 1:1 to test rows and the tensor batching is correct; we make that alignment explicit and also enforce float32 NCHW tensors (ToTensorV2 yields uint8/float with HWC->CHW, but your `_to_nchw` can mis-handle already-CHW cases). Additionally, KL divergence is very sensitive to overconfident predictions, so we apply a minimal “temperature smoothing” (mix with a small uniform prior) after ensembling to reduce extreme probabilities; this usually improves KL on this competition without altering model logic. Finally, we ensure model outputs are treated consistently as logits (some checkpoints may already output probabilities), by detecting if rows already sum ~1 and are within [0,1] and skipping softmax in that case (prevents double-softmax degradation).'
- What this solution (achieved 1.40995) has done: 'Your score (1.40995, lower-is-better) is far above the target (0.42836), so we should make a small change that legitimately improves KL without changing your model or feature pipeline. The biggest low-risk lever here is probability calibration: your current uniform-mix smoothing (alpha=0.02) is likely too weak for KL (which heavily penalizes overconfident wrong classes), so I increase the smoothing slightly to reduce extremes while keeping semantics identical. I also add a tiny epsilon-floor + renormalization inside `_model_to_probs` so every per-model prediction is numerically safe before ensembling (prevents rare near-zero probabilities that can spike KL). Everything else (data loading, spectrogram creation, generator, ensembling, submission formatting) stays the same.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is much worse than the target (0.42836), so we should legitimately reduce KL with minimal changes that don’t touch the model/feature pipeline. The biggest likely calibration issue is that your test-time smoothing is fixed and quite small; for this competition KL often improves by using a slightly stronger uniform-mix prior to avoid overconfident mistakes. I make that smoothing strength configurable and set it to a more conservative value, while also applying the same epsilon-floor+renormalization to the final ensemble output (not just per-model) to prevent rare near-zero probabilities that can spike KL. Everything else (data loading, spectrogram creation, generator, model loading, inference loop, and submission formatting) stays the same.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far above the target (0.42836), so we need a legitimate improvement with minimal risk and without touching the model/feature pipeline. The biggest low-risk lever is calibration: your current uniform-mix smoothing (alpha=0.18) is likely over-smoothing and can hurt KL by washing out useful signal; we reduce it to a milder value that typically improves KL while still preventing overconfident spikes. We also apply the same epsilon-floor+renormalization consistently using a single helper, to avoid any rare near-zero probabilities that can inflate KL. Everything else (data loading, spectrogram building, generator behavior, model loading, inference loop, and submission formatting) stays the same.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far worse than the target (0.42836), so we should improve calibration with the smallest change that doesn’t alter your model/feature pipeline. The biggest likely issue is that you are feeding the model raw 0–255 tensors without the ImageNet-style normalization that these Torch image backbones almost always expect; this can severely degrade probabilities and inflate KL. I add a minimal `A.Normalize(...)` into the existing `val_transform` only (no change to architecture, loss, or feature extraction) so inference matches typical training preprocessing. I keep your existing probability safety (softmax detection, epsilon floor, renorm) and submission formatting unchanged.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far above the target (0.42836), so we should make a minimal, low-risk change that improves probability calibration without touching your model architecture or spectrogram construction. The strongest likely issue is a preprocessing mismatch: you normalize with ImageNet stats after scaling to 0–255, but these models for HMS are typically trained with a simple 0–1 scaling (or no ImageNet centering), and the mismatch can badly distort logits and inflate KL. I keep your existing pipeline and add a single switch to use a “0–1 only” normalization for inference (still using Albumentations + ToTensorV2), keeping the rest identical. I also keep your epsilon-floor+renorm and uniform-mix smoothing unchanged to maintain submission validity and KL stability.'
- What this solution (achieved 1.41844) has done: 'I fix the runtime failure by making model checkpoint discovery robust: if the expected `/kaggle/input/model101-dm-nf-0/...` path is missing, the script automatically fall back to producing a valid, competition-formatted submission using a safe prior derived from `train.csv` (and clipped/renormalized for KL safety). This keeps the core spectrogram-building and generator code intact, but avoids wasting time computing EEG spectrograms when no model exists to use them. I also ensure the output has the exact sample_submission columns and row-wise probabilities sum to 1.0, writing `submission.csv` end-to-end without errors.'

# 9. Code solution

## === cell 0
import glob
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import torch
import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 1
FEATS2 = ["Fp1", "T3", "C3", "O1", "Fp2", "C4", "T4", "O2"]

DATA_TYPE = "both"

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
CLASSES = TARGETS

test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
print("Test shape", test.shape)

test = test.rename({"spectrogram_id": "spec_id"}, axis=1)
if "offset" not in test.columns:
    test["offset"] = 0

PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
files2 = os.listdir(PATH2)
print(f"There are {len(files2)} test spectrogram parquets")

spectrograms2 = {}
for i, f in enumerate(files2):
    if i % 200 == 0:
        print(i, ", ", end="")
    tmp = pd.read_parquet(f"{PATH2}{f}")
    name = int(f.split(".")[0])
    spectrograms2[name] = tmp.iloc[:, 1:].values
print()

PATH_EEG_TEST = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
DISPLAY = 0
EEG_IDS2 = test.eeg_id.unique()
all_eegs2 = {}

print("Setup done (spectrogram parquet cache loaded).")



## === cell 2
import librosa
import pywt


def eeg_from_parquet(parquet_path):
    eeg = pd.read_parquet(parquet_path, columns=FEATS2)
    rows = len(eeg)
    offset = (rows - 10_000) // 2
    eeg = eeg.iloc[offset : offset + 10_000]
    data = np.zeros((10_000, len(FEATS2)), dtype=np.float32)
    for j, col in enumerate(FEATS2):
        x = eeg[col].values.astype("float32")
        m = np.nanmean(x)
        if np.isnan(x).mean() < 1:
            x = np.nan_to_num(x, nan=m)
        else:
            x[:] = 0
        data[:, j] = x
    return data


def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x, wavelet="haar", level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])
    ret = pywt.waverec(coeff, wavelet, mode="per")
    return ret


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

    img = np.zeros((100, 300, 4), dtype="float32")

    if display:
        plt.figure(figsize=(10, 7))
    signals = []
    for k in range(4):
        COLS = FEATS[k]
        for kk in range(4):
            x1 = eeg[COLS[kk]].values
            x2 = eeg[COLS[kk + 1]].values

            m = np.nanmean(x1)
            if np.isnan(x1).mean() < 1:
                x1 = np.nan_to_num(x1, nan=m)
            else:
                x1[:] = 0

            m = np.nanmean(x2)
            if np.isnan(x2).mean() < 1:
                x2 = np.nan_to_num(x2, nan=m)
            else:
                x2[:] = 0

            x = x1 - x2
            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)

            signals.append(x)

            mel_spec = librosa.feature.melspectrogram(
                y=x,
                sr=200,
                hop_length=len(x) // 300,
                n_fft=1024,
                n_mels=100,
                fmin=0,
                fmax=20,
                win_length=128,
            )

            width = (mel_spec.shape[1] // 30) * 30
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[
                :, :width
            ]
            img[:, :, k] += mel_spec_db

        img[:, :, k] /= 4.0

        if display:
            plt.subplot(2, 2, k + 1)
            plt.imshow(img[:, :, k], aspect="auto", origin="lower")

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
        plt.show()

    return img




## === cell 3
class DataGenerator:
    "Generates data for Keras"

    def __init__(
        self,
        data,
        specs=None,
        eeg_specs=None,
        raw_eegs=None,
        augment=False,
        mode="train",
        data_type=DATA_TYPE,
        trans=None,
    ):
        self.data = data
        self.augment = augment
        self.mode = mode
        self.data_type = data_type
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.raw_eegs = raw_eegs
        self.on_epoch_end()
        self.trans = trans

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        X, y = self.data_generation(index)
        if self.augment:
            X = self.augmentation(X)
        return X, y

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def on_epoch_end(self):
        if self.mode == "train":
            self.data = self.data.sample(frac=1).reset_index(drop=True)

    def data_generation(self, index):
        if self.data_type == "both":
            X, y = self.generate_all_specs(index)
        elif self.data_type == "eeg" or self.data_type == "kaggle":
            X, y = self.generate_specs(index)
        elif self.data_type == "raw":
            X, y = self.generate_raw(index)
        return X, y

    def generate_all_specs(self, index):
        X = np.zeros((512, 512, 3), dtype="float32")
        y = np.zeros((6,), dtype="float32")

        row = self.data.iloc[index]
        if self.mode == "test":
            offset = 0
        else:
            offset = int(row.offset / 2)

        eeg = self.eeg_specs[row.eeg_id]
        spec = self.specs[row.spec_id]

        imgs = [
            spec[offset : offset + 300, k * 100 : (k + 1) * 100].T for k in [0, 2, 1, 3]
        ]
        img = np.stack(imgs, axis=-1)
        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)
        img = np.nan_to_num(img, nan=0.0)

        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)

        X[0 + 56 : 100 + 56, :256, 0] = img[:, 22:-22, 0]
        X[100 + 56 : 200 + 56, :256, 0] = img[:, 22:-22, 2]
        X[0 + 56 : 100 + 56, :256, 1] = img[:, 22:-22, 1]
        X[100 + 56 : 200 + 56, :256, 1] = img[:, 22:-22, 3]
        X[0 + 56 : 100 + 56, :256, 2] = img[:, 22:-22, 2]
        X[100 + 56 : 200 + 56, :256, 2] = img[:, 22:-22, 1]

        X[0 + 56 : 100 + 56, 256:, 0] = img[:, 22:-22, 0]
        X[100 + 56 : 200 + 56, 256:, 0] = img[:, 22:-22, 2]
        X[0 + 56 : 100 + 56, 256:, 1] = img[:, 22:-22, 1]
        X[100 + 56 : 200 + 56, 256:, 1] = img[:, 22:-22, 3]

        img = eeg
        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)

        X[200 + 56 : 300 + 56, :256, 0] = img[:, 22:-22, 0]
        X[300 + 56 : 400 + 56, :256, 0] = img[:, 22:-22, 2]
        X[200 + 56 : 300 + 56, :256, 1] = img[:, 22:-22, 1]
        X[300 + 56 : 400 + 56, :256, 1] = img[:, 22:-22, 3]
        X[200 + 56 : 300 + 56, :256, 2] = img[:, 22:-22, 2]
        X[300 + 56 : 400 + 56, :256, 2] = img[:, 22:-22, 1]

        X[200 + 56 : 300 + 56, 256:, 0] = img[:, 22:-22, 0]
        X[300 + 56 : 400 + 56, 256:, 0] = img[:, 22:-22, 2]
        X[200 + 56 : 300 + 56, 256:, 1] = img[:, 22:-22, 1]
        X[300 + 56 : 400 + 56, 256:, 1] = img[:, 22:-22, 3]

        if self.trans is not None:
            X = self.trans(image=X)["image"]

        if self.mode != "test":
            y[:] = row[TARGETS]

        return X, y

    def generate_specs(self, index):
        X = np.zeros((512, 512, 3), dtype="float32")
        y = np.zeros((6,), dtype="float32")

        row = self.data.iloc[index]
        if self.mode == "test":
            offset = 0
        else:
            offset = int(row.offset / 2)

        if self.data_type == "eeg":
            img = self.eeg_specs[row.eeg_id]
        elif self.data_type == "kaggle":
            spec = self.specs[row.spec_id]
            imgs = [
                spec[offset : offset + 300, k * 100 : (k + 1) * 100].T
                for k in [0, 2, 1, 3]
            ]
            img = np.stack(imgs, axis=-1)
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)
            img = np.nan_to_num(img, nan=0.0)

        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)

        X[0 + 56 : 100 + 56, :256, 0] = img[:, 22:-22, 0]
        X[100 + 56 : 200 + 56, :256, 0] = img[:, 22:-22, 2]
        X[0 + 56 : 100 + 56, :256, 1] = img[:, 22:-22, 1]
        X[100 + 56 : 200 + 56, :256, 1] = img[:, 22:-22, 3]
        X[0 + 56 : 100 + 56, :256, 2] = img[:, 22:-22, 2]
        X[100 + 56 : 200 + 56, :256, 2] = img[:, 22:-22, 1]

        X[0 + 56 : 100 + 56, 256:, 0] = img[:, 22:-22, 0]
        X[100 + 56 : 200 + 56, 256:, 0] = img[:, 22:-22, 1]
        X[0 + 56 : 100 + 56, 256:, 1] = img[:, 22:-22, 2]
        X[100 + 56 : 200 + 56, 256:, 1] = img[:, 22:-22, 3]

        X[200 + 56 : 300 + 56, :256, 0] = img[:, 22:-22, 0]
        X[300 + 56 : 400 + 56, :256, 0] = img[:, 22:-22, 1]
        X[200 + 56 : 300 + 56, :256, 1] = img[:, 22:-22, 2]
        X[300 + 56 : 400 + 56, :256, 1] = img[:, 22:-22, 3]
        X[200 + 56 : 300 + 56, :256, 2] = img[:, 22:-22, 3]
        X[300 + 56 : 400 + 56, :256, 2] = img[:, 22:-22, 2]

        X[200 + 56 : 300 + 56, 256:, 0] = img[:, 22:-22, 0]
        X[300 + 56 : 400 + 56, 256:, 0] = img[:, 22:-22, 2]
        X[200 + 56 : 300 + 56, 256:, 1] = img[:, 22:-22, 1]
        X[300 + 56 : 400 + 56, 256:, 1] = img[:, 22:-22, 3]

        if self.mode != "test":
            y[:] = row[TARGETS]

        return X, y




## === cell 4
val_transform_01 = A.Compose(
    [
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2(p=1.0),
    ]
)

val_transform_imagenet = A.Compose(
    [
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        ToTensorV2(p=1.0),
    ]
)


def _ensure_nchw_float32(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    if x.ndim == 3:
        x = x.unsqueeze(0)

    if x.ndim == 4 and x.shape[-1] == 3 and x.shape[1] != 3:
        x = x.permute(0, 3, 1, 2).contiguous()

    return x.to(dtype=torch.float32)


def _eps_floor_and_renorm_np(p: np.ndarray, eps: float) -> np.ndarray:
    p = p.astype(np.float32, copy=False)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def _model_to_probs(model_out: torch.Tensor) -> torch.Tensor:
    x = model_out
    if x.ndim != 2 or x.shape[1] != 6:
        probs = torch.softmax(x, dim=1)
        eps = 1e-6
        probs = probs.clamp_min(eps)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs

    with torch.no_grad():
        row_sums = x.sum(dim=1)
        looks_prob = (
            (x.min(dim=1).values >= -1e-4).all()
            and (x.max(dim=1).values <= 1.0 + 1e-4).all()
            and ((row_sums - 1.0).abs() < 1e-3).float().mean() > 0.95
        )

    if looks_prob:
        eps = 1e-6
        probs = x.clamp_min(eps)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs
    else:
        probs = torch.softmax(x, dim=1)
        eps = 1e-6
        probs = probs.clamp_min(eps)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs


def run_inference_loop(model, test_gen, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for X, _ in tqdm.tqdm(test_gen(), total=len(test_gen.data)):
            X = _ensure_nchw_float32(X).to(device, non_blocking=True)
            out = model(X)
            probs = _model_to_probs(out)
            pred_list.append(probs.detach().cpu().numpy())
    pred_arr = np.concatenate(pred_list, axis=0).astype(np.float32)
    return pred_arr


def _confidence_score(pred: np.ndarray) -> float:
    pred = _eps_floor_and_renorm_np(pred, eps=1e-6)
    return float(np.mean(np.max(pred, axis=1)))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

candidate_globs = [
    "/kaggle/input/model101-dm-nf-0/model_101/*/*.pt",
    "/kaggle/input/model101-dm-nf-0/model_101/**/*.pt",
    "/kaggle/input/**/model_101/*/*.pt",
    "/kaggle/input/**/*.pt",
]
model_paths = []
for g in candidate_globs:
    model_paths.extend(glob.glob(g, recursive=True))
model_paths = sorted({p for p in model_paths if p.lower().endswith(".pt")})
print("Found candidate .pt files:", len(model_paths))

have_models = len(model_paths) > 0

if have_models:
    print("Loading (for transform calibration):", model_paths[0])
    obj0 = torch.load(model_paths[0], map_location=device)
    if isinstance(obj0, torch.nn.Module):
        model0 = obj0
    elif (
        isinstance(obj0, dict)
        and "model" in obj0
        and isinstance(obj0["model"], torch.nn.Module)
    ):
        model0 = obj0["model"]
    else:
        print(f"Unsupported checkpoint format at {model_paths[0]}: type={type(obj0)}")
        have_models = False

if not have_models:
    print(
        "No usable model checkpoints found. Using train-prior probabilities for submission."
    )
    train = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )
    prior = train[CLASSES].sum(axis=0).values.astype(np.float64)
    prior = prior / prior.sum()
    test_pred = np.tile(prior.astype(np.float32), (len(test), 1))
    alpha = 0.02
    uniform = np.full((len(test), 6), 1 / 6, dtype=np.float32)
    test_pred = (1.0 - alpha) * test_pred + alpha * uniform
    test_pred = _eps_floor_and_renorm_np(test_pred, eps=1e-6)
else:
    print("Converting Test EEG to Spectrograms...\n")
    for i, eeg_id in enumerate(EEG_IDS2):
        if i % 200 == 0:
            print(f"{i}/{len(EEG_IDS2)}", end=" | ")
        img = spectrogram_from_eeg(f"{PATH_EEG_TEST}{eeg_id}.parquet", i < DISPLAY)
        all_eegs2[eeg_id] = img
    print("\nDone.")

    calib_n = min(64, len(test))
    test_calib = test.iloc[:calib_n].copy().reset_index(drop=True)

    gen_01 = DataGenerator(
        test_calib,
        mode="test",
        data_type="both",
        specs=spectrograms2,
        eeg_specs=all_eegs2,
        trans=val_transform_01,
    )
    gen_im = DataGenerator(
        test_calib,
        mode="test",
        data_type="both",
        specs=spectrograms2,
        eeg_specs=all_eegs2,
        trans=val_transform_imagenet,
    )

    pred_01 = run_inference_loop(model0, gen_01, device)
    pred_im = run_inference_loop(model0, gen_im, device)

    score_01 = _confidence_score(pred_01)
    score_im = _confidence_score(pred_im)
    print(
        f"Calibration confidence (lower better): 0-1={score_01:.4f}, imagenet={score_im:.4f}"
    )

    chosen_transform = (
        val_transform_imagenet if score_im < score_01 else val_transform_01
    )
    print(
        "Chosen transform:",
        "imagenet" if chosen_transform is val_transform_imagenet else "0-1",
    )

    test_gen_both = DataGenerator(
        test,
        mode="test",
        data_type="both",
        specs=spectrograms2,
        eeg_specs=all_eegs2,
        trans=chosen_transform,
    )

    preds = []
    for model_path in model_paths:
        try:
            print("Loading:", model_path)
            obj = torch.load(model_path, map_location=device)

            if isinstance(obj, torch.nn.Module):
                model = obj
            elif (
                isinstance(obj, dict)
                and "model" in obj
                and isinstance(obj["model"], torch.nn.Module)
            ):
                model = obj["model"]
            else:
                print(
                    f"Skipping unsupported checkpoint format at {model_path}: type={type(obj)}"
                )
                continue

            pred = run_inference_loop(model, test_gen_both, device)
            preds.append(pred)
        except Exception as e:
            print(
                f"Skipping model due to load/inference error: {model_path}\n  {type(e).__name__}: {e}"
            )

    test_pred = np.mean(np.stack(preds, axis=0), axis=0) if len(preds) > 0 else None

    if (
        test_pred is None
        or not isinstance(test_pred, np.ndarray)
        or test_pred.ndim != 2
        or test_pred.shape[0] != len(test)
        or test_pred.shape[1] != 6
    ):
        print("Warning: invalid test_pred shape; using uniform probabilities.")
        test_pred = np.full((len(test), 6), 1 / 6, dtype=np.float32)

    uniform = np.full((len(test), 6), 1 / 6, dtype=np.float32)
    alpha = 0.06
    test_pred = (1.0 - alpha) * test_pred.astype(np.float32) + alpha * uniform
    test_pred = _eps_floor_and_renorm_np(test_pred, eps=1e-6)

test_pred_df = pd.DataFrame(test_pred, columns=CLASSES)
test_pred_df.insert(0, "eeg_id", test["eeg_id"].values)

sample_sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
test_pred_df = test_pred_df[sample_sub.columns]

test_pred_df.to_csv("submission.csv", index=False)
print(test_pred_df.head())
print("Wrote submission.csv with shape:", test_pred_df.shape)
print(
    "Row sums (min/mean/max):",
    test_pred_df[CLASSES].sum(axis=1).min(),
    test_pred_df[CLASSES].sum(axis=1).mean(),
    test_pred_df[CLASSES].sum(axis=1).max(),
)
