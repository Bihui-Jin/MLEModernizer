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

0.4777374467276686

# 6. Current score

0.8425

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the pipeline so it always produces a valid `submission.csv` even when the external model folder is not available in your environment. The only runtime-blocking issue is the hard-coded `/kaggle/input/model101-mixnet-xl-0-05` path, so I replace it with a safe checkpoint discovery over `/kaggle/input/**` and add a deterministic fallback that outputs a valid probability distribution (based on the train vote priors) when no `.pt` weights are found. I also fix a latent bug in `denoise()` (missing return) to prevent unexpected `None` propagation if wavelet denoising is enabled later. These changes are correctness/stability-focused and do not alter the core modeling/inference logic when checkpoints are present.'
- What this solution (achieved 1.47425) has done: 'Your current score (1.419) is far worse than the target (0.478, lower is better), and the most likely reason is that you are usually falling back to the global vote prior because no compatible `.pt` weights are being found/loaded, yielding weak predictions. I make the smallest change that legitimately improves score: (1) prefer loading a known-good public baseline if it exists in the dataset (`*.pth`/`*.ckpt` as well as `*.pt`), and (2) if no weights are present, use a slightly stronger “patient-specific prior” fallback (computed from train vote priors conditioned on `patient_id`) instead of a single global prior. This keeps your core inference logic identical when models are available, and only improves the deterministic fallback behavior to better match the test distribution. The submission writing and probability normalization remain unchanged.'
- What this solution (achieved 0.8425) has done: 'Your score is much worse than the target (lower is better), and with this code that usually indicates either (a) you’re ensembling a lot of unrelated checkpoints found under `/kaggle/input/**` (most not be compatible and/or produce poor predictions) or (b) you’re falling back to priors that are not well-calibrated to the KL metric. I keep your core feature creation and inference loop intact, but restrict checkpoint discovery to the intended competition dataset folders and only accept checkpoints that both load and produce a valid `(1, 6)` output on a quick single-batch sanity check. If no valid model is found, I keep the patient-conditioned prior fallback but make it stronger (and still fully legitimate) by computing *patient + spectrogram* conditioned priors when available, backing off to patient-only, then global. Finally, I apply a tiny Dirichlet-style smoothing to the fallback priors to avoid near-zeros which are heavily penalized by KL.'
- What this solution (achieved 0.8425) has done: 'Your current KL (0.8425, lower is better) is still far above the target (0.4777), and the most likely cause is that you’re often using the conditioned-prior fallback (no usable checkpoints), which is underpowered for this metric. I keep your model/inference logic unchanged when checkpoints exist, but make the fallback legitimately stronger by aggregating train labels at the same granularity as the submission (`eeg_id`) and conditioning on (`patient_id`, `eeg_id`) then (`patient_id`) then global, which better matches how targets are evaluated. I also fix a key mismatch bug in the fallback (you currently condition on `spec_id` but build priors with `spectrogram_id` from train, so it almost never hits the strongest prior). Finally, I keep the same smoothing/normalization so the submission remains valid and numerically safe for KL.'
- What this solution (achieved 0.8425) has done: 'Your current score (0.8425, lower is better) is still far above the target (0.4777), and with this pipeline the biggest likely issue is that you are averaging multiple “compatible-shape” checkpoints that are not actually trained for this exact input construction, which can easily hurt KL badly. I keep your feature creation and inference loop intact, but tighten checkpoint selection so we only ensemble checkpoints that (a) load, (b) produce the right shape, and (c) produce *non-degenerate, numerically sane* probabilities on a small probe batch; if none pass, we keep your existing conditioned-prior fallback. I also change the ensemble reducer from mean-of-probabilities to mean-of-logits (then softmax), which is a minimal change that often improves KL calibration without altering the model or data. Finally, I keep strict probability clipping/renormalization and still write `submission.csv` with the required columns and row sums of 1.'

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

test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
print("Test shape", test.shape)
print(test.head())

PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
files2 = os.listdir(PATH2)
print(f"There are {len(files2)} test spectrogram parquets")

spectrograms2 = {}
for i, f in enumerate(files2):
    if i % 100 == 0:
        print(i, ", ", end="")
    tmp = pd.read_parquet(f"{PATH2}{f}")
    name = int(f.split(".")[0])
    spectrograms2[name] = tmp.iloc[:, 1:].values

print("\nLoaded test spectrograms:", len(spectrograms2))

test = test.rename({"spectrogram_id": "spec_id"}, axis=1)

PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
DISPLAY = 0
EEG_IDS2 = test.eeg_id.unique()
all_eegs2 = {}

print("Converting Test EEG to Spectrograms...")
print()




## === cell 2
def eeg_from_parquet(parquet_path):
    eeg = pd.read_parquet(parquet_path, columns=FEATS2)
    rows = len(eeg)
    offset = (rows - 10_000) // 2
    eeg = eeg.iloc[offset : offset + 10_000]
    data = np.zeros((10_000, len(FEATS2)))
    for j, col in enumerate(FEATS2):
        x = eeg[col].values.astype("float32")
        m = np.nanmean(x)
        if np.isnan(x).mean() < 1:
            x = np.nan_to_num(x, nan=m)
        else:
            x[:] = 0
        data[:, j] = x
    return data


import librosa
import pywt, librosa


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


for i, eeg_id in enumerate(EEG_IDS2):
    img = spectrogram_from_eeg(f"{PATH2}{eeg_id}.parquet", i < DISPLAY)
    all_eegs2[eeg_id] = img

print("Loaded test EEG-derived spectrograms:", len(all_eegs2))




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
        ]  # to match kaggle with eeg
        img = np.stack(imgs, axis=-1)
        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)

        img = np.nan_to_num(img, nan=0.0)

        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)

        X[0 + 56 : 100 + 56, :256, 0] = img[:, 22:-22, 0]  # LL_k
        X[100 + 56 : 200 + 56, :256, 0] = img[:, 22:-22, 2]  # RL_k
        X[0 + 56 : 100 + 56, :256, 1] = img[:, 22:-22, 1]  # LP_k
        X[100 + 56 : 200 + 56, :256, 1] = img[:, 22:-22, 3]  # RP_k
        X[0 + 56 : 100 + 56, :256, 2] = img[:, 22:-22, 2]  # RL_k
        X[100 + 56 : 200 + 56, :256, 2] = img[:, 22:-22, 1]  # LP_k

        X[0 + 56 : 100 + 56, 256:, 0] = img[:, 22:-22, 0]  # LL_k
        X[100 + 56 : 200 + 56, 256:, 0] = img[:, 22:-22, 2]  # RL_k
        X[0 + 56 : 100 + 56, 256:, 1] = img[:, 22:-22, 1]  # LP_k
        X[100 + 56 : 200 + 56, 256:, 1] = img[:, 22:-22, 3]  # RP_K

        img = eeg
        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)
        X[200 + 56 : 300 + 56, :256, 0] = img[:, 22:-22, 0]  # LL_e
        X[300 + 56 : 400 + 56, :256, 0] = img[:, 22:-22, 2]  # RL_e
        X[200 + 56 : 300 + 56, :256, 1] = img[:, 22:-22, 1]  # LP_e
        X[300 + 56 : 400 + 56, :256, 1] = img[:, 22:-22, 3]  # RP_e
        X[200 + 56 : 300 + 56, :256, 2] = img[:, 22:-22, 2]  # RL_e
        X[300 + 56 : 400 + 56, :256, 2] = img[:, 22:-22, 1]  # LP_e

        X[200 + 56 : 300 + 56, 256:, 0] = img[:, 22:-22, 0]  # LL_e
        X[300 + 56 : 400 + 56, 256:, 0] = img[:, 22:-22, 2]  # RL_e
        X[200 + 56 : 300 + 56, 256:, 1] = img[:, 22:-22, 1]  # LP_e
        X[300 + 56 : 400 + 56, 256:, 1] = img[:, 22:-22, 3]  # RP_e

        X = (
            self.trans(image=X)["image"]
            if self.trans is not None
            else torch.from_numpy(X).permute(2, 0, 1)
        )

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
            ]  # to match kaggle with eeg
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
val_transform = A.Compose([ToTensorV2(p=1.0)])




## === cell 5
def _load_model_safely(model_path: str, device: torch.device):
    obj = torch.load(model_path, map_location="cpu")
    if isinstance(obj, torch.nn.Module):
        model = obj
    elif (
        isinstance(obj, dict)
        and "model" in obj
        and isinstance(obj["model"], torch.nn.Module)
    ):
        model = obj["model"]
    else:
        raise TypeError(f"Unsupported checkpoint format at {model_path}: {type(obj)}")
    model.to(device)
    model.eval()
    return model


def run_inference_loop_return_logits(model, test_gen, device):
    model.to(device)
    model.eval()
    logits = np.zeros((len(test_gen), 6), dtype=np.float32)

    with torch.no_grad():
        for i, (x, _) in enumerate(tqdm.tqdm(test_gen(), total=len(test_gen))):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x = x.unsqueeze(0).to(device, non_blocking=True)
            out = model(x)
            if out.ndim != 2 or out.shape[1] != 6:
                raise ValueError(f"Unexpected model output shape: {tuple(out.shape)}")
            logits[i] = out.detach().cpu().numpy().astype(np.float32)[0]

    return logits


def _find_model_paths():
    roots = [
        "/kaggle/input/hms-harmful-brain-activity-classification",
        "/kaggle/input/hms-harmful-brain-activity-classification/**",
    ]
    exts = ["pt", "pth", "ckpt"]
    paths = []
    for r in roots:
        for e in exts:
            paths.extend(glob.glob(f"{r}/**/*.{e}", recursive=True))
    return sorted(set(paths))


def _vote_prior_from_train(train_csv_path: str, targets):
    tr = pd.read_csv(train_csv_path, usecols=targets)
    prior = tr[targets].sum(axis=0).values.astype(np.float64)
    prior = prior / prior.sum()
    return prior.astype(np.float32)


def _conditioned_priors_from_train_eeg_level(train_csv_path: str, targets):
    usecols = ["eeg_id", "patient_id"] + list(targets)
    tr = pd.read_csv(train_csv_path, usecols=usecols)

    grp_pe = tr.groupby(["patient_id", "eeg_id"])[targets].sum()
    grp_pe = grp_pe.div(grp_pe.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    grp_p = tr.groupby("patient_id")[targets].sum()
    grp_p = grp_p.div(grp_p.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    return grp_pe.astype(np.float32), grp_p.astype(np.float32)


def _is_valid_checkpoint_for_task(model_path: str, device: torch.device, test_gen):
    try:
        model = _load_model_safely(model_path, device)
        x0, _ = next(iter(test_gen()))
        if isinstance(x0, np.ndarray):
            x0 = torch.from_numpy(x0)
        x0 = x0.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x0)
        if not isinstance(out, torch.Tensor):
            return False
        if tuple(out.shape) != (1, 6):
            return False

        prob = out.softmax(dim=1).detach().cpu().numpy()[0]
        if not np.isfinite(prob).all():
            return False
        if prob.min() < 0 or prob.max() > 1:
            return False
        entropy = -float(np.sum(prob * np.log(np.clip(prob, 1e-12, 1.0))))
        if entropy < 0.05:
            return False

        return True
    except Exception as e:
        print(
            f"Skipping checkpoint (failed load/shape/prob sanity): {model_path} | {type(e).__name__}: {e}"
        )
        return False


test_gen_both = DataGenerator(
    test,
    mode="test",
    data_type="both",
    specs=spectrograms2,
    eeg_specs=all_eegs2,
    trans=val_transform,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_paths = _find_model_paths()
print(
    f"Discovered {len(model_paths)} candidate checkpoint files under competition input dirs"
)

valid_model_paths = []
for mp in model_paths:
    if _is_valid_checkpoint_for_task(mp, device, test_gen_both):
        valid_model_paths.append(mp)

print(
    f"Keeping {len(valid_model_paths)} checkpoints after compatibility+prob sanity-check"
)

if len(valid_model_paths) > 0:
    logits_list = []
    for model_path in valid_model_paths:
        print("Loading:", model_path)
        model = _load_model_safely(model_path, device)
        lg = run_inference_loop_return_logits(model, test_gen_both, device)
        logits_list.append(lg)

    avg_logits = np.mean(np.stack(logits_list, axis=0), axis=0).astype(np.float32)
    avg_logits = avg_logits - avg_logits.max(axis=1, keepdims=True)
    expv = np.exp(avg_logits).astype(np.float32)
    test_pred = expv / expv.sum(axis=1, keepdims=True)
else:
    train_path = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    global_prior = _vote_prior_from_train(train_path, TARGETS)

    pri_pe, pri_p = _conditioned_priors_from_train_eeg_level(train_path, TARGETS)

    test_pred = np.zeros((len(test), 6), dtype=np.float32)

    alpha = 1e-3  # keep same small smoothing to avoid KL blow-ups from near-zeros

    for i, (pid, eid) in enumerate(
        zip(test["patient_id"].values, test["eeg_id"].values)
    ):
        key = (pid, eid)
        if key in pri_pe.index:
            p = pri_pe.loc[key].values
        elif pid in pri_p.index:
            p = pri_p.loc[pid].values
        else:
            p = global_prior

        p = (p + alpha).astype(np.float32)
        p = p / p.sum()
        test_pred[i] = p

    print(
        "No valid checkpoints found; using eeg-aligned conditioned vote prior fallback."
    )
    print("Global prior:", global_prior)

test_pred = np.clip(test_pred, 1e-12, 1.0)
test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)

CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
test_pred_df = pd.DataFrame(test_pred, columns=CLASSES)

test_pred_df = pd.concat(
    [test[["eeg_id"]].reset_index(drop=True), test_pred_df], axis=1
)
test_pred_df.to_csv("submission.csv", index=False)
print(test_pred_df.head())
print("Wrote submission.csv with shape:", test_pred_df.shape)
print(
    "Row prob sum stats:",
    test_pred_df[CLASSES].sum(axis=1).min(),
    test_pred_df[CLASSES].sum(axis=1).max(),
)
