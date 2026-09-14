# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.4130432990508637

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the inference aggregation so `test_pred` is always a proper 2D NumPy array before building the DataFrame (the current failure happens when no model files are found, leaving `preds` empty and `np.mean` returning a scalar). I also add a safe fallback that creates a valid, normalized submission (uniform probabilities) if the model directory is missing/empty, so you always get a CSV and Kaggle accepts it. Additionally, I remove the per-sample `print(batch_data.shape)` which can massively slow execution and risk timeouts, without changing predictions. Finally, I ensure probabilities are correctly normalized to sum to 1 per row to satisfy the submission constraint.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.4130), so we need a real (but minimal) correctness/performance fix rather than stability tweaks. The biggest issue is that `test_gen_both` is a one-pass generator (`DataGenerator.__call__`) and you reuse it across multiple model checkpoints; after the first model, subsequent models see zero batches, so your “ensemble” is effectively just the first model. I change the inference loop to iterate by index (so it’s repeatable for every model) and also ensure the loaded model is put in eval mode and run on the correct device consistently. This preserves your core feature pipeline and model semantics, but should substantially improve KL by making the ensemble actually average across all checkpoints.'

# 9. Code solution

## === cell 0
import glob
import os
import numpy as np
import pandas as pd

import torch
import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

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

INPUT_ROOT = "/kaggle/input/hms-harmful-brain-activity-classification"

test = pd.read_csv(f"{INPUT_ROOT}/test.csv")
print("Test shape", test.shape)

sub_template = pd.read_csv(f"{INPUT_ROOT}/sample_submission.csv")
test = test.set_index("eeg_id").loc[sub_template["eeg_id"]].reset_index()
print("Aligned test shape to sample_submission:", test.shape)

PATH_SPEC = f"{INPUT_ROOT}/test_spectrograms/"
files2 = os.listdir(PATH_SPEC)
print(f"There are {len(files2)} test spectrogram parquets")

spectrograms2 = {}
for i, f in enumerate(files2):
    if i % 200 == 0:
        print(i, ", ", end="")
    tmp = pd.read_parquet(f"{PATH_SPEC}{f}")
    name = int(f.split(".")[0])
    spectrograms2[name] = tmp.iloc[:, 1:].to_numpy(dtype=np.float32, copy=False)
print()

test = test.rename({"spectrogram_id": "spec_id"}, axis=1)

PATH_EEG = f"{INPUT_ROOT}/test_eegs/"
DISPLAY = 0
EEG_IDS2 = test.eeg_id.unique()
all_eegs2 = {}

print("Converting Test EEG to Spectrograms...")
print()



## === cell 1
import matplotlib.pyplot as plt
import librosa
import pywt
from concurrent.futures import ThreadPoolExecutor, as_completed


def eeg_from_parquet(parquet_path):
    eeg = pd.read_parquet(parquet_path, columns=FEATS2)
    rows = len(eeg)
    offset = (rows - 10_000) // 2
    eeg = eeg.iloc[offset : offset + 10_000]

    arr = eeg.to_numpy(dtype=np.float32, copy=False)  # (10000, 8)
    col_mean = np.nanmean(arr, axis=0)
    all_nan = np.isnan(col_mean)
    col_mean[all_nan] = 0.0
    inds = np.isnan(arr)
    if inds.any():
        arr[inds] = np.take(col_mean, np.where(inds)[1])
    return arr


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
    needed_cols = sorted(set([c for group in FEATS for c in group]))
    eeg = pd.read_parquet(parquet_path, columns=needed_cols)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle : middle + 10_000]

    img = np.zeros((100, 300, 4), dtype="float32")

    if display:
        plt.figure(figsize=(10, 7))
    signals = []

    eeg_np = {c: eeg[c].to_numpy(copy=False) for c in needed_cols}

    hop = 10_000 // 300
    for k in range(4):
        COLS = FEATS[k]
        for kk in range(4):
            x1 = eeg_np[COLS[kk]].astype(np.float32, copy=False)
            x2 = eeg_np[COLS[kk + 1]].astype(np.float32, copy=False)

            m1 = np.nanmean(x1)
            if np.isnan(x1).mean() < 1:
                if np.isnan(x1).any():
                    x1 = np.nan_to_num(x1, nan=m1)
            else:
                x1 = np.zeros_like(x1, dtype=np.float32)

            m2 = np.nanmean(x2)
            if np.isnan(x2).mean() < 1:
                if np.isnan(x2).any():
                    x2 = np.nan_to_num(x2, nan=m2)
            else:
                x2 = np.zeros_like(x2, dtype=np.float32)

            x = (x1 - x2).astype(np.float32, copy=False)

            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            mel_spec = librosa.feature.melspectrogram(
                y=x,
                sr=200,
                hop_length=hop,
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


def _build_one_eeg(eeg_id, display=False):
    return eeg_id, spectrogram_from_eeg(f"{PATH_EEG}{eeg_id}.parquet", display)


max_workers = min(8, (os.cpu_count() or 4))
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    futures = []
    for i, eeg_id in enumerate(EEG_IDS2):
        futures.append(ex.submit(_build_one_eeg, eeg_id, i < DISPLAY))
    for fut in tqdm.tqdm(as_completed(futures), total=len(futures)):
        eeg_id, img = fut.result()
        all_eegs2[eeg_id] = img




## === cell 2
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
        self.data = data.reset_index(drop=True)
        self.augment = augment
        self.mode = mode
        self.data_type = data_type
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.raw_eegs = raw_eegs
        self.trans = trans
        if self.mode == "train":
            self.on_epoch_end()

    def __len__(self):
        return int(len(self.data))

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
        offset = 0 if self.mode == "test" else int(row.offset / 2)

        eeg = self.eeg_specs[row.eeg_id]
        spec = self.specs[row.spec_id]

        imgs = [
            spec[offset : offset + 300, k * 100 : (k + 1) * 100].T for k in [0, 2, 1, 3]
        ]
        img = np.stack(imgs, axis=-1)
        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)
        img = np.nan_to_num(img, nan=0.0)

        mn = np.min(img)
        mx = np.max(img)
        ep = 1e-5
        img = 255.0 * (img - mn) / (mx - mn + ep)

        sl = slice(22, -22)
        X[56:156, :256, 0] = img[:, sl, 0]  # LL_k
        X[156:256, :256, 0] = img[:, sl, 2]  # RL_k
        X[56:156, :256, 1] = img[:, sl, 1]  # LP_k
        X[156:256, :256, 1] = img[:, sl, 3]  # RP_k
        X[56:156, :256, 2] = img[:, sl, 2]  # RL_k
        X[156:256, :256, 2] = img[:, sl, 1]  # LP_k

        X[56:156, 256:, 0] = img[:, sl, 0]  # LL_k
        X[156:256, 256:, 0] = img[:, sl, 2]  # RL_k
        X[56:156, 256:, 1] = img[:, sl, 1]  # LP_k
        X[156:256, 256:, 1] = img[:, sl, 3]  # RP_K

        img = eeg
        mn = np.min(img)
        mx = np.max(img)
        img = 255.0 * (img - mn) / (mx - mn + ep)

        X[256:356, :256, 0] = img[:, sl, 0]  # LL_e
        X[356:456, :256, 0] = img[:, sl, 2]  # RL_e
        X[256:356, :256, 1] = img[:, sl, 1]  # LP_e
        X[356:456, :256, 1] = img[:, sl, 3]  # RP_e
        X[256:356, :256, 2] = img[:, sl, 2]  # RL_e
        X[356:456, :256, 2] = img[:, sl, 1]  # LP_e

        X[256:356, 256:, 0] = img[:, sl, 0]  # LL_e
        X[356:456, 256:, 0] = img[:, sl, 2]  # RL_e
        X[256:356, 256:, 1] = img[:, sl, 1]  # LP_e
        X[356:456, 256:, 1] = img[:, sl, 3]  # RP_e

        if self.trans is not None:
            X = self.trans(image=X)["image"]
        if self.mode != "test":
            y[:] = row[TARGETS]
        return X, y

    def generate_specs(self, index):
        X = np.zeros((512, 512, 3), dtype="float32")
        y = np.zeros((6,), dtype="float32")

        row = self.data.iloc[index]
        offset = 0 if self.mode == "test" else int(row.offset / 2)

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

        mn = np.min(img)
        mx = np.max(img)
        ep = 1e-5
        img = 255.0 * (img - mn) / (mx - mn + ep)

        sl = slice(22, -22)
        X[56:156, :256, 0] = img[:, sl, 0]
        X[156:256, :256, 0] = img[:, sl, 2]
        X[56:156, :256, 1] = img[:, sl, 1]
        X[156:256, :256, 1] = img[:, sl, 3]
        X[56:156, :256, 2] = img[:, sl, 2]
        X[156:256, :256, 2] = img[:, sl, 1]

        X[56:156, 256:, 0] = img[:, sl, 0]
        X[156:256, 256:, 0] = img[:, sl, 1]
        X[56:156, 256:, 1] = img[:, sl, 2]
        X[156:256, 256:, 1] = img[:, sl, 3]

        X[256:356, :256, 0] = img[:, sl, 0]
        X[356:456, :256, 0] = img[:, sl, 1]
        X[256:356, :256, 1] = img[:, sl, 2]
        X[356:456, :256, 1] = img[:, sl, 3]
        X[256:356, :256, 2] = img[:, sl, 3]
        X[356:456, :256, 2] = img[:, sl, 2]

        X[256:356, 256:, 0] = img[:, sl, 0]
        X[356:456, 256:, 0] = img[:, sl, 2]
        X[256:356, 256:, 1] = img[:, sl, 1]
        X[356:456, 256:, 1] = img[:, sl, 3]

        if self.mode != "test":
            y[:] = row[TARGETS]
        return X, y




## === cell 3
val_transform = A.Compose([ToTensorV2(p=1.0)])



## === cell 4
import torch.utils.data as tud


def _to_probabilities(out: torch.Tensor) -> torch.Tensor:
    out = out.float()
    if out.ndim != 2:
        out = out.view(out.shape[0], -1)

    if out.shape[1] != 6:
        return out.softmax(dim=1)

    with torch.no_grad():
        mins = out.min(dim=1).values
        maxs = out.max(dim=1).values
        sums = out.sum(dim=1)
        looks_like_probs = (
            (mins >= -1e-4).all()
            and (maxs <= 1.0 + 1e-4).all()
            and torch.all(torch.isfinite(sums))
            and torch.all((sums - 1.0).abs() < 1e-2)
        )

    if looks_like_probs:
        return out.clamp_min(1e-9)
    return out.softmax(dim=1)


class _TorchDataset(tud.Dataset):
    def __init__(self, gen: DataGenerator):
        self.gen = gen
        self.n = int(len(getattr(gen, "data", gen)))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x, _ = self.gen[idx]
        return x


def run_inference_loop(model, test_ds, device, batch_size=32):
    model.to(device)
    model.eval()

    ds = _TorchDataset(test_ds)
    num_workers = min(4, (os.cpu_count() or 4))
    pin = device.type == "cuda"
    loader = tud.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
    )

    preds = []
    with torch.no_grad():
        for xb in tqdm.tqdm(loader, total=len(loader)):
            xb = xb.to(device, non_blocking=True)
            out = model(xb)
            prob = _to_probabilities(out).detach().cpu().numpy()
            preds.append(prob)

    if len(preds) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    return np.concatenate(preds, axis=0).astype(np.float32)


preds = []

test_ds_both = DataGenerator(
    test,
    mode="test",
    data_type="both",
    specs=spectrograms2,
    eeg_specs=all_eegs2,
    trans=val_transform,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_glob = "/kaggle/input/model101-mixnet-xl-fearture128-flip/model_101/*/*.pt"
model_paths = sorted(glob.glob(model_glob))
print("Found model files:", len(model_paths))

for model_path in model_paths:
    print(model_path)
    model = torch.load(model_path, map_location=device)
    pred = run_inference_loop(model, test_ds_both, device, batch_size=32)
    preds.append(pred)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

n_test = len(test)
n_classes = 6

if len(preds) == 0:
    test_pred = np.full((n_test, n_classes), 1.0 / n_classes, dtype=np.float32)
else:
    for i, p in enumerate(preds):
        if p.shape != (n_test, n_classes):
            raise ValueError(
                f"Model {i} pred shape {p.shape}, expected {(n_test, n_classes)}"
            )
    stacked = np.stack(preds, axis=0)  # (n_models, n_test, 6)
    test_pred = np.mean(stacked, axis=0).astype(np.float32)

test_pred = np.clip(test_pred, 1e-9, 1.0)
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
test_pred_df.insert(0, "eeg_id", sub_template["eeg_id"].values)

assert len(test_pred_df) == len(sub_template), (len(test_pred_df), len(sub_template))
test_pred_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", test_pred_df.shape)
print(test_pred_df.head())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/814541491.py in <cell line: 0>()
    123 
    124 test_pred_df = pd.DataFrame(test_pred, columns=CLASSES)
--> 125 test_pred_df.insert(0, "eeg_id", sub_template["eeg_id"].values)
    126 
    127 assert len(test_pred_df) == len(sub_template), (len(test_pred_df), len(sub_template))

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in insert(self, loc, column, value, allow_duplicates)
   5169             value = value.iloc[:, 0]
   5170 
-> 5171         value, refs = self._sanitize_column(value)
   5172         self._mgr.insert(loc, column, value, refs=refs)
   5173 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5264 
   5265         if is_list_like(value):
-> 5266             com.require_length_match(value, self.index)
   5267         arr = sanitize_array(value, self.index, copy=True, allow_2d=True)
   5268         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/common.py in require_length_match(data, index)
    571     """
    572     if len(data) != len(index):
--> 573         raise ValueError(
    574             "Length of values "
    575             f"({len(data)}) "

ValueError: Length of values (9850) does not match length of index (778942)
