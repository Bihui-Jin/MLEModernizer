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

0.4134873963078464

# 6. Current score

1.39626

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'Implemented a safe fallback for inference when no model files are found and ensured the generated predictions form a valid CSV submission. Added checks for empty model list, provided uniform probabilities as a default, and normalized predictions so each row sums to one. The DataFrame construction now always receives a proper NumPy array.'
- What this solution (achieved 1.40653) has done: 'I add a lightweight post‑processing step that blends the model ensemble predictions with the global class‑frequency prior computed from the training set. This small calibration often reduces Kullback‑Leibler divergence by tempering over‑confident predictions, moving the score closer to the target (lower is better). The change is confined to the inference‑output section, preserves all existing logic, and keeps the same file‑paths and submission format.'
- What this solution (achieved 1.3992) has done: 'I lower the blend weight toward the global prior and add a mild temperature scaling step to smooth the model probabilities, which should reduce over‑confidence and lower the KL‑divergence, moving the score closer to the target.'
- What this solution (achieved 1.39626) has done: 'I lower the blend weight toward the global prior and increase the temperature scaling so the model‐derived probabilities are flattened more strongly. This keeps the original inference pipeline unchanged while moving the averaged predictions closer to the overall class distribution, which typically reduces KL‑divergence and brings the score nearer the target.'
- What this solution (achieved 1.39677) has done: 'The changes adjust the temperature scaling and blend weight in the post‑processing step, making the predicted distributions flatter and giving slightly more influence to the global class prior. This calibration is expected to reduce over‑confidence and lower the KL‑divergence, moving the score closer to the target while keeping the core pipeline unchanged.'
- What this solution (achieved 1.39779) has done: 'I reduce the influence of the model predictions and increase temperature scaling so the final probabilities are closer to the global class prior, which should lower the KL‑divergence and move the score toward the target value.'
- What this solution (achieved 1.39626) has done: 'The script failed because the `rename` call used both `columns` and `axis`, which raises an error, and the test DataFrame lacked the `offset` column that the generator expects. I removed the conflicting `axis` argument, added an `offset` column (set to 0 for test rows), and reorganized the code into numbered cells that run sequentially. These minimal fixes enable the pipeline to load data, generate predictions, apply the calibrated post‑processing, and write a valid `submission.csv` file.'
- What this solution (achieved 1.39677) has done: 'I slightly increase the temperature scaling and give the global class‑frequency prior more weight (lower BLEND_ALPHA) so the predictions become a bit flatter and closer to the overall label distribution, which should reduce the KL‑divergence and move the score toward the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 1.39626) has done: 'We keep the whole pipeline unchanged and only adjust the post‑processing calibration: use a milder temperature flattening (TEMP = 2.0) and give the model a larger influence in the final blend (BLEND_ALPHA = 0.3). This makes the predictions less over‑flattened and lets the learned model signals contribute more, which should lower the KL‑divergence and move the score nearer the target.'

# 9. Code solution

## === cell 0
import os
import glob
import pandas as pd
import numpy as np
import torch
import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import librosa
import pywt

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
test = test.rename(columns={"spectrogram_id": "spec_id"})
test["offset"] = 0

PATH_SPEC = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
files_spec = os.listdir(PATH_SPEC)
print(f"There are {len(files_spec)} test spectrogram parquet files")

spectrograms2 = {}
for i, f in enumerate(files_spec):
    if i % 100 == 0:
        print(i, ", ", end="")
    tmp = pd.read_parquet(os.path.join(PATH_SPEC, f))
    name = int(f.split(".")[0])
    spectrograms2[name] = tmp.iloc[:, 1:].values  # drop first column (usually index)

PATH_EEG = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
DISPLAY = 0
EEG_IDS2 = test.eeg_id.unique()
all_eegs2 = {}

print("\nConverting Test EEG to Spectrograms...")
print()




## === cell 1
def eeg_from_parquet(parquet_path):
    eeg = pd.read_parquet(parquet_path, columns=FEATS2)
    rows = len(eeg)
    offset = (rows - 10_000) // 2
    eeg = eeg.iloc[offset : offset + 10_000]
    data = np.zeros((10_000, len(FEATS2)), dtype="float32")
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
    return np.mean(np.abs(d - np.mean(d, axis=axis)), axis=axis)


def denoise(x, wavelet="haar", level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode="per")


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
            m1 = np.nanmean(x1)
            if np.isnan(x1).mean() < 1:
                x1 = np.nan_to_num(x1, nan=m1)
            else:
                x1[:] = 0
            m2 = np.nanmean(x2)
            if np.isnan(x2).mean() < 1:
                x2 = np.nan_to_num(x2, nan=m2)
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
    img = spectrogram_from_eeg(os.path.join(PATH_EEG, f"{eeg_id}.parquet"), i < DISPLAY)
    all_eegs2[eeg_id] = img




## === cell 2
class DataGenerator:
    "Generates data for Keras (or PyTorch) pipelines"

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
        self.trans = trans
        self.on_epoch_end()

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
            return self.generate_all_specs(index)
        elif self.data_type in ("eeg", "kaggle"):
            return self.generate_specs(index)
        elif self.data_type == "raw":
            return self.generate_raw(index)
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")

    def generate_all_specs(self, index):
        X = np.zeros((512, 512, 3), dtype="float32")
        y = np.zeros((6,), dtype="float32")

        row = self.data.iloc[index]
        offset = 0 if self.mode == "test" else int(row.offset / 2)

        eeg = self.eeg_specs[row.eeg_id]
        spec = self.specs[row.spec_id]

        imgs = [
            spec[offset : offset + 300, k * 100 : (k + 1) * 100].T for k in [0, 2, 1, 3]
        ]  # to match kaggle with eeg
        img = np.stack(imgs, axis=-1)
        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)
        img = np.nan_to_num(img, nan=0.0)

        mn, mx = img.min(), img.max()
        img = 255 * (img - mn) / (mx - mn + 1e-5)

        X[56:156, :256, 0] = img[:, 22:-22, 0]  # LL_k
        X[156:256, :256, 0] = img[:, 22:-22, 2]  # RL_k
        X[56:156, :256, 1] = img[:, 22:-22, 1]  # LP_k
        X[156:256, :256, 1] = img[:, 22:-22, 3]  # RP_k
        X[56:156, :256, 2] = img[:, 22:-22, 2]  # RL_k (dup for channel 2)
        X[156:256, :256, 2] = img[:, 22:-22, 1]  # LP_k

        X[56:156, 256:, 0] = img[:, 22:-22, 0]
        X[156:256, 256:, 0] = img[:, 22:-22, 2]
        X[56:156, 256:, 1] = img[:, 22:-22, 1]
        X[156:256, 256:, 1] = img[:, 22:-22, 3]

        img_e = eeg
        mn, mx = img_e.min(), img_e.max()
        img_e = 255 * (img_e - mn) / (mx - mn + 1e-5)

        X[256:356, :256, 0] = img_e[:, 22:-22, 0]
        X[356:456, :256, 0] = img_e[:, 22:-22, 2]
        X[256:356, :256, 1] = img_e[:, 22:-22, 1]
        X[356:456, :256, 1] = img_e[:, 22:-22, 3]
        X[256:356, :256, 2] = img_e[:, 22:-22, 2]
        X[356:456, :256, 2] = img_e[:, 22:-22, 1]

        X[256:356, 256:, 0] = img_e[:, 22:-22, 0]
        X[356:456, 256:, 0] = img_e[:, 22:-22, 2]
        X[256:356, 256:, 1] = img_e[:, 22:-22, 1]
        X[356:456, 256:, 1] = img_e[:, 22:-22, 3]

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
        else:
            raise ValueError("Unsupported data_type for generate_specs")

        mn, mx = img.min(), img.max()
        img = 255 * (img - mn) / (mx - mn + 1e-5)

        X[56:156, :256, 0] = img[:, 22:-22, 0]
        X[156:256, :256, 0] = img[:, 22:-22, 2]
        X[56:156, :256, 1] = img[:, 22:-22, 1]
        X[156:256, :256, 1] = img[:, 22:-22, 3]
        X[56:156, :256, 2] = img[:, 22:-22, 2]
        X[156:256, :256, 2] = img[:, 22:-22, 1]

        X[56:156, 256:, 0] = img[:, 22:-22, 0]
        X[156:256, 256:, 0] = img[:, 22:-22, 1]
        X[56:156, 256:, 1] = img[:, 22:-22, 2]
        X[156:256, 256:, 1] = img[:, 22:-22, 3]

        X[256:356, :256, 0] = img[:, 22:-22, 0]
        X[356:456, :256, 0] = img[:, 22:-22, 1]
        X[256:356, :256, 1] = img[:, 22:-22, 2]
        X[356:456, :256, 1] = img[:, 22:-22, 3]
        X[256:356, :256, 2] = img[:, 22:-22, 3]
        X[356:456, :256, 2] = img[:, 22:-22, 2]

        X[256:356, 256:, 0] = img[:, 22:-22, 0]
        X[356:456, 256:, 0] = img[:, 22:-22, 2]
        X[256:356, 256:, 1] = img[:, 22:-22, 1]
        X[356:456, 256:, 1] = img[:, 22:-24, 3]  # typo retained as original logic

        if self.mode != "test":
            y[:] = row[TARGETS]
        return X, y




## === cell 3
val_transform = A.Compose([A.Resize(height=896, width=896, p=1.0), ToTensorV2(p=1.0)])




## === cell 4
def run_inference_loop(model, test_gen, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch_data, _ in tqdm.tqdm(test_gen):
            batch_data = batch_data.unsqueeze(0).to(device)  # add batch dim
            logits = model(batch_data)
            pred_list.append(logits.softmax(dim=1).cpu().numpy())
    return np.concatenate(pred_list, axis=0)


test_gen_both = DataGenerator(
    test,
    mode="test",
    data_type="both",
    specs=spectrograms2,
    eeg_specs=all_eegs2,
    trans=val_transform,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preds = []
model_paths = glob.glob("/kaggle/input/model101-mixnet-xl-896/model_101/*/*.pt")
for model_path in model_paths:
    print(f"Loading model: {model_path}")
    model = torch.load(model_path, map_location=device)
    pred = run_inference_loop(model, test_gen_both, device)
    preds.append(pred)

if len(preds) == 0:
    print("No model checkpoints found; using uniform predictions.")
    uniform_prob = np.full(
        (len(test), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32
    )
    test_pred = uniform_prob
else:
    test_pred = np.mean(np.stack(preds, axis=0), axis=0)

TEMP = 2.0  # lower temperature => less aggressive flattening
test_pred = np.power(test_pred, 1.0 / TEMP)
test_pred /= test_pred.sum(axis=1, keepdims=True)

BLEND_ALPHA = 0.3  # give the model a larger contribution
train_df = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
)
train_probs = train_df[TARGETS].div(train_df[TARGETS].sum(axis=1), axis=0)
global_prior = train_probs.mean().values.astype(np.float32)

test_pred = BLEND_ALPHA * test_pred + (1.0 - BLEND_ALPHA) * global_prior

row_sums = test_pred.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
test_pred = test_pred / row_sums

test_pred_df = pd.DataFrame(test_pred, columns=TARGETS)
submission = pd.concat([test[["eeg_id"]].reset_index(drop=True), test_pred_df], axis=1)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
submission.head()
