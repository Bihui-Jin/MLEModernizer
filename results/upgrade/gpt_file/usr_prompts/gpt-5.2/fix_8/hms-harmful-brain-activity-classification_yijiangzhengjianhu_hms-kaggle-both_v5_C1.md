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

0.4038249834600332

# 6. Current score

1.0749

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The failure comes from `test_pred` not being a proper 2D numeric array, which happens when no model files are found (so `preds` stays empty) or when concatenation fails silently due to shape issues. I add a strict check for found model paths and make `run_inference_loop` return a guaranteed `(N, 6)` float32 array; if no models are found, we fall back to a valid, well-formed uniform-probability submission (score be poor but it “yield” and be valid). I also make device selection safe (CPU fallback if CUDA isn’t available) and ensure the test generator yields the expected number of rows with correct column names (`spec_id` present). These changes are minimal and keep the core inference logic intact.'
- What this solution (achieved 1.21414) has done: 'Your current score is far worse than the target (lower-is-better KL; 1.40995 vs 0.4038), so we should improve predictions rather than just ensure validity. The biggest issue is that you compute EEG-derived spectrograms for all test EEGs up-front, which is slow and can force fallback/unintended behavior, and you also ignore `patient_id` even though it can provide a strong prior for label distribution. With minimal changes (no model/loop/architecture changes), I (1) make EEG spectrogram generation lazy and cached inside the generator so inference reliably runs within time/memory, and (2) blend the model probabilities with a patient-level prior learned from train votes (a small convex mix) to reduce KL on this competition. This keeps the same inference semantics (still softmax model outputs; just a light post-calibration blend) and still guarantees a valid submission with rows summing to 1.'
- What this solution (achieved 1.21953) has done: 'Your current score (1.21414, lower-is-better) is far from the target (0.4038), so we should make small, low-risk changes that typically reduce KL without changing your model or feature pipeline. The biggest easy win is fixing test-time preprocessing consistency: your `generate_specs` uses a different channel mapping for the Kaggle spectrograms than `generate_all_specs`, which can systematically harm predictions; we align it to the same mapping used in `both`. Next, we apply a tiny amount of probability “smoothing” toward the global prior (in addition to your patient prior blend) to reduce overconfident miscalibration, which often improves KL while preserving core semantics. Finally, we make the test generator non-shuffling (explicitly) to guarantee stable row order.'
- What this solution (achieved 1.15219) has done: 'Your current score (1.21953, lower-is-better) is still far from the target (0.4038), so we should make small, low-risk calibration changes that usually reduce KL without touching your model/features/training. I (1) stop shuffling the test generator on epoch end to ensure deterministic alignment during multi-model averaging, (2) add a tiny temperature scaling on logits (post-model, pre-softmax) to reduce overconfident predictions (often improves KL), and (3) slightly increase the prior blending strengths to improve robustness on out-of-distribution cases while keeping probabilities valid and normalized. These changes preserve your core pipeline (same inputs, same model, same inference loop) and only adjust post-processing/calibration. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 1.11635) has done: 'Your current KL (1.15219, lower-is-better) is still far above the target (0.4038), so we should improve calibration with the smallest post-processing changes that don’t touch the model or feature pipeline. I fix a quiet but important inefficiency/bug: your test generator reshuffles at the end of inference because `__call__` always triggers `on_epoch_end()`, which can desync ordering across model ensembling; we prevent any shuffling for `mode="test"` and avoid calling `on_epoch_end()` at the end of iteration. Then I make the KL-oriented smoothing slightly stronger but still conservative by (1) a small additional increase of temperature (softer probabilities) and (2) a tiny increase in prior blending, keeping outputs normalized and valid. These changes preserve your core logic (same inputs, same model loading, same softmax) and should move KL downward toward the target.'
- What this solution (achieved 1.07038) has done: 'Your KL is still far above the target (lower-is-better), so we should improve calibration with the smallest post-processing changes that don’t touch your model, data pipeline, or inference loop structure. The most reliable low-risk move for KL on this competition is slightly stronger probability smoothing (blend more with patient and global priors) plus a small extra temperature softening of logits to reduce overconfidence. These changes keep the same predictions semantics (softmax over model outputs) and preserve row alignment and submission validity (still normalized to sum to 1). I only adjust the three scalar calibration knobs (temperature/alpha/beta) and keep everything else intact.'
- What this solution (achieved 1.0749) has done: 'Your current KL (1.07038, lower-is-better) is still far above the target (0.4038), so we should improve reliability of the averaged predictions with very small, low-risk calibration/robustness changes (without changing the model or feature pipeline). The biggest likely remaining issue is that the generator is stateful and can be inadvertently reshuffled/iterated inconsistently across multiple models; I freeze test ordering explicitly and make inference iterate by index (not by the generator’s `__call__`) to guarantee identical sample alignment for every model in the ensemble. Then, to move KL down cautiously, I apply a tiny additional KL-friendly smoothing: an extremely small per-row “epsilon” mix with the global prior (keeps probabilities valid, reduces overconfidence) while leaving your existing priors/temperature logic intact. These are minimal changes and should nudge the score toward the target without altering the core architecture, features, or training.'

# 9. Code solution

## === cell 0
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import tqdm



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

INPUT_DIR = "/kaggle/input/hms-harmful-brain-activity-classification"

test = pd.read_csv(f"{INPUT_DIR}/test.csv")
print("Test shape", test.shape)
print(test.head())

test = test.rename({"spectrogram_id": "spec_id"}, axis=1)

PATH_SPECS_TEST = f"{INPUT_DIR}/test_spectrograms/"
files2 = os.listdir(PATH_SPECS_TEST)
print(f"There are {len(files2)} test spectrogram parquets")

spectrograms2 = {}
for i, f in enumerate(files2):
    if i % 200 == 0:
        print(i, ", ", end="")
    tmp = pd.read_parquet(f"{PATH_SPECS_TEST}{f}")
    name = int(f.split(".")[0])
    spectrograms2[name] = tmp.iloc[:, 1:].values
print()

PATH_EEG_TEST = f"{INPUT_DIR}/test_eegs/"



## === cell 2
import librosa
import pywt


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
        eeg_path=None,
        eeg_cache_size=256,
    ):
        self.data = data
        self.augment = augment
        self.mode = mode
        self.data_type = data_type
        self.specs = specs
        self.eeg_specs = eeg_specs if eeg_specs is not None else {}
        self.raw_eegs = raw_eegs

        self.eeg_path = eeg_path
        self.eeg_cache_size = int(eeg_cache_size)
        self._eeg_cache_order = []

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

    def on_epoch_end(self):
        if self.mode == "train":
            self.data = self.data.sample(frac=1).reset_index(drop=True)

    def _get_eeg_spec(self, eeg_id: int):
        if eeg_id in self.eeg_specs:
            return self.eeg_specs[eeg_id]

        if self.eeg_path is None:
            raise ValueError(
                "eeg_path is None but eeg spectrogram requested and not cached."
            )

        img = spectrogram_from_eeg(
            f"{self.eeg_path}{eeg_id}.parquet", display=False
        ).astype(np.float32)

        self.eeg_specs[eeg_id] = img
        self._eeg_cache_order.append(eeg_id)
        if len(self._eeg_cache_order) > self.eeg_cache_size:
            old_id = self._eeg_cache_order.pop(0)
            if old_id in self.eeg_specs:
                del self.eeg_specs[old_id]
        return img

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

        eeg = self._get_eeg_spec(int(row.eeg_id))
        spec = self.specs[int(row.spec_id)]

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
            img = self._get_eeg_spec(int(row.eeg_id))
        elif self.data_type == "kaggle":
            spec = self.specs[int(row.spec_id)]
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
        X[100 + 56 : 200 + 56, 256:, 0] = img[:, 22:-22, 2]
        X[0 + 56 : 100 + 56, 256:, 1] = img[:, 22:-22, 1]
        X[100 + 56 : 200 + 56, 256:, 1] = img[:, 22:-22, 3]

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

        if self.mode != "test":
            y[:] = row[TARGETS]

        return X, y




## === cell 4
LOGIT_TEMPERATURE = 1.35  # keep: calibration knob already tuned previously


def run_inference_loop(model, test_gen_both, device, logit_temperature: float = 1.0):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(test_gen_both))):
            batch_data, _ = test_gen_both[i]
            batch_data = (
                torch.from_numpy(batch_data).permute(2, 0, 1).unsqueeze(0).to(device)
            )
            out = model(batch_data)
            if logit_temperature is not None and float(logit_temperature) != 1.0:
                out = out / float(logit_temperature)
            prob = out.softmax(dim=1).detach().cpu().numpy().astype(np.float32)
            pred_list.append(prob)

    if len(pred_list) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    return np.concatenate(pred_list, axis=0)


train = pd.read_csv(f"{INPUT_DIR}/train.csv")

vote_cols = TARGETS
train_votes = train[vote_cols].values.astype(np.float32)
train_probs = train_votes / np.clip(train_votes.sum(axis=1, keepdims=True), 1e-8, None)

global_prior = train_probs.mean(axis=0).astype(np.float32)
global_prior = global_prior / global_prior.sum()

patient_prior_map = {}
grp = train.assign(_idx=np.arange(len(train))).groupby("patient_id")["_idx"].apply(list)
for pid, idxs in grp.items():
    p = train_probs[np.array(idxs)].mean(axis=0).astype(np.float32)
    s = float(p.sum())
    if s > 0:
        p = p / s
    else:
        p = global_prior.copy()
    patient_prior_map[int(pid)] = p


def get_patient_prior(pid: int) -> np.ndarray:
    return patient_prior_map.get(int(pid), global_prior)


preds = []

test = test.reset_index(drop=True)

test_gen_both = DataGenerator(
    test,
    mode="test",
    data_type="both",
    specs=spectrograms2,
    eeg_specs=None,
    eeg_path=PATH_EEG_TEST,
    eeg_cache_size=256,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model_paths = sorted(
    glob.glob("/kaggle/input/model90-stage1-stage2/model_90" + "/*/*.pt")
)
print("Found model files:", len(model_paths))

if len(model_paths) > 0:
    for model_path in model_paths:
        print(model_path)
        model = torch.load(model_path, map_location="cpu")
        pred = run_inference_loop(
            model, test_gen_both, device, logit_temperature=LOGIT_TEMPERATURE
        )
        preds.append(pred)

    test_pred = np.mean(np.stack(preds, axis=0), axis=0).astype(np.float32)
else:
    test_pred = np.full((len(test), 6), 1.0 / 6.0, dtype=np.float32)

CLASSES = vote_cols

assert test_pred.shape == (
    len(test),
    6,
), f"Unexpected prediction shape: {test_pred.shape} vs {(len(test), 6)}"

ALPHA = 0.28  # keep: patient-prior blend
priors = np.stack(
    [get_patient_prior(pid) for pid in test["patient_id"].values], axis=0
).astype(np.float32)
test_pred = (1.0 - ALPHA) * test_pred + ALPHA * priors

BETA = 0.14  # keep: global-prior blend
test_pred = (1.0 - BETA) * test_pred + BETA * global_prior.reshape(1, -1)

EPS_SMOOTH = 0.02
test_pred = (1.0 - EPS_SMOOTH) * test_pred + EPS_SMOOTH * global_prior.reshape(1, -1)

test_pred = np.clip(test_pred, 1e-8, 1.0)
test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)

test_pred_df = pd.DataFrame(test_pred, columns=CLASSES)
test_pred_df = pd.concat(
    [test[["eeg_id"]].reset_index(drop=True), test_pred_df], axis=1
)

test_pred_df.to_csv("submission.csv", index=False)
print(test_pred_df.head())
print("Wrote submission.csv with shape:", test_pred_df.shape)
print(
    "Row-sum check (min/max):", test_pred.sum(axis=1).min(), test_pred.sum(axis=1).max()
)
