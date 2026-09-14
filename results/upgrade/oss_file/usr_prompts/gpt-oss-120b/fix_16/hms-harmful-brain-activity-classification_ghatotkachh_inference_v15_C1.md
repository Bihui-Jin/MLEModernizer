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

0.5421195897242879

# 6. Current score

0.99821

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'Implemented safe model loading with fall‑back to uniform predictions, added directory checks, and ensured `finalpred` is always a NumPy array so the submission CSV is generated without errors.'
- What this solution (achieved 1.41937) has done: 'Implemented a simple prior‑based prediction to replace the default uniform fallback.  
The script now loads `train.csv`, computes the overall class vote distribution, and uses this distribution as a constant prediction when no trained models are found. This small change keeps the original pipeline intact while providing a more informed baseline that should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.68479) has done: 'I normalize the per‑patient prior vectors (which were raw vote counts) so they sum to 1, and then re‑normalize the final averaged predictions to guarantee each row is a proper probability distribution. This small correction aligns the fallback predictions with the KL‑divergence metric and should move the score toward the target without altering the core modelling pipeline.'
- What this solution (achieved 0.81597) has done: 'Implemented a blended prior (70% patient‑specific, 30% global) for fallback predictions to provide a more stable baseline, and added a tiny probability clipping before final renormalization to avoid zero probabilities that hurt KL‑divergence. These minor adjustments keep the original pipeline untouched while steering the score closer to the target lower‑is‑better metric.'
- What this solution (achieved 1.0413) has done: 'Implemented a lighter patient‑specific influence by adjusting the fallback blending: use 30 % patient prior + 70 % global prior (instead of the original 70/30). This simple change keeps the core pipeline unchanged while providing a more stable baseline that is expected to lower the KL‑divergence toward the target (lower‑is‑better). The same adjustment is applied to both model fallback blocks.'
- What this solution (achieved 0.81597) has done: 'Implemented a stronger patient‑specific influence by changing the fallback blending from 30 % patient + 70 % global to 70 % patient + 30 % global. This adjustment is applied in both fallback blocks (model 1 and model 2) so the predictions rely more on patient priors, which is expected to lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.0413) has done: 'Implemented two small adjustments to bring the KL‑divergence closer to the target:  
1. Compute per‑patient priors using **sum** of votes (instead of mean) before normalising, which better reflects the amount of annotation data per patient.  
2. Reduce the influence of patient‑specific priors in the fallback blend to **30 % patient + 70 % global**, giving a more stable generic prior when no trained models are present. These changes keep the original pipeline intact while moving the score toward the lower‑is‑better target.'
- What this solution (achieved 0.78827) has done: 'Implemented a stronger patient‑specific influence in the fallback predictions (80 % patient prior + 20 % global prior) for both model branches. This modest adjustment keeps the core pipeline unchanged while steering the averaged predictions toward the per‑patient distributions, which is expected to lower the KL‑divergence and move the metric closer to the target score.'
- What this solution (achieved 0.77747) has done: 'Implemented a higher patient‑specific weight (0.9 vs 0.8) for the fallback priors and added a mild temperature scaling (T = 1.2) to smooth the final probabilities before the last renormalisation. This keeps the original pipeline intact while making the predictions less over‑confident, which is expected to lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.13782) has done: 'The update adds the missing imports, guards against missing EEG/spectrogram data when running in test mode, and ensures the fallback‑only prediction pipeline runs end‑to‑end, producing a valid `submission.csv` whose rows sum to 1. These fixes unblock execution without altering the original modelling logic and move the solution toward the target score.'
- What this solution (achieved 0.77747) has done: 'Implemented stronger patient‑specific priors and milder temperature smoothing to bring the KL‑divergence closer to the target.  
- Updated both fallback blocks to use 90 % patient prior + 10 % global prior (instead of the previous 30/70 split).  
- Reduced temperature scaling from 1.5 to 1.2 so predictions are less overly smoothed while still avoiding zeros.  

These minimal adjustments keep the original pipeline intact but provide more informative, patient‑tailored predictions that lower the metric toward the desired score.'
- What this solution (achieved 0.78397) has done: 'Implemented a higher patient‑specific blend (95 % patient + 5 % global) and increased temperature smoothing to 1.5, which should produce slightly softer probability distributions and lower the KL‑divergence toward the target. Adjusted both fallback branches and the temperature exponent accordingly while preserving the overall pipeline and ensuring the final CSV is correctly written.'
- What this solution (achieved 0.99821) has done: 'Implemented lighter patient‑specific blending (70 % patient + 30 % global) and increased temperature scaling to 2.0. These adjustments keep the original pipeline unchanged while producing softer, more balanced probability distributions, which are expected to lower the KL‑divergence and move the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import pywt
import librosa
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class config:
    model1 = "resnet34d"
    model2 = "vit_base_patch16_224"
    epoch = 10
    lr = 1e-3
    batchsize = 32
    splits = 5
    momentum = 0.9
    MAX_GRAD_NORM = 1e7
    WEIGHT_DECAY = 0.01
    device = "cpu"
    FOLDS = 5
    AMP = True
    PATIENT_WEIGHT = 0.70
    GLOBAL_WEIGHT = 0.30
    TEMPERATURE = 2.0


class paths:
    preloadedeeg = "/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy"
    train_eeg_dir = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs"
    train_spec_dir = (
        "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms"
    )
    train_csv = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    test_csv = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    test_eeg = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs"
    test_spec = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms"
    )
    out = "/kaggle/working/"




## === cell 1
test_df = pd.read_csv(paths.test_csv)
train_df = pd.read_csv(paths.train_csv)

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

class_totals = train_df[TARGETS].sum().values.astype(np.float32)
class_prior = class_totals / class_totals.sum()
print(f"Computed class prior (sum to 1): {class_prior}")

patient_prior_df = train_df.groupby("patient_id")[TARGETS].sum()
patient_prior_dict = {}
for pid, row in patient_prior_df.iterrows():
    vec = row.values.astype(np.float32)
    s = vec.sum()
    if s > 0:
        vec = vec / s
    else:
        vec = class_prior  # fallback to global prior if sum is zero
    patient_prior_dict[pid] = vec
print(
    f"Computed per‑patient priors for {len(patient_prior_dict)} patients (fallback will use these, normalized)."
)



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
            plt.title(f"EEG {parquet_path} - Spectrogram {NAMES[k]}")

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
        plt.title(f"EEG Signals")
        plt.show()
        print("\n" + "#" * 25 + "\n")

    return img




## === cell 3
all_eegs = {}
if os.path.isdir(paths.test_eeg):
    for i in os.listdir(paths.test_eeg):
        sp = spectrogram_from_eeg(os.path.join(paths.test_eeg, i))
        name = int(i.split(".")[0])
        all_eegs[name] = np.array(sp)



## === cell 4
all_eegs  # confirm loading (no change)



## === cell 5
print(f"Test dataframe shape is: {test_df.shape}")
display(test_df.head())



## === cell 6
all_spectrograms = {}
if os.path.isdir(paths.test_spec):
    for i in os.listdir(paths.test_spec):
        sp = pd.read_parquet(os.path.join(paths.test_spec, i))
        name = int(i.split(".")[0])
        all_spectrograms[name] = np.array(sp)



## === cell 7
all_spectrograms  # confirm loading (no change)




## === cell 8
class CustomDataset:
    def __init__(
        self,
        traindf: pd.DataFrame,
        config,
        mode: str = "train",
        specs: dict[int, np.ndarray] = all_spectrograms,
        eegs: dict[int, np.ndarray] = all_eegs,
    ):
        self.traindf = traindf
        self.specs = specs
        self.eeg = eegs
        self.mode = mode

    def __len__(self):
        return len(self.traindf)

    def __getitem__(self, idx):
        X = np.zeros((128, 256, 8), dtype="float32")
        y = np.zeros(6, dtype="float32")
        img = np.ones((128, 256), dtype="float32")
        row = self.traindf.iloc[idx]
        if self.mode == "test":
            r = 0
        else:
            r = 0

        for region in range(4):
            spec = self.specs.get(
                row.spectrogram_id, np.zeros((300, 400), dtype="float32")
            )
            slice_img = spec[r : r + 300, region * 100 : (region + 1) * 100].T
            slice_img = np.clip(slice_img, np.exp(-4), np.exp(8))
            slice_img = np.log(slice_img)

            ep = 1e-6
            mu = np.nanmean(slice_img.flatten())
            std = np.nanstd(slice_img.flatten())
            slice_img = (slice_img - mu) / (std + ep)
            slice_img = np.nan_to_num(slice_img, nan=0.0)
            X[14:-14, :, region] = slice_img[:, 22:-22] / 2.0

            eeg_img = self.eeg.get(row.eeg_id, np.zeros((128, 256), dtype="float32"))
            X[:, :, 4:] = eeg_img

        X = torch.tensor(X)
        spectograms = [X[:, :, i : i + 1] for i in range(4)]
        spectograms = torch.cat(spectograms, dim=0)

        eegs = [X[:, :, i : i + 1] for i in range(4, 8)]
        eegs = torch.cat(eegs, dim=0)

        x = torch.cat([spectograms, eegs], dim=1)
        x = torch.cat([x, x, x], dim=2)
        x = x.permute(2, 0, 1)
        if self.mode != "test":
            y = row[TARGETS].values.astype(np.float32)

        return {"data": x, "target": y}




## === cell 9
customdataset = CustomDataset(test_df, config, mode="test")



## === cell 10
_ = customdataset[0]  # sanity check – no error should occur



## === cell 11
test_loader = DataLoader(
    customdataset,
    batch_size=config.batchsize,
)
X = customdataset[0]["data"]
y = customdataset[0]["target"]
print("Target placeholder (should be zeros in test mode):", y)




## === cell 12
class Custommodel(nn.Module):
    def __init__(
        self,
        config,
        numclass: int = 6,
    ):
        super(Custommodel, self).__init__()
        self.model = timm.create_model(
            config.model1,
            pretrained=False,
            drop_rate=0.1,
            drop_path_rate=0.2,
        )
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.customlayer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, numclass),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.customlayer(x)
        return x




## === cell 13
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    for batch in test_loader:
        x = batch["data"].to(device)
        with torch.no_grad():
            ypred = model(x)
        ypred = softmax(ypred)
        preds.append(ypred.cpu().numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 14
model_dir = "/kaggle/input/resnet34d2"
device = config.device
if os.path.isdir(model_dir):
    preds_list = []
    for fname in os.listdir(model_dir):
        dd = torch.load(os.path.join(model_dir, fname), map_location=device)
        model = Custommodel(config)
        model.load_state_dict(dd["model"])
        testdataset = CustomDataset(test_df, config, mode="test")
        testloader = DataLoader(testdataset, batch_size=config.batchsize)
        model.to(device)
        prediction_dict = inference_function(testloader, model, device)
        preds_list.append(prediction_dict["predictions"])
    predictions = np.mean(np.stack(preds_list), axis=0)
else:
    fallback = []
    for pid in test_df["patient_id"]:
        patient_prior = patient_prior_dict.get(pid, class_prior)
        blended = (
            config.PATIENT_WEIGHT * patient_prior + config.GLOBAL_WEIGHT * class_prior
        )
        fallback.append(blended)
    predictions = np.vstack(fallback).astype(np.float32)



## === cell 15
print("predictions shape:", predictions.shape)



## === cell 16
from torchvision.transforms import transforms

tras = transforms.Compose([transforms.Resize((224, 224))])


class Custommodel2(nn.Module):
    def __init__(self, config, transform, numclass: int = 6):
        super(Custommodel2, self).__init__()
        self.model = timm.create_model(
            config.model2,
            pretrained=False,
        )
        self.model.head = nn.Linear(self.model.head.in_features, numclass)
        self.transform = transform

    def forward(self, x):
        x = self.transform(x)
        x = self.model(x)
        return x




## === cell 17
vt_dir = "/kaggle/input/visiontransformer"
if os.path.isdir(vt_dir):
    preds2_list = []
    for fname in os.listdir(vt_dir):
        dd = torch.load(os.path.join(vt_dir, fname), map_location=device)
        model = Custommodel2(config, tras)
        model.load_state_dict(dd["model"])
        testdataset = CustomDataset(test_df, config, mode="test")
        testloader = DataLoader(testdataset, batch_size=config.batchsize)
        model.to(device)
        pred_dict = inference_function(testloader, model, device)
        preds2_list.append(pred_dict["predictions"])
    predictions2 = np.mean(np.stack(preds2_list), axis=0)
else:
    fallback2 = []
    for pid in test_df["patient_id"]:
        patient_prior = patient_prior_dict.get(pid, class_prior)
        blended = (
            config.PATIENT_WEIGHT * patient_prior + config.GLOBAL_WEIGHT * class_prior
        )
        fallback2.append(blended)
    predictions2 = np.vstack(fallback2).astype(np.float32)



## === cell 18
finalpred = (predictions + predictions2) / 2.0
finalpred = np.clip(finalpred, 1e-6, None)

temperature = config.TEMPERATURE
finalpred = np.power(finalpred, 1.0 / temperature)

row_sums = finalpred.sum(axis=1, keepdims=True)
zero_mask = row_sums == 0
if np.any(zero_mask):
    finalpred[zero_mask[:, 0]] = class_prior
    row_sums = finalpred.sum(axis=1, keepdims=True)
finalpred = finalpred / row_sums



## === cell 19
print("First few rows of final predictions (should sum to 1):")
print(finalpred[:5])
print("Row sums:", finalpred[:5].sum(axis=1))



## === cell 20
sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
sub[TARGETS] = finalpred
submission_path = os.path.join(paths.out, "submission.csv")
sub.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(f"Submission shape: {sub.shape}")
sub.head()
