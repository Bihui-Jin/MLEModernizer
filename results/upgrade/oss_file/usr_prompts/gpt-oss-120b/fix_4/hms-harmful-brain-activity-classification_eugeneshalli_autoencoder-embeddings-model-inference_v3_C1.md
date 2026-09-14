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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

1.0901246270812868

# 6. Current score

1.68479

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The fix removes the failing `umap` import, skips heavy EEG‑spectrogram processing that isn’t needed for inference, and replaces the missing model checkpoints with a simple baseline that uses the overall class vote distribution from the training data. This guarantees a valid `submission.csv` where each row sums to 1, satisfying the competition’s format and avoiding the previous runtime errors.'
- What this solution (achieved 1.68479) has done: 'I import the missing libraries (pandas, numpy, os) at the start so the data loading and baseline code run without NameErrors. The rest of the logic and the “USE_FEATURES = False” flag remain unchanged, ensuring the baseline predictions are generated correctly and a valid submission.csv is written.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
train_df = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
)
TARGETS = train_df.columns[-6:]



## === cell 2
test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
print("Test shape", test.shape)
test.head()



## === cell 3
USE_FEATURES = False

if USE_FEATURES:
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

    test = test.rename({"spectrogram_id": "spec_id"}, axis=1)



## === cell 4
if USE_FEATURES:
    import pywt, librosa

    USE_WAVELET = None

    NAMES = ["LL", "LP", "RP", "RR"]

    FEATS = [
        ["Fp1", "F7", "T3", "T5", "O1"],
        ["Fp1", "F3", "C3", "P3", "O1"],
        ["Fp2", "F8", "T4", "T6", "O2"],
        ["Fp2", "F4", "C4", "P4", "O2"],
    ]

    directory_path = "EEG_Spectrograms/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    def maddest(d, axis=None):
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    def denoise(x, wavelet="haar", level=1):
        coeff = pywt.wavedec(x, wavelet, mode="per")
        sigma = (1 / 0.6745) * maddest(coeff[-level])

        uthresh = sigma * np.sqrt(2 * np.log(len(x)))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])

        ret = pywt.waverec(coeff, wavelet, mode="per")
        return ret

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
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(
                    np.float32
                )[:, :width]

                mel_spec_db = (mel_spec_db + 40) / 40
                img[:, :, k] += mel_spec_db

            img[:, :, k] /= 4.0

            if display:
                plt.subplot(2, 2, k + 1)
                plt.imshow(img[:, :, k], aspect="auto", origin="lower")
                plt.title(f"EEG {{eeg_id}} - Spectrogram {NAMES[k]}")

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
            plt.title(f"EEG {{eeg_id}} Signals")
            plt.show()
            print()
            print("#" * 25)
            print()

        return img




## === cell 5
if USE_FEATURES:
    PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    DISPLAY = 1
    EEG_IDS2 = test.eeg_id.unique()
    all_eegs2 = {}

    print("Converting Test EEG to Spectrograms...")
    print()
    for i, eeg_id in enumerate(EEG_IDS2):
        img = spectrogram_from_eeg(f"{PATH2}{eeg_id}.parquet", i < DISPLAY)
        all_eegs2[eeg_id] = img



## === cell 6
if USE_FEATURES:
    TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
    TARS2 = {y: x for x, y in TARS.items()}

    class EEGDataset(Dataset):
        def __init__(self, data, augment=False, mode="train", eeg_specs=all_eegs2):
            self.data = data
            self.augment = augment
            self.mode = mode
            self.eeg_specs = eeg_specs

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.__getitems__([index])

        def __getitems__(self, indices):
            X, y = self._generate_data(indices)
            if self.augment:
                X = self.__augment(X)
            if self.mode == "train":
                return list(zip(X, y))
            else:
                return X

        def _generate_data(self, indexes):
            X = np.zeros((len(indexes), 4, 128, 256), dtype="float32")
            y = np.zeros((len(indexes), 6), dtype="float32")
            for j, i in enumerate(indexes):
                row = self.data.iloc[i]
                img = self.eeg_specs[row.eeg_id]
                img = np.transpose(img, (2, 0, 1))
                X[j] = img
                if self.mode != "test":
                    y[j,] = row[TARGETS].values
            return X, y

        def _random_transform(self, img):
            composition = albu.Compose([albu.HorizontalFlip(p=0.5)])
            return composition(image=img)["image"]

        def __augment(self, img_batch):
            for i in range(img_batch.shape[0]):
                img_batch[i,] = self._random_transform(img_batch[i,])
            return img_batch




## === cell 7
if USE_FEATURES:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CustomUpsample(nn.Module):
        def __init__(self, scale_factor=(2, 2)):
            super().__init__()
            self.scale_factor = scale_factor

        def forward(self, x):
            return F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")

    class ResNetBlock(nn.Module):
        def __init__(
            self, in_channels, kernel_size, modify=False, bn=True, scale_factor=(1, 1)
        ):
            super().__init__()
            self.modify = modify
            if modify == "downsample":
                self.conv1 = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels * 2,
                    stride=2,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                )
                self.conv2 = nn.Conv2d(
                    in_channels=in_channels * 2,
                    out_channels=in_channels * 2,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                )
                self.bn1 = nn.BatchNorm2d(in_channels * 2) if bn else nn.Identity()
                self.bn2 = nn.BatchNorm2d(in_channels * 2) if bn else nn.Identity()
            elif modify == "upsample":
                self.conv1 = nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=in_channels // 2,
                    stride=2,
                    kernel_size=kernel_size,
                    output_padding=scale_factor,
                    padding=kernel_size // 2,
                    bias=False,
                )
                self.conv2 = nn.Conv2d(
                    in_channels=in_channels // 2,
                    out_channels=in_channels // 2,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                )
                self.bn1 = nn.BatchNorm2d(in_channels // 2)
                self.bn2 = nn.BatchNorm2d(in_channels // 2)
            else:
                self.conv1 = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
                self.conv2 = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
                self.bn1 = nn.BatchNorm2d(in_channels)
                self.bn2 = nn.BatchNorm2d(in_channels)
            self.act = nn.ReLU()
            if modify == "downsample":
                self.proj = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels * 2,
                    stride=2,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            if modify == "upsample":
                self.proj = nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=in_channels // 2,
                    stride=2,
                    kernel_size=kernel_size,
                    output_padding=scale_factor,
                    padding=kernel_size // 2,
                )

        def forward(self, x):
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.act(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.modify:
                x = self.proj(x)
            out = x + out
            out = self.act(out)
            return out

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(4, 16, 7, 1, 7 // 2)
            self.rnb1 = ResNetBlock(16, 3, modify="downsample")
            self.rnb2 = ResNetBlock(32, 3, modify="downsample")
            self.rnb3 = ResNetBlock(64, 3, modify="downsample")
            self.rnb4 = ResNetBlock(128, 3, modify="downsample")
            self.rnb5 = ResNetBlock(256, 3, modify="downsample")
            self.rnb6 = ResNetBlock(512, 3, modify="downsample")
            self.rnb7 = ResNetBlock(1024, 3, modify="downsample")
            self.rnb8 = ResNetBlock(2048, 3, modify="downsample")

        def forward(self, x):
            for layer in [
                self.conv,
                self.rnb1,
                self.rnb2,
                self.rnb3,
                self.rnb4,
                self.rnb5,
                self.rnb6,
                self.rnb7,
                self.rnb8,
            ]:
                x = layer(x)
            return x

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnb1 = ResNetBlock(4096, 3, modify="upsample", scale_factor=(0, 1))
            self.rnb2 = ResNetBlock(2048, 3, modify="upsample")
            self.rnb3 = ResNetBlock(1024, 3, modify="upsample")
            self.rnb4 = ResNetBlock(512, 3, modify="upsample")
            self.rnb5 = ResNetBlock(256, 3, modify="upsample")
            self.rnb6 = ResNetBlock(128, 3, modify="upsample")
            self.rnb7 = ResNetBlock(64, 3, modify="upsample")
            self.rnb8 = ResNetBlock(32, 3, modify="upsample")
            self.conv = nn.Conv2d(16, 4, 3, 1, 3 // 2)

        def forward(self, x):
            for layer in [
                self.rnb1,
                self.rnb2,
                self.rnb3,
                self.rnb4,
                self.rnb5,
                self.rnb6,
                self.rnb7,
                self.rnb8,
                self.conv,
            ]:
                x = layer(x)
            return x




## === cell 8
if USE_FEATURES:

    class SimpleAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(Encoder(), Decoder())

        def forward(self, x):
            return self.net(x)

    class SimpleNet2(nn.Module):
        def __init__(self):
            super(SimpleNet2, self).__init__()
            self.fc1 = nn.Linear(4096, 1024)
            self.dropout1 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, 128)
            self.output = nn.Linear(128, 6)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout1(x)
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return self.output(x)




## === cell 9
global_dist = train_df[TARGETS].sum().astype(np.float32).values
global_dist /= global_dist.sum()

patient_dist = (
    train_df.groupby("patient_id")[TARGETS]
    .sum()
    .astype(np.float32)
    .div(
        train_df.groupby("patient_id")[TARGETS].sum().sum(axis=1),
        axis=0,
    )
    .fillna(0)
)

pred = np.tile(global_dist, (test.shape[0], 1))

patient_ids_in_train = set(patient_dist.index)
mask = test["patient_id"].isin(patient_ids_in_train)
if mask.any():
    patient_rows = test.loc[mask, "patient_id"]
    patient_vals = patient_dist.loc[patient_rows].values
    pred[mask.values] = patient_vals

row_sums = pred.sum(axis=1, keepdims=True)
pred = np.divide(pred, row_sums, where=row_sums != 0)

print("Baseline predictions shape:", pred.shape)



## === cell 10
sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
sub[TARGETS] = pred
sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
sub.head()



## === cell 11
row_sums = sub[TARGETS].sum(axis=1)
print(
    "Row sum statistics: min",
    row_sums.min(),
    "max",
    row_sums.max(),
    "mean",
    row_sums.mean(),
)
