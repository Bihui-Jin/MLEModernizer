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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scipy==1.15.3
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

0.4356722345673653

# 6. Current score

0.77949

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'Your current notebook doesn’t yield a score mainly because it relies on a checkpoint path that almost certainly doesn’t exist in your environment; that means you’re submitting random weights, which score very poorly even if the CSV is valid. To move the KL score toward your target with minimal changes and without changing model/training logic, I (1) robustly load the checkpoint if it exists in standard Kaggle locations and otherwise fall back to a safe uniform-probability submission (valid and usually much better than random logits), and (2) enforce deterministic ordering/alignment of `eeg_id` rows to avoid silent mismatches. These changes preserve architecture, feature extraction, and inference semantics when weights are available, while guaranteeing a valid submission and a reasonable baseline score when they are not.'
- What this solution (achieved 1.41937) has done: 'Your current score (1.40995, lower is better) suggests the checkpoint still isn’t being loaded, so you’re effectively submitting random/untrained outputs; the smallest reliable way to move toward the target is to (1) broaden and harden checkpoint discovery (search common Kaggle input dirs recursively for `HMS_EEG_MODEL0_1.pth`) and verify state_dict key mapping so it actually loads, and (2) if no checkpoint is found, replace the “uniform” fallback with a safer “train prior” fallback computed from `train.csv` vote totals (this usually beats uniform on KL without changing core modeling). Additionally, I enforce deterministic alignment by always building the submission strictly in `sample_submission.csv` order and I add an optional temperature smoothing (very mild) to reduce overconfident probabilities if a checkpoint is found, which often improves KL slightly without changing architecture or training. These changes keep your model and feature extraction intact and only affect weight-loading robustness and probability calibration/post-processing. The script still runs end-to-end and always writes a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 1.19354) has done: 'Your current KL (1.41937; lower is better) is far from the target (0.4357), and the most likely cause is still that the checkpoint isn’t being loaded/used correctly, so you’re effectively submitting a weak fallback. I make two minimal, score-relevant changes: (1) strengthen checkpoint loading by handling common checkpoint formats (raw state_dict, `{"state_dict":...}`, `{"model":...}`) and normalizing key prefixes more robustly, and (2) improve the fallback from a global train prior to a *patient-aware prior* computed from `train.csv` grouped by `patient_id` (and global prior as backup), which usually improves KL without changing model logic. I also keep submission row order exactly as `sample_submission.csv` and ensure probabilities are finite, clipped, and normalized to sum to 1.'
- What this solution (achieved 0.78698) has done: 'Your current score (1.19354, lower is better) is still far from the target (0.4357), and the dominant issue is almost certainly that the intended checkpoint is not being found/loaded (so you’re mostly relying on the weak fallback). I make the smallest score-relevant change by (1) expanding checkpoint discovery to load *any* `.pth` in Kaggle inputs that looks like an HMS EEG model (and prefer ones matching your original name), and (2) if we still must fallback, upgrade the fallback from patient-only priors to a more accurate `(patient_id, spectrogram_id)` conditional prior with smoothing back to patient/global priors. These changes keep your model architecture and inference the same when a checkpoint is available, while materially improving the fallback probabilities (which should reduce KL versus your current fallback). The submission is still written in `sample_submission.csv` order and strictly normalized to sum to 1.'
- What this solution (achieved 0.77954) has done: 'Your current KL (0.78698, lower is better) is still well above the target (0.43567), so we should make the smallest changes that legitimately reduce KL without changing the model/training core. The biggest easy win for KL on this competition is to avoid overconfident probabilities; we do this in both branches by mixing your predictions with a well-formed prior (computed from train vote totals) and applying a very mild “label-smoothing style” blend that preserves semantics but improves calibration. For the checkpoint-loaded path, we keep your exact model/inference and only add: (1) robust probability normalization, and (2) prior-mixing after softmax (not changing architecture or weights). For the fallback path, we keep your existing conditional priors but add a tiny global-prior blend and a slightly stronger smoothing to reduce KL spikes on hard examples.'
- What this solution (achieved 0.77954) has done: 'Your current score (0.77954, lower is better) is still far above the target (0.43567), so we should make the smallest, lowest-risk change that plausibly reduces KL without changing the model/feature logic. The biggest issue is that `domain="test"` makes `TrainingDatasetEEG` use the wrong index mapping and can silently fetch the wrong EEGs (train-style indexing logic applied to test), which damages predictions even if a checkpoint is loaded. I fix only the test-path indexing so it always reads the correct `eeg_id` (no change to architecture, loss, or feature extraction), and I also ensure the loaded-checkpoint path emits properly normalized probabilities (not logs) and applies the existing mild prior-mix for calibration. The fallback conditional-prior branch is kept intact and still writes a valid `submission.csv`.'
- What this solution (achieved 0.77751) has done: 'Your current KL (0.77954, lower is better) is still far above the target (0.43567), so we should make a minimal, low-risk calibration change that reduces KL without touching your model architecture or feature extraction. The biggest safe lever for KL is smoothing overconfident predictions, so I slightly increase the post-softmax prior-mixing strength and temperature scaling in the checkpoint-loaded path, and I slightly increase the final global-prior blend in the fallback conditional-prior path. I also fix a subtle evaluation bug in `test()` where you take `softmax` then `log` instead of using `log_softmax` directly (same semantics, but numerically safer), which can reduce instability if you ever run validation. All changes keep your pipeline end-to-end and still write a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 0.77949) has done: 'Your current KL (0.77751, lower is better) is still far above the target (0.43567), so we should make the smallest, lowest-risk change that reduces KL without changing your model or feature extraction. The safest lever for KL here is calibration: slightly stronger temperature scaling and a slightly stronger blend with a global prior to avoid overconfident predictions, which typically reduces KL. I keep your checkpoint-loading/inference exactly as-is and only adjust post-softmax smoothing (and keep strict normalization) so the submission remains valid. I also make the same mild strengthening in the fallback prior-only path so both branches move in the right direction.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import albumentations as A

import torch as tc
import torch.nn as nn
import torch.nn.functional as F

import timm
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, lfilter

from tqdm import tqdm



## === cell 1
path = "/kaggle/input/hms-harmful-brain-activity-classification"
batch_size = 16
device = "cuda" if tc.cuda.is_available() else "cpu"
domain = "test"  # keep original behavior (submission-only)
read_all_eegs = False
read_all_specs = False
read_all_l2i = False
is_augument = False
is_test_random = True

tc.manual_seed(0)
np.random.seed(0)
if device == "cuda":
    tc.cuda.manual_seed_all(0)

print("device:", device, "| domain:", domain)



## === cell 2
df = pd.read_csv(path + f"/{domain}.csv")

if domain == "train":
    train_df = df.copy()

    train_df = train_df[(train_df.iloc[:, -6:].sum(1) > 6)]

    label_cols = train_df.columns[-6:]
    eeg_ids = train_df.eeg_id.unique()
    train_df = df.groupby("eeg_id")[["patient_id"]].agg("first")
    aux = df.groupby("eeg_id")[label_cols].agg("sum")
    si = df.groupby("eeg_id")[
        ["spectrogram_id", "spectrogram_label_offset_seconds"]
    ].agg("first")

    for k in si:
        train_df[k] = si[k].values

    for label in label_cols:
        train_df[label] = aux[label].values

    y_data = train_df[label_cols].values
    y_data = y_data / y_data.sum(axis=1, keepdims=True)
    train_df[label_cols] = y_data

    train_df = train_df.reset_index()
    train_df = train_df.loc[train_df.eeg_id.isin(eeg_ids)]
    print(f"Train dataframe with unique eeg_id has shape: {train_df.shape}")
    print(train_df.head())

    train_df.iloc[:, -6:].sum(0).plot.bar()
    plt.show()
    df = train_df

print("df shape:", df.shape)




## === cell 3
def softmax(d, dim):
    d = d - np.max(d, axis=dim, keepdims=True)
    ed = np.exp(d)
    return ed / (ed.sum(axis=dim, keepdims=True) + 1e-12)


def butter_lowpass_filter(data, cutoff_freq=40, sampling_rate=200, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data


def butter_highpass_filter(data, cutoff_freq=1, sampling_rate=200, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data


def get_eeg_from_parquet(idx, metadata, eeg_ids, domain_for_io="train"):
    eeg_id = eeg_ids[idx]
    eeg_dir = "train_eegs" if domain_for_io == "train" else "test_eegs"
    eeg_path = path + f"/{eeg_dir}/" + str(eeg_id) + ".parquet"
    eeg = pd.read_parquet(eeg_path).iloc[0:10_000, :]
    eeg = eeg.values.T
    return eeg


def get_spec_from_parquet(idx, metadata, eeg_ids, domain_for_io="train"):
    eeg_id = eeg_ids[idx]

    if domain_for_io == "train":
        occurances_df = metadata.loc[metadata["eeg_id"] == eeg_id, :]
        spec_id = occurances_df.loc[:, "spectrogram_id"].values[0]
        spec_offset = int(
            occurances_df.loc[:, "spectrogram_label_offset_seconds"].values[0] // 2
        )
        spec_path = path + "/train_spectrograms/" + str(spec_id) + ".parquet"
        x = pd.read_parquet(spec_path)
        x = (
            x.values[spec_offset : 300 + spec_offset, 1:]
            .reshape(300, 4, 100)
            .transpose((1, 2, 0))
        )
        return x
    else:
        spec_id = int(
            metadata.loc[metadata["eeg_id"] == eeg_id, "spectrogram_id"].values[0]
        )
        spec_path = path + "/test_spectrograms/" + str(spec_id) + ".parquet"
        x = (
            pd.read_parquet(spec_path)
            .values[:300, 1:]
            .reshape(300, 4, 100)
            .transpose((1, 2, 0))
        )
        return x


def process_spec(x):
    x = np.clip(x, np.exp(-4), np.exp(8))
    x = np.log(x)

    mu = np.nanmean(x)
    sigma = np.nanstd(x)
    x = (x - mu) / (sigma + 1e-5)
    x = np.nan_to_num(x, nan=0.0)
    return x


def line2img(x):
    C, L = x.shape

    mx = x.max(axis=1, keepdims=True)
    mn = x.min(axis=1, keepdims=True)
    nx = (x - mn) / (mx - mn + 1e-5)

    sw = 3
    H, W = 128 - sw, 1024 - sw

    h_inds = (nx * (H - 1)).astype(np.int32)
    w_inds = np.linspace(0, W - 1, L)[None, :].repeat(C, axis=0).astype(np.int32)
    chans = np.array(range(C))[:, None]

    img = np.zeros((C, H + sw, W + sw), dtype=np.uint8)
    for i in range(sw):
        for j in range(sw):
            img[chans, h_inds + i, w_inds + j] = 1

    img = cv2.resize(
        img.transpose(1, 2, 0), (512, 64), interpolation=cv2.INTER_AREA
    ).transpose(2, 0, 1)
    return img




## === cell 4
def get_eeg_frame():
    class TrainingDatasetEEGPreload(Dataset):
        def __init__(self, metadata):
            self.metadata = metadata
            self.eeg_ids = np.array(
                sorted(metadata["eeg_id"].unique()), dtype=np.int64
            ).squeeze()

        def __len__(self):
            return self.eeg_ids.shape[0]

        def __getitem__(self, idx):
            eeg = get_eeg_from_parquet(
                idx, self.metadata, self.eeg_ids, domain_for_io="train"
            )

            is_corrupted = 0
            for i in range(eeg.shape[0]):
                nans = np.isnan(eeg[i]).sum()
                if nans / 1e4 > 0.7:
                    is_corrupted = 1

            eeg = np.nan_to_num(eeg, nan=0).astype(np.float32)
            eeg = tc.from_numpy(eeg).type(tc.float32)

            return eeg, is_corrupted

    bs = 4
    train_dataset = TrainingDatasetEEGPreload(df)
    train_dataloader = DataLoader(
        train_dataset, batch_size=bs, shuffle=False, num_workers=4
    )
    all_eegs = np.zeros((df["eeg_id"].unique().shape[0], 20, 10000), dtype=np.float32)
    all_is_corrupted_list = np.zeros(
        (df["eeg_id"].unique().shape[0],), dtype=np.float32
    )
    j = 0
    for eegs, is_corrupted_list in tqdm(train_dataloader):
        for eeg, is_corrupted in zip(eegs, is_corrupted_list):
            all_eegs[j] = eeg.numpy()
            all_is_corrupted_list[j] = float(is_corrupted)
            j += 1
    return all_eegs, all_is_corrupted_list


def get_spec_frame():
    class TrainingDatasetSpecPreload(Dataset):
        def __init__(self, metadata):
            self.metadata = metadata
            self.eeg_ids = np.array(
                sorted(metadata["eeg_id"].unique()), dtype=np.int64
            ).squeeze()

        def __len__(self):
            return self.eeg_ids.shape[0]

        def __getitem__(self, idx):
            spec = get_spec_from_parquet(
                idx, self.metadata, self.eeg_ids, domain_for_io="train"
            )
            spec = process_spec(spec)
            spec = tc.from_numpy(np.ascontiguousarray(spec)).type(tc.float32)
            return spec

    bs = 4
    train_dataset = TrainingDatasetSpecPreload(df)
    train_dataloader = DataLoader(
        train_dataset, batch_size=bs, shuffle=False, num_workers=4
    )
    all_specs = np.zeros(
        (df["eeg_id"].unique().shape[0], 4, 100, 300), dtype=np.float32
    )
    j = 0
    for specs in tqdm(train_dataloader):
        for spec in specs:
            all_specs[j] = spec.numpy()
            j += 1
    return all_specs


def eeg2img():
    names = [
        "Fp1",
        "F3",
        "C3",
        "P3",
        "F7",
        "T3",
        "T5",
        "O1",
        "Fz",
        "Cz",
        "Pz",
        "Fp2",
        "F4",
        "C4",
        "P4",
        "F8",
        "T4",
        "T6",
        "O2",
        "EKG",
    ]
    pairs = list(zip(range(20), names))
    pairs = {k.lower(): v for v, k in pairs}
    cen_locs = [
        "fp1,fp1,fp1,fp2,fp2,fp2".split(","),
        "f7,f3,fz,fz,f4,f8".split(","),
        "t3,c3,cz,cz,c4,t4".split(","),
        "t5,p3,pz,pz,p4,t6".split(","),
        "o1,o1,o1,o2,o2,o2".split(","),
    ]
    cen_ilocs = [pairs[i] for i in sum(cen_locs, [])]
    cen_ilocs = np.array(cen_ilocs).reshape(5, 6).astype(np.int32)

    def _eeg2img(idx, metadata, eeg_ids):
        eeg = get_eeg_from_parquet(idx, metadata, eeg_ids, domain_for_io="train")

        x2_ = eeg[cen_ilocs, 4000:6000]  # 5,6,L
        x2_ = np.split(x2_, 5, axis=0)
        x2_ = np.concatenate([x2_[i] - x2_[i + 1] for i in range(4)], 0)  # 4,6,L
        x2_ = x2_[:, [0, 1, 4, 5]].reshape(4 * 4, -1)
        x2 = butter_lowpass_filter(x2_.T, cutoff_freq=40).T
        x2 = np.nan_to_num(x2, nan=0)
        x2 = np.clip(x2, -1024, 1024)
        imgs = line2img(x2)
        imgs = (imgs - 0.5) / 0.5
        return imgs

    return _eeg2img


def get_eegimg_frame():
    class TrainingDatasetEEGImgPreload(Dataset):
        def __init__(self, metadata):
            self.metadata = metadata
            self.eeg_ids = np.array(
                sorted(metadata["eeg_id"].unique()), dtype=np.int64
            ).squeeze()
            self.func = eeg2img()

        def __len__(self):
            return self.eeg_ids.shape[0]

        def __getitem__(self, idx):
            imgs = self.func(idx, self.metadata, self.eeg_ids)
            imgs = tc.from_numpy(np.ascontiguousarray(imgs)).type(tc.uint8)
            return imgs

    bs = 4
    train_dataset = TrainingDatasetEEGImgPreload(df)
    train_dataloader = DataLoader(
        train_dataset, batch_size=bs, shuffle=False, num_workers=4
    )
    all_imgs = np.zeros((df["eeg_id"].unique().shape[0], 16, 64, 512), dtype=np.uint8)
    j = 0
    for imgs in tqdm(train_dataloader):
        for img in imgs:
            all_imgs[j] = img.numpy()
            j += 1
    return all_imgs


if read_all_eegs and domain == "train":
    all_eegs, all_is_corrupted_list = get_eeg_frame()
if read_all_specs and domain == "train":
    all_specs = get_spec_frame()
if read_all_l2i and domain == "train":
    all_imgs = get_eegimg_frame()




## === cell 5
class TrainingDatasetEEG(Dataset):
    def __init__(self, metadata, train=True, is_test_random=False):
        self.train = train
        self.metadata = metadata

        neeg_ids = np.array(
            sorted(metadata["eeg_id"].unique()), dtype=np.int64
        ).squeeze()
        if read_all_eegs and domain == "train":
            eeg_ids = np.array(
                [
                    eeg_id
                    for eeg_id, is_corrupted in zip(neeg_ids, all_is_corrupted_list)
                ],
                dtype=np.int64,
            )
        else:
            eeg_ids = neeg_ids

        num_eeg_ids = eeg_ids.shape[0]
        train_size_percentage = 85 if not is_test_random else 100
        train_set_size = int(train_size_percentage / 100 * num_eeg_ids)
        self.train_set_size = train_set_size
        eeg_ids_train = eeg_ids[:train_set_size]
        eeg_ids_valid = (
            eeg_ids[train_set_size:]
            if not is_test_random
            else np.random.choice(eeg_ids_train, 500, replace=False)
        )
        self.eeg_ids = eeg_ids_train if train else eeg_ids_valid

        names = [
            "Fp1",
            "F3",
            "C3",
            "P3",
            "F7",
            "T3",
            "T5",
            "O1",
            "Fz",
            "Cz",
            "Pz",
            "Fp2",
            "F4",
            "C4",
            "P4",
            "F8",
            "T4",
            "T6",
            "O2",
            "EKG",
        ]
        pairs = list(zip(range(20), names))
        pairs = {k.lower(): v for v, k in pairs}
        cen_locs = [
            "fp1,fp1,fp1,fp2,fp2,fp2".split(","),
            "f7,f3,fz,fz,f4,f8".split(","),
            "t3,c3,cz,cz,c4,t4".split(","),
            "t5,p3,pz,pz,p4,t6".split(","),
            "o1,o1,o1,o2,o2,o2".split(","),
        ]
        cen_ilocs = [pairs[i] for i in sum(cen_locs, [])]
        self.cen_ilocs = np.array(cen_ilocs).reshape(5, 6).astype(np.int32)

        self.transform = A.Compose(
            [
                A.Blur(blur_limit=3, p=0.2),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.2
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.OneOf(
                    [
                        A.CoarseDropout(
                            max_holes=12, max_height=60, max_width=64, p=0.05
                        ),
                        A.GridDropout(p=0.05),
                    ],
                    p=0.1,
                ),
                A.ElasticTransform(p=0.05),
                A.GridDistortion(p=0.05),
                A.OpticalDistortion(p=0.05),
            ],
            p=0.2,
        )

    def __len__(self):
        return self.eeg_ids.shape[0]

    def __getitem__(self, idx):
        eeg_id = self.eeg_ids[idx]
        idx_full = idx if self.train else idx + self.train_set_size

        io_domain = "train" if domain == "train" else "test"

        if domain == "train":
            occurances_df = self.metadata.loc[self.metadata["eeg_id"] == eeg_id, :]

        if read_all_eegs and domain == "train":
            eeg = all_eegs[idx_full].copy()
        else:
            if domain == "train":
                eeg = get_eeg_from_parquet(
                    idx_full - self.train_set_size,
                    self.metadata,
                    self.eeg_ids,
                    domain_for_io=io_domain,
                )
                for i in range(eeg.shape[0]):
                    nans = np.isnan(eeg[i]).sum()
                    if nans / 1e4 > 0.7:
                        nidx = idx + 1
                        if nidx >= self.__len__():
                            nidx = 0
                        return self.__getitem__(nidx)
                eeg = np.nan_to_num(eeg, nan=0)
            else:
                eeg = get_eeg_from_parquet(
                    idx, self.metadata, self.eeg_ids, domain_for_io=io_domain
                )
                eeg = np.nan_to_num(eeg, nan=0)

        cutoff_freq = 40
        gn = 0.0
        direction = 1
        polarity = 1.0

        if is_augument and self.train and domain == "train":
            if np.random.rand() > 0.8:
                cutoff_freq = int(np.random.choice(list(range(16, 30, 4))))
            if np.random.rand() > 0.8:
                gnmx = eeg.mean(1)[:, None] * float(
                    np.random.choice((0.025, 0.05, 0.1))
                )
                polarity = np.random.choice((-1.0, 1.0), eeg[0].shape)[None, :]
                gn = polarity * np.random.random(eeg[0].shape)[None, :] * gnmx
            if np.random.rand() > 0.8:
                direction = -1
            if np.random.rand() > 0.8:
                polarity = float(np.random.choice((-1.0, 1.0)))

        eeg = eeg[:, ::direction] + gn
        eeg *= polarity

        x1 = butter_lowpass_filter(eeg.T, cutoff_freq=cutoff_freq).T
        x1 = np.clip(x1, -1024, 1024)

        x2_ = eeg[self.cen_ilocs, :]
        x2_ = np.split(x2_, 5, axis=0)
        x2_ = np.concatenate([x2_[i] - x2_[i + 1] for i in range(4)], 0)
        x2_ = x2_.reshape(4 * 6, -1)
        x2 = butter_lowpass_filter(x2_.T, cutoff_freq=cutoff_freq).T
        x2 = np.clip(x2, -1024, 1024)

        x3_ = eeg[self.cen_ilocs[[0, 2, 4], :][:, [0, 1, 4, 5]], :]
        x3_ = np.split(x3_, 3, axis=0)
        x3_ = np.concatenate([x3_[i] - x3_[i + 1] for i in range(2)], 0)
        x3_ = x3_.reshape(2 * 4, -1)
        x3 = butter_lowpass_filter(x3_.T, cutoff_freq=cutoff_freq).T
        x3 = np.clip(x3, -1024, 1024)

        if read_all_specs and domain == "train":
            x4 = all_specs[idx_full]
        else:
            if domain == "train":
                x4 = get_spec_from_parquet(
                    idx_full - self.train_set_size,
                    self.metadata,
                    self.eeg_ids,
                    domain_for_io=io_domain,
                )
            else:
                x4 = get_spec_from_parquet(
                    idx, self.metadata, self.eeg_ids, domain_for_io=io_domain
                )
            x4 = process_spec(x4)

        if read_all_l2i and domain == "train":
            x5 = all_imgs[idx_full]
        else:
            x5 = x2[:, 4000:6000].reshape(4, 6, -1)[:, [0, 1, 4, 5]].reshape(4 * 4, -1)
            x5 = line2img(x5).astype(np.float32)

        if is_augument and self.train and domain == "train":
            for i in range(16):
                augmented = self.transform(image=x5[i])
                x5[i] = augmented["image"]

        x5 = (x5 - 0.5) / 0.5

        if domain == "train":
            targets = occurances_df.iloc[:, -6:].values.squeeze()
        else:
            targets = np.zeros((6,), dtype=np.float32)

        x1 = tc.from_numpy(np.ascontiguousarray(x1)).type(tc.float32) / 128.0
        x2 = tc.from_numpy(np.ascontiguousarray(x2)).type(tc.float32) / 32.0
        x3 = tc.from_numpy(np.ascontiguousarray(x3)).type(tc.float32)
        x4 = tc.from_numpy(np.ascontiguousarray(x4)).type(tc.float32)
        x5 = tc.from_numpy(np.ascontiguousarray(x5)).type(tc.float32)
        targets = tc.from_numpy(targets).type(tc.float32)
        return (x1, x2, x3, x4, x5), targets.squeeze()




## === cell 6
class DatasetEEG(Dataset):
    def __init__(self, metadata):
        self.metadata = metadata.reset_index(drop=True)
        names = [
            "Fp1",
            "F3",
            "C3",
            "P3",
            "F7",
            "T3",
            "T5",
            "O1",
            "Fz",
            "Cz",
            "Pz",
            "Fp2",
            "F4",
            "C4",
            "P4",
            "F8",
            "T4",
            "T6",
            "O2",
            "EKG",
        ]

        pairs = list(zip(range(20), names))
        pairs = {k.lower(): v for v, k in pairs}
        cen_locs = [
            "fp1,fp1,fp1,fp2,fp2,fp2".split(","),
            "f7,f3,fz,fz,f4,f8".split(","),
            "t3,c3,cz,cz,c4,t4".split(","),
            "t5,p3,pz,pz,p4,t6".split(","),
            "o1,o1,o1,o2,o2,o2".split(","),
        ]
        cen_ilocs = [pairs[i] for i in sum(cen_locs, [])]
        self.cen_ilocs = np.array(cen_ilocs).reshape(5, 6).astype(np.int32)

    def __len__(self):
        return self.metadata.shape[0]

    def __getitem__(self, idx):
        eeg_id = int(self.metadata.loc[idx, "eeg_id"])
        eeg_path = path + "/test_eegs/" + str(eeg_id) + ".parquet"
        eeg = pd.read_parquet(eeg_path).iloc[:10_000, :].values.T
        eeg = np.nan_to_num(eeg, nan=0)

        cutoff_freq = 40

        x1 = butter_lowpass_filter(eeg.T, cutoff_freq=cutoff_freq).T
        x1 = np.clip(x1, -1024, 1024)

        x2_ = eeg[self.cen_ilocs, :]
        x2_ = np.split(x2_, 5, axis=0)
        x2_ = np.concatenate([x2_[i] - x2_[i + 1] for i in range(4)], 0)
        x2_ = x2_.reshape(4 * 6, -1)
        x2 = butter_lowpass_filter(x2_.T, cutoff_freq=cutoff_freq).T
        x2 = np.clip(x2, -1024, 1024)

        x3_ = eeg[self.cen_ilocs[[0, 2, 4], :][:, [0, 1, 4, 5]], :]
        x3_ = np.split(x3_, 3, axis=0)
        x3_ = np.concatenate([x3_[i] - x3_[i + 1] for i in range(2)], 0)
        x3_ = x3_.reshape(2 * 4, -1)
        x3 = butter_lowpass_filter(x3_.T, cutoff_freq=cutoff_freq).T
        x3 = np.clip(x3, -1024, 1024)

        spec_id = int(self.metadata.loc[idx, "spectrogram_id"])
        spec_path = path + "/test_spectrograms/" + str(spec_id) + ".parquet"
        x4 = (
            pd.read_parquet(spec_path)
            .values[:300, 1:]
            .reshape(300, 4, 100)
            .transpose((1, 2, 0))
        )
        x4 = process_spec(x4)

        x5 = x2[:, 4000:6000].reshape(4, 6, -1)[:, [0, 1, 4, 5]].reshape(4 * 4, -1)
        x5 = line2img(x5).astype(np.float32)
        x5 = (x5 - 0.5) / 0.5

        x1 = tc.from_numpy(np.ascontiguousarray(x1)).type(tc.float32) / 128.0
        x2 = tc.from_numpy(np.ascontiguousarray(x2)).type(tc.float32) / 32.0
        x3 = tc.from_numpy(np.ascontiguousarray(x3)).type(tc.float32)
        x4 = tc.from_numpy(np.ascontiguousarray(x4)).type(tc.float32)
        x5 = tc.from_numpy(np.ascontiguousarray(x5)).type(tc.float32)
        return x1, x2, x3, x4, x5, eeg_id




## === cell 7
p_dropout = 0.3 if domain == "train" else 0.0


class BBlock(nn.Module):
    def __init__(self, o, g, s=2, h=None):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv1d(o, o * s, 3, padding=1, groups=g),
            nn.BatchNorm1d(o * s),
            nn.SiLU(),
            nn.Conv1d(o * s, o * s, 3, padding=1, groups=g),
            nn.BatchNorm1d(o * s),
            nn.SiLU(),
            nn.Conv1d(o * s, o * s, 3, padding=1, groups=g),
        )
        self.c_res = nn.Conv1d(o, o * s, 1)
        self.c_add_norm = nn.Sequential(nn.BatchNorm1d(o * s), nn.SiLU())
        o_ = o if h is None else h
        self.c_final = nn.Sequential(
            nn.Conv1d(o * s, o_, 3, padding=1, groups=g), nn.BatchNorm1d(o_), nn.SiLU()
        )

    def forward(self, x):
        c = self.c(x)
        c_res = self.c_res(x)
        c_norm = self.c_add_norm(c + c_res)
        return self.c_final(c_norm)


class BBlock2(nn.Module):
    def __init__(self, o, g, s=2, h=None, is_strided=False, stride=None):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(o, o * s, 3, padding=1, groups=g),
            nn.BatchNorm2d(o * s),
            nn.SiLU(),
            nn.Conv2d(o * s, o * s, 3, padding=1, groups=g),
            nn.BatchNorm2d(o * s),
            nn.SiLU(),
            nn.Conv2d(o * s, o * s, 3, padding=1, groups=g),
        )
        self.c_res = nn.Conv2d(o, o * s, 1, groups=g)
        self.c_add_norm = nn.Sequential(nn.BatchNorm2d(o * s), nn.SiLU())
        o_ = o if h is None else h
        self.c_final = nn.Sequential(
            nn.Conv2d(o * s, o_, 3, stride if is_strided else 1, padding=1, groups=g),
            nn.BatchNorm2d(o_),
            nn.SiLU(),
        )

    def forward(self, x):
        c = self.c(x)
        c_res = self.c_res(x)
        c_norm = self.c_add_norm(c + c_res)
        return self.c_final(c_norm)


class TBlock(nn.Module):
    def __init__(self, o, g, s=2):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv1d(o, o * s, 3, padding=1, groups=g),
            nn.Tanh(),
            nn.Conv1d(o * s, o * s, 3, padding=1, groups=g),
            nn.Tanh(),
            nn.Conv1d(o * s, o * s, 3, padding=1, groups=g),
        )
        self.c_res = nn.Conv1d(o, o * s, 1, groups=g)
        self.c_add_norm = nn.Tanh()
        self.c_final = nn.Sequential(
            nn.Conv1d(o * s, o, 3, padding=1, groups=g), nn.Tanh()
        )

    def forward(self, x):
        c = self.c(x)
        c_res = self.c_res(x)
        c_norm = self.c_add_norm(c + c_res)
        return self.c_final(c_norm)


class EEGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = TBlock(20, 20, s=2)
        self.c2 = nn.Sequential(
            nn.Conv1d(190, 190, 1, groups=190),
            nn.Tanh(),
            nn.AvgPool1d(5),
            nn.Conv1d(190, 32, 3, padding=1, groups=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
        )
        self.c3 = nn.Sequential(
            BBlock(32, 1, s=2, h=16),
            BBlock(16, 1, s=2, h=24),
            nn.AvgPool1d(5),
            BBlock(24, 1, s=2, h=32),
            BBlock(32, 1, s=2, h=48),
            nn.AvgPool1d(5),
            BBlock(48, 1, s=2, h=64),
            BBlock(64, 1, s=2, h=72),
            nn.AvgPool1d(5),
            BBlock(72, 1, s=2, h=96),
        )

        self.d1 = nn.Sequential(
            BBlock(1, 1, s=4, h=4),
            nn.AvgPool1d(5),
            BBlock(4, 1, s=2, h=8),
            nn.AvgPool1d(5),
            BBlock(8, 1, s=2, h=12),
            nn.AvgPool1d(5),
            BBlock(12, 1, s=2, h=16),
            nn.AvgPool1d(5),
            BBlock(16, 1, s=2, h=24),
        )
        self.d2 = nn.Sequential(nn.Dropout(p_dropout), nn.Linear(480, 128), nn.SiLU())

        self.e1 = nn.Sequential(
            BBlock(1, 1, s=4, h=4),
            nn.AvgPool1d(5),
            BBlock(4, 1, s=2, h=8),
            nn.AvgPool1d(5),
            BBlock(8, 1, s=2, h=12),
            nn.AvgPool1d(5),
            BBlock(12, 1, s=2, h=16),
            nn.AvgPool1d(5),
            BBlock(16, 1, s=2, h=24),
        )
        self.e2 = nn.Sequential(nn.Dropout(p_dropout), nn.Linear(576, 128), nn.SiLU())

        self.f1 = nn.Sequential(
            BBlock(4, 1, s=4, h=8),
            nn.AvgPool1d(5),
            BBlock(8, 1, s=2, h=12),
            nn.AvgPool1d(5),
            BBlock(12, 1, s=2, h=16),
            nn.AvgPool1d(5),
            BBlock(16, 1, s=2, h=24),
            nn.AvgPool1d(5),
            BBlock(24, 1, s=2, h=32),
        )
        self.f2 = nn.Sequential(nn.Dropout(p_dropout), nn.Linear(192, 96), nn.SiLU())

        self.g1 = nn.Sequential(
            BBlock(1, 1, s=4, h=8),
            nn.AvgPool1d(2),
            BBlock(8, 1, s=2, h=16),
            nn.AvgPool1d(2),
            BBlock(16, 1, s=2, h=32),
            nn.AvgPool1d(2),
            BBlock(32, 1, s=2, h=64),
            nn.AvgPool1d(2),
            BBlock(64, 1, s=2, h=128),
            nn.AvgPool1d(2),
            BBlock(128, 1, s=1, h=256),
            BBlock(256, 1, s=1, h=324),
            BBlock(324, 1, s=1, h=512),
        )
        self.g2 = nn.Sequential(
            nn.Dropout(p_dropout), nn.Linear(512 * 8, 512), nn.SiLU()
        )

        self.h1 = nn.Sequential(
            BBlock2(1, 1, s=2, h=8),
            nn.AvgPool2d(2),
            BBlock2(8, 1, s=2, h=12),
            nn.AvgPool2d(2),
            BBlock2(12, 1, s=2, h=24),
            nn.AvgPool2d(2),
            BBlock2(24, 1, s=2, h=32),
            nn.AvgPool2d(2),
            BBlock2(32, 1, s=2, h=48),
            nn.AvgPool2d(2),
            BBlock2(48, 1, s=2, h=64),
        )
        self.h2 = nn.Sequential(
            nn.Dropout(p_dropout), nn.Linear(64 * 4, 256), nn.SiLU()
        )

        self.i1 = nn.Sequential(
            BBlock2(1, 1, s=2, h=8, is_strided=True, stride=(1, 2)),
            BBlock2(8, 1, s=2, h=12, is_strided=True, stride=(1, 2)),
            BBlock2(12, 1, s=2, h=24),
            nn.AvgPool2d(2),
            BBlock2(24, 1, s=2, h=32),
            nn.AvgPool2d(2),
            BBlock2(32, 1, s=2, h=48),
            nn.AvgPool2d(2),
            BBlock2(48, 1, s=2, h=64),
        )
        self.i2 = nn.Sequential(
            nn.Dropout(p_dropout), nn.Linear(64 * 16, 512), nn.SiLU()
        )

        self.j1 = timm.create_model("tf_efficientnet_b0", pretrained=False, in_chans=4)
        self.j1.classifier = nn.Linear(self.j1.classifier.in_features, 512)
        self.j2 = nn.Sequential(
            nn.Dropout(p_dropout), nn.Linear(512 * 4, 512), nn.SiLU()
        )

    def ind_feature_extractor(self, x):
        d1 = tc.cat([self.d1(x[:, i, :].unsqueeze(1)) for i in range(x.shape[1])], 1)
        return self.d2(d1.mean(-1))

    def total_feature_extractor(self, x):
        c1 = self.c1(x)
        c2 = []
        for i in range(20):
            X = c1[:, i, :]
            for j in range(i + 1, 20):
                c2.append(X - c1[:, j, :])
        c2 = self.c2(F.tanh(tc.stack(c2, 1)))
        c3 = self.c3(c2)
        return c3.mean(-1)

    def montage_features_extractor(self, x):
        e1 = tc.cat([self.e1(x[:, i, :].unsqueeze(1)) for i in range(x.shape[1])], 1)
        return self.e2(e1.mean(-1))

    def norm(self, x, axis=1):
        mx = x.max(axis, keepdims=True)[0]
        mn = x.min(axis, keepdims=True)[0]
        return (x - mn) / (mx - mn + 1e-5)

    def montage_features_extractor2(self, x):
        with tc.no_grad():
            x = self.norm(x, 2)
        g1 = tc.cat([self.g1(x[:, i, :].unsqueeze(1)) for i in range(x.shape[1])], 1)
        return self.g2(g1.mean(-1))

    def gr_montage_features_extractor(self, x):
        N, S, L = x.shape
        x = x.view(N, 4, 6, L)
        f1 = tc.cat([self.f1(x[:, :, i, :]).mean(-1) for i in range(6)], 1)
        return self.f2(f1)

    def kagg_spec_feature_extractor(self, x):
        h1 = tc.cat(
            [self.h1(x[:, i, ...].unsqueeze(1)).mean((-2, -1)) for i in range(4)], 1
        )
        return self.h2(h1)

    def eeg_img_feature_extractor(self, x):
        i1 = tc.cat(
            [self.i1(x[:, i, ...].unsqueeze(1)).mean((-2, -1)) for i in range(16)], 1
        )
        return self.i2(i1)

    def eeg_img_feature_extractor2(self, x):
        with tc.no_grad():
            N, C, H, W = x.shape
            x = x.view(N, 4, 4, H, W)
        j1 = tc.cat([self.j1(x[:, :, i, ...]) for i in range(4)], 1)
        return self.j2(j1)

    def train_branch(self, func, x, randomize=False):
        return func(x)

    def forward(self, x1, x2, x3, x4, x5):
        h1 = self.train_branch(
            self.total_feature_extractor, x1, True if self.train else False
        )
        h2 = self.train_branch(
            self.ind_feature_extractor, x1, True if self.train else False
        )
        h3 = self.train_branch(
            self.montage_features_extractor, x2, True if self.train else False
        )
        h4 = self.train_branch(
            self.gr_montage_features_extractor, x2, True if self.train else False
        )
        h5 = self.train_branch(
            self.montage_features_extractor2, x3, True if self.train else False
        )
        h6 = self.train_branch(
            self.kagg_spec_feature_extractor, x4, True if self.train else False
        )
        h7 = self.train_branch(
            self.eeg_img_feature_extractor, x5, True if self.train else False
        )
        h8 = self.train_branch(
            self.eeg_img_feature_extractor2, x5, True if self.train else False
        )
        return tc.cat([h1, h2, h3, h4, h5, h6, h7, h8], 1)


class EEGBasedClassifier(nn.Module):
    def __init__(self, load_weights=True, load_out_weights=True):
        super().__init__()
        self.feature_extractor = EEGFeatureExtractor()
        self.out = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(96 + 128 + 128 + 96 + 512 + 256 + 512 + 512, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(p_dropout),
            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.SiLU(),
            nn.Dropout(p_dropout),
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(p_dropout),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(p_dropout),
            nn.Linear(512, 6),
        )

        self.load_out_weights = load_out_weights
        self.ckpt_loaded = False

        def _find_ckpt():
            preferred_names = {
                "HMS_EEG_MODEL0_1.pth",
                "hms_eeg_model0_1.pth",
                "hms_eeg_model.pth",
                "model.pth",
                "best.pth",
                "checkpoint.pth",
            }
            preferred_hits = []
            other_hits = []
            roots = ["/kaggle/input", "/kaggle/working"]
            for r in roots:
                if not os.path.isdir(r):
                    continue
                for dirpath, _, filenames in os.walk(r):
                    for fn in filenames:
                        if not fn.lower().endswith(".pth"):
                            continue
                        fp = os.path.join(dirpath, fn)
                        if fn in preferred_names:
                            preferred_hits.append(fp)
                        elif ("hms" in fn.lower()) and ("eeg" in fn.lower()):
                            other_hits.append(fp)
            preferred_hits = sorted(set(preferred_hits))
            other_hits = sorted(set(other_hits))
            if preferred_hits:
                return preferred_hits[0]
            if other_hits:
                return other_hits[0]
            return None

        def _extract_state_dict(obj):
            if isinstance(obj, dict):
                for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
                    if k in obj and isinstance(obj[k], dict):
                        return obj[k]
            return obj

        def _norm_key(k: str) -> str:
            for pref in ["model.", "module.", "net.", "network."]:
                if k.startswith(pref):
                    k = k[len(pref) :]
            return k

        if load_weights:
            ckpt_path = _find_ckpt()
            if ckpt_path is not None:
                checkpoint_obj = tc.load(ckpt_path, map_location=tc.device(device))
                checkpoint = _extract_state_dict(checkpoint_obj)

                fe_state = self.feature_extractor.state_dict()
                updated = 0
                for key, val in checkpoint.items():
                    nk = _norm_key(key)
                    for cand in [nk, nk.replace("feature_extractor.", "", 1)]:
                        if (cand in fe_state) and (fe_state[cand].shape == val.shape):
                            fe_state[cand] = val
                            updated += 1
                            break
                self.feature_extractor.load_state_dict(fe_state, strict=False)

                if load_out_weights:
                    out_state = self.out.state_dict()
                    out_updated = 0
                    for key, val in checkpoint.items():
                        nk = _norm_key(key)
                        candidates = []
                        if nk.startswith("out."):
                            candidates.append(nk[len("out.") :])
                        candidates.append(nk)
                        for cand in candidates:
                            if (cand in out_state) and (
                                out_state[cand].shape == val.shape
                            ):
                                out_state[cand] = val
                                out_updated += 1
                                break
                    self.out.load_state_dict(out_state, strict=False)
                else:
                    out_updated = 0

                self.ckpt_loaded = True
                print(
                    f"Loaded checkpoint: {ckpt_path} | feature_extractor tensors matched: {updated} | out tensors matched: {out_updated}"
                )
            else:
                print(
                    "Checkpoint not found; will fallback to conditional prior submission (train vote distribution)."
                )

    def forward(self, x1, x2, x3, x4, x5):
        if self.load_out_weights:
            h = self.feature_extractor(x1, x2, x3, x4, x5)
        else:
            with tc.no_grad():
                h = self.feature_extractor(x1, x2, x3, x4, x5)
        return self.out(h)




## === cell 8
model = EEGBasedClassifier(load_weights=True, load_out_weights=True).to(device)
print(
    "trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad)
)




## === cell 9
def test(model, criterion, ret=False):
    test_dataset = TrainingDatasetEEG(df, train=False, is_test_random=is_test_random)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=max(batch_size, 128),
        shuffle=False,
        num_workers=min(batch_size, 4),
    )
    print("Starting Testing")
    model.eval()
    running_loss = 0.0
    ya = []
    yp = []
    for i, data in enumerate(test_dataloader, 0):
        (x1, x2, x3, x4, x5), targets = data
        ya.append(targets.squeeze())

        with tc.cuda.amp.autocast(enabled=(device == "cuda")):
            with tc.no_grad():
                outputs = model(
                    x1.to(device),
                    x2.to(device),
                    x3.to(device),
                    x4.to(device),
                    x5.to(device),
                )
                log_probs = F.log_softmax(outputs, dim=-1)
                loss = criterion(log_probs, targets.to(device))
                yp.append(log_probs.detach().cpu().numpy().squeeze())

        running_loss += loss.item()
    print(f"Test Loss: {running_loss / max(i,1):.6f}")
    if ret:
        return np.concatenate(ya), np.concatenate(yp)


def train(model, optimizer, criterion, scaler, scheduler, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loss_count = 0

        for i, data in enumerate(train_dataloader, 0):
            (x1, x2, x3, x4, x5), targets = data

            with tc.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = model(
                    x1.to(device),
                    x2.to(device),
                    x3.to(device),
                    x4.to(device),
                    x5.to(device),
                )
                outputs = F.log_softmax(outputs, -1)
                loss = criterion(outputs, targets.to(device))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_loss += loss.item()
            loss_count += 1
            if i % 100 == 99:
                print(
                    f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / loss_count:.6f}"
                )
                running_loss = 0.0
                loss_count = 0

        scheduler.step()
        test(model, criterion)

    print("Finished Training")


def _safe_normalize_probs(arr2d, eps=1e-8):
    arr2d = np.asarray(arr2d, dtype=np.float64)
    arr2d = np.nan_to_num(arr2d, nan=0.0, posinf=0.0, neginf=0.0)
    arr2d = np.clip(arr2d, eps, 1.0)
    s = arr2d.sum(axis=1, keepdims=True)
    s[s <= 0] = 1.0
    arr2d = arr2d / s
    return arr2d.astype(np.float32)


def _build_conditional_prior_tables(path_for_train_csv: str):
    vote_cols = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    train_csv = os.path.join(path_for_train_csv, "train.csv")
    if not os.path.exists(train_csv):
        global_prior = np.ones((6,), dtype=np.float32) / 6.0
        return global_prior, {}, {}

    tr = pd.read_csv(train_csv, usecols=["patient_id", "spectrogram_id"] + vote_cols)

    sums_global = tr[vote_cols].sum(axis=0).values.astype(np.float64)
    if (not np.isfinite(sums_global).all()) or sums_global.sum() <= 0:
        global_prior = np.ones((6,), dtype=np.float32) / 6.0
    else:
        global_prior = (sums_global / sums_global.sum()).astype(np.float32)
        global_prior = np.clip(global_prior, 1e-8, 1.0)
        global_prior = global_prior / global_prior.sum()

    grp_p = tr.groupby("patient_id")[vote_cols].sum()
    patient_prior = {}
    for pid, row in grp_p.iterrows():
        v = row.values.astype(np.float64)
        if (not np.isfinite(v).all()) or v.sum() <= 0:
            continue
        p = (v / v.sum()).astype(np.float32)
        p = np.clip(p, 1e-8, 1.0)
        p = p / p.sum()
        patient_prior[int(pid)] = p

    grp_ps = tr.groupby(["patient_id", "spectrogram_id"])[vote_cols].sum()
    pat_spec_prior = {}
    for (pid, sid), row in grp_ps.iterrows():
        v = row.values.astype(np.float64)
        if (not np.isfinite(v).all()) or v.sum() <= 0:
            continue
        p = (v / v.sum()).astype(np.float32)
        p = np.clip(p, 1e-8, 1.0)
        p = p / p.sum()
        pat_spec_prior[(int(pid), int(sid))] = p

    return global_prior, patient_prior, pat_spec_prior


def _mix_with_prior(probs_2d: np.ndarray, prior_1d: np.ndarray, alpha: float):
    """
    Score-relevant for KL: prior-mixing reduces overconfidence and stabilizes KL.
    """
    probs_2d = _safe_normalize_probs(probs_2d)
    prior_1d = np.asarray(prior_1d, dtype=np.float32).reshape(1, -1)
    prior_1d = _safe_normalize_probs(np.repeat(prior_1d, probs_2d.shape[0], axis=0))[0]
    out = (1.0 - alpha) * probs_2d + alpha * prior_1d.reshape(1, -1)
    return _safe_normalize_probs(out)


def submit(model):
    vote_cols = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

    sample = pd.read_csv(path + "/sample_submission.csv")

    global_prior, patient_prior, pat_spec_prior = _build_conditional_prior_tables(path)

    if not getattr(model, "ckpt_loaded", False):
        test_meta = pd.read_csv(
            path + "/test.csv", usecols=["eeg_id", "patient_id", "spectrogram_id"]
        )
        test_meta["eeg_id"] = test_meta["eeg_id"].astype(np.int64)
        test_meta["patient_id"] = test_meta["patient_id"].astype(np.int64)
        test_meta["spectrogram_id"] = test_meta["spectrogram_id"].astype(np.int64)
        meta_map = dict(
            zip(
                test_meta["eeg_id"].values,
                zip(test_meta["patient_id"].values, test_meta["spectrogram_id"].values),
            )
        )

        w_pat_spec = 0.80
        w_pat = 0.18
        w_glob = 0.02

        probs = np.zeros((sample.shape[0], 6), dtype=np.float32)
        eeg_ids = sample["eeg_id"].astype(np.int64).values
        for i, eid in enumerate(eeg_ids):
            pid, sid = meta_map.get(int(eid), (-1, -1))
            p_ps = pat_spec_prior.get((int(pid), int(sid)), None)
            p_p = patient_prior.get(int(pid), None)

            if p_ps is not None:
                p0 = p_ps
                p1 = p_p if p_p is not None else global_prior
                probs[i] = (w_pat_spec * p0) + (w_pat * p1) + (w_glob * global_prior)
            elif p_p is not None:
                probs[i] = (0.95 * p_p) + (0.05 * global_prior)
            else:
                probs[i] = global_prior

        probs = _mix_with_prior(probs, global_prior, alpha=0.10)

        for j, c in enumerate(vote_cols):
            sample[c] = probs[:, j]

        sample.to_csv("submission.csv", index=False)
        print(
            "Checkpoint unavailable -> wrote conditional prior submission.csv with shape:",
            sample.shape,
            "| global_prior:",
            global_prior,
            "| num_patient_priors:",
            len(patient_prior),
            "| num_patient_spectrogram_priors:",
            len(pat_spec_prior),
        )
        print(sample.head())
        return sample

    dataset = DatasetEEG(df)
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=min(batch_size, 4)
    )
    model.eval()

    temperature = 1.55

    eeg_id_list = []
    prob_list = []

    for x1, x2, x3, x4, x5, eeg_ids in tqdm(dataloader, desc="Inference"):
        with tc.cuda.amp.autocast(enabled=(device == "cuda")):
            with tc.no_grad():
                logits = model(
                    x1.to(device),
                    x2.to(device),
                    x3.to(device),
                    x4.to(device),
                    x5.to(device),
                )
                logits = logits / temperature
                probs = F.softmax(logits, -1)

        probs = probs.detach().cpu().numpy().astype(np.float32)
        eeg_ids = np.asarray(eeg_ids, dtype=np.int64)
        eeg_id_list.append(eeg_ids)
        prob_list.append(probs)

    eeg_id_all = np.concatenate(eeg_id_list, axis=0)
    prob_all = np.concatenate(prob_list, axis=0)

    prob_all = _mix_with_prior(prob_all, global_prior, alpha=0.09)

    sub = pd.DataFrame(
        {
            "eeg_id": eeg_id_all.astype(np.int64),
            "seizure_vote": prob_all[:, 0],
            "lpd_vote": prob_all[:, 1],
            "gpd_vote": prob_all[:, 2],
            "lrda_vote": prob_all[:, 3],
            "grda_vote": prob_all[:, 4],
            "other_vote": prob_all[:, 5],
        }
    )

    sub = sample[["eeg_id"]].merge(sub, on="eeg_id", how="left")

    sub[vote_cols] = sub[vote_cols].replace([np.inf, -np.inf], np.nan)
    sub[vote_cols] = sub[vote_cols].fillna(1.0 / 6.0)
    sub[vote_cols] = _safe_normalize_probs(sub[vote_cols].values)

    sub.to_csv("submission.csv", index=False)
    return sub


if domain == "train":
    lr = 6e-6
    epochs = 100

    train_dataset = TrainingDatasetEEG(df, train=True, is_test_random=is_test_random)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(batch_size, 4),
    )

    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = tc.optim.AdamW(model.parameters(), lr=lr)
    scaler = tc.cuda.amp.GradScaler(enabled=(device == "cuda"))
    scheduler = tc.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    train(model, optimizer, criterion, scaler, scheduler, epochs)
else:
    submission = submit(model)
    print("Wrote submission.csv with shape:", submission.shape)
    print(submission.head())
