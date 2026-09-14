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

0.4548077791420204

# 6. Current score

1.43004

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.48867) has done: 'Your code doesn’t yield a Kaggle score mainly because the test-time pipeline can be invalid/unstable: it may silently run with random weights (if no checkpoint is found) and/or produce non-deterministic results, and it also risks submission row misalignment if `eeg_id` dtype/order differs. I make minimal changes to (1) always generate a valid `submission.csv` with correct column order and row count aligned to `sample_submission.csv`, (2) make inference deterministic and safe on CPU/GPU, and (3) slightly improve the “no checkpoint found” fallback by using the *consolidated per-eeg* normalized vote distribution (matching the competition target construction more closely than raw vote sums). Core model/feature logic and loss remain unchanged.'
- What this solution (achieved 1.48867) has done: 'Your current score (1.48867, lower-is-better) is far worse than the target (0.4548), so we should improve test-time predictions with minimal risk and without changing the model itself. The biggest likely issue is that inference is being done on a randomly-initialized checkpoint (or a weak/incorrectly loaded one), producing near-random probabilities; we make the “no/weak checkpoint” behavior much stronger by outputting a per-eeg aggregated label distribution from train.csv (a legitimate prior that matches the target construction). If a checkpoint does load, we still reduce KL risk by blending (mixing) the model probabilities with the learned prior (simple calibration toward the empirical label distribution), which typically improves KL vs. overconfident/shifted predictions. We keep architecture/feature extraction/training/loss unchanged, and we still output a valid submission aligned to `sample_submission.csv` with rows summing to 1.'
- What this solution (achieved 1.48867) has done: 'Your current score (1.48867, lower-is-better) is far from the target (0.4548), so we should improve probability calibration with the smallest safe change that preserves your model and feature pipeline. The biggest low-risk gain for KL on this competition is usually better-calibrated probabilities, so I add temperature scaling (fit on a small deterministic holdout split of consolidated per-eeg labels) and then apply it at test-time. I keep your architecture, data extraction, and loss unchanged, and I leave your existing “blend with prior” idea but make it adaptive based on whether calibration succeeded. I also ensure submission alignment is strictly by `sample_submission.csv` order and that every row sums to 1.'
- What this solution (achieved 1.48867) has done: 'Your current KL (1.48867, lower-is-better) is far above the target (0.4548), so the most reliable minimal improvement is to reduce overconfident/unstable model probabilities by anchoring them more strongly to a label distribution prior learned from train.csv. I keep your model, feature pipeline, and loss unchanged, and only adjust test-time post-processing: (1) compute a stronger per-eeg-id prior table and a global prior, (2) blend model predictions with the prior using a higher alpha and use the per-eeg prior when available, and (3) make the temperature-fit dataset construction correct (include the needed `spectrogram_id`/`eeg_id`/`patient_id` columns) so calibration is less likely to fail silently. These are low-risk changes that typically reduce KL substantially when a checkpoint is weak/mismatched or predictions are miscalibrated. The script still writes a valid `submission.csv` aligned to `sample_submission.csv` with rows summing to 1.'
- What this solution (achieved 1.48867) has done: 'Your current KL (1.48867; lower is better) is far worse than the target (0.4548), so we should improve predictions with the smallest safe change that preserves your model and pipeline. The biggest likely failure mode is that the checkpoint is missing/weak, making model outputs close to random; in that case, a strong prior from `train.csv` is legitimately much better for KL. I keep your architecture/training/inference intact and only adjust the submission-time blending to lean more on the empirical per-`eeg_id` distribution (when available) and fall back to a global prior otherwise. I also make the blend strength adaptive: if the model looks untrustworthy (high-entropy/near-uniform), we smooth more toward the prior; if it’s confident, we smooth less—this typically reduces KL without changing core logic.'
- What this solution (achieved 1.48867) has done: 'Your current KL (1.48867; lower is better) is far above the target (0.4548), so we need a safer, stronger improvement without changing your model/feature pipeline. The smallest high-impact fix is to avoid relying on a potentially missing/weak checkpoint at test time by blending the model probabilities much more toward a robust empirical prior computed from train.csv (global prior), which typically reduces KL dramatically versus near-random outputs. I keep your architecture, data processing, and loss intact; changes are limited to (1) verifying checkpoint loading success more strictly, (2) computing a stable global prior from consolidated per-eeg normalized votes, and (3) using a stronger, fixed blend (and skipping temperature fitting when weights look untrusted) while still aligning exactly to `sample_submission.csv` and ensuring each row sums to 1. This should move the score substantially toward the target band with minimal risk and within Kaggle constraints.'
- What this solution (achieved 1.48867) has done: 'We move your KL score down toward the 0.4548 target by making the test-time probabilities less noisy/overconfident while keeping your model and feature pipeline intact. The biggest low-risk improvement is to use the per-`eeg_id` empirical label distribution from `train.csv` as a prior and blend it into model predictions (and use it directly when the model checkpoint isn’t reliable). We also fix a key issue: your current blending ignores the available per-`eeg_id` distribution table you already computed (`train_eeg_dist`), so we actually use it when possible and fall back to the global prior otherwise. Finally, we keep strict submission alignment to `sample_submission.csv` order and ensure rows sum to 1.'
- What this solution (achieved 1.48867) has done: 'Your current KL (1.48867; lower is better) is far above the target (0.4548), so we should move predictions closer to a robust baseline with minimal risk. The smallest high-impact change is to stop relying on a likely-weak/mismatched checkpoint by (a) using the per-`eeg_id` prior whenever available, and (b) making the prior blending adaptive so untrustworthy (high-entropy) model outputs get smoothed much more strongly toward the prior. This preserves your model, feature extraction, and loss exactly; it only changes test-time post-processing to reduce KL. I also make the per-`eeg_id` lookup safe (avoid KeyErrors) and keep strict alignment to `sample_submission.csv` with row-wise normalization.'
- What this solution (achieved 1.65365) has done: 'Your current KL (1.48867; lower is better) is far above the 0.4548 target, so the smallest reliable way to move closer is to reduce dependence on potentially weak/mismatched model outputs and instead lean more on a robust empirical label prior from `train.csv`. I keep your model/feature pipeline untouched and only change test-time post-processing: use a single strong, fixed blend weight toward the per-`eeg_id`/global prior (rather than the current entropy-based blending which can over-smooth), and ensure the prior table is built from the same filtered/consolidated construction you use for training (vote-sum>6 and per-eeg normalization). I also make checkpoint “trust” stricter (require a higher match ratio) so random/partial loads don’t poison predictions. The script still runs end-to-end and writes a valid `submission.csv` aligned to `sample_submission.csv`, with each row summing to 1.'
- What this solution (achieved 1.57582) has done: 'Your current KL (1.65365; lower is better) is much worse than the target (0.4548), so the safest way to move closer without touching your model/feature pipeline is to strengthen the test-time prior and reduce reliance on potentially miscalibrated logits. I keep your architecture, feature extraction, and loss unchanged, and only adjust submission-time post-processing: (1) build a more metric-aligned prior using Dirichlet/Laplace smoothing on the consolidated per-eeg vote sums, and (2) increase blending toward the per-eeg/global prior with a small adaptive rule so we don’t overreact to noisy model outputs. I also keep strict row alignment to `sample_submission.csv` and ensure every row is normalized to sum to 1. These are minimal, low-risk changes that typically reduce KL substantially when the checkpoint is weak/mismatched.'
- What this solution (achieved 1.48875) has done: 'Your current KL (1.57582; lower is better) is far worse than the 0.4548 target, so the most reliable minimal improvement is to reduce overconfident/noisy model probabilities at submission time without touching the model, features, or training loop. I (1) compute a metric-aligned global prior and per-eeg prior from train.csv using stronger Dirichlet smoothing (reduces extreme probabilities that hurt KL), and (2) switch from the current very-heavy prior blend (alpha≈0.90–0.99) to a more moderate, confidence-adaptive blend (typically improves vs. over-smoothing to the prior). I also (3) ensure deterministic, correctly aligned submission by aggregating duplicate test eeg_ids (if any) and then merging back to `sample_submission.csv` order with strict row-wise normalization.'
- What this solution (achieved 1.43004) has done: 'We keep your entire model/feature/training code unchanged and only adjust the submission-time probability post-processing to legitimately reduce KL (lower-is-better) toward your target. The current score suggests the model outputs are still not reliable/calibrated, so we (1) make the empirical per-eeg/global prior stronger and more metric-aligned via Dirichlet smoothing + mixture-with-uniform, and (2) blend predictions toward this prior with a slightly higher, stable alpha (instead of entropy-adaptive behavior that can under-smooth noisy logits). We also ensure temperature scaling is used only when weights are reliable and we slightly widen the allowed temperature range to avoid overconfident softmax that hurts KL. Submission alignment to `sample_submission.csv` order and per-row normalization are preserved.'

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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.signal import butter, lfilter

from tqdm import tqdm



## === cell 1
path = "/kaggle/input/hms-harmful-brain-activity-classification"
batch_size = 20
device = "cuda" if tc.cuda.is_available() else "cpu"
domain = "test"  # "train" or "test"

read_all_eegs = False
read_all_specs = False
read_all_l2i = False
is_augument = False
is_test_random = True

SEED = 42
np.random.seed(SEED)
tc.manual_seed(SEED)
if tc.cuda.is_available():
    tc.cuda.manual_seed_all(SEED)

tc.backends.cudnn.deterministic = True
tc.backends.cudnn.benchmark = False
try:
    tc.use_deterministic_algorithms(True)
except Exception:
    pass


def _seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    tc.manual_seed(worker_seed)


_dataloader_generator = tc.Generator()
_dataloader_generator.manual_seed(SEED)



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
    df = train_df

print("df shape:", df.shape)
print(df.head())




## === cell 3
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


def get_eeg_from_parquet_by_id(eeg_id, is_train=True):
    eeg_dir = "train_eegs" if is_train else "test_eegs"
    eeg_path = os.path.join(path, eeg_dir, f"{int(eeg_id)}.parquet")
    eeg = pd.read_parquet(eeg_path).iloc[0:10_000, :]
    eeg = eeg.values.T
    return eeg


def get_spec_from_parquet_by_ids(spec_id, spec_offset_seconds=0, is_train=True):
    spec_dir = "train_spectrograms" if is_train else "test_spectrograms"
    spec_path = os.path.join(path, spec_dir, f"{int(spec_id)}.parquet")
    x = pd.read_parquet(spec_path)
    if is_train:
        spec_offset = int(spec_offset_seconds // 2)
    else:
        spec_offset = 0
    x = (
        x.values[spec_offset : 300 + spec_offset, 1:]
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
    chans = np.arange(C)[:, None]

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
    class TrainingDatasetEEG_All(Dataset):
        def __init__(self, metadata):
            self.metadata = metadata
            self.eeg_ids = np.array(
                sorted(metadata["eeg_id"].unique()), dtype=np.int64
            ).squeeze()

        def __len__(self):
            return self.eeg_ids.shape[0]

        def __getitem__(self, idx):
            eeg_id = self.eeg_ids[idx]
            eeg = get_eeg_from_parquet_by_id(eeg_id, is_train=True)

            is_corrupted = 0
            for i in range(eeg.shape[0]):
                nans = np.isnan(eeg[i]).sum()
                if nans / 1e4 > 0.7:
                    is_corrupted = 1

            eeg = np.nan_to_num(eeg, nan=0).astype(np.float32)
            eeg = tc.from_numpy(eeg).type(tc.float32)
            return eeg, is_corrupted

    bs = 4
    train_dataset = TrainingDatasetEEG_All(df)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=4,
        worker_init_fn=_seed_worker,
        generator=_dataloader_generator,
        pin_memory=tc.cuda.is_available(),
        persistent_workers=True if 4 > 0 else False,
    )

    all_eegs = np.zeros((df["eeg_id"].unique().shape[0], 20, 10000), dtype=np.float32)
    all_is_corrupted_list = np.zeros(
        (df["eeg_id"].unique().shape[0],), dtype=np.float32
    )
    j = 0
    for _, (eegs, is_corrupted_list) in enumerate(tqdm(train_dataloader), 0):
        for eeg, is_corrupted in zip(eegs, is_corrupted_list):
            all_eegs[j] = eeg.numpy()
            all_is_corrupted_list[j] = float(is_corrupted)
            j += 1
    return all_eegs, all_is_corrupted_list


def get_spec_frame():
    class TrainingDatasetSpec_All(Dataset):
        def __init__(self, metadata):
            self.metadata = metadata
            self.eeg_ids = np.array(
                sorted(metadata["eeg_id"].unique()), dtype=np.int64
            ).squeeze()

        def __len__(self):
            return self.eeg_ids.shape[0]

        def __getitem__(self, idx):
            eeg_id = self.eeg_ids[idx]
            occ = self.metadata.loc[self.metadata["eeg_id"] == eeg_id, :]
            spec_id = occ.loc[:, "spectrogram_id"].values[0]
            spec_offset_seconds = occ.loc[:, "spectrogram_label_offset_seconds"].values[
                0
            ]
            spec = get_spec_from_parquet_by_ids(
                spec_id, spec_offset_seconds, is_train=True
            )
            spec = process_spec(spec)
            return spec

    bs = 4
    train_dataset = TrainingDatasetSpec_All(df)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=4,
        worker_init_fn=_seed_worker,
        generator=_dataloader_generator,
        pin_memory=tc.cuda.is_available(),
        persistent_workers=True if 4 > 0 else False,
    )

    all_specs = np.zeros(
        (df["eeg_id"].unique().shape[0], 4, 100, 300), dtype=np.float32
    )
    j = 0
    for _, specs in enumerate(tqdm(train_dataloader), 0):
        for spec in specs:
            all_specs[j] = spec.numpy() if isinstance(spec, tc.Tensor) else spec
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

    def _eeg2img_by_id(eeg_id):
        eeg = get_eeg_from_parquet_by_id(eeg_id, is_train=True)

        x2_ = eeg[cen_ilocs, 4000:6000]
        x2_ = np.split(x2_, 5, axis=0)
        x2_ = np.concatenate([x2_[i] - x2_[i + 1] for i in range(4)], 0)
        x2_ = x2_[:, [0, 1, 4, 5]].reshape(4 * 4, -1)
        x2 = butter_lowpass_filter(x2_.T, cutoff_freq=40).T
        x2 = np.nan_to_num(x2, nan=0)
        x2 = np.clip(x2, -1024, 1024)
        imgs = line2img(x2).astype(np.float32)
        imgs = (imgs - 0.5) / 0.5
        return imgs

    return _eeg2img_by_id


def get_eegimg_frame():
    func = eeg2img()
    eeg_ids = np.array(sorted(df["eeg_id"].unique()), dtype=np.int64).squeeze()
    all_imgs = np.zeros((len(eeg_ids), 16, 64, 512), dtype=np.float32)
    for j, eeg_id in enumerate(tqdm(eeg_ids)):
        all_imgs[j] = func(eeg_id)
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
                [eeg_id for eeg_id, _ in zip(neeg_ids, all_is_corrupted_list)],
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
                A.GaussNoise(std_range=(0.04, 0.2), p=0.2),
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
        eeg_id = int(self.eeg_ids[idx])
        occurances_df = self.metadata.loc[self.metadata["eeg_id"] == eeg_id, :]

        if read_all_eegs and domain == "train":
            all_ids = np.array(
                sorted(self.metadata["eeg_id"].unique()), dtype=np.int64
            ).squeeze()
            pos = int(np.where(all_ids == eeg_id)[0][0])
            eeg = all_eegs[pos].copy()
        else:
            eeg = get_eeg_from_parquet_by_id(eeg_id, is_train=True)

            for i in range(eeg.shape[0]):
                nans = np.isnan(eeg[i]).sum()
                if nans / 1e4 > 0.7:
                    nidx = idx + 1
                    if nidx >= self.__len__():
                        nidx = 0
                    return self.__getitem__(nidx)

            eeg = np.nan_to_num(eeg, nan=0)

        cutoff_freq = 40
        gn = 0.0
        direction = 1
        polarity = 1.0

        if is_augument and self.train:
            if np.random.rand() > 0.8:
                cutoff_freq = int(np.random.choice(list(range(16, 30, 4))))

            if np.random.rand() > 0.8:
                gnmx = eeg.mean(1)[:, None] * float(
                    np.random.choice((0.025, 0.05, 0.1))
                )
                pol = np.random.choice((-1.0, 1.0), size=eeg[0].shape)[None, :]
                gn = pol * np.random.random(eeg[0].shape)[None, :] * gnmx

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
            all_ids = np.array(
                sorted(self.metadata["eeg_id"].unique()), dtype=np.int64
            ).squeeze()
            pos = int(np.where(all_ids == eeg_id)[0][0])
            x4 = all_specs[pos]
        else:
            spec_id = int(occurances_df.loc[:, "spectrogram_id"].values[0])
            spec_offset_seconds = float(
                occurances_df.loc[:, "spectrogram_label_offset_seconds"].values[0]
            )
            x4 = get_spec_from_parquet_by_ids(
                spec_id, spec_offset_seconds, is_train=True
            )
            x4 = process_spec(x4)

        if read_all_l2i and domain == "train":
            all_ids = np.array(
                sorted(self.metadata["eeg_id"].unique()), dtype=np.int64
            ).squeeze()
            pos = int(np.where(all_ids == eeg_id)[0][0])
            x5 = all_imgs[pos]
        else:
            x5 = x2[:, 4000:6000].reshape(4, 6, -1)[:, [0, 1, 4, 5]].reshape(4 * 4, -1)
            x5 = line2img(x5).astype(np.float32)

        if is_augument and self.train:
            for i in range(16):
                augmented = self.transform(image=x5[i])
                x5[i] = augmented["image"]

        x5 = (x5 - 0.5) / 0.5

        targets = occurances_df.iloc[:, -6:].values.squeeze()

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
        eeg = get_eeg_from_parquet_by_id(eeg_id, is_train=False)
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
        x4 = get_spec_from_parquet_by_ids(
            spec_id, spec_offset_seconds=0, is_train=False
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
import torch

p_dropout = 0.3 if domain == "train" else 0.0


class BBlock(nn.Module):
    def __init__(self, o, g, s=2, h=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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
        c_final = self.c_final(c_norm)
        return c_final


class BBlock2(nn.Module):
    def __init__(
        self, o, g, s=2, h=None, is_strided=False, stride=None, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
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
        c_final = self.c_final(c_norm)
        return c_final


class TBlock(nn.Module):
    def __init__(self, o, g, s=2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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
        c_final = self.c_final(c_norm)
        return c_final


class EEGFeatureExtractor(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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
        d2 = self.d2(d1.mean(-1))
        return d2

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
        e2 = self.e2(e1.mean(-1))
        return e2

    def norm(self, x, axis=1):
        mx = x.max(axis, keepdims=True)[0]
        mn = x.min(axis, keepdims=True)[0]
        n = (x - mn) / (mx - mn + 1e-5)
        return n

    def montage_features_extractor2(self, x):
        with tc.no_grad():
            x = self.norm(x, 2)
        g1 = tc.cat([self.g1(x[:, i, :].unsqueeze(1)) for i in range(x.shape[1])], 1)
        g2 = self.g2(g1.mean(-1))
        return g2

    def gr_montage_features_extractor(self, x):
        N, S, L = x.shape
        x = x.view(N, 4, 6, L)
        f1 = tc.cat([self.f1(x[:, :, i, :]).mean(-1) for i in range(6)], 1)
        f2 = self.f2(f1)
        return f2

    def kagg_spec_feature_extractor(self, x):
        h1 = tc.cat(
            [self.h1(x[:, i, ...].unsqueeze(1)).mean((-2, -1)) for i in range(4)], 1
        )
        h2 = self.h2(h1)
        return h2

    def eeg_img_feature_extractor(self, x):
        i1 = tc.cat(
            [self.i1(x[:, i, ...].unsqueeze(1)).mean((-2, -1)) for i in range(16)], 1
        )
        i2 = self.i2(i1)
        return i2

    def eeg_img_feature_extractor2(self, x):
        with tc.no_grad():
            N, C, H, W = x.shape
            x = x.view(N, 4, 4, H, W)
        j1 = tc.cat([self.j1(x[:, :, i, ...]) for i in range(4)], 1)
        j2 = self.j2(j1)
        return j2

    def forward(self, x1, x2, x3, x4, x5):
        h1 = self.total_feature_extractor(x1)
        h2 = self.ind_feature_extractor(x1)
        h3 = self.montage_features_extractor(x2)
        h4 = self.gr_montage_features_extractor(x2)
        h5 = self.montage_features_extractor2(x3)
        h6 = self.kagg_spec_feature_extractor(x4)
        h7 = self.eeg_img_feature_extractor(x5)
        h8 = self.eeg_img_feature_extractor2(x5)
        h = tc.cat([h1, h2, h3, h4, h5, h6, h7, h8], 1)
        return h


class EEGBasedClassifier(nn.Module):
    def __init__(
        self, load_weights=True, load_out_weights=True, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
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

        def _find_checkpoint():
            candidates = []
            preferred = "/kaggle/input/hms-dataset/HMS_EEG_MODEL0.pth"
            if os.path.exists(preferred):
                return preferred
            for root, _, files in os.walk("/kaggle/input"):
                for fn in files:
                    if fn.lower().endswith(".pth") and (
                        "hms" in fn.lower() or "eeg" in fn.lower()
                    ):
                        candidates.append(os.path.join(root, fn))
            candidates = sorted(set(candidates))
            if len(candidates) > 0:
                return candidates[0]
            return None

        self.loaded_any_weights = False

        self.load_out_weights = load_out_weights
        if load_weights:
            ckpt_path = _find_checkpoint()
            if ckpt_path is not None and os.path.exists(ckpt_path):
                print(f"Loading checkpoint: {ckpt_path}")
                ckpt_obj = tc.load(ckpt_path, map_location=tc.device(device))
                checkpoint = (
                    ckpt_obj["model"]
                    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj
                    else ckpt_obj
                )

                matched = 0
                total = 0

                new_weights = self.feature_extractor.state_dict()
                for key in checkpoint.keys():
                    if "out" in key:
                        continue
                    nkey = (
                        ".".join(key.split(".")[1:])
                        if key.startswith("module.") or key.startswith("model.")
                        else key
                    )
                    if nkey in new_weights:
                        total += 1
                        if new_weights[nkey].shape == checkpoint[key].shape:
                            new_weights[nkey] = checkpoint[key]
                            matched += 1
                self.feature_extractor.load_state_dict(new_weights)

                if load_out_weights:
                    new_out = self.out.state_dict()
                    for key in checkpoint.keys():
                        if "out" in key:
                            nkey = (
                                ".".join(key.split(".")[1:])
                                if key.startswith("module.") or key.startswith("model.")
                                else key
                            )
                            if (
                                nkey in new_out
                                and new_out[nkey].shape == checkpoint[key].shape
                            ):
                                new_out[nkey] = checkpoint[key]
                    self.out.load_state_dict(new_out)

                match_ratio = matched / max(total, 1)
                self.loaded_any_weights = (matched >= 60) and (match_ratio >= 0.60)
                print(
                    f"Checkpoint tensor match count: {matched} (keys considered: {total}, ratio={match_ratio:.3f}). loaded_any_weights={self.loaded_any_weights}"
                )
            else:
                print(
                    "Warning: no checkpoint found in /kaggle/input. Running with randomly initialized weights."
                )

    def forward(self, x1, x2, x3, x4, x5):
        if self.load_out_weights:
            h = self.feature_extractor(x1, x2, x3, x4, x5)
        else:
            with tc.no_grad():
                h = self.feature_extractor(x1, x2, x3, x4, x5)
        out = self.out(h)
        return out




## === cell 8
model = EEGBasedClassifier(load_weights=True, load_out_weights=True).to(device)
print(
    "Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad)
)
print("Loaded any checkpoint weights:", getattr(model, "loaded_any_weights", False))




## === cell 9
def test(model, criterion, ret=False):
    test_dataset = TrainingDatasetEEG(df, train=False, is_test_random=is_test_random)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=max(batch_size, 128),
        shuffle=False,
        num_workers=min(batch_size, 4),
        pin_memory=tc.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=_dataloader_generator,
        persistent_workers=True if min(batch_size, 4) > 0 else False,
    )
    print("Starting Testing")
    model.eval()
    running_loss = 0.0
    ya = []
    yp = []
    for i, data in enumerate(test_dataloader, 0):
        (x1, x2, x3, x4, x5), targets = data
        ya.append(targets.squeeze())

        with tc.amp.autocast(device_type="cuda", enabled=tc.cuda.is_available()):
            with tc.no_grad():
                outputs = model(
                    x1.to(device),
                    x2.to(device),
                    x3.to(device),
                    x4.to(device),
                    x5.to(device),
                )
                outputs = F.softmax(outputs, -1)
                outputs = tc.log(outputs + 1e-5)
                loss = criterion(outputs, targets.to(device))
                yp.append(outputs.detach().cpu().numpy().squeeze())

        running_loss += loss.item()

    denom = max(i + 1, 1)
    print(f"Test Loss: {running_loss / denom:.6f}")
    print("Finished Testing\n")
    if ret:
        return np.concatenate(ya), np.concatenate(yp)


def train(model, optimizer, criterion, scaler, scheduler, epochs, train_dataloader):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loss_count = 0

        for i, data in enumerate(train_dataloader, 0):
            (x1, x2, x3, x4, x5), targets = data

            with tc.amp.autocast(device_type="cuda", enabled=tc.cuda.is_available()):
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
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            loss_count += 1
            if i % 100 == 99:
                print(
                    f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / loss_count:.6f}"
                )
                running_loss = 0.0
                loss_count = 0

        if scheduler is not None:
            scheduler.step()

        test(model, criterion)

    print("Finished Training")


def _train_eeg_distribution_table(path_, smooth_alpha=1.0):
    train_meta = pd.read_csv(os.path.join(path_, "train.csv"))
    vote_cols = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    train_meta = train_meta.loc[train_meta[vote_cols].sum(axis=1) > 6].copy()
    grp = train_meta.groupby("eeg_id")[vote_cols].sum().astype(np.float64)

    grp = grp + float(smooth_alpha)
    grp = grp.div(grp.sum(axis=1), axis=0)
    grp = grp.fillna(0.0).astype(np.float64)
    return grp, vote_cols


def _label_prior_from_train_csv(path_, smooth_alpha=1.0):
    grp, _ = _train_eeg_distribution_table(path_, smooth_alpha=smooth_alpha)
    prior = grp.mean(axis=0).values.astype(np.float64)
    prior = np.clip(prior, 1e-12, None)
    prior = prior / prior.sum()
    return prior


def _fit_temperature_on_holdout(model, path_, device_, max_holdout=1024):
    vote_cols = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    train_meta = pd.read_csv(os.path.join(path_, "train.csv"))
    train_meta = train_meta.loc[train_meta[vote_cols].sum(axis=1) > 6].copy()
    grp = train_meta.groupby("eeg_id")[vote_cols].sum()
    grp = grp.div(grp.sum(axis=1), axis=0).fillna(0.0)

    eeg_ids = grp.index.values.astype(np.int64)
    rng = np.random.default_rng(SEED)
    rng.shuffle(eeg_ids)
    holdout_ids = eeg_ids[: min(max_holdout, len(eeg_ids))]

    meta_first = (
        train_meta.groupby("eeg_id")[["spectrogram_id", "patient_id"]]
        .first()
        .loc[holdout_ids]
    )
    holdout_meta = meta_first.reset_index()[
        ["spectrogram_id", "eeg_id", "patient_id"]
    ].copy()
    y = grp.loc[holdout_ids].values.astype(np.float64)

    dataset = DatasetEEG(holdout_meta)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=tc.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=_dataloader_generator,
        persistent_workers=True,
    )

    model.eval()
    logits_list = []
    for x1, x2, x3, x4, x5, _eeg_id in dataloader:
        with tc.no_grad():
            out = model(
                x1.to(device_),
                x2.to(device_),
                x3.to(device_),
                x4.to(device_),
                x5.to(device_),
            )
        logits_list.append(out.detach().cpu())
    logits = tc.cat(logits_list, dim=0).to(tc.float32)  # (N,6)
    y_t = tc.from_numpy(y).to(tc.float32)

    logT = tc.tensor([0.0], dtype=tc.float32, requires_grad=True)  # T=exp(logT)
    opt = tc.optim.LBFGS([logT], lr=0.5, max_iter=30, line_search_fn="strong_wolfe")
    crit = nn.KLDivLoss(reduction="batchmean")

    def closure():
        opt.zero_grad(set_to_none=True)
        T = tc.exp(logT).clamp(0.05, 10.0)
        lsm = F.log_softmax(logits / T, dim=1)
        loss = crit(lsm, y_t)
        loss.backward()
        return loss

    try:
        opt.step(closure)
        T = float(tc.exp(logT).clamp(0.05, 10.0).item())
        return T
    except Exception as e:
        print("Temperature fitting failed, falling back to T=1.0. Error:", repr(e))
        return 1.0


def _safe_per_eeg_prior(
    train_eeg_dist: pd.DataFrame, eeg_ids: np.ndarray, global_prior: np.ndarray
):
    idx = train_eeg_dist.index.values.astype(np.int64)
    mask = np.isin(eeg_ids, idx)
    per = np.tile(global_prior[None, :], (len(eeg_ids), 1)).astype(np.float64)
    if mask.any():
        per_vals = train_eeg_dist.reindex(eeg_ids[mask]).values.astype(np.float64)
        per[mask] = per_vals
    per = np.clip(per, 1e-12, 1.0)
    per = per / per.sum(axis=1, keepdims=True)
    return per


def submit(model):
    test_meta = pd.read_csv(os.path.join(path, "test.csv"))
    sample = pd.read_csv(os.path.join(path, "sample_submission.csv"))

    vote_cols = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

    smooth_alpha = 12.0
    uniform_mix = 0.05

    test_meta = test_meta.copy()
    test_meta["eeg_id"] = test_meta["eeg_id"].astype(np.int64)
    sample["eeg_id"] = sample["eeg_id"].astype(np.int64)

    train_eeg_dist, _ = _train_eeg_distribution_table(path, smooth_alpha=smooth_alpha)
    prior = _label_prior_from_train_csv(path, smooth_alpha=smooth_alpha).astype(
        np.float64
    )
    prior = (1.0 - uniform_mix) * prior + uniform_mix * (
        np.ones_like(prior) / len(prior)
    )
    prior = np.clip(prior, 1e-12, None)
    prior = prior / prior.sum()

    if not getattr(model, "loaded_any_weights", False):
        eeg_ids = sample["eeg_id"].values.astype(np.int64)
        base = _safe_per_eeg_prior(train_eeg_dist, eeg_ids, prior)
        sub = pd.DataFrame({"eeg_id": eeg_ids})
        for j, c in enumerate(vote_cols):
            sub[c] = base[:, j]
        sub[vote_cols] = sub[vote_cols].div(sub[vote_cols].sum(axis=1), axis=0)
        print(
            "Completed Submission Inference (smoothed per-eeg/global prior fallback; no reliable checkpoint)"
        )
        return sub[["eeg_id"] + vote_cols]

    T = _fit_temperature_on_holdout(model, path, device, max_holdout=512)
    T = float(np.clip(T, 0.8, 3.0))
    print("Fitted/clipped temperature T =", T)

    test_meta_u = test_meta.drop_duplicates(
        subset=["eeg_id"], keep="first"
    ).reset_index(drop=True)

    dataset = DatasetEEG(test_meta_u)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=min(batch_size, 4),
        pin_memory=tc.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=_dataloader_generator,
        persistent_workers=True if min(batch_size, 4) > 0 else False,
    )
    print("Starting Submission Inference")
    model.eval()

    eeg_ids_all = []
    preds_all = []

    for x1, x2, x3, x4, x5, eeg_ids in tqdm(dataloader):
        with tc.amp.autocast(device_type="cuda", enabled=tc.cuda.is_available()):
            with tc.no_grad():
                outputs = model(
                    x1.to(device),
                    x2.to(device),
                    x3.to(device),
                    x4.to(device),
                    x5.to(device),
                )
                outputs = F.softmax(outputs / float(T), -1).detach().cpu().numpy()
        eeg_ids_all.append(eeg_ids.numpy())
        preds_all.append(outputs)

    eeg_ids_all = np.concatenate(eeg_ids_all).astype(np.int64)
    preds_all = np.concatenate(preds_all).astype(np.float64)

    preds_all = np.clip(preds_all, 1e-12, 1.0)
    preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)

    per_prior = _safe_per_eeg_prior(train_eeg_dist, eeg_ids_all, prior)

    alpha = 0.80
    preds_all = (1.0 - alpha) * preds_all + alpha * per_prior

    preds_all = (1.0 - uniform_mix) * preds_all + uniform_mix * (
        np.ones_like(preds_all) / preds_all.shape[1]
    )

    preds_all = np.clip(preds_all, 1e-12, 1.0)
    preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)

    pred_df = pd.DataFrame(
        {
            "eeg_id": eeg_ids_all,
            "seizure_vote": preds_all[:, 0],
            "lpd_vote": preds_all[:, 1],
            "gpd_vote": preds_all[:, 2],
            "lrda_vote": preds_all[:, 3],
            "grda_vote": preds_all[:, 4],
            "other_vote": preds_all[:, 5],
        }
    )

    sub = sample[["eeg_id"]].merge(pred_df, on="eeg_id", how="left")

    missing = sub[vote_cols].isna().any(axis=1)
    if missing.any():
        sub.loc[missing, vote_cols] = prior[None, :]

    sub[vote_cols] = sub[vote_cols].astype(np.float64)
    sub[vote_cols] = sub[vote_cols].div(sub[vote_cols].sum(axis=1), axis=0)

    print("Completed Submission Inference")
    return sub[["eeg_id"] + vote_cols]


if domain == "train":
    lr = 1e-6
    epochs = 100

    train_dataset = TrainingDatasetEEG(df, train=True, is_test_random=is_test_random)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(batch_size, 4),
        pin_memory=tc.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=_dataloader_generator,
        persistent_workers=True if min(batch_size, 4) > 0 else False,
    )

    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = tc.optim.AdamW(model.parameters(), lr=lr)
    scaler = tc.amp.GradScaler(enabled=tc.cuda.is_available())
    scheduler = tc.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    train(model, optimizer, criterion, scaler, scheduler, epochs, train_dataloader)
else:
    submission = submit(model)
    submission.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission.shape)
    print(submission.head())
