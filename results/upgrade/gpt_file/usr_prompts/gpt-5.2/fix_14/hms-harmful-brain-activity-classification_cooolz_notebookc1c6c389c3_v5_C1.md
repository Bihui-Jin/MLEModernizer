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

0.494549282774905

# 6. Current score

1.40225

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime blocker by making the script robust to missing external weight files: if the listed weights directory doesn’t exist, it fall back to a valid “safe baseline” prediction (uniform probabilities) so a submission is always produced. I also remove a major hidden slowdown/bug (`print(dp)` inside `__getitem__`) and correct a spectrogram slicing bug (the region offset `r` was never updated), while keeping the model and inference logic intact when weights are available. Finally, I guarantee the submission has the correct columns, correct shape, and each row sums to 1 (required to avoid submission failure). These changes are score-neutral when weights are present, and when weights are absent they at least yield a valid submission instead of crashing.'
- What this solution (achieved 1.4076) has done: 'You’re currently far worse than the target (lower-is-better), so we should make a small, safe change that legitimately improves KL without changing the model or training: calibrate the already-softmaxed predictions with mild label-smoothing and a light blend toward the class prior computed from `train.csv` vote distributions. This typically reduces overconfident probabilities (a common KL failure mode) while staying within identical inference semantics (still just post-processing probabilities). I also fix the only likely hidden correctness issue: ensure the spectrogram parquet is read from the same root as `CFG["data"]` (some notebooks mount only one of the duplicate paths), but without changing the feature extraction itself. The submission format, ordering, and row-sum constraints remain guaranteed.'
- What this solution (achieved 1.405) has done: 'Your current score is much worse than the target (lower-is-better), so the safest way to move KL down without changing the model is to reduce overconfidence and improve calibration in post-processing. I keep the same model/inference, but (1) compute the class prior correctly from vote *probabilities* (per-row normalized) instead of raw vote totals, and (2) apply a temperature transform to the already-softmaxed outputs (probability sharpening/flattening) plus a slightly stronger prior blend—both are legitimate probability calibration steps for KL. I also make the data root explicit to avoid accidentally pointing `base_dir` at the file directory instead of the dataset root, which can silently degrade predictions if the wrong path is used. The submission format/row order/row-sum constraints remain guaranteed and a `submission.csv` is always written.'
- What this solution (achieved 1.40231) has done: 'You’re far worse than the target (lower KL is better), so we make the smallest legitimate change that tends to reduce KL without changing the model or inference loop: slightly stronger probability calibration to reduce overconfidence. Specifically, we keep your exact model ensemble output but (1) set temperature a bit higher (flatter distributions) and (2) increase the blend toward the global class prior computed from per-row vote probabilities (not raw totals), which usually improves KL on HMS. We also compute the prior from the canonical dataset path (not via `CFG["data"].replace(...)`) to avoid silently missing `train.csv` if paths differ. Submission formatting and row-sum guarantees remain unchanged.'
- What this solution (achieved 1.40012) has done: 'Your current KL (1.40231, lower-is-better) is still far above the target (0.4945), so we should make a minimal, legitimate change that tends to reduce KL without altering your model or inference loop. The safest lever here is post-processing calibration: slightly flatten the per-row distributions more (temperature) and blend a bit more toward the global class prior from train vote *probabilities*, which commonly reduces overconfidence penalties in KL. I keep the same data loading/model/ensemble logic and only adjust the calibration hyperparameters while preserving the “rows sum to 1” submission constraint. I also make the temperature/prior blend explicit constants so you can easily nudge them if needed after one more Kaggle submission.'
- What this solution (achieved 1.3984) has done: 'Your current KL (1.40012, lower-is-better) is still far above the target (0.4945), so we should make a small, legitimate calibration change that tends to reduce KL without touching the model, dataset, or inference loop. The safest knob here is stronger de-overconfidence: slightly increase the temperature (flatter probabilities) and slightly increase the blend toward the global class prior computed from per-row normalized vote probabilities. This keeps evaluation semantics identical (still produces valid probabilities summing to 1) and is low-risk because it only post-processes the already-softmaxed outputs. I keep all paths and submission formatting unchanged and only adjust the two calibration constants.'
- What this solution (achieved 1.3975) has done: 'Your current KL (1.3984, lower-is-better) is still far above the target (0.4945), so we should make the smallest legitimate change that tends to reduce KL without touching the model/inference loop: strengthen post-hoc probability calibration. I keep the exact ensemble predictions but (a) slightly increase the temperature to further reduce overconfidence and (b) slightly increase the blend toward the global class prior computed from per-row normalized train vote probabilities; both typically reduce KL on this competition when a model is miscalibrated. I also make the spectrogram reading more robust by trying the alternate dataset root path if the primary one isn’t mounted, which can otherwise silently harm predictions or crash. Submission formatting and the “rows sum to 1” constraint remain enforced.'
- What this solution (achieved 1.39675) has done: 'Your current KL (1.3975, lower-is-better) is still far above the target (0.4945), so we should keep the model/inference exactly the same and only adjust post-processing calibration, which is the smallest legitimate lever to reduce KL. I slightly increase the temperature (to further reduce overconfidence) and increase the blend toward the global class prior computed from normalized train vote probabilities (to pull extreme predictions toward realistic marginal frequencies). I also make the fallback spectrogram root consistent with the dataset root(s) already used elsewhere, to avoid silently reading from a wrong/nonexistent location. The submission format, ordering, and row-sum-to-1 constraint remain enforced exactly as before.'
- What this solution (achieved 1.39622) has done: 'Your current KL (1.39675, lower-is-better) is still far above the target (0.49455), so we should keep your model/inference exactly the same and only make the smallest, most reliable KL-improving adjustment: stronger post-hoc calibration to reduce overconfidence. Concretely, we (1) slightly increase the temperature to further flatten distributions and (2) increase the blend toward the global class prior computed from per-row normalized train vote probabilities; both typically reduce KL when predictions are poorly calibrated. We also keep all existing robustness (alternate dataset root for spectrograms/train.csv) and ensure the submission remains valid (correct columns, correct order, each row sums to 1). No training, architecture, feature extraction, or inference-loop logic is changed.'
- What this solution (achieved 1.39622) has done: 'Your KL is still far above the target (lower is better), and we’re already doing the safest kind of improvement (post-hoc probability calibration) without touching the model/inference loop. The minimal next step is to slightly strengthen calibration by (1) increasing the temperature a bit more to reduce overconfidence penalties, and (2) blending a bit more toward the global class prior (computed from per-row normalized train vote probabilities) which typically reduces KL when a model is miscalibrated. I keep all paths, data reading, model ensemble, and submission formatting identical, and only adjust the two calibration constants. This should move the score downward (better) while preserving the existing core logic.'
- What this solution (achieved 1.39632) has done: 'Your current KL (1.39622, lower-is-better) is still far above the target (0.49455), so we should only make the smallest, safest adjustment that can legitimately reduce KL without touching your model, data extraction, or inference loop. The best low-risk lever left is post-processing calibration: slightly stronger flattening (temperature) and a slightly stronger blend toward the global class prior (computed from per-row normalized vote probabilities), which typically reduces overconfidence penalties in KL. I keep everything else identical (paths, dataset, EfficientNet, ensemble averaging, softmax, submission formatting) and only nudge the two calibration constants. The submission still be valid (correct columns/order, 9850 rows, row sums = 1) and write `submission.csv`.'
- What this solution (achieved 1.39905) has done: 'Your current KL (1.39632, lower-is-better) is far above the target (0.49455), and the safest lever that doesn’t touch the model or inference loop is post-hoc probability calibration. Right now you’re blending extremely strongly toward the global prior (alpha=0.82), which can collapse useful signal and often hurts; we reduce that blend and instead rely a bit more on temperature flattening (still a pure probability transform). We also compute the prior slightly more robustly by averaging vote-probabilities grouped by `eeg_id` (so recordings with many overlapping windows don’t dominate the marginal), which typically improves calibration for this competition without changing any modeling logic. All paths, dataset reading, model ensemble, softmax, and submission formatting remain the same, and the script still always writes a valid `submission.csv` with row sums = 1.'
- What this solution (achieved 1.40225) has done: 'Your current KL (1.39905, lower-is-better) is still far above the target (0.49455), and we’re constrained to not touch the model/inference core, so the safest remaining lever is post-hoc probability calibration. I keep your exact ensemble predictions, but adjust the calibration to reduce the over-flattening from the very high temperature (which can wash out useful signal) while slightly increasing the prior blend (which usually stabilizes KL when predictions are noisy/miscalibrated). I also make the calibration constants explicit and keep all row-sum/format guarantees unchanged, so the script still runs end-to-end and always produces a valid `submission.csv`. No architecture, feature extraction, loss, or inference-loop logic is changed.'

# 9. Code solution

## === cell 0
import random
import cv2
import json
import numpy as np
import copy
import pandas as pd
import torch
import gc

import albumentations as A
import os
import librosa
import pickle
import timm
from tqdm import tqdm

import torch.nn as nn
from torch.utils.data import DataLoader



## === cell 1
CFG = {
    "batch_size": 32,
    "num_worker": 4,
    "data": "/kaggle/input/hms-harmful-brain-activity-classification/test.csv",
    "weights": [
        "/kaggle/input/hms-baseline/fold0_epoch_4_val_loss_0.589956.pth",
        "/kaggle/input/hms-baseline/fold1_epoch_4_val_loss_0.519593.pth",
        "/kaggle/input/hms-baseline/fold2_epoch_4_val_loss_0.499242.pth",
        "/kaggle/input/hms-baseline/fold3_epoch_4_val_loss_0.584708.pth",
        "/kaggle/input/hms-baseline/fold4_epoch_4_val_loss_0.655207.pth",
    ],
}

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]




## === cell 2
class AlaskaDataIter:
    def __init__(self, df, training_flag=False, shuffle=False, base_dir=None):
        self.training_flag = training_flag
        self.shuffle = shuffle
        self.raw_data_set_size = None  # decided by self.parse_file
        self.df = df

        self.train_trans = A.Compose([A.HorizontalFlip(p=0.5)])

        TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
        self.TARS2 = {x: y for y, x in TARS.items()}

        self.eeg_nms = [
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

        if base_dir is None:
            base_dir = "/kaggle/input/hms-harmful-brain-activity-classification"
        self.base_dir = base_dir

    def __getitem__(self, item):
        return self.single_map_func(self.df.iloc[item], self.training_flag)

    def __len__(self):
        return len(self.df)

    def single_map_func(self, dp, is_training):
        """Data loading/augmentation function.
        Keep core logic intact; just robust path fallback + correct region slicing.
        """
        EEG = False
        if EEG:
            eeg_path = (
                "../hms-harmful-brain-activity-classification/test_eegs/%s.parquet"
                % (dp["eeg_id"])
            )
            eeg = pd.read_parquet(eeg_path)

            eeg_offside_min = int(dp["eeg_min"])
            eeg_offside_max = int(dp["eeg_max"])
            offset = (eeg_offside_min + eeg_offside_max) // 2
            eeg = eeg.iloc[offset : offset + 10_000]

            waves = eeg.values
            waves = np.transpose(waves, axes=[1, 0])

            for i in range(waves.shape[0]):
                m = np.nanmean(waves[i])
                if np.isnan(waves[i]).mean() < 1:
                    waves[i] = np.nan_to_num(waves[i], nan=m)
                else:
                    waves[i] = 0

            waves = np.array(waves, dtype=np.float32)

            raise NotImplementedError("EEG path disabled in this script (EEG=False).")
        else:
            spec_path = os.path.join(
                self.base_dir,
                "test_spectrograms",
                f"{dp['spectrogram_id']}.parquet",
            )

            if not os.path.exists(spec_path):
                alt_base = "/kaggle/input/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification"
                alt_path = os.path.join(
                    alt_base,
                    "test_spectrograms",
                    f"{dp['spectrogram_id']}.parquet",
                )
                if os.path.exists(alt_path):
                    spec_path = alt_path

            spec = pd.read_parquet(spec_path)
            spec = spec.values[:, 1:]  # drop time column

            images = []
            for region in range(4):
                r = region * 300
                img = spec[r : r + 300, region * 100 : (region + 1) * 100].T
                img = np.clip(img, np.exp(-4), np.exp(8))
                img = np.log(img)
                img = np.nan_to_num(img, nan=0.0)
                images.append(img)

            images = np.stack(images, -1)  # (100, 300, 4)

        if is_training:
            data = self.train_trans(image=images)
            images = data["image"]

        images = np.transpose(images, [2, 0, 1])  # (4, 100, 300)
        return images.astype(np.float32)




## === cell 3
class Net(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=3)
        self.fc = nn.Linear(2048, 6, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bs = x.size(0)

        x1 = [x[:, i : i + 1, :, :] for i in range(4)]
        x1 = torch.cat(x1, dim=2)
        x = torch.cat([x1, x1, x1], dim=1)
        x = self.model.forward_features(x)
        x = self.avg(x)

        x = x.view(bs, -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x




## === cell 4
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for X in tqdm_test_loader:
            X = X.to(device, non_blocking=True)
            with torch.no_grad():
                y_preds = model(X)
                y_preds = softmax(y_preds)
            preds.append(y_preds.detach().cpu().numpy())

    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 5
test_df = pd.read_csv(CFG["data"])
test_df.head(5)



## === cell 6
available_weights = [w for w in CFG["weights"] if os.path.exists(w)]

if len(available_weights) == 0:
    print(
        "WARNING: No model weights found at the configured paths. Using uniform predictions."
    )
    predictions = np.full(
        (len(test_df), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32
    )
else:
    predictions = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base_dir = "/kaggle/input/hms-harmful-brain-activity-classification"

    test_dataset = AlaskaDataIter(
        test_df, training_flag=False, shuffle=False, base_dir=base_dir
    )
    test_loader = DataLoader(
        test_dataset,
        CFG["batch_size"],
        num_workers=CFG["num_worker"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    for model_weight in available_weights:
        model = Net()
        state_dict = torch.load(model_weight, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)

        prediction_dict = inference_function(test_loader, model, device)
        predictions.append(prediction_dict["predictions"])

        del model
        torch.cuda.empty_cache()
        gc.collect()

    predictions = np.array(predictions)
    predictions = np.mean(predictions, axis=0).astype(np.float32)



## === cell 7
dataset_root_primary = "/kaggle/input/hms-harmful-brain-activity-classification"
dataset_root_alt = "/kaggle/input/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification"

train_path = os.path.join(dataset_root_primary, "train.csv")
if not os.path.exists(train_path):
    train_path = os.path.join(dataset_root_alt, "train.csv")

if os.path.exists(train_path):
    train_df_for_prior = pd.read_csv(train_path, usecols=["eeg_id"] + TARGETS)
    train_votes = train_df_for_prior[TARGETS].astype(np.float64)
    row_sum = train_votes.sum(axis=1).values
    row_sum[row_sum == 0] = 1.0
    train_probs = train_votes.div(row_sum, axis=0)
    train_probs["eeg_id"] = train_df_for_prior["eeg_id"].values
    prior = (
        train_probs.groupby("eeg_id")[TARGETS]
        .mean()
        .mean(axis=0)
        .values.astype(np.float64)
    )
    prior = prior / prior.sum()
else:
    prior = np.full((len(TARGETS),), 1.0 / len(TARGETS), dtype=np.float64)

pred = predictions.astype(np.float64)

T = 3.80
pred = np.clip(pred, 1e-12, 1.0)
pred = pred ** (1.0 / T)
pred = pred / pred.sum(axis=1, keepdims=True)

alpha = 0.42
pred = (1.0 - alpha) * pred + alpha * prior[None, :]

eps = 5e-5
pred = (1.0 - eps) * pred + eps / len(TARGETS)

pred = np.clip(pred, 1e-12, 1.0)
pred = pred / pred.sum(axis=1, keepdims=True)
predictions = pred.astype(np.float32)



## === cell 8
if predictions.ndim != 2:
    raise ValueError(
        f"Predictions must be 2D (n_samples, 6). Got shape: {predictions.shape}"
    )

if predictions.shape[0] != len(test_df):
    raise ValueError(
        f"Predictions rows {predictions.shape[0]} != test rows {len(test_df)}"
    )

if predictions.shape[1] != len(TARGETS):
    raise ValueError(
        f"Predictions cols {predictions.shape[1]} != {len(TARGETS)} targets"
    )

predictions = np.clip(predictions, 1e-8, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
print(
    "Row-sum check (min/max):",
    sub[TARGETS].sum(axis=1).min(),
    sub[TARGETS].sum(axis=1).max(),
)
sub.head()
