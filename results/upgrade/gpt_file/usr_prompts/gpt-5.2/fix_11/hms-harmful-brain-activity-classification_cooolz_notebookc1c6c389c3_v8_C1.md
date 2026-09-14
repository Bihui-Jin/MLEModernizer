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

0.4923887280217045

# 6. Current score

1.46552

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime blocker by removing the missing external weight dependency and falling back to a safe, deterministic baseline predictor when weights are not found, so the notebook always produces a valid `submission.csv`. I also fix a logic bug in spectrogram slicing (the row offset `r` was never updated) and ensure the dataset always returns a consistent `float32` tensor shape. Finally, I guarantee the submission probabilities are valid for the KL metric by clipping to a small epsilon and renormalizing each row to sum to 1, preventing submission failures.'
- What this solution (achieved 4.92373) has done: 'I fix the DataLoader crash by making the spectrogram slicing robust to variable spectrogram shapes (some files can be shorter than expected), padding/cropping each region to a consistent (100,300) before stacking. This is a runtime-only fix that preserves the same core spectrogram-based inference logic and model architecture, but ensures every sample returns a consistent float32 tensor shape. I also make inference deterministic by setting `num_workers=0` (avoids worker crashes/overhead in this environment) and add a safe fallback that produces valid uniform probabilities if any unexpected issue occurs during model inference. Finally, I keep the required submission format and explicitly clip+renormalize probabilities so every row sums to 1 for the KL metric.'
- What this solution (achieved 1.91118) has done: 'Your current score (4.92373, lower-is-better) is far worse than the target (0.49239), so we should make the smallest changes that legitimately improve predictions without changing your model architecture or inference loop. The biggest issue is a data/shape mismatch: your dataset builds 4 spectrogram channels, but the model was created with `in_chans=3` and your forward builds a 3-channel input from the 4 channels; this mismatch makes loaded weights ineffective (and even when `strict=False`, the stem weights won’t load as intended), severely hurting score. I change the model to use `in_chans=3` but make the dataset output 3 channels in the exact way your forward expects (stack 4 regions into one tall image, then replicate to 3 channels inside the model), which preserves the core logic while restoring weight compatibility. I also fix a critical spectrogram slicing bug (row slicing should use all time rows, not `region*300`), which currently makes 3 of 4 regions mostly/entirely zero-padded for many files and degrades predictions.'
- What this solution (achieved 1.46943) has done: 'Your current score (1.91118, lower-is-better) is far worse than the target (0.49239), so we should make the smallest changes that legitimately improve prediction quality without altering your model/inference core. The biggest likely accuracy bug is that EfficientNet expects ImageNet-style normalization; your spectrogram “images” are log-scaled but never standardized, so the pretrained backbone (or any weights trained with the usual normalization) behave poorly—adding the standard ImageNet mean/std normalization is a minimal, semantics-preserving fix. I also load fold weights more correctly by unwrapping common checkpoint formats (`state_dict`, `model`) while still using `strict=False`, which increases the chance the fold models actually use their trained parameters. Finally, I keep your probability clipping+renormalization (needed for KL) and leave the architecture, TTA, and data extraction structure intact.'
- What this solution (achieved 1.46943) has done: 'Your current KL (1.469) is still far worse than the target (0.492; lower is better), so the smallest legitimate improvement is to make the checkpoint loading actually work as intended. The main issue is that many training pipelines save weights with keys prefixed by `"module."` or `"model."`, and your current loader doesn’t strip these prefixes—so even with `strict=False`, large parts of the model may not load and you effectively predict with near-random weights. I add a minimal `clean_state_dict_keys()` that (a) strips common prefixes and (b) only keeps keys that match the current model’s shapes, maximizing loaded coverage without changing architecture/loops. Everything else (data extraction, model forward, TTA, normalization, probability renorm, and submission format) stays the same.'
- What this solution (achieved 1.46943) has done: 'Your current KL (1.469, lower-is-better) is still far from the target (0.492), so we should make a small change that is very likely to improve predictions without changing the model architecture or inference loop. The highest-impact minimal fix is to match the channel construction to what EfficientNet-B5 stem expects by turning the single spectrogram “image” into a true 3-channel input at the dataset level (instead of replicating inside `forward` after concatenation logic), which also makes flip-TTA operate on exactly the same representation the model sees. I also ensure the loaded checkpoints apply to the *full* `Net` (including `fc`) by trying both raw keys and keys prefixed with `model.` when cleaning, increasing effective weight coverage while keeping `strict=False`. Everything else (spectrogram extraction, log scaling, EfficientNet-B5 backbone, flip TTA, softmax, probability clipping/renorm, submission format) stays the same.'
- What this solution (achieved 1.46997) has done: 'To move your KL down toward the 0.492 target without changing the model core, the most likely high-impact minimal fix is to make test-time preprocessing match what the fold checkpoints were trained on: the dataset currently log-scales but does not apply the per-image standardization that most HMS spectrogram baselines use, so the loaded weights can behave badly and yield near-random probabilities. I add a lightweight, deterministic per-sample normalization (standardize the stacked log-spectrogram using its own mean/std over valid pixels) while keeping your 3-channel construction, flip-TTA, EfficientNet-B5, softmax, and probability renormalization intact. I also make checkpoint loading slightly more compatible by additionally stripping a common `"net."` prefix (seen in many training scripts), improving the chance that your folds actually load correctly. These are minimal changes aimed specifically at improving prediction calibration/quality for KL, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 1.46655) has done: 'Your current KL (1.46997; lower is better) is far worse than the target (0.49239), so we should make a small, high-confidence fix that improves prediction quality without changing the model architecture or inference loop. The biggest likely issue is a preprocessing mismatch: your dataset standardizes per-sample, but the model then clamps/linearly rescales as if inputs were in log-range [-4,8], effectively distorting the standardized values; I remove that incompatible clamp/rescale so the model sees a consistent standardized “image” and only applies ImageNet normalization. I also switch from the external `albumentations` horizontal flip (unused here) to a minimal, deterministic log-spectrogram normalization that matches common HMS baselines: normalize each sample with robust percentiles, then apply ImageNet mean/std. Finally, I keep the existing probability clipping+renormalization for valid KL submissions and preserve all I/O paths and the overall inference workflow.'
- What this solution (achieved 1.46552) has done: 'Your current KL (1.46655, lower-is-better) is far above the target (0.49239), so we should make a small change that’s very likely to improve predictive signal without altering your model/inference structure. The biggest accuracy issue left is that the test spectrograms are being used “as-is” (only log + robust scaling), while HMS baseline fold checkpoints are typically trained on **normalized** log-spectrograms (mean/std per image or per channel); this mismatch makes the network operate off-distribution. I add a minimal per-sample standardization step **after** log transform and **before** robust scaling (only on non-zero pixels), keeping the same extraction, EfficientNet-B5, flip-TTA, and probability post-processing. I also make weight-path handling robust by auto-discovering fold weights if they exist under `/kaggle/input/**` (without changing the ensemble logic), so you don’t silently fall back to ImageNet-only weights.'

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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader



## === cell 1
CFG = {
    "batch_size": 32,
    "num_worker": 4,
    "data": "/kaggle/input/hms-harmful-brain-activity-classification/test.csv",
    "weights": [
        "/kaggle/input/hms-baseline/fold0_epoch_4_val_loss_0.549755.pth",
        "/kaggle/input/hms-baseline/fold1_epoch_4_val_loss_0.524753.pth",
        "/kaggle/input/hms-baseline/fold2_epoch_4_val_loss_0.511551.pth",
        "/kaggle/input/hms-baseline/fold3_epoch_4_val_loss_0.548698.pth",
        "/kaggle/input/hms-baseline/fold4_epoch_3_val_loss_0.666013.pth",
    ],
    "flip": True,
    "seed": 42,
}


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CFG["seed"])




## === cell 2
class AlaskaDataIter:
    def __init__(self, df, training_flag=False, shuffle=False):
        self.training_flag = training_flag
        self.shuffle = shuffle
        self.raw_data_set_size = None
        self.df = df

        self.train_trans = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
            ]
        )

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

    def __getitem__(self, item):
        x = self.single_map_func(self.df.iloc[item], self.training_flag)
        return torch.from_numpy(np.ascontiguousarray(x)).float()

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _pad_or_crop_2d(a, target_h, target_w, pad_value=0.0):
        a = np.asarray(a)
        h, w = a.shape
        out = np.full((target_h, target_w), pad_value, dtype=a.dtype)
        hh = min(h, target_h)
        ww = min(w, target_w)
        if hh > 0 and ww > 0:
            out[:hh, :ww] = a[:hh, :ww]
        return out

    @staticmethod
    def _standardize_nonzero(img2d, eps=1e-6):
        """
        Score-improving minimal change (distribution match):
        Many HMS spectrogram baselines train on per-image standardized log-spectrograms.
        We standardize using only non-zero pixels (zeros come from padding), keeping
        the same log transform and overall pipeline.
        """
        x = img2d.astype(np.float32)
        mask = x != 0.0
        if not mask.any():
            return x
        v = x[mask]
        mu = float(v.mean())
        sd = float(v.std())
        if (not np.isfinite(mu)) or (not np.isfinite(sd)) or sd < eps:
            return x
        x = x.copy()
        x[mask] = (x[mask] - mu) / sd
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return x

    @staticmethod
    def _robust_scale_01(img2d, eps=1e-6):
        """
        Keep your robust percentile scaling to [0,1] (stable for inference), but now
        applied after standardization for better alignment with fold checkpoints.
        """
        x = img2d.astype(np.float32)
        mask = x != 0.0
        if mask.any():
            v = x[mask]
            lo = float(np.percentile(v, 1.0))
            hi = float(np.percentile(v, 99.0))
        else:
            lo = float(np.min(x))
            hi = float(np.max(x))
        if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < eps:
            return np.zeros_like(x, dtype=np.float32)
        x = np.clip(x, lo, hi)
        x = (x - lo) / (hi - lo)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return x

    def single_map_func(self, dp, is_training):
        spec_path = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/%s.parquet"
            % (dp["spectrogram_id"])
        )
        spec = pd.read_parquet(spec_path)
        spec = np.asarray(spec.values[:, 1:])  # drop time column

        region_imgs = []
        for region in range(4):
            c0 = region * 100
            c1 = (region + 1) * 100
            img = spec[:, c0:c1].T  # (freq=100, time~=300)
            img = self._pad_or_crop_2d(img, 100, 300, pad_value=0.0)

            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)
            img = np.nan_to_num(img, nan=0.0).astype(np.float32)
            region_imgs.append(img)

        stacked = np.concatenate(region_imgs, axis=0).astype(np.float32)  # (400,300)

        stacked = self._standardize_nonzero(stacked)

        stacked = self._robust_scale_01(stacked)

        stacked3 = np.stack([stacked, stacked, stacked], axis=0).astype(np.float32)
        return stacked3




## === cell 3
class Net(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):
        super().__init__()
        self.model = timm.create_model(
            "efficientnet_b5", pretrained=pretrained, in_chans=3
        )
        self.fc = nn.Linear(2048, 6, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

        self.register_buffer(
            "img_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "img_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

    def forward(self, x):
        bs = x.size(0)

        if x.size(1) == 1:
            x = torch.cat([x, x, x], dim=1)  # (B,3,H,W)
        elif x.size(1) != 3:
            x1 = [x[:, i : i + 1, :, :] for i in range(min(4, x.size(1)))]
            x1 = torch.cat(x1, dim=2)
            x = torch.cat([x1, x1, x1], dim=1)

        x = torch.clamp(x, 0.0, 1.0)
        x = (x - self.img_mean) / self.img_std

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

                if CFG["flip"]:
                    X_flip = torch.flip(X, [3])
                    y_preds_flip = model(X_flip)
                    y_preds_flip = softmax(y_preds_flip)
                    y_preds = (y_preds + y_preds_flip) / 2.0

            preds.append(y_preds.detach().cpu().numpy())

    return {"predictions": np.concatenate(preds, axis=0)}


def normalize_probs(p, eps=1e-6):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p.astype(np.float32)


def unwrap_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "net", "model_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def clean_state_dict_keys(state_dict, model):
    """
    Keep: strip common prefixes and only load keys matching shapes.
    """
    if not isinstance(state_dict, dict):
        return state_dict

    model_sd = model.state_dict()
    cleaned = {}

    def _try_add(k_in, v):
        if k_in in model_sd:
            try:
                if hasattr(v, "shape") and hasattr(model_sd[k_in], "shape"):
                    if tuple(v.shape) == tuple(model_sd[k_in].shape):
                        cleaned[k_in] = v
                else:
                    cleaned[k_in] = v
            except Exception:
                return

    for k, v in state_dict.items():
        kk = k
        for pref in ["module.", "model.", "net."]:
            if kk.startswith(pref):
                kk = kk[len(pref) :]

        _try_add(kk, v)
        _try_add("model." + kk, v)

    return cleaned


def discover_weights_if_missing(weight_paths):
    """
    Score-improving minimal change (avoid silent ImageNet-only fallback):
    If provided paths don't exist in this environment, try to find similarly-named
    fold*.pth files under /kaggle/input without changing ensemble semantics.
    """
    existing = [w for w in weight_paths if os.path.exists(w)]
    if len(existing) > 0:
        return existing

    candidates = []
    for root, _, files in os.walk("/kaggle/input"):
        for fn in files:
            if fn.endswith(".pth") and ("fold" in fn):
                candidates.append(os.path.join(root, fn))
    candidates = sorted(candidates)
    return candidates




## === cell 5
test_df = pd.read_csv(CFG["data"])
test_df.head(5)



## === cell 6
available_weights = discover_weights_if_missing(CFG["weights"])

predictions_list = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_dataset = AlaskaDataIter(test_df, training_flag=False, shuffle=False)

num_workers = 0

test_loader = DataLoader(
    test_dataset,
    batch_size=CFG["batch_size"],
    num_workers=num_workers,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)

try:
    if len(available_weights) == 0:
        model = Net(pretrained=True)
        model.to(device)

        prediction_dict = inference_function(test_loader, model, device)
        predictions = prediction_dict["predictions"].astype(np.float32)

        del model
        torch.cuda.empty_cache()
        gc.collect()
    else:
        for model_weight in available_weights:
            model = Net(pretrained=False)
            ckpt = torch.load(model_weight, map_location=device)
            state_dict = unwrap_state_dict(ckpt)
            state_dict = clean_state_dict_keys(state_dict, model)

            model.load_state_dict(state_dict, strict=False)
            model.to(device)

            prediction_dict = inference_function(test_loader, model, device)
            predictions_list.append(prediction_dict["predictions"])

            del model, ckpt, state_dict
            torch.cuda.empty_cache()
            gc.collect()

        predictions = np.mean(np.stack(predictions_list, axis=0), axis=0).astype(
            np.float32
        )

    predictions = normalize_probs(predictions, eps=1e-6)
except Exception as e:
    print("Inference failed with exception; falling back to uniform probabilities.")
    print(repr(e))
    n = len(test_df)
    predictions = np.full((n, 6), 1.0 / 6.0, dtype=np.float32)
    predictions = normalize_probs(predictions, eps=1e-6)



## === cell 7
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
assert predictions.shape[0] == len(sub), (predictions.shape, len(sub))
assert predictions.shape[1] == len(TARGETS), (predictions.shape, len(TARGETS))

sub[TARGETS] = predictions

sub[TARGETS] = sub[TARGETS].clip(1e-6, 1.0)
sub[TARGETS] = sub[TARGETS].div(sub[TARGETS].sum(axis=1), axis=0)

sub.to_csv("submission.csv", index=False)

print(f"Submission shape: {sub.shape}")
print(sub.head())
print(
    "Row sums (min/max):",
    sub[TARGETS].sum(axis=1).min(),
    sub[TARGETS].sum(axis=1).max(),
)
print("Saved to submission.csv")
