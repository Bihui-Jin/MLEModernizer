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

0.4904182350328406

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the missing weight-file crash by making the script robust to absent `/kaggle/input/hms-baseline/*.pth` and falling back to a valid, probability-normalized baseline submission (class priors from train) so a valid `submission.csv` is always produced. I also remove the unintended per-sample `print(dp)` inside the Dataset that would spam logs and slow/timeout inference. Additionally, I ensure the test spectrogram slicing bug is fixed (`r` never incremented), and I guarantee predictions have shape `(n_test, 6)` and each row sums to 1 to avoid submission-format failures. These changes keep the core model/inference logic intact when weights exist, and only use the fallback when they don’t (current state cannot run otherwise).'
- What this solution (achieved 1.41937) has done: 'Your current score is far worse than the target (lower is better), and the main reason is that you’re falling back to train priors because the specified weight files don’t exist, which produces a weak baseline. I keep the exact model and inference logic, but change the weight discovery to load the available baseline fold weights from the competition dataset directory (where the `.pth` files typically reside in Kaggle notebooks) so the ensemble actually runs. I also make `in_chans` match the true 4-channel spectrogram input to avoid silently discarding one channel during the model’s `forward_features`, which should materially reduce KL without changing the overall architecture/training approach. Finally, I keep the probability normalization and submission format checks to ensure a valid CSV.'
- What this solution (achieved 1.41937) has done: 'Your score is far above the target (lower is better), and the most likely reason is that the ensemble is still effectively using the weak fallback (train priors) because the intended fold weights aren’t being found/loaded correctly. I make weight discovery deterministic and restricted to the known competition dataset directory, and I also fix the model’s channel handling so the `in_chans` setting and the `forward()` input match the checkpoint’s expected 3-channel EfficientNet input (your current code sets `in_chans=4` but then feeds 3 channels, which can silently break checkpoint loading and force near-random outputs). Finally, I keep your probability normalization but add a tiny floor after ensembling/softmax to avoid zeros (helps KL stability) without changing evaluation semantics. These are minimal changes that keep the same architecture and inference approach, but should move the score materially toward the target by actually using the trained weights.'
- What this solution (achieved 1.41937) has done: 'Your current score is much worse than the target (lower is better), and the most likely cause is that the inference is not using the trained baseline weights correctly due to a checkpoint key-prefix mismatch; with `strict=False` this can silently leave `fc` randomly initialized and severely hurt KL. I keep the exact model and inference logic, but make weight loading deterministic by stripping common prefixes (`module.`, `model.`, `net.`) and explicitly verifying that a meaningful portion of parameters matched, otherwise skipping that weight. I also ensure we preferentially load the 5 known fold checkpoints (if present) rather than any stray `.pth` files, so the ensemble reflects the intended baseline. These changes should materially move the score toward the target while preserving architecture, feature construction, and prediction semantics.'
- What this solution (achieved 1.41937) has done: 'We keep your model and inference logic exactly as-is, but make sure the ensemble actually uses the intended fold checkpoints by discovering `.pth` files inside the current competition dataset directory (in addition to `/kaggle/input/hms-baseline`). Then we prevent accidental “random head” inference by (1) only accepting checkpoints that load the `fc.*` weights and (2) loading to CPU first, then moving to GPU for inference, which avoids rare device/key issues and improves stability. Finally, we keep your probability normalization but apply a tiny post-ensemble floor consistently to reduce KL blow-ups from near-zeros without changing semantics.'

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
    "train_csv": "/kaggle/input/hms-harmful-brain-activity-classification/train.csv",
    "weights": [
        "/kaggle/input/hms-baseline/fold0_epoch_4_val_loss_0.589956.pth",
        "/kaggle/input/hms-baseline/fold1_epoch_4_val_loss_0.519593.pth",
        "/kaggle/input/hms-baseline/fold2_epoch_4_val_loss_0.499242.pth",
        "/kaggle/input/hms-baseline/fold3_epoch_4_val_loss_0.584708.pth",
        "/kaggle/input/hms-baseline/fold4_epoch_4_val_loss_0.655207.pth",
    ],
    "flip": True,
    "fallback_strategy": "train_priors",  # or 'uniform'
}

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


def normalize_probs(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Make valid per-row probability distributions (helps prevent KL issues with zeros)."""
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, None)
    row_sums = p.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    p = p / row_sums
    return p.astype(np.float32)


def compute_train_priors(train_csv_path: str) -> np.ndarray:
    """Compute class prior probs from vote counts in train.csv."""
    train_df = pd.read_csv(train_csv_path, usecols=TARGETS)
    totals = train_df[TARGETS].sum(axis=0).values.astype(np.float64)
    totals = np.clip(totals, 1e-12, None)
    priors = totals / totals.sum()
    return priors.astype(np.float32)


def discover_weight_files(cfg_weights: list[str]) -> list[str]:
    """
    Score-improving and minimal: ensure we actually find baseline fold weights wherever they are.
    We prioritize explicit CFG paths, then look in common Kaggle input dirs.
    """
    available = [w for w in cfg_weights if os.path.exists(w)]
    if len(available) > 0:
        return available

    roots = [
        "/kaggle/input/hms-baseline",
        "/kaggle/input/hms-harmful-brain-activity-classification",
    ]

    candidates = []
    for root in roots:
        if os.path.isdir(root):
            for fn in os.listdir(root):
                if fn.endswith(".pth"):
                    candidates.append(os.path.join(root, fn))

    if len(candidates) == 0:
        for root in roots:
            if os.path.isdir(root):
                for dirpath, _, filenames in os.walk(root):
                    for fn in filenames:
                        if fn.endswith(".pth"):
                            candidates.append(os.path.join(dirpath, fn))

    expected_like = []
    for p in candidates:
        bn = os.path.basename(p)
        if ("fold" in bn.lower()) and ("epoch" in bn.lower()):
            expected_like.append(p)
    if len(expected_like) > 0:
        candidates = expected_like

    return sorted(set(candidates))


def _strip_prefix_if_present(key: str, prefix: str) -> str:
    return (
        key[len(prefix) :] if isinstance(key, str) and key.startswith(prefix) else key
    )


def sanitize_state_dict_for_model(state_dict: dict, model: nn.Module) -> dict:
    """
    Make checkpoint keys compatible with our Net() definition without changing model logic.
    Handles common wrappers: 'module.', 'model.', 'net.' and checkpoints that store under 'state_dict'.
    """
    if (
        isinstance(state_dict, dict)
        and "state_dict" in state_dict
        and isinstance(state_dict["state_dict"], dict)
    ):
        state_dict = state_dict["state_dict"]

    if not isinstance(state_dict, dict):
        return {}

    cleaned = {}
    for k, v in state_dict.items():
        if not isinstance(k, str):
            continue
        kk = k
        for pref in ("module.",):
            kk = _strip_prefix_if_present(kk, pref)

        kk = _strip_prefix_if_present(kk, "model.")
        kk = _strip_prefix_if_present(kk, "net.")
        kk = _strip_prefix_if_present(kk, "backbone.")

        cleaned[kk] = v

    model_keys = set(model.state_dict().keys())
    if not any(k.startswith("model.") for k in cleaned.keys()):
        mapped = {}
        for k, v in cleaned.items():
            if k in model_keys:
                mapped[k] = v
            else:
                mk = "model." + k
                if mk in model_keys:
                    mapped[mk] = v
                else:
                    mapped[k] = v
        cleaned = mapped

    if "fc.weight" in model_keys and "fc.weight" not in cleaned:
        if (
            "classifier.weight" in cleaned
            and hasattr(cleaned["classifier.weight"], "shape")
            and cleaned["classifier.weight"].shape
            == model.state_dict()["fc.weight"].shape
        ):
            cleaned["fc.weight"] = cleaned["classifier.weight"]
        if (
            "classifier.bias" in cleaned
            and hasattr(cleaned["classifier.bias"], "shape")
            and cleaned["classifier.bias"].shape == model.state_dict()["fc.bias"].shape
        ):
            cleaned["fc.bias"] = cleaned["classifier.bias"]

    return cleaned


def load_checkpoint_strictish(
    model: nn.Module, weight_path: str, device: torch.device
) -> tuple[bool, str]:
    """
    Returns (ok, message). Keep strict=False (core logic unchanged) but ensure enough keys load.
    Minimal score fix: require fc.* present to avoid random classifier head hurting KL a lot.
    """
    raw = torch.load(weight_path, map_location="cpu")
    sd = sanitize_state_dict_for_model(raw, model)
    if len(sd) == 0:
        return False, "empty/invalid state_dict"

    missing, unexpected = model.load_state_dict(sd, strict=False)

    if ("fc.weight" in missing) or ("fc.bias" in missing):
        return False, "fc.* missing (would use random head)"

    total_keys = len(model.state_dict())
    missing_n = len(missing) if isinstance(missing, (list, tuple)) else 0
    loaded_frac = (total_keys - missing_n) / max(total_keys, 1)
    if loaded_frac < 0.80:
        return (
            False,
            f"low loaded_frac={loaded_frac:.2f} (missing {missing_n}/{total_keys})",
        )

    return (
        True,
        f"loaded_frac={loaded_frac:.2f}, unexpected={len(unexpected) if isinstance(unexpected,(list,tuple)) else 0}",
    )




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
        return self.single_map_func(self.df.iloc[item], self.training_flag)

    def __len__(self):
        return len(self.df)

    def single_map_func(self, dp, is_training):
        """Data loading + augmentation."""
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
            images = waves
        else:
            spec_path = (
                "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/%s.parquet"
                % (dp["spectrogram_id"])
            )
            spec = pd.read_parquet(spec_path)
            spec = spec.values[:, 1:]

            images = []
            for region in range(4):
                r0 = region * 300
                r1 = r0 + 300
                c0 = region * 100
                c1 = (region + 1) * 100

                img = spec[r0:r1, c0:c1].T
                img = np.clip(img, np.exp(-4), np.exp(8))
                img = np.log(img)
                img = np.nan_to_num(img, nan=0.0)
                images.append(img)

            images = np.stack(images, -1)

        if is_training:
            data = self.train_trans(image=images)
            images = data["image"]

        images = np.transpose(images, [2, 0, 1]).astype(np.float32)
        return images




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
        x = torch.cat([x1, x1, x1], dim=1)  # 3 channels (matches in_chans=3)
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
        for _, X in enumerate(tqdm_test_loader):
            X = X.to(device, non_blocking=True)
            with torch.no_grad():
                y_preds = model(X)
                y_preds = softmax(y_preds)

            if CFG["flip"]:
                with torch.no_grad():
                    X_flip = torch.flip(X, [3])
                    y_preds_flip = model(X_flip)
                    y_preds_flip = softmax(y_preds_flip)
                    y_preds = (y_preds + y_preds_flip) / 2.0

            preds.append(y_preds.detach().cpu().numpy())

    prediction_dict = {"predictions": np.concatenate(preds, axis=0)}
    return prediction_dict




## === cell 5
test_df = pd.read_csv(CFG["data"])
test_df.head(5)



## === cell 6
available_weights = discover_weight_files(CFG["weights"])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if len(available_weights) == 0:
    if CFG["fallback_strategy"] == "train_priors":
        priors = compute_train_priors(CFG["train_csv"])  # shape (6,)
    else:
        priors = np.ones(6, dtype=np.float32) / 6.0

    predictions = np.tile(priors[None, :], (len(test_df), 1))
    predictions = normalize_probs(predictions, eps=1e-9)
else:
    predictions_list = []
    test_dataset = AlaskaDataIter(test_df, training_flag=False, shuffle=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_worker"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    loaded_weight_paths = []
    for model_weight in available_weights:
        model = Net()

        ok, msg = load_checkpoint_strictish(model, model_weight, device)
        if not ok:
            print(f"Skipping weight (incompatible): {model_weight} -> {msg}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

        print(f"Loaded weight: {model_weight} -> {msg}")
        loaded_weight_paths.append(model_weight)

        model.to(device)

        prediction_dict = inference_function(test_loader, model, device)
        predictions_list.append(prediction_dict["predictions"])

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if len(predictions_list) == 0:
        if CFG["fallback_strategy"] == "train_priors":
            priors = compute_train_priors(CFG["train_csv"])
        else:
            priors = np.ones(6, dtype=np.float32) / 6.0
        predictions = np.tile(priors[None, :], (len(test_df), 1))
        predictions = normalize_probs(predictions, eps=1e-9)
        available_weights = []
    else:
        predictions = np.mean(np.stack(predictions_list, axis=0), axis=0)
        predictions = normalize_probs(predictions, eps=1e-6)
        available_weights = loaded_weight_paths



## === cell 7
assert predictions.shape[0] == len(
    test_df
), f"Pred rows {predictions.shape[0]} != n_test {len(test_df)}"
assert predictions.shape[1] == len(
    TARGETS
), f"Pred cols {predictions.shape[1]} != {len(TARGETS)}"

sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions
sub.to_csv("submission.csv", index=False)

print(f"Found {len(available_weights)} usable weight files.")
if len(available_weights) > 0:
    print("First few weights:", available_weights[:5])
print(f"Submission shape: {sub.shape}")
print(sub.head())
print(
    "Row sum stats:",
    float(sub[TARGETS].sum(axis=1).min()),
    float(sub[TARGETS].sum(axis=1).max()),
)
