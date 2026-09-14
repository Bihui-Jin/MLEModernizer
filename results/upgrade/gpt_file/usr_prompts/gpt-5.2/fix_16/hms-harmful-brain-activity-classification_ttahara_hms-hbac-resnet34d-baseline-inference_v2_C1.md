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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
PyYAML==6.0.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.7427134214724154

# 6. Current score

1.40522

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The submission length mismatch comes from merging predictions onto `sample_submission` without ensuring there is exactly one prediction row per `eeg_id`; if `test.csv` contains duplicated `eeg_id` values (or any mismatch in uniqueness), the merge can change the row count. I fix this by aggregating predictions to one row per `eeg_id` (mean over duplicates) before merging, and by building the submission strictly in the same order and length as `sample_submission`. I also add a safe fallback for any missing/extra ids and keep the core inference/model logic unchanged. This produce a valid `submission.csv` and avoid Kaggle’s “same length” error.'
- What this solution (achieved 1.40541) has done: 'We keep your model and inference loop unchanged and focus on lowering KL by improving probability calibration with a tiny post-processing step. Specifically, we apply a lightweight “prior smoothing” that blends each prediction with the empirical class prior from `train.csv` (converted to probabilities), which typically reduces overconfident mistakes and improves KL. We also add an optional temperature scaling (default 1.0, i.e., no change) so you can nudge confidence down if needed without changing core logic. These changes are confined to submission post-processing and keep row alignment/format guarantees intact.'
- What this solution (achieved 1.40361) has done: 'Your current score (1.40541, lower-is-better) is far from the target (0.7427), so we should improve KL while keeping your model/inference intact. The biggest safe lever here is post-processing calibration: we keep the same predictions but tune the “softening” and “prior blend” to reduce overconfidence, which typically lowers KL. Concretely, we (1) use a slightly stronger prior blend and (2) slightly higher temperature to soften probabilities, while still normalizing and preserving the exact submission alignment/format logic you already fixed. No changes are made to the model, transforms, dataset, or inference loop—only post-processing constants and a tiny numerical-stability safeguard.'
- What this solution (achieved 1.40209) has done: 'Your current KL (1.40361, lower-is-better) is still far above the target (0.7427), so we should improve (decrease) it with the smallest safe changes that don’t touch the model or inference loop. The most reliable lever here is post-processing calibration: we tune the existing prior-blend strength and temperature to further reduce overconfidence, which typically lowers KL without changing evaluation semantics. To keep this bounded and stable, we also add a tiny “floor mix” with uniform probabilities so no class gets extremely small probability (this is legitimate smoothing for KL and only affects post-processing). Everything else (data loading, model, folds, averaging, grouping to one row per eeg_id, submission alignment) remains unchanged.'
- What this solution (achieved 1.40218) has done: 'Your current score (1.40209, lower-is-better) is still far from the target (0.7427), so we should decrease KL with minimal-risk changes that don’t touch your model or inference loop. The safest lever remaining is post-processing calibration: make predictions less overconfident by slightly increasing temperature and slightly increasing the blend toward the empirical train prior, while keeping normalization and row alignment unchanged. I also slightly increase the uniform “floor mix” to reduce near-zero probabilities (which are heavily penalized by KL), and keep all submission integrity checks intact. No architecture/training/data extraction logic is changed.'
- What this solution (achieved 1.40279) has done: 'Your current KL (1.40218, lower-is-better) is still far above the target (0.7427), so we should decrease it with minimal-risk changes that don’t touch the model or inference loop. The biggest safe lever left is post-processing smoothing to avoid near-zero probabilities (heavily penalized by KL) while keeping predictions properly normalized and aligned to `sample_submission`. I slightly increase the uniform floor mix and slightly reduce sharpness by nudging temperature up a bit, while keeping the same prior-blend idea and the same aggregation to one row per `eeg_id`. Everything else (data loading, model definition, inference, fold averaging, submission schema checks) remains unchanged.'
- What this solution (achieved 1.40267) has done: 'Your current KL (1.40279, lower-is-better) is still far above the target (0.7427), so we should keep your model/inference unchanged and only adjust the safest lever: probability calibration post-processing. Specifically, we slightly strengthen the smoothing that reduces near-zero probabilities (which are heavily penalized by KL) by (1) increasing the uniform floor mix a bit and (2) nudging temperature up a bit to reduce overconfidence. We keep the empirical prior blend but make it slightly stronger as well, while preserving strict row alignment to `sample_submission` and exact normalization so the submission remains valid. No changes are made to architecture, transforms, dataset logic, or the inference loop.'
- What this solution (achieved 1.40274) has done: 'We keep your model and inference untouched and only adjust the post-processing calibration constants, because your current KL (1.40267, lower-is-better) is still far above the target (0.7427) and the safest lever to reduce KL is to further reduce overconfidence and avoid near-zero probabilities. Concretely, we (1) increase the blend toward the empirical train prior a bit and (2) slightly increase temperature softening, while (3) adding a small increase to the uniform floor mix to reduce KL spikes from tiny probabilities. We also keep the same strict submission alignment and normalization checks so the output remains valid and stable. No training, architecture, transforms, dataset logic, or inference loop changes are made.'
- What this solution (achieved 1.40277) has done: 'Your current KL (1.40274, lower-is-better) is still far above the target (0.7427), so we should decrease it with the smallest changes that don’t touch your model or inference loop. The safest improvement lever left is post-processing calibration to reduce overconfident errors: we slightly increase temperature softening and slightly strengthen the blend toward the empirical train prior, while also nudging up the uniform floor to avoid near-zero probabilities that are heavily penalized by KL. Everything else (data loading, model definition, inference, fold averaging, aggregation to one row per `eeg_id`, and strict submission alignment/normalization) remains unchanged. This should move KL downward without changing the core modeling logic.'
- What this solution (achieved 1.40312) has done: 'Your current KL (1.40277, lower-is-better) is still far above the target (0.7427), so we should keep your model/inference unchanged and only make a minimal, metric-aligned post-processing adjustment. The safest way to reduce KL is to further suppress overconfident mistakes by slightly increasing probability smoothing: (1) blend a bit more toward the empirical train prior, (2) apply a slightly higher temperature to soften distributions, and (3) slightly increase the uniform floor to avoid near-zero probabilities that are heavily penalized by KL. All row alignment, aggregation to one row per `eeg_id`, and normalization checks remain exactly as before to guarantee a valid submission. No architecture, dataset, transforms, or inference loop logic is modified.'
- What this solution (achieved 1.40338) has done: 'Your current KL (1.40312, lower-is-better) is still far above the target (0.7427), so we should decrease it with the smallest, safest change that doesn’t touch the model or inference loop. The most metric-aligned lever left is to reduce catastrophic KL spikes from near-zero probabilities by slightly increasing the uniform floor smoothing while keeping your existing prior-blend and temperature logic intact. I keep aggregation/alignment to `sample_submission` exactly the same, and only adjust the post-processing constants and add a tiny numerical floor inside temperature scaling for stability. This should move the score downward (better) without changing the core modeling approach.'
- What this solution (achieved 1.40401) has done: 'Your current KL (1.40338, lower-is-better) is still far above the target (0.7427), so we should decrease it with the smallest, safest change that doesn’t touch your model or inference loop. The main remaining lever is post-processing calibration: slightly stronger smoothing to reduce near-zero probabilities (which cause large KL penalties) while keeping strict normalization and submission alignment unchanged. Concretely, we (1) increase the uniform floor mix a bit and (2) very slightly increase temperature to soften overconfident distributions, leaving the prior-blend and all inference code intact. Everything else (file paths, aggregation to one row per eeg_id, and submission integrity checks) stays the same.'
- What this solution (achieved 1.40421) has done: 'Your current KL (1.40401, lower-is-better) is still far above the target (0.7427), so we should move it downward with the smallest change that preserves your model/inference loop. The safest, metric-aligned lever is still post-processing calibration: reduce overconfidence and especially avoid near-zero probabilities (which can create large KL spikes) by slightly increasing smoothing. I keep your exact prior-blend + temperature + uniform-floor logic and only nudge the constants in a direction that should reduce KL (more smoothing), while keeping strict normalization, one-row-per-eeg_id aggregation, and exact sample_submission alignment unchanged. No architecture, dataset, transforms, or inference loop code is modified.'
- What this solution (achieved 1.40522) has done: 'Your current KL (1.40421, lower-is-better) is still far above the target (0.7427), so we should decrease it with minimal risk while preserving your model/inference logic. The safest lever is post-processing calibration: because KL heavily penalizes near-zero probabilities, we increase the minimum-probability smoothing in a controlled way by slightly raising the uniform floor mix and slightly increasing temperature (softer distributions), while keeping the same prior blend mechanism. To avoid accidentally oversmoothing past a useful point, we also slightly reduce the prior-blend weight (so predictions still reflect the model more). All changes are confined to the constants in cell 12; data loading, model, inference, aggregation, and submission alignment remain unchanged.'

# 9. Code solution

## === cell 0
import sys
import os
import gc
import copy
import yaml
import random
import shutil
from time import time
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedGroupKFold

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.cuda import amp

import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



## === cell 2
ROOT = Path("/kaggle")
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
SRC = ROOT / "src"

DATA = INPUT / "hms-harmful-brain-activity-classification"
TRAIN_SPEC = DATA / "train_spectrograms"
TEST_SPEC = DATA / "test_spectrograms"

TRAINED_MODEL = INPUT / "hms-hbac-resnet34d-baseline-training"

TMP = ROOT / "tmp"
TRAIN_SPEC_SPLIT = TMP / "train_spectrograms_split"
TEST_SPEC_SPLIT = TMP / "test_spectrograms_split"
TMP.mkdir(exist_ok=True, parents=True)
TRAIN_SPEC_SPLIT.mkdir(exist_ok=True, parents=True)
TEST_SPEC_SPLIT.mkdir(exist_ok=True, parents=True)

RANDAM_SEED = 1086
CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
N_CLASSES = len(CLASSES)
FOLDS = [0, 1, 2, 3, 4]
N_FOLDS = len(FOLDS)



## === cell 3
test = pd.read_csv(DATA / "test.csv")



## === cell 4
test.head()



## === cell 5
missing_specs = 0
for spec_id in tqdm(
    test["spectrogram_id"].values, desc="Converting test spectrograms to .npy"
):
    out_path = TEST_SPEC_SPLIT / f"{spec_id}.npy"
    if out_path.exists():
        continue
    pq_path = TEST_SPEC / f"{spec_id}.parquet"
    if not pq_path.exists():
        missing_specs += 1
        continue

    spec = pd.read_parquet(pq_path)
    spec_arr = (
        spec.fillna(0).iloc[:, 1:].to_numpy().T.astype("float32")
    )  # (Hz, Time) ~ (400, 300)
    np.save(out_path, spec_arr)

if missing_specs > 0:
    print(f"Warning: {missing_specs} spectrogram parquet files were missing.")




## === cell 6
class HMSHBACSpecModel(nn.Module):

    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        in_channels: int,
        num_classes: int,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
        )

    def forward(self, x):
        h = self.model(x)
        return h




## === cell 7
FilePath = tp.Union[str, Path]
Label = tp.Union[int, float, np.ndarray]


class HMSHBACSpecDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        image_paths: tp.Sequence[FilePath],
        labels: tp.Sequence[Label],
        transform: A.Compose,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        label = self.labels[index]

        img = np.load(img_path)  # shape: (Hz, Time)
        eps = 1e-6
        img_mean = img.mean(axis=(0, 1))
        img = img - img_mean
        img_std = img.std(axis=(0, 1))
        img = img / (img_std + eps)

        img = img[..., None]  # (Hz, Time, 1)
        img = self._apply_transform(img)

        return {"data": img, "target": label}

    def _apply_transform(self, img: np.ndarray):
        transformed = self.transform(image=img)
        img = transformed["image"]
        return img




## === cell 8
class CFG:
    model_name = "resnet34d"
    img_size = 512
    max_epoch = 16
    batch_size = 32
    lr = 1.0e-03
    weight_decay = 1.0e-02
    es_patience = 5
    seed = 1086
    deterministic = True
    enable_amp = True
    device = "cuda" if torch.cuda.is_available() else "cpu"




## === cell 9
def to_device(
    tensors: tp.Union[tp.Tuple[torch.Tensor], tp.Dict[str, torch.Tensor]],
    device: torch.device,
    *args,
    **kwargs,
):
    if isinstance(tensors, tuple):
        return (t.to(device, *args, **kwargs) for t in tensors)
    elif isinstance(tensors, dict):
        return {k: t.to(device, *args, **kwargs) for k, t in tensors.items()}
    else:
        return tensors.to(device, *args, **kwargs)


def get_test_path_label(test: pd.DataFrame):
    """Get file path and dummy target info."""
    img_paths = []
    labels = np.full((len(test), N_CLASSES), -1, dtype="float32")
    for spec_id in test["spectrogram_id"].values:
        img_path = TEST_SPEC_SPLIT / f"{spec_id}.npy"
        img_paths.append(img_path)
    return {"image_paths": img_paths, "labels": [l for l in labels]}


def get_test_transforms(CFG):
    return A.Compose(
        [
            A.Resize(p=1.0, height=CFG.img_size, width=CFG.img_size),
            ToTensorV2(p=1.0),
        ]
    )




## === cell 10
def run_inference_loop(model, loader, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            x = to_device(batch["data"], device)
            y = model(x)
            pred_list.append(y.softmax(dim=1).detach().cpu().numpy())
    pred_arr = np.concatenate(pred_list, axis=0)
    del pred_list
    return pred_arr




## === cell 11
device = torch.device(CFG.device)

have_all_weights = True
for fold_id in range(N_FOLDS):
    model_path = TRAINED_MODEL / f"best_model_fold{fold_id}.pth"
    if not model_path.exists():
        have_all_weights = False
        break

if have_all_weights:
    test_preds_arr = np.zeros((N_FOLDS, len(test), N_CLASSES), dtype="float32")

    test_path_label = get_test_path_label(test)
    test_transform = get_test_transforms(CFG)
    test_dataset = HMSHBACSpecDataset(**test_path_label, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        num_workers=0,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    for fold_id in range(N_FOLDS):
        print(f"\n[fold {fold_id}]")
        model_path = TRAINED_MODEL / f"best_model_fold{fold_id}.pth"
        model = HMSHBACSpecModel(
            model_name=CFG.model_name,
            pretrained=False,
            num_classes=N_CLASSES,
            in_channels=1,
        )
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)

        test_pred = run_inference_loop(model, test_loader, device)
        test_preds_arr[fold_id] = test_pred

        del model, state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    test_pred = test_preds_arr.mean(axis=0)
else:
    print(f"Warning: trained model weights not found at: {TRAINED_MODEL}")
    print("Falling back to uniform probabilities to produce a valid submission.csv.")
    test_pred = np.full((len(test), N_CLASSES), 1.0 / N_CLASSES, dtype="float32")




## === cell 12
def _safe_normalize_probs(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.asarray(p, dtype="float32")
    p = np.nan_to_num(p, nan=1.0 / N_CLASSES, posinf=1.0, neginf=0.0)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def apply_temperature(p: np.ndarray, T: float = 1.0, eps: float = 1e-8) -> np.ndarray:
    p = _safe_normalize_probs(p, eps=eps)
    if T is None or float(T) == 1.0:
        return p
    internal_eps = max(eps, 1e-6)
    T = float(T)
    logp = np.log(np.clip(p, internal_eps, 1.0))
    logp = logp / T
    logp = logp - logp.max(axis=1, keepdims=True)
    pT = np.exp(logp)
    pT = pT / pT.sum(axis=1, keepdims=True)
    return pT.astype("float32")


def compute_train_prior(
    train_csv_path: Path, classes: tp.Sequence[str], eps: float = 1e-8
) -> np.ndarray:
    train_df = pd.read_csv(train_csv_path, usecols=list(classes))
    votes = train_df[classes].to_numpy(dtype="float64")
    row_sum = votes.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    probs = votes / row_sum
    prior = probs.mean(axis=0)
    prior = np.clip(prior, eps, 1.0)
    prior = prior / prior.sum()
    return prior.astype("float32")


PRIOR_BLEND_ALPHA = 0.74  # was 0.80
TEMPERATURE_T = 3.45  # was 3.25
UNIFORM_FLOOR_BETA = 0.32  # was 0.26

test_pred = np.asarray(test_pred, dtype="float32")
test_pred = _safe_normalize_probs(test_pred, eps=1e-8)

train_prior = compute_train_prior(DATA / "train.csv", CLASSES, eps=1e-8)  # shape (6,)

test_pred = (1.0 - PRIOR_BLEND_ALPHA) * test_pred + PRIOR_BLEND_ALPHA * train_prior[
    None, :
]

test_pred = apply_temperature(test_pred, T=TEMPERATURE_T, eps=1e-8)

test_pred = (1.0 - UNIFORM_FLOOR_BETA) * test_pred + UNIFORM_FLOOR_BETA * (
    1.0 / N_CLASSES
)

test_pred = _safe_normalize_probs(test_pred, eps=1e-8)

test_pred_df = pd.DataFrame(test_pred, columns=CLASSES)
test_pred_df = pd.concat(
    [test[["eeg_id"]].reset_index(drop=True), test_pred_df], axis=1
)

test_pred_df = test_pred_df.groupby("eeg_id", as_index=False)[CLASSES].mean()



## === cell 13
smpl_sub = pd.read_csv(DATA / "sample_submission.csv")

sub = smpl_sub[["eeg_id"]].merge(test_pred_df, on="eeg_id", how="left")

missing_rows = sub[CLASSES].isna().any(axis=1).sum()
if missing_rows > 0:
    sub.loc[sub[CLASSES].isna().any(axis=1), CLASSES] = 1.0 / N_CLASSES
    print(
        f"Warning: {missing_rows} eeg_id rows missing preds; filled with uniform probabilities."
    )

vals = sub[CLASSES].to_numpy(dtype="float32")
vals = np.nan_to_num(vals, nan=1.0 / N_CLASSES, posinf=1.0, neginf=0.0)
vals = np.clip(vals, 1e-8, 1.0)
vals = vals / vals.sum(axis=1, keepdims=True)
sub[CLASSES] = vals

sub.to_csv("submission.csv", index=False)
sub.head()



## === cell 14
assert sub.shape[0] == smpl_sub.shape[0], "Submission row count mismatch."
assert list(sub.columns) == ["eeg_id"] + CLASSES, "Submission columns mismatch."
row_sums = sub[CLASSES].sum(axis=1).to_numpy()
assert np.allclose(row_sums, 1.0, atol=1e-4), "Row probabilities do not sum to 1."
assert sub["eeg_id"].equals(
    smpl_sub["eeg_id"]
), "Submission eeg_id order must match sample_submission."
print("Wrote submission.csv with valid probability rows.")
