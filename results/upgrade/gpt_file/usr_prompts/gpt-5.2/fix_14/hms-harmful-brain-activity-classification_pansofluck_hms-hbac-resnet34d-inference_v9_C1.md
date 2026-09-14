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

0.5256717975592473

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the missing pretrained-weights path issue by making the script robust to the absence of `/kaggle/input/hms-hbac-effnetv2-baseline` (so it can still run end-to-end) and ensure a valid CSV is always produced. I also fix the submission length mismatch by aligning predictions to `sample_submission.csv` using a guaranteed 1-row-per-`eeg_id` prediction table (deduplicated and reindexed), and I explicitly renormalize probabilities to sum to 1 per row (required for this competition). If the external fold checkpoints are not available, I fall back to a safe, valid baseline submission using the class prior from `train.csv` (score won’t be good, but it be valid and runnable); if checkpoints are present, the original inference logic is preserved and used.'
- What this solution (achieved 1.41937) has done: 'Your current score (1.41937, lower-is-better) is far worse than the target (0.52567), so we should improve performance, but with minimal changes and without altering the core model/inference logic. The biggest likely issue is a train/test preprocessing mismatch: you normalize each spectrogram by subtracting a single scalar mean and std computed over (Hz, Time), but you do it via `mean(axis=(0,1))` which returns a length-`Time` vector; that silently normalizes per-time-step rather than globally, which can hurt heavily and differ from what the checkpoint expects. I fix the normalization to true global scalar mean/std (keeping the same log/clip semantics) and also make inference optionally use AMP (already enabled in CFG) for speed without changing predictions materially. Everything else (model, weights loading, averaging, submission alignment, renormalization) stays the same.'
- What this solution (achieved 1.41937) has done: 'Your score is much worse than the target (1.41937 vs 0.52567, lower-is-better), so we should improve it with the smallest changes that don’t alter the core model or inference approach. The most likely gain is fixing a train/test mismatch at inference-time: applying the same temporal cropping/aggregation the model was trained with (HMS baselines typically use the central spectrogram window rather than the full 10 minutes). I add a minimal “center-crop along time” inside the Dataset (no model/loss changes) and keep the existing per-image log/clip + global mean/std normalization and fold-averaging. Submission alignment and probability renormalization remain as-is to guarantee validity.'
- What this solution (achieved 1.41937) has done: 'We keep your model and fold-averaging logic unchanged and focus on two minimal, high-impact inference-time fixes that commonly hurt KL divergence when wrong: (1) make the spectrogram normalization consistent with how most HMS baselines expect it (per-frequency z-score across time, not a single global z-score), and (2) crop the spectrogram along time to the central window sized to the model’s input resolution (so the resize doesn’t overly squash a long time axis). These changes don’t alter architecture, loss, or training loops, only preprocessing, and should move your score down toward the 0.525 target if you have the fold checkpoints. Submission alignment and strict probability renormalization remain as-is to guarantee a valid CSV. If checkpoints are missing, it still produces a valid prior-based submission.'
- What this solution (achieved 1.41937) has done: 'Your current score (1.41937, lower-is-better) is far worse than the target (0.52567), so we should improve it with minimal, inference-only changes that keep the same model and fold-averaging logic. The most likely issue is a preprocessing mismatch with the baseline EfficientNetV2 spectrogram checkpoints: they typically expect a fixed center-crop of the time axis and per-frequency normalization, but your current config may not match what the weights were trained on. I (1) make the time crop derive from the native spectrogram width and resize (so the model sees the intended central region rather than a heavily squashed full window), and (2) switch to per-frequency z-score normalization by default (keeping the same clip+log semantics). Submission alignment and probability renormalization stay unchanged to guarantee a valid CSV.'
- What this solution (achieved 1.41937) has done: 'Your current score (1.41937, lower-is-better) is far from the target (0.52567), so the safest way to move toward the target without changing the model/architecture is to fix high-impact inference-time mismatches and aggregation bugs. I (1) fix an important bug in your auto time-crop (it mistakenly uses the Hz dimension instead of the time dimension), so the model sees the intended central time window rather than an overly narrow slice, and (2) add robust checkpoint loading that correctly handles common `.pth` formats (state dict under `state_dict`/`model` and `module.` prefixes), which otherwise silently ruins performance. I also keep your probability renormalization, but add a tiny post-softmax floor before averaging to reduce extreme probabilities that often worsen KL when miscalibrated, without changing core inference semantics. Everything else (model, fold averaging, transforms, submission alignment) remains the same, and it still produces `submission.csv` end-to-end.'
- What this solution (achieved 1.41937) has done: 'Your current score (1.41937, lower-is-better) is far above the target (0.52567), so we should improve inference quality with the smallest changes that keep the same model and fold-averaging logic. The most likely remaining high-impact mismatch is the time-axis handling: you currently feed the full spectrogram time width and then resize it to a square, which can heavily distort temporal patterns versus what these EfficientNetV2 spectrogram checkpoints typically expect. I add an auto center-crop that activates only when `CFG.time_crop==0`, choosing a conservative crop width tied to the model input size to reduce distortion while keeping your existing preprocessing (clip→log→per-freq/global norm) intact. I also make the checkpoint loading slightly more tolerant (still strict on matching keys after cleaning) to avoid silently broken loads that produce very poor KL, and keep the submission alignment + renormalization unchanged to guarantee a valid CSV.'
- What this solution (achieved 1.41937) has done: 'We keep your model, fold-averaging, and overall inference pipeline intact, and only make score-relevant fixes that typically cause very high KL when wrong. Specifically, we (1) switch the per-frequency normalization to match the common HMS baseline convention (normalize each frequency band across time *before* log/clip amplification issues propagate) while keeping your clip→log semantics, and (2) make the auto time-crop width derive from the *post-resize* target size more conservatively (to reduce temporal squashing without over-cropping). We also make the prediction flooring conditional only at the very end (after fold-mean) to avoid biasing the ensemble average toward uniform, which can worsen KL. Submission alignment and strict per-row renormalization remain unchanged to guarantee a valid CSV.'
- What this solution (achieved 1.41937) has done: 'Your current score (1.41937, lower-is-better) is far from the target (0.52567), so we should improve inference quality with minimal, score-relevant changes while keeping the same model and fold-averaging logic. The biggest likely remaining issue is a preprocessing mismatch with the provided EfficientNetV2 fold checkpoints: your current transform resizes the (Hz, Time) spectrogram directly to a square, which can distort time patterns; we instead center-crop the time axis to a width that matches the model’s expected square input before resizing. To avoid over/under-cropping across files, we compute a robust per-dataset crop width once (from the test spectrogram widths) and pass it into the Dataset, keeping all other steps (clip→log→normalization, softmax, fold-mean, renorm, submission alignment) unchanged. This should reduce KL materially if the checkpoints are present, and it remains fully runnable and still produces a valid `submission.csv` even if checkpoints are missing.'
- What this solution (achieved 1.41937) has done: 'Your score is far worse than the target (1.41937 vs 0.52567; lower is better), so we should improve with the smallest inference-only changes that keep your model/ensemble logic intact. The highest-impact issue here is that you’re predicting **one row per `test.csv` row (spectrogram_id)** and then averaging by `eeg_id`, but the competition evaluates one row per `eeg_id`; a safer, score-improving approach is to run inference **once per unique `eeg_id`** (using its single spectrogram) and avoid any unintended averaging/misalignment. I also make the auto time-crop robust by computing widths over the **unique** inference set (not the duplicated rows), which stabilizes preprocessing. Everything else (model, weights loading, transforms, softmax, fold-mean, renorm, submission writing) is preserved.'
- What this solution (achieved 1.41937) has done: 'Your current score (1.41937, lower-is-better) is far worse than the target (0.52567), so we should improve performance with the smallest inference-only changes. The most likely remaining cause of very high KL is a preprocessing mismatch with the EfficientNetV2 fold checkpoints: these HMS spectrogram baselines typically use a **fixed central ~50s crop** of the 10-minute spectrogram (about 300 time steps), whereas your auto-crop currently picks a much wider window tied to `img_size`, which gets aggressively squashed when resized to 512×512. I make the time crop default to a conservative, HMS-standard **300** (only when `CFG.time_crop==0`), while keeping your model, weights, softmax, fold-averaging, and submission alignment/renormalization unchanged. This should move KL down materially toward the target if checkpoints are present, and still produces a valid `submission.csv` in all cases.'
- What this solution (achieved 1.41937) has done: 'Your current score (1.41937, lower-is-better) is far above the target (0.52567), so we should improve it with minimal, inference-only changes while keeping your model/ensemble and all core logic intact. The most likely remaining high-impact mismatch is the **spectrogram preprocessing order**: many HMS EfficientNetV2 spectrogram baselines do `clip -> log -> per-freq normalize` (which you do), but *often also* apply a simple **min-max/standardization stabilization** by removing extreme outliers before normalization; a tiny, safe equivalent is to winsorize the logged spectrogram per-sample. I add a very small per-sample percentile clip in log-space (after `log`, before normalization) to reduce outlier dominance (which tends to blow up KL), without changing architecture or inference flow. Everything else (time_crop=300, norm_mode=per_freq, checkpoint loading, softmax, fold-mean, submission alignment, and per-row renorm) stays the same and it still writes a valid `submission.csv`.'
- What this solution (achieved 1.41937) has done: 'Your current score (1.41937, lower-is-better) is far from the target (0.52567), so we should improve it with minimal, inference-only changes that keep your model/ensemble logic intact. The biggest likely score issue is a preprocessing mismatch: these EfficientNetV2 spectrogram checkpoints are typically trained on **per-frequency normalization across time without extra percentile winsorization**, and the added percentile clip can wash out discriminative peaks and increase KL. I make the percentile clip optional and default it OFF when using checkpoints, while keeping your existing clip→log and per-frequency normalization, and I also ensure we load all test spectrogram `.npy` files for the **unique eeg_id set** only (same predictions, less overhead, less room for mismatch). Everything else—model, folds, softmax, averaging, submission alignment, and per-row renormalization—stays the same.'

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
ROOT = Path.cwd().parent
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
SRC = ROOT / "src"

DATA = INPUT / "hms-harmful-brain-activity-classification"
TRAIN_SPEC = DATA / "train_spectrograms"
TEST_SPEC = DATA / "test_spectrograms"

TRAINED_MODEL = INPUT / "hms-hbac-effnetv2-baseline"

TMP = ROOT / "tmp"
TRAIN_SPEC_SPLIT = TMP / "train_spectrograms_split"
TEST_SPEC_SPLIT = TMP / "test_spectrograms_split"
TMP.mkdir(exist_ok=True)
TRAIN_SPEC_SPLIT.mkdir(exist_ok=True)
TEST_SPEC_SPLIT.mkdir(exist_ok=True)

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
class CFG:
    model_name = "tf_efficientnetv2_l_in21ft1k"
    img_size = 512
    max_epoch = 7
    batch_size = 32
    lr = 1.0e-03
    weight_decay = 1.0e-02
    es_patience = 5
    seed = 1086
    deterministic = True
    enable_amp = True
    device = "cuda"

    time_crop: int = 0  # 0 => set below
    norm_mode: str = "per_freq"  # {"global","per_freq"}

    use_percentile_clip: bool = False
    pct_low: float = 0.5
    pct_high: float = 99.5


PRED_FLOOR: float = 0.0

device = torch.device(CFG.device if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 1086, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(CFG.seed, CFG.deterministic)



## === cell 4
test = pd.read_csv(DATA / "test.csv")
test.head()



## === cell 5
test_unique = test.drop_duplicates(subset=["eeg_id"], keep="first").reset_index(
    drop=True
)

missing = []
for spec_id in test_unique["spectrogram_id"].astype(str).values:
    npy_path = TEST_SPEC_SPLIT / f"{spec_id}.npy"
    if not npy_path.exists():
        missing.append(spec_id)

if len(missing) > 0:
    for spec_id in tqdm(missing, desc="Creating test .npy"):
        spec = pd.read_parquet(TEST_SPEC / f"{spec_id}.parquet")
        spec_arr = spec.fillna(0).values[:, 1:].T.astype("float32")  # (Hz, Time)
        np.save(TEST_SPEC_SPLIT / f"{spec_id}.npy", spec_arr)
else:
    print("All required test .npy files already exist.")




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
        time_crop: int = 0,
        norm_mode: str = "global",
        use_percentile_clip: bool = False,
        pct_low: float = 0.5,
        pct_high: float = 99.5,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.time_crop = int(time_crop) if time_crop is not None else 0
        self.norm_mode = str(norm_mode)

        self.use_percentile_clip = bool(use_percentile_clip)
        self.pct_low = float(pct_low)
        self.pct_high = float(pct_high)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        label = self.labels[index]

        img = np.load(img_path)  # (Hz, Time)

        crop_w = int(self.time_crop) if self.time_crop is not None else 0
        if crop_w > 0 and img.shape[1] > crop_w:
            t = img.shape[1]
            start = (t - crop_w) // 2
            img = img[:, start : start + crop_w]

        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)

        if self.use_percentile_clip:
            lo = np.percentile(img, self.pct_low).astype(np.float32)
            hi = np.percentile(img, self.pct_high).astype(np.float32)
            img = np.clip(img, lo, hi)

        eps = 1e-6
        if self.norm_mode == "per_freq":
            mean = img.mean(axis=1, keepdims=True).astype(np.float32)
            std = img.std(axis=1, keepdims=True).astype(np.float32)
            img = (img - mean) / (std + eps)
        else:
            img_mean = float(img.mean())
            img = img - img_mean
            img_std = float(img.std())
            img = img / (img_std + eps)

        img = img[..., None]  # (Hz, Time, C=1)
        img = self._apply_transform(img)

        return {"data": img, "target": label}

    def _apply_transform(self, img: np.ndarray):
        transformed = self.transform(image=img)
        img = transformed["image"]
        return img




## === cell 8
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


def get_test_path_label(test_df: pd.DataFrame):
    img_paths = []
    labels = np.full((len(test_df), 6), -1, dtype="float32")
    for spec_id in test_df["spectrogram_id"].astype(str).values:
        img_path = TEST_SPEC_SPLIT / f"{spec_id}.npy"
        img_paths.append(img_path)

    test_data = {"image_paths": img_paths, "labels": [l for l in labels]}
    return test_data


def get_test_transforms(CFG):
    test_transform = A.Compose(
        [
            A.Resize(p=1.0, height=CFG.img_size, width=CFG.img_size),
            ToTensorV2(p=1.0),
        ]
    )
    return test_transform




## === cell 9
def _extract_state_dict(state_obj: tp.Any) -> tp.Dict[str, torch.Tensor]:
    """
    Robustly handle common checkpoint formats and prefixes so we actually load the intended weights.
    """
    if isinstance(state_obj, dict):
        for k in ("state_dict", "model", "model_state_dict", "net"):
            if k in state_obj and isinstance(state_obj[k], dict):
                state_obj = state_obj[k]
                break
    if not isinstance(state_obj, dict):
        raise ValueError(
            "Unsupported checkpoint format: expected a dict-like state_dict."
        )

    if any(str(k).startswith("module.") for k in state_obj.keys()):
        state_obj = {k.replace("module.", "", 1): v for k, v in state_obj.items()}

    if any(str(k).startswith("model.") for k in state_obj.keys()):
        state_obj = {k.replace("model.", "", 1): v for k, v in state_obj.items()}

    return state_obj


def run_inference_loop(model, loader, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", leave=False):
            x = to_device(batch["data"], device)
            if CFG.enable_amp and device.type == "cuda":
                with amp.autocast(device_type="cuda", dtype=torch.float16):
                    y = model(x)
            else:
                y = model(x)

            p = y.softmax(dim=1).detach().cpu().numpy()

            if PRED_FLOOR is not None and PRED_FLOOR > 0:
                p = np.clip(p, PRED_FLOOR, None)
                p = p / p.sum(axis=1, keepdims=True)

            pred_list.append(p)
    pred_arr = np.concatenate(pred_list, axis=0)
    del pred_list
    return pred_arr




## === cell 10
if int(CFG.time_crop) == 0:
    CFG.time_crop = 300

print(
    f"Using time_crop={CFG.time_crop} (img_size={CFG.img_size}, norm_mode={CFG.norm_mode}, "
    f"use_percentile_clip={CFG.use_percentile_clip})"
)

smpl_sub = pd.read_csv(DATA / "sample_submission.csv")
test_path_label = get_test_path_label(test_unique)
test_transform = get_test_transforms(CFG)

test_dataset = HMSHBACSpecDataset(
    **test_path_label,
    transform=test_transform,
    time_crop=CFG.time_crop,
    norm_mode=CFG.norm_mode,
    use_percentile_clip=CFG.use_percentile_clip,
    pct_low=CFG.pct_low,
    pct_high=CFG.pct_high,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=CFG.batch_size,
    num_workers=2,
    shuffle=False,
    drop_last=False,
    pin_memory=torch.cuda.is_available(),
)

model_paths = [
    TRAINED_MODEL / f"best_model_fold{fold_id}.pth" for fold_id in range(N_FOLDS)
]
have_all_models = all(p.exists() for p in model_paths)

if have_all_models:
    test_preds_arr = np.zeros((N_FOLDS, len(test_unique), N_CLASSES), dtype=np.float32)
    for fold_id in range(N_FOLDS):
        print(f"\n[fold {fold_id}]")
        model_path = TRAINED_MODEL / f"best_model_fold{fold_id}.pth"
        model = HMSHBACSpecModel(
            model_name=CFG.model_name,
            pretrained=False,
            num_classes=N_CLASSES,
            in_channels=1,
        )
        raw_state = torch.load(model_path, map_location="cpu")
        state = _extract_state_dict(raw_state)
        model.load_state_dict(state, strict=True)
        model.to(device)

        test_pred_fold = run_inference_loop(model, test_loader, device)
        test_preds_arr[fold_id] = test_pred_fold

        del model, raw_state, state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    test_pred = test_preds_arr.mean(axis=0)

    post_floor = 1e-8
    test_pred = np.clip(test_pred, post_floor, None)
    test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)
else:
    print(f"WARNING: Missing fold checkpoints under: {TRAINED_MODEL}")
    print(
        "Creating a valid baseline submission from train label priors (score will be suboptimal)."
    )
    train = pd.read_csv(DATA / "train.csv")
    vote_sums = train[CLASSES].sum(axis=0).astype(np.float64).values
    prior = vote_sums / vote_sums.sum()
    test_pred = np.tile(prior[None, :], (len(test_unique), 1)).astype(np.float32)



## === cell 11
test_pred_df = pd.DataFrame(test_pred, columns=CLASSES)
test_pred_df = pd.concat([test_unique[["eeg_id"]].copy(), test_pred_df], axis=1)

sub = smpl_sub[["eeg_id"]].merge(test_pred_df, on="eeg_id", how="left")

uniform = np.full((1, N_CLASSES), 1.0 / N_CLASSES, dtype=np.float64)
missing_mask = sub[CLASSES].isna().any(axis=1)
if missing_mask.any():
    sub.loc[missing_mask, CLASSES] = uniform

probs = sub[CLASSES].to_numpy(dtype=np.float64)
probs = np.clip(probs, 1e-12, None)
probs /= probs.sum(axis=1, keepdims=True)
sub[CLASSES] = probs

out_path = Path("submission.csv")
sub.to_csv(out_path, index=False)

print(f"Wrote: {out_path.resolve()}")
print("submission shape:", sub.shape)
sub.head()
