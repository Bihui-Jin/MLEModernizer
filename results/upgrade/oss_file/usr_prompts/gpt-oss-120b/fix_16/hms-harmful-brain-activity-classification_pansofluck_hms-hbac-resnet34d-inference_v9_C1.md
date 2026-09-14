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

1.64506

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I guard the model loading with a try/except so missing checkpoint files no longer crash the notebook, and fall back to a simple constant‑probability baseline derived from the average class distribution in the training set. I also replace the merge‑based submission creation with a direct concatenation that guarantees the submission has exactly the same number of rows as the test set and that the probabilities sum to one.'
- What this solution (achieved 1.64506) has done: 'I make two minimal but effective adjustments:  
1. Collect predictions only from folds whose checkpoint files are actually found, and average them instead of averaging in zero‑filled placeholders (which dilute performance).  
2. Replace the “all‑zero” fallback with a smarter baseline that uses per‑patient class probability averages derived from the training data (falling back to the global mean when a patient is unseen). This more informative prior should lower the KL‑divergence score toward the target.'
- What this solution (achieved 1.64506) has done: 'I add a finer‑grained fallback: use per‑eeg_id vote averages when available (they are more specific than per‑patient), then fall back to the existing per‑patient averages, and finally to the global class mean. This small hierarchy should improve the baseline probabilities and thus lower the KL‑divergence score toward the target while leaving the core model logic unchanged.'
- What this solution (achieved 0.72454) has done: 'I add a tiny smoothing step after the model‑or‑baseline predictions: blend a small weight of the global class mean into the predictions and apply a mild temperature scaling (T = 1.2). This keeps the core modeling unchanged but makes the output probabilities a bit less extreme, which should lower the KL‑divergence and move the score toward the target.'
- What this solution (achieved 0.73494) has done: 'I slightly increase the smoothing of the model predictions by (1) computing a hierarchical baseline for every test row and blending it with the averaged fold predictions, (2) raising the global‑mean blend weight and temperature a bit to make the final probabilities less extreme. These minimal adjustments keep the core architecture and training unchanged while moving the KL‑divergence lower toward the target score.'
- What this solution (achieved 1.63065) has done: 'I lower the KL‑divergence by giving the hierarchical baseline and the global‑mean prior much more influence and soften the probabilities further. In the inference cell I increase the baseline blend to 90 %, raise the global‑mean blend to 10 %, and use a higher temperature (1.5). These tweaks keep the original architecture and training untouched while moving the score closer to the target.'
- What this solution (achieved 1.64506) has done: 'I lower the heavy baseline weighting and reduce the extra smoothing so the model’s predictions have more influence, which should decrease the KL‑divergence and move the score closer to the target. Specifically, I change the baseline blend from 0.90 to 0.5, the global‑mean blend from 0.10 to 0.05, and set the temperature to 1.0 (no smoothing). These minimal adjustments keep the core architecture untouched while giving the model more weight and a sharper probability distribution.'
- What this solution (achieved 1.64506) has done: 'I lower the influence of the hierarchical baseline and the global‑mean prior, giving the model predictions more weight, and add a mild temperature scaling (T = 1.2) to soften overly confident outputs. These small adjustments keep the original architecture and training untouched while moving the KL‑divergence score closer to the target.'
- What this solution (achieved 1.64506) has done: 'I adjust the inference blending parameters to give the hierarchical baseline and global‑mean priors a larger influence and increase the temperature scaling, which smooths the predictions and should lower the KL‑divergence toward the target. The core model loading, data handling, and submission logic remain unchanged.'
- What this solution (achieved 1.64506) has done: 'I keep the overall pipeline unchanged and only adjust the post‑processing blending that combines the model’s predictions with the hierarchical baseline and the global class mean. By increasing the baseline and global‑mean blend weights and applying a stronger temperature smoothing, the final probabilities become less extreme, which is expected to lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.64506) has done: 'The changes reduce the heavy reliance on the hierarchical baseline and global‑mean priors and remove the aggressive temperature scaling, giving the model’s predictions more influence and sharpening the final probabilities, which is expected to lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.64506) has done: 'I keep the overall pipeline unchanged and only adjust the post‑processing blending that combines the model’s predictions with the hierarchical baseline and the global class‑mean prior. By increasing the baseline and global‑mean weights and applying a mild temperature scaling (T ≈ 1.2), the final probabilities become less extreme and should lower the KL‑divergence, moving the score closer to the target.'
- What this solution (achieved 1.64506) has done: 'I keep the overall pipeline unchanged and simply give the hierarchical baseline (per‑eeg → per‑patient → global mean) full weight, removing the influence of the model predictions, the extra global‑mean blend, and the temperature scaling. This makes the final probabilities equal to the well‑calibrated baseline, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.64506) has done: 'I adjust the post‑processing blending so that the model’s predictions contribute alongside the hierarchical baseline, add a modest global‑mean blend, and apply a mild temperature scaling (T ≈ 1.3). These small parameter tweaks keep the core pipeline unchanged while moving the KL‑divergence closer to the target 0.525 (aiming to lower the current 1.645).'
- What this solution (achieved 1.64506) has done: 'I slightly increase the reliance on the calibrated hierarchical baseline and the global‑mean prior, and apply a modest temperature smoothing. This keeps the original architecture and training untouched while moving the KL‑divergence score lower toward the target.'

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


device = torch.device(CFG.device)




## === cell 4
test = pd.read_csv(DATA / "test.csv")




## === cell 5
for spec_id in test["spectrogram_id"]:
    spec = pd.read_parquet(TEST_SPEC / f"{spec_id}.parquet")

    spec_arr = (
        spec.fillna(0).values[:, 1:].T.astype("float32")
    )  # (Hz, Time) = (400, 300)

    np.save(TEST_SPEC_SPLIT / f"{spec_id}.npy", spec_arr)




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

        img = np.load(img_path)  # shape: (Hz, Time) = (400, 300)

        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)

        eps = 1e-6
        img_mean = img.mean(axis=(0, 1))
        img = img - img_mean
        img_std = img.std(axis=(0, 1))
        img = img / (img_std + eps)

        img = img[..., None]  # shape: (Hz, Time) -> (Hz, Time, Channel)
        img = self._apply_transform(img)

        return {"data": img, "target": label}

    def _apply_transform(self, img: np.ndarray):
        """apply transform to image and mask"""
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


def get_test_path_label(test: pd.DataFrame):
    """Get file path and dummy target info."""

    img_paths = []
    labels = np.full((len(test), 6), -1, dtype="float32")
    for spec_id in test["spectrogram_id"].values:
        img_path = TEST_SPEC_SPLIT / f"{spec_id}.npy"
        img_paths.append(img_path)

    test_data = {"image_paths": img_paths, "labels": [l for l in labels]}

    return test_data


def get_test_transforms(CFG):
    test_transform = A.Compose(
        [A.Resize(p=1.0, height=CFG.img_size, width=CFG.img_size), ToTensorV2(p=1.0)]
    )
    return test_transform




## === cell 9
def run_inference_loop(model, loader, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            x = to_device(batch["data"], device)
            y = model(x)
            pred_list.append(y.softmax(dim=1).detach().cpu().numpy())

    pred_arr = np.concatenate(pred_list)
    del pred_list
    return pred_arr




## === cell 10
fold_predictions = []  # list of arrays, each (len(test), N_CLASSES)

test_path_label = get_test_path_label(test)
test_transform = get_test_transforms(CFG)
test_dataset = HMSHBACSpecDataset(**test_path_label, transform=test_transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=CFG.batch_size,
    num_workers=4,
    shuffle=False,
    drop_last=False,
)

train = pd.read_csv(DATA / "train.csv")
vote_cols = CLASSES
vote_sums = train[vote_cols].sum(axis=1).replace(0, np.nan)
train_probs = train[vote_cols].div(vote_sums, axis=0).fillna(0)
global_mean = train_probs.mean().values  # shape (6,)

eeg_mean = train_probs.groupby(train["eeg_id"]).mean()
patient_mean = train_probs.groupby(train["patient_id"]).mean()
baseline_rows = []
for _, row in test.iterrows():
    eid = row["eeg_id"]
    pid = row["patient_id"]
    if eid in eeg_mean.index:
        baseline_rows.append(eeg_mean.loc[eid].values)
    elif pid in patient_mean.index:
        baseline_rows.append(patient_mean.loc[pid].values)
    else:
        baseline_rows.append(global_mean)
baseline_pred = np.vstack(baseline_rows)  # (len(test), 6)

for fold_id in range(N_FOLDS):
    print(f"\n[fold {fold_id}]")
    model_path = TRAINED_MODEL / f"best_model_fold{fold_id}.pth"
    try:
        model = HMSHBACSpecModel(
            model_name=CFG.model_name, pretrained=False, num_classes=6, in_channels=1
        )
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        test_pred_fold = run_inference_loop(model, test_loader, device)
        fold_predictions.append(test_pred_fold)
        del model
        torch.cuda.empty_cache()
        gc.collect()
    except FileNotFoundError:
        print(f"⚠️  Model checkpoint {model_path} not found – skipping this fold.")
        continue

if len(fold_predictions) == 0:
    test_pred = baseline_pred
else:
    model_pred = np.mean(np.stack(fold_predictions, axis=0), axis=0)

    baseline_blend = 0.85  # give baseline 85 % weight, model 15 %
    test_pred = (1 - baseline_blend) * model_pred + baseline_blend * baseline_pred

    blend_weight = 0.15  # 15 % global mean contribution
    test_pred = (1 - blend_weight) * test_pred + blend_weight * global_mean

    temperature = 1.5
    test_pred = np.power(test_pred, 1.0 / temperature)

row_sums = test_pred.sum(axis=1, keepdims=True)
test_pred = test_pred / (row_sums + 1e-12)

test_pred_df = pd.DataFrame(test_pred, columns=CLASSES)




## === cell 11
submission = pd.concat(
    [test[["eeg_id"]].reset_index(drop=True), test_pred_df.reset_index(drop=True)],
    axis=1,
)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path} – shape {submission.shape}")
