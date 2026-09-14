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
mne==1.10.2
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

0.431848051039456

# 6. Current score

1.26806

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the runtime failure by making the code robust to missing external weight files: it automatically fall back to a valid “prior” prediction derived from `train.csv` vote frequencies when the `.pth` files aren’t available, so it always produces a submission. I also fix a logic bug in `brain_lead` (it mistakenly uses `RP` twice instead of `RR`) and remove references to undefined variables in the dataset (`eeg_offside_min/max`) to prevent hidden crashes. Finally, I ensure the final predictions have shape `(n_test, 6)`, are valid probabilities (clipped + row-normalized), and write `submission.csv` with the exact required columns.'
- What this solution (achieved 1.1488) has done: 'I fix the hardcoded parquet paths so they point to the actual competition dataset directory, which is why your DataLoader workers can’t find files like `/kaggle/input/.../test_spectrograms/*.parquet`. I also make the train/test parquet base paths configurable and use them consistently for both fallback training and normal inference, without changing the model architectures or training/inference logic. Finally, I add a small guard so that even if a few parquet files are missing/corrupt, the pipeline still produces valid probabilities and writes a proper `submission.csv` with the required columns that sum to 1.'
- What this solution (achieved 1.42416) has done: 'Your current score (1.1488, lower-is-better) is far worse than the target (0.4318), so we should improve calibration/validity with minimal risk while preserving the same model/inference logic. The biggest low-risk gain here is fixing a spectrogram slicing bug (`r` never increments) that currently repeats the same 300-row band for all 4 regions, degrading features and predictions. I also make the test spectrogram extraction robust to varying parquet shapes by explicitly slicing `:400` columns (4*100) after dropping the first column, which avoids silent misalignment and keeps the intended 4-region layout. Finally, I add a tiny post-processing “prior blend” (small epsilon mix with the global class prior from train) to reduce overconfident wrong predictions and typically improve KL divergence without changing the model architecture or training loop.'
- What this solution (achieved 1.41933) has done: 'Your current score (1.42416, lower-is-better) is far worse than the target (0.43185), so we should improve predicted probability calibration with the smallest safe change that doesn’t alter the model architectures or training/inference loops. The biggest low-risk issue is that the fallback training currently trains the spectrogram model on *test* spectrogram files (because `AlaskaDataIter` always reads `test_spectrograms` unless `is_training=True`), which makes fallback training ineffective and hurts KL. I fix this by adding a minimal `mode`/path switch so training reads from `train_spectrograms` while inference still reads from `test_spectrograms`—no change to feature extraction logic, just correct file source. Additionally, I slightly increase the prior blending strength (still small) to reduce overconfident errors, which typically decreases KL for this competition.'
- What this solution (achieved 1.39735) has done: 'Your score is far worse than the target (lower is better), so we should improve KL with minimal risk while preserving your exact model/training/inference structure. The biggest low-risk calibration issue in your current pipeline is that the fallback training uses only 1 epoch and then relies on softmax outputs that can still be too peaky for KL; we can safely reduce KL by (1) slightly increasing label smoothing during fallback training and (2) adding a tiny temperature scaling on logits at inference (a pure calibration step that keeps the same model and features). Additionally, because the ensemble/fallback can still be overconfident, we mildly strengthen the prior blend (still small) and apply a final floor-and-renormalize to guarantee well-behaved probabilities. These changes are minimal, don’t change architectures or data extraction, and are aimed specifically at reducing KL toward your target.'
- What this solution (achieved 1.27396) has done: 'I fix the immediate runtime error by adding the missing `fallback_num_workers` configuration key and using it consistently in the fallback DataLoaders, so `predictions` is always created. I also add a small safety fallback so that if fallback training/inference ever fails for any reason, the code still produces a valid submission by reverting to the global class-prior from `train.csv` (score-neutral vs crashing). Finally, I keep the existing normalization/prior-blend/floor logic intact to ensure every row sums to 1 and the CSV matches the required column names.'
- What this solution (achieved 1.26806) has done: 'Your current score (1.27396, lower-is-better) is still far above the target (0.43185), so we should improve calibration/robustness with minimal, low-risk changes that preserve your exact model/training/inference structure. The biggest legitimate issue left is that when pretrained weights are missing (your common case), the fallback spectrogram model is trained from scratch with a strong dropout and only 1 epoch, which tends to output poorly-calibrated probabilities for KL; we can reduce KL by initializing the classifier bias to the global class prior so early training starts calibrated. Next, we add a tiny, post-softmax “probability sharpening” power transform (<1) to mildly increase entropy and reduce KL blow-ups from overconfident wrong predictions, without changing the model or data. Finally, we keep your prior blending and floor+renormalization, but slightly strengthen the blend to further stabilize predictions toward reasonable distributions.'

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
import mne

import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader



## === cell 1
CFG = {
    "batch_size": 32,
    "num_worker": 4,
    "fallback_num_workers": 2,
    "data": "/kaggle/input/hms-harmful-brain-activity-classification/test.csv",
    "base_dir": "/kaggle/input/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification",
    "weights_spec": [
        "/kaggle/input/hms-baseline/fold0_epoch_4_val_loss_0.549755.pth",
        "/kaggle/input/hms-baseline/fold1_epoch_4_val_loss_0.524753.pth",
        "/kaggle/input/hms-baseline/fold2_epoch_4_val_loss_0.511551.pth",
        "/kaggle/input/hms-baseline/fold3_epoch_4_val_loss_0.548698.pth",
        "/kaggle/input/hms-baseline/fold4_epoch_3_val_loss_0.666013.pth",
    ],
    "weights_eeg": [
        "/kaggle/input/hms-eeg/fold0_epoch_4_val_loss_0.584720.pth",
        "/kaggle/input/hms-eeg/fold1_epoch_4_val_loss_0.590993.pth",
        "/kaggle/input/hms-eeg/fold2_epoch_4_val_loss_0.603758.pth",
        "/kaggle/input/hms-eeg/fold3_epoch_4_val_loss_0.580059.pth",
        "/kaggle/input/hms-eeg/fold4_epoch_4_val_loss_0.545819.pth",
    ],
    "weights_eeg_16chans": [
        "/kaggle/input/hms-eeg-16chans/fold0_epoch_4_val_loss_0.596976.pth",
        "/kaggle/input/hms-eeg-16chans/fold1_epoch_4_val_loss_0.601446.pth",
        "/kaggle/input/hms-eeg-16chans/fold2_epoch_4_val_loss_0.632576.pth",
        "/kaggle/input/hms-eeg-16chans/fold3_epoch_4_val_loss_0.651490.pth",
        "/kaggle/input/hms-eeg-16chans/fold4_epoch_4_val_loss_0.614693.pth",
    ],
    "flip": True,
    "fallback_train": True,
    "fallback_train_rows": 12000,  # cap rows read/used to keep runtime low
    "fallback_epochs": 1,
    "fallback_steps_per_epoch": 220,  # hard cap on gradient steps to stay within timeout
    "fallback_lr": 2e-4,
    "fallback_weight_decay": 1e-4,
    "fallback_label_smoothing": 0.03,
    "prior_blend": 0.18,
    "inference_temperature": 1.15,
    "final_prob_floor": 1e-4,
    "prob_power": 0.90,
}

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


def _normalize_probs(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    row_sums = p.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    p = p / row_sums
    return p.astype(np.float32)


def prior_from_train(train_csv_path: str, n_rows: int) -> np.ndarray:
    train_df = pd.read_csv(train_csv_path, usecols=TARGETS)
    votes = train_df[TARGETS].sum(axis=0).values.astype(np.float64)
    probs = votes / votes.sum()
    probs = np.clip(probs, 1e-6, 1.0)
    probs = probs / probs.sum()
    return np.tile(probs[None, :], (n_rows, 1)).astype(np.float32)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _resolve_base_dir(cfg_base: str) -> str:
    candidates = [
        cfg_base,
        "/kaggle/input/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification",
        "/kaggle/input/hms-harmful-brain-activity-classification",
        "/kaggle/input",
    ]
    for c in candidates:
        if not isinstance(c, str):
            continue
        if os.path.isdir(os.path.join(c, "test_spectrograms")) and os.path.isdir(
            os.path.join(c, "test_eegs")
        ):
            return c
    return cfg_base


CFG["base_dir"] = _resolve_base_dir(CFG["base_dir"])
set_seed(42)




## === cell 2
class AlaskaDataIter:
    def __init__(
        self,
        df,
        training_flag=False,
        shuffle=False,
        use_eeg=False,
        ll=0,
        rr=20,
        base_dir=None,
        mode: str = "test",  # "train" or "test"
    ):

        self.ll = ll
        self.rr = rr
        self.training_flag = training_flag
        self.shuffle = shuffle
        self.raw_data_set_size = None
        self.df = df
        self.base_dir = base_dir or CFG["base_dir"]
        self.mode = mode

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

        self.LL = ["Fp1", "F7", "T3", "T5", "O1"]
        self.RR = ["Fp2", "F8", "T4", "T6", "O2"]
        self.LP = ["Fp1", "F3", "C3", "P3", "O1"]
        self.RP = ["Fp2", "F4", "C4", "P4", "O2"]

        self.leads_dict = {value: index for index, value in enumerate(self.eeg_nms)}
        self.use_eeg = use_eeg

    def __getitem__(self, item):
        return self.single_map_func(self.df.iloc[item], self.training_flag)

    def __len__(self):
        return len(self.df)

    def brain_lead(self, waves):
        waves = copy.deepcopy(waves)
        brain_leads = [self.LL, self.LP, self.RP, self.RR]
        leads = []
        for combine in brain_leads:
            for i in range(len(combine) - 1):
                tmp_lead = (
                    waves[self.leads_dict[combine[i]]]
                    - waves[self.leads_dict[combine[i + 1]]]
                )
                leads.append(tmp_lead)
        data = np.concatenate([leads], axis=0)
        return data

    def _eeg_path(self, eeg_id: int) -> str:
        folder = "train_eegs" if self.mode == "train" else "test_eegs"
        return os.path.join(self.base_dir, folder, f"{int(eeg_id)}.parquet")

    def _spec_path(self, spec_id: int) -> str:
        folder = "train_spectrograms" if self.mode == "train" else "test_spectrograms"
        return os.path.join(self.base_dir, folder, f"{int(spec_id)}.parquet")

    def single_map_func(self, dp, is_training):
        if self.use_eeg:
            eeg_path = self._eeg_path(dp["eeg_id"])
            try:
                eeg = pd.read_parquet(eeg_path)
                offset = 0
                eeg = eeg.iloc[offset * 200 : (offset + 50) * 200]
                waves = eeg.values
                waves = np.transpose(waves, axes=[1, 0])

                for i in range(waves.shape[0]):
                    m = np.nanmean(waves[i])
                    if np.isnan(waves[i]).mean() < 1:
                        waves[i] = np.nan_to_num(waves[i], nan=m)
                    else:
                        waves[i] = 0

                waves = np.array(waves, dtype=np.float64)
                waves = mne.filter.filter_data(
                    waves, 200, self.ll, self.rr, verbose=False
                )
                waves = self.brain_lead(waves)
                data = waves
                return data.astype(np.float32)
            except Exception:
                return np.zeros((16, 10000), dtype=np.float32)
        else:
            spec_id = dp["spectrogram_id"]
            spec_path = self._spec_path(spec_id)
            try:
                spec = pd.read_parquet(spec_path).values

                if spec.ndim != 2 or spec.shape[1] < 2:
                    raise ValueError("Unexpected spectrogram shape")
                spec = spec[:, 1:]  # drop first col (time/index)
                if spec.shape[1] >= 400:
                    spec = spec[:, :400]
                else:
                    pad = 400 - spec.shape[1]
                    spec = np.pad(
                        spec, ((0, 0), (0, pad)), mode="constant", constant_values=0.0
                    )

                total_rows = spec.shape[0]
                needed = 300 * 4
                if total_rows >= needed:
                    start = (total_rows - needed) // 2
                    spec = spec[start : start + needed, :]
                else:
                    spec = np.pad(
                        spec,
                        ((0, needed - total_rows), (0, 0)),
                        mode="constant",
                        constant_values=0.0,
                    )

                images = []
                r = 0
                for region in range(4):
                    img = spec[r : r + 300, region * 100 : (region + 1) * 100].T
                    r += 300

                    img = np.clip(img, np.exp(-4), np.exp(8))
                    img = np.log(img)
                    img = np.nan_to_num(img, nan=0.0)
                    images.append(img)

                images = np.stack(images, -1)
                data = np.transpose(images, [2, 0, 1])
                return data.astype(np.float32)
            except Exception:
                return np.zeros((4, 100, 300), dtype=np.float32)




## === cell 3
class NetSpec(nn.Module):
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
class Transform(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.wave_transform = torchaudio.transforms.Spectrogram(
            n_fft=1024, hop_length=50, power=1
        )
        self.am2db = torchaudio.transforms.AmplitudeToDB(stype="magnitude", top_db=80)
        self.resizer = nn.UpsamplingBilinear2d(size=[160, 320])

    def forward(self, x):
        image = self.wave_transform(x)
        image = self.am2db(image)
        n, c, h, w = image.size()
        image = image[:, :, : int(20 / 100 * h + 30), :]
        image = torch.reshape(image, shape=[n, 4, -1, w])
        return image


class NetEeg(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.preprocess = Transform()
        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=4)
        self.fc = nn.Linear(2048, 6, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bs = x.size(0)
        x = self.preprocess(x)
        x = self.model.forward_features(x)
        x = self.avg(x)
        x = x.view(bs, -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x




## === cell 5
class PositionalEncoder(nn.Module):
    def __init__(self, d_model=16):
        super(PositionalEncoder, self).__init__()
        position = torch.arange(0, 16)
        pe = position.reshape(1, 16, 1, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x[:, :, 0:1, 0:1] = self.pe
        return x


class Transform2(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.wave_transform = torchaudio.transforms.Spectrogram(
            n_fft=1024, hop_length=50, power=1
        )
        self.am2db = torchaudio.transforms.AmplitudeToDB(stype="magnitude", top_db=80)
        self.resizer = nn.UpsamplingBilinear2d(size=[160, 320])
        self.position = PositionalEncoder()

    def forward(self, x):
        image = self.wave_transform(x)
        image = self.am2db(image)
        n, c, h, w = image.size()
        image = image[:, :, : int(20 / 100 * h + 10), :]
        image = self.position(image)
        return image


class NetEeg16chans(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.preprocess = Transform2()
        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=16)
        self.fc = nn.Linear(2048, 6, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bs = x.size(0)
        x = self.preprocess(x)
        x = self.model.forward_features(x)
        x = self.avg(x)
        x = x.view(bs, -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x




## === cell 6
def inference_function(test_loader, model, device, temperature: float = 1.0):
    model.eval()
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for X in tqdm_test_loader:
            X = X.to(device)
            with torch.no_grad():
                logits = model(X)
                if temperature and float(temperature) != 1.0:
                    logits = logits / float(temperature)
                y_preds = torch.softmax(logits, dim=1)
            preds.append(y_preds.detach().cpu().numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 7
test_df = pd.read_csv(CFG["data"])
test_df.head(5)




## === cell 8
class AlaskaTrainSpecIter(AlaskaDataIter):
    def __init__(self, df, training_flag=True, shuffle=False, base_dir=None):
        super().__init__(
            df,
            training_flag=training_flag,
            shuffle=shuffle,
            use_eeg=False,
            base_dir=base_dir,
            mode="train",
        )

    def __getitem__(self, item):
        dp = self.df.iloc[item]
        x = self.single_map_func(dp, self.training_flag)
        y = dp[TARGETS].values.astype(np.float32)
        y = y / (y.sum() + 1e-6)  # votes -> probability distribution
        return x, y


def train_fallback_spec_model(
    train_df: pd.DataFrame, device: torch.device, train_prior: np.ndarray
) -> NetSpec:
    model = NetSpec().to(device)
    model.train()

    with torch.no_grad():
        prior = np.asarray(train_prior, dtype=np.float64)
        prior = np.clip(prior, 1e-6, 1.0)
        prior = prior / prior.sum()
        bias = np.log(prior).astype(np.float32)
        model.fc.bias.copy_(torch.from_numpy(bias).to(device))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG["fallback_lr"],
        weight_decay=CFG["fallback_weight_decay"],
    )

    ls = float(CFG["fallback_label_smoothing"])

    train_ds = AlaskaTrainSpecIter(
        train_df, training_flag=True, shuffle=True, base_dir=CFG["base_dir"]
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG["batch_size"],
        num_workers=int(CFG["fallback_num_workers"]),
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    steps_cap = int(CFG["fallback_steps_per_epoch"])
    for epoch in range(int(CFG["fallback_epochs"])):
        pbar = tqdm(
            train_loader,
            desc=f"FallbackTrain(epoch={epoch+1})",
            total=min(len(train_loader), steps_cap),
        )
        for step, (x, y) in enumerate(pbar):
            if step >= steps_cap:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if ls > 0:
                y = (1.0 - ls) * y + ls / y.shape[1]

            logits = model(x)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = -(y * log_probs).sum(dim=1).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": float(loss.detach().cpu().item())})

    model.eval()
    return model


predictions_list = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _existing_paths(paths):
    return [p for p in paths if isinstance(p, str) and os.path.exists(p)]


spec_weights = _existing_paths(CFG["weights_spec"])
eeg_weights = _existing_paths(CFG["weights_eeg"])
eeg16_weights = _existing_paths(CFG["weights_eeg_16chans"])

any_weights = len(spec_weights) + len(eeg_weights) + len(eeg16_weights)

temp = float(CFG.get("inference_temperature", 1.0))

train_csv_path = os.path.join(CFG["base_dir"], "train.csv")

train_prior_1d = (
    pd.read_csv(train_csv_path, usecols=TARGETS)[TARGETS]
    .sum(axis=0)
    .values.astype(np.float64)
)
train_prior_1d = np.clip(train_prior_1d, 1e-6, np.inf)
train_prior_1d = train_prior_1d / train_prior_1d.sum()

try:
    if any_weights == 0:
        if CFG["fallback_train"]:
            use_cols = ["spectrogram_id"] + TARGETS
            train_df = pd.read_csv(train_csv_path, usecols=use_cols)

            if len(train_df) > int(CFG["fallback_train_rows"]):
                train_df = train_df.sample(
                    n=int(CFG["fallback_train_rows"]), random_state=42
                ).reset_index(drop=True)

            test_dataset = AlaskaDataIter(
                test_df,
                training_flag=False,
                shuffle=False,
                use_eeg=False,
                base_dir=CFG["base_dir"],
                mode="test",
            )
            test_loader = DataLoader(
                test_dataset,
                CFG["batch_size"],
                num_workers=int(CFG["fallback_num_workers"]),
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
            )

            model = train_fallback_spec_model(
                train_df, device=device, train_prior=train_prior_1d
            )
            prediction_dict = inference_function(
                test_loader, model, device, temperature=temp
            )
            predictions = prediction_dict["predictions"].astype(np.float32)

            del model, test_loader, test_dataset, train_df
            torch.cuda.empty_cache()
            gc.collect()
        else:
            predictions = prior_from_train(train_csv_path, n_rows=len(test_df))
    else:
        for model_weight in spec_weights:
            test_dataset = AlaskaDataIter(
                test_df,
                training_flag=False,
                shuffle=False,
                use_eeg=False,
                base_dir=CFG["base_dir"],
                mode="test",
            )
            test_loader = DataLoader(
                test_dataset,
                CFG["batch_size"],
                num_workers=CFG["num_worker"],
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
            )

            model = NetSpec()
            state_dict = torch.load(model_weight, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)

            prediction_dict = inference_function(
                test_loader, model, device, temperature=temp
            )
            predictions_list.append(prediction_dict["predictions"])
            del model, state_dict, test_loader, test_dataset
            torch.cuda.empty_cache()
            gc.collect()

        for model_weight in eeg_weights:
            test_dataset = AlaskaDataIter(
                test_df,
                training_flag=False,
                shuffle=False,
                use_eeg=True,
                base_dir=CFG["base_dir"],
                mode="test",
            )
            test_loader = DataLoader(
                test_dataset,
                CFG["batch_size"],
                num_workers=CFG["num_worker"],
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
            )

            model = NetEeg()
            state_dict = torch.load(model_weight, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)

            prediction_dict = inference_function(
                test_loader, model, device, temperature=temp
            )
            predictions_list.append(prediction_dict["predictions"])
            del model, state_dict, test_loader, test_dataset
            torch.cuda.empty_cache()
            gc.collect()

        for model_weight in eeg16_weights:
            test_dataset = AlaskaDataIter(
                test_df,
                training_flag=False,
                shuffle=False,
                use_eeg=True,
                ll=0.1,
                rr=20,
                base_dir=CFG["base_dir"],
                mode="test",
            )
            test_loader = DataLoader(
                test_dataset,
                CFG["batch_size"],
                num_workers=CFG["num_worker"],
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
            )

            model = NetEeg16chans()
            state_dict = torch.load(model_weight, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)

            prediction_dict = inference_function(
                test_loader, model, device, temperature=temp
            )
            predictions_list.append(prediction_dict["predictions"])
            del model, state_dict, test_loader, test_dataset
            torch.cuda.empty_cache()
            gc.collect()

        predictions = np.mean(np.stack(predictions_list, axis=0), axis=0).astype(
            np.float32
        )

except Exception as e:
    print("Fallback to prior due to error during model path:", repr(e))
    predictions = prior_from_train(train_csv_path, n_rows=len(test_df))

if (
    predictions.ndim != 2
    or predictions.shape[0] != len(test_df)
    or predictions.shape[1] != 6
):
    raise ValueError(
        f"Bad predictions shape {predictions.shape}; expected ({len(test_df)}, 6)."
    )

predictions = _normalize_probs(predictions, eps=1e-6)

prob_power = float(CFG.get("prob_power", 1.0))
if prob_power and abs(prob_power - 1.0) > 1e-8:
    predictions = _normalize_probs(np.power(predictions, prob_power), eps=1e-6)

prior = prior_from_train(train_csv_path, n_rows=len(test_df))
alpha = float(CFG.get("prior_blend", 0.0))
if alpha > 0:
    predictions = _normalize_probs(
        (1.0 - alpha) * predictions + alpha * prior, eps=1e-6
    )

floor = float(CFG.get("final_prob_floor", 0.0))
if floor and floor > 0:
    predictions = _normalize_probs(np.clip(predictions, floor, 1.0), eps=floor)



## === cell 9
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions
sub.to_csv("submission.csv", index=False)
print(f"Using base_dir={CFG['base_dir']}")
print(f"Submission shape: {sub.shape}")
print(sub.head())
print(
    "Row sums (min/mean/max):",
    sub[TARGETS].sum(axis=1).min(),
    sub[TARGETS].sum(axis=1).mean(),
    sub[TARGETS].sum(axis=1).max(),
)
