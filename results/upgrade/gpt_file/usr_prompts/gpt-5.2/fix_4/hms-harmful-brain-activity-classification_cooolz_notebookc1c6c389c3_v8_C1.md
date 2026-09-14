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

4.92373

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime blocker by removing the missing external weight dependency and falling back to a safe, deterministic baseline predictor when weights are not found, so the notebook always produces a valid `submission.csv`. I also fix a logic bug in spectrogram slicing (the row offset `r` was never updated) and ensure the dataset always returns a consistent `float32` tensor shape. Finally, I guarantee the submission probabilities are valid for the KL metric by clipping to a small epsilon and renormalizing each row to sum to 1, preventing submission failures.'
- What this solution (achieved 4.92373) has done: 'I fix the DataLoader crash by making the spectrogram slicing robust to variable spectrogram shapes (some files can be shorter than expected), padding/cropping each region to a consistent (100,300) before stacking. This is a runtime-only fix that preserves the same core spectrogram-based inference logic and model architecture, but ensures every sample returns a consistent float32 tensor shape. I also make inference deterministic by setting `num_workers=0` (avoids worker crashes/overhead in this environment) and add a safe fallback that produces valid uniform probabilities if any unexpected issue occurs during model inference. Finally, I keep the required submission format and explicitly clip+renormalize probabilities so every row sums to 1 for the KL metric.'

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

    def single_map_func(self, dp, is_training):
        """Build 4-channel spectrogram image tensor (C,H,W)."""
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
            waves = self.butter_lowpass_filter(waves)
            avg_waves = self.avg_lead(waves)
            waves = np.concatenate([waves, avg_waves], axis=0)
            images = waves  # placeholder to keep structure
        else:
            spec_path = (
                "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/%s.parquet"
                % (dp["spectrogram_id"])
            )
            spec = pd.read_parquet(spec_path)

            spec = spec.values[:, 1:]
            spec = np.asarray(spec)

            images = []
            for region in range(4):
                r = region * 300
                c0 = region * 100
                c1 = (region + 1) * 100

                img = spec[r : r + 300, c0:c1].T  # expected (100,300)
                img = self._pad_or_crop_2d(img, 100, 300, pad_value=0.0)

                img = np.clip(img, np.exp(-4), np.exp(8))
                img = np.log(img)
                img = np.nan_to_num(img, nan=0.0)
                images.append(img.astype(np.float32))

            images = np.stack(images, -1)  # (100,300,4)

        images = np.transpose(images, [2, 0, 1]).astype(np.float32)  # (4,100,300)
        return images




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

    def forward(self, x):
        bs = x.size(0)
        x1 = [x[:, i : i + 1, :, :] for i in range(4)]
        x1 = torch.cat(x1, dim=2)  # stack along height
        x = torch.cat([x1, x1, x1], dim=1)  # make 3-channel
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




## === cell 5
test_df = pd.read_csv(CFG["data"])
test_df.head(5)



## === cell 6
available_weights = [w for w in CFG["weights"] if os.path.exists(w)]

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
            state_dict = torch.load(model_weight, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)

            prediction_dict = inference_function(test_loader, model, device)
            predictions_list.append(prediction_dict["predictions"])

            del model, state_dict
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
