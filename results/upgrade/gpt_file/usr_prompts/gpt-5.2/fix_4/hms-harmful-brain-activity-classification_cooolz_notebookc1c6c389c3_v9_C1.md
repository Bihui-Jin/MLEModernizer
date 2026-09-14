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

0.4611364180524799

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I first fix the runtime blocker by making the weight loading robust: if the expected Kaggle input folders/files don’t exist, the code fall back to a valid uniform-probability submission rather than crashing. I also fix a logic bug in `brain_lead` (RP was duplicated instead of using RR) and remove references to undefined variables during inference-time EEG slicing. Finally, I guarantee the submission has the correct 6 target columns, correct row count/order, and probabilities that sum to 1.0 per row, so a valid `submission.csv` is always produced end-to-end.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower is better) is far from the target (0.4611), and the most likely reason is that you are effectively submitting near-uniform probabilities because the external weight files in `/kaggle/input/hms-baseline` and `/kaggle/input/hms-eeg` are not available in your environment. The minimal, score-relevant fix is to load and ensemble the provided competition baseline weights *only if they exist*, and otherwise keep your safe uniform fallback. I also ensure inference uses `autocast` on CUDA to fit memory (without changing model logic) and add a tiny numerical stabilization by normalizing after averaging folds (already present) and guaranteeing float64 normalization for KL safety. These changes preserve your architecture/inference semantics while making it actually use learned weights when present, which should move KL much closer to the target band.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower is better) is far above the target (0.4611), and the most likely cause is that you’re averaging probabilities from mismatched/unavailable checkpoints: the code tries to load weights that either don’t exist or don’t match the defined model heads, so it effectively falls back to weak/uniform-like predictions. I make weight loading robust to common Kaggle checkpoint formats (`state_dict`, `model`, `net`) and ensure we only ensemble checkpoints that actually load cleanly into the model, otherwise skip them. I also keep inference semantics identical but add a numerically-stable “logit-ensemble” (average logits then softmax) which is typically better for KL than averaging already-softmaxed probabilities, without changing architecture or training. Finally, I keep the same submission schema/paths and still guarantee a valid CSV even if no checkpoints are usable.'

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
    "data": "/kaggle/input/hms-harmful-brain-activity-classification/test.csv",
    "weights_spec": [
        "/kaggle/input/hms-baseline/fold0_epoch_4_val_loss_0.549755.pth",
        "/kaggle/input/hms-baseline/fold1_epoch_4_val_loss_0.524753.pth",
        "/kaggle/input/hms-baseline/fold2_epoch_4_val_loss_0.511551.pth",
        "/kaggle/input/hms-baseline/fold3_epoch_4_val_loss_0.548698.pth",
        "/kaggle/input/hms-baseline/fold4_epoch_3_val_loss_0.666013.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/fold0_epoch_4_val_loss_0.549755.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/fold1_epoch_4_val_loss_0.524753.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/fold2_epoch_4_val_loss_0.511551.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/fold3_epoch_4_val_loss_0.548698.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/fold4_epoch_3_val_loss_0.666013.pth",
    ],
    "weights_eeg": [
        "/kaggle/input/hms-eeg/fold0_epoch_4_val_loss_0.584720.pth",
        "/kaggle/input/hms-eeg/fold1_epoch_4_val_loss_0.590993.pth",
        "/kaggle/input/hms-eeg/fold2_epoch_4_val_loss_0.603758.pth",
        "/kaggle/input/hms-eeg/fold3_epoch_4_val_loss_0.580059.pth",
        "/kaggle/input/hms-eeg/fold4_epoch_4_val_loss_0.545819.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/fold0_epoch_4_val_loss_0.584720.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/fold1_epoch_4_val_loss_0.590993.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/fold2_epoch_4_val_loss_0.603758.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/fold3_epoch_4_val_loss_0.580059.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/fold4_epoch_4_val_loss_0.545819.pth",
    ],
    "flip": True,
}

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


def _existing_weight_paths(paths):
    return [p for p in paths if isinstance(p, str) and os.path.exists(p)]


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ("state_dict", "model", "net", "model_state_dict", "weights"):
            v = ckpt_obj.get(k, None)
            if isinstance(v, dict):
                ckpt_obj = v
                break
    if not isinstance(ckpt_obj, dict):
        raise TypeError("Checkpoint is not a state_dict/dict-like object.")
    cleaned = {}
    for k, v in ckpt_obj.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v
    return cleaned


def _safe_load_weights(model, weight_path, device):
    try:
        ckpt = torch.load(weight_path, map_location=device)
        sd = _extract_state_dict(ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        total_params = len(list(model.state_dict().keys()))
        if len(missing) > 0.9 * total_params:
            return False, f"Too many missing keys ({len(missing)}/{total_params})"
        return True, ""
    except Exception as e:
        return False, str(e)




## === cell 2
class AlaskaDataIter:
    def __init__(self, df, training_flag=False, shuffle=False, use_eeg=False):
        self.training_flag = training_flag
        self.shuffle = shuffle
        self.raw_data_set_size = None
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

    def single_map_func(self, dp, is_training):
        if self.use_eeg:
            eeg_path = f"/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/{dp['eeg_id']}.parquet"
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
            waves = mne.filter.filter_data(waves, 200, 0, 20, verbose=False)
            waves = self.brain_lead(waves)
            data = waves
        else:
            spec_path = f"/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/{dp['spectrogram_id']}.parquet"
            spec = pd.read_parquet(spec_path)
            spec = spec.values[:, 1:]

            images = []
            r = 0
            for region in range(4):
                img = spec[r : r + 300, region * 100 : (region + 1) * 100].T
                img = np.clip(img, np.exp(-4), np.exp(8))
                img = np.log(img)
                img = np.nan_to_num(img, nan=0.0)
                images.append(img)

            images = np.stack(images, -1)
            data = np.transpose(images, [2, 0, 1])

        return data.astype(np.float32)




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
    def __init__(self):
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
def inference_function_logits(test_loader, model, device):
    model.eval()
    logits = []
    use_amp = device.type == "cuda"
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for X in tqdm_test_loader:
            X = X.to(device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_amp):
                    y = model(X)
            logits.append(y.to("cpu").float().numpy())
    return {"logits": np.concatenate(logits, axis=0)}


def _row_normalize(p, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, None)
    s = p.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    p = p / s
    return p.astype(np.float32)


def _softmax_np(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return _row_normalize(ex)




## === cell 6
test_df = pd.read_csv(CFG["data"])
test_df.head(5)



## === cell 7
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

eeg_weight_paths = _existing_weight_paths(CFG["weights_eeg"])
spec_weight_paths = _existing_weight_paths(CFG["weights_spec"])

predictions = None
all_source_logits = []

loaded_eeg = 0
loaded_spec = 0

if len(eeg_weight_paths) > 0:
    fold_logits = []
    test_dataset = AlaskaDataIter(
        test_df, training_flag=False, shuffle=False, use_eeg=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_worker"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    for model_weight in eeg_weight_paths:
        model = NetEeg()
        ok, msg = _safe_load_weights(model, model_weight, device)
        if not ok:
            del model
            gc.collect()
            continue

        model.to(device)
        prediction_dict = inference_function_logits(test_loader, model, device)
        fold_logits.append(prediction_dict["logits"])
        loaded_eeg += 1

        del model, prediction_dict
        torch.cuda.empty_cache()
        gc.collect()

    if len(fold_logits) > 0:
        eeg_logits = np.mean(np.stack(fold_logits, axis=0), axis=0)
        all_source_logits.append(eeg_logits)

if len(spec_weight_paths) > 0:
    fold_logits = []
    test_dataset = AlaskaDataIter(
        test_df, training_flag=False, shuffle=False, use_eeg=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_worker"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    for model_weight in spec_weight_paths:
        model = NetSpec()
        ok, msg = _safe_load_weights(model, model_weight, device)
        if not ok:
            del model
            gc.collect()
            continue

        model.to(device)
        prediction_dict = inference_function_logits(test_loader, model, device)
        fold_logits.append(prediction_dict["logits"])
        loaded_spec += 1

        del model, prediction_dict
        torch.cuda.empty_cache()
        gc.collect()

    if len(fold_logits) > 0:
        spec_logits = np.mean(np.stack(fold_logits, axis=0), axis=0)
        all_source_logits.append(spec_logits)

if len(all_source_logits) > 0:
    logits = np.mean(np.stack(all_source_logits, axis=0), axis=0)
    predictions = _softmax_np(logits).astype(np.float32)
else:
    n = len(test_df)
    predictions = np.full((n, len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32)

if (
    predictions.ndim != 2
    or predictions.shape[0] != len(test_df)
    or predictions.shape[1] != len(TARGETS)
):
    n = len(test_df)
    predictions = np.full((n, len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32)

predictions = _row_normalize(predictions)



## === cell 8
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
print(sub.head())
print(
    f"Row sums (min/mean/max): {sub[TARGETS].sum(1).min():.6f} / {sub[TARGETS].sum(1).mean():.6f} / {sub[TARGETS].sum(1).max():.6f}"
)
print(
    f"Found EEG weights: {len(eeg_weight_paths)} | Found SPEC weights: {len(spec_weight_paths)} | Loaded EEG: {loaded_eeg} | Loaded SPEC: {loaded_spec}"
)
