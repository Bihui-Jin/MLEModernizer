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

0.4108572550134333

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the missing weight paths by automatically discovering valid `.pth/.pt/.bin` files under `/kaggle/input` (and fall back to a safe uniform-probability submission if none are found), which resolves the `FileNotFoundError` and `IsADirectoryError`. I also ensure the inferred prediction array has shape `(num_test, 6)` and is properly normalized to sum to 1 per row, preventing the “Columns must be same length as key” submission error. Finally, I keep the model definitions and inference logic intact, only adding robust path handling, filtering out directories from weight lists, and a deterministic fallback so the notebook always produces `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.41086), so we should make small, legitimate inference-only fixes that usually improve KL without changing the model architecture or training. The biggest likely issue is that the saved checkpoints aren’t being loaded correctly: many Kaggle weights are stored under keys like `state_dict`/`model` and/or have `module.` prefixes, and your current `torch.load(...); load_state_dict(..., strict=False)` can silently load almost nothing, yielding near-random predictions. I add a robust checkpoint-to-state_dict extraction + key-fixing routine and verify we actually load a meaningful fraction of parameters (otherwise skip that weight), then keep the same ensemble/softmax/normalization and submission format. This should move the score substantially toward the target while preserving the core logic and staying within Kaggle constraints.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.41086), so the most likely “minimal but impactful” fix is ensuring checkpoints are actually loaded and used correctly at inference (otherwise you effectively submit near-uniform/random predictions). I keep your exact model definitions and inference loop, but make weight loading more compatible with common Kaggle checkpoint formats (including `fc.*` vs `model.classifier.*` naming and `module.` prefixes), and I remove the overly-strict “skip if <50% keys match” gate that can discard valid weights for these models. I also add a safe logit-ensemble path (average logits, then softmax once) which typically improves KL vs averaging already-softmaxed probabilities, without changing the architecture or training. Finally, I keep the same normalization/clipping and submission schema so the output remains valid.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far worse than the target (0.41086), so the smallest likely “real fix” is to ensure inference matches how these EfficientNet-B5 checkpoints were trained: correct input normalization and correct weight loading. I keep your model classes and inference loop intact, but (1) add ImageNet mean/std normalization for spectrogram inputs (and a safe standardization for EEG-derived spectrogram tensors) before feeding EfficientNet, and (2) make weight loading more compatible by handling common key prefixes without over-stripping (your current `module.`/`model.` stripping can accidentally break many timm checkpoints). Finally, I keep your logits-averaging ensemble and probability normalization/clipping so the submission remains valid and stable.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.41086), so the most likely minimal improvement is fixing an inference-time bug in the spectrogram construction that makes all 4 region “images” identical (because the row index `r` never advances). I change only that indexing so each of the 4 spectrogram regions uses its own 300-row block, preserving the same preprocessing (clip/log), model architectures, and inference/ensembling. This should make inputs match the intended competition baseline behavior and materially improve KL without altering training logic (there is no training here). I keep the same robust weight discovery/loading, logits averaging, probability normalization, and submission writing.'

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
    "weights_spec": "/kaggle/input/hms-baseline",
    "weights_eeg": "/kaggle/input/hms-eeg",
    "flip": True,
}




## === cell 2
def _gather_weight_files(root: str, must_contain: str | None = None):
    exts = (".pth", ".pt", ".bin", ".ckpt")
    out = []
    if not os.path.exists(root):
        return out
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(exts):
                continue
            full = os.path.join(dirpath, fn)
            if must_contain is not None and must_contain.lower() not in full.lower():
                continue
            out.append(full)
    return sorted(out)


spec_weights = _gather_weight_files(CFG["weights_spec"])
eeg_weights = _gather_weight_files(CFG["weights_eeg"])

if len(spec_weights) == 0 and len(eeg_weights) == 0:
    base_root = "/kaggle/input"
    all_weights = _gather_weight_files(base_root)
    spec_weights = [
        w
        for w in all_weights
        if "spec" in w.lower() or "spect" in w.lower() or "baseline" in w.lower()
    ]
    eeg_weights = [w for w in all_weights if "eeg" in w.lower()]
    if len(spec_weights) == 0 and len(all_weights) > 0:
        spec_weights = all_weights

CFG["weights_spec"] = spec_weights
CFG["weights_eeg"] = eeg_weights

print(f"Found spec weights: {len(CFG['weights_spec'])}")
print(f"Found eeg  weights: {len(CFG['weights_eeg'])}")
if len(CFG["weights_spec"]) > 0:
    print("Example spec weight:", CFG["weights_spec"][0])
if len(CFG["weights_eeg"]) > 0:
    print("Example eeg weight :", CFG["weights_eeg"][0])




## === cell 3
class AlaskaDataIter:
    def __init__(
        self, df, training_flag=False, shuffle=False, use_eeg=False, ll=0, rr=20
    ):

        self.ll = 0
        self.rr = rr
        self.training_flag = training_flag
        self.shuffle = shuffle

        self.raw_data_set_size = None  ##decided by self.parse_file

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

        brain_leads = [self.LL, self.LP, self.RP, self.RP]

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
        """Data augmentation function."""

        if self.use_eeg:
            eeg_path = (
                "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/%s.parquet"
                % (dp["eeg_id"])
            )
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
            waves = mne.filter.filter_data(waves, 200, self.ll, self.rr, verbose=False)

            waves = self.brain_lead(waves)

            data = waves
        else:

            spec_path = (
                "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/%s.parquet"
                % (dp["spectrogram_id"])
            )
            spec = pd.read_parquet(spec_path)

            spec = spec.values[:, 1:]

            images = []

            for region in range(4):
                r = region * 300
                img = spec[r : r + 300, region * 100 : (region + 1) * 100].T
                img = np.clip(img, np.exp(-4), np.exp(8))
                img = np.log(img)

                img = np.nan_to_num(img, nan=0.0)

                images.append(img)

            images = np.stack(images, -1)

            data = np.transpose(images, [2, 0, 1])

        return data.astype(np.float32)




## === cell 4
class NetSpec(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=3)

        self.fc = nn.Linear(2048, 6, bias=True)

        self.dropout = nn.Dropout(0.5)

        self.avg = nn.AdaptiveAvgPool2d(1)

        self.register_buffer(
            "_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x):
        bs = x.size(0)

        x1 = [x[:, i : i + 1, :, :] for i in range(4)]
        x1 = torch.cat(x1, dim=2)
        x = torch.cat([x1, x1, x1], dim=1)

        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        x = (x - x_min) / (x_max - x_min + 1e-6)
        x = (x - self._mean) / self._std

        x = self.model.forward_features(x)
        x = self.avg(x)

        x = x.view(bs, -1)
        x = self.dropout(x)

        x = self.fc(x)
        return x




## === cell 5
class Transform(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.wave_transform = torchaudio.transforms.Spectrogram(
            n_fft=512, hop_length=25, power=1
        )
        self.am2db = torchaudio.transforms.AmplitudeToDB(stype="magnitude", top_db=80)
        self.resizer = nn.UpsamplingBilinear2d(size=[160, 320])

    def forward(self, x):
        bs = x.size(0)

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

        self.register_buffer(
            "_eps", torch.tensor(1e-6, dtype=torch.float32), persistent=False
        )

    def forward(self, x):
        bs = x.size(0)

        x = self.preprocess(x)

        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True)
        x = (x - mean) / (std + self._eps)

        x = self.model.forward_features(x)
        x = self.avg(x)

        x = x.view(bs, -1)
        x = self.dropout(x)

        x = self.fc(x)
        return x




## === cell 6
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for step, X in enumerate(tqdm_test_loader):
            X = X.to(device)
            with torch.no_grad():
                y_preds = model(X)
            y_preds = softmax(y_preds)
            preds.append(y_preds.to("cpu").numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 7
def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "ema_state_dict",
        ]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        if any(torch.is_tensor(v) for v in ckpt_obj.values()):
            return ckpt_obj
    return None


def _strip_prefix_if_present(state_dict, prefix: str):
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    if not all(isinstance(k, str) for k in state_dict.keys()):
        return state_dict
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def _remap_common_head_keys(sd: dict):
    """
    Minimal compatibility remaps for efficientnet checkpoints that store head under different names.
    This preserves core model logic while making weight loading succeed more often, improving KL.
    """
    if not isinstance(sd, dict) or len(sd) == 0:
        return sd

    out = dict(sd)

    has_fc_w = "fc.weight" in out
    has_fc_b = "fc.bias" in out

    cand_w = None
    cand_b = None
    for wkey, bkey in [
        ("model.classifier.weight", "model.classifier.bias"),
        ("classifier.weight", "classifier.bias"),
        ("head.fc.weight", "head.fc.bias"),
        ("model.head.fc.weight", "model.head.fc.bias"),
    ]:
        if wkey in out and bkey in out:
            cand_w, cand_b = wkey, bkey
            break

    if cand_w is not None and (not has_fc_w and not has_fc_b):
        out["fc.weight"] = out[cand_w]
        out["fc.bias"] = out[cand_b]

    return out


def _best_effort_align_state_dict_keys(model: nn.Module, sd: dict):
    model_keys = set(model.state_dict().keys())
    candidates = []
    for cand in [
        sd,
        _strip_prefix_if_present(sd, "module."),
        _strip_prefix_if_present(sd, "model."),
        _strip_prefix_if_present(_strip_prefix_if_present(sd, "module."), "model."),
    ]:
        if not isinstance(cand, dict) or len(cand) == 0:
            continue
        overlap = len(model_keys.intersection(cand.keys()))
        candidates.append((overlap, cand))
    if len(candidates) == 0:
        return sd
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def load_weights_robust(model: nn.Module, weight_path: str, device):
    ckpt = torch.load(weight_path, map_location=device)
    sd = _extract_state_dict(ckpt)
    if sd is None:
        return False, "unrecognized checkpoint format"

    sd = _best_effort_align_state_dict_keys(model, sd)
    sd = _remap_common_head_keys(sd)

    missing, unexpected = model.load_state_dict(sd, strict=False)

    total_keys = len(model.state_dict())
    matched = total_keys - len(missing)
    matched_frac = matched / max(1, total_keys)

    if matched_frac < 0.05:
        return False, f"too few keys matched ({matched}/{total_keys})"

    return (
        True,
        f"loaded (matched {matched}/{total_keys}, unexpected {len(unexpected)}, missing {len(missing)})",
    )




## === cell 8
test_df = pd.read_csv(CFG["data"])
test_df.head(5)



## === cell 9
predictions_list = []
logits_list = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def inference_logits(test_loader, model, device):
    model.eval()
    outs = []
    with tqdm(test_loader, unit="test_batch", desc="Inference(logits)") as tdl:
        for X in tdl:
            X = X.to(device)
            with torch.no_grad():
                y = model(X)
            outs.append(y.to("cpu").numpy())
    return np.concatenate(outs, axis=0)


if len(CFG["weights_spec"]) > 0:
    print("infer with weights_spec")
    for model_weight in CFG["weights_spec"]:
        if not os.path.isfile(model_weight):
            continue
        test_dataset = AlaskaDataIter(test_df, training_flag=False, shuffle=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG["batch_size"],
            num_workers=CFG["num_worker"],
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        model = NetSpec()
        ok, msg = load_weights_robust(model, model_weight, device)
        if not ok:
            print(f"SKIP spec weight: {model_weight} ({msg})")
            del model, test_loader, test_dataset
            torch.cuda.empty_cache()
            gc.collect()
            continue
        print(f"Loaded spec weight: {os.path.basename(model_weight)} ({msg})")
        model.to(device)

        lg = inference_logits(test_loader, model, device)
        logits_list.append(lg)

        del model, test_loader, test_dataset
        torch.cuda.empty_cache()
        gc.collect()

if len(CFG["weights_eeg"]) > 0:
    print("infer with weights_eeg")
    for model_weight in CFG["weights_eeg"]:
        if not os.path.isfile(model_weight):
            continue
        test_dataset = AlaskaDataIter(
            test_df, training_flag=False, shuffle=False, use_eeg=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG["batch_size"],
            num_workers=CFG["num_worker"],
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        model = NetEeg()
        ok, msg = load_weights_robust(model, model_weight, device)
        if not ok:
            print(f"SKIP eeg weight: {model_weight} ({msg})")
            del model, test_loader, test_dataset
            torch.cuda.empty_cache()
            gc.collect()
            continue
        print(f"Loaded eeg weight: {os.path.basename(model_weight)} ({msg})")
        model.to(device)

        lg = inference_logits(test_loader, model, device)
        logits_list.append(lg)

        del model, test_loader, test_dataset
        torch.cuda.empty_cache()
        gc.collect()

if len(logits_list) == 0:
    print(
        "WARNING: No usable model weights found. Writing a uniform-probability submission (valid but low score)."
    )
    predictions = np.full((len(test_df), 6), 1.0 / 6.0, dtype=np.float32)
else:
    logits = np.mean(np.stack(logits_list, axis=0), axis=0).astype(np.float32)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    predictions = (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)

predictions = np.asarray(predictions)
if (
    predictions.ndim != 2
    or predictions.shape[0] != len(test_df)
    or predictions.shape[1] != 6
):
    raise RuntimeError(
        f"Bad predictions shape: got {predictions.shape}, expected ({len(test_df)}, 6)"
    )

predictions = np.clip(predictions, 1e-8, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)



## === cell 10
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
print(
    "Row-sum min/max:", sub[TARGETS].sum(axis=1).min(), sub[TARGETS].sum(axis=1).max()
)
sub.head()
