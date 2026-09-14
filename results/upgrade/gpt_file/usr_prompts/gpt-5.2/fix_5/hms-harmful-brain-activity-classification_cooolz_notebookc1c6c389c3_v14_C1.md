# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.4279889051380929

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the missing weight-path issue by automatically falling back to the competition’s `sample_submission.csv` directory (or, if no external weights are present, using a safe uniform-probability fallback so a valid CSV is still produced). I also filter weight file lists to include only real model files (skip directories like `/`), which resolves the `IsADirectoryError` during `torch.load`. Finally, I guarantee the submission has exactly 6 probability columns that sum to 1 by applying a final normalization step and ensuring the prediction array shape matches `(len(test_df), 6)`.'
- What this solution (achieved 1.40995) has done: 'Your score (1.40995, lower-is-better) is far worse than the target (0.42799), so we should safely improve predictive quality without changing the model architecture or inference semantics. The biggest likely issue is that the code often isn’t actually using the intended weight files because `CFG["weights_spec"]`/`CFG["weights_eeg"]` point to non-existent directories; this can silently fall back to uniform predictions, which scores very poorly on KL. I make the weights discovery robust by searching common Kaggle input locations (including the competition dataset folder) and by correctly loading checkpoints whether they are raw `state_dict` or wrapped dicts (`{"state_dict":...}` / `{"model":...}`), which is a common cause of “loads but wrong”. Finally, I keep the exact same inference and normalization, but enable a small, metric-safe improvement: logit-averaging across models (average logits, then softmax) instead of probability-averaging, which typically reduces KL while preserving the ensemble core logic.'
- What this solution (achieved 1.40995) has done: 'I make the weight discovery/load stricter so you only ensemble checkpoints that actually match the intended model (spec vs eeg) and that load cleanly, because mixing incompatible checkpoints (or partially loading with `strict=False`) typically yields near-random outputs and a very bad KL score. I also change the ensemble step to average logits in a numerically stable way (log-sum-exp in probability space) while preserving the same “ensemble then softmax” semantics, which tends to reduce KL without changing the model architecture or inference pipeline. Finally, I ensure deterministic inference behavior and keep the submission normalization/clipping so every row sums to 1 and the CSV is always valid.'

# 9. Code solution

## === cell 0
import os
import gc
import copy
import json
import pickle
import random

import cv2
import numpy as np
import pandas as pd

import albumentations as A
import librosa
import mne
import timm
from tqdm import tqdm

import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader



## === cell 1
CFG = {
    "batch_size": 32,
    "num_worker": 4,
    "data": "/kaggle/input/hms-harmful-brain-activity-classification/test.csv",
    "sample_sub": "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv",
    "weights_spec": "/kaggle/input/hms-baseline",
    "weights_eeg": "/kaggle/input/hms-eeg",
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

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




## === cell 2
class AlaskaDataIter:
    def __init__(
        self, df, training_flag=False, shuffle=False, use_eeg=False, ll=0, rr=20
    ):
        self.ll = 0
        self.rr = rr
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
            n_fft=512, hop_length=50, power=1
        )
        self.am2db = torchaudio.transforms.AmplitudeToDB(stype="magnitude", top_db=80)
        self.resizer = nn.UpsamplingBilinear2d(size=[160, 320])

    def forward(self, x):
        image = self.wave_transform(x)
        image = self.am2db(image)
        n, c, h, w = image.size()
        image = image[:, :, : int(20 / 100 * h + 10), :]
        image = torch.reshape(image, shape=[n, 2, -1, w])
        x = torch.cat([image[:, 0:1, ...], image[:, 1:2:, ...]], dim=-1)
        return x


class NetEeg(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.preprocess = Transform()
        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=1)
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
def _list_weight_files(path: str):
    if not isinstance(path, str) or not path:
        return []
    if not os.path.exists(path):
        return []
    if os.path.isfile(path):
        return [path] if path.lower().endswith((".pt", ".pth", ".bin")) else []
    files = []
    for fn in sorted(os.listdir(path)):
        fp = os.path.join(path, fn)
        if os.path.isfile(fp) and fn.lower().endswith((".pt", ".pth", ".bin")):
            files.append(fp)
    return files


def _discover_weight_files(hints):
    if isinstance(hints, (list, tuple)):
        hint_paths = list(hints)
    else:
        hint_paths = [hints]

    candidate_roots = [
        "/kaggle/input",
        "/kaggle/input/hms-harmful-brain-activity-classification",
        "/kaggle/input/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification",
        "/kaggle/data",
    ]

    out = []
    seen = set()

    for p in hint_paths:
        for fp in _list_weight_files(p):
            if fp not in seen:
                seen.add(fp)
                out.append(fp)

    if len(out) == 0:
        for root in candidate_roots:
            if not os.path.isdir(root):
                continue
            try:
                lvl1 = [os.path.join(root, d) for d in os.listdir(root)]
            except Exception:
                continue
            for p1 in lvl1:
                if os.path.isfile(p1):
                    if p1.lower().endswith((".pt", ".pth", ".bin")) and p1 not in seen:
                        seen.add(p1)
                        out.append(p1)
                    continue
                if not os.path.isdir(p1):
                    continue
                try:
                    lvl2 = [os.path.join(p1, d) for d in os.listdir(p1)]
                except Exception:
                    continue
                for p2 in lvl2:
                    if os.path.isfile(p2) and p2.lower().endswith(
                        (".pt", ".pth", ".bin")
                    ):
                        if p2 not in seen:
                            seen.add(p2)
                            out.append(p2)

    return sorted(out)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "model_state_dict", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]
                break
        else:
            sd = ckpt
    else:
        sd = ckpt

    if isinstance(sd, dict):
        new_sd = {}
        for key, val in sd.items():
            nk = key
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            new_sd[nk] = val
        sd = new_sd
    return sd


def _filter_weights_by_hint(files, include_any=()):
    if not files:
        return []
    inc = tuple(s.lower() for s in include_any if isinstance(s, str) and s)
    if not inc:
        return files
    kept = []
    for fp in files:
        name = os.path.basename(fp).lower()
        if any(h in name for h in inc):
            kept.append(fp)
    return kept if len(kept) > 0 else files


def _validate_weights_for_model(weight_files, model_ctor):
    ok = []
    for wf in weight_files:
        try:
            m = model_ctor()
            ckpt = torch.load(wf, map_location="cpu")
            sd = _extract_state_dict(ckpt)
            missing, unexpected = m.load_state_dict(sd, strict=False)
            del m, ckpt, sd
            gc.collect()
            if len(missing) == 0:
                ok.append(wf)
        except Exception:
            continue
    return ok


CFG["weights_spec"] = _discover_weight_files(CFG["weights_spec"])
CFG["weights_eeg"] = _discover_weight_files(CFG["weights_eeg"])

CFG["weights_spec"] = _filter_weights_by_hint(
    CFG["weights_spec"], include_any=("spec", "spect", "baseline", "spc")
)
CFG["weights_eeg"] = _filter_weights_by_hint(
    CFG["weights_eeg"], include_any=("eeg", "wave")
)

CFG["weights_spec"] = _validate_weights_for_model(
    CFG["weights_spec"], lambda: NetSpec()
)
CFG["weights_eeg"] = _validate_weights_for_model(CFG["weights_eeg"], lambda: NetEeg())

print("Found spec weights (validated):", len(CFG["weights_spec"]))
print("Found eeg weights (validated):", len(CFG["weights_eeg"]))




## === cell 6
def inference_function(test_loader, model, device):
    model.eval()
    logits = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for step, X in enumerate(tqdm_test_loader):
            X = X.to(device)
            with torch.no_grad():
                y_logits = model(X)
            logits.append(y_logits.detach().to("cpu").numpy())
    return {"logits": np.concatenate(logits, axis=0)}




## === cell 7
test_df = pd.read_csv(CFG["data"])
sample_sub = pd.read_csv(CFG["sample_sub"])

test_df = test_df.merge(sample_sub[["eeg_id"]], on="eeg_id", how="inner")
test_df = test_df.set_index("eeg_id").loc[sample_sub["eeg_id"].values].reset_index()

test_df.head(5)



## === cell 8
predictions_list = []

if len(CFG["weights_spec"]) == 0 and len(CFG["weights_eeg"]) == 0:
    print(
        "No model weights found; creating uniform-probability predictions for a valid submission."
    )
    predictions = np.full((len(test_df), 6), 1.0 / 6.0, dtype=np.float32)
else:
    probs_list = []

    def _log_avg_probs(prob_list, eps=1e-8):
        P = np.stack(prob_list, axis=0).astype(np.float64)  # (M,N,6)
        P = np.clip(P, eps, 1.0)
        logP = np.log(P)
        logP_mean = logP.mean(axis=0)
        out = np.exp(logP_mean)
        out = out / out.sum(axis=1, keepdims=True)
        return out.astype(np.float32)

    if len(CFG["weights_spec"]) > 0:
        print("infer with weights_spec")
        test_dataset = AlaskaDataIter(test_df, training_flag=False, shuffle=False)
        test_loader = DataLoader(
            test_dataset,
            CFG["batch_size"],
            num_workers=CFG["num_worker"],
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        for model_weight in CFG["weights_spec"]:
            model = NetSpec()
            ckpt = torch.load(model_weight, map_location="cpu")
            state_dict = _extract_state_dict(ckpt)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if len(missing) != 0:
                del model, ckpt, state_dict
                gc.collect()
                continue
            model.to(device)

            pred_dict = inference_function(test_loader, model, device)
            logits = pred_dict["logits"].astype(np.float32)
            prob = (
                torch.softmax(torch.from_numpy(logits), dim=1)
                .numpy()
                .astype(np.float32)
            )
            probs_list.append(prob)

            del model, ckpt, state_dict, pred_dict, logits, prob
            torch.cuda.empty_cache()
            gc.collect()

        del test_loader, test_dataset
        torch.cuda.empty_cache()
        gc.collect()

    if len(CFG["weights_eeg"]) > 0:
        print("infer with weights_eeg")
        test_dataset = AlaskaDataIter(
            test_df, training_flag=False, shuffle=False, use_eeg=True
        )
        test_loader = DataLoader(
            test_dataset,
            CFG["batch_size"],
            num_workers=CFG["num_worker"],
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        for model_weight in CFG["weights_eeg"]:
            model = NetEeg()
            ckpt = torch.load(model_weight, map_location="cpu")
            state_dict = _extract_state_dict(ckpt)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if len(missing) != 0:
                del model, ckpt, state_dict
                gc.collect()
                continue
            model.to(device)

            pred_dict = inference_function(test_loader, model, device)
            logits = pred_dict["logits"].astype(np.float32)
            prob = (
                torch.softmax(torch.from_numpy(logits), dim=1)
                .numpy()
                .astype(np.float32)
            )
            probs_list.append(prob)

            del model, ckpt, state_dict, pred_dict, logits, prob
            torch.cuda.empty_cache()
            gc.collect()

        del test_loader, test_dataset
        torch.cuda.empty_cache()
        gc.collect()

    if len(probs_list) == 0:
        print(
            "All discovered weights failed validation at runtime; using uniform fallback."
        )
        predictions = np.full((len(test_df), 6), 1.0 / 6.0, dtype=np.float32)
    else:
        predictions = _log_avg_probs(probs_list, eps=1e-8)

predictions = np.asarray(predictions, dtype=np.float32)
if (
    predictions.ndim != 2
    or predictions.shape[0] != len(test_df)
    or predictions.shape[1] != 6
):
    raise ValueError(
        f"Bad predictions shape {predictions.shape}, expected ({len(test_df)}, 6)"
    )

predictions = np.clip(predictions, 1e-8, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)



## === cell 9
sub = sample_sub[["eeg_id"]].copy()
sub = sub.merge(
    pd.DataFrame({"eeg_id": test_df["eeg_id"].values}), on="eeg_id", how="left"
)

pred_df = pd.DataFrame(predictions, columns=TARGETS)
pred_df.insert(0, "eeg_id", test_df["eeg_id"].values)

sub = sample_sub[["eeg_id"]].merge(pred_df, on="eeg_id", how="left")

miss = sub[TARGETS].isna().any(axis=1)
if miss.any():
    sub.loc[miss, TARGETS] = 1.0 / 6.0

sub[TARGETS] = np.clip(sub[TARGETS].values.astype(np.float32), 1e-8, 1.0)
sub[TARGETS] = sub[TARGETS].div(sub[TARGETS].sum(axis=1), axis=0)

sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
print("Row-sum check:", float(np.max(np.abs(sub[TARGETS].sum(axis=1).values - 1.0))))
sub.head()

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
MemoryError                               Traceback (most recent call last)
/tmp/ipykernel_55/735131470.py in <cell line: 0>()
      1 # Change: build submission strictly from sample_submission.csv order.
      2 sub = sample_sub[["eeg_id"]].copy()
----> 3 sub = sub.merge(
      4     pd.DataFrame({"eeg_id": test_df["eeg_id"].values}), on="eeg_id", how="left"
      5 )

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in merge(self, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
  10830         from pandas.core.reshape.merge import merge
  10831 
> 10832         return merge(
  10833             self,
  10834             right,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    182             validate=validate,
    183         )
--> 184         return op.get_result(copy=copy)
    185 
    186 

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in get_result(self, copy)
    884             self.left, self.right = self._indicator_pre_merge(self.left, self.right)
    885 
--> 886         join_index, left_indexer, right_indexer = self._get_join_info()
    887 
    888         result = self._reindex_and_concat(

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _get_join_info(self)
   1149             )
   1150         else:
-> 1151             (left_indexer, right_indexer) = self._get_join_indexers()
   1152 
   1153             if self.right_index:

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _get_join_indexers(self)
   1123         # make mypy happy
   1124         assert self.how != "asof"
-> 1125         return get_join_indexers(
   1126             self.left_join_keys, self.right_join_keys, sort=self.sort, how=self.how
   1127         )

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in get_join_indexers(left_keys, right_keys, sort, how)
   1757         _, lidx, ridx = left.join(right, how=how, return_indexers=True, sort=sort)
   1758     else:
-> 1759         lidx, ridx = get_join_indexers_non_unique(
   1760             left._values, right._values, sort, how
   1761         )

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in get_join_indexers_non_unique(left, right, sort, how)
   1793     lkey, rkey, count = _factorize_keys(left, right, sort=sort)
   1794     if how == "left":
-> 1795         lidx, ridx = libjoin.left_outer_join(lkey, rkey, count, sort=sort)
   1796     elif how == "right":
   1797         ridx, lidx = libjoin.left_outer_join(rkey, lkey, count, sort=sort)

join.pyx in pandas._libs.join.left_outer_join()

MemoryError: Unable to allocate 1.01 TiB for an array with shape (138147787534,) and data type int64
