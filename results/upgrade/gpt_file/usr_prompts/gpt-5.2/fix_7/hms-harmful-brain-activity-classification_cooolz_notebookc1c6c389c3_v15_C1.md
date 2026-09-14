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

0.4887223828828267

# 6. Current score

1.40813

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the missing/incorrect model-weight paths by auto-discovering valid `.pt/.pth/.bin` files under `/kaggle/input` (or falling back to a uniform-probability submission if no weights exist), which resolves the `FileNotFoundError` and the later `IsADirectoryError`. I also make the weight listing robust (ignore directories) and ensure the prediction array is always shaped `(len(test), 6)` so `sub[TARGETS] = predictions` cannot fail. Finally, I enforce probability normalization (row-sum to 1 with clipping) to guarantee the submission passes Kaggle’s format checks and matches the KL-divergence metric expectations. Core model definitions and inference logic are preserved; changes are limited to plumbing, path resolution, and safe post-processing.'
- What this solution (achieved 1.40995) has done: 'Your score is far worse than the target (lower is better), and the most likely cause is that the current inference is averaging in many unrelated `.pt/.pth/.bin` files discovered under `/kaggle/input`, which effectively corrupts predictions. I make weight discovery strict and deterministic: only use files inside the intended weight directories if they exist, and otherwise fall back to a safe uniform submission (rather than mixing arbitrary weights). I also fix a small but impactful bug in `brain_lead()` where `RR` was accidentally replaced by `RP`, and I ensure test-time augmentation is not silently active (your `flip` flag wasn’t actually used). These are minimal changes that preserve the model and inference core logic, but should substantially reduce KL divergence toward your target.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is much worse than the target (0.4887), and the most plausible cause is that you are loading unintended/incorrect weight files (or loading them with `strict=False` so partially-matching weights silently corrupt the model). I make weight discovery deterministic and strict: only use weight files that (a) are real files, (b) match the expected key shapes for the specific model (NetSpec vs NetEeg), and (c) can be loaded with `strict=True` after removing an optional `module.` prefix. If no compatible weights are found, we still fall back to a valid uniform submission exactly as before. This keeps the model architecture/inference unchanged, but prevents averaging in bad checkpoints and should move KL substantially down toward your target.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995; lower is better) is far worse than the target, and the smallest likely-correct fix is to stop averaging together spectrogram-model predictions (NetSpec) and EEG-model predictions (NetEeg) in a single unweighted mean—this can badly miscalibrate probabilities and inflate KL. I keep your exact models and inference loop, but instead (a) average within each modality separately and (b) combine the two modality means with a simple fixed convex weight (no extra training) that typically improves calibration. I also fix a small but meaningful bug in `brain_lead()` where the right-right chain was accidentally built using `RP` twice (it should use `RR`), which can degrade NetEeg features. Finally, I keep your strict weight compatibility filtering and probability normalization so the submission remains valid.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far worse than the target (0.4887), so we should improve (decrease) it with the smallest low-risk changes that don’t alter your model/inference core. The biggest likely issue is a train/test preprocessing mismatch: in `NetSpec` you concatenate 4 regions then slice only `x[:,0:4,...]`, which drops the 4th region entirely; we fix this by explicitly using only the first 3 regions (consistent with the 3-channel EfficientNet input) in the same way training typically does, which should substantially improve calibration and KL. We also disable the unused/accidentally-defined augmentation path for inference by removing the always-on `HorizontalFlip` transform object (it isn’t applied, but this avoids future accidental activation), and keep the rest identical (weights, strict loading, softmax, normalization, submission format). These are minimal, execution-safe changes aimed at moving KL down toward your target without changing architecture or training loops.'
- What this solution (achieved 1.40813) has done: 'Your KL is far above the target (lower is better), so the smallest likely-correct improvement is to fix a train/test preprocessing mismatch in the spectrogram path: your dataset builds 4 region channels, but `NetSpec` only consumes the first 3 channels; if training used a specific 3-region selection, the current “first 3” choice can be suboptimal and inflate KL. I keep the same model/inference core but change the spectrogram dataset to construct exactly 3 channels by combining the two right-hemisphere regions into one channel (a common baseline trick for this competition), so all 3 EfficientNet input channels carry signal instead of dropping an entire region. I also apply a tiny “prior-mix” smoothing toward the global train label distribution (no retraining, just post-processing) to improve calibration for KL divergence with minimal risk; probabilities remain normalized and the submission format is unchanged. Everything else (weight filtering, strict loading, softmax, ensembling, CSV writing) is preserved.'

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
    "flip": False,
}




## === cell 2
def _discover_weight_files(root_dir: str):
    exts = (".pt", ".pth", ".bin")
    out = []
    if root_dir and os.path.exists(root_dir) and os.path.isdir(root_dir):
        for fn in sorted(os.listdir(root_dir)):
            fp = os.path.join(root_dir, fn)
            if os.path.isfile(fp) and fp.lower().endswith(exts):
                out.append(fp)
    return out


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
        elif "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
        else:
            sd = obj
    else:
        sd = obj
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def _filter_compatible_weights(weight_paths, model_ctor, device):
    """Return only weights that can be loaded strictly (key presence + tensor shapes match)."""
    good = []
    ref_model = model_ctor().to(device)
    ref_sd = ref_model.state_dict()
    del ref_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    for wp in weight_paths:
        try:
            obj = torch.load(wp, map_location=device)
            sd = _extract_state_dict(obj)
            if not isinstance(sd, dict):
                continue

            ok = True
            for k, v in ref_sd.items():
                if k not in sd:
                    ok = False
                    break
                tv = sd[k]
                if not hasattr(tv, "shape") or tuple(tv.shape) != tuple(v.shape):
                    ok = False
                    break

            if ok:
                tmp_model = model_ctor().to(device)
                tmp_model.load_state_dict(sd, strict=True)
                del tmp_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                good.append(wp)
        except Exception:
            continue
    return good


spec_weights = _discover_weight_files(CFG["weights_spec"])
eeg_weights = _discover_weight_files(CFG["weights_eeg"])

CFG["weights_spec"] = spec_weights
CFG["weights_eeg"] = eeg_weights

print("Discovered spec weights:", len(CFG["weights_spec"]))
print("Discovered eeg  weights:", len(CFG["weights_eeg"]))
if len(CFG["weights_spec"]) > 0:
    print("First spec weight:", CFG["weights_spec"][0])
if len(CFG["weights_eeg"]) > 0:
    print("First eeg weight:", CFG["weights_eeg"][0])




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

        self.train_trans = None

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
        """Data augmentation function."""

        if self.use_eeg:
            eeg_path = (
                "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/%s.parquet"
                % (dp["eeg_id"])
            )
            eeg = pd.read_parquet(eeg_path)

            offset = 0  # (eeg_offside_min + eeg_offside_max) // 2

            eeg = eeg.iloc[int(offset * 200) : int((offset + 50) * 200)]

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
            region_imgs = []
            for region in range(4):
                img = spec[r : r + 300, region * 100 : (region + 1) * 100].T
                img = np.clip(img, np.exp(-4), np.exp(8))
                img = np.log(img)
                img = np.nan_to_num(img, nan=0.0)
                region_imgs.append(img)

            img0 = region_imgs[0]
            img1 = region_imgs[1]
            img2 = 0.5 * (region_imgs[2] + region_imgs[3])

            images = np.stack([img0, img1, img2], -1)  # (H, W, 3)
            data = np.transpose(images, [2, 0, 1])  # (3, H, W)

        return data.astype(np.float32)




## === cell 4
class NetSpec(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=3)

        self.fc = nn.Linear(2048, 6, bias=True)

        self.dropout = nn.Dropout(0.5)

        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bs = x.size(0)

        x3 = x[:, :3, :, :]  # [bs, 3, H, W]
        x = self.model.forward_features(x3)
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
def _load_state_dict_safely(model, weight_path, device):
    obj = torch.load(weight_path, map_location=device)
    sd = _extract_state_dict(obj)
    if not isinstance(sd, dict):
        raise ValueError(
            f"Weight file did not contain a valid state_dict: {weight_path}"
        )
    model.load_state_dict(sd, strict=True)
    return model




## === cell 7
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
    prediction_dict = {}
    prediction_dict["predictions"] = np.concatenate(preds)
    return prediction_dict




## === cell 8
test_df = pd.read_csv(CFG["data"])
test_df.head(5)



## === cell 9
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
n_test = len(test_df)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CFG["weights_spec"] = _filter_compatible_weights(CFG["weights_spec"], NetSpec, device)[
    :10
]
CFG["weights_eeg"] = _filter_compatible_weights(CFG["weights_eeg"], NetEeg, device)[:10]

print("Compatible spec weights:", len(CFG["weights_spec"]))
print("Compatible eeg  weights:", len(CFG["weights_eeg"]))
if len(CFG["weights_spec"]) > 0:
    print("First compatible spec weight:", CFG["weights_spec"][0])
if len(CFG["weights_eeg"]) > 0:
    print("First compatible eeg weight:", CFG["weights_eeg"][0])

spec_model_preds = []
eeg_model_preds = []

if len(CFG["weights_spec"]) > 0:
    print("infer with weights_spec")
    test_dataset = AlaskaDataIter(
        test_df, training_flag=False, shuffle=False, use_eeg=False
    )
    test_loader = DataLoader(
        test_dataset,
        CFG["batch_size"],
        num_workers=CFG["num_worker"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    for model_weight in CFG["weights_spec"]:
        model = NetSpec()
        model = _load_state_dict_safely(model, model_weight, device)
        model.to(device)
        prediction_dict = inference_function(test_loader, model, device)
        spec_model_preds.append(prediction_dict["predictions"])
        del model
        if torch.cuda.is_available():
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
        model = _load_state_dict_safely(model, model_weight, device)
        model.to(device)
        prediction_dict = inference_function(test_loader, model, device)
        eeg_model_preds.append(prediction_dict["predictions"])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if (len(spec_model_preds) == 0) and (len(eeg_model_preds) == 0):
    print(
        "WARNING: No compatible model weights found in the configured directories. Falling back to uniform predictions."
    )
    predictions = np.full((n_test, 6), 1.0 / 6.0, dtype=np.float32)
else:
    pred_list = []
    if len(spec_model_preds) > 0:
        pred_spec = np.mean(np.stack(spec_model_preds, axis=0), axis=0).astype(
            np.float32
        )
        pred_list.append(("spec", pred_spec))
    if len(eeg_model_preds) > 0:
        pred_eeg = np.mean(np.stack(eeg_model_preds, axis=0), axis=0).astype(np.float32)
        pred_list.append(("eeg", pred_eeg))

    if len(pred_list) == 1:
        predictions = pred_list[0][1]
    else:
        w_spec = 0.65
        w_eeg = 0.35
        predictions = (w_spec * pred_spec + w_eeg * pred_eeg).astype(np.float32)

predictions = np.asarray(predictions)
if predictions.ndim != 2 or predictions.shape[0] != n_test or predictions.shape[1] != 6:
    raise RuntimeError(
        f"Bad predictions shape: {predictions.shape}, expected ({n_test}, 6)"
    )

train_path = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
try:
    train_df = pd.read_csv(train_path, usecols=TARGETS)
    prior = train_df[TARGETS].sum(axis=0).values.astype(np.float64)
    prior = prior / prior.sum()
except Exception:
    prior = np.full(6, 1.0 / 6.0, dtype=np.float64)

alpha = (
    0.06  # small smoothing toward prior; intended to reduce KL, not to chase best score
)
predictions = (1.0 - alpha) * predictions + alpha * prior[None, :]

eps = 1e-8
predictions = np.clip(predictions, eps, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
print("Row-sum stats:", sub[TARGETS].sum(axis=1).min(), sub[TARGETS].sum(axis=1).max())
sub.head()
