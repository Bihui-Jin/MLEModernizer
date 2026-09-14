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

0.4224930516963866

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the broken weight paths by discovering available `.pth/.pt/.bin` files under `/kaggle/input` (or falling back to uniform probabilities if none are present), and I ensure we only load actual files (not directories), which resolves the `IsADirectoryError`. I also correct the dataset logic that was producing 4 spectrogram “regions” but a 3‑channel model, by selecting 3 channels consistently so the model input matches `in_chans=3` without changing the model architecture. Finally, I make submission creation robust by enforcing the correct `(n_samples, 6)` prediction shape, renormalizing probabilities to sum to 1, and writing `submission.csv` with the required columns.'
- What this solution (achieved 1.40995) has done: 'Your current score is far worse than the target (lower is better), so the smallest safe improvement is to make the inference match the training-time semantics of your weights and reduce systematic distribution shift. I (1) fix a bug in spectrogram construction where the same first 300 time rows are reused for all 4 regions (missing `r += 300`), and (2) make test-time flipping deterministic and consistent by only applying it when `CFG["flip"]` is enabled and averaging original+flipped predictions (TTA) instead of relying on a train-only random flip pipeline. These changes keep the same models, loss, and general pipeline, but usually reduce KL significantly because the input features become correct and predictions become better calibrated/less noisy. I also keep the same submission formatting and probability normalization so the CSV remains valid.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.4225), which strongly suggests a systematic inference mismatch rather than a small tuning issue. The smallest high-impact fix that preserves your core models and loss is to align the test-time EEG preprocessing with how the EEG model expects inputs: `NetEeg` reshapes into 4 channels, but your dataset currently produces 16 differential leads, which is a major shape/semantic mismatch and can silently destroy performance. I change only the EEG branch to output exactly 4 “region” waveforms (LL/LP/RP/RR) using the same differential construction concept, leaving the spectrogram path and both model architectures untouched. I also make the EEG and spectrogram region lists correct (you currently duplicate RP and omit RR), which is a clear bug and should reduce KL toward your target.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.4225), which strongly suggests an inference-time preprocessing mismatch rather than a small calibration issue. The smallest high-impact fix that preserves your model architectures and inference loop is to (1) feed the spectrogram model the 3 channels it expects without duplicating/tiling the input into a very different geometry, and (2) apply the same per-sample normalization that most HMS baselines rely on (standardize each channel) to reduce distribution shift versus what the weights likely saw during training. I also fix flip-TTA so it only runs for spectrograms (where width-flip is meaningful) and not for EEG waveforms (where flipping the time axis is usually not a valid augmentation), which should reduce systematic errors. These changes keep the same weights, loss semantics (softmax probabilities), and output formatting, while aiming to move KL substantially downward toward your target.'
- What this solution (achieved 1.40995) has done: 'Your current KL is much worse than the target (lower is better), which most often comes from inference-time mismatches and/or numerically overconfident probabilities. I make two minimal, score-relevant changes: (1) ensure spectrogram flip-TTA is applied along the correct axis (time/width, not frequency/height) given your tensor layout, and (2) add a very small, deterministic probability smoothing (epsilon-mix with uniform) after ensembling to reduce extreme predictions that are heavily penalized by KL. These keep the same models, weights, and overall inference loop, but should move KL substantially downward toward the target without altering core logic. The submission formatting and per-row normalization remain intact and still produce a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995) is far worse than the target (0.4225, lower-is-better), so the most likely issue is an inference-time mismatch (preprocessing vs what the provided weights expect). I keep your models and inference loop intact, but (1) remove the per-sample standardization you added (it often hurts if the weights were trained on raw/log-clipped inputs), and (2) fix the spectrogram region slicing to match the HMS baseline convention (each “region” is a 100-row band, not 300 rows), which otherwise can feed the network the wrong content. I also make the spectrogram flip-TTA explicitly flip along the time axis (width) only, leaving EEG unchanged, and keep your existing probability normalization/smoothing and submission writing.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995; lower is better) is far worse than the target (0.4225), which points to a systematic inference mismatch rather than minor tuning. The most likely issue is the spectrogram construction: you’re slicing a 100x100 diagonal block per “region” (both rows and columns) which is not how HMS spectrograms are structured, so the model likely sees the wrong content. I minimally fix this by keeping the same 4 region row-bands but using the full time axis (all columns) for each band, then resizing to the expected 100-width so the network still receives a 3x100x100 tensor without changing the model. I also make the flip-TTA for spectrograms explicitly flip along the time/width axis (last dim), consistent with the corrected tensor layout, while leaving the EEG branch unchanged.'
- What this solution (achieved 1.40995) has done: 'Your KL is far worse than the target (lower is better), so we should focus on a likely inference mismatch rather than tuning. The highest-impact minimal fix here is to feed the spectrogram model the same kind of 3-channel input it was likely trained on: instead of taking the first 3 of 4 regions (dropping one region entirely), we convert the 4 region-bands into 3 channels by averaging left (LL+LP), right (RP+RR), and global (mean of all 4), which preserves information and keeps `in_chans=3` unchanged. I also remove spectrogram flip-TTA (horizontal flip in time) because it often hurts this task (EEG events are not time-reversal invariant in these representations) and can inflate KL. Everything else (models, weights loading, inference loop, probability normalization/smoothing, submission format) stays the same.'

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
    "prob_smooth_eps": 0.003,
}




## === cell 2
def _find_weight_files(preferred_path: str, search_root: str = "/kaggle/input"):
    exts = (".pth", ".pt", ".bin", ".ckpt")
    weight_files = []

    if isinstance(preferred_path, str) and os.path.isdir(preferred_path):
        for x in sorted(os.listdir(preferred_path)):
            p = os.path.join(preferred_path, x)
            if os.path.isfile(p) and p.lower().endswith(exts):
                weight_files.append(p)

    if len(weight_files) > 0:
        return weight_files

    for root, _, files in os.walk(search_root):
        for f in files:
            if f.lower().endswith(exts):
                weight_files.append(os.path.join(root, f))

    return sorted(weight_files)


CFG["weights_spec"] = _find_weight_files(CFG["weights_spec"])
CFG["weights_eeg"] = _find_weight_files(CFG["weights_eeg"])

print("Found spec weights:", len(CFG["weights_spec"]))
print("Found eeg weights :", len(CFG["weights_eeg"]))
print("Example spec weight:", CFG["weights_spec"][:3])
print("Example eeg weight :", CFG["weights_eeg"][:3])




## === cell 3
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

    def brain_lead_4ch(self, waves):
        waves = copy.deepcopy(waves)
        brain_leads = [self.LL, self.LP, self.RP, self.RR]

        ch = []
        for combine in brain_leads:
            tmp = 0.0
            for i in range(len(combine) - 1):
                tmp = tmp + (
                    waves[self.leads_dict[combine[i]]]
                    - waves[self.leads_dict[combine[i + 1]]]
                )
            tmp = tmp / (len(combine) - 1)
            ch.append(tmp)

        data = np.stack(ch, axis=0)  # 4 x T
        return data

    def single_map_func(self, dp, is_training):
        if self.use_eeg:
            eeg_path = f'/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/{dp["eeg_id"]}.parquet'
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

            data = self.brain_lead_4ch(waves)  # 4 x T

        else:
            spec_path = f'/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/{dp["spectrogram_id"]}.parquet'
            spec = pd.read_parquet(spec_path)
            spec = spec.values[:, 1:]  # drop time column

            images = []
            for region in range(4):
                r0 = region * 100
                r1 = (region + 1) * 100
                band = spec[r0:r1, :]  # (100, T_full)

                band = np.clip(band, np.exp(-4), np.exp(8))
                band = np.log(band)
                band = np.nan_to_num(band, nan=0.0)

                if band.shape[1] != 100:
                    band = cv2.resize(
                        band.astype(np.float32),
                        dsize=(100, 100),  # (width, height)
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    band = band.astype(np.float32)

                images.append(band)

            images = np.stack(images, -1)  # H x W x 4
            data4 = np.transpose(images, [2, 0, 1])  # 4 x H x W

            left = 0.5 * (data4[0] + data4[1])  # LL + LP
            right = 0.5 * (data4[2] + data4[3])  # RP + RR
            global_ = 0.25 * (data4[0] + data4[1] + data4[2] + data4[3])
            data = np.stack([left, right, global_], axis=0)  # 3 x H x W

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
def inference_function(test_loader, model, device, do_flip_tta: bool = False):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for step, X in enumerate(tqdm_test_loader):
            X = X.to(device)
            with torch.no_grad():
                y = model(X)

                if do_flip_tta:
                    Xf = torch.flip(X, dims=[-1])
                    yf = model(Xf)
                    y = 0.5 * (y + yf)

            y = softmax(y)
            preds.append(y.detach().to("cpu").numpy())

    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 7
test_df = pd.read_csv(CFG["data"])
test_df.head(5)



## === cell 8
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
predictions_list = []


def _safe_load_state_dict(model, path, device):
    obj = torch.load(path, map_location=device)
    if (
        isinstance(obj, dict)
        and "state_dict" in obj
        and isinstance(obj["state_dict"], dict)
    ):
        sd = obj["state_dict"]
        sd = {k.replace("model.", "").replace("net.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
    elif isinstance(obj, dict):
        model.load_state_dict(obj, strict=False)
    else:
        raise ValueError(f"Unsupported weight format in {path}")


if len(CFG["weights_spec"]) == 0 and len(CFG["weights_eeg"]) == 0:
    predictions = np.full(
        (len(test_df), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32
    )
else:
    if len(CFG["weights_spec"]) > 0:
        print("infer with weights_spec")
        test_dataset = AlaskaDataIter(
            test_df, training_flag=False, shuffle=False, use_eeg=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG["batch_size"],
            num_workers=CFG["num_worker"],
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        for model_weight in CFG["weights_spec"]:
            if not (isinstance(model_weight, str) and os.path.isfile(model_weight)):
                continue
            model = NetSpec().to(device)
            _safe_load_state_dict(model, model_weight, device)
            prediction_dict = inference_function(
                test_loader, model, device, do_flip_tta=bool(CFG.get("flip", False))
            )
            predictions_list.append(prediction_dict["predictions"])
            del model
            torch.cuda.empty_cache()
            gc.collect()

    if len(CFG["weights_eeg"]) > 0:
        print("infer with weights_eeg")
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
        for model_weight in CFG["weights_eeg"]:
            if not (isinstance(model_weight, str) and os.path.isfile(model_weight)):
                continue
            model = NetEeg().to(device)
            _safe_load_state_dict(model, model_weight, device)

            prediction_dict = inference_function(
                test_loader, model, device, do_flip_tta=False
            )
            predictions_list.append(prediction_dict["predictions"])
            del model
            torch.cuda.empty_cache()
            gc.collect()

    if len(predictions_list) == 0:
        predictions = np.full(
            (len(test_df), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32
        )
    else:
        predictions = np.mean(np.stack(predictions_list, axis=0), axis=0)

predictions = np.asarray(predictions, dtype=np.float64)
if (
    predictions.ndim != 2
    or predictions.shape[0] != len(test_df)
    or predictions.shape[1] != len(TARGETS)
):
    raise ValueError(
        f"Bad predictions shape: {predictions.shape}, expected ({len(test_df)}, {len(TARGETS)})"
    )

predictions = np.clip(predictions, 1e-6, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

eps = float(CFG.get("prob_smooth_eps", 0.0))
if eps > 0:
    uniform = np.full((1, len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float64)
    predictions = (1.0 - eps) * predictions + eps * uniform
    predictions = predictions / predictions.sum(axis=1, keepdims=True)



## === cell 9
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions.astype(np.float32)
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
print(sub.head())
