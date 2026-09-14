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

0.4422042416692959

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the missing weight-path bug by gracefully falling back to a simple, valid baseline prediction if the expected `/kaggle/input/hms-baseline|hms-eeg|hms-mix` folders aren’t present (so the notebook always produces a submission). I also remove the hard CUDA dependency in the EEG-to-spectrogram precomputation (it currently crashes on environments without an NVIDIA driver) and make mix-mode automatically compute/load the needed `eeg_specs_dict.pkl` on CPU. Finally, I harden inference/submission creation to ensure prediction shape is `(n_test, 6)`, probabilities are normalized to sum to 1, and the submission CSV matches the required columns.'
- What this solution (achieved 1.40995) has done: 'Your current 1.40995 score is far from the 0.4422 target (lower is better), and the main reason is that you’re usually falling back to uniform probabilities because no weights are found. The smallest legitimate improvement is therefore to (1) actually use the provided architecture to infer from whatever weights exist (mix/eeg/spec) instead of defaulting to uniform when mix weights are missing, and (2) average available model families (mix/eeg/spec) to improve calibration without changing any core model logic. I also fix a small but important bug in `brain_lead()` where the last group mistakenly uses `RP` twice instead of `RR`, which should materially reduce error when EEG/mix inference is used. Finally, I keep the strict probability normalization to guarantee valid submissions under the KL metric.'
- What this solution (achieved 1.40995) has done: 'Your current 1.40995 (lower is better) is far from the 0.4422 target, so the smallest meaningful improvement is to ensure the intended EEG lead construction is correct and that inference uses stable, correctly-loaded weights when present. I fix the bug in `brain_lead()` where the final group is mistakenly `RP` twice instead of `RR`, which directly affects EEG and mix inputs and should reduce KL. I also make weight loading robust to common checkpoint formats (raw `state_dict`, nested `state_dict`, `model`, `model_state_dict`) so you don’t silently run with partially-missing weights. Finally, I keep your existing ensembling/normalization semantics unchanged to preserve valid submissions and avoid metric mismatches.'
- What this solution (achieved 1.40995) has done: 'You’re far above the target KL (1.40995 vs 0.4422; lower is better), so we should improve accuracy without changing the overall modeling approach. The biggest minimal win here is to fix a bug in `get_spec()` where `r` is never advanced, so all 4 regions currently reuse the same slice; this materially degrades spec/mix inputs and thus predictions. I also make `DataLoader` use persistent workers to reduce overhead without changing inference results, and I keep your existing ensembling and strict probability normalization (critical for KL + submission validity). No architecture, loss, or training logic is changed—only a clear feature extraction bug fix and safe inference plumbing.'
- What this solution (achieved 1.40995) has done: 'To move your KL score down toward the 0.4422 target (lower is better) without changing the core models, I fix the biggest remaining correctness issue: `NetMix` currently never applies its `Transform()` preprocess, so the EEG half of the mix input is effectively ignored (this can heavily hurt accuracy). I also align the `Transform.forward()` reshape with the actual number of EEG channels produced by `brain_lead()` (16), so the EEG model/mix preprocessing can behave as intended instead of relying on implicit/invalid reshapes. Finally, I keep your existing ensembling and probability normalization, and only add a small safety clamp for the mix tensor placement to avoid silent broadcasting/shape edge cases.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995; lower is better) is still far from the 0.4422 target, so the smallest changes that should materially reduce KL are to (1) fix a critical bug in `Transform.forward()` where the spectrogram is never resized despite `self.resizer` being defined (this creates a size mismatch vs what EfficientNet/HRNet weights expect and can silently degrade outputs), and (2) make `Transform.forward()` robust to the actual input shape for the mix EEG-half (which is currently fed as `(bs,4,128,256)` “image-like”, not raw 1D waveforms). These changes preserve your models, ensembling, and inference flow, but restore the intended preprocessing so loaded weights can behave closer to how they were trained. I also keep strict probability normalization/clipping for submission validity under KL.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995; lower is better) is far above the 0.4422 target, so the most likely reason is that the loaded checkpoints are not being applied correctly (silent partial load due to key-prefix mismatch), leading to near-random/untrained outputs. I minimally harden checkpoint loading by stripping common prefixes (e.g., `module.`) and enforcing a “must-load-most-weights” rule; if a checkpoint fails this, it be skipped instead of degrading the ensemble. I also avoid recomputing the heavy `eeg_specs_dict.pkl` once per mix-weight by instantiating the dataset once per mode (mix/eeg/spec) and reusing the same DataLoader across folds, which preserves model logic but reduces runtime overhead and instability. Finally, I keep your existing probability normalization (critical for KL + submission validity) and submission schema unchanged.'
- What this solution (achieved 1.40995) has done: 'The current score (1.40995 KL; lower is better) is far above the target (0.4422), so we should make a small, correctness-focused change that is likely to materially improve predictions without altering the model architectures or training/inference approach. The biggest remaining feature bug is in `get_spec()`: it currently slices the spectrogram using `region * 100` columns, but the HMS spectrogram parquet stores regions in contiguous 100-column blocks (LL, RL, LP, RP) and the correct column mapping is not `region*100` when using the 4 stacked 300-row bands; this misalignment can heavily degrade spec/mix inputs. I fix `get_spec()` to slice the 4 regions consistently as 4 consecutive 100-column blocks for each 300-row band (keeping the same log/clip processing), which should reduce KL while preserving your exact downstream model logic. I also keep the existing strict probability normalization and submission writing unchanged.'
- What this solution (achieved 1.40995) has done: 'Your KL is far above the target (1.40995 vs 0.4422; lower is better), so we should make the smallest correctness fixes that materially improve model inputs without changing any model architectures or inference semantics. The biggest likely issue here is a feature extraction mismatch in `get_spec()`: the HMS spectrogram parquet columns are grouped by region (LL/RL/LP/RP) across all 1200 rows, not by row-block, so slicing with `c0 = region*100` while also slicing `r0 = region*300` misaligns regions and degrades spec/mix predictions. I fix `get_spec()` to slice the 4 region column-blocks consistently for each of the 4 stacked 300-row bands (so each channel corresponds to the intended region), while keeping the exact same clip/log/NaN handling and output shape. Everything else (models, flips, checkpoint loading, ensembling, normalization, submission writing) stays unchanged.'

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

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)



## === cell 1
CFG = {
    "batch_size": 32,
    "num_worker": 4,
    "data": "/kaggle/input/hms-harmful-brain-activity-classification/test.csv",
    "weights_spec": "/kaggle/input/hms-baseline",
    "weights_eeg": "/kaggle/input/hms-eeg",
    "weights_mix": "/kaggle/input/hms-mix",
    "flip": True,
}




## === cell 2
def _safe_list_weights(folder):
    if isinstance(folder, (list, tuple)):
        return list(folder)
    if folder is None:
        return []
    if not os.path.isdir(folder):
        return []
    files = sorted(os.listdir(folder))
    paths = [os.path.join(folder, x) for x in files]
    paths = [
        p
        for p in paths
        if os.path.isfile(p)
        and (p.endswith(".pt") or p.endswith(".pth") or p.endswith(".bin"))
    ]
    return paths


CFG["weights_spec"] = _safe_list_weights(CFG["weights_spec"])
CFG["weights_eeg"] = _safe_list_weights(CFG["weights_eeg"])
CFG["weights_mix"] = _safe_list_weights(CFG["weights_mix"])

CFG



## === cell 3
import os
import pickle

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch import nn
from tqdm import tqdm

data_dir = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs"
NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]

_SPEC_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EEGToMelTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.wave_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=200,
            hop_length=10000 // 256,
            n_fft=1024,
            n_mels=128,
            f_min=0,
            f_max=20,
            win_length=128,
        )

    def forward(self, x):
        return self.wave_transform(x)


transform_func = EEGToMelTransform().to(_SPEC_DEVICE)


def spectrogram_from_eeg(parquet_path, display=False):
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle : middle + 10_000]

    img = np.zeros((128, 256, 4), dtype="float32")

    for k in range(4):
        COLS = FEATS[k]
        for kk in range(4):
            x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values
            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

            x_tensor = torch.from_numpy(x).to(_SPEC_DEVICE)
            mel_spec = transform_func(x_tensor)
            mel_spec = mel_spec.detach().to("cpu").numpy()

            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[
                :, :width
            ]
            img[:, :, k] += mel_spec_db

        img[:, :, k] /= 4.0

    return img


def build_or_load_test_eeg_specs_dict(pkl_path="eeg_specs_dict.pkl", data_dir=data_dir):
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    all_fs = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
    all_specs = {}
    for item in tqdm(all_fs, desc="Building eeg_specs_dict.pkl"):
        eeg_id = item.rsplit(".", 1)[0]
        eeg_path = os.path.join(data_dir, f"{eeg_id}.parquet")
        all_specs[eeg_id] = spectrogram_from_eeg(eeg_path)

    with open(pkl_path, "wb") as file:
        pickle.dump(all_specs, file)

    return all_specs




## === cell 4
class AlaskaDataIter:
    def __init__(
        self,
        df,
        training_flag=False,
        shuffle=False,
        use_spec=False,
        use_eeg=False,
        use_mix=False,
        ll=0,
        rr=20,
    ):

        self.ll = 0
        self.rr = rr
        self.training_flag = training_flag
        self.shuffle = shuffle

        self.raw_data_set_size = None  ## decided by self.parse_file
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
        self.use_spec = use_spec
        self.use_mix = use_mix

        if self.use_mix:
            self.eeg_specs = build_or_load_test_eeg_specs_dict(
                "eeg_specs_dict.pkl", data_dir=data_dir
            )

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

    def get_eeg(self, dp, is_training):
        eeg_path = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/%s.parquet"
            % (dp["eeg_id"])
        )
        eeg = pd.read_parquet(eeg_path)

        offset = 0
        eeg = eeg.iloc[int(offset * 200) : int(offset * 200) + 10000]

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

        return waves

    def get_spec(self, dp, is_training):
        spec_path = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/%s.parquet"
            % (dp["spectrogram_id"])
        )
        spec = pd.read_parquet(spec_path)

        spec = spec.values[:, 1:]  # drop time column -> shape (1200, 400)

        images = []
        for region in range(4):
            c0 = region * 100
            c1 = c0 + 100

            band_imgs = []
            for band in range(4):
                r0 = band * 300
                r1 = r0 + 300
                band_img = spec[r0:r1, c0:c1].T  # (100, 300)
                band_imgs.append(band_img)

            img = np.mean(np.stack(band_imgs, axis=0), axis=0)  # (100, 300)

            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)
            img = np.nan_to_num(img, nan=0.0)
            images.append(img)

        images = np.stack(images, -1)  # (100,300,4)
        data = np.transpose(images, [2, 0, 1])  # (4,100,300)
        return data

    def get_mix(self, dp, is_training):
        X = np.zeros((128, 256, 8), dtype="float32")
        kg_spec = self.get_spec(dp, is_training=False)  # (4, 100, 300)

        eeg_spec = self.eeg_specs[str(dp["eeg_id"])]  # (128, 256, 4)

        kg_spec = np.transpose(kg_spec, axes=[1, 2, 0])  # (100, 300, 4)
        kg_spec = kg_spec[:, 22:-22, :]  # (100, 256, 4)

        X[14:-14, :, :4] = kg_spec  # (100, 256, 4) into (128, 256, 4)
        X[:, :, 4:] = eeg_spec

        X = np.transpose(X, [2, 0, 1])
        return X

    def single_map_func(self, dp, is_training):
        if self.use_eeg:
            data = self.get_eeg(dp, is_training)
        elif self.use_spec:
            data = self.get_spec(dp, is_training)
        elif self.use_mix:
            data = self.get_mix(dp, is_training)

        return data.astype(np.float32)




## === cell 5
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

        if CFG["flip"]:
            x_flip = torch.flip(x, [3])
            x = torch.cat([x, x_flip], dim=0)

        x = self.model.forward_features(x)
        x = self.avg(x)

        x = x.view((2 * bs if CFG["flip"] else bs), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.softmax(x, -1)

        if CFG["flip"]:
            ans = (x[:bs, ...] + x[bs:, ...]) / 2.0
        else:
            ans = x
        return ans




## === cell 6
class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.wave_transform = torchaudio.transforms.Spectrogram(
            n_fft=512, hop_length=25, power=1
        )
        self.am2db = torchaudio.transforms.AmplitudeToDB(stype="magnitude", top_db=80)
        self.resizer = nn.UpsamplingBilinear2d(size=[160, 320])

    def forward(self, x):
        if x.ndim == 4:
            n, c, h, w = x.shape
            x = x.reshape(n, c, h * w)

        image = self.wave_transform(x)  # (N, C, F, TT)
        image = self.am2db(image)

        n, c, h, w = image.size()
        image = image[:, :, : int(20 / 100 * h + 30), :]
        image = self.resizer(image)

        image = torch.reshape(image, shape=[n, 16, -1, image.size(-1)])
        return image


class NetEeg(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.preprocess = Transform()
        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=16)
        self.fc = nn.Linear(2048, 6, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bs = x.size(0)

        x = self.preprocess(x)

        if CFG["flip"]:
            x_flip = torch.flip(x, [3])
            x = torch.cat([x, x_flip], dim=0)

        x = self.model.forward_features(x)
        x = self.avg(x)
        x = x.view((2 * bs if CFG["flip"] else bs), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.softmax(x, -1)

        if CFG["flip"]:
            ans = (x[:bs, ...] + x[bs:, ...]) / 2.0
        else:
            ans = x

        return ans




## === cell 7
class NetMix(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.preprocess = Transform()
        self.model = timm.create_model("hrnet_w18", pretrained=False, in_chans=3)
        self.fc = nn.Linear(2048, 6, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bs = x.size(0)

        x_kg = x[:, :4, :, :]  # (bs, 4, 128, 256)
        x_eeg = x[:, 4:, :, :]  # (bs, 4, 128, 256)
        x_eeg = self.preprocess(x_eeg)

        n, c16, h, w = x_eeg.shape
        x_eeg4 = x_eeg.view(n, 4, 4, h, w).mean(dim=2)  # (bs, 4, h, w)

        x = torch.cat([x_kg, x_eeg4], dim=1)  # (bs, 8, ..., ...)

        if CFG["flip"]:
            x_flip = torch.flip(x, [3])
            x = torch.cat([x, x_flip], dim=0)

        x1 = [x[:, i : i + 1, :, :] for i in range(4)]
        x1 = torch.cat(x1, dim=2)

        x2 = [x[:, i + 4 : i + 5, :, :] for i in range(4)]
        x2 = torch.cat(x2, dim=2)

        x = torch.cat([x1, x2], dim=3)
        x = torch.cat([x, x, x], dim=1)

        x = self.model.forward_features(x)
        x = self.avg(x)

        x = x.view((2 * bs if CFG["flip"] else bs), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.softmax(x, -1)

        if CFG["flip"]:
            ans = (x[:bs, ...] + x[bs:, ...]) / 2.0
        else:
            ans = x

        return ans




## === cell 8
def inference_function(test_loader, model, device):
    model.eval()
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for step, X in enumerate(tqdm_test_loader):
            X = X.to(device)
            with torch.no_grad():
                y_preds = model(X)
            preds.append(y_preds.to("cpu").numpy())

    prediction_dict = {}
    prediction_dict["predictions"] = np.concatenate(preds, axis=0)
    return prediction_dict




## === cell 9
test_df = pd.read_csv(CFG["data"])
test_df.head(5)



## === cell 10
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _clean_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        for pref in ("module.", "model.", "net."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        cleaned[nk] = v
    return cleaned


def _load_weights_with_sanity(model, model_weight, device, min_load_ratio=0.85):
    ckpt = torch.load(model_weight, map_location="cpu")
    state_dict = _clean_state_dict_keys(_extract_state_dict(ckpt))

    model_state = model.state_dict()
    matched = 0
    total = len(model_state)

    loadable = {}
    for k, v in state_dict.items():
        if k in model_state and hasattr(v, "shape") and v.shape == model_state[k].shape:
            loadable[k] = v
            matched += 1

    load_ratio = matched / max(total, 1)
    if load_ratio < min_load_ratio:
        print(
            f"SKIP weight {os.path.basename(model_weight)}: low load ratio {load_ratio:.2%} ({matched}/{total})"
        )
        return False

    missing, unexpected = model.load_state_dict(loadable, strict=False)
    if len(loadable) == 0:
        print(f"SKIP weight {os.path.basename(model_weight)}: nothing loadable")
        return False

    model.to(device)
    return True


def _build_loader_for_mode(mode_name, test_df):
    if mode_name == "mix":
        use_mix, use_eeg, use_spec = True, False, False
    elif mode_name == "eeg":
        use_mix, use_eeg, use_spec = False, True, False
    elif mode_name == "spec":
        use_mix, use_eeg, use_spec = False, False, True
    else:
        raise ValueError(mode_name)

    test_dataset = AlaskaDataIter(
        test_df,
        training_flag=False,
        shuffle=False,
        use_mix=use_mix,
        use_eeg=use_eeg,
        use_spec=use_spec,
    )

    _nw = int(CFG["num_worker"])
    test_loader = DataLoader(
        test_dataset,
        CFG["batch_size"],
        num_workers=_nw,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(_nw > 0),
    )
    return test_dataset, test_loader


def _predict_with_weights(mode_name, weights, test_df, device):
    if len(weights) == 0:
        return None

    if mode_name == "mix":
        model_ctor = NetMix
    elif mode_name == "eeg":
        model_ctor = NetEeg
    elif mode_name == "spec":
        model_ctor = NetSpec
    else:
        raise ValueError(mode_name)

    test_dataset, test_loader = _build_loader_for_mode(mode_name, test_df)

    preds_all = []
    for model_weight in weights:
        print(f"infer {mode_name} with weight: {os.path.basename(model_weight)}")
        model = model_ctor()

        ok = _load_weights_with_sanity(model, model_weight, device, min_load_ratio=0.85)
        if not ok:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

        prediction_dict = inference_function(test_loader, model, device)
        preds_all.append(prediction_dict["predictions"])

        del model, prediction_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    del test_loader, test_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if len(preds_all) == 0:
        return None

    preds_all = np.asarray(preds_all, dtype=np.float32)
    return preds_all.mean(axis=0)


pred_mix = _predict_with_weights("mix", CFG["weights_mix"], test_df, device)
pred_eeg = _predict_with_weights("eeg", CFG["weights_eeg"], test_df, device)
pred_spec = _predict_with_weights("spec", CFG["weights_spec"], test_df, device)

available = [p for p in [pred_mix, pred_eeg, pred_spec] if p is not None]

if len(available) == 0:
    predictions = np.full((len(test_df), 6), 1.0 / 6.0, dtype=np.float32)
else:
    predictions = np.mean(np.stack(available, axis=0), axis=0)

predictions = np.asarray(predictions, dtype=np.float32)
if (
    predictions.ndim != 2
    or predictions.shape[1] != 6
    or predictions.shape[0] != len(test_df)
):
    predictions = np.full((len(test_df), 6), 1.0 / 6.0, dtype=np.float32)

predictions = np.clip(predictions, 1e-8, 1.0).astype(np.float32)
row_sums = predictions.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
predictions = predictions / row_sums

predictions.shape



## === cell 11
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions.astype(np.float32)
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
sub.head()
