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

0.4373425895911577

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I make the script robust to missing external weight directories by falling back to a valid, score-safe baseline (uniform probabilities) when no model files are found, so a submission is always produced. I also fix the CUDA hard-failure by selecting CPU when CUDA isn’t available, and avoid the heavy precompute cell that tried to create EEG spectrograms on GPU. Finally, I ensure only actual weight files are loaded (not directories) and guarantee the submission has the correct columns, shape, and row-wise probability sums to 1.0.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.4373), and the main reason is that the notebook is almost certainly falling back to uniform predictions because the referenced weight directories (`/kaggle/input/hms-baseline`, `/kaggle/input/hms-eeg`, `/kaggle/input/hms-mix`) don’t exist in your environment. I make a minimal change to auto-discover any `.pt/.pth/.bin` weights inside `/kaggle/input/**` (common when Kaggle datasets are mounted under different names), while keeping your exact model code and inference flow. I also ensure the most relevant branch runs first (mix -> eeg -> spec) only if corresponding weights are found, and keep the uniform fallback as a safe backstop. These changes should materially improve the score toward the target by actually using available weights, without altering the model architecture, data pipeline, or prediction semantics.'
- What this solution (achieved 1.40995) has done: 'Your current score is far worse than the target (lower-is-better), and the most likely cause is that you’re still not actually loading usable checkpoints (or you’re loading incompatible checkpoint dicts), so the pipeline effectively behaves like a weak baseline. I make minimal, inference-only fixes to (1) load common Kaggle checkpoint formats robustly (`state_dict`, `model`, `module.*`), (2) prefer the most relevant single checkpoint per group to reduce the chance of averaging mismatched folds/architectures, and (3) fix a small but important bug in `brain_lead` (RP duplicated instead of RR), which can materially hurt EEG-based inference while preserving the exact modeling approach. These changes keep the same models/feature extraction/loss semantics and only improve correctness and checkpoint usage to move KL toward the target.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.4373), and the most likely remaining cause is still “not actually using a good checkpoint”: either no weights are being found, or the one picked is incompatible/mismatched, leading to near-uniform/garbage probabilities. I make the smallest inference-only change to (1) expand checkpoint discovery to include common extensions like `.ckpt` and (2) automatically choose the checkpoint that best matches each model’s expected parameter keys/shapes (instead of filename heuristics), while keeping your exact dataset/model/inference logic intact. Additionally, I fix a small bug in `get_spec` where the region slice index `r` never increments (all 4 “regions” are identical), which can materially degrade spec/mix inference without changing the overall approach. These changes should move KL substantially toward the target by ensuring the intended inputs and best-compatible weights are actually used.'

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
    "weights_mix": "/kaggle/input/hms-mix",
    "flip": True,
}




## === cell 2
def _list_weight_files(path: str):
    if not isinstance(path, str) or (not path):
        return []
    if not os.path.exists(path):
        return []
    if os.path.isfile(path):
        return [path]
    files = []
    for x in sorted(os.listdir(path)):
        fp = os.path.join(path, x)
        if os.path.isfile(fp) and (
            x.endswith(".pt")
            or x.endswith(".pth")
            or x.endswith(".bin")
            or x.endswith(".ckpt")
        ):
            files.append(fp)
    return files


def _discover_weight_files_under_kaggle_input(max_files_per_group=50):
    """
    Change rationale (score -> target): broaden discovery to .ckpt and search /kaggle/input/**
    so the same inference code can run with real checkpoints (lower KL).
    """
    root = "/kaggle/input"
    if not os.path.exists(root):
        return {"spec": [], "eeg": [], "mix": []}

    spec, eeg, mix = [], [], []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not (
                fn.endswith(".pt")
                or fn.endswith(".pth")
                or fn.endswith(".bin")
                or fn.endswith(".ckpt")
            ):
                continue
            fp = os.path.join(dirpath, fn)
            lfp = fp.lower()
            if "mix" in lfp:
                mix.append(fp)
            elif "eeg" in lfp:
                eeg.append(fp)
            elif "spec" in lfp or "baseline" in lfp or "spect" in lfp:
                spec.append(fp)
            else:
                spec.append(fp)

    spec = sorted(set(spec))[:max_files_per_group]
    eeg = sorted(set(eeg))[:max_files_per_group]
    mix = sorted(set(mix))[:max_files_per_group]
    return {"spec": spec, "eeg": eeg, "mix": mix}


CFG["weights_spec"] = _list_weight_files(CFG["weights_spec"])
CFG["weights_eeg"] = _list_weight_files(CFG["weights_eeg"])
CFG["weights_mix"] = _list_weight_files(CFG["weights_mix"])

if (len(CFG["weights_spec"]) + len(CFG["weights_eeg"]) + len(CFG["weights_mix"])) == 0:
    discovered = _discover_weight_files_under_kaggle_input(max_files_per_group=200)
    CFG["weights_spec"] = discovered["spec"]
    CFG["weights_eeg"] = discovered["eeg"]
    CFG["weights_mix"] = discovered["mix"]

print(
    "Weights discovered -> spec:",
    len(CFG["weights_spec"]),
    "| eeg:",
    len(CFG["weights_eeg"]),
    "| mix:",
    len(CFG["weights_mix"]),
)
CFG



## === cell 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device



## === cell 4
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


class TransformMel(nn.Module):
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


def spectrogram_from_eeg(parquet_path, transform_func, device_infer):
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

            x_tensor = torch.from_numpy(x.astype(np.float32)).to(device_infer)
            mel_spec = transform_func(x_tensor).detach().cpu().numpy()

            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[
                :, :width
            ]
            img[:, :, k] += mel_spec_db

        img[:, :, k] /= 4.0
    return img


if (len(CFG["weights_mix"]) > 0) and (not os.path.exists("eeg_specs_dict.pkl")):
    transform_func = TransformMel().to(device)
    all_fs = os.listdir(data_dir)
    all_specs = {}
    for item in tqdm(all_fs, desc="Precompute EEG specs (mix)"):
        eeg_id = item.rsplit(".", 1)[0]
        eeg_path = f"/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/{eeg_id}.parquet"
        all_specs[eeg_id] = spectrogram_from_eeg(
            eeg_path, transform_func=transform_func, device_infer=device
        )

    with open("eeg_specs_dict.pkl", "wb") as file:
        pickle.dump(all_specs, file)




## === cell 5
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
        self.use_spec = use_spec
        self.use_mix = use_mix

        if self.use_mix:
            if not os.path.exists("eeg_specs_dict.pkl"):
                raise FileNotFoundError(
                    "eeg_specs_dict.pkl not found but use_mix=True. Run precompute cell or disable mix."
                )
            with open("eeg_specs_dict.pkl", mode="rb") as f:
                self.eeg_specs = pickle.load(f)

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
        return data

    def get_mix(self, dp, is_training):
        X = np.zeros((128, 256, 8), dtype="float32")
        kg_spec = self.get_spec(dp, is_training=False)
        eeg_spec = self.eeg_specs[str(dp["eeg_id"])]

        kg_spec = np.transpose(kg_spec, axes=[1, 2, 0])
        X[14:-14, :, :4] = kg_spec[:, 22:-22]
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
        else:
            raise ValueError("One of use_eeg/use_spec/use_mix must be True.")
        return data.astype(np.float32)




## === cell 6
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

        if CFG["flip"]:
            x = x.view(2 * bs, -1)
        else:
            x = x.view(bs, -1)

        x = self.dropout(x)
        x = self.fc(x)
        x = torch.softmax(x, -1)

        if CFG["flip"]:
            ans = (x[:bs, ...] + x[bs:, ...]) / 2.0
        else:
            ans = x
        return ans




## === cell 7
class Transform(nn.Module):
    def __init__(self):
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

        if CFG["flip"]:
            x_flip = torch.flip(x, [3])
            x = torch.cat([x, x_flip], dim=0)

        x = self.model.forward_features(x)
        x = self.avg(x)

        if CFG["flip"]:
            x = x.view(2 * bs, -1)
        else:
            x = x.view(bs, -1)

        x = self.dropout(x)
        x = self.fc(x)
        x = torch.softmax(x, -1)

        if CFG["flip"]:
            ans = (x[:bs, ...] + x[bs:, ...]) / 2.0
        else:
            ans = x
        return ans




## === cell 8
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

        if CFG["flip"]:
            x = x.view(2 * bs, -1)
        else:
            x = x.view(bs, -1)

        x = self.dropout(x)
        x = self.fc(x)
        x = torch.softmax(x, -1)

        if CFG["flip"]:
            ans = (x[:bs, ...] + x[bs:, ...]) / 2.0
        else:
            ans = x
        return ans




## === cell 9
def inference_function(test_loader, model, device):
    model.eval()
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for X in tqdm_test_loader:
            X = X.to(device)
            with torch.no_grad():
                y_preds = model(X)
            preds.append(y_preds.detach().cpu().numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 10
test_df = pd.read_csv(CFG["data"])
test_df.head(5)



## === cell 11
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    return ckpt


def _strip_prefix_if_present(state_dict, prefix="module."):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def _compat_score_for_model(model, state_dict):
    """
    Change rationale (score -> target): pick the checkpoint that best matches this exact model's
    parameter names/shapes to avoid partially-loaded/mismatched weights that yield high KL.
    """
    if not isinstance(state_dict, dict):
        return -1.0
    msd = model.state_dict()
    matched = 0
    matched_shape = 0
    total = len(msd)
    for k, v in state_dict.items():
        if k in msd:
            matched += 1
            try:
                if tuple(v.shape) == tuple(msd[k].shape):
                    matched_shape += 1
            except Exception:
                pass
    if total == 0:
        return -1.0
    return (matched_shape / total) + 0.1 * (matched / total)


def _select_most_compatible_weights(files, model_ctor, topk=1):
    if not files:
        return []
    scored = []
    for fp in files[:200]:
        try:
            model = model_ctor()
            ckpt = torch.load(fp, map_location="cpu")
            sd = _strip_prefix_if_present(_extract_state_dict(ckpt), prefix="module.")
            score = _compat_score_for_model(model, sd)
            scored.append((score, fp))
        except Exception:
            continue
        finally:
            try:
                del model, ckpt, sd
            except Exception:
                pass
            gc.collect()
    scored = sorted(scored, key=lambda x: x[0], reverse=True)
    picked = [fp for s, fp in scored[:topk] if s > 0]
    return picked


CFG["weights_mix"] = _select_most_compatible_weights(CFG["weights_mix"], NetMix, topk=1)
CFG["weights_eeg"] = _select_most_compatible_weights(CFG["weights_eeg"], NetEeg, topk=1)
CFG["weights_spec"] = _select_most_compatible_weights(
    CFG["weights_spec"], NetSpec, topk=1
)

print(
    "Weights selected -> spec:",
    len(CFG["weights_spec"]),
    "| eeg:",
    len(CFG["weights_eeg"]),
    "| mix:",
    len(CFG["weights_mix"]),
)
print(
    "Selected paths:",
    {"spec": CFG["weights_spec"], "eeg": CFG["weights_eeg"], "mix": CFG["weights_mix"]},
)


def _run_ensemble(weights, dataset_kwargs, model_ctor):
    preds_list = []
    for model_weight in weights:
        test_dataset = AlaskaDataIter(
            test_df, training_flag=False, shuffle=False, **dataset_kwargs
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG["batch_size"],
            num_workers=CFG["num_worker"],
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        model = model_ctor()

        ckpt = torch.load(model_weight, map_location=device)
        state_dict = _extract_state_dict(ckpt)
        state_dict = _strip_prefix_if_present(state_dict, prefix="module.")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if (len(missing) > 200) or (len(unexpected) > 200):
            print(
                f"Skip incompatible checkpoint: {model_weight} | missing={len(missing)} unexpected={len(unexpected)}"
            )
            del model, test_loader, test_dataset, ckpt, state_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

        model.to(device)
        pred_dict = inference_function(test_loader, model, device)
        preds_list.append(pred_dict["predictions"])

        del model, test_loader, test_dataset, ckpt, state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if len(preds_list) == 0:
        return None
    return np.mean(np.stack(preds_list, axis=0), axis=0)


pred = None
if len(CFG["weights_mix"]) > 0:
    print("infer with weights_mix:", len(CFG["weights_mix"]))
    pred = _run_ensemble(
        CFG["weights_mix"], dataset_kwargs={"use_mix": True}, model_ctor=NetMix
    )
elif len(CFG["weights_eeg"]) > 0:
    print("infer with weights_eeg:", len(CFG["weights_eeg"]))
    pred = _run_ensemble(
        CFG["weights_eeg"], dataset_kwargs={"use_eeg": True}, model_ctor=NetEeg
    )
elif len(CFG["weights_spec"]) > 0:
    print("infer with weights_spec:", len(CFG["weights_spec"]))
    pred = _run_ensemble(
        CFG["weights_spec"], dataset_kwargs={"use_spec": True}, model_ctor=NetSpec
    )

if pred is None:
    print("No usable model weights found; writing uniform-probability submission.")
    pred = np.full((len(test_df), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32)

pred = np.asarray(pred, dtype=np.float64)
if pred.shape != (len(test_df), len(TARGETS)):
    raise ValueError(
        f"Predictions shape {pred.shape} does not match expected {(len(test_df), len(TARGETS))}."
    )

pred = np.clip(pred, 1e-9, 1.0)
pred = pred / pred.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = pred
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
sub.head()
