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

0.508869027695893

# 6. Current score

1.39779

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the missing weights-path issue by falling back to a safe baseline prediction (uniform probabilities) when the expected model weight folders are not available, so the notebook always runs end-to-end and produces a valid `submission.csv`. I also fix the hardcoded CUDA usage in the EEG→mel-spectrogram precomputation (it fails on CPU-only Kaggle sessions) by selecting `cuda` only if available, and I avoid running that expensive precompute unless mix-weights are present. Finally, I make the weights discovery robust (only actual files, not directories) and guarantee the submission has the correct columns, row count, and per-row probabilities summing to 1.'
- What this solution (achieved 1.41937) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.50887), and the biggest reason is that you’re effectively submitting uniform probabilities whenever the external weight folders aren’t present, which score very poorly on KL-divergence. The smallest legitimate improvement is to replace the uniform fallback with a label-prior fallback computed from `train.csv` (normalized vote frequencies), which typically moves the public KL score substantially toward the target without changing your model logic. I also make weight discovery handle both “folder of .pth files” and “single .pth file” paths so your ensemble actually loads weights when they exist, and keep all model/feature code untouched. Finally, I keep the existing probability normalization/clipping to ensure the submission is always valid.'
- What this solution (achieved 1.41937) has done: 'Your current score (1.41937, lower-is-better) is far from the target (0.50887), and the most likely cause is that your code is still frequently not loading the intended weights and/or is silently mis-loading them (e.g., checkpoint wrapped under `state_dict`, different key prefixes), so you end up close to a weak fallback distribution. I make weight loading robust to common Kaggle checkpoint formats (plain state_dict vs dict with `state_dict`/`model`, and `module.` prefixes) without changing the model architecture or inference logic. I also ensure the spectrogram directory discovery uses existing competition paths (so weights are found when mounted under `/kaggle/input/...`) and keep the prior-based fallback as-is for safety. These changes should move the score downward (better) toward the target by actually using your trained ensemble when weights exist.'
- What this solution (achieved 1.41937) has done: 'Your current score is much worse than the target (lower-is-better), and the most likely reason is that you’re still not actually loading usable weights (so you’re effectively near-prior), plus there’s a subtle but severe bug in `brain_lead()` where `RR` is never used (RP is duplicated), degrading EEG features when EEG/mix models run. I make two minimal, score-relevant fixes: (1) correct the `brain_lead` channel pairing to `LL, LP, RP, RR` (no architecture/training change; just correct feature extraction), and (2) make checkpoint loading slightly more robust by also stripping the common `model.` prefix in addition to `module.` so more weight files load correctly. These changes should legitimately improve predictions and move KL downward toward your target, while keeping the rest of the pipeline and submission semantics identical.'
- What this solution (achieved 1.41937) has done: 'Your current KL (1.41937, lower-is-better) is far from the target (0.50887), so we should make small changes that plausibly improve the validity/strength of predictions without changing the model architectures or inference semantics. The most impactful minimal fix is to correct a serious bug in `get_spec()` where the spectrogram “region” index never advances (`r` stays 0), so all 4 channels come from the same block; fixing this should materially improve Spec and Mix model inputs and reduce KL. I also ensure the spectrogram slicing is consistent with the intended 4-region layout and keep all existing normalization/clipping so the submission remains valid. Everything else (models, softmax, TTA flip, ensemble averaging, fallback prior) is preserved.'
- What this solution (achieved 1.39779) has done: 'Your KL (lower is better) is far worse than the target, so we should make the smallest changes that plausibly improve prediction quality without changing the model architectures or inference semantics. The biggest score risk left is that inference can still silently run with effectively untrained/random weights (because `strict=False` allows large mismatches), which produces near-prior-like outputs and high KL. I (1) enforce a “safe load” policy: only accept a checkpoint if it matches almost all parameters; otherwise skip it and fall back to the train-vote prior, (2) fix a serious NetMix bug where `hrnet_w18` feature dimension is not 2048 (so the FC layer is wrong and weight loading is unreliable), and (3) make the prior computation match the competition’s soft-label distribution by normalizing per-row votes first before averaging. These are minimal, directly score-relevant fixes and still guarantee a valid `submission.csv`.'
- What this solution (achieved 1.39779) has done: 'Your current KL (1.39779, lower-is-better) is still far from the target (0.50887), so the smallest safe improvement is to stop rebuilding the dataset/dataloader for every weight file and instead reuse it per mode, ensuring deterministic inference and avoiding accidental data-order/worker nondeterminism that can hurt ensemble averaging. I also set seeds and deterministic flags (no approximations, no early stopping) to stabilize predictions across runs and make the weight ensemble actually average consistent outputs. Finally, I keep your model architectures, preprocessing, and post-processing identical, but add a tiny guard to ensure test `eeg_id` alignment is preserved and predictions are always finite before normalization (preventing rare NaN propagation that can explode KL). These changes are minimal, execution-safe, and directly aimed at improving the validity/quality of the averaged probabilities.'
- What this solution (achieved 1.39779) has done: 'I make two minimal, score-relevant fixes that keep your modeling/inference logic intact: (1) fix a channel bug in `NetMix` where `hrnet_w18` was created with `in_chans=3` even though the forward pass builds a 3-channel tensor, but the input tensor is assembled from 8 channels; this mismatch can prevent reliable weight loading and degrade predictions, and (2) adjust DataLoader settings to avoid worker-related nondeterminism/ordering issues by using `persistent_workers` and `drop_last=False` explicitly. These changes should increase the chance that mix weights load and are used correctly (reducing KL toward the target) while preserving the same architectures, preprocessing, softmax, and ensembling semantics. The submission writing and probability normalization remain unchanged.'

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
    "train_csv": "/kaggle/input/hms-harmful-brain-activity-classification/train.csv",
}




## === cell 2
def list_weight_files(path: str):
    """
    Allow both (a) a directory of weight files and (b) a single weight file path.
    This increases the chance we actually load weights (instead of falling back), improving KL toward target.
    """
    if not isinstance(path, str) or len(path) == 0:
        return []
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        return []
    files = []
    for fn in sorted(os.listdir(path)):
        fp = os.path.join(path, fn)
        if os.path.isfile(fp) and fp.lower().endswith((".pth", ".pt", ".bin")):
            files.append(fp)
    return files


CFG["weights_spec"] = list_weight_files(CFG["weights_spec"])
CFG["weights_eeg"] = list_weight_files(CFG["weights_eeg"])
CFG["weights_mix"] = list_weight_files(CFG["weights_mix"])

print("Found weights_spec:", len(CFG["weights_spec"]))
print("Found weights_eeg :", len(CFG["weights_eeg"]))
print("Found weights_mix :", len(CFG["weights_mix"]))
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

SPEC_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        image = self.wave_transform(x)
        return image


transform_func = TransformMel().to(SPEC_DEVICE)


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

            x_tensor = torch.from_numpy(x).to(SPEC_DEVICE)
            mel_spec = transform_func(x_tensor)
            mel_spec = mel_spec.detach().to("cpu").numpy()

            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[
                :, :width
            ]
            img[:, :, k] += mel_spec_db

        img[:, :, k] /= 4.0

    return img


if len(CFG["weights_mix"]) > 0:
    all_fs = os.listdir(data_dir)
    all_specs = {}
    for item in tqdm(all_fs):
        eeg_id = item.rsplit(".", 1)[0]
        eeg_path = f"/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/{eeg_id}.parquet"
        eeg_spec = spectrogram_from_eeg(eeg_path)
        all_specs[eeg_id] = eeg_spec

    with open("eeg_specs_dict.pkl", "wb") as file:
        pickle.dump(all_specs, file)
else:
    pass




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

        self.ll = ll
        self.rr = rr
        print(self.ll, self.rr)
        self.training_flag = training_flag
        self.shuffle = shuffle

        self.raw_data_set_size = None  # decided by self.parse_file

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
            if os.path.exists("eeg_specs_dict.pkl"):
                with open("eeg_specs_dict.pkl", mode="rb") as f:
                    self.eeg_specs = pickle.load(f)
            else:
                raise FileNotFoundError(
                    "eeg_specs_dict.pkl not found; mix mode requires precomputed EEG spectrograms."
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
        eeg_path = f"/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/{dp['eeg_id']}.parquet"
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
        spec_path = f"/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/{dp['spectrogram_id']}.parquet"
        spec = pd.read_parquet(spec_path)

        spec = spec.values[:, 1:]

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
            raise ValueError("Must set one of use_eeg/use_spec/use_mix to True")
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




## === cell 7
class NetMix(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.preprocess = Transform()
        self.model = timm.create_model("hrnet_w18", pretrained=False, in_chans=3)
        self.fc = nn.Linear(getattr(self.model, "num_features", 2048), 6, bias=True)
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
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 9
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


seed_everything(42)

test_df = pd.read_csv(CFG["data"])
test_df = test_df.sort_values("eeg_id").reset_index(drop=True)
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
n_test = len(test_df)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

all_model_preds = []


def _extract_state_dict(ckpt):
    """
    Robustly handle common checkpoint wrappers so we actually load trained weights.
    """
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _strip_prefix_if_present(state_dict, prefix="module."):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(str(k).startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {str(k)[len(prefix) :]: v for k, v in state_dict.items()}


def _load_state_dict_safely(
    model: nn.Module, state_dict: dict, mode: str, weight_path: str
):
    """
    Change (score-relevant): prevent 'random-init inference' when checkpoints don't match.
    We accept a checkpoint only if almost all parameters load; otherwise skip this weight file.
    """
    if not isinstance(state_dict, dict):
        print(
            f"[{mode}] {os.path.basename(weight_path)}: invalid state_dict type, skipping"
        )
        return False

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    overlap = len(model_keys & ckpt_keys) / max(1, len(model_keys))

    if overlap < 0.90:
        print(
            f"[{mode}] {os.path.basename(weight_path)}: low key overlap {overlap:.3f}, skipping"
        )
        return False

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing_ratio = len(missing) / max(1, len(model_keys))
    if missing_ratio > 0.05:
        print(
            f"[{mode}] {os.path.basename(weight_path)}: too many missing params "
            f"({len(missing)}/{len(model_keys)}), skipping"
        )
        return False

    if len(unexpected) > 0:
        print(
            f"[{mode}] Loaded with unexpected={len(unexpected)} (ok) from {os.path.basename(weight_path)}"
        )
    return True


def _make_loader_for_mode(mode: str):
    if mode == "eeg":
        test_dataset = AlaskaDataIter(
            test_df, training_flag=False, shuffle=False, use_eeg=True
        )
    elif mode == "spec":
        test_dataset = AlaskaDataIter(
            test_df, training_flag=False, shuffle=False, use_spec=True
        )
    elif mode == "mix":
        test_dataset = AlaskaDataIter(
            test_df, training_flag=False, shuffle=False, use_mix=True
        )
    else:
        raise ValueError("Unknown mode")

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_worker"],
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(CFG["num_worker"] > 0),
    )
    return test_loader


def run_ensemble(weights, mode):
    if len(weights) == 0:
        return None

    test_loader = _make_loader_for_mode(mode)

    preds_list = []
    used = 0
    for model_weight in weights:
        if mode == "eeg":
            model = NetEeg()
        elif mode == "spec":
            model = NetSpec()
        elif mode == "mix":
            model = NetMix()
        else:
            raise ValueError("Unknown mode")

        ckpt = torch.load(model_weight, map_location="cpu")
        state_dict = _extract_state_dict(ckpt)
        state_dict = _strip_prefix_if_present(state_dict, prefix="module.")
        state_dict = _strip_prefix_if_present(state_dict, prefix="model.")

        ok = _load_state_dict_safely(
            model, state_dict, mode=mode, weight_path=model_weight
        )
        if not ok:
            continue

        model.to(device)

        pred_dict = inference_function(test_loader, model, device)
        preds_list.append(pred_dict["predictions"])
        used += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if used == 0:
        return None

    preds_arr = np.mean(np.array(preds_list), axis=0)
    return preds_arr


print("infer with weights_eeg")
pred_eeg = run_ensemble(CFG["weights_eeg"], mode="eeg")
if pred_eeg is not None:
    all_model_preds.append(pred_eeg)

print("infer with weights_spec")
pred_spec = run_ensemble(CFG["weights_spec"], mode="spec")
if pred_spec is not None:
    all_model_preds.append(pred_spec)

print("infer with weights_mix")
pred_mix = run_ensemble(CFG["weights_mix"], mode="mix")
if pred_mix is not None:
    all_model_preds.append(pred_mix)

if len(all_model_preds) == 0:
    train_df = pd.read_csv(CFG["train_csv"], usecols=TARGETS)
    row_sums = train_df[TARGETS].sum(axis=1).replace(0, np.nan)
    soft = (train_df[TARGETS].div(row_sums, axis=0)).fillna(1.0 / len(TARGETS))
    prior = soft.mean(axis=0).astype(np.float64).values
    prior = prior / prior.sum()
    predictions = np.tile(prior[None, :], (n_test, 1)).astype(np.float32)
else:
    predictions = np.mean(np.stack(all_model_preds, axis=0), axis=0).astype(np.float32)

predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)

predictions = np.clip(predictions, 1e-6, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

predictions.shape



## === cell 11
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
print(
    "Row sums (min/max):",
    sub[TARGETS].sum(axis=1).min(),
    sub[TARGETS].sum(axis=1).max(),
)
sub.head()
