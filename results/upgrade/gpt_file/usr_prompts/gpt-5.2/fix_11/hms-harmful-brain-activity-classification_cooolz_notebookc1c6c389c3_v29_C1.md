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

0.5476822909560153

# 6. Current score

1.39779

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I first fix the missing-weight-path failure by automatically falling back to a safe baseline when the external weight directories don’t exist, so inference can run end-to-end. Next, I remove the hard CUDA dependency in the EEG→mel feature precompute and make it device-agnostic (CPU by default in this environment), plus fix a couple of indexing/shape issues that currently break model input construction. Finally, I ensure we always write a valid `submission.csv` with the exact required columns, correct row count, and per-row probabilities that sum to 1 (including a robust fallback to uniform predictions if no usable weights are found). These changes are correctness/stability-oriented (score-neutral compared to a non-running script) and at least produce a valid submission so you can obtain a score to calibrate further.'
- What this solution (achieved 1.40995) has done: 'I make three minimal, score-relevant fixes that typically move KL-divergence down for this competition without changing your model definitions or inference loop structure. First, I fix the spectrogram extraction bug (`r` never increments), which currently makes all 4 “regions” identical and degrades predictions. Second, I ensure the EfficientNet-B5 classifier uses the correct feature dimension from the backbone instead of a hardcoded `2048`, preventing partially-random heads when weights don’t perfectly match (a common hidden score-killer). Third, I add a tiny, metric-aligned post-processing step (probability temperature smoothing + renormalization) to reduce overconfident probabilities, which usually lowers KL when models are weak/mismatched and should help move 1.40995 toward your 0.5477 target.'
- What this solution (achieved 1.40995) has done: 'I make two minimal, score-relevant fixes that preserve your model/inference core logic but should reduce KL from 1.40995 toward your 0.5477 target. First, I calibrate the ensemble probabilities slightly more toward “softer” distributions by increasing the post-softmax temperature smoothing (this often lowers KL when predictions are overconfident/mismatched). Second, I weight the three prediction sources (spec/eeg/mix) with a conservative prior favoring spectrogram models (typically stronger in this competition) instead of a plain mean, while keeping the same models and weight loading. Everything else (data reading, feature extraction, architectures, inference loop, and submission schema) stays the same and still writes a valid `submission.csv` with row-wise probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'I make two minimal, score-directed adjustments that keep your models and inference loop intact but should reduce KL from 1.40995 toward 0.5477. First, I apply the temperature “smoothing” in logit space (equivalently `softmax(log(p)/T)`) instead of `p**(1/T)`; this is a closer-to-correct calibration step for already-softmaxed outputs and typically lowers KL when outputs are miscalibrated. Second, I add a tiny uniform-mixture “label-smoothing prior” after ensembling (e.g., `p = (1-α)p + α/6`) with renormalization; this reduces extreme probabilities, which is usually beneficial for KL on this competition when models are imperfect. Everything else (data reading, feature extraction, model definitions, weight loading, ensembling, and submission schema) remains the same and still writes a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I make two score-directed changes that keep your architectures and inference loops intact but should reduce KL from 1.40995 toward the 0.5477 target by fixing calibration/averaging. First, I remove the extra `softmax` inside each model’s `forward` and instead treat model outputs as logits, then ensemble in logit space and apply one final softmax; averaging probabilities can be overconfident/miscalibrated for KL, while logit-averaging is typically better behaved. Second, I apply your existing temperature + uniform-prior smoothing to the final probabilities only (not implicitly multiple times across flips/models), which reduces overconfidence without changing the core model computations. The script still runs end-to-end and writes a valid `submission.csv` with correct columns and per-row sums of 1.'
- What this solution (achieved 1.40995) has done: 'I make two small, score-relevant adjustments that preserve your model architectures and inference loop while aiming to reduce KL from 1.40995 toward 0.5477 by improving probability calibration. First, I load checkpoints more safely by automatically unwrapping common `state_dict`/EMA key formats and stripping `module.` prefixes so your heads/backbones actually use the trained weights (misloaded weights often produce very high KL). Second, I slightly increase the final uniform-prior smoothing and temperature (post-ensemble only, as you already do) to soften overconfident distributions, which typically lowers KL when models are imperfect. Everything else (feature extraction, model forwards, ensembling in logit space, TTA flip, and submission schema) remains unchanged and still writes a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I make two minimal, score-directed changes that preserve your models and inference flow but should reduce KL from 1.40995 toward 0.5477 by preventing silent weight-misload and by ensuring the mix branch produces sensible logits. First, I implement a “best-effort” checkpoint loader that filters out mismatched keys/shapes (instead of leaving large parts randomly initialized under `strict=False`), which typically drops KL a lot when checkpoints don’t exactly match. Second, I fix `NetMix`’s backbone feature-dimension detection by reading the real classifier input size from the timm model (HRNet often doesn’t expose `num_features` as expected), avoiding a wrong FC shape and ineffective weights. Everything else (feature extraction, architectures, TTA, logit-ensembling, final softmax/temperature/prior smoothing, and writing `submission.csv`) stays the same.'
- What this solution (achieved 1.39779) has done: 'I make two minimal, score-relevant changes that preserve your architectures and overall inference flow but should reduce KL from 1.40995 toward 0.5477 by improving how weights are actually applied and by matching the evaluation target definition. First, I fix EfficientNet-B5’s head input feature detection (your current `getattr(..., "num_features", 2048)` can silently be wrong in timm, causing mostly-random logits even when weights load). Second, I convert your outputs to the same *normalized vote distribution* used by the competition (divide by per-row vote sum) and apply a single global prior based on the *train label distribution* (a safe Bayes/Dirichlet-style shrinkage that typically lowers KL without changing model logic), then renormalize to guarantee valid probabilities.'
- What this solution (achieved 1.39779) has done: 'I make two minimal, score-directed changes that keep your model/inference core intact but should reduce KL from 1.39779 toward the 0.5477 target. First, I fix a key input bug in `NetSpec`: it currently concatenates 4 regions along height, which produces a 1200×100 image and mismatches what EfficientNet-B5 was trained on (typically 600×400); changing the concat to build a 600×400 image usually yields a large KL drop without changing architecture. Second, I adjust the ensemble weighting to be dynamic: if only one branch actually loads usable weights, we should use it at weight 1.0 instead of blending with missing branches’ intended weights (this preserves semantics but avoids unintended down-weighting when some parts are absent). Everything else (feature extraction, models, checkpoint loading, logit-ensembling, temperature/prior smoothing, and submission writing) remains the same.'
- What this solution (achieved 1.39779) has done: 'Your current score (1.39779, lower-is-better) is far above the target (0.54768), so we should improve with minimal, metric-aligned changes. I keep your models and inference loop intact, but fix a likely major input mismatch in `NetSpec` (it currently builds a 600×400 image by duplicating content instead of using all 4 regions), which can make loaded weights behave poorly and inflate KL. Then I slightly increase the post-ensemble Dirichlet-style prior shrinkage and temperature softening (post-processing only) to reduce overconfident probabilities, which usually lowers KL when models are imperfect. Finally, I keep the submission formatting/normalization as-is to ensure a valid CSV with row-wise sums of 1.'

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
    "tta_temperature": 1.75,
    "ensemble_weights": {"spec": 0.50, "eeg": 0.25, "mix": 0.25},
    "prior_smoothing": 0.18,  # alpha in p'=(1-alpha)*p + alpha*prior
}

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]




## === cell 2
def list_weight_files(path: str):
    if not isinstance(path, str) or (not os.path.exists(path)):
        return []
    if os.path.isfile(path):
        return [path]
    files = []
    for x in sorted(os.listdir(path)):
        fp = os.path.join(path, x)
        if os.path.isfile(fp) and (
            fp.endswith(".pth") or fp.endswith(".pt") or fp.endswith(".bin")
        ):
            files.append(fp)
    return files


CFG["weights_spec"] = list_weight_files(CFG["weights_spec"])
CFG["weights_eeg"] = list_weight_files(CFG["weights_eeg"])
CFG["weights_mix"] = list_weight_files(CFG["weights_mix"])

CFG



## === cell 3
data_dir = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs"
NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]


class EEGMelTransform(nn.Module):
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

    def forward(self, x: torch.Tensor):
        return self.wave_transform(x)


def spectrogram_from_eeg(parquet_path, transform_func, device, display=False):
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

            x_tensor = torch.from_numpy(x).to(device=device, dtype=torch.float32)
            mel_spec = transform_func(x_tensor)
            mel_spec = mel_spec.detach().cpu().numpy()

            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[
                :, :width
            ]
            img[:, :, k] += mel_spec_db

        img[:, :, k] /= 4.0

    return img


EGO_SPECS_PKL = "eeg_specs_dict.pkl"
if (len(CFG["weights_mix"]) > 0) and (not os.path.exists(EGO_SPECS_PKL)):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform_func = EEGMelTransform().to(device)
    transform_func.eval()

    all_fs = os.listdir(data_dir)
    all_specs = {}
    for item in tqdm(all_fs, desc="Precompute EEG mel specs"):
        eeg_id = item.rsplit(".", 1)[0]
        eeg_path = f"/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/{eeg_id}.parquet"
        eeg_spec = spectrogram_from_eeg(eeg_path, transform_func, device)
        all_specs[str(eeg_id)] = eeg_spec

    with open(EGO_SPECS_PKL, "wb") as file:
        pickle.dump(all_specs, file)




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
        self.training_flag = training_flag
        self.shuffle = shuffle
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
        self.use_spec = use_spec
        self.use_mix = use_mix

        if self.use_mix:
            with open(EGO_SPECS_PKL, mode="rb") as f:
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

        kg_spec = np.transpose(kg_spec, axes=[1, 2, 0])  # (H,W,C=4)
        X[14:-14, :, :4] = kg_spec[:, 22:-22, :]  # 300 -> 256
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
            raise ValueError("At least one of use_eeg/use_spec/use_mix must be True.")
        return data.astype(np.float32)




## === cell 5
def _infer_timm_in_features(model: nn.Module, fallback: int = 2048) -> int:
    for attr in ["classifier", "fc", "head"]:
        if hasattr(model, attr):
            layer = getattr(model, attr)
            if isinstance(layer, nn.Linear):
                return int(layer.in_features)
    if hasattr(model, "get_classifier"):
        layer = model.get_classifier()
        if isinstance(layer, nn.Linear):
            return int(layer.in_features)
    nf = getattr(model, "num_features", None)
    if nf is not None and int(nf) > 0:
        return int(nf)
    return int(fallback)


class NetSpec(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=3)
        in_features = _infer_timm_in_features(self.model, fallback=2048)
        self.fc = nn.Linear(in_features, 6, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bs = x.size(0)

        top = torch.cat([x[:, 0:1, :, :], x[:, 1:2, :, :]], dim=3)  # (bs,1,300,200)
        bot = torch.cat([x[:, 2:3, :, :], x[:, 3:4, :, :]], dim=3)  # (bs,1,300,200)
        x1 = torch.cat([top, bot], dim=2)  # (bs,1,600,200)
        top2 = torch.cat([x[:, 1:2, :, :], x[:, 0:1, :, :]], dim=3)  # (bs,1,300,200)
        bot2 = torch.cat([x[:, 3:4, :, :], x[:, 2:3, :, :]], dim=3)  # (bs,1,300,200)
        x2 = torch.cat([top2, bot2], dim=2)  # (bs,1,600,200)
        x1 = torch.cat([x1, x2], dim=3)  # (bs,1,600,400)

        x = torch.cat([x1, x1, x1], dim=1)  # (bs,3,600,400)

        if CFG["flip"]:
            x_flip = torch.flip(x, [3])
            x = torch.cat([x, x_flip], dim=0)

        x = self.model.forward_features(x)
        x = self.avg(x)

        x = x.view((2 * bs if CFG["flip"] else bs), -1)
        x = self.dropout(x)
        x = self.fc(x)  # logits

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

    def forward(self, x):
        image = self.wave_transform(x)
        image = self.am2db(image)
        n, c, h, w = image.size()
        image = image[:, :, : int(20 / 100 * h + 10), :]
        image = torch.reshape(image, shape=[n, 4, -1, w])
        return image


class NetEeg(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.preprocess = Transform()
        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=4)
        in_features = _infer_timm_in_features(self.model, fallback=2048)
        self.fc = nn.Linear(in_features, 6, bias=True)
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
        x = self.fc(x)  # logits

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

        in_features = getattr(self.model, "num_features", None)
        if in_features is None or int(in_features) <= 0:
            if hasattr(self.model, "classifier") and isinstance(
                self.model.classifier, nn.Linear
            ):
                in_features = self.model.classifier.in_features
            elif hasattr(self.model, "head") and isinstance(self.model.head, nn.Linear):
                in_features = self.model.head.in_features
            else:
                in_features = 2048  # final fallback (keeps core logic intact)

        self.fc = nn.Linear(int(in_features), 6, bias=True)
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

        x = x.view((2 * bs if CFG["flip"] else bs), -1)
        x = self.dropout(x)
        x = self.fc(x)  # logits

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
        for X in tqdm_test_loader:
            X = X.to(device)
            with torch.no_grad():
                y_preds = model(X)
            preds.append(y_preds.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)




## === cell 9
test_df = pd.read_csv(CFG["data"])
test_df.head(5)



## === cell 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _clean_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "model_state_dict", "net", "ema", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict) and len(ckpt[key]) > 0:
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        return ckpt
    new_sd = {}
    for k, v in ckpt.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        new_sd[nk] = v
    return new_sd


def _load_state_dict_best_effort(model: nn.Module, state_dict: dict):
    model_sd = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k in model_sd and hasattr(v, "shape") and model_sd[k].shape == v.shape:
            filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return missing, unexpected, filtered


def predict_with_weights(weight_files, dataset_kwargs, model_ctor):
    if len(weight_files) == 0:
        return None

    all_model_logits = []
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

    for model_weight in weight_files:
        model = model_ctor()
        ckpt = torch.load(model_weight, map_location=device)
        state_dict = _clean_state_dict(ckpt)

        if isinstance(state_dict, dict):
            missing, unexpected, filtered = _load_state_dict_best_effort(
                model, state_dict
            )
            if len(filtered) < max(10, int(0.1 * len(model.state_dict()))):
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                continue

        model.to(device)
        logits = inference_function(test_loader, model, device)
        all_model_logits.append(logits)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if len(all_model_logits) == 0:
        return None

    return np.mean(np.stack(all_model_logits, axis=0), axis=0)  # (n_samples, 6) logits


pred_eeg = None
pred_spec = None
pred_mix = None

if len(CFG["weights_eeg"]) > 0:
    print("infer with weights_eeg")
    pred_eeg = predict_with_weights(CFG["weights_eeg"], dict(use_eeg=True), NetEeg)

if len(CFG["weights_spec"]) > 0:
    print("infer with weights_spec")
    pred_spec = predict_with_weights(CFG["weights_spec"], dict(use_spec=True), NetSpec)

if len(CFG["weights_mix"]) > 0 and os.path.exists(EGO_SPECS_PKL):
    print("infer with weights_mix")
    pred_mix = predict_with_weights(CFG["weights_mix"], dict(use_mix=True), NetMix)

w = CFG.get("ensemble_weights", {"spec": 1 / 3, "eeg": 1 / 3, "mix": 1 / 3})
parts = []
weights = []
if pred_spec is not None:
    parts.append(pred_spec)
    weights.append(float(w.get("spec", 0.0)))
if pred_eeg is not None:
    parts.append(pred_eeg)
    weights.append(float(w.get("eeg", 0.0)))
if pred_mix is not None:
    parts.append(pred_mix)
    weights.append(float(w.get("mix", 0.0)))

if len(parts) == 0:
    logits_ens = None
else:
    weights = np.array(weights, dtype=np.float32)
    if weights.sum() <= 0:
        logits_ens = np.mean(np.stack(parts, axis=0), axis=0)
    else:
        weights = weights / weights.sum()
        stacked = np.stack(parts, axis=0)  # (n_models, n_samples, 6) logits
        logits_ens = np.tensordot(weights, stacked, axes=([0], [0]))  # (n_samples, 6)

print("Logits available:", logits_ens is not None)
if logits_ens is not None:
    print("logits_ens.shape:", logits_ens.shape)



## === cell 11
train_path = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
train_df = pd.read_csv(train_path, usecols=TARGETS)
train_votes = train_df[TARGETS].values.astype(np.float64)
row_sums = train_votes.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
train_probs = train_votes / row_sums
prior = train_probs.mean(axis=0).astype(np.float32)
prior = prior / prior.sum()
print("Train prior:", dict(zip(TARGETS, prior.round(6))))

sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})

if logits_ens is None:
    preds = np.tile(prior[None, :], (len(test_df), 1)).astype(np.float32)
else:
    logits = logits_ens.astype(np.float32)

    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    preds = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    eps = 1e-8
    preds = np.clip(preds, eps, 1.0)

    T = float(CFG.get("tta_temperature", 1.0))
    if T != 1.0:
        logp = np.log(preds)
        logp = logp / T
        logp = logp - logp.max(axis=1, keepdims=True)
        preds = np.exp(logp)
        preds = preds / preds.sum(axis=1, keepdims=True)

    alpha = float(CFG.get("prior_smoothing", 0.0))
    if alpha > 0:
        preds = (1.0 - alpha) * preds + alpha * prior[None, :]
        preds = preds / preds.sum(axis=1, keepdims=True)

sub[TARGETS] = preds
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
print(
    "Row sums (min/max):",
    sub[TARGETS].sum(axis=1).min(),
    sub[TARGETS].sum(axis=1).max(),
)
sub.head()
