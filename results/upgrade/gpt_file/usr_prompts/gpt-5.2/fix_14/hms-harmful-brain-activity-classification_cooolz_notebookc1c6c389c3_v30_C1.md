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

0.4392794798070529

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the path/weights handling so the notebook no longer crashes when the external weight directories are missing or contain subfolders, and so `torch.load` is only called on actual model files. Because your current run cannot yield a submission, I add a safe fallback that produces a valid probability submission (row-wise sum to 1) even when no weights are available in the environment, without changing the model definitions or inference logic. I also ensure the predictions array always has shape `(len(test_df), 6)` and is properly normalized before writing `submission.csv`. These changes are score-neutral in intent (they unblock execution); if weights are present, the code use them as originally intended.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.43928), so we should improve performance while keeping your architecture and inference logic intact. The biggest issue is that you only run the EEG model, even if spectrogram and mix weights exist; enabling the existing Spec/Mix inference paths and then averaging the three prediction streams is a minimal, legitimate improvement that typically reduces KL a lot. I also make the DataLoader more deterministic/stable (no behavior change in math) and ensure we always normalize/clamp probabilities safely for a valid submission. No training is added; we only use the same models you already defined and the weights you already point to.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is much worse than the target (0.43928), so we should improve predictions with the smallest legitimate changes that keep your model definitions and inference intact. The biggest likely issue is that test-time stochasticity is still enabled because the models are never put into `eval()` mode until inside the inference loop, after dropout layers are already constructed (but eval should still disable them); more importantly, BatchNorm running stats can differ if any accidental train-mode is used—so we force `model.eval()` immediately after loading weights and also disable any nondeterministic CuDNN behavior for stability. Next, we fix a subtle but impactful input mismatch: `NetSpec` is defined with `in_chans=3` but the dataset returns 4 channels; your forward slices 4 regions and then makes RGB by repetition, so it should still work, but we ensure the spec tensor is exactly what the model expects by explicitly selecting only 4 regions and not relying on silent shape quirks. Finally, we do a minimal, metric-aligned probability floor (slightly higher than 1e-9) to reduce extreme KL penalties from near-zero probabilities while still preserving normalization and semantics.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.43928), so we should improve the prediction quality with minimal, safe changes that don’t alter your model architectures or training/inference approach. The biggest likely issue is that the test spectrogram parsing has a bug: the `r` pointer never advances, so all 4 “regions” are extracted from the same time window, hurting `NetSpec` and `NetMix`; fixing this should materially reduce KL while preserving intended semantics. Second, `NetMix` uses an `hrnet_w18` backbone but a hardcoded `Linear(2048,6)` head, which is very likely a dimension mismatch with real weights; we keep the backbone but make the classifier input dimension adaptive from the actual backbone output to correctly load and use your existing weights. Finally, we keep your probability clipping/normalization but add a tiny post-ensemble temperature smoothing (very mild) to reduce overconfident near-zero probabilities that are heavily penalized by KL, which typically improves KL without changing the overall logic.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.43928), so we should improve prediction quality with minimal, metric-aligned changes while keeping your models and inference flow intact. The biggest likely KL penalty here comes from overconfident near-zero probabilities created by applying `softmax` inside the model and then doing an extra log/temperature/exp step on already-normalized probabilities; we replace that with a tiny, safe Dirichlet-style smoothing (epsilon mix with uniform) that reduces catastrophic KL without changing the core inference logic. We also weight the ensemble slightly toward Spec (often stronger than EEG) using a very small change (weighted average rather than equal average) while keeping the same three-stream ensemble semantics. Finally, we make weight loading more robust to common checkpoint formats (`state_dict`, `model`) so that you actually use the available weights instead of partially loading random heads, which should move the score materially toward the target.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.43928), so the smallest likely-impactful fixes are ones that ensure your existing pretrained weights are actually being used as intended and that the output probabilities are less catastrophically overconfident (KL is very sensitive to near-zero probs). I (1) make checkpoint loading robust to common key-prefix formats (e.g., `module.`) so more weights load correctly with the same models, and (2) switch the post-processing from a tiny uniform-mix to a slightly stronger, metric-aligned “floor + renormalize” that directly prevents near-zero probabilities that blow up KL. I also (3) slightly adjust ensemble weights toward Spec (often strongest in this competition) but keep the same three-stream weighted-average ensemble semantics. These are minimal changes that preserve your architectures and inference flow while plausibly moving KL substantially toward the target.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far above the target (0.43928), so we should improve prediction quality with minimal, inference-only fixes that preserve your model definitions and overall flow. The biggest likely score hit is an input mismatch: your `Transform` expects EEG-shaped wave input but `NetMix` never uses it, while `get_mix()` returns an 8-channel image; we fix `NetMix` by removing the unused `Transform` (no architecture change to the actual mix backbone path) and by inferring the correct `fc` input dimension via a tiny dummy forward so your pretrained HRNet weights/head are actually used. Next, we make weight loading stricter-but-safe by adapting common checkpoint formats and only allowing `strict=False` when needed, which reduces accidental partial loading. Finally, we keep your probability floor/renormalization but slightly reduce the floor (still preventing near-zero KL explosions) to avoid over-smoothing that can hurt score.'
- What this solution (achieved 1.40995) has done: 'To move your KL score down toward the target with minimal risk, I’m keeping your models and inference intact and focusing on two high-impact correctness/metric-aligned tweaks. First, I make weight loading more robust by handling common checkpoint wrappers and key prefixes (including `model_state_dict` and nested `state_dict`), and I load the `fc` head weights safely when checkpoint dimensions match—this increases the chance you’re actually using the intended pretrained weights rather than partially-random heads. Second, I replace the relatively aggressive probability floor (0.002) with a smaller, KL-friendly Dirichlet-style smoothing (mixing with uniform) which still prevents near-zero probabilities (catastrophic for KL) but avoids over-smoothing that can hurt accuracy. These changes preserve your core architecture/inference while plausibly reducing KL substantially from 1.41 toward ~0.44.'
- What this solution (achieved 1.40995) has done: 'I keep your model definitions and inference flow intact and only make two minimal, high-impact fixes that plausibly move KL down toward your target. First, I make checkpoint loading robust to checkpoints that store full `model` objects or store weights under nested keys, because silently failing to load the intended weights can easily explain a very poor KL like 1.41. Second, I slightly strengthen the KL-safety post-processing by using a small probability floor + renormalize (still tiny, but more direct than the current uniform-mix) to avoid catastrophic KL from near-zero probabilities while keeping predictions valid and normalized. These changes don’t alter architectures, training loops, or feature extraction, but they increase the chance you’re actually using your pretrained weights and reduce extreme overconfidence penalties.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far above the target (0.4393), so we should improve score by making sure your intended pretrained weights actually load and your inference matches the saved checkpoints. The most likely blocker is that `NetSpec`/`NetEeg` hardcode `fc=Linear(2048,6)` which often mismatches EfficientNet-B5 feature dims (typically 2048 only for some models; for B5 it’s commonly 2048 but can differ depending on timm impl), causing partial/random heads when `strict=False`—so we infer the correct feature dim with a dummy forward (same approach you already used for HRNet in `NetMix`) and then load strictly when possible. Next, we keep your ensemble and post-processing but make checkpoint loading smarter: try strict load; if it fails, attempt to drop only mismatched `fc.*` keys (so backbone loads cleanly) and keep strict=False as last resort. These are minimal, inference-only changes that preserve your architecture/backbones and should materially reduce KL by avoiding partially-random classifiers.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far above the target (0.43928), so we should improve predictive correctness without changing your model architectures or inference flow. The most likely high-impact issue is that your EEG `Transform` outputs 4 channels but your `NetEeg` backbone is created with `in_chans=4`, while `Transform` currently reshapes to 4 channels but starts from 16 leads; if the reshape is not consistent with the intended grouping, it can scramble information and hurt KL. I make the `Transform` reshape deterministic and consistent by explicitly splitting the 16 lead-difference channels into 4 groups (4 each) before the spectrogram, then stacking those 4 group-spectrograms—this preserves the same “EEG→spectrogram→EffNet” core logic while fixing a likely silent mismatch. I also slightly reduce the post-probability floor (still preventing near-zero KL blowups) to avoid over-smoothing that can keep you stuck at high KL.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far above the target (0.4393), so we should improve prediction quality with minimal, inference-only adjustments that preserve your models and data flow. The biggest likely penalty is the aggressive probability floor (5e-5) which over-smooths and can materially worsen KL; I reduce it to a much smaller, KL-safe floor while keeping strict row-normalization. I also make the ensemble weights adaptive: if only one stream is available, it uses it fully; if multiple are available, it defaults to equal-weight averaging (less assumption-heavy than hardcoding 0.60/0.25/0.15) to better match unknown weight quality. Finally, I keep your robust checkpoint loading but ensure we don’t silently skip mix inference due to a missing local pickle by auto-building `eeg_specs_dict.pkl` from test EEGs (using your existing EEG→spectrogram Transform), so `NetMix` can contribute when mix weights exist.'
- What this solution (achieved 1.40995) has done: 'Your current KL is far above the target (lower is better), so the most likely minimal-win improvement is ensuring the spectrogram slicing bug is fixed (so each of the 4 regions comes from the correct time window) and that model checkpoints actually load the right classifier head when possible. I (1) fix `get_spec()` to use the correct row window per region (this is a correctness bug that can severely hurt Spec/Mix streams), (2) make weight loading attempt strict load after dynamically matching `fc` size to the checkpoint when it’s clearly different (so you don’t silently run with a random head), and (3) keep your probability normalization but slightly increase the final probability floor to reduce catastrophic KL from near-zero predictions without changing the ensemble logic. These changes preserve your architectures/backbones and inference flow, but should materially move KL down toward your target.'

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
def _list_weight_files(path):
    exts = (".pt", ".pth", ".bin")
    if not isinstance(path, str) or not os.path.exists(path) or not os.path.isdir(path):
        return []
    out = []
    for name in sorted(os.listdir(path)):
        fp = os.path.join(path, name)
        if os.path.isfile(fp) and name.lower().endswith(exts):
            out.append(fp)
    return out


CFG["weights_spec"] = _list_weight_files(CFG["weights_spec"])
CFG["weights_eeg"] = _list_weight_files(CFG["weights_eeg"])
CFG["weights_mix"] = _list_weight_files(CFG["weights_mix"])

print(
    "Found weight files:",
    {
        "spec": len(CFG["weights_spec"]),
        "eeg": len(CFG["weights_eeg"]),
        "mix": len(CFG["weights_mix"]),
    },
)
CFG




## === cell 3
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
        self.RL = ["Fp2", "F8", "T4", "T6", "O2"]
        self.LP = ["Fp1", "F3", "C3", "P3", "O1"]
        self.RP = ["Fp2", "F4", "C4", "P4", "O2"]

        self.leads_dict = {value: index for index, value in enumerate(self.eeg_nms)}

        self.use_eeg = use_eeg
        self.use_spec = use_spec
        self.use_mix = use_mix

        if self.use_mix:
            with open("eeg_specs_dict.pkl", mode="rb") as f:
                self.eeg_specs = pickle.load(f)

    def __getitem__(self, item):
        return self.single_map_func(self.df.iloc[item], self.training_flag)

    def __len__(self):
        return len(self.df)

    def brain_lead(self, waves):
        waves = copy.deepcopy(waves)

        brain_leads = [self.LL, self.RL, self.LP, self.RP]
        leads = []

        for chain in brain_leads:
            for i in range(len(chain) - 1):
                tmp_lead = (
                    waves[self.leads_dict[chain[i]]]
                    - waves[self.leads_dict[chain[i + 1]]]
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
            r0 = region * 300
            c0 = region * 100
            img = spec[r0 : r0 + 300, c0 : c0 + 100].T
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)
            img = np.nan_to_num(img, nan=0.0)
            images.append(img)

        images = np.stack(images, -1)  # (100,300,4)
        data = np.transpose(images, [2, 0, 1])  # (4,100,300)
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

        return np.ascontiguousarray(data.astype(np.float32))




## === cell 4
class NetSpec(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=3)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

        with torch.no_grad():
            self.model.eval()
            dummy = torch.zeros(
                1, 3, 400, 300
            )  # matches constructed x shape (3, 4*100, 300)
            feat = self.model.forward_features(dummy)
            feat = self.avg(feat).view(1, -1)
            in_features = int(feat.shape[1])

        self.fc = nn.Linear(in_features, 6, bias=True)

    def forward(self, x):
        bs = x.size(0)

        x = x[:, :4, :, :]

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
        n, c, t = x.shape
        if c != 16:
            image = self.wave_transform(x)
            image = self.am2db(image)
            n2, c2, h, w = image.size()
            image = image[:, :, : int(20 / 100 * h + 30), :]
            image = torch.reshape(image, shape=[n2, 4, -1, w])
            return image

        x = x.view(n, 4, 4, t).mean(dim=2)  # (n, 4, t)

        image = self.wave_transform(x)  # (n, 4, f, tt)
        image = self.am2db(image)

        n, c, h, w = image.size()
        image = image[:, :, : int(20 / 100 * h + 30), :]
        return image


class NetEeg(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.preprocess = Transform()

        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=4)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

        with torch.no_grad():
            self.model.eval()
            dummy_wave = torch.zeros(
                1, 16, 10000
            )  # matches brain_lead output (16, 10000)
            dummy_img = self.preprocess(dummy_wave)
            feat = self.model.forward_features(dummy_img)
            feat = self.avg(feat).view(1, -1)
            in_features = int(feat.shape[1])

        self.fc = nn.Linear(in_features, 6, bias=True)

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




## === cell 6
class NetMix(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.model = timm.create_model("hrnet_w18", pretrained=False, in_chans=3)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)

        with torch.no_grad():
            self.model.eval()
            dummy = torch.zeros(1, 3, 128, 256)
            feat = self.model.forward_features(dummy)
            feat = self.avg(feat).view(1, -1)
            in_features = int(feat.shape[1])

        self.fc = nn.Linear(in_features, 6, bias=True)

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




## === cell 7
def inference_function(test_loader, model, device):
    model.eval()
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for step, X in enumerate(tqdm_test_loader):
            X = X.to(device)
            with torch.no_grad():
                y_preds = model(X)
            preds.append(y_preds.to("cpu").numpy())
    prediction_dict = {"predictions": np.concatenate(preds, axis=0)}
    return prediction_dict




## === cell 8
test_df = pd.read_csv(CFG["data"])
test_df.head(5)



## === cell 9
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
try:
    torch.use_deterministic_algorithms(False)
except Exception:
    pass

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


def _unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "model_state_dict", "net", "weights"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def _clean_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if len(keys) == 0:
        return state_dict

    prefixes = ("module.", "model.", "net.")
    for pref in prefixes:
        if sum((isinstance(k, str) and k.startswith(pref)) for k in keys) >= int(
            0.8 * len(keys)
        ):
            state_dict = {k[len(pref) :]: v for k, v in state_dict.items()}
            keys = list(state_dict.keys())
    return state_dict


def _extract_state_dict_from_checkpoint(model, ckpt):
    if isinstance(ckpt, nn.Module):
        return ckpt.state_dict()
    if isinstance(ckpt, dict):
        sd = _unwrap_state_dict(ckpt)
        if isinstance(sd, dict):
            return sd
    return ckpt


def _maybe_resize_fc_from_checkpoint(model, state_dict):
    try:
        if not (isinstance(state_dict, dict) and hasattr(model, "fc")):
            return
        if "fc.weight" not in state_dict:
            return
        w = state_dict["fc.weight"]
        if not hasattr(w, "shape") or len(w.shape) != 2:
            return
        ckpt_out, ckpt_in = int(w.shape[0]), int(w.shape[1])
        if isinstance(model.fc, nn.Linear):
            if model.fc.in_features != ckpt_in or model.fc.out_features != ckpt_out:
                model.fc = nn.Linear(ckpt_in, ckpt_out, bias=True)
    except Exception:
        return


def _try_load_with_fc_fallback(model, state_dict):
    _maybe_resize_fc_from_checkpoint(model, state_dict)
    try:
        model.load_state_dict(state_dict, strict=True)
        return True
    except Exception:
        pass

    if isinstance(state_dict, dict):
        sd2 = dict(state_dict)
        for k in list(sd2.keys()):
            if k.startswith("fc."):
                sd2.pop(k, None)
        try:
            model.load_state_dict(sd2, strict=False)
            return False
        except Exception:
            pass

    model.load_state_dict(state_dict, strict=False)
    return False


def _build_eeg_specs_dict_if_missing(
    test_df, out_path="eeg_specs_dict.pkl", batch_size=64
):
    if os.path.exists(out_path):
        return True

    eeg_dir = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs"
    if not os.path.isdir(eeg_dir):
        return False

    preprocess = Transform().to(device)
    preprocess.eval()

    eeg_nms = [
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
    LL = ["Fp1", "F7", "T3", "T5", "O1"]
    RL = ["Fp2", "F8", "T4", "T6", "O2"]
    LP = ["Fp1", "F3", "C3", "P3", "O1"]
    RP = ["Fp2", "F4", "C4", "P4", "O2"]
    leads_dict = {value: index for index, value in enumerate(eeg_nms)}
    brain_leads = [LL, RL, LP, RP]

    def brain_lead(waves_20xt):
        leads = []
        for chain in brain_leads:
            for i in range(len(chain) - 1):
                tmp = (
                    waves_20xt[leads_dict[chain[i]]]
                    - waves_20xt[leads_dict[chain[i + 1]]]
                )
                leads.append(tmp)
        return np.concatenate([leads], axis=0)  # (16, t)

    def load_one_eeg(eeg_id):
        eeg_path = os.path.join(eeg_dir, f"{eeg_id}.parquet")
        eeg = pd.read_parquet(eeg_path)
        eeg = eeg.iloc[0:10000]
        waves = eeg.values.T  # (20, t)

        for i in range(waves.shape[0]):
            m = np.nanmean(waves[i])
            if np.isnan(waves[i]).mean() < 1:
                waves[i] = np.nan_to_num(waves[i], nan=m)
            else:
                waves[i] = 0.0

        waves = np.asarray(waves, dtype=np.float64)
        waves = mne.filter.filter_data(waves, 200, 0, 20, verbose=False)
        waves16 = brain_lead(waves)  # (16, t)
        return waves16.astype(np.float32)

    eeg_ids = test_df["eeg_id"].tolist()
    out = {}
    buf_ids, buf_waves = [], []

    for eid in tqdm(eeg_ids, desc="Building eeg_specs_dict.pkl", unit="eeg"):
        buf_ids.append(eid)
        buf_waves.append(load_one_eeg(eid))

        if len(buf_ids) >= batch_size:
            x = torch.from_numpy(np.stack(buf_waves, axis=0)).to(
                device
            )  # (n, 16, 10000)
            with torch.no_grad():
                img = preprocess(x)  # (n, 4, f, tt)
            img = img.detach().cpu().numpy()

            for i, eid2 in enumerate(buf_ids):
                e = img[i]  # (4, f, tt)
                e = np.transpose(e, (1, 2, 0))  # (f, tt, 4)
                f, t, c = e.shape
                tgt_f, tgt_t = 128, 256
                ef = np.zeros((tgt_f, tgt_t, 4), dtype=np.float32)
                f0 = max(0, (f - tgt_f) // 2)
                t0 = max(0, (t - tgt_t) // 2)
                e_crop = e[f0 : f0 + min(tgt_f, f), t0 : t0 + min(tgt_t, t), :]
                ef[: e_crop.shape[0], : e_crop.shape[1], :] = e_crop
                out[str(eid2)] = ef

            buf_ids, buf_waves = [], []
            torch.cuda.empty_cache()
            gc.collect()

    if len(buf_ids) > 0:
        x = torch.from_numpy(np.stack(buf_waves, axis=0)).to(device)
        with torch.no_grad():
            img = preprocess(x)
        img = img.detach().cpu().numpy()
        for i, eid2 in enumerate(buf_ids):
            e = img[i]
            e = np.transpose(e, (1, 2, 0))
            f, t, c = e.shape
            tgt_f, tgt_t = 128, 256
            ef = np.zeros((tgt_f, tgt_t, 4), dtype=np.float32)
            f0 = max(0, (f - tgt_f) // 2)
            t0 = max(0, (t - tgt_t) // 2)
            e_crop = e[f0 : f0 + min(tgt_f, f), t0 : t0 + min(tgt_t, t), :]
            ef[: e_crop.shape[0], : e_crop.shape[1], :] = e_crop
            out[str(eid2)] = ef

    with open(out_path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    return True


def _predict_from_weights(weight_files, dataset_kwargs, model_ctor):
    if len(weight_files) == 0:
        return None

    test_dataset = AlaskaDataIter(
        test_df, training_flag=False, shuffle=False, **dataset_kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        CFG["batch_size"],
        num_workers=CFG["num_worker"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(CFG["num_worker"] > 0),
    )

    fold_preds = []
    for model_weight in weight_files:
        model = model_ctor()
        ckpt = torch.load(model_weight, map_location="cpu")
        state_dict = _extract_state_dict_from_checkpoint(model, ckpt)
        state_dict = _clean_state_dict_keys(state_dict)

        _try_load_with_fc_fallback(model, state_dict)
        model.to(device)
        model.eval()

        prediction_dict = inference_function(test_loader, model, device)
        fold_preds.append(prediction_dict["predictions"])

        del model, ckpt, state_dict, prediction_dict
        torch.cuda.empty_cache()
        gc.collect()

    del test_loader, test_dataset
    torch.cuda.empty_cache()
    gc.collect()

    fold_preds = np.array(fold_preds, dtype=np.float64)
    return fold_preds.mean(axis=0)


pred_eeg = _predict_from_weights(CFG["weights_eeg"], {"use_eeg": True}, NetEeg)
pred_spec = _predict_from_weights(CFG["weights_spec"], {"use_spec": True}, NetSpec)

pred_mix = None
if len(CFG["weights_mix"]) > 0:
    if not os.path.exists("eeg_specs_dict.pkl"):
        ok = _build_eeg_specs_dict_if_missing(
            test_df, out_path="eeg_specs_dict.pkl", batch_size=64
        )
        if not ok:
            print("Could not build eeg_specs_dict.pkl; skipping mix inference.")
    if os.path.exists("eeg_specs_dict.pkl"):
        pred_mix = _predict_from_weights(CFG["weights_mix"], {"use_mix": True}, NetMix)


def _ensemble(pred_eeg, pred_spec, pred_mix):
    parts = []
    if pred_spec is not None:
        parts.append(pred_spec)
    if pred_eeg is not None:
        parts.append(pred_eeg)
    if pred_mix is not None:
        parts.append(pred_mix)
    if len(parts) == 0:
        return None
    stacked = np.stack(parts, axis=0)
    return stacked.mean(axis=0)


predictions = _ensemble(pred_eeg, pred_spec, pred_mix)
if predictions is None:
    print("No weights found; writing a valid uniform-probability submission fallback.")
    predictions = np.full((n_test, 6), 1.0 / 6.0, dtype=np.float64)

predictions = np.asarray(predictions, dtype=np.float64)
if predictions.ndim != 2 or predictions.shape[0] != n_test or predictions.shape[1] != 6:
    raise ValueError(
        f"Predictions have invalid shape {predictions.shape}, expected ({n_test}, 6)."
    )

predictions = np.clip(predictions, 1e-12, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

prob_floor = 1e-6
predictions = np.maximum(predictions, prob_floor)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

print(
    "predictions shape:",
    predictions.shape,
    "row sum min/max:",
    predictions.sum(1).min(),
    predictions.sum(1).max(),
)



## === cell 10
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
sub.head()
