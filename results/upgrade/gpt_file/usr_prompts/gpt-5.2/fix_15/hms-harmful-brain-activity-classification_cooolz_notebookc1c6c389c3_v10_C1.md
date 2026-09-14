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

0.4280474113589396

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I first make the script robust to missing pretrained weight files by auto-detecting which listed `.pth` paths actually exist; if none exist, it fall back to a valid uniform-probability submission (so you always get a working `.csv`). Next, I fix the spectrogram slicing bug that left `r=0` unchanged (all 4 “regions” were identical), by correctly slicing the 300-row blocks; this preserves the intended core logic and improves inference quality when weights are available. Finally, I guarantee the submission has exactly 6 probability columns, clamp/renormalize to ensure each row sums to 1, and write `submission.csv` end-to-end without shape errors.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower is better) is far from the target (0.4280), so the most likely “minimal change with big impact” is fixing a clear bug in the EEG montage construction: you accidentally use `RP` twice and never use `RR`, which degrades EEG-model inputs and ensemble quality. I change only that list to `["LL","LP","RP","RR"]` (preserving the exact downstream pipeline and model architecture), and keep everything else the same. Additionally, I enforce a deterministic row order in the DataLoader (no behavior change intended, but reduces accidental nondeterministic drift) and keep the same probability clamping/renormalization to guarantee valid submissions. This should improve the EEG branch (and thus the ensemble) toward the target without changing the core approach.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower is better) is much worse than the target (0.4280), so we should make small, legitimate fixes that improve correctness without changing the model architectures or training/inference approach. The biggest likely issue is input scaling: the EEG branch is filtered but never normalized, and the spectrogram branch is log-scaled but not standardized, which can severely mismatch what the provided EfficientNet weights expect at inference. I add lightweight, per-sample normalization consistent with typical HMS baselines (robust centering/scaling for EEG; mean/std normalization for spectrogram images) while keeping the same data extraction, models, softmax, and ensembling. I also set deterministic seeds/CuDNN flags to reduce run-to-run drift (should not worsen score, just stabilize it).'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower is better) is far from the target (0.4280), so we should make a minimal change that improves correctness and better matches the KL metric without changing the models or training/inference flow. The biggest likely issue is a train/test preprocessing mismatch for the spectrogram branch: `NetSpec` is defined with `in_chans=3` but the dataset returns 4 channels and the forward concatenation is internally inconsistent; this can severely degrade the spectrogram-model predictions even if weights load. I fix `NetSpec` to properly consume 4-channel input (set `in_chans=4` and remove the unnecessary channel replication), keeping the rest of the pipeline identical. I also average multiple windows from each test spectrogram (still deterministic, no sampling) to reduce sensitivity to the unknown 10-second label position within the 10-minute test spectrogram, which typically improves KL for this competition while preserving the same architecture and loss semantics.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower is better) is far from the target (0.4280), so we make one minimal, score-relevant fix that improves correctness without changing the model architectures or inference loop: ensure the EEG preprocessing returns the exact tensor shape that `NetEeg` expects. Right now the EEG branch returns a 2D array `(n_leads, time)` and the `Transform` treats it like `(N,C,L)`, which silently breaks/warps features and can heavily hurt KL. I fix this by reshaping the EEG sample to `(C=4, L=10000)` (the 4 brain “chains” produced by `brain_lead`) before returning it, keeping all filtering/normalization identical. Everything else (weights loading, softmax, ensembling, clipping/renorm, and CSV writing) stays the same.'
- What this solution (achieved 1.40995) has done: 'To move the KL score down toward the target with minimal risk, I fix one clear EEG-shape bug that likely makes the EEG branch predictions very poor: `brain_lead()` produces 16 differential channels, but the code forcibly reshapes it to `(4, -1)`, mixing channels and time and breaking the `Transform` spectrogram conversion. I keep the exact same EEG filtering, robust normalization, model architectures, and inference loop, but return EEG tensors as `(C=4, L=10000)` by averaging the 4 differential pairs per brain region (LL/LP/RP/RR), which matches what `Transform` expects (`torch.reshape(..., [n,4,-1,w])`). I also set `drop_last=False` explicitly and add a small safety check for EEG shape to prevent silent degradation. These changes should improve correctness substantially (and thus reduce KL) without changing the overall approach.'
- What this solution (achieved 1.40995) has done: 'Your KL (1.40995, lower is better) is far above the target (0.428), so we should make minimal, correctness-focused changes that most likely recover the intended ensemble behavior. The biggest high-impact issue consistent with your code is a mismatch between the EEG spectrogram preprocessing output size and what EfficientNet-B5 expects: your `Transform` computes a spectrogram but never resizes, so the spatial resolution can differ from what the pretrained weights were trained on, degrading predictions. I apply the existing `self.resizer` inside `Transform.forward()` (no change to architecture/loops/loss; it’s a missing preprocessing step), and I also make weight loading robust to common checkpoint wrappers (`state_dict` key) to ensure the real weights actually load (otherwise you’re effectively running random-init, which matches your poor KL). These are minimal patches aimed at substantially lowering KL toward the target while keeping everything else the same and still producing a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'The current KL (1.40995, lower is better) is far above the target (0.4280), so we need a minimal change that improves correctness without changing the model architectures or inference loop. The biggest likely remaining issue is that the EEG branch’s `Transform` expects a 3D tensor `(N, C=4, L)` but the DataLoader provides `(N, 4, 10000)` which is correct, yet `torchaudio.transforms.Spectrogram` can behave inconsistently across versions unless the input is explicitly treated as `(batch, channels, time)` and cast to float32/contiguous. I make `Transform.forward()` explicitly enforce shape/dtype/contiguity and use the intended channel-wise spectrogram path, which should recover the behavior the checkpoints were trained with (reducing KL) while preserving the same model and preprocessing steps. I also set `torch.set_float32_matmul_precision("high")` to reduce numeric drift and keep the same submission formatting and probability normalization.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower is better) is far above the target (0.4280), so we need a minimal, high-impact correctness fix rather than tuning. The most likely remaining issue is that your EEG preprocessing builds an EEG “image” with a different frequency band layout than what the EEG EfficientNet checkpoints were trained on: you hard-crop the spectrogram height using a heuristic `: int(20/100*h + 30)` instead of selecting the intended low-frequency band deterministically. I replace that height cropping with an explicit frequency-bin selection consistent with 0–20 Hz given `n_fft=1024` and `sr=200` (so inference matches training expectations), while keeping the same models, weights, inference loop, and submission formatting. I also add a tiny safety clamp to avoid empty crops in edge cases, but otherwise leave everything unchanged.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995; lower is better) is much worse than the target (0.4280), so the most likely “minimal change with large impact” is that your ensemble is effectively not using the intended checkpoints. I (1) make weight discovery point to the *actual* available baseline/eeg checkpoint locations under `/kaggle/input/hms-harmful-brain-activity-classification/` (so real weights load instead of falling back to near-random outputs), and (2) make checkpoint loading robust to common prefixes like `module.` so `load_state_dict` actually matches. I also add a tiny assert/log that confirms how many weights were found/loaded, while keeping your exact models, preprocessing, inference loop, softmax, ensembling, and submission formatting unchanged.'
- What this solution (achieved 1.40995) has done: 'We make two minimal, score-relevant fixes that preserve your ensemble/models/inference loop: (1) stop hard-cropping test EEGs at offset=0 and instead use the central 50s window (25–75s) from each EEG recording, matching the training label’s “central 10 seconds” intent and reducing distribution mismatch; and (2) ensure the test DataLoader uses `persistent_workers=True` and a small `prefetch_factor` when workers>0 to stabilize throughput (no semantics change) so you can finish within the time budget and actually use all folds/weights. Everything else (architectures, preprocessing transforms, softmax, ensembling, clipping/renorm, and submission format) remains unchanged.'
- What this solution (achieved 1.40995) has done: 'Your KL (1.40995; lower is better) is far above the target (0.4280), so we need a minimal correctness fix that is likely preventing the provided checkpoints from being used as intended. The highest-impact issue is that your checkpoint key-cleaning currently strips `"model."` from *all* keys, which breaks loading when the checkpoints were saved with prefixes like `"model.fc.*"` (you end up with `"fc.*"` which won’t match `"model.fc.*"`), effectively leaving much of the network randomly initialized and hurting KL. I change the key-cleaning to only remove the `"module."` prefix (common for DDP) and otherwise keep keys intact, so weights load correctly; everything else (models, preprocessing, inference loop, ensembling, probability normalization, and submission writing) remains the same. This is a small change but is plausibly large in score impact because it directly affects whether inference uses trained weights.'
- What this solution (achieved 1.40995) has done: 'Your KL (1.40995, lower is better) is still far above the target (0.4280), so the most likely minimal-impact, high-gain fix is ensuring the **checkpoint weights actually load onto the correct parameter names** (otherwise you’re close to random-init, which matches the poor score). I keep your exact model architectures and inference loop, but change `_clean_state_dict_keys` to also handle the very common `model.` prefix (and nested `model.model.`) while **not** stripping anything else; then I load with `strict=True` when feasible (and fall back to `strict=False` only if necessary) so we don’t silently run partially-uninitialized networks. This is a small, directly score-relevant correctness change and should move KL substantially downward toward the target without altering evaluation semantics. The submission formatting, clipping, and renormalization remain unchanged.'
- What this solution (achieved 1.40995) has done: 'To move KL down toward the 0.428 target with minimal change, the most likely high-impact fix is ensuring your test-time preprocessing matches what the provided checkpoints were trained on. Your current spectrogram standardization is global-per-sample, which often mismatches these HMS baselines (they typically used per-channel normalization), and your EEG spectrogram uses `AmplitudeToDB` which can be inconsistent with checkpoints trained on `log1p` magnitude. I (1) change spectrogram normalization to per-channel mean/std (same data, same model), and (2) change EEG spectrogram scaling to `log1p` magnitude (same STFT, same crop, same resize) while keeping everything else—including architectures, ensembling, inference loop, and submission formatting—unchanged. These are small, deterministic preprocessing adjustments that commonly reduce KL substantially without altering the overall approach.'

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
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

CFG = {
    "batch_size": 32,
    "num_worker": 4,
    "data": "/kaggle/input/hms-harmful-brain-activity-classification/test.csv",
    "weights_spec": [
        "/kaggle/input/hms-harmful-brain-activity-classification/hms-baseline/fold0_epoch_4_val_loss_0.549755.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/hms-baseline/fold1_epoch_4_val_loss_0.524753.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/hms-baseline/fold2_epoch_4_val_loss_0.511551.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/hms-baseline/fold3_epoch_4_val_loss_0.548698.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/hms-baseline/fold4_epoch_3_val_loss_0.666013.pth",
    ],
    "weights_eeg": [
        "/kaggle/input/hms-harmful-brain-activity-classification/hms-eeg/fold0_epoch_4_val_loss_0.584720.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/hms-eeg/fold1_epoch_4_val_loss_0.590993.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/hms-eeg/fold2_epoch_4_val_loss_0.603758.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/hms-eeg/fold3_epoch_4_val_loss_0.580059.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/hms-eeg/fold4_epoch_4_val_loss_0.545819.pth",
    ],
    "flip": True,
    "spec_time_windows": 5,  # evenly-spaced windows across time for each region (deterministic)
    "eeg_sr": 200,
    "eeg_fmax_hz": 20.0,
    "eeg_fmin_hz": 0.0,
    "eeg_center_seconds": 50.0,  # consolidated test EEGs are ~100s; use 25..75s window
    "eeg_window_seconds": 50.0,
}




## === cell 2
def _existing_paths(paths):
    ex = [p for p in paths if isinstance(p, str) and os.path.exists(p)]
    return ex


CFG["weights_spec_exist"] = _existing_paths(CFG["weights_spec"])
CFG["weights_eeg_exist"] = _existing_paths(CFG["weights_eeg"])

print("Found spec weights:", len(CFG["weights_spec_exist"]))
print("Found eeg weights :", len(CFG["weights_eeg_exist"]))
if len(CFG["weights_spec_exist"]) == 0 and len(CFG["weights_eeg_exist"]) == 0:
    print("WARNING: No weights found; will fall back to uniform predictions.")




## === cell 3
class AlaskaDataIter:
    def __init__(self, df, training_flag=False, shuffle=False, use_eeg=False):

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

        data = np.concatenate([leads], axis=0)  # (16, T)
        return data

    def _robust_zscore(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.4826 * mad  # consistent with std for normal data
        if not np.isfinite(scale) or scale < eps:
            scale = np.std(x)
        if not np.isfinite(scale) or scale < eps:
            scale = 1.0
        x = (x - med) / (scale + eps)
        return x

    def _spec_extract(self, spec_2d: np.ndarray) -> np.ndarray:
        """
        Deterministically average multiple time windows per region.
        Returns: (4, 100, 300) float32.

        Change (score-relevant, minimal): normalize PER-CHANNEL (per region) rather than
        over all channels jointly. This typically matches HMS baseline checkpoints better
        and improves KL without altering model/inference logic.
        """
        images = []
        n_freq, n_time = spec_2d.shape  # expected around (1200, 400)

        win_w = 100
        if n_time <= win_w:
            starts = [0]
        else:
            k = int(CFG.get("spec_time_windows", 1))
            k = max(1, k)
            starts = np.linspace(0, n_time - win_w, num=k)
            starts = [int(round(s)) for s in starts]
            starts = sorted(set([min(max(0, s), n_time - win_w) for s in starts]))

        for region in range(4):
            r = region * 300
            region_block = spec_2d[r : r + 300, :]  # (300, time)

            acc = None
            for s in starts:
                patch = region_block[:, s : s + win_w].T  # (100, 300)
                patch = np.clip(patch, np.exp(-4), np.exp(8))
                patch = np.log(patch)
                patch = np.nan_to_num(patch, nan=0.0)
                if acc is None:
                    acc = patch
                else:
                    acc = acc + patch
            img = acc / float(len(starts))
            images.append(img)

        images = np.stack(images, -1)  # (100, 300, 4)
        data = np.transpose(images, [2, 0, 1]).astype(np.float32)  # (4, 100, 300)

        mu = data.mean(axis=(1, 2), keepdims=True)
        sd = data.std(axis=(1, 2), keepdims=True)
        sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0).astype(np.float32)
        data = (data - mu.astype(np.float32)) / sd
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return data

    def single_map_func(self, dp, is_training):
        """Data augmentation function."""
        if self.use_eeg:
            eeg_path = (
                "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/%s.parquet"
                % (dp["eeg_id"])
            )
            eeg = pd.read_parquet(eeg_path)

            sr = int(CFG.get("eeg_sr", 200))
            win_s = float(CFG.get("eeg_window_seconds", 50.0))
            center_s = float(CFG.get("eeg_center_seconds", 50.0))
            win_len = int(round(win_s * sr))
            start = int(round((center_s - win_s / 2.0) * sr))
            start = max(0, min(start, max(0, len(eeg) - win_len)))
            eeg = eeg.iloc[start : start + win_len]

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

            waves = self.brain_lead(waves)  # (16, 10000)

            waves = self._robust_zscore(waves.astype(np.float32)).astype(np.float32)

            if waves.ndim != 2 or waves.shape[0] != 16:
                raise RuntimeError(
                    f"Unexpected brain_lead shape: {waves.shape}, expected (16, T)"
                )
            waves = waves.reshape(4, 4, -1).mean(axis=1)  # (4, 10000)
            data = waves
        else:
            spec_path = (
                "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/%s.parquet"
                % (dp["spectrogram_id"])
            )
            spec = pd.read_parquet(spec_path)
            spec = spec.values[:, 1:]  # drop time column

            data = self._spec_extract(spec)

        return data.astype(np.float32)




## === cell 4
class NetSpec(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=4)

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
            n_fft=1024, hop_length=50, power=1
        )
        self.resizer = nn.UpsamplingBilinear2d(size=[160, 320])

        self.sr = float(CFG.get("eeg_sr", 200))
        self.n_fft = 1024
        self.fmin = float(CFG.get("eeg_fmin_hz", 0.0))
        self.fmax = float(CFG.get("eeg_fmax_hz", 20.0))

    def forward(self, x):
        if x.ndim != 3:
            raise RuntimeError(f"EEG input must be 3D (N,C,L); got {tuple(x.shape)}")
        x = x.contiguous().float()

        image = self.wave_transform(x)  # (N,C,freq,time), magnitude
        image = torch.log1p(image)

        n, c, h, w = image.size()

        hz_per_bin = self.sr / float(self.n_fft)
        lo = int(np.floor(self.fmin / hz_per_bin))
        hi = int(np.ceil(self.fmax / hz_per_bin)) + 1  # inclusive->exclusive
        lo = max(0, min(lo, h - 1))
        hi = max(lo + 1, min(hi, h))  # ensure non-empty crop
        image = image[:, :, lo:hi, :]

        n, c, h, w = image.size()
        image = torch.reshape(image, shape=[n, 4, -1, w])
        image = self.resizer(image)
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
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for X in tqdm_test_loader:
            X = X.to(device)
            with torch.no_grad():
                y_preds = model(X)
            y_preds = softmax(y_preds)
            preds.append(y_preds.detach().cpu().numpy())
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
n_test = len(test_df)

predictions_list = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _unwrap_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "net", "model_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _clean_state_dict_keys(state_dict, model_keys=None):
    """
    Make weight loading robust to common wrappers so we don't accidentally run
    partially-random models (high KL). We ONLY remove well-known prefixes (module., model.)
    when doing so increases key overlap with the actual model.
    """
    if not isinstance(state_dict, dict):
        return state_dict

    keys = list(state_dict.keys())

    def strip_prefix(prefix: str):
        out = {}
        for k, v in state_dict.items():
            nk = k[len(prefix) :] if k.startswith(prefix) else k
            out[nk] = v
        return out

    if any(k.startswith("module.") for k in keys):
        state_dict = strip_prefix("module.")
        keys = list(state_dict.keys())

    if model_keys is not None:
        model_keys_set = set(model_keys)

        if any(k.startswith("model.") for k in keys):
            stripped = {}
            for k, v in state_dict.items():
                nk = k[len("model.") :] if k.startswith("model.") else k
                stripped[nk] = v

            overlap_before = len(set(keys) & model_keys_set)
            overlap_after = len(set(stripped.keys()) & model_keys_set)
            if overlap_after > overlap_before:
                state_dict = stripped
                keys = list(state_dict.keys())

        if any(k.startswith("model.model.") for k in keys):
            stripped2 = {}
            for k, v in state_dict.items():
                nk = k[len("model.model.") :] if k.startswith("model.model.") else k
                stripped2[nk] = v

            overlap_before = len(set(keys) & model_keys_set)
            overlap_after = len(set(stripped2.keys()) & model_keys_set)
            if overlap_after > overlap_before:
                state_dict = stripped2
                keys = list(state_dict.keys())

    return state_dict


def _run_one_model(weight_path, use_eeg):
    if use_eeg:
        test_dataset = AlaskaDataIter(
            test_df, training_flag=False, shuffle=False, use_eeg=True
        )
        model = NetEeg()
    else:
        test_dataset = AlaskaDataIter(
            test_df, training_flag=False, shuffle=False, use_eeg=False
        )
        model = NetSpec()

    g = torch.Generator()
    g.manual_seed(0)

    loader_kwargs = dict(
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_worker"],
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        generator=g,
    )
    if CFG["num_worker"] > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    test_loader = DataLoader(test_dataset, **loader_kwargs)

    ckpt = torch.load(weight_path, map_location=device)
    state_dict = _unwrap_state_dict(ckpt)

    state_dict = _clean_state_dict_keys(
        state_dict, model_keys=model.state_dict().keys()
    )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(
            f"[load_state_dict strict=True failed] {os.path.basename(weight_path)}: {e}"
        )
        incompatible = model.load_state_dict(state_dict, strict=False)
        if hasattr(incompatible, "missing_keys") and hasattr(
            incompatible, "unexpected_keys"
        ):
            if len(incompatible.missing_keys) > 0:
                print(
                    f"[load_state_dict strict=False] missing_keys={len(incompatible.missing_keys)} for {os.path.basename(weight_path)}"
                )
            if len(incompatible.unexpected_keys) > 0:
                print(
                    f"[load_state_dict strict=False] unexpected_keys={len(incompatible.unexpected_keys)} for {os.path.basename(weight_path)}"
                )

    model.to(device)

    pred = inference_function(test_loader, model, device)["predictions"]
    del model, test_loader, test_dataset, state_dict, ckpt
    torch.cuda.empty_cache()
    gc.collect()
    return pred


for model_weight in CFG["weights_spec_exist"]:
    predictions_list.append(_run_one_model(model_weight, use_eeg=False))

for model_weight in CFG["weights_eeg_exist"]:
    predictions_list.append(_run_one_model(model_weight, use_eeg=True))

if len(predictions_list) == 0:
    predictions = np.full((n_test, 6), 1.0 / 6.0, dtype=np.float32)
else:
    predictions = np.mean(np.stack(predictions_list, axis=0), axis=0).astype(np.float32)

if predictions.ndim != 2 or predictions.shape[0] != n_test:
    raise RuntimeError(
        f"Predictions shape mismatch: got {predictions.shape}, expected ({n_test}, 6)"
    )
if predictions.shape[1] != 6:
    raise RuntimeError(f"Predictions must have 6 columns, got {predictions.shape[1]}")

predictions = np.clip(predictions, 1e-8, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)



## === cell 9
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
print(sub.head())
print(
    "Row-sum check (min/mean/max):",
    sub[TARGETS].sum(axis=1).min(),
    sub[TARGETS].sum(axis=1).mean(),
    sub[TARGETS].sum(axis=1).max(),
)
