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
scipy==1.15.3
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

0.2828732630152467

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I first fix the missing external weight path by making the code fall back to the provided sample_submission (uniform probabilities) when weights are unavailable, instead of crashing. I also fix the weight-file discovery logic so it only loads actual checkpoint files (not directories), preventing the `IsADirectoryError`. Finally, I ensure the prediction array always has shape `(len(test_df), 6)` and is properly normalized to sum to 1 per row so the submission is valid for the KL-divergence metric.'
- What this solution (achieved 1.40995) has done: 'Your current score is far worse than the target (lower is better), and the main reason is that the model is not actually loading the provided 2nd-place weights (your weight path points to a non-existent dataset, so you mostly submit the uniform fallback). I make the code auto-discover and load the EEG-raw checkpoints from the competition dataset folder that already exists in your input tree, while keeping the exact same model, preprocessing, TTA, and averaging logic. I also make weight loading robust to common checkpoint formats (`state_dict`, `model`, `module.*`) so valid weights actually get applied instead of silently failing. These minimal changes should move the score substantially toward your target without altering core architecture/training semantics, and the script still always emit a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your current score is far above the target (lower is better), so we need to actually use the intended trained checkpoints rather than effectively submitting near-uniform probabilities. I make the weight auto-discovery both broader and more precise by specifically searching for the `hms-2nd-place-solution` dataset and an `hms-eeg_raw` subfolder, then I load *all* matching checkpoint files recursively. I also make weight loading tolerant to minor key mismatches by using `strict=False` (to avoid silently skipping all weights due to a single unexpected key) while still preserving the exact same model and inference pipeline. Finally, I set deterministic flags for stability and keep the output normalization exactly as required by the KL metric.'
- What this solution (achieved 1.40995) has done: 'Your current score is far worse than the target (lower is better), so the smallest meaningful improvement is to ensure you actually load the intended checkpoints and map them onto the correct model parameters. I (1) make weight auto-discovery explicitly prefer the `hms-2nd-place-solution/pytorch/hms-eeg_raw` folder and only collect real checkpoint files, and (2) make state-dict extraction more robust by handling common wrappers (including PyTorch Lightning’s `state_dict`) and stripping known prefixes like `model.` in addition to `module.` so weights don’t silently miss most layers. I also (3) fix a likely channel/shape mismatch by ensuring the EEG parquet columns are aligned to the expected 20-channel order before deriving bipolar leads; otherwise the model sees scrambled inputs and performs near-random. These are minimal inference-only fixes that preserve your architecture, preprocessing intent, and softmax-probability semantics, but should move KL substantially toward your target.'
- What this solution (achieved 1.40995) has done: 'Your score is much worse than the target (lower is better), so the smallest meaningful change is to ensure the intended checkpoints are actually found and loaded, instead of silently falling back to near-uniform predictions. I keep the exact same model/inference logic, but (1) fix weight autodiscovery to search the real Kaggle input tree for `hms-2nd-place-solution` robustly, (2) fix EEG column alignment so the bipolar lead computation is fed correctly ordered channels (otherwise the model sees scrambled inputs and performs near-random), and (3) make inference faster/stabler without changing semantics by reusing the same DataLoader per TTA and enabling `torch.inference_mode()`/non-blocking transfers. These are inference-only fixes that should move KL substantially toward your target while still producing a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I make the smallest changes that plausibly move KL toward your target by ensuring the model actually receives the expected EEG channel ordering and by preventing accidental “random-ish” predictions from partially-loaded checkpoints. Specifically, I (1) make EEG column alignment robust by renaming common parquet variants (e.g., `T7/T8/P7/P8`) back to the expected `T3/T4/T5/T6` names before reindexing, and (2) require a high fraction of model parameters to load from each checkpoint (otherwise skip that checkpoint instead of averaging in a bad model). These are inference-only safeguards that preserve your architecture and softmax semantics, but avoid silently using scrambled inputs / broken weights, which is a common cause of very poor KL. The script still always produce a valid `submission.csv` with probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'I make two minimal inference-only fixes that are likely causing your very poor KL: (1) actually load and use the intended EEG-raw checkpoints by relaxing the overly-strict “loaded fraction” filter (your current 0.90 threshold can skip all real checkpoints if their key naming differs slightly), and (2) avoid recomputing/parquet-reading the same EEG twice for TTA by caching the raw EEG array per `eeg_id` inside the dataset, so both normal and flip passes use identical underlying signals (improves stability while keeping the same model and preprocessing). These changes keep your exact model, softmax outputs, and averaging/TTA logic, but should move the score substantially toward the target by ensuring you’re not effectively submitting the uniform fallback. The script still guarantees a valid `submission.csv` with per-row probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'Your current KL is far worse than the target, so the smallest meaningful improvement is to ensure inference matches the 2nd-place EEG-raw pipeline more closely without changing the model itself. I make three minimal fixes that directly affect score: (1) remove dropout randomness at inference by setting `dropout.p=0` after loading weights, (2) stabilize and speed inference by reusing one cached dataset for both normal/flip (so both TTAs use identical base EEG and avoid re-reading/parquet variance), and (3) make checkpoint loading more robust by also stripping `model.model.` (a common nested prefix) and preventing accidental inclusion of non-checkpoint files. These keep the architecture, preprocessing, and softmax semantics identical, but should move KL substantially toward your target by producing deterministic, properly-weighted predictions.'
- What this solution (achieved 1.40995) has done: 'Your current KL is much worse than the target (lower is better), so we should make the smallest changes that help the model produce non-random, properly-aligned probabilities rather than near-uniform outputs. The most likely remaining score killer is a subtle bug in `mirror_eeg`: it uses chained advanced indexing assignment which does not swap in-place correctly in NumPy, so your flip-TTA becomes corrupted and averaging hurts performance. I fix the swap safely using copies (preserving the same TTA intent), and I also ensure EEG column standardization handles common `EEG `-prefixed column names so the channel order is truly correct before bipolar derivation. These are inference-only, minimal changes that keep your architecture, preprocessing, and softmax semantics intact, while plausibly moving KL substantially toward your target.'
- What this solution (achieved 1.40995) has done: 'Your score is far worse than the target (lower is better), so the most likely remaining issue is that the checkpoints are still not actually being used (or are being averaged in incorrectly). I make two minimal, score-relevant fixes: (1) improve checkpoint autodiscovery to also pick up `.pt`/`.pth` files inside common compressed/packaged 2nd-place dataset layouts (often nested under `weights/`, `fold*`, etc.), and (2) fix a critical inference-time shape bug: the model expects input that can be reshaped to `(bs,16,1000,10)`, so the dataset must return `(16,10000)`; currently it returns `(16,10000)` only if the bipolar lead count is 16 and the crop is 10000, but any mismatch silently crashes performance via bad reshaping. I add a strict, minimal safeguard that enforces the expected `(16,10000)` by cropping/padding in time only (no change to feature logic), ensuring the model sees correctly formatted signals. These changes keep your architecture, preprocessing intent, TTA, softmax semantics, and submission format intact, but should move KL substantially toward the target by making the “real model” actually run on correctly shaped inputs.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower is better) is far above the target (0.28287), so the smallest likely win is to ensure we’re actually running the intended trained checkpoints and not silently averaging in “bad/unmatched” ones. I make checkpoint discovery prefer the known `.../pytorch/hms-eeg_raw` tree and add a quick sanity filter to only keep weights whose loaded fraction is high enough to be a real match, instead of mixing in partially-loaded/random models. I also fix a subtle but important EEG flip-TTA bug: the current `mirror_eeg` is swapping indices that don’t exist after bipolar conversion (16ch), which can crash or corrupt flip predictions; we swap the correct left/right bipolar groups (8 channels each). These are inference-only changes that preserve your model architecture, preprocessing intent, and softmax semantics while plausibly moving KL much closer to the target.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower is better) is still far from the target (0.28287), so the smallest meaningful improvement is to ensure we (a) actually find the intended 2nd-place EEG-raw checkpoints and (b) don’t accidentally skip all of them due to an overly-strict “loaded fraction” filter. I minimally adjust checkpoint autodiscovery to explicitly prefer the common fold subdirectories (like `/.../hms-eeg_raw/1`, `/2`, …) and relax `min_loaded_frac` slightly so real checkpoints with minor key naming differences are still used. I also keep your exact model/inference logic but ensure the state-dict extraction covers an additional common wrapper (`ema_state_dict`) to avoid silently loading nothing. These changes are inference-only, preserve the architecture/softmax semantics, and should move KL substantially toward your target by replacing near-uniform outputs with real checkpoint predictions.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995; lower is better) suggests the pipeline is still effectively not using the intended 2nd-place weights (or is averaging in mismatched checkpoints), so the smallest score-relevant changes are to (1) make weight auto-discovery explicitly prefer the actual `/kaggle/input/hms-2nd-place-solution/pytorch/hms-eeg_raw` tree and only select checkpoint-like files, and (2) tighten checkpoint acceptance to avoid averaging in “partially loaded” models that behave near-random. I also fix a likely EEG input scale mismatch by applying the same per-channel standardization (mean/std) after filtering; this preserves your architecture and inference semantics (still softmax probs) but typically makes pretrained EEG models behave correctly. Finally, I keep the submission normalization and add a strict sanity print so you can confirm how many checkpoints are truly used (to avoid silently falling back again).'
- What this solution (achieved 1.40995) has done: 'We make two score-relevant, minimal inference-only changes to move KL down toward your target: first, ensure the EEG input tensor is contiguous before the model’s fixed `view()` reshape (non-contiguous tensors can silently produce incorrect reshapes and near-random predictions). Second, we stop disabling dropout at inference (you already call `model.eval()`, so dropout is off anyway; forcing `p=0.0` can create a mismatch vs the checkpointed model definition and can reduce performance if any weights expect the module graph unchanged). These changes keep the exact model, preprocessing, TTA, checkpoint averaging, and softmax semantics intact, while addressing a common “random-like inference” failure mode and avoiding an unnecessary model-structure mutation.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995; lower is better) strongly indicates inference is still effectively broken (near-uniform / near-random), so the smallest change likely to move toward the target is to make checkpoint loading actually match the model’s parameter names. I fix `load_state_dict` handling (it currently mis-assigns the returned `(missing_keys, unexpected_keys)` tuple) and make the “loaded fraction” calculation use the correct missing-key list, so valid checkpoints are not mistakenly skipped/averaged incorrectly. I also ensure the DataLoader does not respawn workers each time (stability/latency) and keep everything else (model, preprocessing, TTA, softmax, submission normalization) identical. This should materially reduce KL by ensuring real weights are used and averaged correctly.'

# 9. Code solution

## === cell 0
import random
import cv2
import json
import copy
import torch
import gc
import os
import librosa
import pickle
import timm
import mne

import numpy as np

import albumentations as A
import pandas as pd
from tqdm import tqdm

import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.signal import butter, lfilter



## === cell 1
CFG = {
    "batch_size": 32,
    "num_worker": 4,
    "data": "/kaggle/input/hms-harmful-brain-activity-classification/test.csv",
    "weights_eeg_raw": "/kaggle/input/hms-2nd-place-solution/pytorch/hms-eeg_raw/1",
    "sample_sub": "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv",
}




## === cell 2
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


def list_weight_files(path: str, recursive: bool = False):
    if not isinstance(path, str) or not os.path.exists(path):
        return []
    exts = (".pt", ".pth", ".bin", ".ckpt")
    files = []
    if os.path.isfile(path):
        if path.lower().endswith(exts):
            return [path]
        return []
    if not os.path.isdir(path):
        return []

    if not recursive:
        for fn in sorted(os.listdir(path)):
            fp = os.path.join(path, fn)
            if os.path.isfile(fp) and fn.lower().endswith(exts):
                files.append(fp)
        return files

    for dirpath, _, filenames in os.walk(path):
        for fn in filenames:
            if fn.lower().endswith(exts):
                fp = os.path.join(dirpath, fn)
                if os.path.isfile(fp):
                    files.append(fp)
    return sorted(files)


def autodiscover_eeg_raw_weight_dir():
    """
    Why (score): current KL suggests we still aren't reliably using the intended checkpoints.
    Minimal change: explicitly prioritize the known competition dataset layout and only then broaden.
    """
    preferred_dirs = [
        CFG.get("weights_eeg_raw", ""),
        "/kaggle/input/hms-2nd-place-solution/pytorch/hms-eeg_raw/1",
        "/kaggle/input/hms-2nd-place-solution/pytorch/hms-eeg_raw/2",
        "/kaggle/input/hms-2nd-place-solution/pytorch/hms-eeg_raw/3",
        "/kaggle/input/hms-2nd-place-solution/pytorch/hms-eeg_raw/4",
        "/kaggle/input/hms-2nd-place-solution/pytorch/hms-eeg_raw/5",
        "/kaggle/input/hms-2nd-place-solution/pytorch/hms-eeg_raw",
    ]

    candidates = []
    for root in preferred_dirs:
        if isinstance(root, str) and os.path.isdir(root):
            candidates.append(root)

    scan_root = "/kaggle/input"
    if os.path.isdir(scan_root):
        for dirpath, _, _ in os.walk(scan_root):
            low = dirpath.lower()
            if ("2nd" in low or "second" in low) and (
                "eeg_raw" in low or "eeg-raw" in low
            ):
                candidates.append(dirpath)

    best_dir, best_count = None, 0
    for c in sorted(set(candidates)):
        cnt = len(list_weight_files(c, recursive=True))
        if cnt > best_count:
            best_dir, best_count = c, cnt
    return best_dir


_w = list_weight_files(CFG["weights_eeg_raw"], recursive=True)
if len(_w) == 0:
    discovered = autodiscover_eeg_raw_weight_dir()
    if discovered is not None:
        CFG["weights_eeg_raw"] = discovered
        _w = list_weight_files(CFG["weights_eeg_raw"], recursive=True)

CFG["weights_eeg_raw"] = _w
print("Found EEG raw weight files:", len(CFG["weights_eeg_raw"]))
print("Example weight files:", CFG["weights_eeg_raw"][:10])




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
        flip=False,
        use_mne_filter=True,
    ):

        self.flip_eeg = flip
        self.ll = ll
        self.rr = rr

        print(self.ll, self.rr, "with mne filter:", use_mne_filter)

        self.training_flag = training_flag
        self.shuffle = shuffle

        self.raw_data_set_size = None  ##decided by self.parse_file

        self.df = df.reset_index(drop=True)

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
        self.mid = ["Fz", "Cz", "Pz"]
        self.leads_dict = {value: index for index, value in enumerate(self.eeg_nms)}

        self.use_eeg = use_eeg
        self.use_spec = use_spec
        self.use_mix = use_mix
        self.use_mne_filter = use_mne_filter

        self._eeg_cache = {}

        self.expected_ch = 16
        self.expected_t = 10000

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

    def mirror_spec(self, data):
        indx = [1, 0, 3, 2]
        return data[..., indx]

    def mirror_eeg(self, data):
        """
        Why (score): flip-TTA must correctly swap left/right bipolar groups in 16ch space.
        """
        data = np.asarray(data)
        if data.shape[0] != 16:
            return data
        out = data.copy()
        out[0:4, :] = data[4:8, :]
        out[4:8, :] = data[0:4, :]
        out[8:12, :] = data[12:16, :]
        out[12:16, :] = data[8:12, :]
        return out

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        return butter(order, [lowcut, highcut], fs=fs, btype="band")

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def _standardize_eeg_column_names(self, eeg: pd.DataFrame) -> pd.DataFrame:
        """
        Why (score): ensure channel names match expected 20-ch order before reindexing/bipolar derivation.
        """
        if any(isinstance(c, str) and c.startswith("EEG ") for c in eeg.columns):
            eeg = eeg.rename(
                columns={
                    c: c.replace("EEG ", "", 1) if isinstance(c, str) else c
                    for c in eeg.columns
                }
            )

        rename_map = {
            "T7": "T3",
            "T8": "T4",
            "P7": "T5",
            "P8": "T6",
            "Fpz": "Fz",
        }
        cols = list(eeg.columns)
        intersect = set(cols) & set(rename_map.keys())
        if len(intersect) > 0:
            eeg = eeg.rename(columns={k: rename_map[k] for k in intersect})
        return eeg

    def _load_and_preprocess_base_eeg(self, eeg_id: int):
        eeg_path = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/%s.parquet"
            % (eeg_id)
        )
        eeg = pd.read_parquet(eeg_path)
        eeg = self._standardize_eeg_column_names(eeg)

        if all(c in eeg.columns for c in self.eeg_nms):
            eeg = eeg.reindex(columns=self.eeg_nms)
        else:
            missing_cols = [c for c in self.eeg_nms if c not in eeg.columns]
            for c in missing_cols:
                eeg[c] = 0.0
            eeg = eeg.reindex(columns=self.eeg_nms)

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

        waves = self.brain_lead(waves)
        waves = np.array(waves, dtype=np.float64)
        waves = np.clip(waves, -1024, 1024)

        if self.use_mne_filter:
            waves = mne.filter.filter_data(waves, 200, self.ll, self.rr, verbose=False)
        else:
            waves = self.butter_bandpass_filter(waves, 0.5, 20, 200, 2)

        mu = np.mean(waves, axis=1, keepdims=True)
        sig = np.std(waves, axis=1, keepdims=True)
        waves = (waves - mu) / (sig + 1e-6)

        return waves

    def _enforce_expected_shape(self, waves: np.ndarray) -> np.ndarray:
        """
        Why (score): Net1d assumes exactly (16,10000) for its fixed reshape.
        """
        waves = np.asarray(waves)
        if waves.ndim != 2:
            waves = waves.reshape(waves.shape[0], -1)

        if waves.shape[0] < self.expected_ch:
            pad = np.zeros(
                (self.expected_ch - waves.shape[0], waves.shape[1]), dtype=waves.dtype
            )
            waves = np.concatenate([waves, pad], axis=0)
        elif waves.shape[0] > self.expected_ch:
            waves = waves[: self.expected_ch]

        t = waves.shape[1]
        if t < self.expected_t:
            pad = np.zeros((waves.shape[0], self.expected_t - t), dtype=waves.dtype)
            waves = np.concatenate([waves, pad], axis=1)
        elif t > self.expected_t:
            waves = waves[:, : self.expected_t]

        return waves

    def get_eeg(self, dp, is_training, flip=False):
        eeg_id = int(dp["eeg_id"])
        base = self._eeg_cache.get(eeg_id, None)
        if base is None:
            base = self._load_and_preprocess_base_eeg(eeg_id)
            base = self._enforce_expected_shape(base)
            self._eeg_cache[eeg_id] = base

        waves = base.copy()
        if flip:
            waves = self.mirror_eeg(waves)
        return waves

    def single_map_func(self, dp, is_training):
        data = self.get_eeg(dp, is_training, self.flip_eeg)
        return data.astype(np.float32)




## === cell 4
class Net1d(nn.Module):
    def __init__(
        self,
    ):
        super(Net1d, self).__init__()
        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, out_features=6, bias=True)
        self.dropout = nn.Dropout(p=0.5)

    def extract_features(self, x):
        feature1 = self.model.forward_features(x)
        return feature1

    def forward(self, x):
        bs = x.size(0)

        x = x.contiguous()

        reshaped_tensor = x.view(bs, 16, 1000, 10)
        reshaped_and_permuted_tensor = reshaped_tensor.permute(0, 1, 3, 2)
        reshaped_and_permuted_tensor = reshaped_and_permuted_tensor.reshape(
            bs, 16 * 10, 1000
        )
        x = torch.unsqueeze(reshaped_and_permuted_tensor, dim=1)
        x = torch.cat([x, x, x], dim=1)
        bs = x.size(0)

        x = self.extract_features(x)
        x = self.pool(x)
        x = x.view(bs, -1)

        x = self.dropout(x)
        x = self.fc(x)
        x = torch.softmax(x, -1)
        return x




## === cell 5
test_df = pd.read_csv(CFG["data"])
test_df.head(5)




## === cell 6
def inference_function(test_loader, model, device, double_input=False):
    model.eval()
    preds = []
    with torch.inference_mode():
        with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
            for step, X in enumerate(tqdm_test_loader):
                if double_input:
                    wave, spec = X
                    wave = wave.to(device, non_blocking=True)
                    spec = spec.to(device, non_blocking=True)
                    y_preds = model(wave, spec)
                else:
                    X = X.to(device, non_blocking=True)
                    y_preds = model(X)
                preds.append(y_preds.to("cpu").numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 7
def _extract_state_dict(ckpt):
    """
    Why (score): avoid silently not loading weights due to checkpoint wrapper variations.
    """
    if isinstance(ckpt, dict):
        for k in [
            "state_dict",
            "ema_state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
        ]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _strip_known_prefixes(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    def strip_prefix(sd, prefix):
        if any(isinstance(k, str) and k.startswith(prefix) for k in sd.keys()):
            return {k.replace(prefix, "", 1): v for k, v in sd.items()}
        return sd

    state_dict = strip_prefix(state_dict, "module.")
    state_dict = strip_prefix(state_dict, "model.model.")
    state_dict = strip_prefix(state_dict, "model.")
    state_dict = strip_prefix(state_dict, "net.")
    return state_dict


def _loaded_fraction(missing_keys, model: nn.Module) -> float:
    """
    Why (score): use the real missing-key list; previously this was fed the wrong object if load_state_dict
    outputs were mis-assigned, causing good checkpoints to be skipped (near-uniform predictions -> bad KL).
    """
    try:
        total = len(list(model.state_dict().keys()))
        if total <= 0:
            return 0.0
        return float((total - len(missing_keys)) / total)
    except Exception:
        return 0.0


def run_weight_eeg_raw():
    if not CFG["weights_eeg_raw"]:
        print("No EEG raw weights available; skipping model inference.")
        return None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("infer with run_weight_eeg_raw on device:", device)

    loader_kwargs = dict(
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_worker"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(CFG["num_worker"] > 0),
    )

    min_loaded_frac = 0.85

    shared_dataset = AlaskaDataIter(
        test_df,
        training_flag=False,
        shuffle=False,
        use_eeg=True,
        ll=0.5,
        rr=20,
        flip=False,
    )

    predictions = []
    used = 0
    skipped = 0

    for model_weight in CFG["weights_eeg_raw"]:
        if not (isinstance(model_weight, str) and os.path.isfile(model_weight)):
            continue

        model = Net1d()
        ckpt = torch.load(model_weight, map_location="cpu")
        state_dict = _strip_known_prefixes(_extract_state_dict(ckpt))

        incompatible = model.load_state_dict(state_dict, strict=False)
        missing_keys = list(getattr(incompatible, "missing_keys", []))
        unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))

        frac = _loaded_fraction(missing_keys, model)

        if frac < min_loaded_frac:
            skipped += 1
            print(
                f"SKIP {os.path.basename(model_weight)}: loaded_fraction={frac:.3f} "
                f"(missing={len(missing_keys)} unexpected={len(unexpected_keys)})"
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

        used += 1
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            print(
                f"Loaded {os.path.basename(model_weight)} loaded_fraction={frac:.3f} "
                f"missing={len(missing_keys)} unexpected={len(unexpected_keys)}"
            )

        model.to(device)

        shared_dataset.flip_eeg = False
        test_loader = DataLoader(shared_dataset, **loader_kwargs)
        prediction_dict = inference_function(test_loader, model, device)
        predictions.append(prediction_dict["predictions"])

        shared_dataset.flip_eeg = True
        test_loader_flip = DataLoader(shared_dataset, **loader_kwargs)
        prediction_dict = inference_function(test_loader_flip, model, device)
        predictions.append(prediction_dict["predictions"])

        del model, test_loader, test_loader_flip
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print(f"Checkpoints used: {used}, skipped: {skipped}")

    if len(predictions) == 0:
        print("No valid weight files loaded; skipping model inference.")
        return None

    predictions = np.array(predictions)
    predictions = np.mean(predictions, axis=0)
    return predictions


TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

predictions = run_weight_eeg_raw()

if predictions is None:
    sample_sub = pd.read_csv(CFG["sample_sub"])
    predictions = sample_sub[TARGETS].values.astype(np.float32)

predictions = np.asarray(predictions, dtype=np.float32)
if (
    predictions.ndim != 2
    or predictions.shape[0] != len(test_df)
    or predictions.shape[1] != len(TARGETS)
):
    raise ValueError(
        f"Predictions has invalid shape {predictions.shape}; expected ({len(test_df)}, {len(TARGETS)})"
    )

predictions = np.clip(predictions, 1e-8, None)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
print(
    "Row sum check (min/max):",
    sub[TARGETS].sum(axis=1).min(),
    sub[TARGETS].sum(axis=1).max(),
)
sub.head()
