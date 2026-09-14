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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
PyWavelets==1.8.0
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

0.6909634962658863

# 6. Current score

1.41047

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41047) has done: 'I fix the missing-weight crash by making the weight path robust: if the expected checkpoint isn’t available, the code fall back to an untrained EfficientNet and still run end-to-end (this is score-worse but valid, and necessary since no valid submission was produced). I also fix a dataset bug where `label_cols` is undefined and the EEG spectrogram assignment is incorrectly indented inside the spectrogram loop. Finally, I ensure inference uses the correct loader variable, enforce proper probability normalization (sum-to-1) for submission safety, and guarantee the prediction array has shape `(len(test_df), 6)` so the CSV writes without the pandas shape error.'
- What this solution (achieved 1.41047) has done: 'Your current score (1.41047, lower-is-better) is far worse than the target (0.69096), and the main reason is that you’re effectively running an untrained/random EfficientNet because the referenced checkpoint path is not present in your inputs. The smallest change that should materially move the score toward the target is to ensure the script actually uses real pretrained weights available in this environment; we do that by searching `/kaggle/input/**` for a matching `tf_efficientnet_b3*.pth` and using all matches as an ensemble (same architecture/inference, just real weights). As a safety net, we keep your random-init fallback if nothing is found, and we keep the same normalization logic to guarantee a valid KL-safe submission. No changes are made to the model architecture, feature extraction, loss, or training approach (there is no training here), only to robustly locate and load the intended weights.'
- What this solution (achieved 1.41047) has done: 'Your current score is far worse than the target because the script is likely loading either no checkpoint (random init) or the wrong `.pth` files (e.g., optimizer-only, different architectures), so predictions are essentially untrained. I keep your exact model/dataset/inference logic, but make weight loading both stricter (only accept checkpoints that actually match EfficientNet-B3 keys) and safer (normalize common key prefixes like `module.`), so we reliably use real compatible weights when present. This should materially decrease KL toward the target without changing architecture, feature extraction, or evaluation semantics. As a final safeguard, if no compatible weights are found, it still fall back to the current random-init behavior and write a valid submission.'
- What this solution (achieved 1.41047) has done: 'Your score gap is large (1.41047 vs target 0.69096, lower-is-better), and the most likely cause is that inference is still effectively “random/untrained” because no compatible weights are actually being loaded (or the loaded checkpoints don’t contain the classifier head you use). I keep your exact dataset, preprocessing, model architecture, and inference loop, but make checkpoint discovery/load stricter and more compatible by (1) filtering to EfficientNet-B3 checkpoints that contain the final classifier weights, (2) remapping common head names (`classifier.*` / `head.*`) into your `custom_layers.2.*` so the 6-class head is actually loaded, and (3) only ensembling truly compatible checkpoints (otherwise skipping them). This should materially reduce KL toward the target without changing core logic, and still safely falls back to the current behavior if no valid checkpoints exist. The submission writing and probability normalization remain unchanged to guarantee a valid CSV.'
- What this solution (achieved 1.41047) has done: 'Your score is much worse than the target (1.41047 vs 0.69096, lower-is-better), and the most likely cause is that you’re still not actually loading any real compatible weights, so inference is effectively random. I keep your exact dataset, feature construction, model, and inference loop, but make checkpoint discovery more reliable and the load mapping more compatible by (1) prioritizing likely competition weight folders/filenames, (2) handling common key prefixes including `state_dict` keys like `model.model.*`, and (3) remapping EfficientNet classifier keys (`model.classifier.*` / `classifier.*`) into your `custom_layers.2.*` so the 6-class head is truly loaded. I also add a final “sanity filter” that only ensembles checkpoints that load a substantial portion of backbone weights and have the head loaded, which should move KL toward the target without changing evaluation semantics. Submission writing and probability normalization remain unchanged to ensure a valid CSV.'
- What this solution (achieved 1.41047) has done: 'Your current score is far above the target (1.41047 vs 0.69096, lower-is-better), so the most direct minimal improvement is to ensure you are actually using valid trained EfficientNet-B3 weights instead of effectively-random initialization. I keep your exact dataset construction, model, and inference loop, but make checkpoint discovery/loading more robust by (1) verifying checkpoint compatibility via a dry-run `load_state_dict` (so we don’t falsely skip good weights due to naming/prefix differences) and (2) expanding key remapping to cover more common timm/lightning conventions so the 6-class head loads reliably. This should reduce KL by using real learned parameters while preserving the core logic and producing the same submission schema. Submission normalization stays as-is to guarantee sum-to-1 rows and KL-safe clipping.'

# 9. Code solution

## === cell 0
import albumentations as A
import gc
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pywt
import random
import timm
import torch
import torch.nn as nn

from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, List, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")




## === cell 1
class config:
    BATCH_SIZE = 32
    MODEL = "tf_efficientnet_b3"
    NUM_WORKERS = 0  # keep 0 to avoid parquet multiprocessing overhead/timeouts
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False


class paths:
    MODEL_WEIGHTS = (
        "/kaggle/input/hba-efficientnet-weights/tf_efficientnet_b3_epoch_15.pth"
    )
    OUTPUT_DIR = "/kaggle/working/"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TEST_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    TEST_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ("model", "state_dict", "net", "weights", "ema", "model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict) and len(ckpt[k]) > 0:
                return ckpt[k]
    return ckpt


def _strip_common_prefixes(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state.items():
        nk = k
        for pref in (
            "module.",
            "model.",
            "net.",
            "encoder.",
            "backbone.",
            "student.",
            "teacher.",
            "model.model.",
            "model.module.",
            "pl_module.",
        ):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        out[nk] = v
    return out


def _remap_head_keys_for_custom_model(
    state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    remapped = dict(state)

    if "custom_layers.2.weight" in remapped and "custom_layers.2.bias" in remapped:
        return remapped

    candidates = [
        ("classifier.weight", "classifier.bias"),
        ("head.weight", "head.bias"),
        ("fc.weight", "fc.bias"),
        ("model.classifier.weight", "model.classifier.bias"),
        ("model.head.weight", "model.head.bias"),
        ("model.fc.weight", "model.fc.bias"),
        ("module.classifier.weight", "module.classifier.bias"),
        ("module.head.weight", "module.head.bias"),
        ("module.fc.weight", "module.fc.bias"),
        ("model.custom_layers.2.weight", "model.custom_layers.2.bias"),
        ("custom_layers.2.weight", "custom_layers.2.bias"),
    ]

    for wkey, bkey in candidates:
        if wkey in remapped and bkey in remapped:
            remapped["custom_layers.2.weight"] = remapped[wkey]
            remapped["custom_layers.2.bias"] = remapped[bkey]
            break

    return remapped


_tmp_model = timm.create_model(config.MODEL, pretrained=False)
_expected_backbone_keys = set(
    nn.Sequential(*list(_tmp_model.children())[:-2]).state_dict().keys()
)
del _tmp_model


def _ckpt_has_6class_head(state: Dict[str, torch.Tensor]) -> bool:
    for wkey, bkey in [
        ("custom_layers.2.weight", "custom_layers.2.bias"),
        ("classifier.weight", "classifier.bias"),
        ("head.weight", "head.bias"),
        ("fc.weight", "fc.bias"),
        ("model.classifier.weight", "model.classifier.bias"),
        ("model.head.weight", "model.head.bias"),
        ("model.fc.weight", "model.fc.bias"),
    ]:
        if wkey in state and bkey in state:
            w = state[wkey]
            b = state[bkey]
            if hasattr(w, "shape") and hasattr(b, "shape"):
                if (
                    len(w.shape) == 2
                    and w.shape[0] == 6
                    and len(b.shape) == 1
                    and b.shape[0] == 6
                ):
                    return True
    return False


def _score_ckpt_path(p: str) -> Tuple[int, int, int]:
    base = os.path.basename(p).lower()
    parent = os.path.dirname(p).lower()
    is_b3 = int("tf_efficientnet_b3" in base or "tf_efficientnet_b3" in parent)
    has_epoch15 = int("epoch_15" in base)
    has_hba = int("hba" in parent or "harmful" in parent)
    return (has_epoch15, is_b3, has_hba)


def _is_compatible_by_dryrun(path_pth: str) -> Tuple[bool, float, bool]:
    """
    Change: instead of guessing compatibility via raw key overlap only,
    do a cheap CPU dry-run load into our exact CustomModel structure (strict=False),
    then measure whether head and substantial backbone were actually loadable.
    """
    try:
        ckpt = torch.load(path_pth, map_location="cpu")
        state = _extract_state_dict(ckpt)
        if not isinstance(state, dict) or len(state) == 0:
            return (False, 0.0, False)

        state = _strip_common_prefixes(state)
        state = _remap_head_keys_for_custom_model(state)

        probe = CustomModel(config)
        missing, unexpected = probe.load_state_dict(state, strict=False)

        head_loaded = ("custom_layers.2.weight" not in missing) and (
            "custom_layers.2.bias" not in missing
        )
        loaded_keys = set(probe.state_dict().keys()) - set(missing)
        backbone_ratio = len(loaded_keys & _expected_backbone_keys) / max(
            1, len(_expected_backbone_keys)
        )

        del probe, ckpt, state
        gc.collect()
        return (head_loaded and backbone_ratio >= 0.40, backbone_ratio, head_loaded)
    except Exception:
        return (False, 0.0, False)


candidate_patterns = [
    "/kaggle/input/**/hba*/*tf_efficientnet_b3*.pth",
    "/kaggle/input/**/tf_efficientnet_b3*.pth",
    "/kaggle/input/**/*.pth",
]

found: List[str] = []
for pat in candidate_patterns:
    found.extend(glob(pat, recursive=True))
found = sorted(set(found))

preferred: List[str] = []
if os.path.exists(paths.MODEL_WEIGHTS):
    preferred = [paths.MODEL_WEIGHTS]
else:
    found_sorted = sorted(found, key=_score_ckpt_path, reverse=True)
    preferred = found_sorted[:300]

model_weights: List[str] = []
for p in preferred:
    try:
        ckpt = torch.load(p, map_location="cpu")
        state = _extract_state_dict(ckpt)
        if not isinstance(state, dict) or len(state) == 0:
            continue
        state = _strip_common_prefixes(state)
        state = _remap_head_keys_for_custom_model(state)

        if not _ckpt_has_6class_head(state):
            continue
    except Exception:
        continue

    ok, backbone_ratio, head_loaded = _is_compatible_by_dryrun(p)
    if ok:
        model_weights.append(p)
    else:
        if "tf_efficientnet_b3" in os.path.basename(p).lower():
            print(
                f"Skip ckpt after dry-run (backbone_loaded={backbone_ratio:.2%}, head_loaded={head_loaded}): {p}"
            )

model_weights = sorted(set(model_weights), key=_score_ckpt_path, reverse=True)[:5]

if len(model_weights) == 0:
    print(
        "No compatible EfficientNet-B3 weights with a 6-class head found under /kaggle/input/**. Will use random init."
    )
else:
    print(f"Found {len(model_weights)} compatible checkpoint(s) to ensemble. Example:")
    print(model_weights[:3])




## === cell 2
USE_WAVELET = None

NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


def maddest(d, axis: int = None):
    """Denoise helper."""
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x: np.ndarray, wavelet: str = "haar", level: int = 1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])
    output = pywt.waverec(coeff, wavelet, mode="per")
    return output


def spectrogram_from_eeg(parquet_path, display=False):
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle : middle + 10_000]

    img = np.zeros((128, 256, 4), dtype="float32")

    if display:
        plt.figure(figsize=(10, 7))

    for k in range(4):
        COLS = FEATS[k]

        for kk in range(4):
            x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values

            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)

            mel_spec = librosa.feature.melspectrogram(
                y=x,
                sr=200,
                hop_length=len(x) // 256,
                n_fft=1024,
                n_mels=128,
                fmin=0,
                fmax=20,
                win_length=128,
            )

            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[
                :, :width
            ]
            mel_spec_db = (mel_spec_db + 40) / 40
            img[:, :, k] += mel_spec_db

        img[:, :, k] /= 4.0

        if display:
            plt.subplot(2, 2, k + 1)
            plt.imshow(img[:, :, k], aspect="auto", origin="lower")
            plt.title(f"Spectrogram {NAMES[k]}")

    if display:
        plt.show()

    return img


def plot_spectrogram(spectrogram_path: str):
    sample_spect = pd.read_parquet(spectrogram_path)
    split_spect = {
        "LL": sample_spect.filter(regex="^LL", axis=1),
        "RL": sample_spect.filter(regex="^RL", axis=1),
        "RP": sample_spect.filter(regex="^RP", axis=1),
        "LP": sample_spect.filter(regex="^LP", axis=1),
    }

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    axes = axes.flatten()
    label_interval = 5
    for i, split_name in enumerate(split_spect.keys()):
        ax = axes[i]
        img = ax.imshow(
            np.log(split_spect[split_name]).T,
            cmap="viridis",
            aspect="auto",
            origin="lower",
        )
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label("Log(Value)")
        ax.set_title(split_name)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time")

        frequencies = [
            column_name[3:] for column_name in split_spect[split_name].columns
        ]
        ax.set_yticks(
            np.arange(0, len(split_spect[split_name].columns), label_interval)
        )
        ax.set_yticklabels(frequencies[::label_interval])
    plt.tight_layout()
    plt.show()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


seed_everything(config.SEED)




## === cell 3
test_df = pd.read_csv(paths.TEST_CSV)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()




## === cell 4
paths_spectrograms = glob(paths.TEST_SPECTROGRAMS + "*.parquet")
print(f"There are {len(paths_spectrograms)} spectrogram parquets")
all_spectrograms: Dict[int, np.ndarray] = {}

for file_path in tqdm(paths_spectrograms):
    aux = pd.read_parquet(file_path)
    name = int(os.path.basename(file_path).split(".")[0])
    all_spectrograms[name] = aux.iloc[:, 1:].values
    del aux

if config.VISUALIZE and len(paths_spectrograms) > 0:
    idx = np.random.randint(0, len(paths_spectrograms))
    spectrogram_path = paths_spectrograms[idx]
    plot_spectrogram(spectrogram_path)




## === cell 5
paths_eegs = glob(paths.TEST_EEGS + "*.parquet")
print(f"There are {len(paths_eegs)} EEG parquets")
all_eegs: Dict[int, np.ndarray] = {}

counter = 0
for file_path in tqdm(paths_eegs):
    eeg_id = int(os.path.basename(file_path).split(".")[0])
    eeg_spectrogram = spectrogram_from_eeg(file_path, counter < 1 and config.VISUALIZE)
    all_eegs[eeg_id] = eeg_spectrogram
    counter += 1

print("Loaded spectrogram dict size:", len(all_spectrograms))
print("Loaded eeg-spectrogram dict size:", len(all_eegs))




## === cell 6
class CustomModel(nn.Module):
    def __init__(self, config, num_classes: int = 6):
        super(CustomModel, self).__init__()
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True
        self.model = timm.create_model(config.MODEL, pretrained=False)
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, num_classes),
        )

    def __reshape_input(self, x):
        """
        Reshapes input (128, 256, 8) -> (3, 512, 512) like input expected by EfficientNet.
        """
        spectrograms = [x[:, :, :, i : i + 1] for i in range(4)]
        spectrograms = torch.cat(spectrograms, dim=1)

        eegs = [x[:, :, :, i : i + 1] for i in range(4, 8)]
        eegs = torch.cat(eegs, dim=1)

        if self.USE_KAGGLE_SPECTROGRAMS and self.USE_EEG_SPECTROGRAMS:
            x = torch.cat([spectrograms, eegs], dim=2)
        elif self.USE_EEG_SPECTROGRAMS:
            x = eegs
        else:
            x = spectrograms

        x = torch.cat([x, x, x], dim=3)
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, x):
        x = self.__reshape_input(x)
        x = self.features(x)
        x = self.custom_layers(x)
        return x




## === cell 7
class CustomDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config,
        augment: bool = False,
        mode: str = "train",
        specs: Dict[int, np.ndarray] = None,
        eeg_specs: Dict[int, np.ndarray] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.augment = augment
        self.mode = mode
        self.spectrograms = all_spectrograms if specs is None else specs
        self.eeg_spectrograms = all_eegs if eeg_specs is None else eeg_specs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X, y = self.__data_generation(index)
        if self.augment:
            X = self.__transform(X)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )

    def __data_generation(self, index):
        X = np.zeros((128, 256, 8), dtype="float32")
        y = np.zeros(6, dtype="float32")
        row = self.df.iloc[index]

        r = 0

        spec = self.spectrograms[int(row.spectrogram_id)]
        for region in range(4):
            img = spec[r : r + 300, region * 100 : (region + 1) * 100].T

            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            ep = 1e-6
            mu = np.nanmean(img)
            std = np.nanstd(img)
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)

            X[14:-14, :, region] = img[:, 22:-22] / 2.0

        eeg_img = self.eeg_spectrograms[int(row.eeg_id)]
        X[:, :, 4:] = eeg_img

        if self.mode != "test":
            y = row[TARGETS].values.astype(np.float32)

        return X, y

    def __transform(self, img):
        transforms = A.Compose([A.HorizontalFlip(p=0.5)])
        return transforms(image=img)["image"]




## === cell 8
test_dataset = CustomDataset(test_df, config, mode="test", augment=False)
test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True,
    drop_last=False,
)
X0, y0 = test_dataset[0]
print(f"X shape: {X0.shape}")
print(f"y shape: {y0.shape}")




## === cell 9
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for step, (X, y) in enumerate(tqdm_test_loader):
            X = X.to(device, non_blocking=True)
            with torch.no_grad():
                y_preds = model(X)
                y_preds = softmax(y_preds)
            preds.append(y_preds.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)




## === cell 10
predictions_list = []

if len(model_weights) == 0:
    print(
        "No compatible checkpoint weights found. Running with random initialized model (valid submission, poor score expected)."
    )
    model = CustomModel(config).to(device)
    preds = inference_function(test_loader, model, device)
    predictions_list.append(preds)
else:
    for model_weight in model_weights:
        model = CustomModel(config)

        checkpoint = torch.load(model_weight, map_location="cpu")
        state = _extract_state_dict(checkpoint)
        if not isinstance(state, dict):
            print(f"Skip non-dict state in: {model_weight}")
            del model, checkpoint
            gc.collect()
            continue

        state = _strip_common_prefixes(state)
        state = _remap_head_keys_for_custom_model(state)

        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 0 or len(unexpected) > 0:
            print(
                f"[{os.path.basename(model_weight)}] load_state_dict(strict=False): missing={len(missing)}, unexpected={len(unexpected)}"
            )

        head_loaded = ("custom_layers.2.weight" not in missing) and (
            "custom_layers.2.bias" not in missing
        )
        loaded_keys = set(model.state_dict().keys()) - set(missing)
        backbone_loaded = len(loaded_keys & _expected_backbone_keys) / max(
            1, len(_expected_backbone_keys)
        )

        if (not head_loaded) or (backbone_loaded < 0.40):
            print(
                f"Skip checkpoint (head_loaded={head_loaded}, backbone_loaded={backbone_loaded:.2%}): {model_weight}"
            )
            del model, checkpoint, state
            torch.cuda.empty_cache()
            gc.collect()
            continue

        model.to(device)
        preds = inference_function(test_loader, model, device)
        predictions_list.append(preds)

        del model, checkpoint, state
        torch.cuda.empty_cache()
        gc.collect()

if len(predictions_list) == 0:
    print(
        "All checkpoints were skipped during loading; falling back to random-init model."
    )
    model = CustomModel(config).to(device)
    preds = inference_function(test_loader, model, device)
    predictions_list.append(preds)

predictions = np.mean(np.stack(predictions_list, axis=0), axis=0)
print("Raw predictions shape:", predictions.shape)




## === cell 11
if predictions.ndim != 2 or predictions.shape[1] != 6:
    raise ValueError(
        f"Predictions must be (n_samples, 6). Got shape={predictions.shape}"
    )

row_sums = predictions.sum(axis=1, keepdims=True)
row_sums = np.where(row_sums == 0, 1.0, row_sums)
predictions = predictions / row_sums
predictions = np.clip(predictions, 1e-12, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions
out_path = os.path.join(paths.OUTPUT_DIR, "submission.csv")
sub.to_csv(out_path, index=False)
print(f"Saved submission to: {out_path}")
print(f"Submission shape: {sub.shape}")
print(
    "Row prob sum stats:",
    sub[TARGETS].sum(axis=1).min(),
    sub[TARGETS].sum(axis=1).max(),
)
sub.head()
