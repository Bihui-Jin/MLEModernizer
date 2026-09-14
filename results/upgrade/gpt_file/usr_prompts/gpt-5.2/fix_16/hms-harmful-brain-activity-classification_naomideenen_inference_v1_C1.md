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

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
numpy==1.26.4
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

0.4814297300688451

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the merge runtime error by removing the overly-strict `validate="one_to_one"` constraint and instead enforcing uniqueness explicitly on both sides (dropping duplicates in `sample_sub` and aggregating predictions by `eeg_id`). I also add a safety check to ensure we always output exactly the sample submission’s row order and that each row sums to 1, so the submission is valid. These changes don’t alter the model/ensemble logic, only the final alignment step to reliably write `submission.csv`. The pipeline then run end-to-end and produce a valid CSV submission file.'
- What this solution (achieved 1.40995) has done: 'Your current score is far worse than the target (lower-is-better), so the most likely issue is that the predictions are effectively close to uniform or otherwise mis-calibrated because the spectrogram preprocessing is not consistent with how these EfficientNet checkpoints were trained. I make the smallest changes that keep the same ensemble/models and averaging logic, but fix two high-impact inference semantics: (1) use the correct spectrogram magnitude channels (exclude the time column only if present, not by blindly dropping the first column), and (2) apply the standard ImageNet normalization that timm EfficientNet models expect unless the checkpoint was trained without it. I also add a very light “probability sharpening/softening” temperature on the *final* averaged probabilities (not changing model/loss), defaulting to 1.0 but set to a conservative value that typically improves KL for this competition by reducing over-uniform predictions. Submission writing/alignment remains identical and still guarantees row sums to 1.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.4814), so the highest-impact minimal fix is to ensure the inference preprocessing matches what the EfficientNet spectrogram checkpoints most likely expect. I keep the exact same ensemble/models and averaging logic, but (1) align spectrogram orientation to a consistent (freq, time) convention without relying on a fragile transpose, (2) guarantee a stable float32 pipeline through preprocessing/normalization, and (3) make the post-ensemble temperature optional and default to neutral (1.0) so we don’t accidentally worsen KL via miscalibration. These are small semantic fixes that commonly move HMS spectrogram EfficientNet submissions from “nearly uniform/wrong scaling” toward a much better KL, while preserving your overall approach and submission alignment guarantees.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far worse than the target (0.4814), so the most likely “minimal but high-impact” fix is that the spectrogram tensor being fed to the EfficientNet checkpoints is oriented incorrectly (time/frequency swapped) and/or scaled differently than the training pipeline. I keep the same ensemble members and averaging logic, but make the preprocessing more robust by auto-detecting whether the parquet is (time×freq) or (freq×time) and only transposing when needed, plus ensure we always drop non-feature columns safely. I also make timm normalization conditional (default ON, but easy to revert) and add a very small epsilon/renorm safety exactly as before so the submission remains valid. These changes preserve your model/loss/ensemble core logic, but should move predictions away from near-uniform “wrong-shape” behavior and reduce KL toward the target.'
- What this solution (achieved 1.40995) has done: 'Your KL is much worse than the target, so the most likely issue is a mismatch between how the parquet spectrogram is being turned into the (1,H,W) image and how the EfficientNet checkpoints were trained. I keep the same ensemble/models and averaging, but fix two inference semantics that often cause near-uniform/wrong predictions: (1) robustly extract only the true spectrogram feature matrix (dropping any index-like columns such as `time`/`Unnamed: 0` and ensuring numeric-only), and (2) resize while preserving the expected “frequency-as-height” orientation by choosing transpose based on typical spectrogram dimensions rather than `h<w`. Finally, I keep your probability renorm safety but also apply a conservative *logit-temperature* on the ensemble logits (equivalent to calibrating confidence) instead of exponentiating probabilities, which usually improves KL without changing the model core.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.4814), so the most likely issue is a preprocessing mismatch that makes the ensemble output close to uninformative. I keep the same models/ensemble/softmax logic, but fix two high-impact inference semantics: (1) robustly determine spectrogram orientation using column-name cues (Hz vs region) and only transpose when needed, and (2) use timm’s built-in per-model `data_config` normalization for EfficientNet (instead of a single fixed mean/std), while still respecting your existing 1‑channel setup. I also avoid recomputing the normalized image inside each model loop (same result, faster) and keep the exact same submission alignment and “row sums to 1” guarantees.'
- What this solution (achieved 1.40995) has done: 'Your KL is far above the target (lower is better), so the most likely cause is a preprocessing mismatch making the ensemble output close to uninformative. I keep the same models/ensemble averaging/softmax, but fix two minimal high-impact inference semantics: (1) use timm’s per-model input normalization consistently across *all* models by first mapping the spectrogram image into the expected 0–1 range, then applying the model’s mean/std; and (2) make the spectrogram orientation decision more robust by preferring “frequency-as-height” using typical HMS spectrogram shapes rather than a fragile `h>w` heuristic. I also ensure each model uses its own mean/std (your current code accidentally reuses the first model’s norm for whole groups), which can materially affect logits while preserving architecture and evaluation semantics. The submission writing and probability renormalization remain unchanged and still produce a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your KL is much worse than the target (lower-is-better), so the most likely minimal high-impact issue is that the input normalization is mismatched: applying `sigmoid()` to already log-scaled/standardized spectrogram values collapses dynamic range and makes predictions close to uninformative. I keep the same models, ensemble averaging, and softmax, but replace the `sigmoid`-based “0–1 mapping” with a per-image min–max scaling to [0,1] (a common spectrogram-to-image convention) before applying timm mean/std. I also keep your orientation/column dropping logic, but add a tiny safety to ensure the min–max scaling never divides by ~0. Submission alignment and row-sum-to-1 guarantees remain unchanged.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995; lower is better) is far from the target (0.4814), so we should make a minimal but high-impact inference fix rather than change the ensemble or models. The most likely issue is a mismatch between how the spectrogram parquet values are converted into an “image” and what the EfficientNet checkpoints expect; your pipeline currently log-scales then standardizes, then min–maxes after resize, which can wash out amplitude cues. I keep the same models, same ensemble averaging, same softmax, and same submission alignment, but change only the 0–1 mapping step to be done on the raw log-spectrogram (before any z-score) and then apply either dataset-wide z-score or timm normalization consistently. This preserves core logic while making the input distribution much closer to typical spectrogram-image training, which should reduce KL substantially toward the target.'
- What this solution (achieved 1.40995) has done: 'Your KL is much worse than the target (lower-is-better), so the smallest high-impact improvement is to fix a likely normalization bug that can make the EfficientNet inputs almost constant/uninformative. Right now you z-score the min–maxed spectrogram and then clamp it to [0,1] before “timm norm”, which destroys most signal (nearly everything clamps to 0 or 1). I keep your exact models/ensemble/softmax and just change normalization to: log-spectrogram → per-image min–max to [0,1] → (optional) dataset-wide z-score (no clamping) for the “datawide” models, and skip applying ImageNet mean/std to 1‑channel non-natural images (set `apply_timm_norm=False`) to avoid further distribution mismatch. Submission alignment/row-sum-to-1 guarantees remain identical.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995; lower is better) is far from the target (0.4814), so we need a minimal but high-impact inference semantics fix rather than touching the ensemble/models. The biggest likely issue is that you min–max to [0,1] and then immediately apply a dataset-wide z-score computed on *log-spectrogram space*, which makes inputs badly mismatched (and can push most values into a narrow band), yielding near-uninformative predictions. I keep the exact same models, averaging, and softmax, but change the “datawide” normalizations to use the log-spectrogram directly for z-scoring (no min–max), while leaving the instance-wise branch as-is. I also set `logit_temperature` back to neutral (1.0) to avoid adding miscalibration on top of already-mismatched preprocessing, which should move KL materially toward the target without altering core logic.'

# 9. Code solution

## === cell 0
import os
import gc
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=Warning)
gc.collect()

DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification"




## === cell 1
class Config:
    seed = 3131
    image_size = (512, 512)  # (H, W)
    num_folds = 5

    dataset_wide_mean = -0.2972692229201065  # From Train notebook
    dataset_wide_std = 2.5997336315611026  # From Train notebook

    apply_timm_norm = False

    logit_temperature = 1.0


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(Config.seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)


## === cell 2
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

test_df = test_df[["eeg_id", "spectrogram_id"]].copy()
test_df["path"] = test_df["spectrogram_id"].apply(
    lambda x: f"{DATA_DIR}/test_spectrograms/{x}.parquet"
)

display(test_df.head())
print("test_df rows:", len(test_df), "sample_submission rows:", len(sample_sub))
gc.collect()




## === cell 3
def try_load_state_dict(model: nn.Module, ckpt_path: str) -> bool:
    if not os.path.exists(ckpt_path):
        return False
    state = torch.load(ckpt_path, map_location=torch.device("cpu"))
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(state, strict=False)
    return True


def find_ckpt_fallback(expected_rel: str) -> str | None:
    p = Path(expected_rel)
    if p.exists():
        return str(p)
    fname = p.name
    hits = list(Path("/kaggle/input").rglob(fname))
    return str(hits[0]) if len(hits) > 0 else None


models = []
for i in range(Config.num_folds):
    model_effnet_b0 = timm.create_model(
        "efficientnet_b0", pretrained=False, num_classes=6, in_chans=1
    )
    ckpt_expected = (
        f"/kaggle/input/hms-train-efficientnetb0/efficientnet_b0_fold{i}.pth"
    )
    ckpt = find_ckpt_fallback(ckpt_expected)
    if ckpt and try_load_state_dict(model_effnet_b0, ckpt):
        models.append(model_effnet_b0)

models_datawide1 = []
for i in range(Config.num_folds):
    model_effnet_b1 = timm.create_model(
        "efficientnet_b1", pretrained=False, num_classes=6, in_chans=1
    )
    ckpt_expected = f"/kaggle/input/train/efficientnet_b1_fold{i}.pth"
    ckpt = find_ckpt_fallback(ckpt_expected)
    if ckpt and try_load_state_dict(model_effnet_b1, ckpt):
        models_datawide1.append(model_effnet_b1)

models_datawide2 = []
for i in range(Config.num_folds):
    model_effnet_b1 = timm.create_model(
        "efficientnet_b1", pretrained=False, num_classes=6, in_chans=1
    )
    ckpt_expected = f"/kaggle/input/efficientnet-b0-naomi/efficientnet_b1_fold{i}_datawide_ReduceLROnPlateau_0.001_False.pth"
    ckpt = find_ckpt_fallback(ckpt_expected)
    if ckpt and try_load_state_dict(model_effnet_b1, ckpt):
        models_datawide2.append(model_effnet_b1)

    model_effnet_b1b = timm.create_model(
        "efficientnet_b1", pretrained=False, num_classes=6, in_chans=1
    )
    ckpt_expected = f"/kaggle/input/efficientnet-b0-naomi/efficientnet_b1_fold{i}_datawide_CosineAnnealingLR_0.001_False.pth"
    ckpt = find_ckpt_fallback(ckpt_expected)
    if ckpt and try_load_state_dict(model_effnet_b1b, ckpt):
        models_datawide2.append(model_effnet_b1b)

for m in models + models_datawide1 + models_datawide2:
    m.to(DEVICE)
    m.eval()

print(
    f"Loaded models: instance-wise={len(models)}, datawide1={len(models_datawide1)}, datawide2={len(models_datawide2)}"
)
gc.collect()


## === cell 4
norm_csv_expected = "/kaggle/input/efficientnet-b0-naomi/normalization.csv"
norm_csv = find_ckpt_fallback(norm_csv_expected) or norm_csv_expected
if os.path.exists(norm_csv):
    df_normalization = pd.read_csv(norm_csv)
    data_mean2 = float(df_normalization["mean"].values[0])
    data_std2 = float(df_normalization["std"].values[0])
else:
    data_mean2 = float(Config.dataset_wide_mean)
    data_std2 = float(Config.dataset_wide_std)

print("data_mean2, data_std2:", data_mean2, data_std2)


## === cell 5
paths_to_parquets = test_df["path"].values
test_predictions = []


def _infer_orientation_from_columns(df: pd.DataFrame) -> str:
    cols = [str(c).strip() for c in df.columns]
    cols = [
        c
        for c in cols
        if c.lower() not in ("time", "t") and not c.lower().startswith("unnamed")
    ]
    if len(cols) == 0:
        return "unknown"

    freq_like = 0
    for c in cols[:200]:
        try:
            float(c)
            freq_like += 1
            continue
        except Exception:
            pass
        if "_" in c:
            left = c.split("_", 1)[0]
            try:
                float(left)
                freq_like += 1
                continue
            except Exception:
                pass

    if freq_like / max(1, min(len(cols), 200)) > 0.2:
        return "time_by_freq"
    return "freq_by_time_or_other"


def preprocess(path_to_parquet: str) -> np.ndarray:
    df = pd.read_parquet(path_to_parquet)

    drop_cols = []
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ("time", "t") or cl.startswith("unnamed"):
            drop_cols.append(c)
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df.select_dtypes(include=[np.number])
    arr = df.to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=-1.0, posinf=-1.0, neginf=-1.0).astype(np.float32)

    orientation = _infer_orientation_from_columns(df)
    if orientation == "time_by_freq":
        data = arr.T  # (freq, time)
    else:
        h, w = arr.shape
        if w >= 300 and h <= 600:
            data = arr.T
        elif h >= 300 and w <= 600:
            data = arr
        else:
            data = arr.T if w > h else arr

    data = np.clip(data, np.exp(-6), np.exp(10)).astype(np.float32)
    data = np.log(data).astype(np.float32)
    return data


def _to_image_tensor(data_point: np.ndarray) -> torch.Tensor:
    x = torch.tensor(data_point, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=Config.image_size, mode="bilinear", align_corners=False)
    return x[0]  # (1,H,W)


def _get_timm_norm_for_model(model: nn.Module) -> tuple[float, float]:
    cfg = timm.data.resolve_model_data_config(model)
    mean = cfg.get("mean", (0.0,))
    std = cfg.get("std", (1.0,))
    mean0 = float(mean[0] if isinstance(mean, (list, tuple)) else mean)
    std0 = float(std[0] if isinstance(std, (list, tuple)) else std)
    return mean0, std0


def _minmax01_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    return (x - xmin) / (xmax - xmin + 1e-6)


def _apply_timm_norm_assuming_01(
    x_img01: torch.Tensor, mean0: float, std0: float
) -> torch.Tensor:
    if not Config.apply_timm_norm:
        return x_img01
    mean = torch.tensor([mean0], device=x_img01.device, dtype=x_img01.dtype).view(
        1, 1, 1
    )
    std = torch.tensor([std0], device=x_img01.device, dtype=x_img01.dtype).view(1, 1, 1)
    return (x_img01 - mean) / (std + 1e-6)


def normalize_datawide1(
    data_point: np.ndarray, mean0: float, std0: float
) -> torch.Tensor:
    xz = (data_point - float(Config.dataset_wide_mean)) / (
        float(Config.dataset_wide_std) + 1e-6
    )
    x_img = _to_image_tensor(xz)
    return _apply_timm_norm_assuming_01(x_img, mean0, std0)


def normalize_datawide2(
    data_point: np.ndarray, data_mean: float, data_std: float, mean0: float, std0: float
) -> torch.Tensor:
    xz = (data_point - float(data_mean)) / (float(data_std) + 1e-6)
    x_img = _to_image_tensor(xz)
    return _apply_timm_norm_assuming_01(x_img, mean0, std0)


def normalize_instance_wise(
    data_point: np.ndarray, mean0: float, std0: float
) -> torch.Tensor:
    x01 = _minmax01_np(data_point)
    dm = float(x01.mean())
    ds = float(x01.std())
    xz = (x01 - dm) / (ds + 1e-6)
    x_img = _to_image_tensor(xz)
    return _apply_timm_norm_assuming_01(x_img, mean0, std0)


def _predict_logits_one(model: nn.Module, x_img: torch.Tensor) -> np.ndarray:
    x = x_img.unsqueeze(0).to(DEVICE, non_blocking=True)  # (1,1,H,W)
    with torch.no_grad():
        logits = model(x)[0].detach().cpu().numpy().astype(np.float32)
    return logits


total_models = len(models) + len(models_datawide1) + len(models_datawide2)
print("Total ensemble members:", total_models)

model_norms = []
for m in models + models_datawide1 + models_datawide2:
    model_norms.append(_get_timm_norm_for_model(m))

Tlog = float(Config.logit_temperature)
for path in paths_to_parquets:
    if total_models == 0:
        test_predictions.append(np.ones(6, dtype=np.float32) / 6.0)
        continue

    preprocessed_data = preprocess(path)
    logits_list = []

    if len(models) > 0:
        for j, m in enumerate(models):
            mean0, std0 = model_norms[j]
            img_inst = normalize_instance_wise(preprocessed_data, mean0, std0)
            logits_list.append(_predict_logits_one(m, img_inst))

    offset = len(models)
    if len(models_datawide1) > 0:
        for j, m in enumerate(models_datawide1):
            mean0, std0 = model_norms[offset + j]
            img_dw1 = normalize_datawide1(preprocessed_data, mean0, std0)
            logits_list.append(_predict_logits_one(m, img_dw1))
    offset += len(models_datawide1)

    if len(models_datawide2) > 0:
        for j, m in enumerate(models_datawide2):
            mean0, std0 = model_norms[offset + j]
            img_dw2 = normalize_datawide2(
                preprocessed_data, data_mean2, data_std2, mean0, std0
            )
            logits_list.append(_predict_logits_one(m, img_dw2))

    ensemble_logits = np.mean(np.stack(logits_list, axis=0), axis=0)

    if Tlog != 1.0:
        ensemble_logits = ensemble_logits / Tlog

    z = ensemble_logits - np.max(ensemble_logits)
    expz = np.exp(z)
    probs = expz / np.sum(expz)
    test_predictions.append(probs.astype(np.float32))

test_predictions = np.asarray(test_predictions, dtype=np.float32)
test_predictions = np.clip(test_predictions, 1e-8, None)
test_predictions = test_predictions / test_predictions.sum(axis=1, keepdims=True)

print("Predictions shape:", test_predictions.shape, "Expected:", len(test_df))
gc.collect()


## === cell 6
labels = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
vote_cols = [f"{l}_vote" for l in labels]

pred_df = pd.DataFrame(test_predictions, columns=vote_cols)
pred_df["eeg_id"] = test_df["eeg_id"].values

pred_df = pred_df.groupby("eeg_id", as_index=False)[vote_cols].mean()

sample_sub_unique = sample_sub.drop_duplicates(subset=["eeg_id"], keep="first").copy()

submission_out = sample_sub_unique[["eeg_id"]].merge(pred_df, on="eeg_id", how="left")

missing_mask = submission_out[vote_cols].isna().any(axis=1)
if missing_mask.any():
    submission_out.loc[missing_mask, vote_cols] = 1.0 / 6.0

submission_out[vote_cols] = submission_out[vote_cols].astype(np.float64).clip(1e-8, 1.0)
submission_out[vote_cols] = submission_out[vote_cols].div(
    submission_out[vote_cols].sum(axis=1), axis=0
)

submission_out = submission_out[["eeg_id"] + vote_cols]

if len(sample_sub) != len(sample_sub_unique):
    submission_out = sample_sub[["eeg_id"]].merge(
        submission_out, on="eeg_id", how="left"
    )
    missing_mask = submission_out[vote_cols].isna().any(axis=1)
    if missing_mask.any():
        submission_out.loc[missing_mask, vote_cols] = 1.0 / 6.0
    submission_out[vote_cols] = (
        submission_out[vote_cols].astype(np.float64).clip(1e-8, 1.0)
    )
    submission_out[vote_cols] = submission_out[vote_cols].div(
        submission_out[vote_cols].sum(axis=1), axis=0
    )
    submission_out = submission_out[["eeg_id"] + vote_cols]

assert list(submission_out.columns) == ["eeg_id"] + vote_cols
assert len(submission_out) == len(sample_sub)
row_sums = submission_out[vote_cols].sum(axis=1).values
assert np.all(np.isfinite(row_sums))
assert np.max(np.abs(row_sums - 1.0)) < 1e-6

submission_out.to_csv("submission.csv", index=False)
display(submission_out.head())
print("Wrote:", os.path.abspath("submission.csv"), "rows:", len(submission_out))
gc.collect()
