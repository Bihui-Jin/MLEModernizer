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

    timm_mean = 0.485
    timm_std = 0.229

    apply_timm_norm = True

    prob_temperature = 1.0


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


def preprocess(path_to_parquet: str) -> np.ndarray:
    df = pd.read_parquet(path_to_parquet)

    drop_cols = []
    for c in df.columns:
        if c.lower() in ("time",):
            drop_cols.append(c)
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df.select_dtypes(include=[np.number])

    arr = df.to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=-1.0, posinf=-1.0, neginf=-1.0).astype(np.float32)

    h, w = arr.shape
    if h < w:
        data = arr.T
    else:
        data = arr

    data = np.clip(data, np.exp(-6), np.exp(10)).astype(np.float32)
    data = np.log(data).astype(np.float32)
    return data


def _to_image_tensor(data_point: np.ndarray) -> torch.Tensor:
    x = torch.tensor(data_point, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=Config.image_size, mode="bilinear", align_corners=False)
    return x[0]  # (1,H,W)


def _apply_timm_norm(x_img: torch.Tensor) -> torch.Tensor:
    if not Config.apply_timm_norm:
        return x_img
    mean = torch.tensor(
        [Config.timm_mean], device=x_img.device, dtype=x_img.dtype
    ).view(1, 1, 1)
    std = torch.tensor([Config.timm_std], device=x_img.device, dtype=x_img.dtype).view(
        1, 1, 1
    )
    return (x_img - mean) / (std + 1e-6)


def normalize_datawide1(data_point: np.ndarray) -> torch.Tensor:
    eps = 1e-6
    data_point = (data_point - Config.dataset_wide_mean) / (
        Config.dataset_wide_std + eps
    )
    x_img = _to_image_tensor(data_point)
    return _apply_timm_norm(x_img)


def normalize_datawide2(
    data_point: np.ndarray, data_mean: float, data_std: float
) -> torch.Tensor:
    eps = 1e-6
    data_point = (data_point - data_mean) / (data_std + eps)
    x_img = _to_image_tensor(data_point)
    return _apply_timm_norm(x_img)


def normalize_instance_wise(data_point: np.ndarray) -> torch.Tensor:
    eps = 1e-6
    data_mean = float(data_point.mean())
    data_std = float(data_point.std())
    data_point = (data_point - data_mean) / (data_std + eps)
    x_img = _to_image_tensor(data_point)
    return _apply_timm_norm(x_img)


def _predict_one(model: nn.Module, x_img: torch.Tensor) -> np.ndarray:
    x = x_img.unsqueeze(0).to(DEVICE, non_blocking=True)  # (1,1,H,W)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
    return probs


total_models = len(models) + len(models_datawide1) + len(models_datawide2)
print("Total ensemble members:", total_models)

for path in paths_to_parquets:
    if total_models == 0:
        test_predictions.append(np.ones(6, dtype=np.float32) / 6.0)
        continue

    test_predictions_per_model = []
    preprocessed_data = preprocess(path)

    for m in models:
        img = normalize_instance_wise(preprocessed_data)
        test_predictions_per_model.append(_predict_one(m, img))

    for m in models_datawide1:
        img = normalize_datawide1(preprocessed_data)
        test_predictions_per_model.append(_predict_one(m, img))

    for m in models_datawide2:
        img = normalize_datawide2(preprocessed_data, data_mean2, data_std2)
        test_predictions_per_model.append(_predict_one(m, img))

    ensemble_prediction = np.mean(np.stack(test_predictions_per_model, axis=0), axis=0)
    test_predictions.append(ensemble_prediction)

test_predictions = np.asarray(test_predictions, dtype=np.float32)

test_predictions = np.clip(test_predictions, 1e-8, None)
test_predictions = test_predictions / test_predictions.sum(axis=1, keepdims=True)

T = float(Config.prob_temperature)
if T != 1.0:
    p = np.clip(test_predictions, 1e-8, 1.0)
    p = p ** (1.0 / T)
    test_predictions = p / p.sum(axis=1, keepdims=True)

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
