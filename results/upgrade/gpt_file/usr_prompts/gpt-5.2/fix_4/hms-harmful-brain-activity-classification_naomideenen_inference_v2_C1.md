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

0.4827280437240149

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The merge in cell 5 fails because `eeg_id` is not unique in `sample_submission.csv` and/or `test.csv` under this environment, so I stop merging and instead build the spectrogram parquet `path` by mapping `eeg_id -> spectrogram_id` from `test.csv` while preserving the sample submission row order. That fixes the missing `path` column and unblocks inference so `test_predictions` is defined. I also make the parquet preprocessing robust to unexpected column layouts (e.g., no leading “time” column) to prevent shape/index errors without changing the model logic. Finally, I ensure the output probabilities are clipped and row-normalized and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.4827), so we need a small but meaningful improvement without changing the model or training. The biggest likely issue is that the inference is running in mixed precision or otherwise non-deterministically and, more importantly, may be producing overly confident probabilities that are heavily penalized by KL; adding a tiny, legitimate post-processing “label-smoothing” blend with uniform usually reduce KL on this competition. I keep the exact model ensemble and preprocessing, but (1) ensure consistent float32 inference with `torch.cuda.amp.autocast(False)` and (2) apply a small uniform-mixture calibration (e.g., 0.02) after ensembling and renormalize, which commonly improves KL without altering the core logic. I also add a safe fallback: if no weights are found, output the sample_submission’s uniform probabilities directly so you never submit garbage.'

# 9. Code solution

## === cell 0
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"



## === cell 1
import gc
import os
import random
import warnings
import numpy as np
import pandas as pd
from IPython.display import display

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

warnings.filterwarnings("ignore", category=Warning)
gc.collect()



## === cell 2
from pathlib import Path

DATA_ROOT = Path("/kaggle/input")


def find_first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return str(p)
    return None


def rglob_paths(root, pattern):
    root = Path(root)
    if not root.exists():
        return []
    return sorted([str(p) for p in root.rglob(pattern)])


def safe_load_state_dict(model, weights_path, map_location="cpu"):
    """Load weights safely. Returns True if loaded, False otherwise."""
    try:
        state = torch.load(weights_path, map_location=torch.device(map_location))
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]
            stripped = {}
            for k, v in state.items():
                nk = k
                for pref in ("model.", "module."):
                    if nk.startswith(pref):
                        nk = nk[len(pref) :]
                stripped[nk] = v
            state = stripped
        model.load_state_dict(state, strict=True)
        return True
    except Exception:
        return False




## === cell 3
class Config:
    seed = 3131
    image_transform = transforms.Resize((512, 512))
    num_folds = 5
    dataset_wide_mean = -0.2972692229201065  # From Train notebook
    dataset_wide_std = 2.5997336315611026  # From Train notebook

    uniform_mix = 0.02  # 0.00 keeps original behavior; small >0 often improves KL


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(Config.seed)



## === cell 4
COMP_DIR = "/kaggle/input/hms-harmful-brain-activity-classification"
TEST_CSV = f"{COMP_DIR}/test.csv"
SAMPLE_SUB = f"{COMP_DIR}/sample_submission.csv"
TEST_SPEC_DIR = f"{COMP_DIR}/test_spectrograms"

assert os.path.exists(TEST_CSV), f"Missing {TEST_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.exists(TEST_SPEC_DIR), f"Missing {TEST_SPEC_DIR}"



## === cell 5
test_df = pd.read_csv(TEST_CSV)
submission = pd.read_csv(SAMPLE_SUB)

eeg_to_spec = test_df.drop_duplicates("eeg_id").set_index("eeg_id")["spectrogram_id"]
submission["spectrogram_id"] = submission["eeg_id"].map(eeg_to_spec)

submission["path"] = submission["spectrogram_id"].apply(
    lambda x: f"{TEST_SPEC_DIR}/{int(x)}.parquet" if pd.notna(x) else None
)

display(submission.head())
gc.collect()




## === cell 6
def load_fold_models(model_name, num_classes, in_chans, weight_globs):
    loaded = []
    for fold in range(Config.num_folds):
        candidates = []
        for g in weight_globs:
            g_fold = g.format(fold=fold)
            candidates.extend(rglob_paths(DATA_ROOT, g_fold))
        candidates = sorted(set(candidates))
        if not candidates:
            continue

        m = timm.create_model(
            model_name, pretrained=False, num_classes=num_classes, in_chans=in_chans
        )
        ok = False
        for w in candidates:
            if safe_load_state_dict(m, w, map_location="cpu"):
                ok = True
                break
        if ok:
            loaded.append(m)
    return loaded


models = load_fold_models(
    model_name="efficientnet_b0",
    num_classes=6,
    in_chans=1,
    weight_globs=[
        "**/efficientnet_b0_fold{fold}.pth",
        "**/efficientnet_b0_fold{fold}*.pth",
    ],
)

models_datawide1 = load_fold_models(
    model_name="efficientnet_b1",
    num_classes=6,
    in_chans=1,
    weight_globs=[
        "**/efficientnet_b1_fold{fold}.pth",
        "**/efficientnet_b1_fold{fold}*.pth",
    ],
)

models_datawide2 = load_fold_models(
    model_name="efficientnet_b1",
    num_classes=6,
    in_chans=1,
    weight_globs=[
        "**/efficientnet_b1_fold{fold}_datawide_*.pth",
        "**/efficientnet_b1_fold{fold}*datawide*.pth",
    ],
)

print(
    f"Loaded models: instance_wise={len(models)}, datawide1={len(models_datawide1)}, datawide2={len(models_datawide2)}"
)
gc.collect()



## === cell 7
norm_csv = find_first_existing(
    [
        "/kaggle/input/efficientnet-b0-naomi/normalization.csv",
    ]
)

if norm_csv is not None:
    df_normalization = pd.read_csv(norm_csv)
    data_mean2 = float(df_normalization["mean"].values[0])
    data_std2 = float(df_normalization["std"].values[0])
else:
    data_mean2 = float(Config.dataset_wide_mean)
    data_std2 = float(Config.dataset_wide_std)

print("data_mean2, data_std2:", data_mean2, data_std2)



## === cell 8
ensemble_size = 3  # kept for compatibility (not directly used)

paths_to_parquets = submission["path"].values
test_predictions = []


def preprocess(path_to_parquet):
    df = pd.read_parquet(path_to_parquet)
    df = df.fillna(-1)

    vals = df.to_numpy()
    if vals.shape[1] >= 2:
        col0 = vals[:, 0]
        if np.issubdtype(col0.dtype, np.number):
            diffs = np.diff(col0)
            if np.all(np.isfinite(diffs)) and np.mean(diffs >= 0) > 0.99:
                vals = vals[:, 1:]

    data = vals.T  # [H, W]
    data = np.clip(data, np.exp(-6), np.exp(10))
    data = np.log(data)
    return data


def normalize_datawide1(data_point):
    eps = 1e-6
    data_point = (data_point - Config.dataset_wide_mean) / (
        Config.dataset_wide_std + eps
    )
    data_tensor = torch.unsqueeze(
        torch.tensor(data_point, dtype=torch.float32), dim=0
    )  # [1,H,W]
    data_point = Config.image_transform(data_tensor)  # [1,512,512]
    return data_point


def normalize_datawide2(data, data_mean, data_std):
    eps = 1e-6
    data = (data - data_mean) / (data_std + eps)
    data_tensor = torch.unsqueeze(
        torch.tensor(data, dtype=torch.float32), dim=0
    )  # [1,H,W]
    data = Config.image_transform(data_tensor)
    return data


def normalize_instance_wise(data_point):
    eps = 1e-6
    data_mean = data_point.mean(axis=(0, 1))
    data_std = data_point.std(axis=(0, 1))
    data_point = (data_point - data_mean) / (data_std + eps)
    data_tensor = torch.unsqueeze(
        torch.tensor(data_point, dtype=torch.float32), dim=0
    )  # [1,H,W]
    data_point = Config.image_transform(data_tensor)
    return data_point


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for m in models + models_datawide1 + models_datawide2:
    m.to(device)
    m.eval()

uniform_pred = np.ones(6, dtype=np.float32) / 6.0

if (len(models) + len(models_datawide1) + len(models_datawide2)) == 0:
    test_predictions = np.tile(
        uniform_pred[None, :], (len(paths_to_parquets), 1)
    ).astype(np.float32)
else:
    for path in paths_to_parquets:
        if (path is None) or (not os.path.exists(path)):
            test_predictions.append(uniform_pred.copy())
            continue

        preprocessed_data = preprocess(path)
        test_predictions_per_model = []

        for m in models:
            x = (
                normalize_instance_wise(preprocessed_data).unsqueeze(0).to(device)
            )  # [1,1,512,512]
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                out = m(x.float())
                pred = F.softmax(out, dim=1)[0].detach().cpu().numpy()
            test_predictions_per_model.append(pred)

        for m in models_datawide1:
            x = normalize_datawide1(preprocessed_data).unsqueeze(0).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                out = m(x.float())
                pred = F.softmax(out, dim=1)[0].detach().cpu().numpy()
            test_predictions_per_model.append(pred)

        for m in models_datawide2:
            x = (
                normalize_datawide2(preprocessed_data, data_mean2, data_std2)
                .unsqueeze(0)
                .to(device)
            )
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                out = m(x.float())
                pred = F.softmax(out, dim=1)[0].detach().cpu().numpy()
            test_predictions_per_model.append(pred)

        if len(test_predictions_per_model) == 0:
            ensemble_prediction = uniform_pred.copy()
        else:
            ensemble_prediction = np.mean(test_predictions_per_model, axis=0)

        if Config.uniform_mix > 0:
            ensemble_prediction = (1.0 - Config.uniform_mix) * ensemble_prediction + (
                Config.uniform_mix * uniform_pred
            )

        ensemble_prediction = np.clip(ensemble_prediction, 1e-9, 1.0)
        ensemble_prediction = ensemble_prediction / ensemble_prediction.sum()

        test_predictions.append(ensemble_prediction)

    test_predictions = np.asarray(test_predictions, dtype=np.float32)

print("test_predictions shape:", test_predictions.shape)
gc.collect()



## === cell 9
submission_out = pd.read_csv(SAMPLE_SUB)
labels = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]

assert (
    test_predictions.shape[0] == submission_out.shape[0]
), "Prediction rows != submission rows"
assert test_predictions.shape[1] == 6, "Predictions must have 6 columns"

for i, lab in enumerate(labels):
    submission_out[f"{lab}_vote"] = test_predictions[:, i]

probs = submission_out[[f"{l}_vote" for l in labels]].to_numpy(dtype=np.float64)
probs = np.clip(probs, 1e-15, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)
submission_out[[f"{l}_vote" for l in labels]] = probs

submission_out.to_csv("submission.csv", index=False)
display(submission_out.head())
print("Wrote submission.csv with shape:", submission_out.shape)
gc.collect()
