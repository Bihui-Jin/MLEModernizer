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
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.4827), so the smallest high-impact adjustment is to reduce overconfident predictions, which KL punishes heavily. I keep the exact same ensemble and preprocessing, but (1) increase the uniform-mixture calibration strength slightly (a controlled “label smoothing” on probabilities) and (2) weight that smoothing up when the ensemble itself is peaky (low entropy), which is a minimal post-processing change that often improves KL without touching the model. I also ensure the spectrogram_id mapping cannot silently drop rows by falling back to test.csv ordering if any NaNs appear, preventing uniform fallback on valid rows. The script still runs end-to-end and writes a valid `submission.csv` with row-wise probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'We keep your exact model ensemble and preprocessing unchanged and focus on post-processing that reduces KL by avoiding overconfident predictions (a common cause of high KL like 1.41). Specifically, we slightly increase the uniform-mixture calibration and make it more conservative by allowing a higher max mix when predictions are very peaky, which should move the score down toward the 0.48 target without changing model logic. We also add a tiny temperature-style softening on the final probabilities (implemented purely as a probability transform, not changing logits/model) to further reduce overconfidence while preserving row sums to 1. All I/O paths remain the same and the script still writes a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your KL (1.40995, lower-is-better) is much worse than the target (0.4827), so we need a small but meaningful reduction in overconfidence without changing the model ensemble or preprocessing. The highest-impact minimal fix is to calibrate the *final* probabilities a bit more conservatively: increase the uniform-mixture smoothing slightly and apply a slightly stronger probability-temperature softening, both of which typically reduce KL by avoiding extremely small probabilities. I keep your exact inference and ensembling logic, only adjusting the post-processing hyperparameters and adding one safety clamp so every row stays strictly valid (positive and sums to 1). The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'We keep your model loading, preprocessing, and ensembling intact, and only adjust the final probability calibration to reduce KL (your 1.41 is typically caused by overconfident/peaky probabilities). Specifically, we make the uniform-mixture smoothing slightly stronger (still adaptive) and reduce the probability “temperature softening” (your current transform actually sharpens when `temperature>1`, which usually *worsens* KL). We also apply the softening as an invertible “flattening” power transform (no change to model outputs/logits), then re-normalize and clamp to keep rows valid. These are minimal post-processing-only changes aimed at moving the score down toward the 0.48 target without altering core inference logic.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.4827), so the most likely minimal win is to further reduce overconfident/peaky predictions, which KL penalizes heavily. I keep your exact model loading, preprocessing, and ensembling unchanged, and only fix the probability “softening” transform (your current `p**t` with `t>1` actually sharpens) to a flattening power transform that legitimately increases entropy. Then I slightly strengthen (but cap) the existing adaptive uniform-mixture smoothing to avoid near-zero probabilities while preserving valid row sums. This stays within post-processing only, should move KL down toward the target, and still writes a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'We keep your model loading, preprocessing, and ensembling exactly as-is, and only adjust the final probability calibration to better match the KL metric (your current 1.41 is typically caused by overconfident/peaky predictions that KL heavily penalizes). Specifically, we (1) slightly strengthen the uniform-mixture smoothing and (2) increase entropy a bit more via the existing probability “temperature softening” (which is already implemented as a flattening power transform). We also ensure the mix/softening happens in float64 and that the final clamp uses a slightly higher floor to avoid near-zero probabilities that can explode KL, while still strictly renormalizing rows to sum to 1. These are minimal post-processing-only changes intended to move the score down toward the 0.4827 target without altering the core inference logic.'
- What this solution (achieved 1.40995) has done: 'We need to reduce your KL from 1.40995 toward 0.4827 (lower is better), and the most plausible minimal lever without touching your model/feature logic is safer probability calibration to avoid near-zero probabilities and overly peaky distributions (which KL punishes heavily). I keep your entire inference/ensemble exactly as-is, but (1) compute the final calibration in float64, (2) apply a tiny Dirichlet-prior style add-α smoothing (equivalent to mixing with uniform but numerically safer), and (3) slightly raise the probability floor while renormalizing to ensure no class gets an extremely small probability. These changes are post-processing only, preserve evaluation semantics (valid probabilities summing to 1), and are designed to move the score downward toward the target rather than chasing the absolute best score. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'We need to move KL down from 1.40995 toward 0.4827 (lower is better) without changing your model or preprocessing, so the safest lever is probability calibration/post-processing to avoid extreme/overconfident outputs that KL heavily penalizes. I keep the exact ensemble/inference logic intact and only (1) slightly strengthen the probability floor and Dirichlet add-α smoothing (numerically stable, reduces near-zero probs), and (2) apply the same calibration only once (right now it’s effectively applied twice: inside the loop and again in the submission cell), which can distort probabilities and hurt KL. These are minimal, metric-aligned changes that should reduce KL while preserving valid probabilities summing to 1. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I keep your model loading, preprocessing, and ensembling exactly the same and only adjust the *final probability calibration* to better match the KL metric (your 1.40995 indicates the submission is still too peaky/overconfident). Specifically, I (1) slightly increase the Dirichlet-style add-α smoothing and probability floor to avoid near-zero class probabilities, and (2) make the probability softening a bit stronger (higher temperature) to increase entropy—both are post-processing only and preserve valid row-wise sums to 1. I also ensure this calibration is applied only once (it currently is, and we keep it that way) and keep all I/O paths unchanged while still writing a valid `submission.csv`. These minimal changes are aimed at reducing KL toward your 0.4827 target without touching the core inference logic.'
- What this solution (achieved 1.40995) has done: 'We need to reduce KL from 1.40995 toward 0.4827 (lower is better) without changing the model or preprocessing, so the safest lever is probability post-processing to avoid overconfident (near-zero) class probabilities that KL punishes heavily. I keep your ensemble and all transforms the same, but make the final calibration slightly more conservative by (1) increasing entropy via a mildly stronger probability flattening (temperature) and (2) strengthening the Dirichlet-style add-α smoothing and probability floor. These are minimal changes applied only at the end of inference, and we still renormalize so every row sums to 1 and the submission remains valid. I/O paths and submission format remain unchanged.'
- What this solution (achieved 1.40995) has done: 'We need to move your KL down from 1.40995 toward 0.4827 (lower is better), and the most likely minimal lever without touching the model/preprocessing is stronger, numerically-safe probability calibration to avoid near-zero/overconfident outputs that KL punishes heavily. I keep your exact model loading, preprocessing, and ensembling unchanged, but adjust only the final post-processing hyperparameters: slightly stronger entropy-increasing softening and a stronger Dirichlet-style add-α + probability floor. I also ensure the final calibration is applied once in a single place (at the end of inference, as it already is) and keep strict row-wise normalization so the submission always validates. This should reduce extreme probabilities and move the score meaningfully toward the target without changing core logic.'

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

    uniform_mix_base = 0.42  # was 0.36
    uniform_mix_adaptive = 0.32  # was 0.28
    uniform_mix_max = 0.85  # was 0.80

    prob_temperature = 4.20  # was 3.20

    dirichlet_alpha = 0.14  # was 0.09
    final_prob_floor = 1.0e-2  # was 6.0e-3


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

if submission["spectrogram_id"].isna().any():
    tmp = submission[["eeg_id"]].merge(
        test_df[["eeg_id", "spectrogram_id"]], on="eeg_id", how="left"
    )
    submission["spectrogram_id"] = tmp["spectrogram_id"].values

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


def adaptive_uniform_mix(p, base, adaptive, max_mix):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-15, 1.0)
    p = p / p.sum()
    ent = -np.sum(p * np.log(p))
    ent_norm = ent / np.log(6.0)  # 0..1
    extra = adaptive * (1.0 - ent_norm)
    mix = base + extra
    mix = float(np.clip(mix, 0.0, max_mix))
    return mix


def soften_probs(p, temperature):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-15, 1.0)
    p = p / p.sum()
    if temperature is None:
        return p
    t = float(temperature)
    if t <= 1.0:
        return p
    p = p ** (1.0 / t)
    p = p / p.sum()
    return p


def final_dirichlet_smooth_and_floor(p, alpha, floor):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-15, 1.0)
    p = p / p.sum()
    if alpha is not None and float(alpha) > 0.0:
        a = float(alpha)
        p = p + a
        p = p / p.sum()
    if floor is not None and float(floor) > 0.0:
        f = float(floor)
        p = np.clip(p, f, 1.0)
        p = p / p.sum()
    return p


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
            x = normalize_instance_wise(preprocessed_data).unsqueeze(0).to(device)
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

        mix = adaptive_uniform_mix(
            ensemble_prediction,
            base=Config.uniform_mix_base,
            adaptive=Config.uniform_mix_adaptive,
            max_mix=Config.uniform_mix_max,
        )
        if mix > 0:
            ensemble_prediction = (1.0 - mix) * ensemble_prediction + (
                mix * uniform_pred
            )

        ensemble_prediction = soften_probs(ensemble_prediction, Config.prob_temperature)

        ensemble_prediction = final_dirichlet_smooth_and_floor(
            ensemble_prediction,
            alpha=Config.dirichlet_alpha,
            floor=Config.final_prob_floor,
        )

        test_predictions.append(ensemble_prediction.astype(np.float32))

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
