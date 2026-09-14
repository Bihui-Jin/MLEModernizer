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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
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

1.0901246270812868

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I remove the two problematic imports that trigger the protobuf `MessageFactory.GetPrototype` crash (they aren’t used by the rest of the pipeline), so the notebook can start successfully. Then I fix the missing external model weight paths by adding a safe fallback that produces valid probabilistic predictions (uniform distribution) when the Kaggle input checkpoints aren’t available, ensuring a submission CSV is always generated. I also make the DataLoader compatible with Kaggle (set `num_workers=0`) and enforce numeric stability by clipping/renormalizing probabilities so every row sums to 1. Finally, I keep the core inference logic unchanged when weights are present; the fallback only activates if weights are missing, which is necessary to produce any submission at all.'
- What this solution (achieved 1.40995) has done: 'Your current score is much worse than the target (lower is better), and the biggest likely cause is that you’re not actually using the provided test spectrogram parquet features you loaded (you build `spectrograms2` but never feed them to the model), so predictions are coming from an EEG-only pipeline that doesn’t match the checkpoint’s expected inputs/feature distribution. With minimal changes and without altering model architecture or inference semantics, I (1) make the dataset prefer `spectrograms2` keyed by `spec_id` when available (fallback to EEG-derived spectrograms), (2) ensure `spec_id` is present and used, and (3) keep the same normalization/softmax/averaging and submission formatting. This should legitimately improve KL toward the target by aligning inference inputs with what these public checkpoints are typically trained on, while keeping everything else the same.'
- What this solution (achieved 1.40995) has done: 'Your score is worse than the target (lower KL is better), so the safest way to move toward the target without changing core model logic is to ensure the inference inputs match what the checkpoint expects and that predictions are aggregated correctly. I make the dataset consistently use the provided `test_spectrograms` tensors (already loaded into `spectrograms2`) without an extra erroneous transpose that currently scrambles the channels, and I keep the EEG-derived spectrograms only as a fallback when a spectrogram parquet is missing. I also remove the non-functional “5 folds” loop (it reloads identical weights 5 times and just repeats the same prediction), replacing it with a single pass that preserves identical inference semantics but avoids redundant averaging artifacts and time. Finally, I keep the same probability clipping/renormalization to guarantee valid submissions.'
- What this solution (achieved 1.40995) has done: 'I keep your model and inference pipeline intact, but fix two high-impact correctness issues that can easily worsen KL: (1) your targets are raw vote counts, yet you evaluate against probabilities, so I normalize the training targets (when present) to proper distributions inside the Dataset; this doesn’t change inference but ensures any future train/val use is metric-consistent and prevents accidental misuse. (2) For the provided spectrogram parquets, I switch the per-sample normalization from “divide by max” to the same `(x+40)/40` style scaling used in your EEG->mel path (clipped to [0,1]), which is a minimal, semantics-preserving calibration change to better match what these public checkpoints typically expect. Finally, I add a safe assertion that the prediction rows align 1:1 with `test` ordering before writing `submission.csv` to avoid silent misalignment.'
- What this solution (achieved 1.40995) has done: 'Your KL is worse than the target (lower is better), so the smallest likely gain without touching model architecture is to improve input calibration/compatibility with the checkpoint and reduce avoidable distribution shift. I (1) apply the same dB-range clipping used in your EEG->mel path to the provided test spectrogram parquets (instead of simple `(x+40)/40` after zero-filling), which better matches how these models are typically trained, and (2) add a tiny temperature scaling on the softmax (T>1) to slightly smooth overconfident predictions, which often lowers KL when miscalibrated. I keep the rest of inference identical, keep probability clipping/renormalization, and still produce a valid `submission.csv`. These are minimal, metric-aligned tweaks expected to move KL downward toward your target without changing the core pipeline.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower is better) is worse than the target (1.0901), so we should cautiously improve calibration/compatibility while keeping your model/inference logic intact. The biggest low-risk win here is to ensure the **provided test spectrogram parquets** are scaled like your EEG-derived mel dB pipeline (they are not mel-power; your current `[-40,0]` clamp is likely mismatched), so I switch `_spec_to_img` to a robust per-sample dB-like normalization using high-percentile reference then clamp to `[-40,0]` and map to `[0,1]` (same final convention as before). I also add a tiny, metric-aligned smoothing (blend a small epsilon of uniform distribution) after softmax to reduce overconfidence, which commonly reduces KL when miscalibrated, without changing the model architecture or training approach. Everything else (data loading, encoder->classifier inference, CSV writing, row-sum enforcement) stays the same.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower is better) is far from the target (1.0901), so we should make a small, metric-aligned calibration change rather than altering any model structure. The two safest levers here are (1) ensure the provided parquet spectrograms are normalized in a stable “dB-like then [0,1]” way without accidental over-compression, and (2) slightly increase post-softmax smoothing to reduce overconfident probabilities (KL is very sensitive to overconfidence). I keep your exact model/encoder/classifier and inference flow, and only tune the temperature + uniform-mix and make `_spec_to_img` use a consistent log scaling regardless of sign (with a small shift if needed), then still clip+renormalize so every row sums to 1 and the submission is valid. These are minimal changes expected to reduce KL toward the target without touching the core pipeline.'
- What this solution (achieved 1.40995) has done: 'To move your KL score down toward the target with minimal risk, I keep your model and inference loop intact and only adjust the post-processing calibration that strongly affects KL. Specifically, I reduce the current softmax smoothing/temperature (which can over-flatten predictions and hurt KL when the model is already uncertain), and instead apply a very small label-smoothing-style floor after softmax that prevents extreme zeros without overly diluting the distribution. I also make sure the parquet spectrogram normalization matches your EEG-mel path even more closely by using `librosa.power_to_db(..., ref=np.max)` on a nonnegative “power-like” version of the parquet values (same transform family as your EEG path), while keeping the same output shape and [0,1] scaling. Everything else (data loading, model weights, encoder usage, submission formatting, row-sum enforcement) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'We need to move KL down (lower is better) from 1.40995 toward 1.09012, so we should make minimal, metric-aligned calibration changes rather than altering the model. The safest lever here is post-softmax probability smoothing: your current combination of temperature + uniform mix + probability floor can easily over-flatten distributions and hurt KL; we replace it with a single tiny “Dirichlet-like” additive smoothing (epsilon) that prevents near-zeros without washing out the model signal. We keep the exact encoder+classifier inference and the same spectrogram construction, only adjusting the probability post-processing and making it consistent between the checkpoint path and the uniform-fallback path. This should reduce overconfidence/underconfidence extremes and typically lowers KL toward the target while preserving core logic and producing a valid submission.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower is better) is substantially worse than the target (1.09012), so we should make a small, metric-aligned change that improves calibration without changing the model or data flow. The safest lever is post-softmax probability calibration: instead of a fixed additive smoothing (which can sometimes harm KL by over-flattening), we apply a tiny “floor then renormalize” that only prevents near-zero probabilities (KL is very sensitive to zeros) while preserving relative confidence. This keeps the encoder→classifier inference exactly the same and only adjusts the final probability vector. I also ensure the same post-processing is applied consistently both in the checkpoint path and the uniform-fallback path.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower is better) is worse than the target (1.09012), so we should make the smallest inference-time change that improves calibration without touching the model or data flow. The lowest-risk lever for KL here is to reduce extreme probabilities by applying a tiny uniform-mix (Dirichlet-like) smoothing after softmax, while keeping your existing probability floor + renormalization and the exact same encoder→classifier inference. I also ensure the parquet spectrogram normalization cannot produce degenerate all-zeros dB frames (which can distort the distribution) by adding a tiny epsilon before `power_to_db`, preserving the same transform family and output range. These changes are minimal, metric-aligned, and keep the submission format and row-sum constraints intact.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995; lower is better) is worse than the target (1.09012), so we should make the smallest metric-aligned inference-time calibration tweak rather than changing the model or data flow. The biggest low-risk lever for KL here is post-softmax smoothing: your current uniform-mix (0.02) can over-flatten predictions and worsen KL if the model is already underconfident, so I reduce it to a much smaller value while keeping your existing probability floor + renormalization intact. I also make the probability-flooring in cell 10 use the same `PROB_FLOOR` constant (instead of a different hardcoded `1e-8`) to keep post-processing consistent and avoid accidental near-zero probabilities. Everything else (spectrogram usage, encoder→classifier inference, dataloader, submission format) remains unchanged and still writes a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'We need to move KL down from 1.40995 toward 1.09012 (lower is better) without changing the model or inference flow, so the safest minimal lever is probability calibration/post-processing. I reduce the current uniform-mix smoothing slightly (it can over-flatten and worsen KL) and instead apply a tiny temperature scaling on the logits (T>1) to soften only when the model is overconfident, which often improves KL. I also make the final probability post-processing consistent by doing all smoothing/flooring/renorm exactly once in one place to avoid double-clipping effects. Everything else (dataset construction, spectrogram usage, encoder→classifier, batch loop, submission formatting) remains unchanged and still writes a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I keep your model/inference pipeline intact and only adjust the *final probability calibration* in a way that usually reduces KL when a model is miscalibrated. Concretely, I slightly lower the current logit temperature (your 1.15 can over-flatten) and slightly increase the uniform-mix smoothing (your 0.001 is often too small to prevent harmful near-zeros under KL). I also make sure we don’t apply the probability floor/renorm twice in a way that can unintentionally distort distributions by only doing it once at the very end (still enforcing valid row sums). These changes are minimal, metric-aligned, and preserve the exact architecture, inputs, and inference flow.'
- What this solution (achieved 1.40995) has done: 'To move KL down toward your 1.0901 target without changing the model or inference flow, I’m making two minimal, metric-aligned calibration fixes. First, I apply your `PROB_FLOOR` *after* all smoothing/softmax (already done) but reduce it slightly because `1e-4` can over-flatten distributions and worsen KL; a smaller floor still prevents zeros (which KL hates) while preserving model signal. Second, I slightly reduce the uniform-mix smoothing because it can also over-flatten; we keep temperature scaling unchanged to avoid altering core semantics too much. Everything else (data loading, spectrogram usage, encoder→classifier inference, CSV formatting, row-sum enforcement) stays the same and still produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import gc
import time
from pathlib import Path

from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import albumentations as albu


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)



## === cell 1
train_df = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
)
TARGETS = train_df.columns[-6:].tolist()
print("Targets:", TARGETS)



## === cell 2
test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
print("Test shape", test.shape)
test.head()



## === cell 3
PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
files2 = os.listdir(PATH2)
print(f"There are {len(files2)} test spectrogram parquets")

spectrograms2 = {}
for i, f in enumerate(files2):
    if i % 200 == 0:
        print(i, ", ", end="")
    tmp = pd.read_parquet(f"{PATH2}{f}")
    name = int(f.split(".")[0])
    spectrograms2[name] = tmp.iloc[:, 1:].values

test = test.rename({"spectrogram_id": "spec_id"}, axis=1)



## === cell 4
import pywt, librosa

USE_WAVELET = None

NAMES = ["LL", "LP", "RP", "RR"]
FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]

directory_path = "EEG_Spectrograms/"
if not os.path.exists(directory_path):
    os.makedirs(directory_path)


def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x, wavelet="haar", level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])
    ret = pywt.waverec(coeff, wavelet, mode="per")
    return ret


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

    return img




## === cell 5
PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
DISPLAY = 0  # keep non-interactive; no plotting in Kaggle submit runs
EEG_IDS2 = test.eeg_id.unique()
all_eegs2 = {}

print("Converting Test EEG to Spectrograms...")
print()
for i, eeg_id in enumerate(tqdm(EEG_IDS2)):
    img = spectrogram_from_eeg(f"{PATH2}{eeg_id}.parquet", i < DISPLAY)
    all_eegs2[eeg_id] = img

print("Built test EEG spectrogram dict:", len(all_eegs2))



## === cell 6
TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
TARS2 = {x: y for y, x in TARS.items()}


class EEGDataset(Dataset):
    def __init__(
        self, data, augment=False, mode="train", eeg_specs=None, spec_specs=None
    ):
        self.data = data.reset_index(drop=True)
        self.augment = augment
        self.mode = mode
        self.eeg_specs = eeg_specs if eeg_specs is not None else {}
        self.spec_specs = spec_specs if spec_specs is not None else {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X, y = self._generate_data([index])
        if self.augment:
            X = self.__augment(X)
        if self.mode == "train":
            return X[0], y[0]
        else:
            return X[0]

    def _spec_to_img(self, spec_arr_2d):
        """
        Keep the same spectrogram->image mapping family as the EEG mel path:
        power_to_db(ref=np.max) -> clamp[-40,0] -> map to [0,1].

        Stability: epsilon before power_to_db to avoid all-zero frames.
        """
        a = np.asarray(spec_arr_2d, dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

        T = a.shape[0]
        if T >= 256:
            a = a[:256]
        else:
            pad = np.zeros((256 - T, a.shape[1]), dtype=np.float32)
            a = np.concatenate([a, pad], axis=0)

        Fdim = a.shape[1]
        if Fdim >= 512:
            a = a[:, :512]
        else:
            pad = np.zeros((256, 512 - Fdim), dtype=np.float32)
            a = np.concatenate([a, pad], axis=1)

        amin = float(np.min(a))
        if amin < 0.0:
            a = a - amin
        a = np.maximum(a, 0.0)

        a = a + 1e-10

        a_db = librosa.power_to_db(a, ref=np.max).astype(np.float32)
        a_db = np.clip(a_db, -40.0, 0.0)

        a_db = a_db.reshape(256, 4, 128)  # (256,4,128)
        img_hwc = np.transpose(a_db, (2, 0, 1))  # (128,256,4)

        img_hwc = (img_hwc + 40.0) / 40.0
        img_hwc = np.clip(img_hwc, 0.0, 1.0)
        return img_hwc.astype(np.float32)

    def _generate_data(self, indexes):
        X = np.zeros((len(indexes), 4, 128, 256), dtype="float32")
        y = np.zeros((len(indexes), 6), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]

            img_hwc = None
            if "spec_id" in row.index:
                sid = int(row.spec_id)
                if sid in self.spec_specs:
                    img_hwc = self._spec_to_img(self.spec_specs[sid])  # (128,256,4)

            if img_hwc is None:
                eid = int(row.eeg_id)
                img_hwc = self.eeg_specs[eid]  # (128,256,4)

            img = np.transpose(img_hwc, (2, 0, 1))  # -> (4,128,256)
            X[j] = img

            if self.mode != "test":
                vv = row[TARGETS].values.astype(np.float32)
                s = float(np.sum(vv))
                if s > 0:
                    vv = vv / s
                else:
                    vv[:] = 1.0 / 6.0
                y[j,] = vv

        return X, y

    def _random_transform(self, img):
        composition = albu.Compose(
            [
                albu.HorizontalFlip(p=0.5),
            ]
        )
        img_hwc = np.transpose(img, (1, 2, 0))
        out = composition(image=img_hwc)["image"]
        return np.transpose(out, (2, 0, 1))

    def __augment(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i] = self._random_transform(img_batch[i])
        return img_batch




## === cell 7
class ResNetBlock(nn.Module):
    def __init__(
        self, in_channels, kernel_size, modify=False, bn=True, scale_factor=(1, 1)
    ):
        super().__init__()
        self.modify = modify
        if modify == "downsample":
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels * 2,
                stride=2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            )
            self.conv2 = nn.Conv2d(
                in_channels=in_channels * 2,
                out_channels=in_channels * 2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            )
            if bn:
                self.bn1 = nn.BatchNorm2d(in_channels * 2)
                self.bn2 = nn.BatchNorm2d(in_channels * 2)
            else:
                self.bn1 = nn.Identity()
                self.bn2 = nn.Identity()

        elif modify == "upsample":
            self.conv1 = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                stride=2,
                kernel_size=kernel_size,
                output_padding=scale_factor,
                padding=kernel_size // 2,
                bias=False,
            )
            self.conv2 = nn.Conv2d(
                in_channels=in_channels // 2,
                out_channels=in_channels // 2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            )
            self.bn1 = nn.BatchNorm2d(in_channels // 2)
            self.bn2 = nn.BatchNorm2d(in_channels // 2)
        else:
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            self.conv2 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(in_channels)

        self.act = nn.ReLU()

        if modify == "downsample":
            self.proj = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels * 2,
                stride=2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        if modify == "upsample":
            self.proj = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                stride=2,
                kernel_size=kernel_size,
                output_padding=scale_factor,
                padding=kernel_size // 2,
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.modify:
            x = self.proj(x)
        out = x + out
        out = self.act(out)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 16, 7, 1, 7 // 2)
        self.rnb1 = ResNetBlock(16, 3, modify="downsample")
        self.rnb2 = ResNetBlock(32, 3, modify="downsample")
        self.rnb3 = ResNetBlock(64, 3, modify="downsample")
        self.rnb4 = ResNetBlock(128, 3, modify="downsample")
        self.rnb5 = ResNetBlock(256, 3, modify="downsample")
        self.rnb6 = ResNetBlock(512, 3, modify="downsample")
        self.rnb7 = ResNetBlock(1024, 3, modify="downsample")
        self.rnb8 = ResNetBlock(2048, 3, modify="downsample")

    def forward(self, x):
        x = self.conv(x)
        x = self.rnb1(x)
        x = self.rnb2(x)
        x = self.rnb3(x)
        x = self.rnb4(x)
        x = self.rnb5(x)
        x = self.rnb6(x)
        x = self.rnb7(x)
        x = self.rnb8(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnb1 = ResNetBlock(4096, 3, modify="upsample", scale_factor=(0, 1))
        self.rnb2 = ResNetBlock(2048, 3, modify="upsample")
        self.rnb3 = ResNetBlock(1024, 3, modify="upsample")
        self.rnb4 = ResNetBlock(512, 3, modify="upsample")
        self.rnb5 = ResNetBlock(256, 3, modify="upsample")
        self.rnb6 = ResNetBlock(128, 3, modify="upsample")
        self.rnb7 = ResNetBlock(64, 3, modify="upsample")
        self.rnb8 = ResNetBlock(32, 3, modify="upsample")
        self.conv = nn.Conv2d(16, 4, 3, 1, 3 // 2)

    def forward(self, x):
        x = self.rnb1(x)
        x = self.rnb2(x)
        x = self.rnb3(x)
        x = self.rnb4(x)
        x = self.rnb5(x)
        x = self.rnb6(x)
        x = self.rnb7(x)
        x = self.rnb8(x)
        x = self.conv(x)
        return x




## === cell 8
class SimpleAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(Encoder(), Decoder())

    def forward(self, x):
        return self.net(x)


class SimpleNet2(nn.Module):
    def __init__(self):
        super(SimpleNet2, self).__init__()
        self.fc1 = nn.Linear(4096, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.output = nn.Linear(128, 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output(x)
        return x


class SimpleNet1(nn.Module):
    def __init__(self):
        super(SimpleNet1, self).__init__()
        self.fc1 = nn.Linear(4096, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 128)
        self.output = nn.Linear(128, 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x




## === cell 9
def find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def prob_floor_renorm_np(p: np.ndarray, floor: float) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    p = np.nan_to_num(p, nan=1.0 / 6.0, posinf=1.0 / 6.0, neginf=1.0 / 6.0)
    p = np.maximum(p, floor)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def prob_floor_renorm_torch(p: torch.Tensor, floor: float) -> torch.Tensor:
    p = torch.nan_to_num(p, nan=1.0 / 6.0, posinf=1.0 / 6.0, neginf=1.0 / 6.0)
    p = torch.clamp(p, min=floor)
    p = p / p.sum(dim=1, keepdim=True)
    return p


ae_ckpt_candidates = [
    "/kaggle/input/best_model.pth/pytorch/hms-autoencoder/1/best_model.pth",
]
clf_ckpt_candidates = [
    "/kaggle/input/simplenet1/pytorch/2/1/model_simplenet2_20_e3.pth",
]

ae_ckpt = find_first_existing(ae_ckpt_candidates)
clf_ckpt = find_first_existing(clf_ckpt_candidates)

print("AE checkpoint found:", ae_ckpt)
print("CLF checkpoint found:", clf_ckpt)

test_ds = EEGDataset(test, mode="test", eeg_specs=all_eegs2, spec_specs=spectrograms2)
test_loader = DataLoader(
    test_ds, shuffle=False, batch_size=64, num_workers=0, pin_memory=(DEVICE == "cuda")
)

PROB_FLOOR = 1e-6

UNIFORM_MIX = 0.002

LOGIT_TEMPERATURE = 1.08

if (ae_ckpt is None) or (clf_ckpt is None):
    pred = np.full((len(test), 6), 1.0 / 6.0, dtype=np.float32)
    pred = prob_floor_renorm_np(pred, PROB_FLOOR)
    print(
        "WARNING: Missing checkpoints. Using uniform predictions. pred shape:",
        pred.shape,
    )
else:
    model_ae = SimpleAE()
    model_ae = nn.DataParallel(model_ae)
    state_ae = torch.load(ae_ckpt, map_location="cpu")
    model_ae.load_state_dict(state_ae)
    model_ae.to(DEVICE).eval()

    encoder = nn.DataParallel(model_ae.module.net[0])
    encoder.to(DEVICE).eval()

    model = SimpleNet2()
    state_clf = torch.load(clf_ckpt, map_location="cpu")
    model.load_state_dict(state_clf)
    model.to(DEVICE).eval()

    fold_preds = []
    with torch.inference_mode():
        for test_batch in test_loader:
            test_batch = torch.as_tensor(test_batch, device=DEVICE, dtype=torch.float32)
            emb = encoder(test_batch)
            embs = emb.squeeze(2, 3)
            pred_logits = model(embs)

            pred_logits = pred_logits / LOGIT_TEMPERATURE
            probabilities = F.softmax(pred_logits, dim=1)

            if UNIFORM_MIX > 0:
                probabilities = (1.0 - UNIFORM_MIX) * probabilities + UNIFORM_MIX * (
                    1.0 / 6.0
                )

            fold_preds.append(probabilities.detach().cpu().numpy())

    pred = np.concatenate(fold_preds, axis=0).astype(np.float32)
    print("Test preds shape", pred.shape)



## === cell 10
pred = prob_floor_renorm_np(pred, PROB_FLOOR)

assert pred.shape[0] == len(
    test
), f"pred rows ({pred.shape[0]}) != test rows ({len(test)})"

sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
sub[TARGETS] = pred

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
print("Submission shape", sub.shape)
print("Saved:", sub_path)
sub.head()



## === cell 11
row_sums = sub.iloc[:, -6:].sum(axis=1)
print(row_sums.describe())
print("All close to 1:", np.allclose(row_sums.values, 1.0, atol=1e-5))
