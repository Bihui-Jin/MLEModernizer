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

cudf-polars-cu12==25.6.0
geopandas==0.14.4
librosa==0.11.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
polars==1.25.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scipy==1.15.3
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

0.4134363371417311

# 6. Current score

1.46633

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.48999) has done: 'I fix the tensor reshaping bug by correctly expanding the 4×2500 EEG chain into 16 channels (4 regions × 4 leads) instead of forcing an invalid 16×2500 reshape, which currently crashes inference and prevents creating a submission. I also make row extraction from the Polars test frame robust (Polars slicing returns a DataFrame) and add a safe fallback if the expected EEG parquet is missing. Finally, I ensure `preds_final` always has shape `(n_test, 6)` and write a valid `submission.csv` with the required columns that sum to 1 per row.'
- What this solution (achieved 1.48999) has done: 'Your current score (1.48999, lower-is-better) is far worse than the target (0.4134), so we should improve performance with minimal, metric-aligned fixes. The biggest issue is that you are not loading any trained weights: `from_pretrained()` is called but `load_model()` is never used, so predictions are essentially random/constant-like and KL be poor. I minimally add robust checkpoint discovery/loading inside each fold directory (common filenames like `*.pt/*.pth/*.bin/*.ckpt`) while keeping the exact same preprocessing and `knn_predict()` inference path. I also make the test-row iteration Polars-safe and add a tiny epsilon + renormalization guard to ensure every row sums to 1 (required for submission validity).'
- What this solution (achieved 1.48999) has done: 'Your current score (1.48999, lower-is-better) is far from the target (0.4134), so we should improve predictive signal with the smallest changes that don’t alter the modeling approach. The biggest remaining performance issue is that inference is feeding the model a different tensor layout than the fallback model expects (and likely different from the original external model too), which makes predictions effectively miscalibrated/noisy. I keep the same preprocessing and `knn_predict()` path, but fix the tensor shape to be consistent: treat the EEG as 16 channels and pass it as `(B, 16, T)` and only add singleton dims if the model explicitly needs them. I also add a tiny robust adapter that tries both common input layouts for `knn_predict()` (3D and 4D) without changing the core logic, so the same code works whether the external model is available or we fall back.'
- What this solution (achieved 1.48999) has done: 'Your score is much worse than the target (lower-is-better), so the smallest high-impact fix is to ensure we actually use the provided spectrograms as the competition expects, while keeping the same ensemble + `knn_predict()` inference path. Right now you compute a time-series EEG chain but never use the already-provided `test_spectrograms/`, which likely makes the external pretrained KNN model misaligned and yields near-random probabilities. I minimally add a spectrogram loader (center crop to a fixed 256×256) and feed that into `knn_predict()` first, falling back to the current EEG pipeline only if the spectrogram parquet is missing or incompatible. I also keep the exact same probability clipping + renormalization to guarantee valid submissions.'
- What this solution (achieved 1.48335) has done: 'We make the smallest metric-aligned change that should move KL down: ensure the model always receives the correct input layout for each modality, instead of pushing a 2D spectrogram through the EEG `knn_predict()` path (which likely makes predictions near-random). Concretely, we keep your existing EEG pipeline unchanged, but fix the spectrogram branch to convert the spectrogram into a “16-channel over time” representation via simple band-averaging so it matches the `(B, 16, T)` interface your inference expects. We also normalize consistently and add a final safety guard that, if the model output shape is wrong for any reason, falls back to uniform probabilities (valid submission, avoids NaNs harming KL). These changes preserve your architecture/training approach while improving input compatibility, which is a high-impact correctness fix given your current score is far from target.'
- What this solution (achieved 1.48539) has done: 'The score gap to the target is large (1.483 → 0.413, lower-is-better), and the biggest likely cause in your current pipeline is that your “spectrogram branch” feeds a generic 16×T band-averaged representation into a model that was almost certainly trained on *the competition’s 4-region spectrogram layout* (LL/LP/RP/RL), so inference is misaligned and looks close to random. I keep your exact ensemble/inference approach (`knn_predict` on a (B,16,T) tensor, mean-std normalization, averaging models) but change only the spectrogram-to-16×T conversion to use the 4 region groups and 4 frequency bands per region (4×4=16), which matches the dataset’s spectrogram schema and should move KL down materially. I also make checkpoint discovery slightly more robust by searching recursively inside each fold directory (many Kaggle datasets nest checkpoints), without changing how weights are loaded. Everything else (EEG fallback path, normalization, probability clipping/renorm, submission writing) stays the same.'
- What this solution (achieved 1.48335) has done: 'Your current score (1.485, lower-is-better) is far from the target (0.413), so we need a small but high-impact correctness fix rather than tuning. The biggest issue is that the spectrogram branch is using the **wrong axis semantics**: the parquet is shaped roughly `(freq, time)`, but the current conversion treats it as `(time, freq)`, which scrambles features and makes predictions near-random. I minimally fix `_load_test_spectrogram()` to return the matrix as `(freq_bins, time_steps)` and update `_spec2d_to_eeglike_16xt()` to build 16 channels from **4 regions × 4 frequency bands** over the correct frequency axis, then resample along the time axis to `T=2500`. Everything else (model loading, normalization, ensemble averaging, EEG fallback, submission writing) stays the same.'
- What this solution (achieved 1.47099) has done: 'Your current KL (1.483, lower-is-better) is far from the target (0.413), so we need a small correctness fix that improves signal without changing the model or training logic. The biggest likely issue is that the spectrogram parquet layout is being converted into 16×T features using an incorrect assumption (splitting the frequency axis into 4 “regions”), which scrambles the intended LL/LP/RP/RL groups and produces near-random predictions. I minimally change spectrogram loading to preserve the original column structure and explicitly build 16 channels as **4 regions × 4 frequency bands** by selecting columns by region suffix (LL/LP/RP/RL), then band-averaging along the frequency-bin axis and resampling time to 2500. Everything else (checkpoint discovery/loading, normalization, ensemble averaging, EEG fallback, and probability renorm for valid submission) stays the same.'
- What this solution (achieved 1.4886) has done: 'The score is far worse than the target (KL 1.47 vs 0.413, lower is better), so we need a small but high-impact correctness/calibration fix without changing your model/inference approach. The most likely issue is a mismatch between what the external KNN model expects and what we feed it: your spectrogram conversion currently averages over raw frequency rows uniformly, but HMS spectrograms are on a log-like scale and benefit from stabilizing/standardizing per-region columns before band pooling. I minimally (1) apply a safe `log1p` transform + per-column standardization inside `_load_test_spectrogram()` (only for spectrogram branch), and (2) add a tiny temperature smoothing on the final probabilities (mix with a small uniform prior) to reduce overconfident KL penalties while preserving semantics and valid probability sums. Everything else (checkpoint discovery/loading, `(B,16,T)` interface, ensemble averaging, EEG fallback, and submission writing) stays the same.'
- What this solution (achieved 1.49405) has done: 'We make two minimal, metric-aligned fixes that should materially reduce KL (lower is better) without changing your core modeling/inference flow. First, we remove the uniform-mixing “calibration” (it tends to hurt good models and can also mask useful signal), and instead only apply a tiny epsilon floor + renormalization (already required for valid submissions). Second, we make the spectrogram preprocessing closer to common HMS solutions by applying a per-sample robust normalization (median/IQR) rather than per-column standardization, which is unstable across test items and can distort KNN distances. Everything else (model loading, `(B,16,T)` interface, ensemble averaging, fallback paths, submission schema) remains unchanged.'
- What this solution (achieved 1.44306) has done: 'The current score is far above the target (KL 1.49 vs 0.413; lower is better), so the most likely “minimal but high-impact” improvement is to ensure the inference output is calibrated to the competition’s soft-label distribution (vote proportions) and not overly peaky. Without changing your model or feature extraction, I add a tiny, metric-aligned post-processing step: apply a fixed temperature to soften probabilities before the existing epsilon+renormalization (this is a standard KL-reduction calibration that preserves semantics and always yields valid probabilities). I also make the spectrogram path selection deterministic by preferring spectrogram inference whenever the file exists (as you already do), and add a safety check to prevent degenerate all-equal outputs from numerical issues. These changes are small, fast, and directly target KL behavior while keeping your core pipeline intact.'
- What this solution (achieved 1.4568) has done: 'Your current KL (1.443, lower-is-better) is still far from the target (0.413), so the smallest likely high-impact improvement is to make the final probability post-processing better aligned to KL on soft labels without changing your model/inference pipeline. I keep your feature extraction, model loading, and `knn_predict()` ensemble exactly the same, but replace the fixed temperature with a safer, more KL-friendly smoothing: a tiny uniform prior mix plus an optional mild temperature (applied only if outputs are overly peaky). This reduces the KL penalty from overconfident wrong classes while preserving the model’s ranking signal, and still guarantees valid probability rows that sum to 1. I also make the probability finalization apply identically to both spectrogram and EEG branches and add a stricter finite/shape guard to avoid rare numerical blow-ups.'
- What this solution (achieved 1.47192) has done: 'Your current KL (1.4568, lower-is-better) is far worse than the target (0.4134), so we need a small but high-impact, metric-aligned correction rather than tuning. The most likely culprit is that the spectrogram preprocessing applies a per-sample median/IQR normalization and a fairly strong uniform prior mix, both of which can distort KNN distances and wash out signal; I switch to a simpler per-sample z-score after `log1p` (still stable) and reduce the prior mix to near-zero while keeping the same model, features, and inference flow. I also make the spectrogram time axis handling more faithful by aggregating over time bins (columns) in a stable way and only interpolating to 2500 at the very end, keeping the same `(B,16,T)` interface. Everything else (model loading, ensemble averaging, EEG fallback, and strict probability normalization) remains the same to keep changes minimal and runtime within limits.'
- What this solution (achieved 1.46633) has done: 'We make one minimal but high-impact metric-aligned correction: remove the extra *per-channel* mean/std normalization before `knn_predict()` (for both spectrogram- and EEG-derived 16×T inputs), because an external KNN-based model typically expects features in the same absolute scale it was trained on and this normalization can destroy distance structure and hurt KL badly. We keep your spectrogram preprocessing (log1p + global per-sample z-score) and the entire model/ensemble/inference flow unchanged, and only adjust probability finalization slightly by lowering temperature to avoid oversmoothing if the model is already underconfident. Everything else (checkpoint discovery/loading, fallback behavior, shape guards, and strict row-wise probability normalization for submission validity) stays the same.'

# 9. Code solution

## === cell 0
from tqdm.auto import tqdm
import pandas as pd
import polars as pl
import numpy as np
import argparse
import os
import glob
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Optional, Tuple, List, Dict, Any



## === cell 1
DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"
SPEC_DIR = os.path.join(DATA_DIR, "test_spectrograms/")
EEG_DIR = os.path.join(DATA_DIR, "test_eegs/")

df_train = pl.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test = pl.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_submission = pl.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

df_test.head()



## === cell 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## === cell 3
def load_model(path: str, model: nn.Module) -> nn.Module:
    ckpt = torch.load(path, map_location=torch.device("cpu"))
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    return model




## === cell 4
try:
    sys.path.append("/kaggle/input/hms-models/")
    from eeg_cnn_rnn_knn import EegModel as _ExternalEegModel  # type: ignore

    CurrModel = _ExternalEegModel
    USING_FALLBACK = False
except Exception as e:
    USING_FALLBACK = True

    class EegModel(nn.Module):
        """
        Fallback model with the same public API used below:
          - from_pretrained(fold_path)
          - knn_predict(x) -> (batch, 6) probabilities

        Note: this fallback is only for correctness/runnability. It expects x as either:
          - (B, C, T) where C=16, or
          - (B, C, 1, T) and will squeeze the singleton dim.
        """

        def __init__(self, n_classes: int = 6):
            super().__init__()
            self.n_classes = n_classes
            self.fc = nn.Linear(16 * 4, n_classes, bias=True)

        def from_pretrained(self, fold_path: str):
            return self

        @torch.no_grad()
        def knn_predict(self, x: torch.Tensor) -> torch.Tensor:
            x = x.float()
            if x.ndim == 4:
                if x.shape[2] == 1:
                    x = x.squeeze(2)
            if x.ndim != 3:
                raise ValueError(
                    f"Expected x.ndim==3 (or 4 with singleton), got {x.shape}"
                )

            mean = x.mean(dim=-1)  # (B, 16)
            std = x.std(dim=-1)  # (B, 16)
            abs_mean = x.abs().mean(dim=-1)  # (B, 16)
            rms = torch.sqrt((x**2).mean(dim=-1))  # (B, 16)
            feats = torch.cat([mean, std, abs_mean, rms], dim=1)  # (B, 64)
            logits = self.fc(feats)
            probs = torch.softmax(logits, dim=1)
            return probs

    CurrModel = EegModel

print("Using fallback model:", USING_FALLBACK)



## === cell 5
import librosa
from scipy.ndimage import convolve

KERNEL = np.array([-1, -1, -1, 0, 1, 1, 1])


def compute_spec(eeg: np.ndarray) -> np.ndarray:
    spectrogram = librosa.stft(
        eeg,
        n_fft=1024,
        hop_length=39,
        win_length=256,
        window="hann",
        center=True,
        pad_mode="constant",
        out=None,
        dtype=None,
    )
    spectrogram = librosa.power_to_db(np.abs(spectrogram) ** 2, ref=np.max).astype(
        np.float32
    )
    spectrogram = (spectrogram + 80) / 80
    spectrogram = spectrogram**2
    return spectrogram[:256][::2, ::2]


def spec(eeg: np.ndarray) -> np.ndarray:
    sp = compute_spec(eeg)
    eeg2 = convolve(eeg, KERNEL)
    sp = sp + compute_spec(eeg2)
    return sp / 2


def compute_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    Fp1 = df_eeg["Fp1"].to_numpy()
    Fp2 = df_eeg["Fp2"].to_numpy()
    F3 = df_eeg["F3"].to_numpy()
    F4 = df_eeg["F4"].to_numpy()
    F7 = df_eeg["F7"].to_numpy()
    F8 = df_eeg["F8"].to_numpy()
    C3 = df_eeg["C3"].to_numpy()
    C4 = df_eeg["C4"].to_numpy()
    P3 = df_eeg["P3"].to_numpy()
    P4 = df_eeg["P4"].to_numpy()
    T3 = df_eeg["T3"].to_numpy()
    T4 = df_eeg["T4"].to_numpy()
    T5 = df_eeg["T5"].to_numpy()
    T6 = df_eeg["T6"].to_numpy()
    O1 = df_eeg["O1"].to_numpy()
    O2 = df_eeg["O2"].to_numpy()

    ll = np.stack([spec(Fp1 - F7), spec(F7 - T3), spec(T3 - T5), spec(T5 - O1)])
    lp = np.stack([spec(Fp1 - F3), spec(F3 - C3), spec(C3 - P3), spec(P3 - O1)])
    rp = np.stack([spec(Fp2 - F4), spec(F4 - C4), spec(C4 - P4), spec(P4 - O2)])
    rl = np.stack([spec(Fp2 - F8), spec(F8 - T4), spec(T4 - T6), spec(T6 - O2)])

    chain = np.stack([ll, lp, rp, rl])[:, 0]
    return chain


def preprocess_chain(chain: np.ndarray) -> np.ndarray:
    return chain


def compute_spec_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_chain(df_eeg)
    chain = preprocess_chain(chain)
    return chain




## === cell 6
import math
from typing import Union, Tuple, List


def bin_array(
    array,
    bin_size,
    axis=-1,
    pad_dir="symmetric",
    mode="edge",
    return_padding=False,
    **padding_kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    if axis == -1:
        axis = array.ndim - 1

    curr_len = array.shape[axis]
    n_bins = math.ceil(curr_len / bin_size)
    new_len = n_bins * bin_size

    new_shape = list(array.shape)
    new_shape[axis] = n_bins
    new_shape.insert(axis + 1, bin_size)

    padding = [(0, 0)] * array.ndim
    if curr_len != new_len:
        if pad_dir == "left":
            pad_l = new_len - curr_len
            pad_r = 0
        elif pad_dir == "right":
            pad_l = 0
            pad_r = new_len - curr_len
        else:
            pad_l = (new_len - curr_len) // 2
            pad_r = (new_len - curr_len) - pad_l

        padding[axis] = (pad_l, pad_r)
        array = np.pad(array, padding, mode=mode, **padding_kwargs)

    array = array.reshape(new_shape)

    if return_padding:
        return array, padding
    return array




## === cell 7
from scipy.signal import butter, filtfilt


def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4, btype="lowpass"):
    b, a = butter(N=order, Wn=cutoff_freq / (0.5 * fs), btype=btype, analog=False)
    return filtfilt(b, a, eeg_data)


def compute_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = butter_filter(eeg, cutoff_freq=np.array([0.25, 50]), btype="bandpass")
    eeg = bin_array(eeg, bin_size=4, mode="reflect").mean(axis=-1)  # 200Hz -> 50Hz
    return eeg


def compute_eeg_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    Fp1 = df_eeg["Fp1"].to_numpy()
    Fp2 = df_eeg["Fp2"].to_numpy()
    F3 = df_eeg["F3"].to_numpy()
    F4 = df_eeg["F4"].to_numpy()
    F7 = df_eeg["F7"].to_numpy()
    F8 = df_eeg["F8"].to_numpy()
    C3 = df_eeg["C3"].to_numpy()
    C4 = df_eeg["C4"].to_numpy()
    P3 = df_eeg["P3"].to_numpy()
    P4 = df_eeg["P4"].to_numpy()
    T3 = df_eeg["T3"].to_numpy()
    T4 = df_eeg["T4"].to_numpy()
    T5 = df_eeg["T5"].to_numpy()
    T6 = df_eeg["T6"].to_numpy()
    O1 = df_eeg["O1"].to_numpy()
    O2 = df_eeg["O2"].to_numpy()

    ll = np.stack(
        [
            compute_eeg(Fp1 - F7),
            compute_eeg(F7 - T3),
            compute_eeg(T3 - T5),
            compute_eeg(T5 - O1),
        ]
    )
    lp = np.stack(
        [
            compute_eeg(Fp1 - F3),
            compute_eeg(F3 - C3),
            compute_eeg(C3 - P3),
            compute_eeg(P3 - O1),
        ]
    )
    rp = np.stack(
        [
            compute_eeg(Fp2 - F4),
            compute_eeg(F4 - C4),
            compute_eeg(C4 - P4),
            compute_eeg(P4 - O2),
        ]
    )
    rl = np.stack(
        [
            compute_eeg(Fp2 - F8),
            compute_eeg(F8 - T4),
            compute_eeg(T4 - T6),
            compute_eeg(T6 - O2),
        ]
    )

    chain = np.stack([ll, lp, rp, rl])
    return chain


def compute_eeg_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_eeg_chain(df_eeg)
    return chain




## === cell 8
torch.manual_seed(0)
np.random.seed(0)

np.set_printoptions(formatter={"all": lambda x: f"{x:0.3f}"})


def _fix_length_last_axis(x: np.ndarray, target_len: int) -> np.ndarray:
    t = x.shape[-1]
    if t == target_len:
        return x
    if t > target_len:
        return x[..., :target_len]
    pad = target_len - t
    return np.pad(
        x, [(0, 0)] * (x.ndim - 1) + [(0, pad)], mode="constant", constant_values=0
    )


def _row_to_eeg_id(df_row: pl.DataFrame) -> int:
    if isinstance(df_row, pl.DataFrame):
        return int(df_row["eeg_id"][0])
    return int(df_row["eeg_id"])


def _row_to_spec_id(df_row: pl.DataFrame) -> int:
    if isinstance(df_row, pl.DataFrame):
        return int(df_row["spectrogram_id"][0])
    return int(df_row["spectrogram_id"])


def _center_crop_or_pad_2d(x: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = x.shape
    if h > target_h:
        top = (h - target_h) // 2
        x = x[top : top + target_h, :]
    if w > target_w:
        left = (w - target_w) // 2
        x = x[:, left : left + target_w]
    h, w = x.shape
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h or pad_w:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        x = np.pad(x, [(pad_top, pad_bottom), (pad_left, pad_right)], mode="constant")
    return x


def _load_test_spectrogram(spec_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Keep: log1p + per-sample global z-score to stabilize scale.
    Return sp2d as (F, Ncols) and original numeric column names for region grouping.
    """
    df = pl.read_parquet(spec_path).fill_null(0)

    num_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric()]
    if len(num_cols) == 0:
        raise ValueError("No numeric columns in spectrogram parquet.")

    x = df.select(num_cols).to_numpy()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    x = _center_crop_or_pad_2d(x, 256, x.shape[1])

    x = np.log1p(np.maximum(x, 0.0)).astype(np.float32)

    mu = float(x.mean())
    sig = float(x.std()) + 1e-6
    x = (x - mu) / sig

    return x, num_cols


def _spec2d_to_eeglike_16xt(
    sp2d: np.ndarray, colnames: List[str], target_t: int = 2500
) -> np.ndarray:
    """
    Keep same 16-channel construction (4 regions × 4 frequency bands), then interpolate time to target_t.
    """
    sp2d = np.asarray(sp2d, dtype=np.float32)
    sp2d = np.nan_to_num(sp2d, nan=0.0, posinf=0.0, neginf=0.0)

    F, N = sp2d.shape  # (freq_bins, time_bins/columns)

    regions = ["LL", "LP", "RP", "RL"]
    region_cols: Dict[str, List[int]] = {r: [] for r in regions}
    for j, c in enumerate(colnames):
        cu = str(c).upper()
        for r in regions:
            if cu.endswith(f"_{r}") or cu.endswith(r) or f"_{r}_" in cu:
                region_cols[r].append(j)
                break

    if any(len(region_cols[r]) == 0 for r in regions):
        edges = np.linspace(0, N, 4 + 1).astype(int)
        for i, r in enumerate(regions):
            region_cols[r] = list(range(edges[i], edges[i + 1]))

    feats16 = []
    band_edges = np.linspace(0, F, 4 + 1).astype(int)
    for r in regions:
        cols_idx = region_cols[r]
        region_mat = sp2d[:, cols_idx]  # (F, Tr)
        for b in range(4):
            fa, fb = band_edges[b], band_edges[b + 1]
            if fb <= fa:
                band_ts = np.zeros((region_mat.shape[1],), dtype=np.float32)
            else:
                band_ts = region_mat[fa:fb, :].mean(axis=0)
            feats16.append(band_ts.astype(np.float32))

    maxT = max(ch.shape[0] for ch in feats16)
    x = np.stack(
        [
            (
                ch
                if ch.shape[0] == maxT
                else np.pad(ch, (0, maxT - ch.shape[0]), mode="edge")
            )
            for ch in feats16
        ],
        axis=0,
    ).astype(
        np.float32
    )  # (16, maxT)

    T = x.shape[1]
    if T != target_t:
        old = np.linspace(0.0, 1.0, T, dtype=np.float32)
        new = np.linspace(0.0, 1.0, target_t, dtype=np.float32)
        x_rs = np.empty((16, target_t), dtype=np.float32)
        for c in range(16):
            x_rs[c] = np.interp(new, old, x[c]).astype(np.float32)
        x = x_rs
    return x


@torch.no_grad()
def _model_predict_robust(model: nn.Module, xt_3d: torch.Tensor) -> torch.Tensor:
    """
    Try (B, C, T) first, fall back to (B, C, 1, T) if needed.
    """
    try:
        return model.knn_predict(xt_3d)
    except Exception:
        return model.knn_predict(xt_3d.unsqueeze(2))


def _apply_temperature(p: np.ndarray, temperature: float) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32).reshape(-1)
    if temperature is None or temperature <= 0:
        return p
    p = np.clip(p, 1e-12, 1.0)
    logp = np.log(p)
    logp = logp / float(temperature)
    logp = logp - np.max(logp)
    p2 = np.exp(logp).astype(np.float32)
    p2 = p2 / (p2.sum() + 1e-12)
    return p2


def _finalize_probs(
    p: np.ndarray,
    eps: float = 1e-6,
    prior_mix: float = 0.01,
    temperature: float = 1.05,
) -> np.ndarray:
    """
    Change (metric-aligned, minimal): keep only mild softening and tiny prior to reduce KL blow-ups,
    but avoid over-smoothing (lower temperature than before).
    """
    p = np.asarray(p, dtype=np.float32).reshape(-1)

    if p.shape[0] != 6 or (not np.isfinite(p).all()):
        return np.ones(6, dtype=np.float32) / 6.0

    s = float(p.sum())
    if not np.isfinite(s) or s <= 0:
        return np.ones(6, dtype=np.float32) / 6.0

    p = p / s
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()

    p = _apply_temperature(p, temperature=temperature)

    prior_mix = float(np.clip(prior_mix, 0.0, 0.25))
    if prior_mix > 0:
        u = np.ones(6, dtype=np.float32) / 6.0
        p = (1.0 - prior_mix) * p + prior_mix * u

    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return p


@torch.no_grad()
def gen_ensemble_pred(models, df_row: pl.DataFrame) -> np.ndarray:
    eeg_id = _row_to_eeg_id(df_row)
    spec_id = _row_to_spec_id(df_row)

    spec_path = os.path.join(SPEC_DIR, f"{spec_id}.parquet")
    if os.path.exists(spec_path):
        try:
            sp2d, colnames = _load_test_spectrogram(spec_path)  # (F, Ncols)
            x16 = _spec2d_to_eeglike_16xt(sp2d, colnames, target_t=2500)  # (16,2500)

            xt = torch.tensor(x16, device=device).unsqueeze(0)  # (1,16,2500)

            preds = []
            for model in models:
                model.eval()
                pred = _model_predict_robust(model, xt)
                pred = pred.detach().float().cpu().numpy().reshape(-1)
                preds.append(pred)

            preds = np.mean(preds, axis=0).astype(np.float32)
            return _finalize_probs(preds, eps=1e-6)
        except Exception:
            pass

    eeg_path = os.path.join(EEG_DIR, f"{eeg_id}.parquet")
    if not os.path.exists(eeg_path):
        return np.ones(6, dtype=np.float32) / 6.0

    x = compute_eeg_from_file(eeg_path)  # (4,4,T)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = _fix_length_last_axis(x, 2500)

    x = x.reshape(16, 2500)

    xt = torch.tensor(x, device=device).unsqueeze(0)  # (1,16,2500)

    preds = []
    for model in models:
        model.eval()
        pred = _model_predict_robust(model, xt)
        pred = pred.detach().float().cpu().numpy().reshape(-1)
        preds.append(pred)

    preds = np.mean(preds, axis=0).astype(np.float32)
    return _finalize_probs(preds, eps=1e-6)




## === cell 9
LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]




## === cell 10
def _find_ckpt_in_fold(fold_path: str) -> Optional[str]:
    """
    Many published Kaggle model datasets nest checkpoints in subfolders.
    Search recursively while keeping the same loading semantics.
    """
    if not os.path.isdir(fold_path):
        return None
    patterns = [
        "**/*.pth",
        "**/*.pt",
        "**/*.bin",
        "**/*.ckpt",
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(os.path.join(fold_path, pat), recursive=True))
    candidates = [p for p in candidates if os.path.isfile(p)]
    preferred = []
    for name in ["best", "last", "final", "checkpoint"]:
        preferred.extend([p for p in candidates if name in os.path.basename(p).lower()])
    ordered = preferred + [p for p in candidates if p not in preferred]
    return ordered[0] if len(ordered) else None


fold_dirs = ["/kaggle/input/hms-models/knn_0/*"]

models = []
found_any = False
loaded_any_ckpt = False

for fold_dir in fold_dirs:
    for fold_path in glob.glob(fold_dir):
        found_any = True
        model = CurrModel()
        if hasattr(model, "from_pretrained"):
            model.from_pretrained(fold_path)

        ckpt_path = _find_ckpt_in_fold(fold_path)
        if ckpt_path is not None:
            try:
                model = load_model(ckpt_path, model)
                loaded_any_ckpt = True
            except Exception:
                pass

        model = model.to(device)
        models.append(model)

if not found_any:
    model = CurrModel().to(device)
    models = [model]

print("n_models:", len(models), "loaded_any_ckpt:", loaded_any_ckpt)



## === cell 11
from tqdm.auto import tqdm

preds_final = []
for i in tqdm(range(len(df_test))):
    df_row = df_test.slice(i, 1)
    pred = gen_ensemble_pred(models, df_row)
    preds_final.append(pred)

preds_final = np.asarray(preds_final, dtype=np.float32)
print("preds_final shape:", preds_final.shape)



## === cell 12
if (
    preds_final.ndim != 2
    or preds_final.shape[1] != len(LABELS)
    or preds_final.shape[0] != len(df_test)
):
    raise ValueError(
        f"Invalid preds_final shape {preds_final.shape}, expected ({len(df_test)}, {len(LABELS)})"
    )

df_sub = pd.DataFrame({"eeg_id": df_test["eeg_id"].to_list()})
df_sub[LABELS] = preds_final

arr = df_sub[LABELS].to_numpy(dtype=np.float64)
arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
arr = np.clip(arr, 1e-12, 1.0)
arr = arr / arr.sum(axis=1, keepdims=True)
df_sub[LABELS] = arr.astype(np.float32)

df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
df_sub.head()
