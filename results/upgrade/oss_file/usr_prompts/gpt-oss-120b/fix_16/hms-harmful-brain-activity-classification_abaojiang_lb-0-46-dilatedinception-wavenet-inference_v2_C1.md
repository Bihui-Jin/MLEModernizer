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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
PyYAML==6.0.3
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

0.5976041136518271

# 6. Current score

0.97422

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The fix adds handling for missing model files by loading no models safely and falling back to a uniform probability prediction, preventing the `UnboundLocalError`. It also guards the inference function to return a proper tensor when no models are loaded, ensuring the submission dataframe receives a correctly‑shaped array and the CSV is written without errors.'
- What this solution (achieved 1.41937) has done: 'I compute a class‑frequency prior from the training data and use it as a fallback prediction when no pretrained models are found. This replaces the uniform‑distribution fallback with a more informative prior, which should lower the KL‑divergence (the metric is lower‑is‑better) and move the score toward the target. The change only adds a few lines for loading the training CSV, computing the normalized vote distribution, and using it in `_infer` when `models` is empty.'
- What this solution (achieved 1.68479) has done: 'The fixes add all required imports, rename the first cell to start at 1, and ensure every referenced class/function (Path, pandas, numpy, torch, tqdm, Dataset, DataLoader, etc.) is defined before use. No core‑logic changes are made; the fallback‑to‑training‑set and patient priors remain, so the model’s predictions (or priors when no model is found) are correctly turned into a CSV submission whose rows sum to 1, moving the KL‑divergence score toward the target.'
- What this solution (achieved 1.68479) has done: 'Implemented a missing import for `Type` which caused a `NameError` during model class definitions. This fix allows the model architecture to be defined successfully and enables the rest of the pipeline (including fallback prior‑based predictions) to run, producing a valid submission CSV.'
- What this solution (achieved 0.81597) has done: 'We blend the patient‑specific prior with the overall training prior instead of using the patient prior alone. This simple calibration keeps the original architecture and inference flow unchanged while giving probabilities that are less extreme, which should lower the KL‑divergence and move the score toward the target.'
- What this solution (achieved 0.78698) has done: 'We give the patient‑specific prior more weight (increase `blend_alpha`), because the patient prior is usually more informative than the global prior and this simple change has been shown to lower the KL‑divergence without altering the model architecture or training flow.'
- What this solution (achieved 0.85517) has done: 'I lower the reliance on the patient‑specific prior by reducing the blending factor (blend_alpha) from 0.95 to 0.6. This makes the overall prediction closer to the global train prior, which empirically moves the KL‑divergence toward the target lower score while keeping all core logic unchanged.'
- What this solution (achieved 0.79593) has done: 'I adjust the blending factor to give more weight to the patient‑specific prior (which has proven to lower KL‑divergence) and add a tiny smoothing step so the probability vectors are not overly sharp. These small tweaks keep the original architecture and inference flow unchanged while moving the validation score closer to the target.'
- What this solution (achieved 0.82997) has done: 'I keep the overall architecture unchanged and only adjust the prior‑blending smoothing step, which directly influences the KL‑divergence score. By increasing the additive epsilon from 0.02 to 0.05 we make the probability vectors slightly less confident, which typically lowers KL loss when only prior‑based predictions are used. The rest of the pipeline (data loading, transformations, model loading) remains identical, guaranteeing a valid submission.csv output.'
- What this solution (achieved 0.77739) has done: 'The fix corrects the typo in the EEG transformer (`apply_butter_lowpass_filter` and `_butter_lowpass_filter`) and adds a wrapper for the low‑pass filter method. It also slightly adjusts the prior‑blending hyper‑parameters (higher `blend_alpha` and smaller epsilon) to move the KL‑divergence score toward the target while keeping the core model unchanged.'
- What this solution (achieved 0.93917) has done: 'I slightly reduce the reliance on the patient‑specific prior and increase the smoothing factor when only prior‑based predictions are used. Lowering `blend_alpha` makes the global training prior contribute more, and a larger epsilon smooths the probability vectors, both of which tend to lower the KL‑divergence and move the score closer to the target while keeping the original model architecture unchanged.'
- What this solution (achieved 0.97422) has done: 'The fix adds all missing imports and defines a lightweight configuration, loads the train and test metadata, computes global and patient‑specific priors, and then performs inference using only these priors (no neural‑network models are required).  The blending weight and smoothing ε are set to values that keep the probability rows well‑calibrated and close to the target KL‑divergence while preserving the original pipeline structure.  Finally, a correctly formatted `submission.csv` is written.'

# 9. Code solution

## === cell 0
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from scipy.signal import butter, lfilter


class CFG:
    base_path: Path = Path("/kaggle/input/hms-harmful-brain-activity-classification")
    feats: List[str] = []
    cast_eegs: bool = False
    device: torch.device = torch.device("cpu")
    model_path: Path = base_path / "models"
    batch_size: int = 64
    dataset: Dict[str, Any] = {}


EEG_PTS: int = 200 * 50
N_CLASSES: int = 6
TGT_VOTE_COLS: List[str] = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

DATA_PATH = CFG.base_path

test = pd.read_csv(DATA_PATH / "test.csv")
print(f"Test data shape | {test.shape}")

train = pd.read_csv(DATA_PATH / "train.csv")
train_votes_sum = train[TGT_VOTE_COLS].sum()
TRAIN_PRIOR = (train_votes_sum / train_votes_sum.sum()).values.astype(np.float32)

patient_prior_df = (
    train.groupby("patient_id")[TGT_VOTE_COLS]
    .sum()
    .apply(lambda row: row / row.sum(), axis=1)
)
PATIENT_PRIOR = {
    pid: row.values.astype(np.float32) for pid, row in patient_prior_df.iterrows()
}



## === cell 1
models: List[nn.Module] = []
if CFG.model_path.exists():
    pth_files = sorted(CFG.model_path.glob("*.pth"))
    if pth_files:
        for fold, file in enumerate(pth_files):
            print(f"Loading model from {file}...")
            model = torch.load(file, map_location=CFG.device)  # placeholder loading
            models.append(model)
    else:
        print("No .pth files found; inference will use prior‑based predictions.")
else:
    print("Model path does not exist; inference will use prior‑based predictions.")



## === cell 2
if not models:
    blend_alpha: float = 0.50  # weight for patient‑specific prior
    eps: float = 0.05  # smoothing constant

    patient_ids = test["patient_id"].values
    patient_prior_matrix = np.stack(
        [PATIENT_PRIOR.get(pid, TRAIN_PRIOR) for pid in patient_ids]
    )

    y_preds = blend_alpha * patient_prior_matrix + (1.0 - blend_alpha) * TRAIN_PRIOR

    y_preds = y_preds / y_preds.sum(axis=1, keepdims=True)

    y_preds = (y_preds + eps) / (1.0 + eps * N_CLASSES)

    print(f"Row 0 sum after blending & smoothing: {y_preds[0].sum():.6f}")
else:
    test_data = {"meta": test, "eeg": {}}  # placeholder – not used
    test_loader = DataLoader(
        EEGDataset(test_data, "test", **CFG.dataset),
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=0,
    )

    @torch.no_grad()
    def _infer(
        inputs: Dict[str, torch.Tensor], models: List[nn.Module]
    ) -> torch.Tensor:
        batch_size = inputs["x"].shape[0]
        prior_tensor = torch.tensor(TRAIN_PRIOR, device=CFG.device, dtype=torch.float32)
        if not models:
            return prior_tensor.unsqueeze(0).repeat(batch_size, 1)

        n_models = len(models)
        agg = None
        for i, model in enumerate(models):
            model.eval()
            out = F.softmax(model(inputs), dim=1) / n_models
            agg = out if agg is None else agg + out
        return agg

    preds = []
    for batch in test_loader:
        batch["x"] = batch["x"].to(CFG.device)
        preds.append(_infer(batch, models).cpu().numpy())
    y_preds = np.vstack(preds)



## === cell 3
submission = pd.DataFrame({"eeg_id": test["eeg_id"].values})
submission[TGT_VOTE_COLS] = y_preds
submission_path = Path("submission.csv")
submission.to_csv(submission_path, index=False)

print("===== Submission Preview =====")
print(submission.head())
print(f"Submission saved to {submission_path.resolve()}")
