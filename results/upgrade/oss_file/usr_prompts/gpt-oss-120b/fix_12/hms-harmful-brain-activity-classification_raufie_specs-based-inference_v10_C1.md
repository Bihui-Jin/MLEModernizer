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

0.6620293823888312

# 6. Current score

1.42461

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The fix adds a safe fallback when the pretrained weights are missing: it computes class‑level prior probabilities from the training votes and uses those as constant predictions for every test row, guaranteeing a correctly‑shaped CSV submission. It also guards the weight loading with a try‑except block, removes the heavy‑weight loading of all spectrogram files (which isn’t needed for the fallback), and ensures the submission file is written with the proper columns and sum‑to‑one probabilities.'
- What this solution (achieved 1.41937) has done: 'I fixed the Na N‑fill logic that caused a `ValueError` and ensured the fallback priors are always defined. I also added a tiny safety normalization after computing predictions so each row sums exactly to 1, guaranteeing a valid submission CSV.'
- What this solution (achieved 1.45413) has done: 'I keep the existing fallback‑prior logic but add a modest temperature‑scaling step to sharpen the probability vectors (raising them to a power > 1 and renormalising). This small calibration can lower the KL divergence without changing the core model or data handling, moving the score closer to the target while preserving a valid CSV submission.'
- What this solution (achieved 1.41496) has done: 'I keep the overall fallback‑prior approach but make the predictions a blend of the per‑eeg prior and the overall class prior (so very rare or missing eeg_id rows are smoothed) and replace the sharpening temperature > 1 with a slight softening (temperature < 1). This small calibration keeps the core logic unchanged while moving the KL‑divergence lower, bringing the score nearer to the target.'
- What this solution (achieved 1.43069) has done: 'I reduce the reliance on noisy per‑eeg priors by lowering the blending weight (blend_alpha) and switch the temperature scaling from softening (<1) to a modest sharpening (>1). This keeps the fallback‑prior logic unchanged while likely producing probability vectors that better match the true test distribution, moving the KL‑divergence closer to the target lower score.'
- What this solution (achieved 1.41136) has done: 'I lower the reliance on noisy per‑eeg priors by reducing `blend_alpha` to 0.10 and soften the probability vectors with a temperature < 1 (set `temperature = 0.80`). This calibration keeps the original fallback logic unchanged while aiming to produce smoother, better‑aligned predictions, thereby decreasing the KL‑divergence toward the target score.'
- What this solution (achieved 1.43069) has done: 'I adjust the fallback blending to rely more on the per‑eeg priors (increase `blend_alpha`) and apply a modest sharpening temperature (> 1) so the probability vectors are less uniform, which should lower the KL‑divergence toward the target. I also replace the forced `FileNotFoundError` with a genuine existence check for the pretrained weights, keeping the same fallback logic if the file is absent.'
- What this solution (achieved 1.41937) has done: 'I lower the reliance on the noisy per‑eeg priors by setting the blending weight `blend_alpha` to 0, so the prediction for every test row is the global class prior. I also remove the temperature‑sharpening step (set `temperature = 1.0`), keeping the probabilities exactly the class prior and ensuring they sum to 1. These minimal adjustments keep the original fallback logic unchanged while moving the KL‑divergence toward the lower target score.'
- What this solution (achieved 1.41101) has done: 'I keep the overall fallback‑prior logic but give the model a stronger per‑eeg signal (increase `blend_alpha`), smooth the resulting probabilities with a temperature < 1 and add a tiny uniform smoothing term. These lightweight calibrations stay within the original design yet are expected to lower the KL‑divergence, moving the score closer to the target while still producing a valid CSV submission.'
- What this solution (achieved 1.42461) has done: 'I adjust the fallback calibration to rely a bit less on the noisy per‑eeg priors, remove the uniform smoothing, and apply a slight temperature‐sharpening (> 1). These small changes keep the overall fallback logic intact while nudging the predicted distributions toward the true class frequencies, which should lower the KL‑divergence and move the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import random
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from tqdm import tqdm
from typing import Dict




## === cell 1
class config:
    BATCH_SIZE = 64
    MODEL = "tf_efficientnet_b4"
    NUM_WORKERS = 0
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False


class paths:
    MODEL_WEIGHTS = (
        "/kaggle/input/hba-efficientnet-weights/tf_efficientnet_b4_epoch_7.pth"
    )
    OUTPUT_DIR = "/kaggle/working/"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TRAIN_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")




## === cell 2
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


seed_everything(config.SEED)

train_df = pd.read_csv(paths.TRAIN_CSV)
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
class_counts = train_df[TARGETS].sum().values.astype(np.float64)
class_prior = class_counts / class_counts.sum()
print("Class prior probabilities:", class_prior)

eeg_class_counts = train_df.groupby("eeg_id")[TARGETS].sum()
fill_vals = pd.Series(class_prior, index=TARGETS)
eeg_prior = eeg_class_counts.div(eeg_class_counts.sum(axis=1), axis=0).fillna(fill_vals)
print("Per‑eeg_id prior computed for", len(eeg_prior), "ids")




## === cell 3
test_df = pd.read_csv(paths.TEST_CSV)
print(f"Test dataframe shape: {test_df.shape}")




## === cell 4
class CustomModel(nn.Module):
    def __init__(self, cfg, num_classes: int = 6):
        super(CustomModel, self).__init__()
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True
        self.model = timm.create_model(cfg.MODEL, pretrained=False)
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, num_classes),
        )

    def __reshape_input(self, x):
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


predictions = None
try:
    if os.path.exists(paths.MODEL_WEIGHTS):
        model = CustomModel(config)
        checkpoint = torch.load(paths.MODEL_WEIGHTS, map_location=device)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()
        raise FileNotFoundError  # Force fallback when inference code is not implemented
    else:
        raise FileNotFoundError
except FileNotFoundError:
    blend_alpha = 0.6  # weight of per‑eeg prior (0 ≤ α ≤ 1)
    epsilon = 0.0  # uniform smoothing factor
    temperature = 1.1  # >1 to sharpen distribution

    pred_list = [
        eeg_prior.loc[eid].values if eid in eeg_prior.index else class_prior
        for eid in test_df["eeg_id"].values
    ]
    per_eeg_preds = np.vstack(pred_list)  # shape (n_test, 6)

    blended = blend_alpha * per_eeg_preds + (1 - blend_alpha) * class_prior

    scaled = blended**temperature

    predictions = scaled

except Exception as e:
    print(f"Unexpected error ({e}) – using global prior predictions.")
    predictions = np.tile(class_prior, (len(test_df), 1))

predictions = np.clip(predictions, 1e-9, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)




## === cell 5
submission = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
submission[TARGETS] = predictions
submission_path = os.path.join(paths.OUTPUT_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(f"Submission shape: {submission.shape}")
