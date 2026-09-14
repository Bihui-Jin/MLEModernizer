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

0.5993002550597261

# 6. Current score

1.41496

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I fix the file‑loading error, simplify the pipeline by skipping unnecessary heavy data loading, and add a fallback baseline prediction that uses the average vote distribution from the training data. This ensures a valid `submission.csv` is created without crashing, and the probabilities sum to 1 for each row, moving the solution toward the target score.'
- What this solution (achieved 1.40995) has done: 'I fixed the `fillna` error by using a Series for the overall‑average values, ensured the prediction array always has shape (N_test, 6), and added a tiny blending step that mixes the patient‑average baseline with a uniform distribution. This simple calibration (chosen by a cheap grid‑search on the training set) lowers the KL‑divergence without altering the core model logic, and guarantees a valid `submission.csv` with probabilities that sum to 1.'
- What this solution (achieved 1.39779) has done: 'I replace the uniform‑distribution blending with a blend that uses the overall class‑frequency vector (computed from the training data) instead of a plain uniform vector. This small calibration often reduces KL loss because the overall class distribution is a better prior than a uniform one, and it keeps the same overall pipeline and core logic unchanged. The code now searches for the best blending weight on the training set using this overall‑average prior and then applies the same weight to the test predictions, finally renormalising so each row sums to 1.'
- What this solution (achieved 1.24697) has done: 'I add a simple per‑patient weighting that leans on the patient‑specific average when that patient appears enough times in the training set, otherwise falls back to the overall class distribution. This modest calibration is expected to lower the KL‑divergence (bringing the score closer to the target) while keeping the original baseline logic unchanged. I also remove the final global blending step that could overwrite the new weighted predictions.'
- What this solution (achieved 1.39779) has done: 'I apply the α blending factor that was calibrated on the training set to the test‑time predictions. After the per‑patient weighted average is computed, I blend it with the overall class distribution using the best α, clip for numerical stability and renormalize so each row sums to 1. This small calibration is expected to lower the KL divergence and move the score closer to the target without altering the core model logic.'
- What this solution (achieved 1.15565) has done: 'I adjust the patient‑weighting scheme to give stronger influence to patient‑specific averages (using the mean count as a normalizer) and enforce a modest blending factor α (≥ 0.2) that was previously allowed to become 0. These small calibrations keep the original baseline logic but are expected to lower the KL‑divergence, moving the score toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 1.39779) has done: 'The adjustment reduces the enforced blending floor and computes a more discriminative patient‑weighting (using the maximum patient count instead of the mean). The optimal α is now taken directly from the training‑set search without a hard lower bound, giving a calibrated prediction that stays valid (rows sum to 1) while moving the KL‑divergence closer to the target.'
- What this solution (achieved 1.39779) has done: 'I fine‑tune the calibration step that blends the patient‑specific averages with the overall class distribution. By using a finer search for the optimal blending factor α (step 0.01 instead of 0.05) and capping the patient‑weight at 0.8 we obtain a slightly better calibrated probability set, which should lower the KL‑divergence and move the score closer to the target while keeping the original pipeline unchanged.'
- What this solution (achieved 1.39779) has done: 'I adjust the patient‑weighting step in the fallback baseline: instead of capping the weight at 0.8, I use the full raw weight (patient count / max count). This lets the model rely more on patient‑specific averages, which should lower the KL‑divergence and move the score closer to the target while keeping the core logic unchanged.'
- What this solution (achieved 1.39779) has done: 'I add a small calibration search that jointly optimises the patient‑count smoothing k and the blending factor α on the training set, then apply the best‑found parameters to the test predictions. This only tweaks the fallback baseline (no model changes) and is expected to lower the KL‑divergence, moving the score closer to the target while still producing a valid submission.csv​.'
- What this solution (achieved 1.41937) has done: 'I improve the fallback baseline by using vote‑count based class distributions per patient and per eeg_id instead of simple probability averages, and I apply these more informative priors before the calibrated blending step. This tighter prior typically lowers KL‑divergence, moving the score closer to the target while keeping the overall pipeline unchanged. The rest of the script remains the same, and the final submission file is still written to /kaggle/working/submission.csv.'
- What this solution (achieved 1.41706) has done: 'I keep the overall fallback‑average pipeline unchanged but add a tiny temperature‑scaling step after the probabilities are renormalised. Raising the predicted probabilities to a power < 1 (here 0.95) makes them slightly smoother, which often lowers KL‑divergence when the raw averages are a bit over‑confident, moving the score closer to the target while preserving the original logic.'
- What this solution (achieved 1.41496) has done: 'I blend the EEG‑specific priors with the overall class distribution (using the already‑tuned α) instead of using them raw, and I smooth the final probabilities a bit more by changing the temperature from 0.95 to 0.90. This keeps the original fallback logic while making the predictions less confident, which should lower the KL‑divergence and move the score nearer the target.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import timm
from tqdm import tqdm




## === cell 1
import albumentations as A
import gc
import librosa
import matplotlib.pyplot as plt
import pywt
import random
import time
from albumentations.pytorch import ToTensorV2
from glob import glob
from typing import Dict, List

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s")




## === cell 2
class config:
    model = "resnet34d"
    epoch = 10
    lr = 1e-3
    batchsize = 32
    splits = 5
    momentum = 0.9
    MAX_GRAD_NORM = 1e7
    WEIGHT_DECAY = 0.01
    device = "cpu"
    FOLDS = 5
    AMP = True


class paths:
    preloadedeeg = "/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy"
    train_eeg_dir = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs"
    train_spec_dir = (
        "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms"
    )
    train_csv = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    test_csv = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    test_eeg = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs"
    test_spec = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms"
    )
    out = "/kaggle/working/"




## === cell 3
USE_WAVELET = None
NAMES = ["LL", "LP", "RP", "RR"]
FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
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




## === cell 4
test_df = pd.read_csv(paths.test_csv)
print(f"Test dataframe shape: {test_df.shape}")




## === cell 5
all_eegs = {}
all_spectrograms = {}




## === cell 6
class CustomDataset(Dataset):
    def __init__(
        self,
        traindf: pd.DataFrame,
        config,
        mode: str = "train",
        specs: dict[int, np.ndarray] = None,
        eegs: dict[int, np.ndarray] = None,
    ):
        self.traindf = traindf
        self.specs = specs or {}
        self.eegs = eegs or {}
        self.mode = mode

    def __len__(self):
        return len(self.traindf)

    def __getitem__(self, idx):
        row = self.traindf.iloc[idx]
        X = torch.zeros((3, 128, 256), dtype=torch.float32)  # dummy tensor
        y = torch.zeros(6, dtype=torch.float32)
        return {"data": X, "target": y}




## === cell 7
class Custommodel(nn.Module):
    def __init__(self, cfg, numclass: int = 6):
        super(Custommodel, self).__init__()
        self.model = timm.create_model(
            cfg.model,
            pretrained=False,
            drop_rate=0.1,
            drop_path_rate=0.2,
        )
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.customlayer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, numclass),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.customlayer(x)
        return x




## === cell 8
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    for batch in test_loader:
        x = batch["data"].to(device)
        with torch.no_grad():
            ypred = model(x)
        ypred = softmax(ypred)
        preds.append(ypred.cpu().numpy())
    return {"predictions": np.concatenate(preds, axis=0)}


model_dir = "/kaggle/input/resnet34d2"
if os.path.isdir(model_dir) and len(os.listdir(model_dir)) > 0:
    ensemble_preds = []
    for fname in os.listdir(model_dir):
        ckpt = torch.load(os.path.join(model_dir, fname), map_location="cpu")
        model = Custommodel(config)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        test_dataset = CustomDataset(test_df, config, mode="test")
        test_loader = DataLoader(test_dataset, batch_size=config.batchsize)
        pred_dict = inference_function(test_loader, model, device)
        ensemble_preds.append(pred_dict["predictions"])
    predictions = np.mean(ensemble_preds, axis=0)
else:
    train_df = pd.read_csv(paths.train_csv)

    vote_cols = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

    overall_counts = train_df[vote_cols].sum()
    overall_avg_vals = (overall_counts / overall_counts.sum()).values  # (6,)

    patient_votes_sum = train_df.groupby("patient_id")[vote_cols].sum()
    patient_avg = patient_votes_sum.div(patient_votes_sum.sum(axis=1), axis=0)

    eeg_votes_sum = train_df.groupby("eeg_id")[vote_cols].sum()
    eeg_avg = eeg_votes_sum.div(eeg_votes_sum.sum(axis=1), axis=0)

    patient_counts = train_df.groupby("patient_id").size()
    max_count = patient_counts.max()

    eps = 1e-12

    def kl_loss(p_true, p_pred):
        return np.mean(np.sum(p_true * np.log(p_true / p_pred), axis=1))

    k_candidates = [0, 1, 2, 5, 10, 20]
    best_score = np.inf
    best_k = 0
    best_alpha = 0.0

    patient_avg_filled_train = (
        patient_avg.reindex(train_df["patient_id"])
        .fillna(pd.Series(overall_avg_vals, index=vote_cols))
        .values
    )
    prob_rows = train_df[vote_cols].div(train_df[vote_cols].sum(axis=1), axis=0).values

    for k in k_candidates:
        weight_train = (
            patient_counts.reindex(train_df["patient_id"]).fillna(0) + k
        ) / (
            max_count + k
        )  # (N_train,)
        weight_train = weight_train.values.reshape(-1, 1)  # (N_train,1)

        pred_train = (
            weight_train * patient_avg_filled_train
            + (1.0 - weight_train) * overall_avg_vals
        )
        pred_train = np.clip(pred_train, eps, 1)

        for alpha in np.arange(0.0, 1.0001, 0.01):
            blended = alpha * pred_train + (1 - alpha) * overall_avg_vals
            blended = np.clip(blended, eps, 1)
            score = kl_loss(prob_rows, blended)
            if score < best_score:
                best_score = score
                best_k = k
                best_alpha = alpha

    patient_avg_filled_test = (
        patient_avg.reindex(test_df["patient_id"])
        .fillna(pd.Series(overall_avg_vals, index=vote_cols))
        .values
    )
    patient_counts_test = patient_counts.reindex(test_df["patient_id"]).fillna(0).values
    weight_test = (patient_counts_test + best_k) / (max_count + best_k)
    weight_test = weight_test.reshape(-1, 1)  # (N_test,1)

    eeg_avg_filled_test = (
        eeg_avg.reindex(test_df["eeg_id"])
        .fillna(pd.Series(overall_avg_vals, index=vote_cols))
        .values
    )
    has_eeg_prior = ~np.isnan(eeg_avg_filled_test).any(axis=1)

    predictions = np.empty((len(test_df), 6), dtype=np.float32)

    if has_eeg_prior.any():
        blended_eeg = (
            best_alpha * eeg_avg_filled_test[has_eeg_prior]
            + (1 - best_alpha) * overall_avg_vals
        )
        blended_eeg = np.clip(blended_eeg, eps, 1)
        predictions[has_eeg_prior] = blended_eeg

    idx_rest = np.where(~has_eeg_prior)[0]
    if idx_rest.size > 0:
        pred_rest = (
            weight_test[idx_rest] * patient_avg_filled_test[idx_rest]
            + (1.0 - weight_test[idx_rest]) * overall_avg_vals
        )
        pred_rest = best_alpha * pred_rest + (1 - best_alpha) * overall_avg_vals
        predictions[idx_rest] = pred_rest

    predictions = np.clip(predictions, eps, 1)
    predictions = predictions / predictions.sum(axis=1, keepdims=True)

    temperature = (
        0.90  # increased smoothing (lower temperature) to reduce over‑confidence
    )
    predictions = np.power(predictions, temperature)
    predictions = predictions / predictions.sum(axis=1, keepdims=True)




## === cell 9
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
submission = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
submission[TARGETS] = predictions
submission_path = os.path.join(paths.out, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission shape: {submission.shape}")
print(f"Saved to {submission_path}")
