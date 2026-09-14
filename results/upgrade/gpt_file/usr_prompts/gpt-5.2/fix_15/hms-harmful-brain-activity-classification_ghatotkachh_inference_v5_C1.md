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

0.6376157212923325

# 6. Current score

0.85484

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the missing checkpoint directory error by removing the dependency on `/kaggle/input/10ep5foldresent18` and instead generating a valid prediction using the empirical class-prior from `train.csv` (a stable baseline that produces a correct submission). I also fix multiple runtime issues in the current pipeline (undefined `targets`, invalid spectrogram indexing with `min/max`, huge debug `print`, and shape mismatch when writing the submission). Finally, I enforce strict probability normalization (row-sum = 1) and output the submission with the required `.csv` suffix and correct columns so it always yields a valid Kaggle submission.'
- What this solution (achieved 1.54604) has done: 'Your current solution is a pure class-prior baseline, which is stable but leaves a large gap to the target KL score. To move the score down toward your target without changing the “core approach” (still no model training/inference), I make one minimal, legitimate improvement: compute the prior at the *eeg_id* level (averaging per recording) instead of at the row level, so overlapping train windows don’t overweight certain recordings/classes. Then I add a tiny amount of symmetric smoothing (Dirichlet/Laplace) before normalization to better match the competition’s probability/KL sensitivity while keeping the same semantics and guaranteeing valid probabilities. The pipeline still run fast, produce a valid `submission.csv`, and keep all I/O paths unchanged.'
- What this solution (achieved 1.20859) has done: 'Your current baseline is stable but too “generic” because it predicts the same prior for every test EEG; to move the KL score down toward your target without changing the core approach (still a non-model baseline), I compute a patient-level empirical prior and blend it with the global prior. This is a minimal, legitimate improvement because test.csv includes `patient_id`, and train labels vary by patient; using a patient prior often reduces KL versus a single global prior. I keep the existing eeg_id-level de-overlap and smoothing, and add a small interpolation weight plus a safe fallback to the global prior for unseen patients. The output remains a valid probability distribution per row and writes `submission.csv` to the same path.'
- What this solution (achieved 1.10136) has done: 'Your current approach is a stable non-model prior baseline, but it’s still far from the target KL, so the smallest legitimate improvement is to make the “personalization” more specific without changing the core logic (still just priors + blending + smoothing). I switch the patient prior from a mean-of-means to a properly weighted sum over eeg_id (so patients with many windows don’t distort the distribution), and I compute the global prior using the same weighted aggregation for consistency. Then I add a tiny per-patient shrinkage based on how much data that patient has (more shrink to global for low-support patients), which usually reduces KL while keeping predictions valid and stable. The submission writing, columns, and strict row-normalization remain unchanged.'
- What this solution (achieved 1.12691) has done: 'Your current prior+patient-blend baseline is stable but still above the target KL, so the smallest likely improvement is to make the prior conditional a bit more specific without changing the approach. I keep the same “vote-sum → smoothed prior → blend → normalize” core logic, but add a second, low-risk conditioning signal available in `test.csv`: `spectrogram_id`. We compute a spectrogram-level prior from train (aggregated at eeg_id to avoid overlap bias), then blend global + patient + spectrogram priors with the same shrinkage-by-support idea so low-support groups fall back to global. This usually reduces KL versus patient-only personalization while keeping outputs valid probabilities and runtime fast.'
- What this solution (achieved 1.22769) has done: 'Your current approach is a blended prior baseline; to move KL down toward the target with minimal risk, I keep the same logic but tune the blend weights and shrinkage so the more informative conditional prior (spectrogram_id) is trusted a bit more and patient prior a bit less (patients can be heterogeneous). I also slightly reduce the symmetric smoothing alpha so conditional priors are less “washed out,” which typically improves KL while still preventing zeros. Finally, I add a tiny final mixing with the global prior for safety/calibration (doesn’t change the core approach, just stabilizes extreme group priors). The script still runs fast and produces a valid `submission.csv` with strictly normalized probabilities.'
- What this solution (achieved 1.32351) has done: 'Your current score (1.22769, lower-is-better) is still far above the target (0.6376), so we should improve—but with minimal, low-risk changes that keep the same “blended priors + shrinkage + smoothing” core logic. The most likely issue is a mismatch between what you predict (probabilities over vote *counts*) and what the metric expects (KL vs normalized vote *distribution*), so I (1) compute priors using per-eeg normalized label distributions (each eeg_id contributes equally, rather than being weighted by number of annotators/votes) and (2) make patient/spec priors consistent with the same normalization. I keep your blending/shrinkage structure intact, only lightly retuning weights to avoid overconfident conditional priors, and keep strict row-normalization and submission format unchanged.'
- What this solution (achieved 1.22028) has done: 'We keep your blended-prior baseline exactly as-is in spirit (global + patient + spectrogram with support-based shrinkage + smoothing), but fix one likely source of poor KL: test spectrogram_ids often won’t match train spectrogram_ids (different IDs), so your spectrogram prior frequently falls back to global and adds noise when it does match by chance. The minimal improvement is to replace the spectrogram_id conditioning with an eeg-derived conditioning signal that exists in both train and test and is consistent: per-patient priors only, but strengthened via a small hierarchical “patient→global” shrinkage that depends on patient support, plus a tiny temperature calibration to reduce overconfident distributions (helps KL). This preserves your “no model” approach, keeps runtime tiny, guarantees valid probabilities, and is a small, low-risk change intended to move KL down toward the target.'
- What this solution (achieved 1.10299) has done: 'Your current baseline already has the right “blended priors + shrinkage + smoothing” core, but it likely underperforms because the priors are computed as *mean of per-eeg distributions*, which ignores confidence/annotator-count information that the KL metric is sensitive to. I keep the same approach, but compute global and patient priors from properly aggregated vote totals (Dirichlet-smoothed) at the eeg_id level, so eegs contribute proportionally to their label evidence while still avoiding window-overlap bias. Then I keep your existing shrinkage blend and temperature, only slightly retuning weights to rely a bit more on patient signal (since it becomes more reliable under the new aggregation) while keeping strong fallback to global. This is a minimal change intended to reduce KL (lower-is-better) toward your target without introducing any model training or new features.'
- What this solution (achieved 1.1743) has done: 'I keep your “blended priors + shrinkage + smoothing + temperature” approach exactly the same, but fix one likely KL issue: you’re training priors on raw vote totals, which overweights samples with more annotators and can miscalibrate probabilities for KL. I instead compute priors from per-eeg normalized vote distributions (each eeg_id contributes one distribution), then build patient priors as the mean of those eeg-level distributions, while keeping the same shrinkage blending in inference. This is a minimal semantic change (still just priors, no model) that typically improves calibration for KL and should move the score down toward your target. I keep the submission writing and strict normalization unchanged.'
- What this solution (achieved 1.12432) has done: 'Your current score (1.1743, lower-is-better) is still far above the target (0.6376), so we should improve—but with minimal changes that keep the same “global prior + patient prior + shrinkage + smoothing + temperature” core logic. The biggest low-risk gain is to compute patient priors with a Dirichlet-style aggregation (sum of eeg-level *distributions*) rather than a plain mean, which is more stable for KL and lets support-weighted patients be represented better without window-overlap bias. Then we make the shrinkage use the same “effective sample size” and lightly retune the patient weight/temperature to reduce overconfident predictions (KL is very sensitive to overconfidence). All I/O paths and submission formatting remain unchanged, and the script still runs fast and writes a valid `submission.csv`.'
- What this solution (achieved 0.87055) has done: 'I keep your “global prior + patient prior + shrinkage + smoothing + temperature” approach intact, but fix one core calibration issue: patient priors are currently built by summing *eeg-level distributions*, which effectively upweights EEGs with fewer annotator votes (noisier labels) equally to high-vote EEGs and can hurt KL. I instead compute patient priors by aggregating *eeg-level vote counts* (already de-overlapped at eeg_id), then convert to probabilities with the same Dirichlet smoothing—this is a minimal change and usually improves KL calibration. I also compute patient support using total votes (effective evidence) rather than just number of eeg_ids, and use that same evidence in shrinkage; everything else (weights, smoothing, temperature, I/O, normalization) remains the same. This should move the score down (lower-is-better) toward your target without changing the overall logic or runtime.'
- What this solution (achieved 0.85486) has done: 'Your current score (0.87055, lower-is-better) is still above the target (0.6376), so we should improve slightly while keeping the same “global prior + patient prior + shrinkage + smoothing + temperature” core logic. The most likely low-risk gain is better calibration for KL by (1) computing the global prior with the same vote-count aggregation (Dirichlet-smoothed) used for patients, instead of a mean of per-eeg distributions, and (2) reducing the temperature slightly (less flattening) so predictions are not overly uniform. I also make the final global mix a bit smaller to avoid washing out patient signal when it’s well-supported, while keeping all safeguards (clipping + strict normalization) and the exact submission schema. These changes are minimal, keep runtime tiny, and are directly aimed at lowering KL toward your target.'
- What this solution (achieved 0.85484) has done: 'Your current score (0.85486, lower-is-better) is still above the target (0.6376), so we should improve slightly without changing the core “global prior + patient prior + shrinkage + smoothing + temperature” approach. The most likely low-risk gain is to compute the global prior with the same *patient-level weighting* used at inference time (patients contribute equally rather than being dominated by patients with many EEGs/votes), which usually improves KL calibration. I keep your patient prior construction and shrinkage intact, but swap the global prior computation to a patient-balanced vote aggregation (still de-overlapped at eeg_id), and leave all I/O and submission formatting unchanged. This is a minimal semantic change (still priors-only) and should move KL down toward the target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os



## === cell 1
import librosa
import albumentations as A
import gc
import matplotlib.pyplot as plt
import math
import multiprocessing
import random
import time
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from albumentations.pytorch import ToTensorV2
from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, List

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")




## === cell 2
class config:
    model = "resnet18d"
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


TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



## === cell 3
train_df = pd.read_csv(paths.train_csv)
test_df = pd.read_csv(paths.test_csv)

print("Train:", train_df.shape, "Test:", test_df.shape)
train_df.head(), test_df.head()



## === cell 4
alpha = 0.06
eps = 1e-12

eeg_votes = train_df.groupby("eeg_id")[TARGETS].sum().astype(np.float64)
eeg_meta = train_df.groupby("eeg_id")[["patient_id"]].first()

eeg_votes_with_pid = eeg_votes.join(eeg_meta[["patient_id"]], how="left")

patient_vote_sum = (
    eeg_votes_with_pid.groupby("patient_id")[TARGETS].sum().astype(np.float64)
)
global_vote_totals_balanced = patient_vote_sum.mean(axis=0).values.astype(np.float64)

prior_global = global_vote_totals_balanced + alpha
prior_global = prior_global / max(prior_global.sum(), eps)

patient_support_votes = patient_vote_sum.sum(axis=1).astype(np.float64)

patient_priors: Dict[int, np.ndarray] = {}
for pid, row in patient_vote_sum.iterrows():
    v = row.values.astype(np.float64) + alpha
    v = v / max(v.sum(), eps)
    patient_priors[int(pid)] = v

print(
    "Global prior:",
    dict(zip(TARGETS, prior_global)),
    "Sum:",
    float(prior_global.sum()),
)
print("Patient priors computed for:", len(patient_priors), "patients")
if len(patient_support_votes):
    print(
        "Patient support (total votes) min/median/max:",
        float(patient_support_votes.min()),
        float(patient_support_votes.median()),
        float(patient_support_votes.max()),
    )
else:
    print("Patient support (total votes) min/median/max: 0 0 0")



## === cell 5
w_patient_base = 0.70
k_shrink_patient = 18.0  # still in "votes" units (effective evidence)

final_global_mix = 0.01
temperature = 1.03

predictions = np.zeros((len(test_df), len(TARGETS)), dtype=np.float64)
test_pids = test_df["patient_id"].values

for i, pid in enumerate(test_pids):
    pid_int = int(pid)
    p = prior_global

    pp = patient_priors.get(pid_int, None)
    if pp is not None:
        n_p = float(patient_support_votes.get(pid_int, 0.0))
        w_p_eff = w_patient_base * (n_p / (n_p + k_shrink_patient)) if n_p > 0 else 0.0
        p = (1.0 - w_p_eff) * p + w_p_eff * pp

    if final_global_mix > 0:
        p = (1.0 - final_global_mix) * p + final_global_mix * prior_global

    p = np.clip(p, 1e-12, 1.0)
    p = np.exp(np.log(p) / temperature)
    p = p / max(p.sum(), eps)

    predictions[i] = p

predictions = np.clip(predictions, 1e-12, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

print(
    "Predictions shape:",
    predictions.shape,
    "Row-sum min/max:",
    float(predictions.sum(1).min()),
    float(predictions.sum(1).max()),
)



## === cell 6
sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
sub[TARGETS] = predictions

vals = sub[TARGETS].values.astype(np.float64)
vals = np.clip(vals, 1e-12, 1.0)
vals = vals / vals.sum(axis=1, keepdims=True)
sub[TARGETS] = vals

out_path = os.path.join(paths.out, "submission.csv")
sub.to_csv(out_path, index=False)
print("Saved:", out_path, "shape:", sub.shape)
sub.head()



## === cell 7
assert list(sub.columns) == ["eeg_id"] + TARGETS
row_sums = sub[TARGETS].sum(axis=1).values
assert np.all(np.isfinite(row_sums))
assert np.allclose(row_sums, 1.0, atol=1e-6)
print("Submission format OK. Example row sum:", float(row_sums[0]))
