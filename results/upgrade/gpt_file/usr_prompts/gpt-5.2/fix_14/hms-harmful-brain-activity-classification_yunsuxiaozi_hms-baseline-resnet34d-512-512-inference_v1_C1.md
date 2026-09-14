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

0.748099

# 6. Current score

1.4033

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.51482) has done: 'I fix the failing merge by removing the incorrect one-to-one validation and instead build the spectrogram paths directly from `test.csv`/`sample_submission.csv` aligned by `eeg_id`, which also resolves the missing `path` column. I also fix the tensor shape handling around `transforms.Resize` by ensuring the spectrogram is treated as a single-channel image `(1,H,W)` and only adding the batch dimension once before passing to the model. Finally, I keep the core “prior-only” model logic unchanged, but ensure predictions are properly normalized and the submission matches the required columns and row order, writing a valid `submission.csv`.'
- What this solution (achieved 1.41937) has done: 'Your current solution is a fixed “prior-only” model, so the only legitimate way to move the KL score toward your target (lower is better) without changing core modeling logic is to calibrate the prior to better match the true label distribution. I compute the empirical class prior from `train.csv` using the vote counts (aggregated per `eeg_id` to match test granularity) and use that as the model prior. This keeps the same architecture, forward pass, and softmax semantics, but should substantially reduce KL versus an arbitrary hand-set prior. I also keep your submission alignment and probability normalization exactly as required.'
- What this solution (achieved 1.48502) has done: 'Your current “prior-only” model is already the right minimal approach for this score-matching objective, but the prior can be made closer to what the test distribution likely looks like without changing the model architecture or inference loop. I keep the same PriorModel and softmax semantics, but replace the global vote prior with a patient-balanced prior (average per-patient class distribution) to reduce over-representation of patients with many overlapping train segments. I also apply a very small amount of smoothing to the prior (Dirichlet-style) to avoid overly confident probabilities that tend to hurt KL. Everything else (paths, transforms, forward pass, normalization, submission schema) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 1.49017) has done: 'Your current score is worse than the target (lower is better), so the smallest safe way to move KL down without changing the model/inference core is to make the constant prior closer to the (unknown) test distribution. I keep the same PriorModel + softmax pipeline, but compute the prior at the correct granularity by first aggregating vote counts per `eeg_id` (not per `(patient_id, eeg_id)`), then patient-balancing by averaging per-patient EEG distributions; this avoids overweighting patients with many overlapping segments. I also switch smoothing from adding a constant to probabilities (which distorts normalization) to adding a tiny Dirichlet pseudocount to the *counts* before normalization, which is a minimal semantic change but typically reduces overconfident priors that hurt KL. Everything else (paths, spectrogram reading, resize, submission formatting/normalization) stays the same and still writes `submission.csv`.'
- What this solution (achieved 1.49017) has done: 'Your current pipeline is “prior-only”, so the most direct way to reduce KL (lower is better) without changing model/inference logic is to make that constant prior closer to the test label distribution. The smallest safe improvement is to compute the prior at the correct granularity: first aggregate vote counts per `eeg_id` (not per `(patient_id, eeg_id)`), then patient-balance by averaging EEG-level class distributions per patient, and finally average across patients. I keep your Dirichlet-style pseudocount smoothing (on counts before normalization) and keep the same `PriorModel` + softmax and the same spectrogram reading loop (even though it’s unused by the model). This should move the score down toward your target while preserving the core semantics and producing the same valid `submission.csv` format.'
- What this solution (achieved 1.49017) has done: 'Your current score (1.49017, lower-is-better) is far above the target (0.748099), so we should improve it, but with minimal changes that preserve your “prior-only” model semantics. The biggest issue is that the prior is accidentally computed at the wrong granularity because you group by `["patient_id","eeg_id"]` first (so an EEG with multiple subsamples becomes multiple rows and is then implicitly overweighted in the patient mean). I fix this by first aggregating votes per `eeg_id` (matching test granularity), then computing an EEG-level distribution, and only then patient-balancing by averaging across each patient’s EEGs. Everything else (PriorModel, softmax, spectrogram loading loop, submission formatting/normalization) stays the same, and it still writes a valid `submission.csv`.'
- What this solution (achieved 1.40263) has done: 'Your current “prior-only” model is fundamentally limited, so the most direct way to reduce KL (lower is better) without changing the model/inference core is to better calibrate the constant prior to match the test distribution as closely as we can from train metadata. I keep the same PriorModel + softmax pipeline, but compute a more robust prior by (1) using an EEG-level distribution, (2) patient-balancing, and (3) blending that patient-balanced prior with the global EEG-weighted prior (a small mixture often reduces KL when the true test mix lies between the two). I also slightly increase the Dirichlet pseudocount smoothing on counts to avoid overly sharp priors that are typically penalized by KL. Everything else (spectrogram loading loop, tensor shapes, submission formatting and normalization, paths) remains the same and still writes a valid `submission.csv`.'
- What this solution (achieved 1.4033) has done: 'Your current pipeline is “prior-only”, so the safest way to move KL down toward 0.748099 (lower is better) without changing core logic is to calibrate that constant prior closer to the expected test distribution and avoid overconfident probabilities. I keep the same PriorModel+softmax and the same inference loop, but adjust the prior computation by (1) aggregating votes per `eeg_id`, (2) patient-balancing, and (3) using a simple, robust ensemble of multiple smoothed priors (different Dirichlet pseudocount strengths and mixture weights) averaged together to reduce sensitivity and typically improve KL. This is still the same constant-probability model (no feature usage), just a more stable prior estimate. Submission formatting and probability normalization remain unchanged and a valid `submission.csv` is written.'
- What this solution (achieved 1.39701) has done: 'Your current score is much worse than the target (lower is better), and because the model is “prior-only”, the only safe lever (without changing core logic) is improving how that constant prior is estimated and avoiding overconfident probabilities that inflate KL. I keep the same PriorModel+softmax inference exactly, but (1) compute priors from vote counts with a correct Bayesian Dirichlet posterior mean (pseudocount added to counts, normalized by total votes + K*alpha), and (2) apply a small temperature (>1) to slightly soften the fixed logits (same architecture/forward, just calibrated confidence), which typically reduces KL for constant predictors. Everything else (data reading loop, transforms, submission alignment/normalization, output filename) remains unchanged and still produces a valid `submission.csv`.'
- What this solution (achieved 1.39448) has done: 'Your current pipeline is a constant “prior-only” predictor, so the safest way to reduce KL toward the target (lower is better) is to better calibrate that fixed prior without changing any model/training core (there is no training). I keep the same PriorModel+softmax inference, but adjust two calibration knobs that directly affect KL for constant predictors: (1) blend in a tiny amount of uniform probability (reduces overconfidence, typically improves KL), and (2) slightly increase the temperature to further soften predictions. I also ensure determinism/consistency by setting `cudnn.benchmark=False` (no semantic change here since the model ignores inputs), while keeping all I/O paths and submission formatting identical.'
- What this solution (achieved 1.39312) has done: 'Your current score (1.39448, lower-is-better) is still far above the target (0.748099), so we should improve it with the smallest changes that keep your “prior-only” model semantics intact. The biggest, low-risk gain is to stop doing expensive spectrogram I/O and transforms that don’t affect predictions (your model ignores inputs), which also removes a potential hidden source of numerical/shape issues while keeping outputs identical in meaning. Then, to move KL down, we only adjust calibration knobs already present in your logic: slightly increase uniform mixing and soften a bit more with temperature, which typically reduces KL for constant predictors by avoiding overconfident probabilities. Everything else (prior estimation, model class, softmax, submission columns/order, and row-wise normalization) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 1.4033) has done: 'Your current “prior-only” setup can only improve by making the constant class-probability vector closer to the (unknown) test label distribution while avoiding overconfident probabilities that inflate KL. I keep the same PriorModel + softmax inference and the same submission formatting, but replace the hand-tuned temperature/uniform mix with a tiny, deterministic grid-search on a small train-held-out split to pick the best (temperature, uniform_mix) by KL computed against normalized vote targets. This does not change the model architecture or training loop (there is none), it just calibrates two existing knobs using a validation proxy so the score should move down toward your target. I also ensure we compute KL in the same way as Kaggle (targets and predictions normalized, clipped), and still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import pandas as pd  # 导入csv文件的库
import numpy as np  # 进行矩阵运算的库
import torch  # 一个深度学习的库Pytorch
import torch.nn as nn  # neural network,神经网络
import torch.nn.functional as F  # 神经网络函数库
import torchvision.transforms as transforms  # Pytorch下面的图像处理库,用于对图像进行数据增强
import random
import warnings  # 避免一些可以忽略的报错

warnings.filterwarnings("ignore")  # 控制警告信息的输出方式和级别



## === cell 1
LABELS = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
VOTE_COLS = [f"{l}_vote" for l in LABELS]

train_df = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
)

eeg_level_votes = train_df.groupby(["eeg_id"], as_index=False)[VOTE_COLS].sum()
eeg_level_patient = train_df.groupby(["eeg_id"], as_index=False)[["patient_id"]].first()
eeg_level = eeg_level_votes.merge(eeg_level_patient, on="eeg_id", how="left")


def compute_priors(alpha_count: float):
    counts = eeg_level[VOTE_COLS].to_numpy(np.float64)
    counts = np.where(np.isfinite(counts), counts, 0.0)

    K = counts.shape[1]
    post = counts + float(alpha_count)
    denom = counts.sum(axis=1, keepdims=True) + float(alpha_count) * K
    denom = np.where(denom == 0, 1.0, denom)
    eeg_probs = post / denom  # (n_eeg, 6)

    global_prior = eeg_probs.mean(axis=0)

    eeg_probs_df = pd.DataFrame(eeg_probs, columns=VOTE_COLS)
    eeg_probs_df["patient_id"] = eeg_level["patient_id"].values
    patient_mean = eeg_probs_df.groupby("patient_id", as_index=False)[VOTE_COLS].mean()
    patient_prior = patient_mean[VOTE_COLS].mean(axis=0).to_numpy(np.float64)
    return global_prior, patient_prior


alphas = [0.5, 2.0, 6.0]
mixes = [0.40, 0.60, 0.80]

priors = []
for a in alphas:
    global_prior, patient_prior = compute_priors(alpha_count=a)
    for m in mixes:
        p = m * patient_prior + (1.0 - m) * global_prior
        p = p / np.clip(p.sum(), 1e-12, None)
        priors.append(p)

prior = np.mean(np.stack(priors, axis=0), axis=0)
prior = prior / np.clip(prior.sum(), 1e-12, None)
prior = prior.astype(np.float32)


class PriorModel(nn.Module):
    def __init__(self, prior_probs, temperature: float = 1.0):
        super().__init__()
        p = torch.tensor(prior_probs, dtype=torch.float32)
        p = p / p.sum()
        logits = torch.log(p.clamp_min(1e-12))
        self.register_buffer("logits", logits)
        self.temperature = float(temperature)

    def forward(self, x):
        b = x.shape[0]
        return (self.logits / self.temperature).unsqueeze(0).expand(b, -1)




## === cell 2
class Config:
    seed = 2024
    image_transform = transforms.Resize((512, 512))




## === cell 3
def seed_everything(seed):
    torch.backends.cudnn.deterministic = True  # 将cuda加速的随机数生成器设为确定性模式
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)  # pytorch的随机种子
    np.random.seed(seed)  # numpy的随机种子
    random.seed(seed)  # python内置的随机种子


seed_everything(Config.seed)




## === cell 4
def _kl_divergence(true_probs: np.ndarray, pred_probs: np.ndarray) -> float:
    true_probs = np.asarray(true_probs, dtype=np.float64)
    pred_probs = np.asarray(pred_probs, dtype=np.float64)

    true_probs = np.clip(true_probs, 1e-12, 1.0)
    true_probs = true_probs / true_probs.sum(axis=1, keepdims=True)

    pred_probs = np.clip(pred_probs, 1e-12, 1.0)
    pred_probs = pred_probs / pred_probs.sum(axis=1, keepdims=True)

    return float(
        np.mean(np.sum(true_probs * (np.log(true_probs) - np.log(pred_probs)), axis=1))
    )


counts = eeg_level[VOTE_COLS].to_numpy(np.float64)
counts = np.where(np.isfinite(counts), counts, 0.0)
den = counts.sum(axis=1, keepdims=True)
den = np.where(den == 0, 1.0, den)
y_eeg = counts / den  # (n_eeg, 6) normalized vote distribution

rng = np.random.default_rng(Config.seed)
n = len(eeg_level)
idx = np.arange(n)
rng.shuffle(idx)
val_size = max(2000, int(0.2 * n))  # fixed-ish size for stability
val_idx = idx[:val_size]

temps = [1.0, 1.2, 1.35, 1.55, 1.8, 2.2]
uniform_mixes = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]

best = None
best_params = None

base_prior = prior.astype(np.float64)
for t in temps:
    p_temp = np.power(np.clip(base_prior, 1e-12, 1.0), 1.0 / float(t))
    p_temp = p_temp / np.clip(p_temp.sum(), 1e-12, None)

    for um in uniform_mixes:
        p = (1.0 - float(um)) * p_temp + float(um) * (
            np.ones_like(p_temp) / p_temp.size
        )
        p = p / np.clip(p.sum(), 1e-12, None)

        pred_val = np.tile(p[None, :], (len(val_idx), 1))
        kl = _kl_divergence(y_eeg[val_idx], pred_val)

        if (best is None) or (kl < best):
            best = kl
            best_params = (float(t), float(um), p.astype(np.float32))

best_temperature, best_uniform_mix, tuned_prior = best_params

model = PriorModel(tuned_prior, temperature=best_temperature)

print(
    f"Chosen calibration: temperature={best_temperature}, uniform_mix={best_uniform_mix}, val_KL={best:.6f}"
)



## === cell 5
test_df = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
)
submission = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)

test_unique = test_df.drop_duplicates(subset=["eeg_id"], keep="first")[
    ["eeg_id", "spectrogram_id"]
]
submission = submission.merge(test_unique, on="eeg_id", how="left")

if submission["spectrogram_id"].isna().any():
    missing = (
        submission.loc[submission["spectrogram_id"].isna(), "eeg_id"].head(5).tolist()
    )
    raise ValueError(f"Missing spectrogram_id for some eeg_id. Examples: {missing}")

submission["path"] = submission["spectrogram_id"].apply(
    lambda x: "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    + str(x)
    + ".parquet"
)
submission.head()



## === cell 6
paths = submission["path"].values
model.eval()
test_pred = []

dummy_x = torch.zeros((1, 1, 1, 1), dtype=torch.float32)

with torch.no_grad():
    for _ in paths:
        logits = model(dummy_x)  # (1, 6)
        pred = F.softmax(logits, dim=1)[0]  # (6,)
        test_pred.append(pred.cpu().numpy())

test_pred = np.asarray(test_pred, dtype=np.float32)
test_pred[:2], test_pred.shape



## === cell 7
labels = LABELS

row_sums = test_pred.sum(axis=1, keepdims=True)
row_sums = np.where(row_sums == 0, 1.0, row_sums)
test_pred = test_pred / row_sums

out = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
if len(out) != len(test_pred):
    raise ValueError(
        f"Row count mismatch: sample_submission has {len(out)} rows, preds have {len(test_pred)} rows"
    )

for i, lab in enumerate(labels):
    out[f"{lab}_vote"] = test_pred[:, i].astype(np.float32)

probs = out[[f"{lab}_vote" for lab in labels]].to_numpy(np.float64)
probs = np.clip(probs, 1e-12, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)
out[[f"{lab}_vote" for lab in labels]] = probs.astype(np.float32)

out.to_csv("submission.csv", index=False)
out.head()
