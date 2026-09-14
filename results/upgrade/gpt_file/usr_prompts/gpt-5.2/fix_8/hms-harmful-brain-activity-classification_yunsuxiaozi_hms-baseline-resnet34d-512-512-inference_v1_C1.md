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

1.49017

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.51482) has done: 'I fix the failing merge by removing the incorrect one-to-one validation and instead build the spectrogram paths directly from `test.csv`/`sample_submission.csv` aligned by `eeg_id`, which also resolves the missing `path` column. I also fix the tensor shape handling around `transforms.Resize` by ensuring the spectrogram is treated as a single-channel image `(1,H,W)` and only adding the batch dimension once before passing to the model. Finally, I keep the core “prior-only” model logic unchanged, but ensure predictions are properly normalized and the submission matches the required columns and row order, writing a valid `submission.csv`.'
- What this solution (achieved 1.41937) has done: 'Your current solution is a fixed “prior-only” model, so the only legitimate way to move the KL score toward your target (lower is better) without changing core modeling logic is to calibrate the prior to better match the true label distribution. I compute the empirical class prior from `train.csv` using the vote counts (aggregated per `eeg_id` to match test granularity) and use that as the model prior. This keeps the same architecture, forward pass, and softmax semantics, but should substantially reduce KL versus an arbitrary hand-set prior. I also keep your submission alignment and probability normalization exactly as required.'
- What this solution (achieved 1.48502) has done: 'Your current “prior-only” model is already the right minimal approach for this score-matching objective, but the prior can be made closer to what the test distribution likely looks like without changing the model architecture or inference loop. I keep the same PriorModel and softmax semantics, but replace the global vote prior with a patient-balanced prior (average per-patient class distribution) to reduce over-representation of patients with many overlapping train segments. I also apply a very small amount of smoothing to the prior (Dirichlet-style) to avoid overly confident probabilities that tend to hurt KL. Everything else (paths, transforms, forward pass, normalization, submission schema) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 1.49017) has done: 'Your current score is worse than the target (lower is better), so the smallest safe way to move KL down without changing the model/inference core is to make the constant prior closer to the (unknown) test distribution. I keep the same PriorModel + softmax pipeline, but compute the prior at the correct granularity by first aggregating vote counts per `eeg_id` (not per `(patient_id, eeg_id)`), then patient-balancing by averaging per-patient EEG distributions; this avoids overweighting patients with many overlapping segments. I also switch smoothing from adding a constant to probabilities (which distorts normalization) to adding a tiny Dirichlet pseudocount to the *counts* before normalization, which is a minimal semantic change but typically reduces overconfident priors that hurt KL. Everything else (paths, spectrogram reading, resize, submission formatting/normalization) stays the same and still writes `submission.csv`.'
- What this solution (achieved 1.49017) has done: 'Your current pipeline is “prior-only”, so the most direct way to reduce KL (lower is better) without changing model/inference logic is to make that constant prior closer to the test label distribution. The smallest safe improvement is to compute the prior at the correct granularity: first aggregate vote counts per `eeg_id` (not per `(patient_id, eeg_id)`), then patient-balance by averaging EEG-level class distributions per patient, and finally average across patients. I keep your Dirichlet-style pseudocount smoothing (on counts before normalization) and keep the same `PriorModel` + softmax and the same spectrogram reading loop (even though it’s unused by the model). This should move the score down toward your target while preserving the core semantics and producing the same valid `submission.csv` format.'
- What this solution (achieved 1.49017) has done: 'Your current score (1.49017, lower-is-better) is far above the target (0.748099), so we should improve it, but with minimal changes that preserve your “prior-only” model semantics. The biggest issue is that the prior is accidentally computed at the wrong granularity because you group by `["patient_id","eeg_id"]` first (so an EEG with multiple subsamples becomes multiple rows and is then implicitly overweighted in the patient mean). I fix this by first aggregating votes per `eeg_id` (matching test granularity), then computing an EEG-level distribution, and only then patient-balancing by averaging across each patient’s EEGs. Everything else (PriorModel, softmax, spectrogram loading loop, submission formatting/normalization) stays the same, and it still writes a valid `submission.csv`.'

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

warnings.filterwarnings(
    "ignore"
)  # filterwarnings()方法是用于设置警告过滤器的方法，它可以控制警告信息的输出方式和级别。




## === cell 1
LABELS = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
VOTE_COLS = [f"{l}_vote" for l in LABELS]

train_df = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
)

eeg_level_votes = train_df.groupby(["eeg_id"], as_index=False)[VOTE_COLS].sum()
eeg_level_patient = train_df.groupby(["eeg_id"], as_index=False)[["patient_id"]].first()
eeg_level = eeg_level_votes.merge(eeg_level_patient, on="eeg_id", how="left")

alpha_count = 0.5  # tiny pseudocount per class; keeps "prior-only" semantics but reduces overconfidence for KL
votes = eeg_level[VOTE_COLS].to_numpy(np.float64) + alpha_count
sums = votes.sum(axis=1, keepdims=True)
sums = np.where(sums == 0, 1.0, sums)
eeg_probs = votes / sums

eeg_probs_df = pd.DataFrame(eeg_probs, columns=VOTE_COLS)
eeg_probs_df["patient_id"] = eeg_level["patient_id"].values
eeg_probs_df["eeg_id"] = eeg_level["eeg_id"].values

patient_mean = eeg_probs_df.groupby("patient_id", as_index=False)[VOTE_COLS].mean()
prior = patient_mean[VOTE_COLS].mean(axis=0).to_numpy(np.float64)

prior = prior / prior.sum()
prior = prior.astype(np.float32)


class PriorModel(nn.Module):
    def __init__(self, prior_probs):
        super().__init__()
        p = torch.tensor(prior_probs, dtype=torch.float32)
        p = p / p.sum()
        self.register_buffer("logits", torch.log(p.clamp_min(1e-12)))

    def forward(self, x):
        b = x.shape[0]
        return self.logits.unsqueeze(0).expand(b, -1)


model = PriorModel(prior)




## === cell 2
class Config:
    seed = 2024
    image_transform = transforms.Resize((512, 512))




## === cell 3
def seed_everything(seed):
    torch.backends.cudnn.deterministic = True  # 将cuda加速的随机数生成器设为确定性模式
    torch.backends.cudnn.benchmark = True  # 保持原代码语义
    torch.manual_seed(seed)  # pytorch的随机种子
    np.random.seed(seed)  # numpy的随机种子
    random.seed(seed)  # python内置的随机种子


seed_everything(Config.seed)




## === cell 4
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




## === cell 5
paths = submission["path"].values
model.eval()
test_pred = []

with torch.no_grad():
    for path in paths:
        eps = 1e-6
        data = pd.read_parquet(path)

        arr = data.fillna(-1).values[:, 1:].T  # (freq, time)
        arr = arr[:, 0:300]

        arr = np.clip(arr, np.exp(-6), np.exp(10))
        arr = np.log(arr)

        data_mean = arr.mean(axis=(0, 1))
        data_std = arr.std(axis=(0, 1))
        arr = (arr - data_mean) / (data_std + eps)

        x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        x = Config.image_transform(x)  # (1, 512, 512)
        x = x.unsqueeze(0)  # (B=1, C=1, 512, 512)

        logits = model(x)  # (1, 6)
        pred = F.softmax(logits, dim=1)[0]  # (6,)
        test_pred.append(pred.cpu().numpy())

test_pred = np.asarray(test_pred, dtype=np.float32)
test_pred[:2], test_pred.shape




## === cell 6
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
