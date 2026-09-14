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
mne==1.10.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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

0.4277328111095559

# 6. Current score

1.47425

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I make the weight‑folder lookups robust (setting empty lists when the directories are missing) and replace the model‑inference loop with a simple uniform‑probability baseline, which guarantees a valid .csv submission and avoids the previous file‑not‑found and shape errors.'
- What this solution (achieved 1.41937) has done: 'I replace the uniform‑probability baseline with a simple prior‑based baseline that uses the overall class vote distribution from the training data. This keeps all model code untouched, adds only a small data‑loading step, and usually yields a lower KL divergence (moving the score toward the target). The change is limited to the prediction construction cell and preserves the submission format.'
- What this solution (achieved 1.68479) has done: 'The fix adds the missing standard library and package imports (os, copy, random, numpy, pandas, torch, torch.nn, timm, albumentations, tqdm, torchaudio, and mne) at the start of the notebook so that all later cells can access the required symbols. No core logic or model architecture is altered; the prior‑based baseline remains unchanged, and a valid `submission.csv` is now written correctly.'
- What this solution (achieved 1.0413) has done: 'I blend the patient‑specific priors with the overall global prior (using a modest weight) and add a tiny epsilon before renormalising so every row still sums to 1. This simple smoothing often reduces KL divergence without changing any model architecture or training logic, moving the score closer to the target.'
- What this solution (achieved 0.85517) has done: 'I increase the weight given to the patient‑specific prior (alpha) from 0.3 to 0.6 so that predictions rely more on patient history, which typically reduces KL divergence while keeping the same blending and normalisation logic. No other parts of the pipeline are altered.'
- What this solution (achieved 0.85517) has done: '## A brief outline
I added the missing imports, defined a default `CFG` dictionary (including paths, batch size, and weight lists), and placed the helper `_list_weights` function. This resolves all `NameError`s and lets the script fall back to the prior‑based baseline when model weights are absent, producing a valid `submission.csv` with correctly normalised probabilities.'
- What this solution (achieved 1.41937) has done: 'I fixed the path‑handling bug that tried to list weights twice (which caused a TypeError) and simplified the prediction blending by relying only on the global class prior (setting alpha to 0). This removes the patient‑specific prior that was hurting the KL‑divergence, while keeping the overall pipeline unchanged and guaranteeing a correctly‑formatted submission.csv that sums to 1 per row.'
- What this solution (achieved 0.77767) has done: 'I increase the weight of the patient‑specific prior (α) from 0.0 to 0.9 so that predictions rely much more on each patient’s historical vote distribution. This simple change keeps the overall pipeline unchanged, still guarantees that each row sums to 1, and moves the KL‑divergence score closer to the target without altering any model architecture or training logic.'
- What this solution (achieved 1.47425) has done: 'I keep the overall pipeline unchanged and only adjust the blending weight α that mixes the patient‑specific prior with the global class prior. By evaluating a small validation split of the training data for several α values and picking the one that yields the lowest average KL‑divergence, we can move the score closer to the target without touching the model architecture or any training code.'
- What this solution (achieved 1.47425) has done: 'Implemented a finer α‑grid search to better tune the blend between patient‑specific and global class priors. By evaluating 101 α values (step 0.01) on the same validation split, we select a more optimal blending weight, which is expected to lower the KL divergence and move the score closer to the target. No other logic or model architecture was altered.'
- What this solution (achieved 1.47425) has done: 'I replace the fixed α blending with a smoothing‑based blend that weights the patient‑specific prior by the amount of annotation votes it has. This simple adjustment keeps all model code unchanged, guarantees each row sums to 1, and typically lowers KL divergence, moving the score closer to the target.'
- What this solution (achieved 1.47425) has done: 'I keep the existing prior‑blending logic but, when model weights are present, blend a small proportion of the model’s softmax predictions with the prior‑based probabilities. This adds useful signal without altering the architecture, and the blend weight (β = 0.2) is modest enough to keep the predictions well‑behaved, moving the KL‑divergence toward the target while still guaranteeing a valid CSV.'
- What this solution (achieved 1.47425) has done: 'I force the solution to ignore the (likely un‑trained) model predictions and rely fully on the patient‑specific prior, which historically lowers the KL‑divergence. By setting `beta = 0` and using a smoothing value of 0 (maximizing the patient prior weight), the predictions become deterministic and the submission stays valid while moving the score toward the lower‑is‑better target.'

# 9. Code solution

## === cell 0
import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import albumentations as A
import torchaudio
from tqdm import tqdm
import mne


def _list_weights(path):
    if os.path.isdir(path):
        return [os.path.join(path, x) for x in sorted(os.listdir(path))]
    else:
        return []


BASE_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
CFG = {
    "weights_spec": _list_weights(os.path.join(BASE_PATH, "weights_spec")),
    "weights_eeg": _list_weights(os.path.join(BASE_PATH, "weights_eeg")),
    "data": os.path.join(BASE_PATH, "test.csv"),
    "batch_size": 32,
    "num_worker": 0,
}
CFG




## === cell 1
class AlaskaDataIter:
    def __init__(
        self, df, training_flag=False, shuffle=False, use_eeg=False, ll=0, rr=20
    ):

        self.ll = ll
        self.rr = rr
        self.training_flag = training_flag
        self.shuffle = shuffle

        self.raw_data_set_size = None  ## decided by self.parse_file

        self.df = df

        self.train_trans = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
            ]
        )

        TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
        self.TARS2 = {x: y for y, x in TARS.items()}

        self.eeg_nms = [
            "Fp1",
            "F3",
            "C3",
            "P3",
            "F7",
            "T3",
            "T5",
            "O1",
            "Fz",
            "Cz",
            "Pz",
            "Fp2",
            "F4",
            "C4",
            "P4",
            "F8",
            "T4",
            "T6",
            "O2",
            "EKG",
        ]

        self.LL = ["Fp1", "F7", "T3", "T5", "O1"]
        self.RR = ["Fp2", "F8", "T4", "T6", "O2"]
        self.LP = ["Fp1", "F3", "C3", "P3", "O1"]
        self.RP = ["Fp2", "F4", "C4", "P4", "O2"]

        self.leads_dict = {value: index for index, value in enumerate(self.eeg_nms)}

        self.use_eeg = use_eeg

    def __getitem__(self, item):
        return self.single_map_func(self.df.iloc[item], self.training_flag)

    def __len__(self):
        return len(self.df)

    def brain_lead(self, waves):
        waves = copy.deepcopy(waves)
        brain_leads = [self.LL, self.LP, self.RR, self.RP]
        leads = []
        for combine in brain_leads:
            for i in range(len(combine) - 1):
                tmp_lead = (
                    waves[self.leads_dict[combine[i]]]
                    - waves[self.leads_dict[combine[i + 1]]]
                )
                leads.append(tmp_lead)
        data = np.concatenate([leads], axis=0)
        return data

    def single_map_func(self, dp, is_training):
        """Data augmentation function."""
        if self.use_eeg:
            eeg_path = f"/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/{dp['eeg_id']}.parquet"
            eeg = pd.read_parquet(eeg_path)

            offset = 0  # placeholder; could be randomized for training
            if random.uniform(0, 1) < 1.0 and is_training:
                offset += random.uniform(-1, 1)
                offset = np.clip(offset, a_min=0, a_max=0)

            eeg = eeg.iloc[int(offset) * 200 : int(offset + 50) * 200]
            waves = eeg.values
            waves = np.transpose(waves, axes=[1, 0])

            for i in range(waves.shape[0]):
                m = np.nanmean(waves[i])
                if np.isnan(waves[i]).mean() < 1:
                    waves[i] = np.nan_to_num(waves[i], nan=m)
                else:
                    waves[i] = 0

            waves = np.array(waves, dtype=np.float64)
            waves = mne.filter.filter_data(waves, 200, self.ll, self.rr, verbose=False)
            waves = self.brain_lead(waves)
            data = waves
        else:
            spec_path = f"/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/{dp['spectrogram_id']}.parquet"
            spec = pd.read_parquet(spec_path)
            spec = spec.values[:, 1:]

            images = []
            for region in range(4):
                img = spec[0:300, region * 100 : (region + 1) * 100].T
                img = np.clip(img, np.exp(-4), np.exp(8))
                img = np.log(img)
                img = np.nan_to_num(img, nan=0.0)
                images.append(img)

            images = np.stack(images, -1)
            data = np.transpose(images, [2, 0, 1])

        return data.astype(np.float32)




## === cell 2
class NetSpec(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=3)
        self.fc = nn.Linear(2048, num_classes, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bs = x.size(0)
        x1 = [x[:, i : i + 1, :, :] for i in range(4)]
        x1 = torch.cat(x1, dim=2)
        x = torch.cat([x1, x1, x1], dim=1)  # now 3 channels as required
        x = self.model.forward_features(x)
        x = self.avg(x)
        x = x.view(bs, -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x




## === cell 3
class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.wave_transform = torchaudio.transforms.Spectrogram(
            n_fft=1024, hop_length=50, power=1
        )
        self.am2db = torchaudio.transforms.AmplitudeToDB(stype="magnitude", top_db=80)
        self.resizer = nn.UpsamplingBilinear2d(size=[160, 320])

    def forward(self, x):
        bs = x.size(0)
        image = self.wave_transform(x)
        image = self.am2db(image)
        n, c, h, w = image.size()
        image = image[:, :, : int(20 / 100 * h + 30), :]
        image = torch.reshape(image, shape=[n, 4, -1, w])
        return image


class NetEeg(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.preprocess = Transform()
        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=4)
        self.fc = nn.Linear(2048, num_classes, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bs = x.size(0)
        x = self.preprocess(x)
        x = self.model.forward_features(x)
        x = self.avg(x)
        x = x.view(bs, -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x




## === cell 4
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for X in tqdm_test_loader:
            X = X.to(device)
            with torch.no_grad():
                y_preds = model(X)
            y_preds = softmax(y_preds)
            preds.append(y_preds.cpu().numpy())
    return np.concatenate(preds)


test_df = pd.read_csv(CFG["data"])
train_path = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
train_df = pd.read_csv(train_path)

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

global_class_votes = train_df[TARGETS].sum().values.astype(np.float64)
global_total = global_class_votes.sum()
global_prior = global_class_votes / global_total  # (6,)

patient_group = train_df.groupby("patient_id")[TARGETS].sum()
patient_totals_series = patient_group.sum(axis=1)  # total votes per patient
patient_totals = patient_totals_series.values[:, None]  # column vector
patient_priors = patient_group.values / patient_totals
patient_prior_dict = {
    pid: prior.astype(np.float32)
    for pid, prior in zip(patient_group.index, patient_priors)
}
patient_total_dict = patient_totals_series.to_dict()  # for fast lookup


def _row_target_distribution(row):
    votes = row[TARGETS].values.astype(np.float32)
    total = votes.sum()
    if total == 0:
        return global_prior.astype(np.float32)
    return votes / total


def _kl(p, q):
    eps = 1e-12
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * np.log(p / q))


val_mask = np.random.RandomState(42).choice(
    [False, True], size=len(train_df), p=[0.9, 0.1]
)
val_df = train_df[val_mask].reset_index(drop=True)

smoothing_candidates = [0, 1, 5, 10, 20, 50]
best_smoothing = 0
best_kl = float("inf")

for s in smoothing_candidates:
    kl_sum = 0.0
    for _, row in val_df.iterrows():
        pid = row["patient_id"]
        patient_prior = patient_prior_dict.get(pid, global_prior.astype(np.float32))
        patient_total = patient_total_dict.get(pid, 0.0)

        weight = patient_total / (patient_total + s) if (patient_total + s) > 0 else 0.0
        blended = weight * patient_prior + (1 - weight) * global_prior.astype(
            np.float32
        )
        blended = blended + 1e-12
        blended = blended / blended.sum()

        true_dist = _row_target_distribution(row)
        kl_sum += _kl(true_dist, blended)

    avg_kl = kl_sum / len(val_df)
    if avg_kl < best_kl:
        best_kl = avg_kl
        best_smoothing = s

print(f"Chosen smoothing based on validation KL: {best_smoothing}")

best_smoothing = 0  # use pure patient prior when available
beta = 0.0  # completely drop model contribution

if CFG["weights_spec"]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_spec = NetSpec()
    checkpoint_path = CFG["weights_spec"][0]
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model_spec.load_state_dict(state["model"])
    else:
        model_spec.load_state_dict(state)
    model_spec.to(device)

    test_dataset = AlaskaDataIter(
        test_df, training_flag=False, shuffle=False, use_eeg=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=CFG["num_worker"],
        pin_memory=True,
    )
    _ = inference_function(test_loader, model_spec, device)

num_rows = len(test_df)
predictions = np.empty((num_rows, len(TARGETS)), dtype=np.float32)

for i, (_, row) in enumerate(test_df.iterrows()):
    pid = row["patient_id"]
    patient_prior = patient_prior_dict.get(pid, global_prior.astype(np.float32))
    patient_total = patient_total_dict.get(pid, 0.0)

    weight = (
        patient_total / (patient_total + best_smoothing)
        if (patient_total + best_smoothing) > 0
        else 0.0
    )
    blended = weight * patient_prior + (1 - weight) * global_prior.astype(np.float32)

    blended = blended + 1e-12
    blended = blended / blended.sum()

    final_vec = blended
    predictions[i] = final_vec.astype(np.float32)




## === cell 5
sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
sub[TARGETS] = predictions
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
sub.head()
