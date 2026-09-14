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

0.4587981659456721

# 6. Current score

0.7224

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The fix adds safe handling for missing weight directories, forces all tensors to run on CPU (since the environment has no GPU), and provides a fallback that creates uniform class probabilities when no model weights are available. This eliminates the file‑not‑found, CUDA, and mismatched‑shape errors, ensuring the pipeline runs end‑to‑end and writes a valid `submission.csv` whose rows sum to 1.'
- What this solution (achieved 1.39779) has done: 'I add a lightweight fallback that uses the empirical class distribution from the training data instead of a uniform guess when no model weights are found. This small change keeps the core architecture untouched while giving more informed probabilities, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.64506) has done: 'I fixed the shape‑mismatch when falling back to the patient‑wise mean predictions. Now each patient’s probability dict is converted to a fixed‑order NumPy array (matching the TARGETS order), and the fallback uses the overall training mean array. This resolves the stacking error, guarantees a valid `predictions` array, and keeps the submission generation unchanged while preserving the intended fallback logic.'
- What this solution (achieved 0.86455) has done: 'I replace the fallback logic that uses per‑patient means with a simple blend of the patient‑specific mean (when available) and the overall training mean. This keeps the core model untouched, guarantees valid probability rows, and is expected to move the KL‑divergence closer to the target (lower is better).'
- What this solution (achieved 1.64506) has done: 'I adjust the fallback inference logic so that when no model weights are available it uses the patient‑specific mean probability distribution directly (or the overall training mean when the patient is unseen) instead of blending it with the overall mean. This makes the predictions more tailored to each patient and is expected to reduce the KL‑divergence, moving the score closer to the lower target value.'
- What this solution (achieved 0.86455) has done: 'I modify the fallback inference (cell 10) to blend the patient‑specific mean with the overall training mean instead of using one or the other. This gives a more informed prediction for known patients while still anchoring unseen patients to the global distribution, and I renormalize the blended vector so each row sums to 1. This simple change is expected to lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.64506) has done: 'I adjust the fallback inference logic to rely entirely on the patient‑specific mean distribution when it exists (setting α = 1.0) and fall back to the global training mean otherwise. This eliminates the unnecessary blending that was hurting performance, keeping the rows normalized and preserving the original pipeline.'
- What this solution (achieved 0.7289) has done: 'I added the missing imports (`pandas`, `numpy`, `os`, `albumentations`, `torch`, `torch.utils.data.DataLoader`, `gc`, `mne`) at the start, fixed the order of cells to start from 1, and ensured all referenced names are defined before use. This resolves the NameError issues, allows the fallback‑prediction logic to run when no model weights are present, and guarantees that a correctly formatted `submission.csv` is written with rows that sum to 1.'
- What this solution (achieved 1.64506) has done: 'I keep the overall pipeline unchanged but improve the fallback prediction logic: when model weights are missing, I use the patient‑specific mean distribution directly (or the global training mean for unseen patients) instead of blending it with the global mean. This gives more accurate probabilities for known patients and keeps rows normalized, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.71539) has done: 'I adjust the fallback prediction logic to smooth patient‑specific mean probabilities with the overall training mean, using each patient’s sample count as a weighting factor. This creates a more robust estimate that should lower the KL‑divergence toward the target value while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.7224) has done: 'I reduce the smoothing constant used when blending patient‑specific averages with the global training mean, giving more weight to the patient‑specific distribution (which is usually more informative). This small change keeps the overall pipeline unchanged while moving the KL‑divergence closer to the lower target score.'

# 9. Code solution

## === cell 0
import os
import gc
import pickle

import numpy as np
import pandas as pd
import albumentations as A
import torch
from torch.utils.data import DataLoader
import mne
import librosa
import timm
import torchaudio
from torch import nn
from tqdm import tqdm

CFG = {
    "batch_size": 32,
    "num_worker": 4,
    "data": "/kaggle/input/hms-harmful-brain-activity-classification/test.csv",
    "weights_spec": "/kaggle/input/hms-baseline",
    "weights_eeg": "/kaggle/input/hms-eeg",
    "weights_mix": "/kaggle/input/hms-mix",
    "flip": True,
}

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

train_votes = train_df[TARGETS].values.astype(np.float32)
row_sums = train_votes.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
train_probs = train_votes / row_sums
TRAIN_MEAN = train_probs.mean(axis=0)  # shape (6,)

patient_df = pd.DataFrame(train_probs, columns=TARGETS)
patient_df["patient_id"] = train_df["patient_id"].values
PATIENT_MEAN = patient_df.groupby("patient_id")[TARGETS].mean().to_dict("index")
PATIENT_COUNTS = train_df.groupby("patient_id").size().to_dict()

CFG




## === cell 1
def list_weights(path):
    return (
        [os.path.join(path, x) for x in sorted(os.listdir(path))]
        if os.path.isdir(path)
        else []
    )


CFG["weights_spec"] = list_weights(CFG["weights_spec"])
CFG["weights_eeg"] = list_weights(CFG["weights_eeg"])
CFG["weights_mix"] = list_weights(CFG["weights_mix"])

CFG



## === cell 2
data_dir = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs"
NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]


class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.wave_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=200,
            hop_length=10000 // 256,
            n_fft=1024,
            n_mels=128,
            f_min=0,
            f_max=20,
            win_length=128,
        )

    def forward(self, x):
        mel = self.wave_transform(x)  # (n_mels, time)
        return mel


transform_func = Transform().to(torch.device("cpu"))


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
            x_tensor = torch.from_numpy(x).float()
            mel_spec = transform_func(x_tensor)
            mel_spec = mel_spec.cpu().numpy()
            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[
                :, :width
            ]
            img[:, :, k] += mel_spec_db
        img[:, :, k] /= 4.0
    return img


all_fs = os.listdir(data_dir)
all_specs = {}

for item in tqdm(all_fs, desc="Creating EEG spectrograms"):
    eeg_id = item.rsplit(".", 1)[0]
    eeg_path = f"/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/{eeg_id}.parquet"
    eeg_spec = spectrogram_from_eeg(eeg_path)
    all_specs[eeg_id] = eeg_spec

with open("eeg_specs_dict.pkl", "wb") as file:
    pickle.dump(all_specs, file)




## === cell 3
class AlaskaDataIter:
    def __init__(
        self,
        df,
        training_flag=False,
        shuffle=False,
        use_spec=False,
        use_eeg=False,
        use_mix=False,
        ll=0,
        rr=20,
    ):
        self.ll = ll
        self.rr = rr
        self.training_flag = training_flag
        self.shuffle = shuffle
        self.df = df
        self.train_trans = A.Compose([A.HorizontalFlip(p=0.5)])
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
        self.use_spec = use_spec
        self.use_mix = use_mix
        if self.use_mix:
            with open("eeg_specs_dict.pkl", mode="rb") as f:
                self.eeg_specs = pickle.load(f)

    def __getitem__(self, item):
        return self.single_map_func(self.df.iloc[item], self.training_flag)

    def __len__(self):
        return len(self.df)

    def brain_lead(self, waves):
        brain_leads = [self.LL, self.LP, self.RP, self.RP]
        leads = []
        for combine in brain_leads:
            for i in range(len(combine) - 1):
                tmp_lead = (
                    waves[self.leads_dict[combine[i]]]
                    - waves[self.leads_dict[combine[i + 1]]]
                )
                leads.append(tmp_lead)
        return np.concatenate([leads], axis=0)

    def get_eeg(self, dp, is_training):
        eeg_path = f'/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/{dp["eeg_id"]}.parquet'
        eeg = pd.read_parquet(eeg_path)
        offset = 0
        eeg = eeg.iloc[int(offset * 200) : int(offset * 200) + 10000]
        waves = eeg.values.T
        for i in range(waves.shape[0]):
            m = np.nanmean(waves[i])
            if np.isnan(waves[i]).mean() < 1:
                waves[i] = np.nan_to_num(waves[i], nan=m)
            else:
                waves[i] = 0
        waves = np.array(waves, dtype=np.float64)
        waves = mne.filter.filter_data(waves, 200, self.ll, self.rr, verbose=False)
        waves = self.brain_lead(waves)
        return waves

    def get_spec(self, dp, is_training):
        spec_path = f'/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/{dp["spectrogram_id"]}.parquet'
        spec = pd.read_parquet(spec_path).values[:, 1:]
        images = []
        r = 0
        for region in range(4):
            img = spec[r : r + 300, region * 100 : (region + 1) * 100].T
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)
            img = np.nan_to_num(img, nan=0.0)
            images.append(img)
        images = np.stack(images, -1)
        return np.transpose(images, [2, 0, 1])

    def get_mix(self, dp, is_training):
        X = np.zeros((128, 256, 8), dtype="float32")
        kg_spec = self.get_spec(dp, is_training=False)
        eeg_spec = self.eeg_specs[str(dp["eeg_id"])]
        kg_spec = np.transpose(kg_spec, axes=[1, 2, 0])
        X[14:-14, :, :4] = kg_spec[:, 22:-22]
        X[:, :, 4:] = eeg_spec
        return np.transpose(X, [2, 0, 1])

    def single_map_func(self, dp, is_training):
        if self.use_eeg:
            data = self.get_eeg(dp, is_training)
        elif self.use_spec:
            data = self.get_spec(dp, is_training)
        elif self.use_mix:
            data = self.get_mix(dp, is_training)
        else:
            raise ValueError("No data modality selected.")
        return data.astype(np.float32)




## === cell 4
class NetSpec(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=3)
        self.fc = nn.Linear(2048, 6, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bs = x.size(0)
        x1 = [x[:, i : i + 1, :, :] for i in range(4)]
        x1 = torch.cat(x1, dim=2)
        x = torch.cat([x1, x1, x1], dim=1)
        if CFG["flip"]:
            x_flip = torch.flip(x, [3])
            x = torch.cat([x, x_flip], dim=0)
        x = self.model.forward_features(x)
        x = self.avg(x)
        x = x.view(2 * bs if CFG["flip"] else bs, -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.softmax(x, -1)
        if CFG["flip"]:
            x = (x[:bs] + x[bs:]) / 2.0
        return x




## === cell 5
class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.wave_transform = torchaudio.transforms.Spectrogram(
            n_fft=512, hop_length=25, power=1
        )
        self.am2db = torchaudio.transforms.AmplitudeToDB(stype="magnitude", top_db=80)
        self.resizer = nn.UpsamplingBilinear2d(size=[160, 320])

    def forward(self, x):
        image = self.wave_transform(x)
        image = self.am2db(image)
        n, c, h, w = image.size()
        image = image[:, :, : int(20 / 100 * h + 5), :]
        image = torch.reshape(image, (n, 4, -1, w))
        return image


class NetEeg(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.preprocess = Transform()
        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=4)
        self.fc = nn.Linear(2048, 6, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bs = x.size(0)
        x = self.preprocess(x)
        if CFG["flip"]:
            x_flip = torch.flip(x, [3])
            x = torch.cat([x, x_flip], dim=0)
        x = self.model.forward_features(x)
        x = self.avg(x)
        x = x.view(2 * bs if CFG["flip"] else bs, -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.softmax(x, -1)
        if CFG["flip"]:
            x = (x[:bs] + x[bs:]) / 2.0
        return x




## === cell 6
class NetMix(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.preprocess = Transform()
        self.model = timm.create_model("hrnet_w18", pretrained=False, in_chans=3)
        self.fc = nn.Linear(2048, 6, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bs = x.size(0)
        if CFG["flip"]:
            x_flip = torch.flip(x, [3])
            x = torch.cat([x, x_flip], dim=0)
        x1 = torch.cat([x[:, i : i + 1, :, :] for i in range(4)], dim=2)
        x2 = torch.cat([x[:, i + 4 : i + 5, :, :] for i in range(4)], dim=2)
        x = torch.cat([x1, x2], dim=3)
        x = torch.cat([x, x, x], dim=1)
        x = self.model.forward_features(x)
        x = self.avg(x)
        x = x.view(2 * bs if CFG["flip"] else bs, -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.softmax(x, -1)
        if CFG["flip"]:
            x = (x[:bs] + x[bs:]) / 2.0
        return x




## === cell 7
def inference_function(test_loader, model, device):
    model.eval()
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for X in tqdm_test_loader:
            X = X.to(device)
            with torch.no_grad():
                y_preds = model(X)
            preds.append(y_preds.cpu().numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 8
test_df = pd.read_csv(CFG["data"])
test_df.head(5)



## === cell 9
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if len(CFG["weights_eeg"]) == 0:
    pred_list = []
    smoothing_const = 1.0  # was 5.0
    for pid in test_df["patient_id"]:
        patient_probs = PATIENT_MEAN.get(pid)
        if patient_probs is None:
            blended = TRAIN_MEAN
        else:
            patient_arr = np.array(
                [patient_probs[col] for col in TARGETS], dtype=np.float32
            )
            cnt = PATIENT_COUNTS.get(pid, 0)
            alpha = cnt / (
                cnt + smoothing_const
            )  # larger alpha → more patient influence
            blended = alpha * patient_arr + (1.0 - alpha) * TRAIN_MEAN
        blended = blended / blended.sum()
        pred_list.append(blended.astype(np.float32))
    predictions = np.stack(pred_list, axis=0).astype(np.float32)
else:
    predictions_list = []
    print("infer with weights_eeg")
    for model_weight in CFG["weights_eeg"]:
        test_dataset = AlaskaDataIter(
            test_df, training_flag=False, shuffle=False, use_eeg=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG["batch_size"],
            num_workers=CFG["num_worker"],
            shuffle=False,
        )

        model = NetEeg()
        state_dict = torch.load(model_weight, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)

        pred_dict = inference_function(test_loader, model, device)
        predictions_list.append(pred_dict["predictions"])

        torch.cuda.empty_cache()
        gc.collect()
    predictions = np.mean(np.stack(predictions_list, axis=0), axis=0)

predictions.shape



## === cell 10
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
sub[TARGETS] = predictions
row_sums = sub[TARGETS].sum(axis=1).replace(0, 1)
sub[TARGETS] = sub[TARGETS].div(row_sums, axis=0)
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
sub.head()
