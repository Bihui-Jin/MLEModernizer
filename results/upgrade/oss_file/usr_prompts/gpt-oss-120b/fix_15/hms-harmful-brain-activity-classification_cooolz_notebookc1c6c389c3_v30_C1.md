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

0.4392794798070529

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I make the script robust to missing weight directories by safely handling the file‑listing step, skip loading nonexistent checkpoints, and fall back to uniform probability predictions when no weights are available. This prevents the `FileNotFoundError` and the later shape mismatch, ensuring a valid `submission.csv` with rows that sum to 1.'
- What this solution (achieved 1.40995) has done: 'I added the missing imports, corrected the cell numbering to start at 1, and ensured all referenced modules (os, copy, pickle, gc, numpy, pandas, torch & torch.nn, DataLoader, albumentations, timm, torchaudio, tqdm) are available. This fixes the NameError crashes, allows the data loader and model classes to be instantiated, and guarantees a valid `submission.csv` is written with correctly normalized probability rows.'
- What this solution (achieved 1.41937) has done: 'I add a train‑data path to the config and compute class‑frequency priors from the training votes. When no model weights are found the script fall back to these priors instead of a uniform guess, and when predictions exist they be softly blended (15 % prior, 85 % model) before renormalising. This modest calibration keeps the original model code untouched while moving the KL score toward the target.'
- What this solution (achieved 1.41937) has done: 'I reduced the weight of the model predictions and applied a modest temperature scaling to make the final probability distribution less confident, which should lower the KL divergence and move the score closer to the target. The core data loading, model architectures, and inference loops remain unchanged.'
- What this solution (achieved 1.41937) has done: 'I keep the original pipeline but adjust the blending step so that the final predictions rely entirely on the class‑frequency prior (which is more stable and moves the KL divergence toward the target). This reduces the influence of potentially noisy model outputs, removes the extra temperature scaling, and keeps the submission format unchanged.'
- What this solution (achieved 1.41937) has done: 'I keep the overall pipeline unchanged but give the model predictions a modest influence and smooth the final probabilities. By setting `blend_model` to 0.2 we combine 20 % of the model output with 80 % of the class‑frequency prior, which is expected to lower the KL divergence toward the target. A light temperature scaling (`temp = 1.2`) then makes the blended distribution slightly less confident while still staying normalized.'
- What this solution (achieved 1.41937) has done: 'I keep the overall pipeline unchanged but replace the blending step with a pure class‑frequency prior, which is more stable and moves the KL divergence toward the target ≈ 0.44. By ignoring the noisy model outputs and removing the extra temperature scaling, the predictions stay normalized and should reduce the score. I also renumber the cells so they start at 1 as required.'
- What this solution (achieved 1.41937) has done: 'I blend the model predictions with the class‑frequency prior instead of discarding them, and renormalize the final probabilities so each row sums to 1. This modest use of the model outputs should lower the KL divergence and move the score toward the target.'
- What this solution (achieved 1.41937) has done: 'I keep the data loading, model definitions, and inference loops unchanged, but replace the blending step with a pure class‑frequency prior. Using only the prior (which already sums to 1 per row) eliminates noisy model predictions and moves the KL divergence much closer to the target ≈ 0.44.'
- What this solution (achieved 1.41937) has done: 'I blend the model predictions with the class‑frequency prior instead of discarding them. By averaging any loaded model outputs and mixing them (e.g., 30 % model + 70 % prior) then renormalising each row, the final probabilities become more informed and the KL divergence moves lower toward the target score.'
- What this solution (achieved 1.41937) has done: 'The update removes the model‑prediction contribution in the final blending step, relying solely on the class‑frequency prior which is already well‑calibrated and yields a KL score close to the target (≈0.44). This small change keeps the core pipeline unchanged while moving the evaluation metric toward the desired lower value.'
- What this solution (achieved 1.41937) has done: 'Implemented missing imports and a robust configuration, added safe weight‑listing that handles non‑existent directories, and defined all required classes (`AlaskaDataIter`, model wrappers, transform) with proper `nn` imports. The script now falls back to class‑frequency priors when no model checkpoints are found, guaranteeing a valid normalized `submission.csv`. This fixes all NameError/FileNotFound errors and produces a submission whose KL score is close to the target ≈ 0.44 (lower is better).'

# 9. Code solution

## === cell 0
import os
import copy
import pickle
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
import timm
import torchaudio
from tqdm import tqdm

CFG = {
    "weights_spec": "/kaggle/input/hms-harmful-brain-activity-classification/weights/spec",
    "weights_eeg": "/kaggle/input/hms-harmful-brain-activity-classification/weights/eeg",
    "weights_mix": "/kaggle/input/hms-harmful-brain-activity-classification/weights/mix",
    "batch_size": 32,
    "num_worker": 2,
    "flip": False,
    "data": "/kaggle/input/hms-harmful-brain-activity-classification/test.csv",
    "train_data": "/kaggle/input/hms-harmful-brain-activity-classification/train.csv",
}




## === cell 1
def list_weights(path):
    """Return a sorted list of checkpoint files if the directory exists, else empty list."""
    if os.path.isdir(path):
        return [os.path.join(path, x) for x in sorted(os.listdir(path))]
    return []


CFG["weights_spec"] = list_weights(CFG["weights_spec"])
CFG["weights_eeg"] = list_weights(CFG["weights_eeg"])
CFG["weights_mix"] = list_weights(CFG["weights_mix"])

CFG




## === cell 2
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
        self.RL = ["Fp2", "F8", "T4", "T6", "O2"]
        self.LP = ["Fp1", "F3", "C3", "P3", "O1"]
        self.RP = ["Fp2", "F4", "C4", "P4", "O2"]

        self.leads_dict = {value: index for index, value in enumerate(self.eeg_nms)}

        self.use_eeg = use_eeg
        self.use_spec = use_spec
        self.use_mix = use_mix

        if self.use_mix:
            with open("eeg_specs_dict.pkl", "rb") as f:
                self.eeg_specs = pickle.load(f)

    def __getitem__(self, item):
        return self.single_map_func(self.df.iloc[item], self.training_flag)

    def __len__(self):
        return len(self.df)

    def brain_lead(self, waves):
        waves = copy.deepcopy(waves)
        brain_leads = [self.LL, self.RL, self.LP, self.RP]
        leads = []
        for chain in brain_leads:
            for i in range(len(chain) - 1):
                tmp = (
                    waves[self.leads_dict[chain[i]]]
                    - waves[self.leads_dict[chain[i + 1]]]
                )
                leads.append(tmp)
        return np.concatenate(leads, axis=0)

    def get_eeg(self, dp, is_training):
        eeg_path = f'/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/{dp["eeg_id"]}.parquet'
        eeg = pd.read_parquet(eeg_path)
        offset = 0
        eeg = eeg.iloc[int(offset * 200) : int(offset * 200) + 10000]
        waves = np.transpose(eeg.values, (1, 0))

        for i in range(waves.shape[0]):
            m = np.nanmean(waves[i])
            if np.isnan(waves[i]).mean() < 1:
                waves[i] = np.nan_to_num(waves[i], nan=m)
            else:
                waves[i] = 0

        waves = mne.filter.filter_data(waves, 200, self.ll, self.rr, verbose=False)
        waves = self.brain_lead(waves)
        return waves.astype(np.float32)

    def get_spec(self, dp, is_training):
        spec_path = f'/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/{dp["spectrogram_id"]}.parquet'
        spec = pd.read_parquet(spec_path).values[:, 1:]

        images = []
        for region in range(4):
            img = spec[0:300, region * 100 : (region + 1) * 100].T
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)
            img = np.nan_to_num(img, nan=0.0)
            images.append(img)

        images = np.stack(images, -1)  # shape (100, 300, 4)
        data = np.transpose(images, (2, 0, 1))  # shape (4, 100, 300)
        return data.astype(np.float32)

    def get_mix(self, dp, is_training):
        X = np.zeros((128, 256, 8), dtype="float32")
        kg_spec = self.get_spec(dp, is_training=False)
        eeg_spec = self.eeg_specs[str(dp["eeg_id"])]
        kg_spec = np.transpose(kg_spec, (1, 2, 0))
        X[14:-14, :, :4] = kg_spec[:, 22:-22]
        X[:, :, 4:] = eeg_spec
        X = np.transpose(X, (2, 0, 1))
        return X

    def single_map_func(self, dp, is_training):
        if self.use_eeg:
            data = self.get_eeg(dp, is_training)
        elif self.use_spec:
            data = self.get_spec(dp, is_training)
        elif self.use_mix:
            data = self.get_mix(dp, is_training)
        else:
            raise ValueError("No data modality selected.")
        return data




## === cell 3
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
        x1 = torch.cat(x1, dim=2)  # (bs, 1, H, W) -> (bs, 1, H, 4*W)
        x = torch.cat([x1, x1, x1], dim=1)  # make 3 channels

        if CFG["flip"]:
            x = torch.cat([x, torch.flip(x, [3])], dim=0)

        x = self.model.forward_features(x)
        x = self.avg(x)

        if CFG["flip"]:
            x = x.view(2 * bs, -1)
        else:
            x = x.view(bs, -1)

        x = self.dropout(x)
        x = self.fc(x)
        return torch.softmax(x, dim=-1)




## === cell 4
class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.wave_transform = torchaudio.transforms.Spectrogram(
            n_fft=512, hop_length=25, power=1
        )
        self.am2db = torchaudio.transforms.AmplitudeToDB(stype="magnitude", top_db=80)
        self.resizer = nn.UpsamplingBilinear2d(size=(160, 320))

    def forward(self, x):
        img = self.wave_transform(x)
        img = self.am2db(img)
        n, c, h, w = img.size()
        img = img[:, :, : int(20 / 100 * h + 30), :]
        img = torch.reshape(img, (n, 4, -1, w))
        return img




## === cell 5
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

        if CFG["flip"]:
            x = torch.cat([x, torch.flip(x, [3])], dim=0)

        x = self.model.forward_features(x)
        x = self.avg(x)

        if CFG["flip"]:
            x = x.view(2 * bs, -1)
        else:
            x = x.view(bs, -1)

        x = self.dropout(x)
        x = self.fc(x)
        return torch.softmax(x, dim=-1)




## === cell 6
class NetMix(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.preprocess = Transform()
        self.model = timm.create_model("hrnet_w18", pretrained=False, in_chans=3)
        self.fc = nn.Linear(2048, num_classes, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bs = x.size(0)

        if CFG["flip"]:
            x = torch.cat([x, torch.flip(x, [3])], dim=0)

        x1 = torch.cat([x[:, i : i + 1, :, :] for i in range(4)], dim=2)
        x2 = torch.cat([x[:, i + 4 : i + 5, :, :] for i in range(4)], dim=2)
        x = torch.cat([x1, x2], dim=3)  # (bs, 1, H, 8*W)
        x = torch.cat([x, x, x], dim=1)  # make 3 channels

        x = self.model.forward_features(x)
        x = self.avg(x)

        if CFG["flip"]:
            x = x.view(2 * bs, -1)
        else:
            x = x.view(bs, -1)

        x = self.dropout(x)
        x = self.fc(x)
        return torch.softmax(x, dim=-1)




## === cell 7
def inference_function(test_loader, model, device):
    model.eval()
    preds = []
    with tqdm(test_loader, unit="batch", desc="Inference") as pbar:
        for X in pbar:
            X = X.to(device)
            with torch.no_grad():
                y_pred = model(X)
            preds.append(y_pred.cpu().numpy())
    return np.concatenate(preds, axis=0)




## === cell 8
test_df = pd.read_csv(CFG["data"])
train_df = pd.read_csv(CFG["train_data"])

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

class_totals = train_df[TARGETS].sum().values.astype(np.float64)
prior_probs = class_totals / class_totals.sum()
prior_predictions = np.tile(prior_probs, (len(test_df), 1)).astype(np.float32)

predictions_list = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_inference(
    weight_paths, model_cls, use_spec=False, use_eeg=False, use_mix=False
):
    for weight_path in weight_paths:
        dataset = AlaskaDataIter(
            test_df,
            training_flag=False,
            shuffle=False,
            use_spec=use_spec,
            use_eeg=use_eeg,
            use_mix=use_mix,
        )
        loader = DataLoader(
            dataset,
            batch_size=CFG["batch_size"],
            num_workers=CFG["num_worker"],
            shuffle=False,
        )
        model = model_cls()
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)

        preds = inference_function(loader, model, device)
        predictions_list.append(preds)

        torch.cuda.empty_cache()
        gc.collect()


if CFG["weights_eeg"]:
    print("Running EEG models...")
    run_inference(CFG["weights_eeg"], NetEeg, use_eeg=True)
else:
    print("No EEG weights found; using prior only.")

if CFG["weights_spec"]:
    print("Running Spectrogram models...")
    run_inference(CFG["weights_spec"], NetSpec, use_spec=True)
else:
    print("No Spectrogram weights found; using prior only.")

if CFG["weights_mix"]:
    print("Running Mix models...")
    run_inference(CFG["weights_mix"], NetMix, use_mix=True)
else:
    print("No Mix weights found; using prior only.")

if predictions_list:
    avg_model_preds = np.mean(predictions_list, axis=0)
    avg_model_preds = avg_model_preds / avg_model_preds.sum(axis=1, keepdims=True)
    predictions = 0.7 * avg_model_preds + 0.3 * prior_predictions
else:
    predictions = prior_predictions

predictions = predictions / predictions.sum(axis=1, keepdims=True)
print("Final prediction shape:", predictions.shape)



## === cell 9
sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
sub[TARGETS] = predictions
sub.to_csv("submission.csv", index=False)
print(f"Submission written: submission.csv (shape {sub.shape})")
