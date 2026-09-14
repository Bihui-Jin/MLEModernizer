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

0.4568878693320788

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The fix adds the missing imports (`os`, `albumentations as A`, `copy`, `gc`, `torch.utils.data.DataLoader`, and `timm`) and moves the configuration/setup into the first cell so that weight listing works before it’s used. This resolves the NameErrors, allows the data iterator to be created, and ensures predictions have the correct shape for building the submission file.'
- What this solution (achieved 1.41937) has done: 'I add all missing imports, define a minimal `CFG` configuration (including data paths, batch size, worker count, flip flag, and empty weight lists), and guard the expensive EEG‑spectrogram generation so it only runs when model weights are provided. This resolves the `NameError`s, ensures the data loading and inference code can execute, and yields a valid `submission.csv` with properly normalised probabilities, moving the solution from “no score” to a baseline submission (lower KL‑divergence than the target).'
- What this solution (achieved 1.68479) has done: 'I replace the simple prior‑only baseline with a patient‑aware baseline: for each test sample the code now looks up the historical average vote distribution of the same patient in the training data (if available) and uses that as the prediction, falling back to the overall prior otherwise. This adds only a few lines, keeps the same overall pipeline, and is expected to lower the KL‑divergence (move the score toward the target).'
- What this solution (achieved 0.77861) has done: 'I blend the patient‑level averages with the overall class prior instead of using the patient mean alone. This adds a simple smoothing (α ≈ 0.6) that keeps the predictions anchored to the global distribution, which usually lowers KL divergence and moves the score closer to the target while preserving the existing pipeline.'
- What this solution (achieved 0.82593) has done: 'I lower the blending weight α from 0.6 to 0.3 so the prediction relies more on the global class prior, which is typically more stable than noisy patient‑specific averages. This small adjustment is expected to reduce the KL‑divergence (move the score closer to the target) while preserving the original pipeline and all other logic.'
- What this solution (achieved 0.77861) has done: 'I increase the blending factor α so the patient‑specific distribution contributes more weight (closer to the prior‑only baseline that performed better). This simple change is expected to lower the KL‑divergence, moving the score toward the target while keeping the core pipeline unchanged.'
- What this solution (achieved 0.85517) has done: 'I replace the simple row‑wise patient mean with a patient‑wise vote‑count proportion (which reflects the true number of annotations per patient) and keep the same blending logic. This gives a more accurate patient‑specific distribution and should lower the KL‑divergence, moving the score closer to the target while preserving the overall pipeline.'
- What this solution (achieved 1.41937) has done: 'I lower the blending factor `alpha` to 0 so the predictions rely solely on the global class prior, which removes noisy patient‑specific estimates and is expected to reduce the KL‑divergence toward the target score. All other logic and file handling remain unchanged.'

# 9. Code solution

## === cell 0
import os
import copy
import gc
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
import torchaudio
import librosa
import tqdm
from tqdm import tqdm
import albumentations as A
import timm
import mne

CFG = {
    "data": "/kaggle/input/hms-harmful-brain-activity-classification/test.csv",
    "weights_eeg": [],  # empty → use prior‑based baseline
    "weights_spec": [],  # not used in this script
    "weights_mix": [],  # not used in this script
    "batch_size": 32,
    "num_worker": 0,
    "flip": False,  # no test‑time augmentation
    "device": "cpu",
}

if CFG["weights_eeg"] or CFG["weights_mix"]:
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
            mel = self.wave_transform(x)
            return mel

    transform_func = Transform().to("cpu")

    def spectrogram_from_eeg(parquet_path):
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
                x_tensor = torch.from_numpy(x.astype(np.float32)).to("cpu")
                mel_spec = transform_func(x_tensor)
                mel_spec = mel_spec.cpu().numpy()
                width = (mel_spec.shape[1] // 32) * 32
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(
                    np.float32
                )[:, :width]
                img[:, :, k] += mel_spec_db
            img[:, :, k] /= 4.0
        return img

    all_fs = os.listdir(data_dir)
    all_specs = {}
    for item in tqdm.tqdm(all_fs, desc="Building EEG specs"):
        eeg_id = item.rsplit(".", 1)[0]
        eeg_path = f"/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/{eeg_id}.parquet"
        eeg_spec = spectrogram_from_eeg(eeg_path)
        all_specs[eeg_id] = eeg_spec

    with open("eeg_specs_dict.pkl", "wb") as file:
        pickle.dump(all_specs, file)




## === cell 1
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
        TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRD": 3, "GRD": 4, "Other": 5}
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
        waves = copy.deepcopy(waves)
        brain_leads = [self.LL, self.LP, self.RP, self.RR]
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

    def get_eeg(self, dp, is_training):
        eeg_path = f'/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/{dp["eeg_id"]}.parquet'
        eeg = pd.read_parquet(eeg_path)
        offset = 0
        eeg = eeg.iloc[int(offset * 200) : int(offset * 200) + 10000]
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
        return waves

    def get_spec(self, dp, is_training):
        spec_path = f'/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/{dp["spectrogram_id"]}.parquet'
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
        return data

    def get_mix(self, dp, is_training):
        X = np.zeros((128, 256, 8), dtype="float32")
        kg_spec = self.get_spec(dp, is_training=False)
        eeg_spec = self.eeg_specs[str(dp["eeg_id"])]
        kg_spec = np.transpose(kg_spec, axes=[1, 2, 0])
        X[14:-14, :, :4] = kg_spec[:, 22:-22]
        X[:, :, 4:] = eeg_spec
        X = np.transpose(X, [2, 0, 1])
        return X

    def single_map_func(self, dp, is_training):
        if self.use_eeg:
            data = self.get_eeg(dp, is_training)
        elif self.use_spec:
            data = self.get_spec(dp, is_training)
        elif self.use_mix:
            data = self.get_mix(dp, is_training)
        else:
            data = np.zeros((1, 1, 1), dtype=np.float32)  # dummy placeholder
        return data.astype(np.float32)




## === cell 2
def inference_function(test_loader, model, device):
    model.eval()
    preds = []
    with tqdm.tqdm(
        test_loader, unit="test_batch", desc="Inference"
    ) as tqdm_test_loader:
        for X in tqdm_test_loader:
            X = X.to(device)
            with torch.no_grad():
                y_preds = model(X)
            preds.append(y_preds.cpu().numpy())
    return np.concatenate(preds)




## === cell 3
test_df = pd.read_csv(CFG["data"])
test_df.head()




## === cell 4
if CFG["weights_eeg"]:
    predictions = []
    device = torch.device(CFG["device"])
    for model_weight in CFG["weights_eeg"]:
        test_dataset = AlaskaDataIter(
            test_df, training_flag=False, shuffle=False, use_eeg=True, ll=0, rr=40
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
        preds = inference_function(test_loader, model, device)
        predictions.append(preds)
        torch.cuda.empty_cache()
        gc.collect()
    predictions = np.mean(np.stack(predictions), axis=0)
else:
    train_path = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    vote_cols = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)

        prior_counts = train_df[vote_cols].sum().values.astype(np.float32)
        prior_probs = prior_counts / prior_counts.sum()

        patient_counts = train_df.groupby("patient_id")[vote_cols].sum()
        patient_probs = patient_counts.div(patient_counts.sum(axis=1), axis=0)

        test_with_patient = test_df[["eeg_id", "patient_id"]].merge(
            patient_probs, left_on="patient_id", right_index=True, how="left"
        )
        for col in vote_cols:
            test_with_patient[col].fillna(
                prior_probs[vote_cols.index(col)], inplace=True
            )

        patient_probs_arr = test_with_patient[vote_cols].values.astype(np.float32)

        alpha = 0.0
        blended = alpha * patient_probs_arr + (1 - alpha) * prior_probs

        row_sums = blended.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        predictions = blended / row_sums
    else:
        prior_probs = np.full(6, 1 / 6, dtype=np.float32)
        predictions = np.tile(prior_probs, (len(test_df), 1))




## === cell 5
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
sub.to_csv("submission.csv", index=False)
print(f"Submission saved: submission.csv (shape {sub.shape})")
