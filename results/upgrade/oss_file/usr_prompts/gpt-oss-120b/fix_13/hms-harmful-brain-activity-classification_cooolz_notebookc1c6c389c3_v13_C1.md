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

0.4224930516963866

# 6. Current score

0.79764

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I added guards so missing weight folders don’t cause crashes and provided a safe fallback that creates uniform class probabilities when no pretrained models are found. This lets the pipeline run end‑to‑end, produces a correctly‑shaped `submission.csv` with probabilities that sum to 1, and avoids the earlier file‑not‑found and shape‑mismatch errors while keeping the original architecture unchanged.'
- What this solution (achieved 1.41937) has done: 'I keep the original pipeline unchanged and only adjust the fallback when no pretrained weights are found. Instead of using a uniform 1/6 probability for every class, I compute class‑wise priors from the train metadata (summing all vote counts and normalising). This simple calibration respects the required format, guarantees rows sum to 1, and should lower the KL divergence compared with the uniform baseline, moving the score toward the target.'
- What this solution (achieved 1.68479) has done: 'I fixed the fill‑na error in the fallback probability calculation, replacing the NumPy array with a Pandas Series so the patient‑wise priors are correctly filled. This allows the fallback branch to run, creates a proper (N, 6) prediction array, and enables the final cell to write a valid `submission.csv` where each row’s probabilities sum to 1. No other logic is changed, preserving the original model‑based pipeline.'
- What this solution (achieved 0.76744) has done: 'I keep the original data loading, model definitions and inference logic unchanged, but I improve the fallback “no‑weights” predictions and blend them with any model‑based predictions that do exist.  
1) Compute class priors per patient with Laplace smoothing (add‑one) and also a smoothed overall prior.  
2) When no pretrained weights are found the code already uses the patient‑wise prior.  
3) When at least one model provides predictions, I now add the smoothed prior as an additional “model” and average all predictions together. This modest calibration moves the KL‑divergence toward the target without altering the core architecture or training procedure.'
- What this solution (achieved 0.81798) has done: 'I smooth the class‑wise priors more heavily and blend each patient’s prior with the overall prior (90 % patient, 10 % overall). This reduces over‑confident patient‑specific probabilities, keeping rows summing to 1 while moving the KL‑divergence closer to the target lower score.'
- What this solution (achieved 1.06043) has done: 'I soften the prior calculations and give the overall class distribution more influence, which should reduce over‑confident patient‑specific predictions and move the KL‑divergence toward the lower target value. This is done by increasing the Laplace smoothing term and changing the patient/overall blend to 0.5 / 0.5, keeping the rest of the pipeline unchanged.'
- What this solution (achieved 1.4053) has done: 'Implemented missing imports, corrected undefined names, and refined the fallback prior blending to use overall class priors (which are more stable) while keeping the temperature scaling that softens predictions. These changes fix runtime errors, ensure a valid `submission.csv` is written, and modestly improve the KL‑divergence score toward the target.'
- What this solution (achieved 0.96005) has done: 'Implemented fixes to restore end‑to‑end execution and improve the KL‑divergence score:

* Added missing `torchaudio` import.
* Corrected the fallback prior fill‑na operation to use a Pandas Series, avoiding the ndarray error.
* Increased Laplace smoothing (`SMOOTH = 2.0`) for more stable priors.
* Raised the temperature scaling factor (`TEMPERATURE = 3.0`) to soften predictions and better align with the KL metric.
* Adjusted cell ordering to start at 1 while preserving original logic.'
- What this solution (achieved 0.79764) has done: 'I reduced the Laplace smoothing, blended patient‑wise priors with the overall class prior (instead of using the overall prior only for missing patients), and changed the temperature scaling to sharpen the final probabilities (temperature < 1). These tweaks keep the original architecture and inference flow unchanged while moving the KL‑divergence closer to the target lower score.'

# 9. Code solution

## === cell 0
import os, copy, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
import albumentations as A
import timm
from tqdm.auto import tqdm

CFG = {
    "batch_size": 32,
    "num_worker": 4,
    "data": "/kaggle/input/hms-harmful-brain-activity-classification/test.csv",
    "weights_spec": "/kaggle/input/hms-baseline",
    "weights_eeg": "/kaggle/input/hms-eeg",
    "flip": True,
}




## === cell 1
def _list_weights(dir_path):
    if os.path.isdir(dir_path):
        return [os.path.join(dir_path, x) for x in sorted(os.listdir(dir_path))]
    else:
        return []


CFG["weights_spec"] = _list_weights(CFG["weights_spec"])
CFG["weights_eeg"] = _list_weights(CFG["weights_eeg"])
CFG




## === cell 2
class AlaskaDataIter:
    def __init__(
        self, df, training_flag=False, shuffle=False, use_eeg=False, ll=0, rr=20
    ):

        self.ll = ll
        self.rr = rr
        self.training_flag = training_flag
        self.shuffle = shuffle

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

    def single_map_func(self, dp, is_training):
        """Data augmentation function."""
        if self.use_eeg:
            eeg_path = f"/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/{dp['eeg_id']}.parquet"
            eeg = pd.read_parquet(eeg_path)
            offset = 0
            eeg = eeg.iloc[offset * 200 : (offset + 50) * 200]
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
            data = np.transpose(images, [2, 0, 1])
        return data.astype(np.float32)




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
        x1 = torch.cat(x1, dim=2)
        x = torch.cat([x1, x1, x1], dim=1)
        x = self.model.forward_features(x)
        x = self.avg(x)
        x = x.view(bs, -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x




## === cell 4
class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.wave_transform = torchaudio.transforms.Spectrogram(
            n_fft=512, hop_length=25, power=1
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
        x = self.model.forward_features(x)
        x = self.avg(x)
        x = x.view(bs, -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x




## === cell 6
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




## === cell 7
test_df = pd.read_csv(CFG["data"])

predictions = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if CFG["weights_spec"]:
    print("infer with weights_spec")
    for model_weight in CFG["weights_spec"]:
        test_dataset = AlaskaDataIter(test_df, training_flag=False, shuffle=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG["batch_size"],
            num_workers=CFG["num_worker"],
            shuffle=False,
        )
        model = NetSpec()
        state_dict = torch.load(model_weight, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        preds = inference_function(test_loader, model, device)
        predictions.append(preds)
        torch.cuda.empty_cache()
        gc.collect()
else:
    print("No spectrogram weights found – skipping spectrogram inference.")

if CFG["weights_eeg"]:
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
        preds = inference_function(test_loader, model, device)
        predictions.append(preds)
        torch.cuda.empty_cache()
        gc.collect()
else:
    print("No EEG weights found – skipping EEG inference.")

train_path = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
train_df = pd.read_csv(train_path)

vote_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

SMOOTH = 0.5
overall_sums = train_df[vote_cols].sum().astype(np.float64) + SMOOTH
overall_total = overall_sums.sum()
overall_prior = overall_sums / overall_total  # Series indexed by vote_cols

patient_groups = train_df.groupby("patient_id")[vote_cols].sum()
patient_sums = patient_groups + SMOOTH
patient_priors = patient_sums.div(patient_sums.sum(axis=1), axis=0)  # (n_patients,6)

patient_prior_df = test_df[["patient_id"]].merge(
    patient_priors.reset_index(), on="patient_id", how="left"
)

ALPHA = 0.7
patient_filled = patient_prior_df[vote_cols].fillna(overall_prior)
fallback_array = (
    ALPHA * patient_filled.values + (1 - ALPHA) * overall_prior.values.reshape(1, -1)
).astype(np.float32)

if len(predictions) == 0:
    predictions = fallback_array
else:
    predictions.append(fallback_array)
    predictions = np.array(predictions)
    predictions = np.mean(predictions, axis=0)

TEMPERATURE = 0.7
predictions = np.power(predictions, 1.0 / TEMPERATURE)
predictions = predictions / predictions.sum(axis=1, keepdims=True)




## === cell 8
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
if predictions.shape[0] != len(test_df):
    raise ValueError(
        f"Prediction rows {predictions.shape[0]} do not match test rows {len(test_df)}"
    )
sub[TARGETS] = predictions
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
sub.head()
