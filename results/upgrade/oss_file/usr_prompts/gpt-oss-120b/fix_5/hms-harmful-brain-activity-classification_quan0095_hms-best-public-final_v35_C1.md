# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

No external packages required in the script and installed.

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

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.signal as signal
from scipy.signal import butter
import librosa  # retained from original code (unused but kept)

NAMES = ["LL", "LP", "RP", "RR"]
FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]



## === cell 1
SFREQ = 200
filter_range = [0.5, 40]
b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")


def stft_spec_from_eeg(parquet_path):
    EEG_LENGTH = 50
    eeg = pd.read_parquet(parquet_path)
    time_start = round((50 - EEG_LENGTH) / 2 * 200)
    time_stop = round((50 + EEG_LENGTH) / 2 * 200)
    eeg = eeg.iloc[time_start:time_stop]
    img = np.zeros((128, 256, 4), dtype="float32")
    for k in range(4):
        COLS = FEATS[k]
        for kk in range(4):
            eeg_1 = eeg[COLS[kk]]
            eeg_1 = eeg_1.fillna(eeg_1.mean()).values
            eeg_2 = eeg[COLS[kk + 1]]
            eeg_2 = eeg_2.fillna(eeg_2.mean()).values
            new_eeg = eeg_1 - eeg_2
            fs = 200
            nperseg = len(new_eeg) // 256
            f, t, spec = signal.stft(new_eeg, fs, nperseg=nperseg, noverlap=0, nfft=256)
            spec = np.log1p(np.abs(spec)).astype("float32")
            img[:, :, k] += spec[:128, 1:257]
        img[:, :, k] /= 4
    img = np.concatenate((img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]), 0)
    return img




## === cell 2
DEBUG = False  # default to inference mode
if DEBUG:
    test = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )[:40]
    SPEC_PATH = (
        "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    )
    EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
else:
    test = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    )
    SPEC_PATH = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
print("Test rows:", test.shape)




## === cell 3
def seed_everything(seed=2024):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seed_everything()



## === cell 4
try:
    import timm
except Exception:
    timm = None



## === cell 5
import torch.utils.data as data
from torch.utils.data import DataLoader
from skimage.transform import resize
import gc




## === cell 6
class DummyBackbone(nn.Module):
    def __init__(self, num_classes=6, in_chans=1):
        super().__init__()
        self.num_classes = num_classes
        self.in_chans = in_chans

    def forward_features(self, x):
        return torch.zeros(x.shape[0], 768, device=x.device)

    def forward(self, x):
        return self.forward_features(x)


class Net(nn.Module):
    def __init__(self, back_bone, device_id):
        super().__init__()
        if timm is not None:
            self.spec_model = timm.create_model(
                "vit_small_patch14_reg4_dinov2.lvd142m",
                num_classes=6,
                pretrained=False,
                in_chans=1,
            )
            self.eeg_model = timm.create_model(
                "vit_small_patch14_reg4_dinov2.lvd142m",
                num_classes=6,
                pretrained=False,
                in_chans=1,
            )
            self.raw_50s_model = timm.create_model(
                back_bone, num_classes=6, pretrained=False, in_chans=1
            )
            self.raw_10s_model = timm.create_model(
                back_bone, num_classes=6, pretrained=False, in_chans=1
            )
        else:
            self.spec_model = DummyBackbone()
            self.eeg_model = DummyBackbone()
            self.raw_50s_model = DummyBackbone()
            self.raw_10s_model = DummyBackbone()
        for m in [
            self.spec_model,
            self.eeg_model,
            self.raw_50s_model,
            self.raw_10s_model,
        ]:
            if hasattr(m, "fc_norm"):
                m.fc_norm = nn.Identity()
            if hasattr(m, "head_drop"):
                m.head_drop = nn.Identity()
            if hasattr(m, "head"):
                m.head = nn.Identity()
        self.head = nn.Linear(768 * 2 + 384 * 2, 6)

    def forward(self, spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_imgs):
        spec_imgs = spec_imgs.permute(0, 3, 1, 2).contiguous()
        eeg_imgs = eeg_imgs.permute(0, 3, 1, 2).contiguous()
        raw_50s_imgs = raw_50s_imgs.permute(0, 3, 1, 2).contiguous()
        raw_10s_imgs = raw_10s_imgs.permute(0, 3, 1, 2).contiguous()
        spec_f = self.spec_model.forward_features(spec_imgs)[:, 0]
        eeg_f = self.eeg_model.forward_features(eeg_imgs)[:, 0]
        raw50_f = self.raw_50s_model.forward_features(raw_50s_imgs)[:, 0]
        raw10_f = self.raw_10s_model.forward_features(raw_10s_imgs)[:, 0]
        feat = torch.cat((spec_f, eeg_f, raw50_f, raw10_f), dim=1)
        return self.head(feat)




## === cell 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_models = []
model_weights = [
    "/kaggle/input/hms-bestlb-vitbase/fold_0_exp_7_bestlb.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_1_exp_7_bestlb.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_2_exp_7_bestlb.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_3_exp_7_bestlb.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_4_exp_7_bestlb.pth",
]
for w in model_weights:
    model = Net("vit_base_patch14_reg4_dinov2.lvd142m", device).to(device)
    if os.path.exists(w):
        state = torch.load(w, map_location=device)
        model.load_state_dict(state)
    model.eval()
    vit_models.append(model)




## === cell 8
class ImageFolder(data.Dataset):
    def __init__(self, df, imgsize):
        self.df = df.reset_index(drop=True)
        self.imgsize = imgsize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        eeg_id = str(row.eeg_id)
        spec_id = str(row.spectrogram_id)
        spec = pd.read_parquet(os.path.join(SPEC_PATH, f"{spec_id}.parquet"))
        spec_arr = spec.values[:, 1:].T.astype("float32")[:, :300]
        spec_arr = resize(spec_arr, self.imgsize)
        eeg_path = os.path.join(EEG_PATH, f"{eeg_id}.parquet")
        eeg_img = stft_spec_from_eeg(eeg_path)
        eeg_img = resize(eeg_img, self.imgsize)
        raw_10l = np.zeros(self.imgsize, dtype="float32")
        raw_10c = np.zeros(self.imgsize, dtype="float32")
        raw_10r = np.zeros(self.imgsize, dtype="float32")
        raw_50 = np.zeros(self.imgsize, dtype="float32")
        spec_arr = np.expand_dims(spec_arr, -1)
        raw_10l = np.expand_dims(raw_10l, -1)
        raw_10c = np.expand_dims(raw_10c, -1)
        raw_10r = np.expand_dims(raw_10r, -1)
        raw_50 = np.expand_dims(raw_50, -1)
        eeg_img = np.expand_dims(eeg_img, -1)
        eps = 1e-6
        spec_arr = np.clip(spec_arr, np.exp(-4), np.exp(8))
        spec_arr = np.log(spec_arr + eps)
        spec_arr = np.nan_to_num(spec_arr, nan=0.0)
        m = eeg_img.mean(axis=(0, 1), keepdims=True)
        s = eeg_img.std(axis=(0, 1), keepdims=True)
        eeg_img = (eeg_img - m) / (s + eps)
        m = spec_arr.mean(axis=(0, 1), keepdims=True)
        s = spec_arr.std(axis=(0, 1), keepdims=True)
        spec_arr = (spec_arr - m) / (s + eps)
        return (
            spec_arr.astype("float32"),
            eeg_img.astype("float32"),
            raw_50.astype("float32"),
            raw_10l.astype("float32"),
            raw_10c.astype("float32"),
            raw_10r.astype("float32"),
            eeg_id,
        )




## === cell 9
test_dataset = ImageFolder(test, (518, 518))
test_loader = DataLoader(
    test_dataset, batch_size=32, pin_memory=False, num_workers=8, drop_last=False
)
eeg_id_list = []
prob_list = []
with torch.no_grad():
    for (
        spec_imgs,
        eeg_imgs,
        raw_50s_imgs,
        raw_10s_l_imgs,
        raw_10s_c_imgs,
        raw_10s_r_imgs,
        eeg_ids,
    ) in test_loader:
        spec_imgs = spec_imgs.to(device)
        eeg_imgs = eeg_imgs.to(device)
        raw_50s_imgs = raw_50s_imgs.to(device)
        raw_10s_l_imgs = raw_10s_l_imgs.to(device)
        raw_10s_c_imgs = raw_10s_c_imgs.to(device)
        raw_10s_r_imgs = raw_10s_r_imgs.to(device)
        ensemble_probs = torch.zeros(spec_imgs.size(0), 6, device=device)
        for mdl in vit_models:
            probs = (
                mdl(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_l_imgs).softmax(dim=1)
                + mdl(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs).softmax(dim=1)
                + mdl(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_r_imgs).softmax(dim=1)
            ) / 3
            ensemble_probs += probs
        ensemble_probs /= len(vit_models)
        probs_np = ensemble_probs.cpu().numpy()
        for i, eid in enumerate(eeg_ids):
            eeg_id_list.append(eid)
            prob_list.append(probs_np[i])



## === cell 10
for m in vit_models:
    del m
torch.cuda.empty_cache()
gc.collect()



## === cell 11
col_names = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
probs_array = np.vstack(prob_list) if prob_list else np.empty((0, 6))
sub = pd.DataFrame(probs_array, columns=col_names)
sub.insert(0, "eeg_id", eeg_id_list)
sub.to_csv("submission.csv", index=False)
print(sub.head())
