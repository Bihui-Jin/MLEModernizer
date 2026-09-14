# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

0.2879761379598549

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
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




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1351186145.py in <cell line: 0>()
      1 SFREQ = 200
      2 filter_range = [0.5, 40]
----> 3 b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")
      4 
      5 

NameError: name 'signal' is not defined

## === cell 1
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




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1262130683.py in <cell line: 0>()
      9     EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
     10 else:
---> 11     test = pd.read_csv(
     12         "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
     13     )

NameError: name 'pd' is not defined

## === cell 2
def seed_everything(seed=2024):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seed_everything()




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3401120418.py in <cell line: 0>()
      7 
      8 
----> 9 seed_everything()
     10 
     11 

/tmp/ipykernel_55/3401120418.py in seed_everything(seed)
      1 def seed_everything(seed=2024):
----> 2     torch.backends.cudnn.deterministic = True
      3     torch.backends.cudnn.benchmark = False
      4     torch.manual_seed(seed)
      5     np.random.seed(seed)

NameError: name 'torch' is not defined

## === cell 3
try:
    import timm
except Exception:
    timm = None




## === cell 4
import torch.utils.data as data
from torch.utils.data import DataLoader
import gc
import torch.nn.functional as F




## === cell 5
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




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/964179854.py in <cell line: 0>()
----> 1 class DummyBackbone(nn.Module):
      2     def __init__(self, num_classes=6, in_chans=1):
      3         super().__init__()
      4         self.num_classes = num_classes
      5         self.in_chans = in_chans

NameError: name 'nn' is not defined

## === cell 6
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




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/455953455.py in <cell line: 0>()
----> 1 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      2 vit_models = []
      3 model_weights = [
      4     "/kaggle/input/hms-bestlb-vitbase/fold_0_exp_7_bestlb.pth",
      5     "/kaggle/input/hms-bestlb-vitbase/fold_1_exp_7_bestlb.pth",

NameError: name 'torch' is not defined

## === cell 7
class ImageFolder(data.Dataset):
    def __init__(self, df, imgsize):
        self.df = df.reset_index(drop=True)
        self.imgsize = imgsize  # (H, W)
        self._zero_raw = torch.zeros(self.imgsize, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.df)

    def _load_and_preprocess_spec(self, spec_id):
        spec = pd.read_parquet(os.path.join(SPEC_PATH, f"{spec_id}.parquet"))
        spec_arr = spec.values[:, 1:].T.astype("float32")[:, :300]
        spec_tensor = torch.from_numpy(spec_arr).unsqueeze(0).unsqueeze(0)  # (1,1,F,T)
        spec_tensor = F.interpolate(
            spec_tensor, size=self.imgsize, mode="bilinear", align_corners=False
        )
        eps = 1e-6
        spec_tensor = torch.clamp(
            spec_tensor, torch.exp(torch.tensor(-4.0)), torch.exp(torch.tensor(8.0))
        )
        spec_tensor = torch.log(spec_tensor + eps)
        spec_tensor = torch.nan_to_num(spec_tensor, nan=0.0)
        m = spec_tensor.mean()
        s = spec_tensor.std()
        spec_tensor = (spec_tensor - m) / (s + eps)
        return spec_tensor.squeeze(0).squeeze(0)  # (H, W)

    def _load_and_preprocess_eeg(self, eeg_id):
        eeg_path = os.path.join(EEG_PATH, f"{eeg_id}.parquet")
        eeg_img = stft_spec_from_eeg(eeg_path)  # (H, W)
        eeg_tensor = torch.from_numpy(eeg_img).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        eeg_tensor = F.interpolate(
            eeg_tensor, size=self.imgsize, mode="bilinear", align_corners=False
        )
        eps = 1e-6
        m = eeg_tensor.mean()
        s = eeg_tensor.std()
        eeg_tensor = (eeg_tensor - m) / (s + eps)
        return eeg_tensor.squeeze(0).squeeze(0)  # (H, W)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        eeg_id = str(row.eeg_id)
        spec_id = str(row.spectrogram_id)

        spec_arr = self._load_and_preprocess_spec(spec_id)  # (H, W)
        eeg_img = self._load_and_preprocess_eeg(eeg_id)  # (H, W)

        spec_arr = spec_arr.unsqueeze(-1)
        eeg_img = eeg_img.unsqueeze(-1)

        raw_10l = self._zero_raw
        raw_10c = self._zero_raw
        raw_10r = self._zero_raw
        raw_50 = self._zero_raw

        return (
            spec_arr.float(),
            eeg_img.float(),
            raw_50.float(),
            raw_10l.float(),
            raw_10c.float(),
            raw_10r.float(),
            eeg_id,
        )




## === cell 8
test_dataset = ImageFolder(test, (518, 518))
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    pin_memory=False,
    num_workers=8,
    drop_last=False,
    persistent_workers=True,
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




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/260235148.py in <cell line: 0>()
----> 1 test_dataset = ImageFolder(test, (518, 518))
      2 test_loader = DataLoader(
      3     test_dataset,
      4     batch_size=32,
      5     pin_memory=False,

NameError: name 'test' is not defined

## === cell 9
for m in vit_models:
    del m
torch.cuda.empty_cache()
gc.collect()




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/527718076.py in <cell line: 0>()
----> 1 for m in vit_models:
      2     del m
      3 torch.cuda.empty_cache()
      4 gc.collect()
      5 

NameError: name 'vit_models' is not defined

## === cell 10
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

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1696768055.py in <cell line: 0>()
      7     "other_vote",
      8 ]
----> 9 probs_array = np.vstack(prob_list) if prob_list else np.empty((0, 6))
     10 sub = pd.DataFrame(probs_array, columns=col_names)
     11 sub.insert(0, "eeg_id", eeg_id_list)

NameError: name 'prob_list' is not defined
