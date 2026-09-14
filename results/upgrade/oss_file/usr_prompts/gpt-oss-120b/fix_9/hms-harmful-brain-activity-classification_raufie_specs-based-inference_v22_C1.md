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
PyWavelets==1.8.0
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

0.6909634962658863

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I replace the failing model‑loading inference with a simple baseline that uses the class distribution from the training data as predictions for every test sample. This removes the missing‑file error, guarantees the output probabilities sum to 1, and creates a valid `submission.csv` file.'
- What this solution (achieved 1.68479) has done: 'I replace the simple global‑frequency baseline with a patient‑aware baseline: compute class vote distributions per patient from the training data and use them for any test rows that share a patient_id; fall back to the overall class prior when a patient is unseen. This keeps the core logic unchanged, guarantees probabilities sum to 1, and should lower the KL score toward the target.'
- What this solution (achieved 1.68479) has done: 'I keep the existing data‑loading and dataset code, but replace the simple patient‑prior baseline with an actual model inference step. The script now loads the EfficientNet weights, runs the test loader through the model, and writes the softmax probabilities to `submission.csv`. If the weight file is missing or loading fails, it gracefully falls back to the previous patient‑wise prior so the code always produces a valid submission.'
- What this solution (achieved 1.68479) has done: 'I keep the overall pipeline unchanged but enhance the fallback baseline: in addition to the patient‑wise prior it also try an eeg_id‑wise prior (which is often more specific) before falling back to the global class prior. This small refinement should produce predictions that are closer to the true label distribution and therefore reduce the KL divergence toward the target while preserving the original logic and file output.'
- What this solution (achieved 1.68479) has done: 'I added all missing imports, defined the device, corrected the use of undefined variables (glob, np, pd, nn, Dataset, DataLoader, label_cols), reordered the loading of spectrogram and EEG data before the dataset class, and fixed the dataset constructor to receive the pre‑loaded dictionaries. These fixes let the script run end‑to‑end, produce a properly normalised probability matrix, and write a valid `submission.csv` file. The fallback patient/eeg‑wise prior remains unchanged, providing a reasonable score while keeping the core logic intact.'
- What this solution (achieved 1.68479) has done: 'The fix corrects the shape mismatch in the dataset by extracting a 100 × 256 region from each spectrogram (center‑cropping and padding when needed) before assigning it to the input tensor. This eliminates the broadcasting error, allows the DataLoader to produce valid samples, and lets the rest of the pipeline run to generate a properly normalised submission CSV. No other logic is altered, preserving the original model‑fallback behavior and keeping the score‑related code unchanged.'

# 9. Code solution

## === cell 0
test_df = pd.read_csv(paths.TEST_CSV)
print(f"Test dataframe shape: {test_df.shape}")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2074763077.py in <cell line: 0>()
----> 1 test_df = pd.read_csv(paths.TEST_CSV)
      2 print(f"Test dataframe shape: {test_df.shape}")
      3 

NameError: name 'pd' is not defined

## === cell 1
paths_spectrograms = glob.glob(os.path.join(paths.TEST_SPECTROGRAMS, "*.parquet"))
print(f"There are {len(paths_spectrograms)} spectrogram parquet files")
all_spectrograms = {}
for fp in tqdm(paths_spectrograms, desc="Loading spectrograms"):
    df = pd.read_parquet(fp)
    eid = int(os.path.basename(fp).split(".")[0])
    all_spectrograms[eid] = df.iloc[:, 1:].values



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4080582316.py in <cell line: 0>()
----> 1 paths_spectrograms = glob.glob(os.path.join(paths.TEST_SPECTROGRAMS, "*.parquet"))
      2 print(f"There are {len(paths_spectrograms)} spectrogram parquet files")
      3 all_spectrograms = {}
      4 for fp in tqdm(paths_spectrograms, desc="Loading spectrograms"):
      5     df = pd.read_parquet(fp)

NameError: name 'glob' is not defined

## === cell 2
USE_WAVELET = None
NAMES = ["LL", "LP", "RP", "RR"]
FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]


def maddest(d, axis: int = None):
    return np.mean(np.abs(d - np.mean(d, axis)), axis)


def denoise(x: np.ndarray, wavelet: str = "haar", level: int = 1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = [pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:]]
    return pywt.waverec(coeff, wavelet, mode="per")


def spectrogram_from_eeg(parquet_path, display=False):
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle : middle + 10_000]

    img = np.zeros((128, 256, 4), dtype="float32")
    signals = []
    for k in range(4):
        COLS = FEATS[k]
        for kk in range(4):
            x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values
            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0
            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            mel_spec = librosa.feature.melspectrogram(
                y=x,
                sr=200,
                hop_length=len(x) // 256,
                n_fft=1024,
                n_mels=128,
                fmin=0,
                fmax=20,
                win_length=128,
            )
            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[
                :, :width
            ]
            mel_spec_db = (mel_spec_db + 40) / 40
            img[:, :, k] += mel_spec_db
        img[:, :, k] /= 4.0

        if display:
            plt.subplot(2, 2, k + 1)
            plt.title(f"EEG Spectrogram {NAMES[k]}")
            plt.imshow(img[:, :, k], aspect="auto", origin="lower")
    return img


paths_eegs = glob.glob(os.path.join(paths.TEST_EEGS, "*.parquet"))
print(f"There are {len(paths_eegs)} EEG parquet files")
all_eegs = {}
for i, fp in enumerate(tqdm(paths_eegs, desc="Loading EEGs")):
    eid = int(os.path.basename(fp).split(".")[0])
    all_eegs[eid] = spectrogram_from_eeg(fp, display=False)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2338741790.py in <cell line: 0>()
     13 
     14 
---> 15 def denoise(x: np.ndarray, wavelet: str = "haar", level: int = 1):
     16     coeff = pywt.wavedec(x, wavelet, mode="per")
     17     sigma = (1 / 0.6745) * maddest(coeff[-level])

NameError: name 'np' is not defined

## === cell 3
class CustomModel(nn.Module):
    def __init__(self, cfg, num_classes: int = 6):
        super().__init__()
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True
        self.model = timm.create_model(cfg.MODEL, pretrained=False)
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, num_classes),
        )

    def __reshape_input(self, x):
        spect = torch.cat(
            [x[:, :, :, i : i + 1] for i in range(4)], dim=1
        )  # (B, 128, 256, 4)
        eeg = torch.cat(
            [x[:, :, :, i : i + 1] for i in range(4, 8)], dim=1
        )  # (B, 128, 256, 4)

        if self.USE_KAGGLE_SPECTROGRAMS and self.USE_EEG_SPECTROGRAMS:
            x = torch.cat([spect, eeg], dim=2)  # (B, 128, 512, 4)
        elif self.USE_EEG_SPECTROGRAMS:
            x = eeg
        else:
            x = spect

        x = torch.cat([x, x, x], dim=3)  # (B, 128, 512, 12)
        x = x.permute(0, 3, 1, 2)  # (B, 12, 128, 512)
        return x

    def forward(self, x):
        x = self.__reshape_input(x)
        x = self.features(x)
        x = self.custom_layers(x)
        return x




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/468277259.py in <cell line: 0>()
----> 1 class CustomModel(nn.Module):
      2     def __init__(self, cfg, num_classes: int = 6):
      3         super().__init__()
      4         self.USE_KAGGLE_SPECTROGRAMS = True
      5         self.USE_EEG_SPECTROGRAMS = True

NameError: name 'nn' is not defined

## === cell 4
class CustomDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, cfg, spectrograms, eeg_spectrograms, mode="test"
    ):
        self.df = df
        self.cfg = cfg
        self.spectrograms = spectrograms
        self.eeg_spectrograms = eeg_spectrograms
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        X = np.zeros((128, 256, 8), dtype="float32")
        y = np.zeros(len(TARGETS), dtype="float32")

        spec = self.spectrograms[row.spectrogram_id]  # shape (H, W)
        h, w = spec.shape
        start_row = max(0, (h - 100) // 2)
        img = spec[start_row : start_row + 100, :256]  # (100, 256)

        if img.shape[0] < 100:
            pad_rows = 100 - img.shape[0]
            img = np.pad(img, ((0, pad_rows), (0, 0)), mode="constant")
        if img.shape[1] < 256:
            pad_cols = 256 - img.shape[1]
            img = np.pad(img, ((0, 0), (0, pad_cols)), mode="constant")

        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)
        ep = 1e-6
        mu, std = np.nanmean(img), np.nanstd(img)
        img = (img - mu) / (std + ep)
        img = np.nan_to_num(img, nan=0.0)

        X[14:-14, :, 0] = img / 2.0  # shape (100, 256)

        X[:, :, 4:] = self.eeg_spectrograms[row.eeg_id]

        if self.mode != "test":
            y = row[label_cols].values.astype(np.float32)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return X_tensor, y_tensor




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1907359537.py in <cell line: 0>()
----> 1 class CustomDataset(Dataset):
      2     def __init__(
      3         self, df: pd.DataFrame, cfg, spectrograms, eeg_spectrograms, mode="test"
      4     ):
      5         self.df = df

NameError: name 'Dataset' is not defined

## === cell 5
test_dataset = CustomDataset(
    test_df,
    config,
    spectrograms=all_spectrograms,
    eeg_spectrograms=all_eegs,
    mode="test",
)
test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True,
    drop_last=False,
)

X_sample, y_sample = test_dataset[0]
print(f"Sample X shape: {X_sample.shape}, y shape: {y_sample.shape}")




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3103347059.py in <cell line: 0>()
----> 1 test_dataset = CustomDataset(
      2     test_df,
      3     config,
      4     spectrograms=all_spectrograms,
      5     eeg_spectrograms=all_eegs,

NameError: name 'CustomDataset' is not defined

## === cell 6
def inference_function(loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    with torch.no_grad():
        for X, _ in tqdm(loader, desc="Inference"):
            X = X.to(device)
            out = model(X)
            out = softmax(out)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)




## === cell 7
use_model = False
predictions = None

try:
    model = CustomModel(config, num_classes=len(TARGETS)).to(device)
    state_dict = torch.load(paths.MODEL_WEIGHTS, map_location=device)
    model.load_state_dict(state_dict)
    preds = inference_function(test_loader, model, device)
    row_sums = preds.sum(axis=1, keepdims=True)
    preds = preds / np.where(row_sums == 0, 1, row_sums)
    predictions = preds.astype(np.float32)
    use_model = True
    print("Model inference succeeded.")
except Exception as e:
    print(f"Model inference failed ({e}); using fallback prior.")

if not use_model:
    train_path = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    train_df = pd.read_csv(train_path)

    class_counts = train_df[TARGETS].sum()
    global_prior = class_counts / class_counts.sum()

    patient_votes = train_df.groupby("patient_id")[TARGETS].sum()
    patient_prior = patient_votes.div(patient_votes.sum(axis=1), axis=0)
    patient_totals = patient_votes.sum(axis=1)

    eeg_votes = train_df.groupby("eeg_id")[TARGETS].sum()
    eeg_prior = eeg_votes.div(eeg_votes.sum(axis=1), axis=0)
    eeg_totals = eeg_votes.sum(axis=1)

    test_patient_ids = test_df["patient_id"].values
    test_eeg_ids = test_df["eeg_id"].values

    predictions = np.zeros((len(test_df), len(TARGETS)), dtype=np.float32)

    for i, (pid, eid) in enumerate(zip(test_patient_ids, test_eeg_ids)):
        has_patient = pid in patient_prior.index
        has_eeg = eid in eeg_prior.index
        if has_patient and has_eeg:
            pw = patient_totals.loc[pid]
            ew = eeg_totals.loc[eid]
            combined = (
                patient_prior.loc[pid].values * pw + eeg_prior.loc[eid].values * ew
            )
            predictions[i] = combined / (pw + ew)
        elif has_patient:
            predictions[i] = patient_prior.loc[pid].values
        elif has_eeg:
            predictions[i] = eeg_prior.loc[eid].values
        else:
            predictions[i] = global_prior.values

    row_sums = predictions.sum(axis=1, keepdims=True)
    predictions = predictions / np.where(row_sums == 0, 1, row_sums)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1831570785.py in <cell line: 0>()
     17 if not use_model:
     18     train_path = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
---> 19     train_df = pd.read_csv(train_path)
     20 
     21     class_counts = train_df[TARGETS].sum()

NameError: name 'pd' is not defined

## === cell 8
sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
sub[TARGETS] = predictions
sub_path = os.path.join(paths.OUTPUT_DIR, "submission.csv")
sub.to_csv(sub_path, index=False)
print(f"Submission saved to {sub_path}")
print(f"Submission shape: {sub.shape}")
sub.head()

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1300523028.py in <cell line: 0>()
----> 1 sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
      2 sub[TARGETS] = predictions
      3 sub_path = os.path.join(paths.OUTPUT_DIR, "submission.csv")
      4 sub.to_csv(sub_path, index=False)
      5 print(f"Submission saved to {sub_path}")

NameError: name 'pd' is not defined
