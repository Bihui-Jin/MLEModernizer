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

cupy-cuda12x==13.6.0
fastai==2.8.5
geopandas==0.14.4
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
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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

0.4668129151277632

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the two blockers preventing any valid submission: (1) CuPy failing due to an incompatible CUDA driver by replacing the CuPy spectrogram generation with a CPU SciPy/Numpy equivalent, and (2) missing external pretrained model files by falling back to a safe uniform-probability submission when those checkpoints aren’t available. I also correct the PATH bug (`test_`/`train_` suffix) so parquet reads resolve to the provided dataset folders. Finally, I ensure the written `submission.csv` has the exact required columns and each row sums to 1, so Kaggle accepts it end-to-end.'
- What this solution (achieved 1.40995) has done: 'I fix the path construction that currently points to non-existent `test_/eegs` and `test_/spectrograms` folders, which is what triggers the FileNotFoundError in both the spectrogram read and the custom-spec generation loop. I also make `LL/RL/LP/RP` channel lists deterministic and derived once from an existing spectrogram parquet so dataset indexing can’t silently break. With those fixes, the custom spectrogram `.npy` files be created successfully and the rest of the pipeline can run (or, if external checkpoints are truly missing, it still generate a valid uniform submission). These changes are execution-critical and score-positive (they allow using real features instead of crashing/falling back mid-run), while keeping the modeling core unchanged.'
- What this solution (achieved 1.40995) has done: 'I fix the runtime blocker by making checkpoint loading robust: if the external Kaggle datasets with pretrained fold models aren’t mounted, the code no longer crash at `torch.load`. Instead, it fall back to a valid, well-formed probability submission (uniform) so a `.csv` is always produced end-to-end. This is a minimal change that preserves the existing inference logic when checkpoints exist, and only changes behavior in the missing-checkpoint case. I also add a small safety check to ensure the written submission has the exact required columns and row-wise probabilities sum to 1.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"



## === cell 1
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/hms-harmful-brain-activity-classification",
    "/kaggle/data/hms-harmful-brain-activity-classification",
]


def _resolve_data_root():
    for root in DATA_ROOT_CANDIDATES:
        if os.path.isdir(root):
            if os.path.isdir(os.path.join(root, "test_eegs")) and os.path.isdir(
                os.path.join(root, "test_spectrograms")
            ):
                return root
    root = "/kaggle/input/hms-harmful-brain-activity-classification"
    return root


DATA_ROOT = _resolve_data_root()

PATH = {
    "root": DATA_ROOT,
    "test_eegs": os.path.join(DATA_ROOT, "test_eegs"),
    "train_eegs": os.path.join(DATA_ROOT, "train_eegs"),
    "test_spectrograms": os.path.join(DATA_ROOT, "test_spectrograms"),
    "train_spectrograms": os.path.join(DATA_ROOT, "train_spectrograms"),
}

for k, p in PATH.items():
    if k == "root":
        continue
    if not os.path.exists(p):
        raise FileNotFoundError(f"Expected dataset folder missing: {k} -> {p}")

models = {
    "eeg": [
        "/kaggle/input/eegs-cv-kaggle/EEGS_CNN_0",
        "/kaggle/input/eegs-cv-kaggle/EEGS_CNN_1",
        "/kaggle/input/eegs-cv-kaggle/EEGS_CNN_2",
        "/kaggle/input/eegs-cv-kaggle/EEGS_CNN_3",
        "/kaggle/input/eegs-cv-kaggle/EEGS_CNN_4",
    ],
    "spec": [
        "/kaggle/input/specs-cv-kaggle/SPECS_CNN_0",
        "/kaggle/input/specs-cv-kaggle/SPECS_CNN_1",
        "/kaggle/input/specs-cv-kaggle/SPECS_CNN_2",
        "/kaggle/input/specs-cv-kaggle/SPECS_CNN_3",
        "/kaggle/input/specs-cv-kaggle/SPECS_CNN_4",
    ],
    "spec_fixed": [
        "/kaggle/input/specs-cv-kaggle-fixed/SPECS_CNN_0",
        "/kaggle/input/specs-cv-kaggle-fixed/SPECS_CNN_1",
        "/kaggle/input/specs-cv-kaggle-fixed/SPECS_CNN_2",
        "/kaggle/input/specs-cv-kaggle-fixed/SPECS_CNN_3",
        "/kaggle/input/specs-cv-kaggle-fixed/SPECS_CNN_4",
    ],
    "custom": [
        "/kaggle/input/custom-cv-kaggle/custom_SPECS_CNN_0",
        "/kaggle/input/custom-cv-kaggle/custom_SPECS_CNN_1",
        "/kaggle/input/custom-cv-kaggle/custom_SPECS_CNN_2",
        "/kaggle/input/custom-cv-kaggle/custom_SPECS_CNN_3",
        "/kaggle/input/custom-cv-kaggle/custom_SPECS_CNN_4",
    ],
    "custom_fixed": [
        "/kaggle/input/custom-cv-kaggle-fixed/custom_SPECS_CNN_0",
        "/kaggle/input/custom-cv-kaggle-fixed/custom_SPECS_CNN_1",
        "/kaggle/input/custom-cv-kaggle-fixed/custom_SPECS_CNN_2",
        "/kaggle/input/custom-cv-kaggle-fixed/custom_SPECS_CNN_3",
        "/kaggle/input/custom-cv-kaggle-fixed/custom_SPECS_CNN_4",
    ],
    "HMS": [
        "/kaggle/input/hms-xtra/HMSmodel_0",
        "/kaggle/input/hms-xtra/HMSmodel_1",
        "/kaggle/input/hms-xtra/HMSmodel_2",
        "/kaggle/input/hms-xtra/HMSmodel_3",
        "/kaggle/input/hms-xtra/HMSmodel_4",
    ],
}
DEBUG = False




## === cell 2
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # faster convs with fixed shapes
    torch.backends.cudnn.deterministic = False


set_seed(42)




## === cell 3
def normalize(spec, epsilon=1e-6, NATURAL=False):
    if NATURAL:
        spec = np.clip(spec, np.exp(-4), np.exp(8))
        spec = np.log(spec)
    else:
        spec = np.clip(spec, np.exp(-4), np.exp(6))
        spec = np.log10(spec)

    mask = ~np.isnan(spec)
    mean = np.mean(spec[mask])
    std = np.std(spec[mask])
    spec[mask] = spec[mask] - mean
    if std > 0:
        spec[mask] /= std + epsilon

    return spec




## === cell 4
from scipy.signal import butter, lfilter


_BLP_BA = butter(4, 20 / (0.5 * 200), btype="low", analog=False)


def butter_lowpass_filter(data, cutoff_freq=20, sampling_rate=200, order=4):
    b, a = _BLP_BA
    return lfilter(b, a, data, axis=0)




## === cell 5
from scipy.ndimage import gaussian_filter
from scipy.signal import (
    iirnotch,
    filtfilt,
    butter as scipy_butter,
    spectrogram as scipy_spectrogram,
)


_FS = 200
_B_NOTCH, _A_NOTCH = iirnotch(w0=60, Q=30, fs=_FS)
_NYQ = 0.5 * _FS
_LOW = 0.7 / _NYQ
_HIGH = 20 / _NYQ
_B_BP, _A_BP = scipy_butter(5, [_LOW, _HIGH], btype="band")

_ELECTRODE_PAIR_NAME_LOCATIONS = {
    "LL": ["Fp1", "F7", "T3", "T5", "O1"],
    "RL": ["Fp2", "F8", "T4", "T6", "O2"],
    "LP": ["Fp1", "F3", "C3", "P3", "O1"],
    "RP": ["Fp2", "F4", "C4", "P4", "O2"],
}


def create_spectrogram_cpu(
    eeg_data,
    eeg_id,
    low_cut_freq=0.7,
    high_cut_freq=20,
    order_band=5,
    nperseg=500,
    noverlap=200,
    nfft=1024,
    sigma_gaussian=0.7,
    mean_montage_names=4,
):
    spectrogram_out = None
    processed_eeg = {}

    for i, (montage, locs) in enumerate(_ELECTRODE_PAIR_NAME_LOCATIONS.items()):
        for j in range(4):
            s = eeg_data[locs[j]].values - eeg_data[locs[j + 1]].values
            if np.isnan(s).mean() < 1:
                m = np.nanmean(s)
                s = np.nan_to_num(s, nan=m)
            else:
                s = np.zeros_like(s)

            try:
                s = filtfilt(_B_NOTCH, _A_NOTCH, s)
            except Exception:
                pass
            try:
                s = filtfilt(_B_BP, _A_BP, s)
            except Exception:
                pass

            f, t, Sxx = scipy_spectrogram(
                s,
                fs=_FS,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                scaling="density",
                mode="psd",
            )
            valid = (f >= 0.59) & (f <= 20)
            Sxx = Sxx[valid, :].astype(np.float32)

            Sxx = np.clip(Sxx, np.exp(-4), np.exp(6))
            Sxx = np.log10(Sxx)
            mean = Sxx.mean()
            std = Sxx.std()
            Sxx = (Sxx - mean) / (std + 1e-6)

            if spectrogram_out is None:
                spectrogram_out = np.zeros(
                    (Sxx.shape[0], Sxx.shape[1], 4), dtype=np.float32
                )
            spectrogram_out[:, :, i] += Sxx

            processed_eeg[f"{locs[j]}_{locs[j + 1]}"] = s
            processed_eeg[montage] = processed_eeg.get(montage, 0) + s

        spectrogram_out[:, :, i] /= float(mean_montage_names)

    if sigma_gaussian and sigma_gaussian > 0:
        spectrogram_out = gaussian_filter(
            spectrogram_out, sigma=(sigma_gaussian, sigma_gaussian, 0)
        )

    ekg = eeg_data["EKG"].values
    if np.isnan(ekg).mean() < 1:
        ekg = np.nan_to_num(ekg, nan=np.nanmean(ekg))
    else:
        ekg = np.zeros_like(ekg)
    try:
        ekg = filtfilt(_B_NOTCH, _A_NOTCH, ekg)
        ekg = filtfilt(_B_BP, _A_BP, ekg)
    except Exception:
        pass
    processed_eeg["EKG"] = ekg

    return spectrogram_out, processed_eeg




## === cell 6
submission = pd.read_csv(os.path.join(PATH["root"], "sample_submission.csv"))
votes = [c for c in submission.columns if "_vote" in c]

test = pd.read_csv(os.path.join(PATH["root"], "test.csv"))
if DEBUG:
    test = test.iloc[:200].copy()



## === cell 7
any_spec_id = int(test.spectrogram_id.iloc[0])
spec0 = pd.read_parquet(
    os.path.join(PATH["test_spectrograms"], f"{any_spec_id}.parquet")
)
LL = [c for c in spec0.columns if "LL" in c]
RL = [c for c in spec0.columns if "RL" in c]
LP = [c for c in spec0.columns if "LP" in c]
RP = [c for c in spec0.columns if "RP" in c]
if not (LL and RL and LP and RP):
    raise RuntimeError("Failed to infer spectrogram channel groups (LL/RL/LP/RP).")
del spec0
gc.collect()



## === cell 8
custom_dir = "/kaggle/working/custom"
os.makedirs(custom_dir, exist_ok=True)

from concurrent.futures import ProcessPoolExecutor, as_completed


def _build_one_custom(eeg_id: int):
    out_path = os.path.join(custom_dir, f"{eeg_id}.npy")
    if os.path.exists(out_path):
        return eeg_id, True
    eeg = pd.read_parquet(os.path.join(PATH["test_eegs"], f"{eeg_id}.parquet"))
    mask = np.isnan(eeg.values).max(1)
    spec_cpu, _ = create_spectrogram_cpu(
        eeg,
        eeg_id,
        low_cut_freq=0.7,
        high_cut_freq=20,
        order_band=5,
        nperseg=500,
        noverlap=200,
        nfft=1024,
        sigma_gaussian=0.7,
        mean_montage_names=4,
    )
    _, w, _ = spec_cpu.shape
    mask_r = resize(
        mask.astype(np.float32), (w, 1), preserve_range=True, anti_aliasing=False
    )
    spec_cpu[:, mask_r[:, 0] > 0.5, :] = 0
    np.save(out_path, spec_cpu.astype(np.float32))
    return eeg_id, True


eeg_ids = test.eeg_id.unique().tolist()
max_workers = min(8, os.cpu_count() or 2)
if DEBUG:
    max_workers = 2

with ProcessPoolExecutor(max_workers=max_workers) as ex:
    futs = [ex.submit(_build_one_custom, int(eid)) for eid in eeg_ids]
    for _ in tqdm.tqdm(
        as_completed(futs),
        total=len(futs),
        desc="Building custom specs (CPU, parallel)",
    ):
        pass

gc.collect()




## === cell 9
class HMS_DS(torch.utils.data.Dataset):
    def __init__(self, df, mode="test"):
        self.START = (10000 - 2048) // 2
        self.data = df.reset_index(drop=True)
        self.mode = mode

        self.eeg_ids = self.data["eeg_id"].to_numpy(np.int64)
        self.spec_ids = (
            self.data["spectrogram_id"].to_numpy(np.int64)
            if "spectrogram_id" in self.data.columns
            else None
        )

        self._eeg_buf = np.zeros((8, 2048), dtype=np.float32)
        self._spec_buf = np.zeros((1, 256, 256), dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eeg_id = int(self.eeg_ids[idx])
        spec_id = int(self.spec_ids[idx])

        eeg = pd.read_parquet(os.path.join(PATH["test_eegs"], f"{eeg_id}.parquet"))
        eeg = eeg.iloc[self.START : self.START + 2048]

        mask = ~np.isnan(eeg["Fp1"])
        self._eeg_buf[:, :] = 0.0

        self._eeg_buf[0][mask] = eeg["Fp1"][mask] - eeg["T3"][mask]
        self._eeg_buf[1][mask] = eeg["T3"][mask] - eeg["O1"][mask]
        self._eeg_buf[2][mask] = eeg["Fp1"][mask] - eeg["C3"][mask]
        self._eeg_buf[3][mask] = eeg["C3"][mask] - eeg["O1"][mask]
        self._eeg_buf[4][mask] = eeg["Fp2"][mask] - eeg["C4"][mask]
        self._eeg_buf[5][mask] = eeg["C4"][mask] - eeg["O2"][mask]
        self._eeg_buf[6][mask] = eeg["Fp2"][mask] - eeg["T4"][mask]
        self._eeg_buf[7][mask] = eeg["T4"][mask] - eeg["O2"][mask]

        eeg_np = np.clip(self._eeg_buf, -1024, 1024) / 32.0
        eeg_np = butter_lowpass_filter(eeg_np)
        eeg_t = torch.from_numpy(eeg_np).float()

        spec_df = pd.read_parquet(
            os.path.join(PATH["test_spectrograms"], f"{spec_id}.parquet")
        )
        spec_df = spec_df[LL + RL + LP + RP]
        t0 = 22
        spec_df = spec_df[t0 : t0 + 256]

        self._spec_buf[0, :, :64] = normalize(resize(spec_df[LP].values, (256, 64)))
        self._spec_buf[0, :, 64:128] = normalize(resize(spec_df[LL].values, (256, 64)))
        self._spec_buf[0, :, 128:-64] = normalize(resize(spec_df[RP].values, (256, 64)))
        self._spec_buf[0, :, -64:] = normalize(resize(spec_df[RL].values, (256, 64)))
        self._spec_buf[np.isnan(self._spec_buf)] = 0.0
        spec_t = torch.from_numpy(self._spec_buf.copy()).float()

        custom_path = os.path.join(custom_dir, f"{eeg_id}.npy")
        custom_spec = np.load(custom_path)
        custom_spec = np.concatenate(
            (
                custom_spec[:, :, 3],
                custom_spec[:, :, 2],
                custom_spec[:, :, 1],
                custom_spec[:, :, 0],
            ),
            axis=0,
        )
        custom_spec = resize(custom_spec, (256, 300))
        t1 = 22
        self._spec_buf[0] = custom_spec[:, t1 : t1 + 256]
        self._spec_buf[np.isnan(self._spec_buf)] = 0.0
        custom_t = torch.from_numpy(self._spec_buf.copy()).float()

        return eeg_id, spec_id, eeg_t, spec_t, custom_t




## === cell 10
import torchvision


class EEG2Img(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(1, 3, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.proj(x)


class EEGBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = EEG2Img()
        self.backbone = torchvision.models.efficientnet_b0(weights=None)

    def forward(self, x):
        x = self.adapter(x)
        return self.backbone(x)


def make_effnet_1ch():
    m = torchvision.models.efficientnet_b0(weights=None)
    conv0 = m.features[0][0]
    if conv0.in_channels != 1:
        new_conv = nn.Conv2d(
            1,
            conv0.out_channels,
            kernel_size=conv0.kernel_size,
            stride=conv0.stride,
            padding=conv0.padding,
            bias=(conv0.bias is not None),
        )
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        if new_conv.bias is not None:
            nn.init.zeros_(new_conv.bias)
        m.features[0][0] = new_conv
    return m


class HMSmodel(nn.Module):
    def __init__(self, models_list):
        super().__init__()
        self.models = torch.nn.ModuleList(models_list)
        self.FC = nn.Linear(1280 * len(models_list), 6).to(device)

        for model in self.models:
            for p in model.parameters():
                p.requires_grad = False

        weights = []
        bias = 0
        for model in self.models:
            model.features[8][0].weight.requires_grad = True
            model.classifier[0] = nn.Dropout(0.8)
            weights.append(model.classifier[1].weight)
            bias += model.classifier[1].bias
            model.classifier[1] = nn.Identity()

        self.FC.weight = nn.Parameter(torch.cat(weights, 1).to(device))
        self.FC.bias = nn.Parameter(bias.to(device))

    def forward(self, X):
        X = torch.cat([self.models[i](X[i]) for i in range(len(self.models))], 1)
        OUT = self.FC(X)
        return OUT




## === cell 11


def _all_required_checkpoints_exist():
    req = []
    for fold in [0, 1, 2, 3, 4]:
        req.extend(
            [
                models["eeg"][fold],
                models["spec_fixed"][fold],
                models["custom_fixed"][fold],
                models["HMS"][fold],
            ]
        )
    return all(os.path.exists(p) for p in req)


def _write_uniform_submission(out_path="submission.csv"):
    sub = pd.read_csv(os.path.join(PATH["root"], "sample_submission.csv"))
    votes_local = [c for c in sub.columns if c.endswith("_vote")]
    sub[votes_local] = 1.0 / len(votes_local)
    sub = sub[["eeg_id"] + votes_local]
    sub.to_csv(out_path, index=False)
    print(f"Wrote {out_path} (uniform fallback)", sub.shape)
    print(sub.head())


if not _all_required_checkpoints_exist():
    _write_uniform_submission("submission.csv")
else:

    def _load_fold_models(fold: int):
        eeg_m = torch.load(models["eeg"][fold], map_location="cpu")
        spec_m = torch.load(models["spec_fixed"][fold], map_location="cpu")
        cust_m = torch.load(models["custom_fixed"][fold], map_location="cpu")
        hms_m = torch.load(models["HMS"][fold], map_location="cpu")
        return eeg_m, spec_m, cust_m, hms_m

    batch_size = 64
    num_workers = min(4, os.cpu_count() or 2)
    if DEBUG:
        num_workers = 0

    test_ds = HMS_DS(test, mode="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    all_fold_preds = []
    all_ids = None

    for fold in [0, 1, 2, 3, 4]:
        eeg_model, spec_model, custom_model, hms = _load_fold_models(fold)
        eeg_model.to(device).eval()
        spec_model.to(device).eval()
        custom_model.to(device).eval()
        hms.to(device).eval()

        preds = []
        ids = []
        with torch.no_grad():
            for batch in tqdm.tqdm(test_loader, desc=f"Inference fold {fold}"):
                eeg_id, spec_id, eeg_t, spec_t, custom_t = batch
                eeg_t = eeg_t.to(device, non_blocking=True)
                spec_t = spec_t.to(device, non_blocking=True)
                custom_t = custom_t.to(device, non_blocking=True)
                out = hms((eeg_t, spec_t, custom_t))
                prob = F.softmax(out, dim=1).detach().cpu().numpy().astype(np.float64)
                preds.append(prob)
                ids.extend([int(x) for x in eeg_id])

        preds = np.concatenate(preds, axis=0)
        all_fold_preds.append(preds)
        if all_ids is None:
            all_ids = np.asarray(ids, dtype=np.int64)

        del eeg_model, spec_model, custom_model, hms
        torch.cuda.empty_cache()
        gc.collect()

    preds = np.mean(np.stack(all_fold_preds, axis=0), axis=0)

    pred_df = pd.DataFrame(preds, columns=votes)
    pred_df["eeg_id"] = all_ids.tolist()

    sub = pd.read_csv(os.path.join(PATH["root"], "sample_submission.csv"))
    sub = sub.merge(pred_df, on="eeg_id", how="left", suffixes=("", "_pred"))

    for c in votes:
        if c not in sub.columns or sub[c].isna().any():
            if f"{c}_pred" in sub.columns:
                sub[c] = sub[f"{c}_pred"]
            else:
                sub[c] = sub[c].fillna(1.0 / len(votes))
        if f"{c}_pred" in sub.columns:
            sub[c] = sub[c].fillna(sub[f"{c}_pred"])
            sub.drop(columns=[f"{c}_pred"], inplace=True)

    sub[votes] = sub[votes].astype(np.float64)
    sub[votes] = sub[votes].clip(1e-12, 1.0)
    sub[votes] = sub[votes].div(sub[votes].sum(axis=1), axis=0)

    sub = sub[["eeg_id"] + votes]
    assert list(sub.columns) == ["eeg_id"] + votes
    row_sums = sub[votes].sum(axis=1).to_numpy()
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        sub[votes] = sub[votes].div(sub[votes].sum(axis=1), axis=0)

    sub.to_csv("submission.csv", index=False)
    print("Wrote submission.csv", sub.shape)
    print(sub.head())
