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
PyYAML==6.0.3
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

0.5976041136518271

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import gc
import os
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import warnings
from pathlib import Path

try:
    from tqdm.notebook import tqdm
except Exception:
    from tqdm import tqdm

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader




## === cell 1
def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 2
DATA_PATH = Path("/kaggle/input/hms-harmful-brain-activity-classification")


class CFG:
    exp_id = "0217-15-11-37"
    model_path = Path("/kaggle/input/0217-15-11-37")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    feats = ["Fp1", "T3", "C3", "O1", "Fp2", "C4", "T4", "O2"]
    cast_eegs = True
    dataset = {
        "eeg": {
            "n_feats": 8,
            "apply_chris_magic_ch8": True,
            "normalize": True,
            "apply_butter_lowpass_filter": True,
            "apply_mu_law_encoding": False,
            "downsample": 5,
        }
    }

    batch_size = 32


N_CLASSES = 6
TGT_VOTE_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
EEG_FREQ = 200  # Hz
EEG_WLEN = 50  # sec
EEG_PTS = int(EEG_FREQ * EEG_WLEN)



## === cell 3
if not (DATA_PATH / "test.csv").exists():
    DATA_PATH = Path("/kaggle/input") / "hms-harmful-brain-activity-classification"

assert (DATA_PATH / "test.csv").exists(), f"Cannot find test.csv under {DATA_PATH}"




## === cell 4
def _get_eeg_window(file: Path) -> np.ndarray:
    """Return cropped EEG window (middle 50-sec window)."""
    eeg = pd.read_parquet(file, columns=CFG.feats)
    n_pts = len(eeg)

    if n_pts < EEG_PTS:
        pad_len = EEG_PTS - n_pts
        eeg = eeg.copy()
        for c in CFG.feats:
            col = eeg[c].values
            if CFG.cast_eegs:
                col = col.astype("float32")
            mean = np.nanmean(col)
            if np.isnan(col).mean() < 1:
                col = np.nan_to_num(col, nan=mean)
            else:
                col[:] = 0
            col = np.concatenate([col, np.zeros(pad_len, dtype=col.dtype)])
            eeg[c] = col[: len(eeg) + pad_len]
        eeg = eeg.iloc[:EEG_PTS]
    else:
        offset = (n_pts - EEG_PTS) // 2
        eeg = eeg.iloc[offset : offset + EEG_PTS]

    eeg_win = np.zeros((EEG_PTS, len(CFG.feats)))
    for j, col in enumerate(CFG.feats):
        if CFG.cast_eegs:
            eeg_raw = eeg[col].values.astype("float32")
        else:
            eeg_raw = eeg[col].values

        mean = np.nanmean(eeg_raw)
        if np.isnan(eeg_raw).mean() < 1:
            eeg_raw = np.nan_to_num(eeg_raw, nan=mean)
        else:
            eeg_raw[:] = 0
        eeg_win[:, j] = eeg_raw

    return eeg_win




## === cell 5
test = pd.read_csv(DATA_PATH / "test.csv")
print(f"Test data shape | {test.shape}")

test = test.drop_duplicates(subset=["eeg_id"], keep="first").reset_index(drop=True)
print(f"Test data shape after drop_duplicates(eeg_id) | {test.shape}")



## === cell 6
uniq_eeg_ids = test["eeg_id"].unique()
n_uniq_eeg_ids = len(uniq_eeg_ids)

all_eegs = {}
missing = 0
for i, eeg_id in tqdm(enumerate(uniq_eeg_ids), total=n_uniq_eeg_ids):
    f = DATA_PATH / "test_eegs" / f"{eeg_id}.parquet"
    if not f.exists():
        missing += 1
        continue
    eeg_win = _get_eeg_window(f)
    all_eegs[eeg_id] = eeg_win

assert len(all_eegs) == len(
    uniq_eeg_ids
), f"Missing {missing} EEG parquet files in test_eegs."
print(f"Demo EEG shape | {list(all_eegs.values())[0].shape}")



## === cell 7
gc.collect()




## === cell 8
class EEGDataset(Dataset):
    """Dataset for pure raw EEG signals."""

    def __init__(
        self,
        data: Dict[str, Any],
        split: str,
        **dataset_cfg: Any,
    ) -> None:
        self.metadata = data["meta"].reset_index(drop=True)
        self.all_eegs = data["eeg"]
        self.dataset_cfg = dataset_cfg

        self.eeg_params = dataset_cfg["eeg"]
        self.eeg_trafo = _EEGTransformer(**self.eeg_params)

        self._infer = True if split == "test" else False

        self._set_n_samples()

        self._stream_X = True if self.all_eegs is None else False
        self._X, self._y = self._transform()

    def _set_n_samples(self) -> None:
        nunique = self.metadata["eeg_id"].nunique()
        assert (
            len(self.metadata) == nunique
        ), f"metadata has {len(self.metadata)} rows but {nunique} unique eeg_id."
        self._n_samples = len(self.metadata)

    def _transform(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Transform feature and target matrices."""
        if self.eeg_params["downsample"] is not None:
            eeg_len = int(EEG_PTS / self.eeg_params["downsample"])
        else:
            eeg_len = int(EEG_PTS)

        if not self._stream_X:
            X = np.zeros(
                (self._n_samples, eeg_len, self.eeg_params["n_feats"]), dtype="float32"
            )
        else:
            X = None

        y = (
            np.zeros((self._n_samples, N_CLASSES), dtype="float32")
            if not self._infer
            else None
        )

        for i, row in tqdm(self.metadata.iterrows(), total=len(self.metadata)):
            if not self._stream_X:
                eeg = self.all_eegs[row["eeg_id"]]
                x = self.eeg_trafo.transform(eeg)
                X[i] = x

            if not self._infer:
                y[i] = row[TGT_VOTE_COLS].values.astype("float32")

        return X, y

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        if self._X is None:
            eeg_id = self.metadata.loc[idx, "eeg_id"]
            eeg = self.all_eegs[eeg_id] if self.all_eegs is not None else None
            if eeg is None:
                raise RuntimeError(
                    "Streaming mode requires access to EEGs; all_eegs is None."
                )
            x = self.eeg_trafo.transform(eeg).astype("float32")
        else:
            x = self._X[idx, ...]

        data_sample = {"x": torch.tensor(x, dtype=torch.float32)}
        if not self._infer:
            data_sample["y"] = torch.tensor(self._y[idx, :], dtype=torch.float32)

        return data_sample


class _EEGTransformer(object):
    """Data transformer for raw EEG signals."""

    FEAT2CODE = {f: i for i, f in enumerate(CFG.feats)}

    def __init__(
        self,
        n_feats: int,
        apply_chris_magic_ch8: bool = True,
        normalize: bool = True,
        apply_butter_lowpass_filter: bool = True,
        apply_mu_law_encoding: bool = False,
        downsample: Optional[int] = None,
    ) -> None:
        self.n_feats = n_feats
        self.apply_chris_magic_ch8 = apply_chris_magic_ch8
        self.normalize = normalize
        self.apply_butter_lowpass_filter = apply_butter_lowpass_filter
        self.apply_mu_law_encoding = apply_mu_law_encoding
        self.downsample = downsample

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_ = x.copy()
        if self.apply_chris_magic_ch8:
            x_ = self._apply_chris_magic_ch8(x_)

        if self.normalize:
            x_ = np.clip(x_, -1024, 1024)
            x_ = np.nan_to_num(x_, nan=0) / 32.0

        if self.apply_butter_lowpass_filter:
            x_ = self._butter_lowpass_filter(x_)

        if self.apply_mu_law_encoding:
            x_ = self._quantize_data(x_, 1)

        if self.downsample is not None:
            x_ = x_[:: self.downsample, :]

        return x_

    def _apply_chris_magic_ch8(self, x: np.ndarray) -> np.ndarray:
        x_tmp = np.zeros((EEG_PTS, self.n_feats), dtype="float32")

        x_tmp[:, 0] = x[:, self.FEAT2CODE["Fp1"]] - x[:, self.FEAT2CODE["T3"]]
        x_tmp[:, 1] = x[:, self.FEAT2CODE["T3"]] - x[:, self.FEAT2CODE["O1"]]

        x_tmp[:, 2] = x[:, self.FEAT2CODE["Fp1"]] - x[:, self.FEAT2CODE["C3"]]
        x_tmp[:, 3] = x[:, self.FEAT2CODE["C3"]] - x[:, self.FEAT2CODE["O1"]]

        x_tmp[:, 4] = x[:, self.FEAT2CODE["Fp2"]] - x[:, self.FEAT2CODE["C4"]]
        x_tmp[:, 5] = x[:, self.FEAT2CODE["C4"]] - x[:, self.FEAT2CODE["O2"]]

        x_tmp[:, 6] = x[:, self.FEAT2CODE["Fp2"]] - x[:, self.FEAT2CODE["T4"]]
        x_tmp[:, 7] = x[:, self.FEAT2CODE["T4"]] - x[:, self.FEAT2CODE["O2"]]

        return x_tmp

    def _butter_lowpass_filter(self, data, cutoff_freq=20, sampling_rate=200, order=4):
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        filtered_data = lfilter(b, a, data, axis=0)
        return filtered_data

    def _quantize_data(self, data, classes):
        mu_x = self._mu_law_encoding(data, classes)
        return mu_x

    def _mu_law_encoding(self, data, mu):
        mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
        return mu_x




## === cell 9
gc.collect()



## === cell 10
test_data = {"meta": test, "eeg": all_eegs}
test_ds = EEGDataset(test_data, "test", **CFG.dataset)

_num_workers = 2 if os.cpu_count() and os.cpu_count() >= 4 else 0

test_loader = DataLoader(
    test_ds,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
)
print(f"There are {len(test_loader.dataset)} test samples to infer.")



## === cell 11
gc.collect()




## === cell 12
class DilatedInceptionWaveNet(nn.Module):
    """WaveNet architecture with dilated inception conv."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

        kernel_size = [2, 3, 6, 7]

        self.wave_module = nn.Sequential(
            _WaveBlock(12, 1, 16, kernel_size, _DilatedInception),
            _WaveBlock(8, 16, 32, kernel_size, _DilatedInception),
            _WaveBlock(4, 32, 64, kernel_size, _DilatedInception),
            _WaveBlock(1, 64, 64, kernel_size, _DilatedInception),
        )
        self.output = nn.Sequential(
            nn.Linear(64 * 4, 64), nn.ReLU(), nn.Linear(64, N_CLASSES)
        )

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        x = inputs["x"]
        bs, length, in_dim = x.shape
        x = x.transpose(1, 2).unsqueeze(dim=2)  # (B, C, N, L)

        x_ll_1 = self.wave_module(x[:, 0:1, :])
        x_ll_2 = self.wave_module(x[:, 1:2, :])
        x_ll = (
            F.adaptive_avg_pool2d(x_ll_1, (1, 1))
            + F.adaptive_avg_pool2d(x_ll_2, (1, 1))
        ) / 2

        x_rl_1 = self.wave_module(x[:, 2:3, :])
        x_rl_2 = self.wave_module(x[:, 3:4, :])
        x_rl = (
            F.adaptive_avg_pool2d(x_rl_1, (1, 1))
            + F.adaptive_avg_pool2d(x_rl_2, (1, 1))
        ) / 2

        x_lp_1 = self.wave_module(x[:, 4:5, :])
        x_lp_2 = self.wave_module(x[:, 5:6, :])
        x_lp = (
            F.adaptive_avg_pool2d(x_lp_1, (1, 1))
            + F.adaptive_avg_pool2d(x_lp_2, (1, 1))
        ) / 2

        x_rp_1 = self.wave_module(x[:, 6:7, :])
        x_rp_2 = self.wave_module(x[:, 7:8, :])
        x_rp = (
            F.adaptive_avg_pool2d(x_rp_1, (1, 1))
            + F.adaptive_avg_pool2d(x_rp_2, (1, 1))
        ) / 2

        x = torch.cat([x_ll, x_rl, x_lp, x_rp], axis=1).reshape(bs, -1)
        output = self.output(x)

        return output


class _WaveBlock(nn.Module):
    def __init__(
        self,
        n_layers: int,
        in_dim: int,
        h_dim: int,
        kernel_size: Union[int, List[int]],
        conv_module: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()

        self.n_layers = n_layers
        self.dilation_rates = [2**l for l in range(n_layers)]

        self.in_conv = nn.Conv2d(in_dim, h_dim, kernel_size=(1, 1))
        self.gated_tcns = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        for layer in range(n_layers):
            c_in, c_out = h_dim, h_dim
            self.gated_tcns.append(
                _GatedTCN(
                    in_dim=c_in,
                    h_dim=c_out,
                    kernel_size=kernel_size,
                    dilation_factor=self.dilation_rates[layer],
                    conv_module=conv_module,
                )
            )
            self.skip_convs.append(nn.Conv2d(h_dim, h_dim, kernel_size=(1, 1)))

        nn.init.xavier_uniform_(
            self.in_conv.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.zeros_(self.in_conv.bias)
        for i in range(len(self.skip_convs)):
            nn.init.xavier_uniform_(
                self.skip_convs[i].weight, gain=nn.init.calculate_gain("relu")
            )
            nn.init.zeros_(self.skip_convs[i].bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_conv(x)

        x_skip = x
        for layer in range(self.n_layers):
            x = self.gated_tcns[layer](x)
            x = self.skip_convs[layer](x)
            x_skip = x_skip + x

        return x_skip


class _GatedTCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        kernel_size: Union[int, List[int]],
        dilation_factor: int,
        dropout: Optional[float] = None,
        conv_module: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()

        if conv_module is None:
            self.filt = nn.Conv2d(
                in_channels=in_dim,
                out_channels=h_dim,
                kernel_size=(1, kernel_size),
                dilation=dilation_factor,
            )
            self.gate = nn.Conv2d(
                in_channels=in_dim,
                out_channels=h_dim,
                kernel_size=(1, kernel_size),
                dilation=dilation_factor,
            )
        else:
            self.filt = conv_module(
                in_channels=in_dim,
                out_channels=h_dim,
                kernel_size=kernel_size,
                dilation=dilation_factor,
            )
            self.gate = conv_module(
                in_channels=in_dim,
                out_channels=h_dim,
                kernel_size=kernel_size,
                dilation=dilation_factor,
            )

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x: Tensor) -> Tensor:
        x_filt = torch.tanh(self.filt(x))
        x_gate = torch.sigmoid(self.gate(x))
        h = x_filt * x_gate
        if self.dropout is not None:
            h = self.dropout(h)

        return h


class _DilatedInception(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: List[int], dilation: int
    ) -> None:
        super().__init__()

        n_kernels = len(kernel_size)
        assert (
            out_channels % n_kernels == 0
        ), "`out_channels` must be divisible by #kernels."
        h_dim = out_channels // n_kernels

        self.convs = nn.ModuleList()
        for k in kernel_size:
            pad = (dilation * (k - 1)) // 2
            self.convs.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=h_dim,
                    kernel_size=(1, k),
                    padding=(0, pad),
                    dilation=dilation,
                ),
            )

    def forward(self, x: Tensor) -> Tensor:
        x_convs = []
        for conv in self.convs:
            x_convs.append(conv(x))
        h = torch.cat(x_convs, dim=1)
        return h




## === cell 13
gc.collect()




## === cell 14
def _unwrap_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    """
    Change (unblocks loading): handle checkpoints saved as {'state_dict': ...} / {'model': ...}.
    """
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        return obj
    raise TypeError(f"Unsupported checkpoint type: {type(obj)}")


def _strip_known_prefixes(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Change (unblocks loading): strip common wrappers so load_state_dict succeeds.
    """
    if not isinstance(state, dict) or len(state) == 0:
        return state

    prefixes = ("module.", "model.", "net.", "encoder.")
    keys = list(state.keys())
    for p in prefixes:
        n_pref = sum(k.startswith(p) for k in keys)
        if n_pref >= max(1, len(keys) // 2):
            return {k[len(p) :]: v for k, v in state.items()}
    return state


def _autodetect_model_files() -> List[Path]:
    """
    Change (unblocks submission): guard against non-existent CFG.model_path.
    Previously, rglob on a non-existent path could raise, preventing submission.csv
    and yielding "Not yielded".
    """
    preferred_roots = [
        CFG.model_path,
        Path("/kaggle/input") / CFG.exp_id,
        DATA_PATH / CFG.exp_id,
        Path("/kaggle/input") / CFG.exp_id / CFG.exp_id,
        DATA_PATH / CFG.exp_id / CFG.exp_id,
    ]

    exp_hits: List[Path] = []
    for r in preferred_roots + [Path("/kaggle/input")]:
        if r is None or not Path(r).exists():
            continue
        r = Path(r)
        if r.is_file() and r.suffix == ".pth":
            if CFG.exp_id in str(r):
                exp_hits.append(r)
            continue
        if r.is_dir():
            for p in r.rglob("*.pth"):
                if CFG.exp_id in str(p):
                    exp_hits.append(p)

    def _dedup(paths: List[Path]) -> List[Path]:
        seen = set()
        uniq = []
        for p in sorted(paths, key=lambda x: str(x)):
            s = str(p.resolve())
            if s not in seen:
                seen.add(s)
                uniq.append(p)
        return uniq

    exp_hits = _dedup(exp_hits)
    if len(exp_hits) > 0:
        return exp_hits

    any_hits: List[Path] = []
    for r in [CFG.model_path, Path("/kaggle/input")]:
        if r is None or not Path(r).exists():
            continue
        r = Path(r)
        if r.is_file() and r.suffix == ".pth":
            any_hits.append(r)
            continue
        if r.is_dir():
            for p in r.rglob("*.pth"):
                any_hits.append(p)

    return _dedup(any_hits)


model_files = _autodetect_model_files()
print(f"Found {len(model_files)} .pth file(s).")
if len(model_files) > 0:
    print(f"First weight file: {model_files[0]}")
else:
    print(
        "WARNING: No .pth files found; will use train-derived prior predictions (patient-smoothed)."
    )

models: List[nn.Module] = []

for fold, file in enumerate(model_files):
    try:
        print(f"Load model from {file}...")
        fold_model = DilatedInceptionWaveNet()
        state_obj = torch.load(file, map_location="cpu")
        state = _unwrap_state_dict(state_obj)
        state = _strip_known_prefixes(state)

        incompat = fold_model.load_state_dict(state, strict=False)
        missing = (
            list(incompat.missing_keys) if hasattr(incompat, "missing_keys") else []
        )
        unexpected = (
            list(incompat.unexpected_keys)
            if hasattr(incompat, "unexpected_keys")
            else []
        )
        if len(missing) > 0 or len(unexpected) > 0:
            print(
                f"NOTE: Loaded with strict=False for {file}. Missing={len(missing)}, unexpected={len(unexpected)}."
            )

        fold_model = fold_model.to(CFG.device)
        fold_model.eval()
        models.append(fold_model)
    except Exception as e:
        print(f"WARNING: Failed to load {file}: {repr(e)}. Skipping.")
        continue

if len(models) == 0:
    print(
        "WARNING: No models loaded; will use train-derived prior predictions (patient-smoothed)."
    )
else:
    print(f"Loaded {len(models)} model(s).")



## === cell 15
gc.collect()




## === cell 16
@torch.inference_mode()
def _infer(inputs: Dict[str, Tensor], models: List[nn.Module]) -> Tensor:
    if len(models) == 0:
        raise RuntimeError("No models loaded; inference with models is unavailable.")

    n_models = len(models)
    y_pred = None
    for i, model in enumerate(models):
        y_pred_fold = F.softmax(model(inputs), dim=1) / n_models  # (B, N_CLASSES)
        if y_pred is None:
            y_pred = y_pred_fold
        else:
            y_pred += y_pred_fold

    return y_pred




## === cell 17
def _kl_divergence(p_true: np.ndarray, p_pred: np.ndarray, eps: float = 1e-12) -> float:
    p_true = np.clip(p_true, eps, 1.0)
    p_true = p_true / np.clip(p_true.sum(axis=1, keepdims=True), eps, None)
    p_pred = np.clip(p_pred, eps, 1.0)
    p_pred = p_pred / np.clip(p_pred.sum(axis=1, keepdims=True), eps, None)
    return float(np.mean(np.sum(p_true * (np.log(p_true) - np.log(p_pred)), axis=1)))


@torch.inference_mode()
def _predict_proba_for_eegs(
    eeg_ids: np.ndarray,
    all_eegs_dict: Dict[int, np.ndarray],
    models: List[nn.Module],
    batch_size: int,
) -> np.ndarray:
    meta = pd.DataFrame({"eeg_id": eeg_ids})
    ds = EEGDataset({"meta": meta, "eeg": all_eegs_dict}, "test", **CFG.dataset)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    out = []
    for batch in loader:
        batch["x"] = batch["x"].to(CFG.device, non_blocking=True)
        out.append(_infer(batch, models).detach().cpu().numpy())
    return np.vstack(out).astype("float32")


train_path = DATA_PATH / "train.csv"
assert train_path.exists(), f"Cannot find train.csv under {DATA_PATH}"
train_df = pd.read_csv(train_path, usecols=["eeg_id", "patient_id"] + TGT_VOTE_COLS)

prior_global = train_df[TGT_VOTE_COLS].sum(axis=0).values.astype("float64")
prior_global = prior_global / max(prior_global.sum(), 1e-12)
prior_global = np.clip(prior_global, 1e-12, 1.0)
prior_global = (prior_global / prior_global.sum()).astype("float32")

patient_votes = train_df.groupby("patient_id")[TGT_VOTE_COLS].sum()
patient_probs = patient_votes.div(patient_votes.sum(axis=1), axis=0).astype("float32")

PATIENT_PRIOR_SMOOTH = 0.15
patient_probs_sm = (
    (1.0 - PATIENT_PRIOR_SMOOTH) * patient_probs.values
    + PATIENT_PRIOR_SMOOTH * prior_global[None, :]
).astype("float32")
patient_probs_sm = np.clip(patient_probs_sm, 1e-12, 1.0)
patient_probs_sm = patient_probs_sm / patient_probs_sm.sum(axis=1, keepdims=True)
patient_prior_map = {
    int(pid): patient_probs_sm[i] for i, pid in enumerate(patient_probs.index.values)
}

prior = prior_global.copy()

ALPHA_PRIOR = 0.02 if len(models) == 0 else 0.005

if len(models) > 0:
    agg = train_df.groupby(["eeg_id", "patient_id"], as_index=False)[
        TGT_VOTE_COLS
    ].sum()
    y_true = agg[TGT_VOTE_COLS].values.astype("float32")
    y_true = y_true / np.clip(y_true.sum(axis=1, keepdims=True), 1e-12, None)

    patients = agg["patient_id"].values.astype("int64")
    uniq_pat = np.unique(patients)
    rng = np.random.RandomState(42)
    rng.shuffle(uniq_pat)
    n_val_pat = max(1, int(0.1 * len(uniq_pat)))
    val_pat = set(uniq_pat[:n_val_pat])

    val_mask = np.array([p in val_pat for p in patients])
    eeg_ids_val = agg.loc[val_mask, "eeg_id"].values.astype("int64")
    y_true_val = y_true[val_mask]

    MAX_TUNE_EEGS = 256
    if len(eeg_ids_val) > MAX_TUNE_EEGS:
        eeg_ids_val = eeg_ids_val[:MAX_TUNE_EEGS]
        y_true_val = y_true_val[:MAX_TUNE_EEGS]

    val_eegs: Dict[int, np.ndarray] = {}
    for eeg_id in tqdm(eeg_ids_val, total=len(eeg_ids_val), desc="Load val EEGs"):
        f = DATA_PATH / "train_eegs" / f"{eeg_id}.parquet"
        if f.exists():
            val_eegs[int(eeg_id)] = _get_eeg_window(f)

    eeg_ids_val_loaded = np.array(
        [eid for eid in eeg_ids_val if int(eid) in val_eegs], dtype="int64"
    )

    if len(eeg_ids_val_loaded) >= 32:
        y_true_map = {int(eid): y_true_val[i] for i, eid in enumerate(eeg_ids_val)}
        y_true_val_loaded = np.vstack(
            [y_true_map[int(eid)] for eid in eeg_ids_val_loaded]
        ).astype("float32")

        y_pred_val = _predict_proba_for_eegs(
            eeg_ids=eeg_ids_val_loaded,
            all_eegs_dict=val_eegs,
            models=models,
            batch_size=CFG.batch_size,
        )

        alphas = np.array([0.0, 0.002, 0.005, 0.01, 0.02, 0.03], dtype="float32")
        best_alpha = float(ALPHA_PRIOR)
        best_kl = float("inf")
        for a in alphas:
            mix = (1.0 - a) * y_pred_val + a * prior_global[None, :]
            kl = _kl_divergence(y_true_val_loaded, mix)
            if kl < best_kl:
                best_kl = kl
                best_alpha = float(a)
        ALPHA_PRIOR = best_alpha
        print(f"Tuned ALPHA_PRIOR={ALPHA_PRIOR:.3f} on patient-val (KL={best_kl:.6f}).")
    else:
        print(
            "WARNING: Too few validation EEGs loaded to tune ALPHA_PRIOR; using default."
        )



## === cell 18
if len(models) > 0:
    y_preds_list = []
    for i, batch_data in enumerate(test_loader):
        batch_data["x"] = batch_data["x"].to(CFG.device, non_blocking=True)
        y_pred = _infer(batch_data, models)
        y_preds_list.append(y_pred.detach().cpu().numpy())
    y_preds = np.vstack(y_preds_list).astype("float32")
else:
    y_prior = np.zeros((len(test_loader.dataset), N_CLASSES), dtype="float32")
    test_pids = test_loader.dataset.metadata["patient_id"].values.astype("int64")
    for i, pid in enumerate(test_pids):
        y_prior[i] = patient_prior_map.get(int(pid), prior_global)
    y_preds = y_prior

y_preds = (1.0 - float(ALPHA_PRIOR)) * y_preds + float(ALPHA_PRIOR) * prior_global[
    None, :
]

print(f"y_preds shape: {y_preds.shape}")
print(f"Sum of row 0 in y_preds (pre-final-norm): {float(np.sum(y_preds[0, :])):.6f}.")



## === cell 19
y_preds = np.nan_to_num(
    y_preds, nan=1.0 / N_CLASSES, posinf=1.0 / N_CLASSES, neginf=1.0 / N_CLASSES
).astype("float32")
y_preds = np.clip(y_preds, 1e-12, 1.0).astype("float32")
row_sums = y_preds.sum(axis=1, keepdims=True)
row_sums = np.clip(row_sums, 1e-12, None)
y_preds = y_preds / row_sums

print(
    f"Post-normalization sum row 0: {y_preds[0].sum():.6f}, min={y_preds.min():.3e}, max={y_preds.max():.3e}"
)



## === cell 20
sample_sub_path = DATA_PATH / "sample_submission.csv"
assert sample_sub_path.exists(), f"Cannot find sample_submission.csv under {DATA_PATH}"
sample_sub = pd.read_csv(sample_sub_path, usecols=["eeg_id"] + TGT_VOTE_COLS)

sample_sub = sample_sub.drop_duplicates(subset=["eeg_id"], keep="first").reset_index(
    drop=True
)
assert sample_sub[
    "eeg_id"
].is_unique, "sample_submission eeg_id is still not unique after drop_duplicates."

pred_df = pd.DataFrame({"eeg_id": test_loader.dataset.metadata["eeg_id"].values})
assert pred_df[
    "eeg_id"
].is_unique, (
    "Prediction metadata eeg_id is not unique (test should be unique per eeg_id)."
)
assert (
    len(pred_df) == y_preds.shape[0]
), f"Rows {len(pred_df)} != preds {y_preds.shape[0]}"
pred_df[TGT_VOTE_COLS] = y_preds

sub = sample_sub[["eeg_id"]].merge(
    pred_df, on="eeg_id", how="left", validate="one_to_one"
)
assert (
    sub[TGT_VOTE_COLS].isna().sum().sum() == 0
), "Found missing predictions after merge."

vals = sub[TGT_VOTE_COLS].values.astype("float64")
vals = np.nan_to_num(
    vals, nan=1.0 / N_CLASSES, posinf=1.0 / N_CLASSES, neginf=1.0 / N_CLASSES
)
vals = np.clip(vals, 1e-12, 1.0)
vals = vals / np.clip(vals.sum(axis=1, keepdims=True), 1e-12, None)
sub[TGT_VOTE_COLS] = vals.astype("float32")

assert (
    list(sub.columns) == ["eeg_id"] + TGT_VOTE_COLS
), f"Bad submission columns: {sub.columns.tolist()}"
row_sum = sub[TGT_VOTE_COLS].sum(axis=1).values
assert np.all(np.isfinite(row_sum)), "Non-finite row sums in submission."
assert np.allclose(
    row_sum, 1.0, atol=1e-5
), f"Row sums not 1 within tolerance. min={row_sum.min()}, max={row_sum.max()}"

sub.to_csv("submission.csv", index=False)

print("===== Submission Demo =====")
print(sub.head())
print(f"Row-sum check (first 5): {sub[TGT_VOTE_COLS].sum(axis=1).head().values}")
print(
    f"Saved submission.csv with shape {sub.shape} at {Path('submission.csv').resolve()}"
)
