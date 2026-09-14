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

Lower is better

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
test_loader = DataLoader(
    test_ds,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=0,
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
            self.convs.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=h_dim,
                    kernel_size=(1, k),
                    padding="same",
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
    Change (stability): handle checkpoints saved as {'state_dict': ...} / {'model': ...}
    without altering model architecture or weights.
    """
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        return obj
    raise TypeError(f"Unsupported checkpoint type: {type(obj)}")


def _autodetect_model_files(preferred_dir: Path) -> List[Path]:
    """
    Change (score-impacting): robustly find .pth files so we avoid the weak prior fallback
    when weights are available somewhere under /kaggle/input.
    """
    candidates: List[Path] = []
    if preferred_dir is not None:
        candidates.append(preferred_dir)
    candidates.append(Path("/kaggle/input") / CFG.exp_id)
    candidates.append(DATA_PATH / CFG.exp_id)

    found: List[Path] = []
    for d in candidates:
        try:
            if d is not None and d.exists() and d.is_dir():
                files = sorted(d.glob("*.pth"))
                if len(files) > 0:
                    return files
        except Exception:
            pass

    base = Path("/kaggle/input")
    if base.exists():
        for d1 in base.iterdir():
            if not d1.is_dir():
                continue
            try:
                files = sorted(d1.glob("*.pth"))
                if len(files) > 0:
                    found.extend(files)
                    continue
                for d2 in d1.iterdir():
                    if d2.is_dir():
                        files2 = sorted(d2.glob("*.pth"))
                        if len(files2) > 0:
                            found.extend(files2)
            except Exception:
                continue

    return sorted(found)


model_files = _autodetect_model_files(CFG.model_path)
print(f"Found {len(model_files)} .pth file(s).")
if len(model_files) > 0:
    print(f"First weight file: {model_files[0]}")
else:
    print("WARNING: No .pth files found anywhere under /kaggle/input (bounded search).")

models: List[nn.Module] = []
if len(model_files) == 0:
    print("WARNING: Running fallback mode (train prior).")
else:
    for fold, file in enumerate(model_files):
        print(f"Load model from {file}...")
        fold_model = DilatedInceptionWaveNet()
        state_obj = torch.load(file, map_location=CFG.device)
        state = _unwrap_state_dict(state_obj)
        fold_model.load_state_dict(state, strict=True)
        fold_model = fold_model.to(CFG.device)
        models.append(fold_model)
    print(f"Loaded {len(models)} model(s).")



## === cell 15
gc.collect()




## === cell 16
@torch.no_grad()
def _infer(inputs: Dict[str, Tensor], models: List[nn.Module]) -> Tensor:
    if len(models) == 0:
        raise RuntimeError("No models loaded; inference with models is unavailable.")

    n_models = len(models)
    y_pred = None
    for i, model in enumerate(models):
        model.eval()
        y_pred_fold = F.softmax(model(inputs), dim=1) / n_models  # (B, N_CLASSES)
        if y_pred is None:
            y_pred = y_pred_fold
        else:
            y_pred += y_pred_fold

    return y_pred




## === cell 17
train_path = DATA_PATH / "train.csv"
assert train_path.exists(), f"Cannot find train.csv under {DATA_PATH}"
train_votes = pd.read_csv(train_path, usecols=TGT_VOTE_COLS)
prior = train_votes[TGT_VOTE_COLS].sum(axis=0).values.astype("float64")
prior = prior / max(prior.sum(), 1e-12)
prior = np.clip(prior, 1e-12, 1.0)
prior = prior / prior.sum()
prior = prior.astype("float32")

ALPHA_PRIOR = 0.02  # 2% prior mix

if len(models) > 0:
    y_preds_list = []
    for i, batch_data in enumerate(test_loader):
        batch_data["x"] = batch_data["x"].to(CFG.device, non_blocking=True)
        y_pred = _infer(batch_data, models)
        y_preds_list.append(y_pred.detach().cpu().numpy())
    y_preds = np.vstack(y_preds_list).astype("float32")
else:
    y_preds = np.tile(prior[None, :], (len(test_loader.dataset), 1)).astype("float32")

y_preds = (1.0 - ALPHA_PRIOR) * y_preds + ALPHA_PRIOR * prior[None, :]

print(f"y_preds shape: {y_preds.shape}")
print(f"Sum of row 0 in y_preds (pre-final-norm): {float(np.sum(y_preds[0, :])):.6f}.")



## === cell 18
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



## === cell 19
sample_sub_path = DATA_PATH / "sample_submission.csv"
assert sample_sub_path.exists(), f"Cannot find sample_submission.csv under {DATA_PATH}"
sample_sub = pd.read_csv(sample_sub_path, usecols=["eeg_id"] + TGT_VOTE_COLS)

pred_df = pd.DataFrame({"eeg_id": test_loader.dataset.metadata["eeg_id"].values})
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
vals = np.clip(vals, 1e-12, 1.0)
vals = vals / np.clip(vals.sum(axis=1, keepdims=True), 1e-12, None)
sub[TGT_VOTE_COLS] = vals.astype("float32")

sub.to_csv("submission.csv", index=False)

print("===== Submission Demo =====")
print(sub.head())
print(f"Row-sum check (first 5): {sub[TGT_VOTE_COLS].sum(axis=1).head().values}")
print(
    f"Saved submission.csv with shape {sub.shape} at {Path('submission.csv').resolve()}"
)

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
MergeError                                Traceback (most recent call last)
/tmp/ipykernel_55/2970212934.py in <cell line: 0>()
     10 pred_df[TGT_VOTE_COLS] = y_preds
     11 
---> 12 sub = sample_sub[["eeg_id"]].merge(
     13     pred_df, on="eeg_id", how="left", validate="one_to_one"
     14 )

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in merge(self, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
  10830         from pandas.core.reshape.merge import merge
  10831 
> 10832         return merge(
  10833             self,
  10834             right,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    168         )
    169     else:
--> 170         op = _MergeOperation(
    171             left_df,
    172             right_df,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in __init__(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)
    811         # are in fact unique.
    812         if validate is not None:
--> 813             self._validate_validate_kwd(validate)
    814 
    815     def _maybe_require_matching_dtypes(

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _validate_validate_kwd(self, validate)
   1651                 )
   1652             if not left_unique:
-> 1653                 raise MergeError(
   1654                     "Merge keys are not unique in left dataset; not a one-to-one merge"
   1655                 )

MergeError: Merge keys are not unique in left dataset; not a one-to-one merge
