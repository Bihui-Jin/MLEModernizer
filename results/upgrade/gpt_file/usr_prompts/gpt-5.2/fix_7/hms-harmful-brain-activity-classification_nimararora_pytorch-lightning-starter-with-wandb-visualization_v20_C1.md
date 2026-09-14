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
joblib==1.5.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pyarrow==19.0.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
PyYAML==6.0.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
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
transformers==4.53.3
wandb==0.21.0

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

0.8884738274865326

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
CONFIG_STR = """
development: false
experiment: exp015
notes: impute zero

data_dir: /kaggle/input/hms-harmful-brain-activity-classification
output_dir: /kaggle/working
temp_dir: /kaggle/temp

processed_data_dir: null
prev_notebook_ver: /kaggle/input/pytorch-lightning-starter-with-wandb-visualization

seed: 42

model:
    class_name: BasicConvolution
    BasicConvolution:
        num_conv_layers: 16
        num_kernels: 64
        kernel_size: 3
        stride: 1
        dilation: 1
        activation: "tanh"
        fcn_nodes: 128
        dropout1_prob: 0
        dropout2_prob: 0

train:
    split_by_col: eeg_id
    one_per_split_by_col: true
    num_folds: 5
    batch_size: 256
    num_workers: 4
    accelerator: auto
    precision: 32
    gradient_clip_val: 1.0
    accumulate_grad_batches: 1
    check_val_every_n_epoch: 1
    deterministic: true
    impute_zero: true
    max_epochs: 17
    max_time: "00:01:00:00"
    limit_val_batches: 1.0
    limit_train_batches: 1.0

optimizer:
  lr: 0.0005
  class_name: AdamW
  SGD:
    momentum: 0.9
    weight_decay: 0
    nesterov: false
    dampening: 0
  AdamW:
    weight_decay: 0.01
  scheduler: cosine
  num_warmup_steps: 0
"""
print(CONFIG_STR)

from types import SimpleNamespace
import logging
import json
import yaml
import wandb
import os
from kaggle_secrets import UserSecretsClient
import pandas as pd
from pathlib import Path
import numpy as np
import joblib
from tqdm.notebook import tqdm
from typing import List, Optional


def configure_logger(level=logging.INFO):
    root_logger = logging.getLogger()
    if len(root_logger.handlers) < 2:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        root_logger.setLevel(level)


def info(module: str, message: str):
    logger = logging.getLogger(module)
    logger.info(message)


def json_to_py(json_cfg):
    return json.loads(json.dumps(json_cfg), object_hook=lambda d: SimpleNamespace(**d))


def wandb_login() -> bool:
    try:
        user_secrets = UserSecretsClient()
        os.environ["WANDB_API_KEY"] = user_secrets.get_secret("WANDB_API_KEY")
        os.environ["WANDB_ENTITY"] = user_secrets.get_secret("WANDB_ENTITY")
        return wandb.login()
    except Exception:
        return False


def impute_mean(df: pd.DataFrame) -> pd.DataFrame:
    arr = df.to_numpy(copy=True)
    if arr.size == 0:
        return df
    col_means = np.nanmean(arr, axis=0)
    inds = np.where(np.isnan(arr))
    if inds[0].size:
        arr[inds] = col_means[inds[1]]
    return pd.DataFrame(arr, columns=df.columns, index=df.index)


def impute_zero(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(0)


def create_directory(directory_path) -> bool:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        return True
    return False


def parquet_to_numpy(
    src_dir, dest_dir, do_impute_zero: bool, restrict_ids: Optional[set] = None
):
    create_directory(dest_dir)

    def copy_one_file(src_file, dest_dir, do_impute_zero):
        prefix = src_file.name.split(".")[0]
        out_path = Path(dest_dir) / (prefix + ".npy")
        if out_path.exists():
            return
        df = pd.read_parquet(src_file)
        if do_impute_zero:
            arr = impute_zero(df).to_numpy()
        else:
            arr = impute_mean(df).to_numpy()
        np.save(out_path, arr)

    all_files = list(Path(src_dir).glob("*.parquet"))
    if restrict_ids is not None:
        restrict_ids_str = set(map(str, restrict_ids))
        all_files = [p for p in all_files if p.stem in restrict_ids_str]

    if len(all_files) == 0:
        return

    joblib.Parallel(n_jobs=-1, backend="loky", batch_size=256)(
        joblib.delayed(copy_one_file)(filename, dest_dir, do_impute_zero)
        for filename in tqdm(all_files)
    )


def merge_csvs(src_csvs: List[str], tgt_csv: str, gby_col: str):
    dfs = [pd.read_csv(file) for file in src_csvs]
    result_df = pd.concat(dfs, ignore_index=True)
    result_df = result_df.groupby(gby_col, as_index=False).mean(numeric_only=True)
    result_df.to_csv(tgt_csv, index=False)




## === cell 1
import pytorch_lightning as ptl
from pathlib import Path, PosixPath
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Optional

EEG_SAMPRATE = 200
EEG_LENGTH = 50
EEG_FEATURES = 20
EEG_CLASSES = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]


def _pad_or_truncate(arr: np.ndarray, target_rows: int) -> np.ndarray:
    if arr.shape[0] == target_rows:
        return arr
    if arr.shape[0] > target_rows:
        return arr[:target_rows]
    pad = np.zeros((target_rows - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


class OffsetEEG(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        eeg_dir: PosixPath,
        spec_dir: PosixPath,
        processed_data: bool,
        do_impute_zero: bool,
        one_per_col: Optional[str],
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.eeg_dir = eeg_dir
        self.spec_dir = spec_dir
        self.processed_data = processed_data
        self.do_impute_zero = do_impute_zero
        self.one_per_col = one_per_col
        self.num_rows = EEG_SAMPRATE * EEG_LENGTH

        self._label_cols = [f"{cls}_vote" for cls in EEG_CLASSES]

        self._eeg_id = self.df["eeg_id"].to_numpy()
        self._eeg_sub_id = (
            self.df["eeg_sub_id"].to_numpy()
            if "eeg_sub_id" in self.df.columns
            else np.zeros(len(self.df), dtype=np.int64)
        )
        self._offset = (
            self.df["eeg_label_offset_seconds"].to_numpy(dtype=np.float32)
            * EEG_SAMPRATE
        ).astype(np.int64, copy=False)

        labels = self.df[self._label_cols].to_numpy(dtype=np.float32, copy=True)
        sums = labels.sum(axis=1, keepdims=True).astype(np.float32, copy=False)
        bad = sums.squeeze(1) <= 0
        if np.any(bad):
            labels[bad] = 1.0 / len(EEG_CLASSES)
            sums[bad] = 1.0
        labels /= sums
        self._target = labels

        if self.one_per_col is not None:
            ids = self.df[self.one_per_col].to_numpy()
            self.unique_ids, inv = np.unique(ids, return_inverse=True)
            self._rows_by_uid = [[] for _ in range(len(self.unique_ids))]
            for i, u in enumerate(inv):
                self._rows_by_uid[u].append(i)

    def __len__(self):
        return len(self.unique_ids) if self.one_per_col is not None else len(self.df)

    def __getitem__(self, index):
        if self.one_per_col is not None:
            rows = self._rows_by_uid[index]
            row_idx = rows[np.random.randint(len(rows))]
        else:
            row_idx = index

        eeg_id = self._eeg_id[row_idx]
        eeg_sub_id = self._eeg_sub_id[row_idx]
        offset = int(self._offset[row_idx])

        eeg_file_path = self.eeg_dir / f"{eeg_id}.npy"
        full = np.load(eeg_file_path, mmap_mode="r")
        if full.ndim == 1:
            full = full.reshape(-1, EEG_FEATURES)
        offset = max(0, min(offset, max(0, full.shape[0] - 1)))
        eeg = full[offset : offset + self.num_rows]
        eeg = _pad_or_truncate(np.asarray(eeg), self.num_rows)

        assert eeg.shape == (self.num_rows, EEG_FEATURES)

        return dict(
            eeg=eeg.T.astype(np.float32, copy=False),
            name=f"{eeg_id}_{eeg_sub_id}",
            target=self._target[row_idx],
        )


class FullEEG(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        eeg_dir: PosixPath,
        spec_dir: PosixPath,
        do_impute_zero: bool,
        processed_data: bool = False,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.eeg_dir = eeg_dir
        self.spec_dir = spec_dir
        self.do_impute_zero = do_impute_zero
        self.processed_data = processed_data
        self.num_rows = EEG_SAMPRATE * EEG_LENGTH

        self._eeg_id = self.df["eeg_id"].to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        eeg_id = self._eeg_id[index]

        eeg_file_path = self.eeg_dir / f"{eeg_id}.npy"
        full = np.load(eeg_file_path, mmap_mode="r")
        if full.ndim == 1:
            full = full.reshape(-1, EEG_FEATURES)
        eeg = _pad_or_truncate(np.asarray(full), self.num_rows)

        assert eeg.shape == (self.num_rows, EEG_FEATURES)
        return dict(eeg=eeg.T.astype(np.float32, copy=False), name=str(eeg_id))


class BACDataModule(ptl.LightningDataModule):
    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.cfg = cfg
        self.data_dir = Path(self.cfg.data_dir)

        if self.cfg.processed_data_dir is not None:
            self.proc_root = Path(self.cfg.processed_data_dir)
        else:
            self.proc_root = Path(self.cfg.output_dir) / "_processed_cache"
        self.train_eegs = self.proc_root / "train_eegs"
        self.test_eegs = self.proc_root / "test_eegs"
        self.train_specs = Path(self.data_dir) / "train_spectrograms"
        self.test_specs = Path(self.data_dir) / "test_spectrograms"
        self.processed_data = True

    def prepare_data(self):
        if not self.test_eegs.exists() or (
            len(list(self.test_eegs.glob("*.npy"))) == 0
        ):
            info(
                "BACDataModule", f"Processing test eegs into directory {self.test_eegs}"
            )
            parquet_to_numpy(
                Path(self.data_dir) / "test_eegs",
                self.test_eegs,
                self.cfg.train.impute_zero,
                restrict_ids=None,
            )

    def setup(self, stage: str):
        info("BACDataModule", f"setup {stage=}")
        if stage == "fit":
            self.full_train_df = pd.read_csv(self.data_dir / "train.csv")
            unique_ids = self.full_train_df[self.cfg.train.split_by_col].unique()
            validation_ids = unique_ids[
                self.cfg.train.fold_idx :: self.cfg.train.num_folds
            ]
            validation_mask = self.full_train_df[self.cfg.train.split_by_col].isin(
                validation_ids
            )
            self.train_df = self.full_train_df[~validation_mask]
            self.val_df = self.full_train_df[validation_mask]
            info(
                "BACDataModule",
                f"Training fold {self.cfg.train.fold_idx} of {self.cfg.train.num_folds}"
                f" folds has {len(self.train_df)} train labels and {len(self.val_df)} validation labels"
                f" (split by {self.cfg.train.split_by_col}).",
            )
        elif stage == "test":
            self.test_df = pd.read_csv(self.data_dir / "test.csv")
            info("BACDataModule", f"Testing on {len(self.test_df)} EEGs.")
        else:
            raise ValueError(f"Unsupported {stage=}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            OffsetEEG(
                self.train_df,
                self.train_eegs,
                self.train_specs,
                processed_data=True,
                do_impute_zero=self.cfg.train.impute_zero,
                one_per_col=(
                    self.cfg.train.split_by_col
                    if self.cfg.train.one_per_split_by_col
                    else None
                ),
            ),
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(self.cfg.train.num_workers > 0),
            prefetch_factor=4 if self.cfg.train.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            OffsetEEG(
                self.val_df,
                self.train_eegs,
                self.train_specs,
                processed_data=True,
                do_impute_zero=self.cfg.train.impute_zero,
                one_per_col=(
                    self.cfg.train.split_by_col
                    if self.cfg.train.one_per_split_by_col
                    else None
                ),
            ),
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(self.cfg.train.num_workers > 0),
            prefetch_factor=4 if self.cfg.train.num_workers > 0 else None,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            FullEEG(
                self.test_df,
                self.test_eegs,
                self.test_specs,
                self.cfg.train.impute_zero,
                processed_data=True,
            ),
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(self.cfg.train.num_workers > 0),
            prefetch_factor=4 if self.cfg.train.num_workers > 0 else None,
        )




## === cell 2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as ptl
from transformers import get_cosine_schedule_with_warmup
from types import SimpleNamespace
from pathlib import Path
import pandas as pd
import wandb
import importlib
from collections import OrderedDict


class BasicConvolution(nn.Module):
    def __init__(
        self,
        input_len,
        input_features,
        output_classes,
        num_conv_layers,
        num_kernels,
        kernel_size,
        stride,
        dilation,
        activation,
        dropout1_prob,
        fcn_nodes,
        dropout2_prob,
    ):
        super().__init__()
        layers = OrderedDict()
        for l in range(num_conv_layers):
            layers[f"conv{l+1}"] = nn.Conv1d(
                input_features,
                num_kernels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=True,
            )
            layers[f"bn{l+1}"] = nn.BatchNorm1d(num_kernels)
            if activation == "relu":
                layers[f"relu{l+1}"] = nn.ReLU(inplace=True)
            elif activation == "tanh":
                layers[f"tanh{l+1}"] = nn.Tanh()
            else:
                raise ValueError(f"Unknown {activation=}")
            input_features = num_kernels
            input_len = self.calculate_conv_output_size(
                input_len, kernel_size, stride, dilation
            )
            if (l + 1) % 2 == 0:
                layers[f"max{l+1}"] = nn.MaxPool1d(2)
                input_len = self.calculate_conv_output_size(input_len, 2, 2, 1)
        self.conv = nn.Sequential(layers)
        self.dropout1 = nn.Dropout1d(dropout1_prob)
        self.fc1 = nn.Linear(num_kernels * input_len, fcn_nodes)
        self.dropout2 = nn.Dropout(dropout2_prob)
        self.fc2 = nn.Linear(fcn_nodes, output_classes)

    def calculate_conv_output_size(self, input_length, kernel_size, stride, dilation):
        return (input_length - dilation * (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class BACModelModule(ptl.LightningModule):
    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.output_dir = Path(self.cfg.output_dir)
        model_class_name = self.cfg.model.class_name
        model_class = globals()[model_class_name]
        self.model_params = vars(getattr(cfg.model, model_class_name))
        self.model_params.update(
            dict(
                input_len=EEG_SAMPRATE * EEG_LENGTH,
                input_features=EEG_FEATURES,
                output_classes=len(EEG_CLASSES),
            )
        )
        self.model = model_class(**self.model_params)
        self.loss_fn = nn.KLDivLoss(reduction="batchmean")

    def forward(self, batch):
        out = {}
        logits = self.model(batch["eeg"])
        if torch.isnan(logits).any():
            torch.save(logits, self.output_dir / "logits.pt")
            torch.save(batch["eeg"], self.output_dir / "eeg.pt")
            torch.save(self.model.state_dict(), self.output_dir / "model.ckpt")
            torch.save(self.model_params, self.output_dir / "model.params")
            raise ValueError("logits are nan")

        logprobs = F.log_softmax(logits, dim=1)
        if "target" in batch:
            out["loss"] = self.loss_fn(logprobs, batch["target"])
        out["prob"] = np.exp(logprobs.detach().cpu().numpy())
        return out

    def on_train_epoch_start(self):
        self.train_epoch_loss = 0.0
        self.train_epoch_cnt = 0
        self.epoch_metrics = {}
        self.epoch_conf_mat = None

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        self.train_epoch_loss += out["loss"].item() * len(batch["name"])
        self.train_epoch_cnt += len(batch["name"])
        self.log(
            "loss",
            out["loss"].item(),
            batch_size=len(batch["name"]),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return out["loss"]

    def on_validation_epoch_start(self):
        self.val_epoch_loss = 0.0
        self.val_epoch_cnt = 0
        self.val_epoch_true = []
        self.val_epoch_prob = []
        self.val_epoch_name = []

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        self.val_epoch_loss += out["loss"].item() * len(batch["name"])
        self.val_epoch_cnt += len(batch["name"])
        self.val_epoch_true.extend(batch["target"].cpu().numpy())
        self.val_epoch_prob.extend(out["prob"])
        self.val_epoch_name.extend(batch["name"])
        return out["loss"]

    def on_validation_epoch_end(self):
        self.epoch_metrics["val_loss"] = self.val_epoch_loss / self.val_epoch_cnt
        df = pd.DataFrame(
            data=np.concatenate(
                [
                    np.reshape(self.val_epoch_name, (-1, 1)),
                    self.val_epoch_true,
                    self.val_epoch_prob,
                ],
                axis=1,
            ),
            columns=["name"]
            + [f"true_{cls}" for cls in EEG_CLASSES]
            + [f"prob_{cls}" for cls in EEG_CLASSES],
        )
        df.to_csv(self.output_dir / "val.csv", index=False)

        self.epoch_conf_mat = wandb.plot.confusion_matrix(
            probs=np.array(self.val_epoch_prob),
            y_true=np.argmax(self.val_epoch_true, axis=1),
            class_names=EEG_CLASSES,
        )
        del (
            self.val_epoch_loss,
            self.val_epoch_cnt,
            self.val_epoch_true,
            self.val_epoch_prob,
            self.val_epoch_name,
        )

    def on_train_epoch_end(self):
        self.epoch_metrics["train_loss"] = self.train_epoch_loss / self.train_epoch_cnt
        self.log_dict(
            self.epoch_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True
        )
        if self.epoch_conf_mat is not None and self.cfg.wandb_enabled:
            self.logger.experiment.log({"conf_mat": self.epoch_conf_mat})
        del (
            self.train_epoch_loss,
            self.train_epoch_cnt,
            self.epoch_metrics,
            self.epoch_conf_mat,
        )

    def configure_optimizers(self):
        optim_class_name = self.cfg.optimizer.class_name
        optim_module = importlib.import_module(".optim", "torch")
        optim_class = getattr(optim_module, optim_class_name)
        extra_params = getattr(self.cfg.optimizer, optim_class_name, {})
        optimizer = optim_class(
            self.parameters(), lr=self.cfg.optimizer.lr, **vars(extra_params)
        )
        if not hasattr(self.cfg.optimizer, "scheduler"):
            return optimizer
        if self.cfg.optimizer.scheduler.lower() != "cosine":
            raise ValueError(f"Unsupported scheduler '{self.cfg.optimizer.scheduler}'")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_warmup_steps=self.cfg.optimizer.num_warmup_steps,
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1, "name": "lr"}
        ]

    def on_test_epoch_start(self):
        self.test_names = []
        self.test_probs = []

    def test_step(self, batch, batch_idx):
        out = self.forward(batch)
        self.test_names.extend(batch["name"])
        self.test_probs.extend(out["prob"])

    def on_test_epoch_end(self):
        probs = np.asarray(self.test_probs, dtype=np.float64)
        probs = np.nan_to_num(
            probs,
            nan=1.0 / len(EEG_CLASSES),
            posinf=1.0 / len(EEG_CLASSES),
            neginf=1.0 / len(EEG_CLASSES),
        )
        probs = np.clip(probs, 1e-12, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)

        df = pd.DataFrame(
            data=np.concatenate([np.reshape(self.test_names, (-1, 1)), probs], axis=1),
            columns=["eeg_id"] + [f"{cls}_vote" for cls in EEG_CLASSES],
        )
        df.to_csv(self.output_dir / "submission.csv", index=False)
        del self.test_names, self.test_probs




## === cell 3
import yaml
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, RichModelSummary
from pytorch_lightning.loggers import WandbLogger


def _ensure_train_cache_once(json_cfg: dict):
    cfg = json_to_py(json_cfg)
    data_dir = Path(cfg.data_dir)

    if cfg.processed_data_dir is not None:
        proc_root = Path(cfg.processed_data_dir)
    else:
        proc_root = Path(cfg.output_dir) / "_processed_cache"
    train_eegs = proc_root / "train_eegs"
    create_directory(train_eegs)

    train_df = pd.read_csv(data_dir / "train.csv")
    all_ids = set(train_df["eeg_id"].unique())

    missing = []
    for eid in all_ids:
        if not (train_eegs / f"{eid}.npy").exists():
            missing.append(eid)

    if len(missing) == 0:
        info("train", f"Train cache already complete: {len(all_ids)} eegs.")
        return

    info(
        "train",
        f"Building train cache: {len(missing)}/{len(all_ids)} missing eegs -> {train_eegs}",
    )
    parquet_to_numpy(
        data_dir / "train_eegs",
        train_eegs,
        cfg.train.impute_zero,
        restrict_ids=set(missing),
    )


def train(config_str):
    configure_logger()
    json_cfg = yaml.safe_load(config_str)

    json_cfg["wandb_enabled"] = (
        not json_cfg.get("development", False)
    ) and wandb_login()
    if not json_cfg["wandb_enabled"]:
        info("train", "WANDB was disabled.")

    if json_cfg.get("development", False):
        json_cfg["train"]["num_folds"] = 2

    _ensure_train_cache_once(json_cfg)

    submissions = []
    num_folds = json_cfg["train"]["num_folds"]
    for fold_idx in range(num_folds):
        json_cfg["train"]["fold_idx"] = fold_idx
        info("train", f"Training fold {fold_idx} of {num_folds}")
        submissions.append(str(train_one_fold(json_cfg)))

    out_path = Path(json_cfg["output_dir"]) / "submission.csv"
    merge_csvs(submissions, out_path, "eeg_id")

    sample = pd.read_csv(Path(json_cfg["data_dir"]) / "sample_submission.csv")
    merged = pd.read_csv(out_path)
    merged["eeg_id"] = merged["eeg_id"].astype(sample["eeg_id"].dtype, copy=False)
    merged = sample[["eeg_id"]].merge(merged, on="eeg_id", how="left")
    vote_cols = [f"{c}_vote" for c in EEG_CLASSES]
    merged[vote_cols] = merged[vote_cols].astype(np.float64)

    merged[vote_cols] = merged[vote_cols].fillna(1.0 / len(EEG_CLASSES))
    merged[vote_cols] = np.clip(merged[vote_cols].to_numpy(), 1e-12, 1.0)
    merged[vote_cols] = merged[vote_cols].to_numpy() / merged[vote_cols].to_numpy().sum(
        axis=1, keepdims=True
    )

    merged.to_csv(out_path, index=False)
    info("train", f"Wrote final submission to: {out_path}")


def train_one_fold(json_cfg):
    cfg = json_to_py(json_cfg)

    exp_name = f"{cfg.experiment}_{cfg.train.fold_idx}"
    cfg.output_dir = Path(cfg.output_dir) / exp_name
    cfg.temp_dir = Path(cfg.temp_dir) / exp_name
    cfg.seed = cfg.seed + 13 * cfg.train.fold_idx

    seed_everything(cfg.seed)
    create_directory(cfg.output_dir)
    create_directory(cfg.temp_dir)

    lr_monitor = LearningRateMonitor("epoch")
    model_summary = RichModelSummary(max_depth=3)

    if cfg.prev_notebook_ver is not None:
        ckpt_path = Path(cfg.prev_notebook_ver) / exp_name / "trainer.ckpt"
        if ckpt_path.exists():
            cfg.wandb_enabled = False
    else:
        ckpt_path = Path(cfg.output_dir) / "trainer.ckpt"

    run = wandb.init(
        job_type=f"fold {cfg.train.fold_idx} of {cfg.train.num_folds}",
        dir=cfg.temp_dir,
        config=json_cfg,
        project="HMS - Harmful Brain Activity Classification",
        reinit=True,
        group=cfg.experiment,
        name=exp_name,
        notes=cfg.notes,
        mode="disabled" if not cfg.wandb_enabled else "online",
    )
    pl_logger = WandbLogger(experiment=run)

    data = BACDataModule(cfg)
    model = BACModelModule(cfg)

    trainer = Trainer(
        default_root_dir=cfg.temp_dir,
        deterministic=cfg.train.deterministic,
        accelerator=cfg.train.accelerator,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.gradient_clip_val,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        callbacks=[lr_monitor, model_summary],
        logger=pl_logger,
        num_sanity_val_steps=0,
        sync_batchnorm=True,
        check_val_every_n_epoch=1,
        max_time=cfg.train.max_time,
        log_every_n_steps=1 if cfg.development else 5,
        max_epochs=2 if cfg.development else cfg.train.max_epochs,
        limit_train_batches=2 if cfg.development else cfg.train.limit_train_batches,
        limit_val_batches=2 if cfg.development else cfg.train.limit_val_batches,
        enable_checkpointing=False,
    )

    if ckpt_path.exists():
        info("train", f"Reusing existing checkpoint {ckpt_path}")
        trainer.fit(model, data, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, data)
        info("train", f"Checkpointing model to {ckpt_path}.")
        trainer.save_checkpoint(ckpt_path)

    trainer.test(model, datamodule=data)
    run.finish()
    return cfg.output_dir / "submission.csv"




## === cell 4
train(CONFIG_STR)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2690731650.py in <cell line: 0>()
----> 1 train(CONFIG_STR)

/tmp/ipykernel_55/1217752204.py in train(config_str)
     65         json_cfg["train"]["fold_idx"] = fold_idx
     66         info("train", f"Training fold {fold_idx} of {num_folds}")
---> 67         submissions.append(str(train_one_fold(json_cfg)))
     68 
     69     out_path = Path(json_cfg["output_dir"]) / "submission.csv"

/tmp/ipykernel_55/1217752204.py in train_one_fold(json_cfg)
    149         trainer.fit(model, data, ckpt_path=ckpt_path)
    150     else:
--> 151         trainer.fit(model, data)
    152         info("train", f"Checkpointing model to {ckpt_path}.")
    153         trainer.save_checkpoint(ckpt_path)

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in fit(self, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)
    558         self.training = True
    559         self.should_stop = False
--> 560         call._call_and_handle_interrupt(
    561             self, self._fit_impl, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path
    562         )

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/call.py in _call_and_handle_interrupt(trainer, trainer_fn, *args, **kwargs)
     47         if trainer.strategy.launcher is not None:
     48             return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
---> 49         return trainer_fn(*args, **kwargs)
     50 
     51     except _TunerExitException:

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in _fit_impl(self, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)
    596             model_connected=self.lightning_module is not None,
    597         )
--> 598         self._run(model, ckpt_path=ckpt_path)
    599 
    600         assert self.state.stopped

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in _run(self, model, ckpt_path)
   1009         # RUN THE TRAINER
   1010         # ----------------------------
-> 1011         results = self._run_stage()
   1012 
   1013         # ----------------------------

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in _run_stage(self)
   1053                 self._run_sanity_check()
   1054             with torch.autograd.set_detect_anomaly(self._detect_anomaly):
-> 1055                 self.fit_loop.run()
   1056             return None
   1057         raise RuntimeError(f"Unexpected state {self.state}")

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/fit_loop.py in run(self)
    214             try:
    215                 self.on_advance_start()
--> 216                 self.advance()
    217                 self.on_advance_end()
    218             except StopIteration:

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/fit_loop.py in advance(self)
    456         with self.trainer.profiler.profile("run_training_epoch"):
    457             assert self._data_fetcher is not None
--> 458             self.epoch_loop.run(self._data_fetcher)
    459 
    460     def on_advance_end(self) -> None:

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/training_epoch_loop.py in run(self, data_fetcher)
    150         while not self.done:
    151             try:
--> 152                 self.advance(data_fetcher)
    153                 self.on_advance_end(data_fetcher)
    154             except StopIteration:

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/training_epoch_loop.py in advance(self, data_fetcher)
    308         else:
    309             dataloader_iter = None
--> 310             batch, _, __ = next(data_fetcher)
    311             # TODO: we should instead use the batch_idx returned by the fetcher, however, that will require saving the
    312             # fetcher state so that the batch_idx is correct after restarting

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/fetchers.py in __next__(self)
    132         elif not self.done:
    133             # this will run only when no pre-fetching was done.
--> 134             batch = super().__next__()
    135         else:
    136             # the iterator is empty

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/fetchers.py in __next__(self)
     59         self._start_profiler()
     60         try:
---> 61             batch = next(self.iterator)
     62         except StopIteration:
     63             self.done = True

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/utilities/combined_loader.py in __next__(self)
    339     def __next__(self) -> _ITERATOR_RETURN:
    340         assert self._iterator is not None
--> 341         out = next(self._iterator)
    342         if isinstance(self._iterator, _Sequential):
    343             return out

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/utilities/combined_loader.py in __next__(self)
     76         for i in range(n):
     77             try:
---> 78                 out[i] = next(self.iterators[i])
     79             except StopIteration:
     80                 self._consumed[i] = True

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

FileNotFoundError: Caught FileNotFoundError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/781919268.py", line 90, in __getitem__
    full = np.load(eeg_file_path, mmap_mode="r")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/exp015_0/_processed_cache/train_eegs/1487394339.npy'
