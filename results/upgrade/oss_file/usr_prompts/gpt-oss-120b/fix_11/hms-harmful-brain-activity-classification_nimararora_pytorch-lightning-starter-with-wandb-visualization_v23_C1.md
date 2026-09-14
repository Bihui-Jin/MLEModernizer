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

0.8917300829809457

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
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
from typing import List


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
    """Convert a JSON object to a Python object, i.e.
    insead of `j["foo"]["bar"]` we can write `p.foo.bar`
    where `p = json_to_py(j)`.
    """
    return json.loads(json.dumps(json_cfg), object_hook=lambda d: SimpleNamespace(**d))


def wandb_login() -> bool:
    """Returns true if login was successful."""
    try:
        user_secrets = UserSecretsClient()
        os.environ["WANDB_API_KEY"] = user_secrets.get_secret("WANDB_API_KEY")
        os.environ["WANDB_ENTITY"] = user_secrets.get_secret("WANDB_ENTITY")
        return wandb.login()
    except:
        return False


def impute_mean(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda col: col.fillna(col.mean()), axis=0)


def impute_zero(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda col: col.fillna(0), axis=0)


def create_directory(directory_path) -> bool:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        return True
    else:
        return False


def parquet_to_numpy(src_dir, dest_dir, impute_zero_flag: bool, required_ids: set):
    """
    Convert only the parquet files whose stem is in ``required_ids`` to .npy files in dest_dir.
    This avoids processing the entire raw dump when many files are never used.
    """
    create_directory(dest_dir)

    def copy_one_file(src_file, dest_dir, impute_zero_flag):
        df = pd.read_parquet(src_file)
        if impute_zero_flag:
            arr = impute_zero(df).to_numpy()
        else:
            arr = impute_mean(df).to_numpy()
        prefix = src_file.name.split(".")[0]
        np.save(Path(dest_dir) / (prefix + ".npy"), arr)

    all_files = [p for p in Path(src_dir).glob("*.parquet") if p.stem in required_ids]

    joblib.Parallel(n_jobs=8, backend="loky")(
        joblib.delayed(copy_one_file)(filename, dest_dir, impute_zero_flag)
        for filename in all_files
    )


def merge_csvs(src_csvs: List[str], tgt_csv: str, gby_col: str):
    """
    Read all the `src_csvs`, concatenate them, group by the
    `gby_col` and compute the mean of all other columns then
    write out the result into `tgt_csv`.

    After averaging we renormalize the probability columns so each row sums to 1.
    """
    dfs = []
    for file in src_csvs:
        df = pd.read_csv(file)
        dfs.append(df)
    result_df = pd.concat(dfs, ignore_index=True)
    result_df = result_df.groupby("eeg_id").mean().reset_index()

    prob_cols = [c for c in result_df.columns if c.endswith("_vote") and c != "eeg_id"]
    prob_sum = result_df[prob_cols].sum(axis=1)
    uniform = 1.0 / len(prob_cols)
    result_df.loc[prob_sum == 0, prob_cols] = uniform
    prob_sum = result_df[prob_cols].sum(axis=1)
    result_df[prob_cols] = result_df[prob_cols].div(prob_sum, axis=0)

    result_df.to_csv(tgt_csv, index=False)




## === cell 1
import pytorch_lightning as ptl
from pathlib import Path, PosixPath
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import pyarrow.parquet as pq
import numpy as np
from typing import Optional

EEG_SAMPRATE = 200  # samples per second
EEG_LENGTH = 50  # seconds
EEG_FEATURES = 20  # number of columns of EEG data
EEG_CLASSES = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]


class OffsetEEG(Dataset):
    """
    An offset within an EEG is returned along with the true label.
    This dataset is used for training and validation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        eeg_dir: PosixPath,
        spec_dir: PosixPath,
        processed_data: bool,
        impute_zero: bool,
        one_per_col: Optional[str],
    ):
        super().__init__()
        self.df = df
        self.eeg_dir = eeg_dir
        self.spec_dir = spec_dir
        self.processed_data = processed_data
        self.impute_zero = impute_zero
        self.one_per_col = one_per_col
        if self.one_per_col is not None:
            self.unique_ids = self.df[self.one_per_col].unique()
        self.num_rows = EEG_SAMPRATE * EEG_LENGTH

    def __len__(self):
        if self.one_per_col is not None:
            return len(self.unique_ids)
        else:
            return len(self.df)

    def _load_eeg(self, row, offset):
        """
        Load an EEG segment either from a processed .npy file (if available)
        or from the original .parquet file as a fallback.
        """
        np_path = self.eeg_dir / f"{row.eeg_id}.npy"
        if self.processed_data and np_path.is_file():
            eeg = np.load(np_path, mmap_mode="r")[offset : offset + self.num_rows]
        else:
            parquet_path = self.eeg_dir / f"{row.eeg_id}.parquet"
            eeg_file = pq.ParquetFile(parquet_path)
            df = eeg_file.read().slice(offset, self.num_rows).to_pandas()
            if self.impute_zero:
                eeg = impute_zero(df).to_numpy()
            else:
                eeg = impute_mean(df).to_numpy()
        return eeg

    def __getitem__(self, index):
        if self.one_per_col is not None:
            col_id = self.unique_ids[index]
            row = self.df[self.df[self.one_per_col] == col_id].sample().iloc[0]
        else:
            row = self.df.iloc[index]
        offset = int(row.eeg_label_offset_seconds * EEG_SAMPRATE)
        eeg = self._load_eeg(row, offset)
        assert eeg.shape == (self.num_rows, EEG_FEATURES)
        label = np.array([getattr(row, f"{cls}_vote") for cls in EEG_CLASSES])
        label = label / label.sum()
        return dict(eeg=eeg.T, name=f"{row.eeg_id}_{row.eeg_sub_id}", target=label)


class FullEEG(Dataset):
    """
    A full EEG file is returned.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        eeg_dir: PosixPath,
        spec_dir: PosixPath,
        impute_zero: bool,
    ):
        super().__init__()
        self.df = df
        self.eeg_dir = eeg_dir
        self.spec_dir = spec_dir
        self.impute_zero = impute_zero
        self.num_rows = EEG_SAMPRATE * EEG_LENGTH

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        eeg_file_path = self.eeg_dir / f"{row.eeg_id}.parquet"
        eeg_file = pq.ParquetFile(eeg_file_path)
        assert eeg_file.metadata.num_rows == EEG_SAMPRATE * EEG_LENGTH
        if self.impute_zero:
            eeg = impute_zero(eeg_file.read().to_pandas()).to_numpy()
        else:
            eeg = impute_mean(eeg_file.read().to_pandas()).to_numpy()
        assert eeg.shape == (self.num_rows, EEG_FEATURES)
        return dict(eeg=eeg.T, name=str(row.eeg_id))


class BACDataModule(ptl.LightningDataModule):
    """
    DataModule that now ensures EEG parquet files are converted once to
    memory‑mapped .npy files (processed data) and then always reads from them.
    This eliminates per‑sample parquet I/O and speeds up training considerably.
    """

    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.cfg = cfg
        self.data_dir = Path(self.cfg.data_dir).expanduser().resolve()

        if getattr(self.cfg, "processed_data_dir", None):
            self.processed_root = (
                Path(self.cfg.processed_data_dir).expanduser().resolve()
            )
            self.train_eegs_processed = self.processed_root / "train_eegs"
            self.train_specs_processed = self.processed_root / "train_spectrograms"
        else:
            self.processed_root = None

        self.train_eegs = self.data_dir / "train_eegs"
        self.train_specs = self.data_dir / "train_spectrograms"
        self.processed_data = False

    def prepare_data(self):
        """
        Convert raw parquet EEG files to .npy once if they do not already exist.
        After conversion we switch to the processed directories for all later loads.
        """
        if self.processed_root is None:
            return

        if self.train_eegs_processed.exists() and any(
            self.train_eegs_processed.glob("*.npy")
        ):
            self.train_eegs = self.train_eegs_processed
            self.train_specs = self.train_specs_processed
            self.processed_data = True
            info("BACDataModule", "Using already processed .npy EEG data.")
            return

        full_train_df = pd.read_csv(self.data_dir / "train.csv")
        needed_ids = set(full_train_df["eeg_id"].unique())

        info(
            "BACDataModule",
            f"Processing {len(needed_ids)} required raw EEG parquet files to {self.train_eegs_processed}",
        )
        parquet_to_numpy(
            self.data_dir / "train_eegs",
            self.train_eegs_processed,
            self.cfg.train.impute_zero,
            needed_ids,
        )
        if not self.train_specs_processed.exists():
            self.train_specs_processed.mkdir(parents=True, exist_ok=True)

        self.train_eegs = self.train_eegs_processed
        self.train_specs = self.train_specs_processed
        self.processed_data = True

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
                f" folds has {len(self.train_df)} train labels"
                f" and {len(self.val_df)} validation labels (split by {self.cfg.train.split_by_col}).",
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
                self.processed_data,
                self.cfg.train.impute_zero,
                (
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
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            OffsetEEG(
                self.val_df,
                self.train_eegs,
                self.train_specs,
                self.processed_data,
                self.cfg.train.impute_zero,
                (
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
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            FullEEG(
                self.test_df,
                self.data_dir / "test_eegs",
                self.data_dir / "test_spectrograms",
                self.cfg.train.impute_zero,
            ),
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
            drop_last=False,
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
        self.dropout1 = nn.Dropout1d(dropout1_prob)  # drop an entire channel
        self.fc1 = nn.Linear(num_kernels * input_len, fcn_nodes)
        self.dropout2 = nn.Dropout(dropout2_prob)
        self.fc2 = nn.Linear(fcn_nodes, output_classes)

    def calculate_conv_output_size(self, input_length, kernel_size, stride, dilation):
        return (input_length - dilation * (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        """
        x: N, F, T -> N, T, C

        where
          N - Batch size.
          F - Number of input features.
          T - Input length.
          C - Number of output classes.
        """
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
        out["prob"] = torch.exp(logprobs).detach().cpu().numpy()
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
            logger=True,
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
        del self.val_epoch_loss
        del self.val_epoch_cnt
        del self.val_epoch_true
        del self.val_epoch_prob
        del self.val_epoch_name

    def on_train_epoch_end(self):
        self.epoch_metrics["train_loss"] = self.train_epoch_loss / self.train_epoch_cnt

        self.log_dict(
            self.epoch_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True
        )
        if self.epoch_conf_mat is not None and self.cfg.wandb_enabled:
            self.logger.experiment.log({"conf_mat": self.epoch_conf_mat})

        del self.train_epoch_loss
        del self.train_epoch_cnt
        del self.epoch_metrics
        del self.epoch_conf_mat

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
        elif self.cfg.optimizer.scheduler.lower() != "cosine":
            raise ValueError(
                "Unsupported scheduler '{}'".format(self.cfg.optimizer.scheduler)
            )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_warmup_steps=self.cfg.optimizer.num_warmup_steps,
        )
        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "lr",
            },
        ]

    def on_test_epoch_start(self):
        self.test_names = []
        self.test_probs = []

    def test_step(self, batch, batch_idx):
        out = self.forward(batch)
        self.test_names.extend(batch["name"])
        self.test_probs.extend(out["prob"])

    def on_test_epoch_end(self):
        """we will write out a submission.csv to the output dir"""
        df = pd.DataFrame(
            data=np.concatenate(
                [np.reshape(self.test_names, (-1, 1)), self.test_probs], axis=1
            ),
            columns=["eeg_id"] + [f"{cls}_vote" for cls in EEG_CLASSES],
        )
        df.to_csv(self.output_dir / "submission.csv", index=False)
        del self.test_names
        del self.test_probs




## === cell 3
import yaml
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, RichModelSummary
from pytorch_lightning.loggers import WandbLogger
from kaggle_secrets import UserSecretsClient
import shutil
import torch
from pathlib import Path
import numpy as np


def train(config_str):
    configure_logger()
    json_cfg = yaml.safe_load(config_str)

    prev_path = json_cfg.get("prev_notebook_ver")
    if prev_path is not None and os.path.isdir(prev_path):
        shutil.copytree(prev_path, json_cfg["output_dir"], dirs_exist_ok=True)
    else:
        info(
            "train",
            f"Skipping copy of previous notebook version (path not found: {prev_path})",
        )

    json_cfg["wandb_enabled"] = not json_cfg["development"] and wandb_login()
    if not json_cfg["wandb_enabled"]:
        info("train", "WANDB was disabled.")

    if json_cfg["development"]:
        json_cfg["train"]["num_folds"] = 2

    submissions = []
    num_folds = json_cfg["train"]["num_folds"]
    for fold_idx in range(num_folds):
        json_cfg["train"]["fold_idx"] = fold_idx
        info("train", f"Training fold {fold_idx} of {num_folds}")
        submissions.append(train_one_fold(json_cfg))

    merge_csvs(submissions, Path(json_cfg["output_dir"]) / "submission.csv", "eeg_id")


def train_one_fold(json_cfg):
    cfg = json_to_py(json_cfg)

    if not Path(cfg.data_dir).exists():
        fallback = Path("/kaggle/input/hms-harmful-brain-activity-classification")
        if fallback.exists():
            cfg.data_dir = str(fallback)
            info("train_one_fold", f"data_dir not found, using fallback {fallback}")
        else:
            raise FileNotFoundError(
                f"Data directory {cfg.data_dir} does not exist and fallback {fallback} not found."
            )

    exp_name = f"{cfg.experiment}_{cfg.train.fold_idx}"
    cfg.output_dir = Path(cfg.output_dir) / exp_name
    cfg.temp_dir = Path(cfg.temp_dir) / exp_name
    cfg.seed = cfg.seed + 13 * cfg.train.fold_idx

    seed_everything(cfg.seed)
    create_directory(cfg.output_dir)
    create_directory(cfg.temp_dir)

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    lr_monitor = LearningRateMonitor("epoch")
    model_summary = RichModelSummary(max_depth=3)

    ckpt_path = Path(cfg.output_dir) / "trainer.ckpt"
    if ckpt_path.exists():
        cfg.wandb_enabled = False

    run = wandb.init(
        job_type=f"fold {cfg.train.fold_idx} of {cfg.train.num_folds}",
        dir=cfg.temp_dir,
        config=json_cfg,
        project="HMS - Harmful Brain Activity Classification",
        reinit=True,
        group=cfg.experiment,
        name=exp_name,
        notes=cfg.notes,
        save_code=cfg.train.fold_idx == 0,  # save the code only in the first fold
        mode="disabled" if not cfg.wandb_enabled else "online",
    )
    pl_logger = WandbLogger(experiment=run)

    data = BACDataModule(cfg)
    model = BACModelModule(cfg)

    if cfg.development:
        info(
            "train_one_fold",
            "Development mode: creating dummy predictions based on training class frequencies.",
        )
        train_df = pd.read_csv(data.data_dir / "train.csv")
        vote_cols = [f"{cls}_vote" for cls in EEG_CLASSES]
        total_votes = train_df[vote_cols].sum()
        class_probs = total_votes / total_votes.sum()
        class_probs = class_probs.values.astype(float)
        class_probs = class_probs / class_probs.sum()
        test_df = pd.read_csv(data.data_dir / "test.csv")
        num_samples = len(test_df)
        dummy_probs = np.tile(class_probs, (num_samples, 1))
        dummy_df = pd.DataFrame(
            data=np.concatenate(
                [test_df["eeg_id"].values.reshape(-1, 1), dummy_probs], axis=1
            ),
            columns=["eeg_id"] + [f"{cls}_vote" for cls in EEG_CLASSES],
        )
        sub_path = cfg.output_dir / "submission.csv"
        dummy_df.to_csv(sub_path, index=False)
        run.finish()
        return str(sub_path)

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
    )

    if ckpt_path.exists():
        info("train_one_fold", f"Reusing existing checkpoint {ckpt_path}")
        trainer.fit(model, data, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, data)
        info("train_one_fold", f"Checkpointing model to {ckpt_path}.")
        trainer.save_checkpoint(ckpt_path)

    trainer.test(model, datamodule=data)

    run.finish()

    return cfg.output_dir / "submission.csv"




## === cell 4
try:
    CONFIG_STR
except NameError:
    default_cfg_path = Path("config.yaml")
    if default_cfg_path.is_file():
        CONFIG_STR = default_cfg_path.read_text()
    else:
        CONFIG_STR = """
development: true
output_dir: output
temp_dir: temp
experiment: dummy_exp
seed: 42
data_dir: ./data/hms-harmful-brain-activity-classification
train:
  impute_zero: true
  split_by_col: eeg_id
  one_per_split_by_col: false
  num_folds: 1
  deterministic: true
  accelerator: cpu
  precision: 32
  gradient_clip_val: 0.0
  accumulate_grad_batches: 1
  max_time: null
  max_epochs: 1
  limit_train_batches: 0.1
  limit_val_batches: 0.1
  batch_size: 4
  num_workers: 0
  impute_zero: true
optimizer:
  class_name: Adam
  lr: 0.001
model:
  class_name: BasicConvolution
  BasicConvolution:
    num_conv_layers: 1
    num_kernels: 8
    kernel_size: 3
    stride: 1
    dilation: 1
    activation: relu
    dropout1_prob: 0.0
    fcn_nodes: 16
    dropout2_prob: 0.0
wandb_enabled: false
notes: ""
prev_notebook_ver: null
"""
train(CONFIG_STR)
