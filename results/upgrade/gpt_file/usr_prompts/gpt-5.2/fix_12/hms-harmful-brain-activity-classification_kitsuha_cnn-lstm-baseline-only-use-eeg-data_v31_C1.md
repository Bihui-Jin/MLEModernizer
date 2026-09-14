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

1.1107197547284728

# 6. Current score

1.66223

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.82374) has done: 'I fix the environment import crash by removing TensorFlow/Keras (not used in the PyTorch pipeline) which is triggering the protobuf `MessageFactory.GetPrototype` error. Then I fix the missing training CSV path by using the `normalized_data.csv` you already write to the working directory, and make the dataset indexing robust by switching from `.loc[idx]` to `.iloc[idx]`. Finally, I ensure test DataLoader order matches `test.csv` (no shuffle) and write predictions aligned to `eeg_id`, guaranteeing the submission has exactly 9850 rows and each row sums to 1.'
- What this solution (achieved 1.31898) has done: 'I fix the cache-writing bug that causes the `.npy` files to be saved with an extra `.npy` extension (so the subsequent `np.load()` can’t find them), which is the root cause of the `FileNotFoundError` cascading through training and inference. I also make the cache write truly atomic by writing to a temp filename that ends with `.npy`, then renaming it to the final `.npy` path. These are runtime/correctness fixes and do not change the model, features, loss, or training loop semantics. After that, the pipeline run end-to-end and produce `submission.csv` with the required columns and per-row probabilities summing to 1.'
- What this solution (achieved 1.66223) has done: 'The crash is because your spectrogram time dimension varies after cropping, so the default DataLoader collate can’t stack tensors of different lengths; fixing this requires padding (or truncation) per-batch while keeping the same feature extraction and model. I add a minimal custom `collate_fn` that right-pads (or trims) each batch to a consistent time length, and I also make the dataset cache include the crop configuration so you don’t accidentally reuse incompatible cached arrays. With that in place, training and test inference run end-to-end, and the script always write a valid `submission.csv` with 9850 rows, correct columns, and per-row probabilities summing to 1. These changes are runtime/correctness fixes and should be score-neutral to slightly positive versus the current broken run.'

# 9. Code solution

## === cell 0
import os, gc, glob, warnings, random, time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)



## === cell 1
PATH = "/kaggle/input/hms-harmful-brain-activity-classification/"
TRAIN_CSV = os.path.join(PATH, "train.csv")
TEST_CSV = os.path.join(PATH, "test.csv")
SAMPLE_SUB_CSV = os.path.join(PATH, "sample_submission.csv")

SPEC_PATH = os.path.join(PATH, "train_spectrograms") + "/"
TEST_SPEC_PATH = os.path.join(PATH, "test_spectrograms") + "/"

df = pd.read_csv(TRAIN_CSV)
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))



## === cell 2
g_eeg = df.groupby("eeg_id", sort=False)

agg = g_eeg.agg(
    spectrogram_id=("spectrogram_id", "first"),
    spectrogram_label_offset_seconds=("spectrogram_label_offset_seconds", "first"),
    min=("spectrogram_label_offset_seconds", "min"),
    max=("spectrogram_label_offset_seconds", "max"),
    patient_id=("patient_id", "first"),
    expert_consensus=("expert_consensus", "first"),
)
vote_sums = g_eeg[TARGETS].sum()
train = pd.concat([agg, vote_sums], axis=1)

y_data = train[TARGETS].values.astype(np.float32)
y_data = y_data / y_data.sum(axis=1, keepdims=True)
train[TARGETS] = y_data
train["target"] = train["expert_consensus"]
train = train.reset_index(drop=False)
print("Train non-overlapp eeg_id shape:", train.shape)



## === cell 3
ycol = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]

normalized_data = train[
    ["eeg_id", "spectrogram_id", "spectrogram_label_offset_seconds"] + ycol + ["target"]
].copy()
normalized_data[ycol] = normalized_data[ycol].astype("float32")
normalized_data[ycol] = normalized_data[ycol].div(
    normalized_data[ycol].sum(axis=1), axis=0
)

print("Prepared normalized_data shape:", normalized_data.shape)



## === cell 4
idx = np.arange(len(normalized_data))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
val_frac = 0.05
val_n = int(len(idx) * val_frac)
val_idx = idx[:val_n]
trn_idx = idx[val_n:]

train_df = normalized_data.iloc[trn_idx].reset_index(drop=True)
val_df = normalized_data.iloc[val_idx].reset_index(drop=True)
print("Split sizes:", len(train_df), len(val_df))




## === cell 5
class SpectrogramDataset(Dataset):
    """
    Spectrogram feature extraction with caching.

    Bugfix: cropping can yield slightly different time lengths across parquet files (due to
    different underlying T), which breaks default DataLoader stacking. We keep the dataset
    returning variable-length (C,T) and solve batching via a custom collate_fn (next cell).
    Also, cache_tag is used to prevent mixing incompatible cached arrays.
    """

    def __init__(
        self,
        df_in: pd.DataFrame,
        spec_path: str,
        ycol=None,
        n_freq_bins: int = 8,
        cache_dir: str = "spec_cache",
        cache_tag: str = "train",
        build_cache: bool = True,
        crop_seconds: int = 600,
        use_offset: bool = True,
    ):
        self.df = df_in.reset_index(drop=True)
        self.spec_path = spec_path
        self.ycol = ycol
        self.n_freq_bins = int(n_freq_bins)

        self.crop_seconds = int(crop_seconds)
        self.use_offset = bool(use_offset)

        self.cache_dir = os.path.join(cache_dir, cache_tag)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.spec_ids = self.df["spectrogram_id"].astype(np.int64).to_numpy()
        if self.use_offset and ("spectrogram_label_offset_seconds" in self.df.columns):
            self.offset_seconds = (
                self.df["spectrogram_label_offset_seconds"]
                .astype(np.float32)
                .to_numpy()
            )
        else:
            self.offset_seconds = None

        if self.ycol is not None:
            self.labels = self.df[self.ycol].to_numpy(dtype=np.float32, copy=True)
        else:
            self.labels = None

        if build_cache:
            self._build_cache_for_unique_ids()

    def __len__(self):
        return len(self.spec_ids)

    @staticmethod
    def _fill_nan_inplace(x: np.ndarray) -> np.ndarray:
        return np.nan_to_num(x, nan=0.0, copy=False)

    def _cache_path(self, spec_id: int) -> str:
        return os.path.join(self.cache_dir, f"{spec_id}.npy")

    def _is_valid_cache(self, arr: np.ndarray) -> bool:
        if not isinstance(arr, np.ndarray):
            return False
        if arr.dtype != np.float32:
            return False
        if arr.ndim != 2:
            return False
        if arr.shape[0] != self.n_freq_bins:
            return False
        if arr.shape[1] <= 0:
            return False
        return True

    def _maybe_crop_time(
        self, spec: np.ndarray, offset_seconds: float | None
    ) -> np.ndarray:
        if self.crop_seconds <= 0:
            return spec

        T = spec.shape[1]
        if T <= 1:
            return spec

        bins_per_second = T / 600.0  # 10 minutes
        crop_bins = int(round(self.crop_seconds * bins_per_second))
        crop_bins = max(8, min(crop_bins, T))

        if (
            (offset_seconds is None)
            or (not np.isfinite(offset_seconds))
            or (not self.use_offset)
        ):
            center = T // 2
        else:
            center = int(round(float(offset_seconds) * bins_per_second))
            center = max(0, min(center, T - 1))

        start = center - crop_bins // 2
        start = max(0, min(start, T - crop_bins))
        end = start + crop_bins
        return spec[:, start:end]

    def _build_spec_array_from_parquet(
        self, spec_id: int, offset_seconds: float | None
    ) -> np.ndarray:
        spec_file_path = f"{self.spec_path}{spec_id}.parquet"
        spec_df = pd.read_parquet(spec_file_path)
        spec = spec_df.to_numpy(dtype=np.float32, copy=False)
        self._fill_nan_inplace(spec)

        if spec.shape[0] > spec.shape[1]:
            spec = spec.T  # now (freq, time)

        spec = self._maybe_crop_time(spec, offset_seconds)

        m = float(spec.mean())
        s = float(spec.std()) + 1e-6
        spec = (spec - m) / s

        if spec.shape[0] >= self.n_freq_bins:
            spec = spec[: self.n_freq_bins]
        else:
            pad = np.zeros(
                (self.n_freq_bins - spec.shape[0], spec.shape[1]), dtype=np.float32
            )
            spec = np.concatenate([spec, pad], axis=0)

        return spec.astype(np.float32, copy=False)

    def _write_cache_atomic(self, cpath: str, arr: np.ndarray) -> None:
        tmp_path = cpath + ".tmp.npy"
        np.save(tmp_path, arr, allow_pickle=False)
        try:
            os.replace(tmp_path, cpath)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _load_cache_mmap(self, cpath: str) -> np.ndarray:
        return np.load(cpath, allow_pickle=False, mmap_mode="r")

    def _load_or_build_spec(
        self, spec_id: int, offset_seconds: float | None
    ) -> np.ndarray:
        cpath = self._cache_path(spec_id)
        if os.path.exists(cpath):
            try:
                arr = self._load_cache_mmap(cpath)
                if self._is_valid_cache(arr):
                    return arr
                else:
                    try:
                        os.remove(cpath)
                    except OSError:
                        pass
            except Exception:
                try:
                    os.remove(cpath)
                except OSError:
                    pass

        spec = self._build_spec_array_from_parquet(spec_id, offset_seconds)
        self._write_cache_atomic(cpath, spec)
        return self._load_cache_mmap(cpath)

    def _build_cache_for_unique_ids(self) -> None:
        uniq_ids = pd.unique(self.spec_ids)
        t0 = time.time()
        built = 0

        for sid in uniq_ids:
            sid = int(sid)
            cpath = self._cache_path(sid)
            if os.path.exists(cpath):
                try:
                    arr = self._load_cache_mmap(cpath)
                    if self._is_valid_cache(arr):
                        continue
                    else:
                        try:
                            os.remove(cpath)
                        except OSError:
                            pass
                except Exception:
                    try:
                        os.remove(cpath)
                    except OSError:
                        pass

            offset_seconds = None
            if self.offset_seconds is not None:
                j = int(np.flatnonzero(self.spec_ids == sid)[0])
                offset_seconds = float(self.offset_seconds[j])

            spec = self._build_spec_array_from_parquet(sid, offset_seconds)
            self._write_cache_atomic(cpath, spec)
            built += 1

        dt = time.time() - t0
        print(
            f"Cache ready [{os.path.basename(self.cache_dir)}]: unique_ids={len(uniq_ids)}, newly_built={built}, time={dt:.1f}s"
        )

    def __getitem__(self, idx):
        spec_id = int(self.spec_ids[idx])
        offset_seconds = None
        if self.offset_seconds is not None:
            offset_seconds = float(self.offset_seconds[idx])

        spec = self._load_or_build_spec(spec_id, offset_seconds)
        x = torch.from_numpy(spec)  # (channels, time), float32 (mmap-backed)

        if self.labels is None:
            return x

        labels = torch.from_numpy(self.labels[idx])
        return x, labels


def pad_collate_1d(batch, pad_value: float = 0.0):
    """
    Right-pad (or right-trim) variable-length (C,T) tensors to a common T within the batch.
    Keeps model inputs consistent and avoids DataLoader stack runtime error.
    """
    if len(batch) == 0:
        return batch

    has_labels = isinstance(batch[0], (tuple, list)) and len(batch[0]) == 2
    if has_labels:
        xs, ys = zip(*batch)
    else:
        xs = batch
        ys = None

    max_T = max(int(x.shape[1]) for x in xs)
    C = int(xs[0].shape[0])

    out_x = torch.full((len(xs), C, max_T), float(pad_value), dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        T = int(x.shape[1])
        if T >= max_T:
            out_x[i] = x[:, :max_T]
        else:
            out_x[i, :, :T] = x

    if has_labels:
        out_y = torch.stack(ys, dim=0)
        return out_x, out_y
    return out_x


num_workers = min(4, os.cpu_count() or 2)
CROP_SECONDS = 120

train_dataset = SpectrogramDataset(
    train_df,
    SPEC_PATH,
    ycol=ycol,
    n_freq_bins=8,
    cache_dir="spec_cache",
    cache_tag=f"train_crop{CROP_SECONDS}s_nf8",
    build_cache=True,
    crop_seconds=CROP_SECONDS,
    use_offset=True,
)
val_dataset = SpectrogramDataset(
    val_df,
    SPEC_PATH,
    ycol=ycol,
    n_freq_bins=8,
    cache_dir="spec_cache",
    cache_tag=f"val_crop{CROP_SECONDS}s_nf8",
    build_cache=True,
    crop_seconds=CROP_SECONDS,
    use_offset=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
    collate_fn=pad_collate_1d,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker,
    collate_fn=pad_collate_1d,
)

X0, y0 = train_dataset[0]
print("Train sample shapes (single):", X0.shape, y0.shape)




## === cell 6
class CNNLSTM(nn.Module):
    def __init__(self, in_channels=8, num_classes=6):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.5)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.75)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lstm1 = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = nn.Sequential(
            nn.Linear(128 * 2, 64), nn.Tanh(), nn.Linear(64, 1)
        )
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.pool2(x)

        x = x.permute(0, 2, 1)  # (batch, seq, channels)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        att_weights = F.softmax(self.attention(x), dim=1)  # (batch, seq, 1)
        x = torch.sum(att_weights * x, dim=1)  # (batch, features)

        x = self.fc(x)
        return x




## === cell 7
input_channels = 8
num_classes = 6

model = CNNLSTM(in_channels=input_channels, num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.KLDivLoss(reduction="batchmean")
optimizer = optim.Adam(model.parameters(), lr=0.001)

if hasattr(torch, "compile"):
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("torch.compile enabled")
    except Exception as e:
        print("torch.compile unavailable:", repr(e))


def eval_kl(model, loader):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            log_probs = F.log_softmax(outputs, dim=1)
            target_probs = labels.clamp(min=1e-7)
            target_probs = target_probs / target_probs.sum(dim=1, keepdim=True)

            kl_per_sample = (target_probs * (torch.log(target_probs) - log_probs)).sum(
                dim=1
            )
            total += float(kl_per_sample.sum().item())
            n += int(kl_per_sample.numel())

    return total / max(1, n)


num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)

        log_probs = F.log_softmax(outputs, dim=1)
        target_probs = labels.clamp(min=1e-7)
        target_probs = target_probs / target_probs.sum(dim=1, keepdim=True)

        loss = criterion(log_probs, target_probs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / max(1, len(train_loader))
    val_loss = eval_kl(model, val_loader)
    print(f"Epoch {epoch+1}, Train KL: {train_loss:.6f}, Val KL: {val_loss:.6f}")



## === cell 8
test_df = pd.read_csv(TEST_CSV)

testdataset = SpectrogramDataset(
    test_df,
    TEST_SPEC_PATH,
    ycol=None,
    n_freq_bins=8,
    cache_dir="spec_cache",
    cache_tag=f"test_crop{CROP_SECONDS}s_nf8",
    build_cache=True,
    crop_seconds=CROP_SECONDS,
    use_offset=False,
)

test_dataloader = DataLoader(
    testdataset,
    batch_size=64,
    shuffle=False,  # must align with test.csv order
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker,
    collate_fn=pad_collate_1d,
)

model.eval()
predictions = []
with torch.no_grad():
    for inputs in test_dataloader:
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        predictions.append(probabilities.detach().cpu().numpy())

predictions = (
    np.concatenate(predictions, axis=0)
    if len(predictions)
    else np.zeros((0, 6), dtype=np.float32)
)
print("Predictions shape:", predictions.shape)



## === cell 9
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

columns = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

assert len(predictions) == len(test_df), (len(predictions), len(test_df))

pred_df = pd.DataFrame(predictions, columns=columns)
pred_df.insert(0, "eeg_id", test_df["eeg_id"].values)

pred_df = pred_df.groupby("eeg_id", sort=False, as_index=False)[columns].mean()

sub = sample_sub[["eeg_id"]].merge(pred_df, on="eeg_id", how="left", sort=False)

assert len(sub) == len(sample_sub), (len(sub), len(sample_sub))
assert sub[columns].isna().sum().sum() == 0, sub[columns].isna().sum()

sub[columns] = sub[columns].clip(lower=1e-7)
sub[columns] = sub[columns].div(sub[columns].sum(axis=1), axis=0)

sub.to_csv("submission.csv", index=False)
print("Submission shape:", sub.shape)
print(
    "Row sum stats:",
    float(sub[columns].sum(axis=1).min()),
    float(sub[columns].sum(axis=1).max()),
)
print(sub.head())
