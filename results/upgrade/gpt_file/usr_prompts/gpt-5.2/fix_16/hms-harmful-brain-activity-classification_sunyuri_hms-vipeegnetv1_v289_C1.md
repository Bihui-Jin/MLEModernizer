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

3.13

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
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri
email: syuri@tju.edu.cn
"""

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_USE_UPB", "0")

import sys

for _m in list(sys.modules.keys()):
    if _m.startswith("google.protobuf"):
        del sys.modules[_m]

PLATFORM = "kaggle"  # *** local kaggle *** local training or online testing
NEEDTRAIN = False  # train the model

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

LOAD_MODELS_FROM = "models20241118b"  # the path of trained model weights for testing

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

if not os.path.exists(os.path.join(LOAD_DATA_FROM, "train.csv")):
    alt = "/kaggle/input"
    if os.path.exists(os.path.join(alt, "train.csv")):
        LOAD_DATA_FROM = alt

SFREQ = 200  # EEG sampling rate
RSFREQ = 200  # resampled EEG sampling rate

EEG_LENGTH = 50  # the length of EEG data used for each sample
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16  # 16 18

EEG_MULTIPLY = 10

IMG_LENGTH = 20
IMG_HIGH = 324
IMG_WIDE = 324

SPE_HIGH = 100  # the height of the spectrogram
SPE_WIDE = 256  # the width of the spectrogram  10 * 30

STFT_LENGTH = 50
STFT_HIGH = 32  # the height of the STFT (eeg spectrogram)
STFT_WIDE = round(STFT_LENGTH / 0.4)  # the width of the STFT (eeg spectrogram)  50 * 5

filter_range = [0.5, 45]  # eeg filtering range
filter_range2 = [0.1, 35]  # eeg filtering range

SEED = 2024  # seed

BATCHSIZE = 16  # batch size

LEARN_RATE = 1e-3
EPOCHS = 15
PATIENCE = 5

SPLITS = 5

READ_EEG_FILES = False  # preprocess eeg
READ_SPE_FILES = False  # preprocess spectrogram

spectrograms = {}  # preprocessed spectrograms for training
eegs = {}  # preprocessed eegs for training
stfts = {}  # preprocessed short-time fourier transform plots for training
imgs = {}

spectrograms_test = {}
eegs_test = {}
stfts_test = {}
imgs_test = {}

BRAIN = [
    "Fp1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",  # LL
    "Fp1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",  # LP
    "Fz-Cz",
    "Cz-Pz",
    "Fp2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",  # RP
    "Fp2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",  # RL
]

TEST_BATCHSIZE = 128

import warnings

warnings.filterwarnings("ignore")

import io
from PIL import Image
import pandas as pd, numpy as np

from scipy import signal
import time
import gc

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))



## === cell 1
if NEEDTRAIN:
    TARGETS_RAW = [i + "_raw" for i in TARGETS]

    if READ_EEG_FILES:
        train = df.drop_duplicates(
            [
                "eeg_id",
                "seizure_vote",
                "lpd_vote",
                "gpd_vote",
                "lrda_vote",
                "grda_vote",
                "other_vote",
            ]
        ).reset_index(drop=True)
        train["sign_id"] = train.index.values
        df["sign_id"] = df.index.values

        y_data = train[TARGETS].values
        train[TARGETS_RAW] = y_data
        y_data = y_data / y_data.sum(axis=1, keepdims=True)
        train[TARGETS] = y_data

        train.to_csv("train.csv", index=False)
    else:
        train = pd.read_csv("train.csv")




## === cell 2
def _weights_exist() -> bool:
    if not os.path.isdir(LOAD_MODELS_FROM):
        return False
    needed = [
        os.path.join(LOAD_MODELS_FROM, f"fold{i}_stage2.h5") for i in range(SPLITS)
    ]
    return all(os.path.exists(p) for p in needed)




## === cell 3
if not NEEDTRAIN:
    import hashlib
    import multiprocessing as mp

    try:
        import pyarrow.parquet as pq  # type: ignore

        _HAVE_PYARROW = True
    except Exception:
        pq = None
        _HAVE_PYARROW = False

    def _cache_path(tag: str) -> str:
        key = f"{tag}|seed={SEED}|data={LOAD_DATA_FROM}"
        h = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
        return os.path.join("/kaggle/working", f"hms_cache_{tag}_{h}.npz")

    _BRAIN_PAIRS = [tuple(ch.split("-")) for ch in BRAIN]

    def _read_parquet_fast(path: str) -> pd.DataFrame:
        if _HAVE_PYARROW:
            return pq.read_table(path).to_pandas()
        return pd.read_parquet(path)

    def _safe_col_np(df_local: pd.DataFrame, col: str) -> np.ndarray:
        if col in df_local.columns:
            return np.asarray(df_local[col].values, dtype=np.float32)
        return np.zeros(len(df_local), dtype=np.float32)

    def _bandpower_features_from_mean(xm: np.ndarray, fs: int = 200) -> np.ndarray:
        f, pxx = signal.welch(
            xm,
            fs=fs,
            nperseg=min(512, xm.shape[0]),
            noverlap=0,
            scaling="density",
        )
        pxx = np.nan_to_num(pxx, nan=0.0, posinf=0.0, neginf=0.0)

        def bp(lo, hi):
            m = (f >= lo) & (f < hi)
            if not np.any(m):
                return 0.0
            return float(np.trapz(pxx[m], f[m]))

        return np.array([bp(1, 4), bp(4, 8), bp(8, 13), bp(13, 30)], dtype=np.float64)

    def _eeg_stats_from_parquet_path(path: str) -> np.ndarray:
        d = _read_parquet_fast(path)

        sigs = []
        for a_ch, b_ch in _BRAIN_PAIRS:
            va = _safe_col_np(d, a_ch)
            vb = _safe_col_np(d, b_ch)
            v = va - vb
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            sigs.append(v)
        x = np.stack(sigs, axis=0)  # (18, T)

        abs_mean = float(np.mean(np.abs(x)))
        std = float(np.std(x))
        p95 = float(np.quantile(np.abs(x), 0.95))
        ll = float(np.mean(np.abs(np.diff(x, axis=1))))
        rms = float(np.sqrt(np.mean(x * x)))

        xm = np.nan_to_num(np.mean(x, axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )
        bpf = _bandpower_features_from_mean(xm, fs=RSFREQ)  # 4 dims
        tot = float(np.sum(bpf) + 1e-12)
        bpf = np.log((bpf + 1e-12) / tot)

        return np.concatenate(
            [np.array([abs_mean, std, p95, ll, rms], dtype=np.float64), bpf], axis=0
        )

    def _spec_features_from_parquet_path(path: str) -> np.ndarray:
        d = _read_parquet_fast(path)
        x = np.asarray(d.iloc[:, 1:].values, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, a_min=np.exp(-6), a_max=np.exp(8))
        lx = np.log(x + 1e-12)

        C = lx.shape[1]
        if C < 8:
            m = lx.mean()
            s = lx.std()
            q = np.quantile(lx, 0.9)
            return np.array([m, s, q], dtype=np.float64)

        blocks = np.array_split(lx, 4, axis=1)

        feats = []
        for b in blocks:
            v = b.mean(axis=0)
            q30 = float(np.quantile(v, 0.30))
            q60 = float(np.quantile(v, 0.60))
            q90 = float(np.quantile(v, 0.90))
            mu = float(np.mean(v))
            sd = float(np.std(v))
            feats.extend([mu, sd, q30, q60, q90])

        t_mu = lx.mean(axis=1)
        feats.append(float(np.std(t_mu)))
        feats.append(float(np.quantile(t_mu, 0.95) - np.quantile(t_mu, 0.05)))

        return np.asarray(feats, dtype=np.float64)

    def _mp_map_deterministic(func, items, workers: int):
        if workers <= 1:
            return [func(x) for x in items]
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers, maxtasksperchild=200) as pool:
            return pool.map(func, items, chunksize=16)

    def _fallback_predict_from_eeg_features(test_df: pd.DataFrame) -> np.ndarray:
        vote_cols = list(TARGETS)

        eps_prior = 1.0
        train_votes = df[vote_cols].sum(axis=0).values.astype(np.float64)
        global_prior = (train_votes + eps_prior) / np.sum(train_votes + eps_prior)

        df_patient = df[["patient_id"] + vote_cols].copy()
        pv = df_patient.groupby("patient_id")[vote_cols].sum()
        pv = (pv + eps_prior).div((pv + eps_prior).sum(axis=1), axis=0)

        train_eeg_dir = os.path.join(LOAD_DATA_FROM, "train_eegs")
        test_eeg_dir = os.path.join(LOAD_DATA_FROM, "test_eegs")

        eeg_soft = df.groupby("eeg_id")[vote_cols].sum()
        eeg_soft = (eeg_soft + 1e-12).div((eeg_soft + 1e-12).sum(axis=1), axis=0)
        eeg_patient = df.groupby("eeg_id")["patient_id"].agg(lambda x: x.iloc[0])

        train_eeg_ids = eeg_soft.index.values
        train_u = pd.DataFrame(
            {
                "eeg_id": train_eeg_ids,
                "patient_id": eeg_patient.reindex(train_eeg_ids).values,
            }
        )
        for c in vote_cols:
            train_u[c] = eeg_soft[c].values

        max_train_eegs = 8500
        if len(train_u) > max_train_eegs:
            rs = np.random.RandomState(SEED)
            idx = rs.choice(len(train_u), size=max_train_eegs, replace=False)
            train_u = train_u.iloc[np.sort(idx)].reset_index(drop=True)

        cache_file = _cache_path("fallback_eeg_train")
        if os.path.exists(cache_file):
            z = np.load(cache_file, allow_pickle=False)
            train_stats = z["train_stats"]
            train_targets = z["train_targets"]
        else:
            work_paths = []
            work_targets = []
            for eeg_id, *probs in train_u[["eeg_id"] + vote_cols].itertuples(
                index=False, name=None
            ):
                p = os.path.join(train_eeg_dir, f"{int(eeg_id)}.parquet")
                if not os.path.exists(p):
                    continue
                vt = np.array(probs, dtype=np.float64)
                vt = vt / np.clip(vt.sum(), 1e-12, None)
                work_paths.append(p)
                work_targets.append(vt)

            workers = min(8, max(1, (os.cpu_count() or 2) - 1))
            feats = _mp_map_deterministic(
                _eeg_stats_from_parquet_path, work_paths, workers
            )
            train_stats_list = []
            train_targets_list = []
            for s, t in zip(feats, work_targets):
                if s is None:
                    continue
                train_stats_list.append(s)
                train_targets_list.append(t)

            train_stats = np.asarray(train_stats_list, dtype=np.float64)
            train_targets = np.asarray(train_targets_list, dtype=np.float64)
            np.savez_compressed(
                cache_file, train_stats=train_stats, train_targets=train_targets
            )

        if train_stats.shape[0] < 300:
            preds = np.tile(global_prior[None, :], (len(test_df), 1))
            preds = np.clip(preds, 1e-7, 1.0)
            preds = preds / preds.sum(axis=1, keepdims=True)
            return preds.astype(np.float32)

        mu = train_stats.mean(axis=0, keepdims=True)
        sig = train_stats.std(axis=0, keepdims=True) + 1e-6
        train_z = (train_stats - mu) / sig  # (N,F)

        K = train_targets.shape[1]
        F = train_z.shape[1]
        class_mu = np.zeros((K, F), dtype=np.float64)
        class_prior = train_targets.mean(axis=0).astype(np.float64)
        class_prior = (class_prior + 1e-12) / np.sum(class_prior + 1e-12)

        for k in range(K):
            w = train_targets[:, k : k + 1]
            denom = np.sum(w) + 1e-12
            class_mu[k] = np.sum(train_z * w, axis=0) / denom

        resid2 = np.zeros(F, dtype=np.float64)
        for k in range(K):
            w = train_targets[:, k]
            diff = train_z - class_mu[k][None, :]
            resid2 += (w[:, None] * (diff * diff)).sum(axis=0)
        resid2 = resid2 / (np.sum(train_targets) + 1e-12)
        resid2 = np.maximum(resid2, 1e-6)

        shrink = 0.35
        var = (1.0 - shrink) * resid2 + shrink * 1.0
        inv_var = 1.0 / (var + 1e-12)

        y_row = df[vote_cols].values.astype(np.float64)
        y_row = y_row / np.clip(y_row.sum(axis=1, keepdims=True), 1e-12, None)
        dominance = np.max(y_row, axis=1)
        dom_med = float(np.median(dominance))
        temp = 1.10 + 1.3 * (0.62 - dom_med)
        temp = float(np.clip(temp, 1.02, 2.0))

        alpha_global = float(np.clip(0.16 + 0.14 * (0.60 - dom_med), 0.10, 0.28))
        alpha_patient = 0.16  # keep modest

        cache_test = _cache_path("fallback_eeg_test")
        if os.path.exists(cache_test):
            zt = np.load(cache_test, allow_pickle=False)
            test_stats = zt["test_stats"]
        else:
            test_paths = [
                os.path.join(test_eeg_dir, f"{int(eid)}.parquet")
                for eid in test_df["eeg_id"].values
            ]
            workers = min(8, max(1, (os.cpu_count() or 2) - 1))
            test_feats = _mp_map_deterministic(
                _eeg_stats_from_parquet_path, test_paths, workers
            )
            test_stats = np.asarray(test_feats, dtype=np.float64)
            np.savez_compressed(cache_test, test_stats=test_stats)

        preds = np.zeros((len(test_df), K), dtype=np.float64)
        patient_ids = test_df["patient_id"].values
        for i in range(len(test_df)):
            patient_id = patient_ids[i]

            patient_prior = None
            if patient_id in pv.index:
                patient_prior = pv.loc[patient_id].values.astype(np.float64)
                patient_prior = np.clip(patient_prior, 1e-12, 1.0)
                patient_prior = patient_prior / np.sum(patient_prior)

            try:
                s = test_stats[i]
                zrow = (s[None, :] - mu) / sig  # (1,F)
                d2 = ((zrow - class_mu[None, :, :]) ** 2) * inv_var[None, None, :]
                d2 = np.sum(d2, axis=-1)[0]  # (K,)
                logits = (-0.5 * d2 + np.log(class_prior + 1e-12)) / temp
                probs = np.exp(logits - np.max(logits))
                probs = probs / (np.sum(probs) + 1e-12)
            except Exception:
                probs = global_prior.copy()

            probs = (1.0 - alpha_global) * probs + alpha_global * global_prior
            if patient_prior is not None:
                probs = (1.0 - alpha_patient) * probs + alpha_patient * patient_prior

            probs = np.clip(probs, 1e-7, 1.0)
            probs = probs / np.sum(probs)
            preds[i] = probs

        return preds.astype(np.float32)

    def _fallback_predict_from_spectrogram_features(
        test_df: pd.DataFrame,
    ) -> np.ndarray:
        vote_cols = list(TARGETS)

        eps_prior = 1.0
        train_votes = df[vote_cols].sum(axis=0).values.astype(np.float64)
        global_prior = (train_votes + eps_prior) / np.sum(train_votes + eps_prior)

        df_patient = df[["patient_id"] + vote_cols].copy()
        pv = df_patient.groupby("patient_id")[vote_cols].sum()
        pv = (pv + eps_prior).div((pv + eps_prior).sum(axis=1), axis=0)

        train_spec_dir = os.path.join(LOAD_DATA_FROM, "train_spectrograms")
        test_spec_dir = os.path.join(LOAD_DATA_FROM, "test_spectrograms")

        spec_soft = df.groupby("spectrogram_id")[vote_cols].sum()
        spec_soft = (spec_soft + 1e-12).div((spec_soft + 1e-12).sum(axis=1), axis=0)
        spec_patient = df.groupby("spectrogram_id")["patient_id"].agg(
            lambda x: x.iloc[0]
        )

        train_spec_ids = spec_soft.index.values
        train_u = pd.DataFrame(
            {
                "spectrogram_id": train_spec_ids,
                "patient_id": spec_patient.reindex(train_spec_ids).values,
            }
        )
        for c in vote_cols:
            train_u[c] = spec_soft[c].values

        max_train_specs = 6500
        if len(train_u) > max_train_specs:
            rs = np.random.RandomState(SEED)
            idx = rs.choice(len(train_u), size=max_train_specs, replace=False)
            train_u = train_u.iloc[np.sort(idx)].reset_index(drop=True)

        cache_file = _cache_path("fallback_spe_train")
        if os.path.exists(cache_file):
            z = np.load(cache_file, allow_pickle=False)
            train_stats = z["train_stats"]
            train_targets = z["train_targets"]
        else:
            work_paths = []
            work_targets = []
            for spec_id, *probs in train_u[["spectrogram_id"] + vote_cols].itertuples(
                index=False, name=None
            ):
                p = os.path.join(train_spec_dir, f"{int(spec_id)}.parquet")
                if not os.path.exists(p):
                    continue
                vt = np.array(probs, dtype=np.float64)
                vt = vt / np.clip(vt.sum(), 1e-12, None)
                work_paths.append(p)
                work_targets.append(vt)

            workers = min(8, max(1, (os.cpu_count() or 2) - 1))
            feats = _mp_map_deterministic(
                _spec_features_from_parquet_path, work_paths, workers
            )

            train_stats_list = []
            train_targets_list = []
            for s, t in zip(feats, work_targets):
                if s is None:
                    continue
                train_stats_list.append(s)
                train_targets_list.append(t)

            train_stats = np.asarray(train_stats_list, dtype=np.float64)
            train_targets = np.asarray(train_targets_list, dtype=np.float64)
            np.savez_compressed(
                cache_file, train_stats=train_stats, train_targets=train_targets
            )

        if train_stats.shape[0] < 300:
            preds = np.tile(global_prior[None, :], (len(test_df), 1))
            preds = np.clip(preds, 1e-7, 1.0)
            preds = preds / preds.sum(axis=1, keepdims=True)
            return preds.astype(np.float32)

        mu = train_stats.mean(axis=0, keepdims=True)
        sig = train_stats.std(axis=0, keepdims=True) + 1e-6
        train_z = (train_stats - mu) / sig

        K = train_targets.shape[1]
        F = train_z.shape[1]
        class_mu = np.zeros((K, F), dtype=np.float64)
        class_prior = train_targets.mean(axis=0).astype(np.float64)
        class_prior = (class_prior + 1e-12) / np.sum(class_prior + 1e-12)

        for k in range(K):
            w = train_targets[:, k : k + 1]
            denom = np.sum(w) + 1e-12
            class_mu[k] = np.sum(train_z * w, axis=0) / denom

        resid2 = np.zeros(F, dtype=np.float64)
        for k in range(K):
            w = train_targets[:, k]
            diff = train_z - class_mu[k][None, :]
            resid2 += (w[:, None] * (diff * diff)).sum(axis=0)
        resid2 = resid2 / (np.sum(train_targets) + 1e-12)
        resid2 = np.maximum(resid2, 1e-6)

        shrink = 0.45
        var = (1.0 - shrink) * resid2 + shrink * 1.0
        inv_var = 1.0 / (var + 1e-12)

        temp = 1.05
        alpha_global = 0.14
        alpha_patient = 0.14

        cache_test = _cache_path("fallback_spe_test")
        if os.path.exists(cache_test):
            zt = np.load(cache_test, allow_pickle=False)
            test_stats = zt["test_stats"]
        else:
            test_paths = [
                os.path.join(test_spec_dir, f"{int(sid)}.parquet")
                for sid in test_df["spectrogram_id"].values
            ]
            workers = min(8, max(1, (os.cpu_count() or 2) - 1))
            test_feats = _mp_map_deterministic(
                _spec_features_from_parquet_path, test_paths, workers
            )
            test_stats = np.asarray(test_feats, dtype=np.float64)
            np.savez_compressed(cache_test, test_stats=test_stats)

        preds = np.zeros((len(test_df), K), dtype=np.float64)
        patient_ids = test_df["patient_id"].values
        for i in range(len(test_df)):
            patient_id = patient_ids[i]

            patient_prior = None
            if patient_id in pv.index:
                patient_prior = pv.loc[patient_id].values.astype(np.float64)
                patient_prior = np.clip(patient_prior, 1e-12, 1.0)
                patient_prior = patient_prior / np.sum(patient_prior)

            try:
                s = test_stats[i]
                zrow = (s[None, :] - mu) / sig
                d2 = ((zrow - class_mu[None, :, :]) ** 2) * inv_var[None, None, :]
                d2 = np.sum(d2, axis=-1)[0]
                logits = (-0.5 * d2 + np.log(class_prior + 1e-12)) / temp
                probs = np.exp(logits - np.max(logits))
                probs = probs / (np.sum(probs) + 1e-12)
            except Exception:
                probs = global_prior.copy()

            probs = (1.0 - alpha_global) * probs + alpha_global * global_prior
            if patient_prior is not None:
                probs = (1.0 - alpha_patient) * probs + alpha_patient * patient_prior

            probs = np.clip(probs, 1e-7, 1.0)
            probs = probs / np.sum(probs)
            preds[i] = probs

        return preds.astype(np.float32)

    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    test["sign_id"] = test.index.values
    print("Test shape", test.shape)

    sample_sub = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))
    sample_sub = sample_sub.copy()
    TARGETS = [c for c in sample_sub.columns if c != "eeg_id"]

    if not _weights_exist():
        print(f"WARNING: Model weights not found under: {LOAD_MODELS_FROM}")
        print("Using improved EEG+spectrogram feature fallback (two-posteriors blend).")

        preds_eeg = _fallback_predict_from_eeg_features(test)
        preds_spe = _fallback_predict_from_spectrogram_features(test)

        preds_eeg = np.asarray(preds_eeg, dtype=np.float64)
        preds_spe = np.asarray(preds_spe, dtype=np.float64)

        w_spe = 0.62
        preds_all = (1.0 - w_spe) * preds_eeg + w_spe * preds_spe

        t_post = 1.08
        preds_all = np.clip(preds_all, 1e-12, 1.0)
        preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)
        preds_all = preds_all ** (1.0 / t_post)
        preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)

        preds_all = np.clip(preds_all, 1e-7, 1.0)
        preds_all = preds_all / np.sum(preds_all, axis=1, keepdims=True)

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[TARGETS] = preds_all.astype(np.float32)
        sub = sub[sample_sub.columns]
        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
        print(sub.head())
    else:
        import tensorflow as tf
        from tensorflow.keras import optimizers
        import matplotlib
        import matplotlib.pyplot as plt

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) <= 1:
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            print(f"Using {len(gpus)} GPU")
        else:
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using {len(gpus)} GPUs")

        np.random.seed(SEED)
        os.environ["PYTHONHASHSEED"] = str(SEED)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        tf.random.set_seed(SEED)
        tf.keras.utils.set_random_seed(SEED)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

        MIX = True
        if MIX:
            try:
                tf.config.optimizer.set_experimental_options(
                    {"auto_mixed_precision": True}
                )
                print("Mixed precision enabled")
            except Exception:
                print("Mixed precision not available; continuing")
        else:
            print("Using full precision")

        length = round(32 / (EEG_MULTIPLY / 10))
        x = np.linspace(1, length, length)
        y = x * 0
        y[15:] = 1
        WEIGHTS = np.concatenate((y[: round(length / 2)], y[: round(length / 2)][::-1]))
        WEIGHTS = WEIGHTS / np.sum(WEIGHTS)
        WEIGHTS = np.reshape(WEIGHTS, [1, -1, 1])  # (1, T, 1)
        EEG_WEIGHTS_f = tf.convert_to_tensor(WEIGHTS, dtype=tf.float32)

        length = 8
        x = np.linspace(1, length, length)
        y = x * 0
        y[3:] = 1
        WEIGHTS = np.concatenate((y[: round(length / 2)], y[: round(length / 2)][::-1]))
        WEIGHTS = WEIGHTS / np.sum(WEIGHTS)
        WEIGHTS = np.reshape(WEIGHTS, [1, -1, 1])  # (1, T, 1)
        SPE_WEIGHTS_f = tf.convert_to_tensor(WEIGHTS, dtype=tf.float32)

        class DataGenerator(tf.keras.utils.Sequence):
            def __init__(
                self,
                dataframe,
                batch_size=32,
                shuffle=False,
                sample_weights=False,
                mode="train",
                eegs=None,
                stfts=None,
                specs=None,
                imgs=None,
            ):

                self.dataframe = dataframe
                self.batch_size = batch_size
                self.shuffle = shuffle
                self.sample_weights = sample_weights
                self.mode = mode
                self.eegs = eegs
                self.stfts = stfts
                self.specs = specs
                self.imgs = imgs
                self.on_epoch_end()

            def __len__(self):
                ct = int(np.ceil(len(self.dataframe) / self.batch_size))
                return ct

            def __getitem__(self, index):
                indexes = self.indexes[
                    index * self.batch_size : (index + 1) * self.batch_size
                ]
                x, y, sample_weights = self.__data_generation(indexes)

                if self.mode == "test":
                    return x
                return x, y, sample_weights

            def on_epoch_end(self):
                self.indexes = np.arange(len(self.dataframe))
                if self.shuffle:
                    np.random.shuffle(self.indexes)

            def __data_generation(self, indexes):
                if "spe" in DATATYPE:
                    x_spe = np.zeros(
                        (len(indexes), 4, SPE_HIGH, SPE_WIDE), dtype="float32"
                    )
                if "eeg" in DATATYPE:
                    x_eeg = np.zeros(
                        (
                            len(indexes),
                            EEG_CHANNEL_USED * EEG_MULTIPLY,
                            round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                        ),
                        dtype="float32",
                    )
                if "stft" in DATATYPE:
                    x_stft = np.zeros(
                        (len(indexes), STFT_HIGH * 9, STFT_WIDE * 2), dtype="float32"
                    )
                if "img" in DATATYPE:
                    x_img = np.zeros(
                        (len(indexes), IMG_HIGH, IMG_WIDE, 3), dtype="float32"
                    )

                y_out = np.zeros((len(indexes), len(TARGETS)), dtype="float32")
                sample_weights_out = np.zeros((len(indexes), 1), dtype="float32")

                for j, i in enumerate(indexes):
                    row = self.dataframe.iloc[i]
                    sign_id = row.sign_id
                    if self.mode != "test":
                        sample_weight = 1.0

                    if self.mode == "test":
                        r_spe = 0
                        r_eeg = 0
                        r_stft = 0
                    else:
                        r_spe = 0
                        r_eeg = 0

                    if "spe" in DATATYPE:
                        spe = []
                        for k in range(4):
                            spe.append(
                                np.reshape(
                                    self.specs[row.spectrogram_id][
                                        r_spe : (r_spe + 300), k * 100 : (k + 1) * 100
                                    ].T,
                                    (1, 100, 300),
                                )
                            )
                        spe = np.concatenate(spe, axis=0)

                    if "eeg" in DATATYPE:
                        eeg = self.eegs[row.eeg_id][
                            :, round(r_eeg * RSFREQ) : round((r_eeg + 50) * RSFREQ)
                        ]

                    if "stft" in DATATYPE:
                        stft_t = self.stfts[-row.eeg_id]
                        r_stft = (np.where(stft_t >= (r_eeg - min(stft_t))))[0][0]
                        stft = self.stfts[row.eeg_id][
                            :, :, r_stft : (r_stft + STFT_WIDE)
                        ]
                        if stft.shape[2] < STFT_WIDE:
                            stft = np.concatenate((stft, stft[:, :, ::-1]), 2)
                            stft = stft[:, :, :STFT_WIDE]

                    if "img" in DATATYPE:
                        img = self.imgs[sign_id]

                    if "spe" in DATATYPE:
                        spe[np.isnan(spe)] = 0
                        spe = np.clip(spe, a_min=np.exp(-4), a_max=np.exp(6))
                        spe = np.log(spe)
                        spe = spe[
                            :,
                            :,
                            round((spe.shape[2] - SPE_WIDE) / 2) : -round(
                                (spe.shape[2] - SPE_WIDE) * 1.0 / 2
                            ),
                        ]
                        spe = (spe - np.mean(spe, keepdims=True)) / (
                            np.std(spe, keepdims=True) + 1e-6
                        )
                        x_spe[j] = spe

                    if "eeg" in DATATYPE:
                        eeg = eeg[
                            :,
                            round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2) : round(
                                (EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2
                            ),
                        ]
                        eeg_save = np.zeros(
                            (x_eeg.shape[1], x_eeg.shape[2]), dtype=np.float32
                        )
                        eeg = np.concatenate(
                            (
                                eeg[0 : round(EEG_CHANNEL_USED / 2), :],
                                eeg[-round(EEG_CHANNEL_USED / 2) :, :],
                            ),
                            axis=0,
                        )
                        eeg2 = eeg.copy()
                        eeg[4:8, :] = eeg2[12:16, :]
                        eeg[8:12, :] = eeg2[4:8, :]
                        eeg[12:16, :] = eeg2[8:12, :]
                        for ii in range(eeg_save.shape[0]):
                            eeg_save[ii, :] = eeg[
                                ii // EEG_MULTIPLY, ii % EEG_MULTIPLY :: EEG_MULTIPLY
                            ]

                        eeg = (eeg_save - np.mean(eeg_save, keepdims=True)) / (
                            np.std(eeg_save, keepdims=True) + 1e-6
                        )
                        x_eeg[j] = eeg

                    if self.mode != "test":
                        ysum = float(np.sum(row[TARGETS].values))
                        if ysum <= 0:
                            y_out[j] = np.ones(len(TARGETS), dtype=np.float32) / len(
                                TARGETS
                            )
                        else:
                            y_out[j] = row[TARGETS].values / ysum
                        if self.sample_weights:
                            sample_weights_out[j] = sample_weight
                        else:
                            sample_weights_out[j] = 1

                x = []
                if "spe" in DATATYPE:
                    x.append(x_spe)
                if "eeg" in DATATYPE:
                    x.append(x_eeg)
                if "stft" in DATATYPE:
                    x.append(x_stft)
                if "img" in DATATYPE:
                    x.append(x_img)

                return x, y_out, sample_weights_out

        class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
            def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
                super(CosineAnnealingLRScheduler, self).__init__()
                self.total_step = total_step
                self.warm_step = 1 if warmth_rate == 0 else int(warmth_rate)
                self.lr_max = lr_max
                self.lr_min = lr_min

            @tf.function
            def __call__(self, step):
                step = step + 1
                if step < self.warm_step:
                    lr = self.lr_max / self.warm_step * step
                else:
                    lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                        1.0
                        + tf.cos(
                            (step - self.warm_step)
                            / (self.total_step - self.warm_step)
                            * np.pi
                        )
                    )
                return lr

        def _make_effnet_b0(name: str):
            base = tf.keras.applications.EfficientNetB0(
                include_top=False, weights=None, input_shape=None
            )
            base._name = name
            return base

        def build_model():
            inp = []
            y = None

            def _apply_time_weights_4d(feat4d, weights_1d, name_prefix: str):
                x = tf.keras.layers.Permute((2, 1, 3), name=f"{name_prefix}_permute")(
                    feat4d
                )

                def _mul_with_resized_weights(t):
                    x_local, w_local = t
                    wlen = tf.shape(w_local)[1]
                    xlen = tf.shape(x_local)[1]
                    w = tf.cond(
                        tf.equal(wlen, xlen),
                        lambda: w_local,
                        lambda: tf.image.resize(
                            w_local, size=(xlen, 1), method="bilinear"
                        ),
                    )
                    w = w / (tf.reduce_sum(w, axis=1, keepdims=True) + 1e-12)
                    return x_local * w

                x = tf.keras.layers.Reshape(
                    (x.shape[1], x.shape[2] * x.shape[3]), name=f"{name_prefix}_reshape"
                )(x)
                x = tf.keras.layers.Lambda(
                    _mul_with_resized_weights, name=f"{name_prefix}_mul"
                )([x, weights_1d])
                x = tf.keras.layers.Lambda(
                    lambda t: tf.reduce_sum(t, axis=1, keepdims=True),
                    name=f"{name_prefix}_sum",
                )(x)
                return x

            if "spe" in DATATYPE:
                inp_spe = tf.keras.Input(shape=(4, SPE_HIGH, SPE_WIDE))
                x_spe = tf.keras.layers.Concatenate(axis=1)(
                    [
                        inp_spe[:, 0, :, :],
                        inp_spe[:, 1, :, :],
                        inp_spe[:, 2, :, :],
                        inp_spe[:, 3, :, :],
                    ]
                )
                x_spe = tf.keras.layers.Reshape((x_spe.shape[1], x_spe.shape[2], 1))(
                    x_spe
                )
                x_spe = tf.keras.layers.Concatenate(axis=-1)([x_spe, x_spe, x_spe])

                base_model_spe = _make_effnet_b0("spe_extractor")
                x_spe = base_model_spe(x_spe)

                x_spe = _apply_time_weights_4d(x_spe, SPE_WEIGHTS_f, "spe_weights")
                x_spe = tf.keras.layers.GlobalAveragePooling1D()(x_spe)
                x_spe = tf.keras.layers.Dropout(0.2)(x_spe)

                inp.append(inp_spe)
                y = x_spe

            if "eeg" in DATATYPE:
                inp_eeg = tf.keras.Input(
                    shape=(
                        EEG_CHANNEL_USED * EEG_MULTIPLY,
                        round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                    )
                )
                x_eeg = tf.keras.layers.Reshape(
                    (inp_eeg.shape[1], inp_eeg.shape[2], 1)
                )(inp_eeg)
                x_eeg = tf.keras.layers.Concatenate(axis=-1)([x_eeg, x_eeg, x_eeg])

                base_model_eeg = _make_effnet_b0("eeg_extractor")
                x_eeg = base_model_eeg(x_eeg)

                x_eeg = _apply_time_weights_4d(x_eeg, EEG_WEIGHTS_f, "eeg_weights")
                x_eeg = tf.keras.layers.GlobalAveragePooling1D()(x_eeg)
                x_eeg = tf.keras.layers.Dropout(0.2)(x_eeg)

                inp.append(inp_eeg)
                if y is not None:
                    y = tf.keras.layers.Concatenate(axis=1)([y, x_eeg])
                else:
                    y = x_eeg

            if "stft" in DATATYPE:
                inp_stft = tf.keras.Input(shape=(STFT_HIGH * 9, STFT_WIDE * 2))
                x_stft = tf.keras.layers.Reshape(
                    (inp_stft.shape[1], inp_stft.shape[2], 1)
                )(inp_stft)
                x_stft = tf.keras.layers.Concatenate(axis=-1)([x_stft, x_stft, x_stft])

                base_model_stft = _make_effnet_b0("stft_extractor")
                x_stft = base_model_stft(x_stft)
                x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)

                inp.append(inp_stft)
                if y is not None:
                    y = tf.keras.layers.Concatenate(axis=1)([y, x_stft])
                else:
                    y = x_stft

            if "img" in DATATYPE:
                inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3))
                base_model_img = _make_effnet_b0("img_extractor")
                x_img = base_model_img(inp_img)
                x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)

                inp.append(inp_img)
                if y is not None:
                    y = tf.keras.layers.Concatenate(axis=1)([y, x_img])
                else:
                    y = x_img

            y = tf.keras.layers.Dense(
                len(TARGETS), activation="softmax", dtype="float32"
            )(y)
            model = tf.keras.Model(inputs=inp, outputs=y)
            return model

        preds_all = []
        models = []

        with strategy.scope():
            for model_i in range(SPLITS):
                print(f"Fold {model_i + 1}")
                model = build_model()
                wpath = os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.h5")
                model.load_weights(wpath, by_name=True, skip_mismatch=True)
                models.append(model)

        if "spe" in DATATYPE:
            PATH_test = os.path.join(LOAD_DATA_FROM, "test_spectrograms") + "/"
            files_test = os.listdir(PATH_test)
            print(f"There are {len(files_test)} test spectrogram parquets")
            for i, f in enumerate(files_test):
                if i % 100 == 0:
                    print(i, ", ", end="")
                tmp = pd.read_parquet(f"{PATH_test}{f}")
                name = int(f.split(".")[0])
                spectrograms_test[name] = tmp.iloc[:, 1:].values

        PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"

        if (
            ("spe" in DATATYPE)
            or ("eeg" in DATATYPE)
            or ("stft" in DATATYPE)
            or ("img" in DATATYPE)
        ):
            b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")
            b2, a2 = signal.butter(
                3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass"
            )

            batch_start = 0

            for i, eeg_id in enumerate(test.eeg_id):
                if i % 100 == 0:
                    print(i, ", ", end="")

                eeg_default = pd.read_parquet(
                    os.path.join(PATH_test, (str(eeg_id) + ".parquet"))
                )

                eeg = []
                for channel in BRAIN:
                    a_ch, b_ch = channel.split("-")
                    if (a_ch in eeg_default.columns) and (b_ch in eeg_default.columns):
                        eeg_temp = (
                            eeg_default.loc[:, a_ch] - eeg_default.loc[:, b_ch]
                        ).values
                    else:
                        eeg_temp = np.zeros(len(eeg_default), dtype=np.float32)
                    eeg_temp = np.asarray(eeg_temp, dtype=np.float32)
                    eeg_temp[np.isnan(eeg_temp)] = 0
                    eeg.append(np.reshape(eeg_temp, (1, -1)))
                eeg = np.concatenate(eeg, axis=0)

                if SFREQ != RSFREQ:
                    eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

                if "stft" in DATATYPE:
                    eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)
                    ff, tt, ss = signal.spectrogram(
                        eeg2, axis=1, fs=RSFREQ, nperseg=RSFREQ, noverlap=60, nfft=160
                    )
                    ss[np.isnan(ss)] = 0
                    ss = ss[:, (ff > 0) * (ff <= 20), :]

                if "img" in DATATYPE:
                    eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)
                    eeg2 = np.clip(eeg2, a_min=-1024, a_max=1024)

                    train_plot = test[test.eeg_id == eeg_id].reset_index(drop=True)
                    for j in range(len(train_plot)):
                        eeg_plot = eeg2[:, 0 : EEG_LENGTH * RSFREQ]
                        eeg_plot = eeg_plot[
                            :,
                            round((EEG_LENGTH - IMG_LENGTH) / 2 * RSFREQ) : round(
                                (EEG_LENGTH + IMG_LENGTH) / 2 * RSFREQ
                            ),
                        ]

                        img_save = np.zeros(
                            (eeg_plot.shape[0], 36, IMG_WIDE), dtype=np.float32
                        )
                        for ii in range(eeg_plot.shape[0]):
                            fig = plt.figure(clear=True, figsize=(3.93, 2 / 18 * 2))
                            fig.patch.set_facecolor("black")

                            plt.plot(eeg_plot[ii, :] + 100, color="red", linewidth=0.2)
                            plt.xlim(-5, eeg_plot.shape[1] + 5)
                            plt.ylim(0, 200)
                            plt.axis("off")

                            byte_stream = io.BytesIO()
                            plt.savefig(
                                byte_stream, format="png", bbox_inches="tight", dpi=100
                            )
                            byte_stream.seek(0)
                            img = Image.open(byte_stream)
                            img = np.array(img)[:, :, :1]
                            img = img / 255
                            img = np.array(img, dtype=np.float32)
                            byte_stream.truncate()
                            plt.close("all")

                            if img.shape != (36, IMG_WIDE, 1):
                                img = np.concatenate((img, img, img), 2)
                                img = np.array(
                                    tf.image.resize(img, (36, IMG_WIDE)),
                                    dtype=np.float32,
                                )
                            img = img[:, :, 0]
                            img_save[ii, :, :] = img

                        imgs_test[train_plot.sign_id[j]] = img_save

                eeg = signal.filtfilt(b, a, eeg, axis=1)
                eeg = np.clip(eeg, a_min=-1024, a_max=1024)

                if "eeg" in DATATYPE:
                    eegs_test[eeg_id] = eeg
                if "stft" in DATATYPE:
                    stfts_test[eeg_id] = ss
                    stfts_test[-eeg_id] = tt

                is_batch_end = ((i + 1) % TEST_BATCHSIZE == 0) or (
                    (i + 1) == len(test.eeg_id)
                )
                if is_batch_end:
                    batch_end = i + 1
                    batch_df = test.iloc[batch_start:batch_end].reset_index(drop=True)

                    preds = []
                    test_gen = DataGenerator(
                        batch_df,
                        shuffle=False,
                        sample_weights=False,
                        batch_size=TEST_BATCHSIZE,
                        mode="test",
                        specs=spectrograms_test,
                        eegs=eegs_test,
                        stfts=stfts_test,
                        imgs=imgs_test,
                    )
                    for model_i in range(SPLITS):
                        pred = models[model_i].predict(test_gen, verbose=0)
                        preds.append(pred)
                    pred = np.mean(preds, axis=0)

                    eegs_test = {}
                    stfts_test = {}
                    imgs_test = {}
                    gc.collect()

                    if len(preds_all) == 0:
                        preds_all = pred.copy()
                    else:
                        preds_all = np.concatenate((preds_all, pred), axis=0)

                    batch_start = batch_end

        preds_all = np.asarray(preds_all, dtype=np.float32)
        if preds_all.shape[0] != len(test):
            raise RuntimeError(
                f"Prediction rows ({preds_all.shape[0]}) != test rows ({len(test)}). Check batching."
            )

        preds_all = np.clip(preds_all, 1e-7, 1.0)
        preds_all = preds_all / np.sum(preds_all, axis=1, keepdims=True)

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[TARGETS] = preds_all
        sub = sub[sample_sub.columns]
        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
        print(sub.head())
