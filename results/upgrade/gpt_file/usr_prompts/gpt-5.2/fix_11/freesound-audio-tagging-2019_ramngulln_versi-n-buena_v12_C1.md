# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Develop a model to tag audio data automatically using a diverse vocabulary of 80 categories.

## Metric
The task consists of predicting the audio labels (tags) for every test clip. Some test clips bear one label while others bear several labels. The predictions are to be done at the clip level, i.e., no start/end timestamps for the sound events are required.

The primary metric is label-weighted label-ranking average precision. 

The  "label-weighted" part means that the overall score is the average over all the *labels* in the test set, where each label receives equal weight (by contrast, plain *lrap* gives each *test item* equal weight).

## Submission Format
For each `fname` in the test set, you must predict the probability of each label. The file should contain a header and have the following format:

```
fname,Accelerating_and_revving_and_vroom,...Zipper_(clothing)
000ccb97.wav,0.1,....,0.3
0012633b.wav,0.0,...,0.8
```

## Dataset
The following 5 audio files in the curated train set have a wrong label, due to a bug in the file renaming process:\
`f76181c4.wav, 77b925c2.wav, 6a1f682a.wav, c7db12aa.wav, 7752cc8a.wav`

The audio file `1d44b0bd.wav` in the curated train set was found to be corrupted (contains no signal) due to an error in format conversion.

- **train_curated.csv** - ground truth labels for the curated subset of the training audio files (see Data Fields below)
- **train_noisy.csv** - ground truth labels for the noisy subset of the training audio files (see Data Fields below)
- **sample_submission.csv** - a sample submission file in the correct format, including the correct sorting of the sound categories; it contains the list of audio files found in the test.zip folder (corresponding to the public leaderboard)
- **train_curated.zip** - a folder containing the audio (.wav) training files of the curated subset
- **train_noisy.zip** - a folder containing the audio (.wav) training files of the noisy subset
- **test.zip** - a folder containing the audio (.wav) test files for the public leaderboard

### Columns
Each row of the train_curated.csv and train_noisy.csv files contains the following information:

- **fname**: the audio file name, eg, `0006ae4e.wav`
- **labels**: the audio classification label(s) (ground truth). Note that the number of labels per clip can be one, eg, `Bark` or more, eg, `"Walk_and_footsteps,Slam"`.

# 2. Python version

3.9

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
pillow==11.3.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (276 lines)
            sample_submission.csv (3362 lines)
            sample_submission.csv.zip (20.7 kB)
            test.zip (2.2 GB)
            train_curated.csv (4971 lines)
            train_curated.csv.zip (39.3 kB)
            train_curated.zip (2.4 GB)
            train_noisy.csv (19816 lines)
            train_noisy.csv.zip (154.2 kB)
            train_noisy.zip (21.5 GB)
            freesound-audio-tagging-2019/
                description.md (276 lines)
                sample_submission.csv (3362 lines)
                ... and 8 other files
                freesound-audio-tagging-2019/
                test/
                    4260ebea.wav (1.0 MB)
                    426eb1e0.wav (654.5 kB)
                    ... and 3359 other files
                    test/
                train_curated/
                    0006ae4e.wav (621.0 kB)
                    0019ef41.wav (181.3 kB)
                    ... and 4968 other files
                train_noisy/
                    00097e21.wav (1.3 MB)
                    000b6cfb.wav (1.3 MB)
                    ... and 19813 other files
            test/
                4260ebea.wav (1.0 MB)
                426eb1e0.wav (654.5 kB)
                ... and 3359 other files
                test/
            train_curated/
                0006ae4e.wav (621.0 kB)
                0019ef41.wav (181.3 kB)
                ... and 4968 other files
            train_noisy/
                00097e21.wav (1.3 MB)
                000b6cfb.wav (1.3 MB)
                ... and 19813 other files
        input/
            description.md (276 lines)
            sample_submission.csv (3362 lines)
            sample_submission.csv.zip (20.7 kB)
            test.zip (2.2 GB)
            train_curated.csv (4971 lines)
            train_curated.csv.zip (39.3 kB)
            train_curated.zip (2.4 GB)
            train_noisy.csv (19816 lines)
            train_noisy.csv.zip (154.2 kB)
            train_noisy.zip (21.5 GB)
            freesound-audio-tagging-2019/
                description.md (276 lines)
                sample_submission.csv (3362 lines)
                ... and 8 other files
                freesound-audio-tagging-2019/
                test/
                    4260ebea.wav (1.0 MB)
                    426eb1e0.wav (654.5 kB)
                    ... and 3359 other files
                    test/
                train_curated/
                    0006ae4e.wav (621.0 kB)
                    0019ef41.wav (181.3 kB)
                    ... and 4968 other files
                train_noisy/
                    00097e21.wav (1.3 MB)
                    000b6cfb.wav (1.3 MB)
                    ... and 19813 other files
            test/
                4260ebea.wav (1.0 MB)
                426eb1e0.wav (654.5 kB)
                ... and 3359 other files
                test/
                    4260ebea.wav (1.0 MB)
                    426eb1e0.wav (654.5 kB)
                    ... and 3359 other files
                    test/
            train_curated/
                0006ae4e.wav (621.0 kB)
                0019ef41.wav (181.3 kB)
                ... and 4968 other files
            train_noisy/
                00097e21.wav (1.3 MB)
                000b6cfb.wav (1.3 MB)
                ... and 19813 other files
        working/
            freesound-audio-tagging-2019/
                description.md (276 lines)
                sample_submission.csv (3362 lines)
                ... and 8 other files
                freesound-audio-tagging-2019/
                test/
                    4260ebea.wav (1.0 MB)
                    426eb1e0.wav (654.5 kB)
                    ... and 3359 other files
                    test/
                train_curated/
                    0006ae4e.wav (621.0 kB)
                    0019ef41.wav (181.3 kB)
                    ... and 4968 other files
                train_noisy/
                    00097e21.wav (1.3 MB)
                    000b6cfb.wav (1.3 MB)
                    ... and 19813 other files
```

-> data/freesound-audio-tagging-2019/sample_submission.csv has 3361 rows and 81 columns.
The columns are: fname, Accelerating_and_revving_and_vroom, Accordion, Acoustic_guitar, Applause, Bark, Bass_drum, Bass_guitar, Bathtub_(filling_or_washing), Bicycle_bell, Burping_and_eructation, Bus, Buzz, Car_passing_by, Cheering... and 66 more columns

-> data/freesound-audio-tagging-2019/train_curated.csv has 4970 rows and 2 columns.
The columns are: fname, labels

-> data/freesound-audio-tagging-2019/train_noisy.csv has 19815 rows and 2 columns.
The columns are: fname, labels

-> data/sample_submission.csv has 3361 rows and 81 columns.
The columns are: fname, Accelerating_and_revving_and_vroom, Accordion, Acoustic_guitar, Applause, Bark, Bass_drum, Bass_guitar, Bathtub_(filling_or_washing), Bicycle_bell, Burping_and_eructation, Bus, Buzz, Car_passing_by, Cheering... and 66 more columns

-> data/train_curated.csv has 4970 rows and 2 columns.
The columns are: fname, labels

-> data/train_noisy.csv has 19815 rows and 2 columns.
The columns are: fname, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.1257248064317963

# 6. Current score

0.19022

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.08412) has done: 'I fix the broken file paths so audio is actually found (your `TRN_CURATED` points to a non-existent folder and test extraction was reading from the wrong place), and I stop extracting the huge `test.zip` by directly reading wavs from the already-unzipped `../input/freesound-audio-tagging-2019/test/`. I also make the code compatible with the installed fastai v2 (your notebook is written for fastai v1, causing `get_transforms/cnn_learner/ImageList` errors), while keeping the same core idea: convert wav→mel→RGB images and train a ResNet18 CNN for multilabel audio tagging. Finally, I ensure a valid `submission.csv` is written with the exact columns/order from `sample_submission.csv`. These changes are required for end-to-end execution and should yield a reasonable score rather than “Not yielded”.'
- What this solution (achieved 0.10886) has done: 'The timeout is dominated by two things: (1) precomputing and compressing mel “images” for *every* train/valid/test file (CPU-heavy `librosa` + `np.savez_compressed`), and (2) a very long training schedule (115 epochs total) plus an expensive full sorting-based lwlrap each epoch. To finish within 600s without changing the model/training logic, we keep the same mel+image pipeline but remove the up-front precompute pass and instead cache lazily on first access using fast uncompressed `.npy` memmaps (provably equivalent data, far less CPU). We also make validation lwlrap computation mathematically identical but much faster by computing it only on the already-computed validation probabilities with a vectorized stable implementation and by avoiding repeated large Python list concatenations overhead. Finally, we tune DataLoader worker/prefetch settings to reduce IPC overhead and make GPU/CPU transfer more efficient while keeping determinism and the same training loop/epochs.'
- What this solution (achieved 0.19022) has done: 'The timeout is dominated by repeated CPU-bound audio→mel conversion inside the DataLoader across 115 epochs; even with caching, each epoch redoes random crops and still loads/decodes many files, and lwlrap computation loops per sample. I (1) precompute and persist the mel→RGB uint8 arrays for train/valid/test once (same functions, same parameters) using parallel workers, then make the Dataset read only cached .npy files during training/inference to eliminate repeated librosa work. I also (2) optimize the lwlrap calculation by removing the per-sample Python loop using vectorized bincount accumulation, preserving identical ranking semantics (stable mergesort) and outputs up to float rounding. Finally, I (3) tune DataLoader settings for cached data (more workers, lower prefetch) and avoid repeated allocations where safe, without changing the model, loss, schedules, or epoch counts.'

# 9. Code solution

## === cell 0
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from tqdm.notebook import tqdm as tqdm_nb
except Exception:
    from tqdm import tqdm as tqdm_nb

import PIL
from PIL import Image as PILImage

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

print("Listing ../input:")
print(os.listdir("../input"))




## === cell 1
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 2
DATA = Path("../input/freesound-audio-tagging-2019")

CSV_TRN_CURATED = DATA / "train_curated.csv"
CSV_TRN_NOISY = DATA / "train_noisy.csv"
CSV_SUBMISSION = DATA / "sample_submission.csv"

TRN_CURATED = DATA / "train_curated"
TRN_NOISY = DATA / "train_noisy"
TEST_DIR = DATA / "test"  # already extracted in this environment per listing

WORK = Path("/kaggle/working")
IMG_TRN_CURATED = WORK / "image/trn_curated"
IMG_TRN_NOISY = WORK / "image/trn_noisy"
IMG_TEST = WORK / "image/test"

for folder in [WORK, IMG_TRN_CURATED, IMG_TRN_NOISY, IMG_TEST]:
    folder.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(CSV_TRN_CURATED)
test_df = pd.read_csv(CSV_SUBMISSION)

print(df.shape, test_df.shape)
print("TRN_CURATED exists:", TRN_CURATED.exists())
print("TEST_DIR exists:", TEST_DIR.exists())



## === cell 3
df = df.reset_index(drop=True)
df.head()



## === cell 4
import librosa
import librosa.display

try:
    import soundfile as sf
except Exception:
    sf = None


class conf:
    sampling_rate = 44100
    duration = 2
    hop_length = 347 * duration  # to make time steps ~128
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20
    samples = sampling_rate * duration


def read_audio(conf, pathname, trim_long_data):
    if sf is not None:
        try:
            y, sr = sf.read(str(pathname), dtype="float32", always_2d=False)
            if y is None or (hasattr(y, "__len__") and len(y) == 0):
                return np.zeros(conf.samples, dtype=np.float32)
            if y.ndim == 2:
                y = y.mean(axis=1, dtype=np.float32)
            if sr != conf.sampling_rate:
                raise RuntimeError("unexpected sample rate")
        except Exception:
            y = None
    else:
        y = None

    if y is None:
        try:
            y, sr = librosa.load(str(pathname), sr=conf.sampling_rate, mono=True)
        except Exception:
            return np.zeros(conf.samples, dtype=np.float32)

        if y is None or len(y) == 0:
            return np.zeros(conf.samples, dtype=np.float32)

    try:
        y, _ = librosa.effects.trim(y)
    except Exception:
        pass

    if len(y) > conf.samples:
        y = y[: conf.samples]
    else:
        padding = conf.samples - len(y)
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), "constant")
    return y.astype(np.float32, copy=False)


def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=conf.sampling_rate,
        n_mels=conf.n_mels,
        hop_length=conf.hop_length,
        n_fft=conf.n_fft,
        fmin=conf.fmin,
        fmax=conf.fmax,
    )
    spectrogram = librosa.power_to_db(spectrogram)
    return spectrogram.astype(np.float32)


def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):
    x = read_audio(conf, pathname, trim_long_data)
    mels = audio_to_melspectrogram(conf, x)
    if debug_display:
        plt.figure(figsize=(10, 3))
        librosa.display.specshow(
            mels,
            x_axis="time",
            y_axis="mel",
            sr=conf.sampling_rate,
            hop_length=conf.hop_length,
            fmin=conf.fmin,
            fmax=conf.fmax,
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title("Log-frequency power spectrogram")
        plt.tight_layout()
        plt.show()
    return mels


example_path = TRN_CURATED / df.loc[0, "fname"]
print("Example wav exists:", example_path.exists(), example_path)




## === cell 5
def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    mean = X.mean() if mean is None else mean
    std = X.std() if std is None else std
    Xstd = (X - mean) / (std + eps)

    _min, _max = Xstd.min(), Xstd.max()
    norm_max = _max if norm_max is None else norm_max
    norm_min = _min if norm_min is None else norm_min

    if (norm_max - norm_min) > eps:
        V = np.clip(Xstd, norm_min, norm_max)
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(Xstd, dtype=np.uint8)

    return np.repeat(V[:, :, None], 3, axis=2)


class _MelImageCache:
    def __init__(self, max_items=512):
        self.max_items = int(max_items)
        self._data = {}
        self._order = []

    def get(self, key):
        return self._data.get(key, None)

    def put(self, key, value):
        if key in self._data:
            return
        self._data[key] = value
        self._order.append(key)
        if len(self._order) > self.max_items:
            old = self._order.pop(0)
            self._data.pop(old, None)


print("Feature extraction helpers ready.")



## === cell 6
label_cols = [c for c in test_df.columns if c != "fname"]
num_classes = len(label_cols)
label_to_idx = {l: i for i, l in enumerate(label_cols)}


def labels_to_multihot(label_str):
    y = np.zeros(num_classes, dtype=np.float32)
    for lab in str(label_str).split(","):
        lab = lab.strip()
        if lab in label_to_idx:
            y[label_to_idx[lab]] = 1.0
    return y


Y_train = np.stack([labels_to_multihot(s) for s in df["labels"].values], axis=0)
print("Y_train:", Y_train.shape, "classes:", num_classes)



## === cell 7
import hashlib
import concurrent.futures as cf

FEATURE_CACHE_DIR = WORK / "melcache_v2_npy"
FEATURE_CACHE_DIR.mkdir(exist_ok=True, parents=True)

_CONF_KEY = (
    f"sr{conf.sampling_rate}_dur{conf.duration}_hop{conf.hop_length}_fmin{conf.fmin}_fmax{conf.fmax}"
    f"_nmels{conf.n_mels}_nfft{conf.n_fft}_samples{conf.samples}"
)
_CONF_HASH = hashlib.md5(_CONF_KEY.encode("utf-8")).hexdigest()[:10]


def _cache_path_for(source_dir: Path, fname: str) -> Path:
    src_tag = source_dir.name  # train_curated / train_noisy / test
    stem = Path(fname).stem
    return FEATURE_CACHE_DIR / f"{src_tag}__{stem}__{_CONF_HASH}.npy"


def _load_cached_img(npy_path: Path):
    try:
        return np.load(npy_path, mmap_mode="r")
    except Exception:
        return None


def _save_cached_img(npy_path: Path, img_uint8: np.ndarray):
    tmp_path = npy_path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        np.save(f, img_uint8, allow_pickle=False)
    os.replace(str(tmp_path), str(npy_path))


def _build_cache_for_files(source_dir: Path, fnames, desc: str, max_workers: int):
    source_dir = Path(source_dir)
    fnames = list(map(str, fnames))

    def _one(fname: str):
        npy_path = _cache_path_for(source_dir, fname)
        if npy_path.exists():
            return 0
        wav_path = source_dir / fname
        mels = read_as_melspectrogram(
            conf, wav_path, trim_long_data=False, debug_display=False
        )
        img = mono_to_color(mels)
        try:
            _save_cached_img(npy_path, img)
            return 1
        except Exception:
            return 0

    wrote = 0
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for r in tqdm_nb(ex.map(_one, fnames), total=len(fnames), desc=desc):
            wrote += int(r)
    return wrote


class MelImageDataset(Dataset):
    def __init__(
        self,
        fnames,
        source_dir,
        Y=None,
        train=True,
        cache_max_items=512,
        require_cache=False,
    ):
        self.fnames = list(map(str, fnames))
        self.source_dir = Path(source_dir)
        self.Y = Y
        self.train = train
        self.cache = _MelImageCache(max_items=cache_max_items)
        self.require_cache = bool(require_cache)

    def __len__(self):
        return len(self.fnames)

    def _crop_square(self, img_arr):
        h, w, _ = img_arr.shape
        if w == h:
            return img_arr
        if w > h:
            if self.train:
                left = random.randint(0, w - h)
            else:
                left = (w - h) // 2
            return img_arr[:, left : left + h, :]
        pad = h - w
        left_pad = pad // 2
        right_pad = pad - left_pad
        return np.pad(img_arr, ((0, 0), (left_pad, right_pad), (0, 0)), mode="constant")

    def _load_image_array(self, fname):
        cached = self.cache.get(fname)
        if cached is not None:
            return cached

        npy_path = _cache_path_for(self.source_dir, fname)
        if npy_path.exists():
            img = _load_cached_img(npy_path)
            if img is not None:
                self.cache.put(fname, img)
                return img

        if self.require_cache:
            raise FileNotFoundError(f"Missing cached feature: {npy_path}")

        wav_path = self.source_dir / fname
        mels = read_as_melspectrogram(
            conf, wav_path, trim_long_data=False, debug_display=False
        )
        img = mono_to_color(mels)
        try:
            _save_cached_img(npy_path, img)
        except Exception:
            pass
        self.cache.put(fname, img)
        return img

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = self._load_image_array(fname)

        img = self._crop_square(img)
        if not isinstance(img, np.ndarray):
            img = np.asarray(img)
        img = np.ascontiguousarray(img)

        x = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
        if self.Y is None:
            return x
        y = torch.from_numpy(self.Y[idx])
        return x, y


n = len(df)
indices = np.arange(n)
rng = np.random.RandomState(42)
rng.shuffle(indices)
valid_size = int(0.2 * n)
valid_idx = indices[:valid_size]
train_idx = indices[valid_size:]

cpu_cnt = os.cpu_count() or 2
cache_workers = min(8, max(2, cpu_cnt))  # threads are OK; sf/librosa release GIL often

w1 = _build_cache_for_files(
    TRN_CURATED,
    df.loc[train_idx, "fname"].values,
    "Caching train_curated (train)",
    cache_workers,
)
w2 = _build_cache_for_files(
    TRN_CURATED,
    df.loc[valid_idx, "fname"].values,
    "Caching train_curated (valid)",
    cache_workers,
)
w3 = _build_cache_for_files(
    TEST_DIR, test_df["fname"].values, "Caching test", cache_workers
)
print(
    f"Cached new files: train={w1}, valid={w2}, test={w3} (cache dir: {FEATURE_CACHE_DIR})"
)

bs = 64
use_cuda = torch.cuda.is_available()


def _seed_worker(worker_id):
    base_seed = 42
    s = base_seed + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


g = torch.Generator()
g.manual_seed(42)

num_workers = min(8, max(2, cpu_cnt // 2)) if cpu_cnt > 2 else 0

train_ds = MelImageDataset(
    df.loc[train_idx, "fname"].values,
    source_dir=TRN_CURATED,
    Y=Y_train[train_idx],
    train=True,
    cache_max_items=4096,
    require_cache=True,
)
valid_ds = MelImageDataset(
    df.loc[valid_idx, "fname"].values,
    source_dir=TRN_CURATED,
    Y=Y_train[valid_idx],
    train=False,
    cache_max_items=4096,
    require_cache=True,
)
test_ds = MelImageDataset(
    test_df["fname"].values,
    source_dir=TEST_DIR,
    Y=None,
    train=False,
    cache_max_items=4096,
    require_cache=True,
)

train_dl = DataLoader(
    train_ds,
    batch_size=bs,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=use_cuda,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker,
    generator=g,
)
valid_dl = DataLoader(
    valid_ds,
    batch_size=bs,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=use_cuda,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker,
)
test_dl = DataLoader(
    test_ds,
    batch_size=bs,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=use_cuda,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker,
)

len(train_ds), len(valid_ds), len(test_ds)




## === cell 8
def calculate_per_class_lwlrap(truth, scores):
    truth = (truth > 0).astype(np.bool_)
    num_samples, num_classes = scores.shape

    order = np.argsort(-scores, axis=1, kind="mergesort")
    truth_sorted = np.take_along_axis(truth, order, axis=1)

    hits = np.cumsum(truth_sorted, axis=1, dtype=np.int32)
    ranks = (np.arange(num_classes, dtype=np.float64) + 1.0)[None, :]
    precisions = hits / ranks

    labels_per_class = truth.sum(axis=0).astype(np.float64)
    sum_precisions_per_class = np.zeros(num_classes, dtype=np.float64)

    pos_mask = truth_sorted
    if np.any(pos_mask):
        cls_ids = order[pos_mask]
        w = precisions[pos_mask]
        sum_precisions_per_class += np.bincount(
            cls_ids, weights=w, minlength=num_classes
        )

    per_class_lwlrap = sum_precisions_per_class / np.maximum(1.0, labels_per_class)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class) + 1e-12)
    return per_class_lwlrap, weight_per_class


def lwlrap_overall(truth, scores):
    per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(truth, scores)
    return float((per_class_lwlrap * weight_per_class).sum())




## === cell 9
from torchvision import models

model = models.resnet18(weights=None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(device)

if torch.cuda.is_available():
    model = model.to(memory_format=torch.channels_last)

criterion = nn.BCEWithLogitsLoss()


def run_block(epochs, lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if torch.cuda.is_available():
                xb = xb.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * xb.size(0)

        scheduler.step()

        model.eval()
        va_loss = 0.0
        all_scores = np.empty((len(valid_ds), num_classes), dtype=np.float32)
        all_truth = np.empty((len(valid_ds), num_classes), dtype=np.float32)
        ofs = 0
        with torch.no_grad():
            for xb, yb in valid_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                if torch.cuda.is_available():
                    xb = xb.contiguous(memory_format=torch.channels_last)

                logits = model(xb)
                loss = criterion(logits, yb)
                va_loss += loss.item() * xb.size(0)

                probs = (
                    torch.sigmoid(logits)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                y_np = yb.detach().cpu().numpy().astype(np.float32, copy=False)
                bs_cur = probs.shape[0]
                all_scores[ofs : ofs + bs_cur] = probs
                all_truth[ofs : ofs + bs_cur] = y_np
                ofs += bs_cur

        va_lwlrap = lwlrap_overall(all_truth, all_scores)

        print(
            f"block lr={lr:g} ep {ep+1}/{epochs} "
            f"train_loss={tr_loss/len(train_ds):.4f} "
            f"valid_loss={va_loss/len(valid_ds):.4f} "
            f"valid_lwlrap={va_lwlrap:.4f}"
        )




## === cell 10
run_block(epochs=5, lr=1e-1)
run_block(epochs=10, lr=1e-2)
run_block(epochs=20, lr=3e-3)
run_block(epochs=20, lr=1e-3)
run_block(epochs=50, lr=3e-3)  # keep schedule identical to provided script
run_block(epochs=10, lr=1e-3)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/582230506.py in <cell line: 0>()
----> 1 run_block(epochs=5, lr=1e-1)
      2 run_block(epochs=10, lr=1e-2)
      3 run_block(epochs=20, lr=3e-3)
      4 run_block(epochs=20, lr=1e-3)
      5 run_block(epochs=50, lr=3e-3)  # keep schedule identical to provided script

/tmp/ipykernel_55/907680187.py in run_block(epochs, lr)
     19         model.train()
     20         tr_loss = 0.0
---> 21         for xb, yb in train_dl:
     22             xb = xb.to(device, non_blocking=True)
     23             yb = yb.to(device, non_blocking=True)

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

FileNotFoundError: Caught FileNotFoundError in DataLoader worker process 5.
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
  File "/tmp/ipykernel_55/4174426617.py", line 130, in __getitem__
    img = self._load_image_array(fname)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/4174426617.py", line 114, in _load_image_array
    raise FileNotFoundError(f"Missing cached feature: {npy_path}")
FileNotFoundError: Missing cached feature: /kaggle/working/melcache_v2_npy/train_curated__ede8d21c__d79d4ecc84.npy


## === cell 11
model.eval()
all_test_scores = np.empty((len(test_ds), num_classes), dtype=np.float32)
ofs = 0
with torch.no_grad():
    for xb in test_dl:
        xb = xb.to(device, non_blocking=True)
        if torch.cuda.is_available():
            xb = xb.contiguous(memory_format=torch.channels_last)
        logits = model(xb)
        probs = (
            torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32, copy=False)
        )
        bs_cur = probs.shape[0]
        all_test_scores[ofs : ofs + bs_cur] = probs
        ofs += bs_cur
all_test_scores.shape



## === cell 12
sub = pd.read_csv(CSV_SUBMISSION)
sub[label_cols] = all_test_scores.astype(np.float32)

out_path = WORK / "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", sub.shape)
sub.head()



## === cell 13
print("Done.")
