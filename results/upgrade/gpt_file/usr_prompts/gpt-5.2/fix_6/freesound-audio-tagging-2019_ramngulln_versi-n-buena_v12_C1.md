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

0.10886

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.08412) has done: 'I fix the broken file paths so audio is actually found (your `TRN_CURATED` points to a non-existent folder and test extraction was reading from the wrong place), and I stop extracting the huge `test.zip` by directly reading wavs from the already-unzipped `../input/freesound-audio-tagging-2019/test/`. I also make the code compatible with the installed fastai v2 (your notebook is written for fastai v1, causing `get_transforms/cnn_learner/ImageList` errors), while keeping the same core idea: convert wav→mel→RGB images and train a ResNet18 CNN for multilabel audio tagging. Finally, I ensure a valid `submission.csv` is written with the exact columns/order from `sample_submission.csv`. These changes are required for end-to-end execution and should yield a reasonable score rather than “Not yielded”.'
- What this solution (achieved 0.10886) has done: 'The timeout is dominated by two things: (1) precomputing and compressing mel “images” for *every* train/valid/test file (CPU-heavy `librosa` + `np.savez_compressed`), and (2) a very long training schedule (115 epochs total) plus an expensive full sorting-based lwlrap each epoch. To finish within 600s without changing the model/training logic, we keep the same mel+image pipeline but remove the up-front precompute pass and instead cache lazily on first access using fast uncompressed `.npy` memmaps (provably equivalent data, far less CPU). We also make validation lwlrap computation mathematically identical but much faster by computing it only on the already-computed validation probabilities with a vectorized stable implementation and by avoiding repeated large Python list concatenations overhead. Finally, we tune DataLoader worker/prefetch settings to reduce IPC overhead and make GPU/CPU transfer more efficient while keeping determinism and the same training loop/epochs.'

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
    y, sr = librosa.load(str(pathname), sr=conf.sampling_rate, mono=True)
    if 0 < len(y):
        y, _ = librosa.effects.trim(y)
    if len(y) > conf.samples:
        if trim_long_data:
            y = y[: conf.samples]
        else:
            y = y[: conf.samples]
    else:
        padding = conf.samples - len(y)
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), "constant")
    return y


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
    X = np.stack([X, X, X], axis=-1)

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
    return V


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
    tmp = npy_path.with_suffix(".tmp.npy")
    np.save(tmp, img_uint8, allow_pickle=False)
    os.replace(tmp, npy_path)


class MelImageDataset(Dataset):
    def __init__(self, fnames, source_dir, Y=None, train=True, cache_max_items=512):
        self.fnames = list(map(str, fnames))
        self.source_dir = Path(source_dir)
        self.Y = Y
        self.train = train
        self.cache = _MelImageCache(max_items=cache_max_items)

    def __len__(self):
        return len(self.fnames)

    def _crop_square(self, img_arr):
        h, w, _ = img_arr.shape
        if w == h:
            cropped = img_arr
        elif w > h:
            if self.train:
                left = random.randint(0, w - h)
            else:
                left = (w - h) // 2
            cropped = img_arr[:, left : left + h, :]
        else:
            pad = h - w
            left_pad = pad // 2
            right_pad = pad - left_pad
            cropped = np.pad(
                img_arr, ((0, 0), (left_pad, right_pad), (0, 0)), mode="constant"
            )
        return cropped

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
        img = self._crop_square(np.asarray(img))  # ensure ndarray if mmap
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

train_ds = MelImageDataset(
    df.loc[train_idx, "fname"].values,
    source_dir=TRN_CURATED,
    Y=Y_train[train_idx],
    train=True,
    cache_max_items=1024,
)
valid_ds = MelImageDataset(
    df.loc[valid_idx, "fname"].values,
    source_dir=TRN_CURATED,
    Y=Y_train[valid_idx],
    train=False,
    cache_max_items=1024,
)
test_ds = MelImageDataset(
    test_df["fname"].values,
    source_dir=TEST_DIR,
    Y=None,
    train=False,
    cache_max_items=1024,
)

bs = 64
use_cuda = torch.cuda.is_available()

num_workers = 2 if (os.cpu_count() or 2) >= 4 else 1

train_dl = DataLoader(
    train_ds,
    batch_size=bs,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=use_cuda,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)
valid_dl = DataLoader(
    valid_ds,
    batch_size=bs,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=use_cuda,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)
test_dl = DataLoader(
    test_ds,
    batch_size=bs,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=use_cuda,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)

len(train_ds), len(valid_ds), len(test_ds)




## === cell 8
def calculate_per_class_lwlrap(truth, scores):
    truth = (truth > 0).astype(np.bool_)
    num_samples, num_classes = scores.shape

    order = np.argsort(-scores, axis=1, kind="mergesort")
    truth_sorted = np.take_along_axis(truth, order, axis=1)

    hits = np.cumsum(truth_sorted, axis=1)
    ranks = (np.arange(num_classes, dtype=np.float64) + 1.0)[None, :]
    precisions = hits / ranks

    labels_per_class = truth.sum(axis=0).astype(np.float64)
    sum_precisions_per_class = np.zeros(num_classes, dtype=np.float64)

    for i in range(num_samples):
        pos = truth_sorted[i]
        if not np.any(pos):
            continue
        cls_ids = order[i, pos]
        sum_precisions_per_class[cls_ids] += precisions[i, pos]

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
run_block(
    epochs=50, lr=3e-3
)  # original used slice(1e-3,3e-3); keep a single lr to stay minimal
run_block(epochs=10, lr=1e-3)  # original used slice(1e-4,1e-3); keep single lr



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_get_data(self, timeout)
   1250         try:
-> 1251             data = self._data_queue.get(timeout=timeout)
   1252             return (True, data)

/usr/lib/python3.11/queue.py in get(self, block, timeout)
    179                         raise Empty
--> 180                     self.not_empty.wait(remaining)
    181             item = self._get()

/usr/lib/python3.11/threading.py in wait(self, timeout)
    330                 if timeout > 0:
--> 331                     gotit = waiter.acquire(True, timeout)
    332                 else:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/signal_handling.py in handler(signum, frame)
     72         # Python can still get and update the process status successfully.
---> 73         _error_if_any_worker_fails()
     74         if previous_handler is not None:

RuntimeError: DataLoader worker (pid 96) is killed by signal: Aborted. 

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3126339880.py in <cell line: 0>()
----> 1 run_block(epochs=5, lr=1e-1)
      2 run_block(epochs=10, lr=1e-2)
      3 run_block(epochs=20, lr=3e-3)
      4 run_block(epochs=20, lr=1e-3)
      5 run_block(

/tmp/ipykernel_55/810315234.py in run_block(epochs, lr)
     16         model.train()
     17         tr_loss = 0.0
---> 18         for xb, yb in train_dl:
     19             xb = xb.to(device, non_blocking=True)
     20             yb = yb.to(device, non_blocking=True)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1456 
   1457             assert not self._shutdown and self._tasks_outstanding > 0
-> 1458             idx, data = self._get_data()
   1459             self._tasks_outstanding -= 1
   1460             if self._dataset_kind == _DatasetKind.Iterable:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _get_data(self)
   1408         elif self._pin_memory:
   1409             while self._pin_memory_thread.is_alive():
-> 1410                 success, data = self._try_get_data()
   1411                 if success:
   1412                     return data

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_get_data(self, timeout)
   1262             if len(failed_workers) > 0:
   1263                 pids_str = ", ".join(str(w.pid) for w in failed_workers)
-> 1264                 raise RuntimeError(
   1265                     f"DataLoader worker (pid(s) {pids_str}) exited unexpectedly"
   1266                 ) from e

RuntimeError: DataLoader worker (pid(s) 96) exited unexpectedly

## === cell 11
model.eval()
all_test_scores = np.empty((len(test_ds), num_classes), dtype=np.float32)
ofs = 0
with torch.no_grad():
    for xb in test_dl:
        xb = xb.to(device, non_blocking=True)
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
