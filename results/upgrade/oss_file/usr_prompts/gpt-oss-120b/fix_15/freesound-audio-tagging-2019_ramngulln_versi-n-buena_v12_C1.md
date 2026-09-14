# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os, zipfile, numpy as np, pandas as pd, librosa, librosa.display
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import label_ranking_average_precision_score
import concurrent.futures  # will be used for both Thread and Process pools

os.environ["OMP_NUM_THREADS"] = "1"  # avoid thread oversubscription in librosa




## === cell 1
BASE = Path("../input")
DATA_ROOT = BASE / "freesound-audio-tagging-2019"

ZIP_TRAIN_CURATED = DATA_ROOT / "train_curated.zip"
ZIP_TRAIN_NOISY = DATA_ROOT / "train_noisy.zip"
ZIP_TEST = DATA_ROOT / "test.zip"

CSV_TRAIN_CURATED = DATA_ROOT / "train_curated.csv"
CSV_TRAIN_NOISY = DATA_ROOT / "train_noisy.csv"
CSV_SUBMISSION = DATA_ROOT / "sample_submission.csv"

if not (DATA_ROOT / "train_curated").exists():
    with zipfile.ZipFile(ZIP_TRAIN_CURATED, "r") as z:
        z.extractall(path=DATA_ROOT)
if not (DATA_ROOT / "train_noisy").exists():
    with zipfile.ZipFile(ZIP_TRAIN_NOISY, "r") as z:
        z.extractall(path=DATA_ROOT)
if not (DATA_ROOT / "test").exists():
    with zipfile.ZipFile(ZIP_TEST, "r") as z:
        z.extractall(path=DATA_ROOT)

df_curated = pd.read_csv(CSV_TRAIN_CURATED)
df_noisy = pd.read_csv(CSV_TRAIN_NOISY)
df_train = pd.concat([df_curated, df_noisy], ignore_index=True)

test_df = pd.read_csv(CSV_SUBMISSION)




## === cell 2
class Conf:
    sampling_rate = 44100
    duration = 2  # seconds
    hop_length = 347 * duration  # approx 128 time steps
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20
    samples = sampling_rate * duration


def read_audio(conf, pathname):
    """
    Load exactly `conf.duration` seconds of audio at the target sampling rate.
    This yields the same length as the original pipeline after trim/pad,
    but avoids reading the whole file and the costly `librosa.effects.trim`.
    """
    y, sr = librosa.load(
        pathname,
        sr=conf.sampling_rate,
        mono=True,
        duration=conf.duration,  # load only the required portion
    )
    if len(y) < conf.samples:
        pad = conf.samples - len(y)
        offset = pad // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), "constant")
    return y


def audio_to_melspectrogram(conf, audio):
    S = librosa.feature.melspectrogram(
        y=audio,
        sr=conf.sampling_rate,
        n_mels=conf.n_mels,
        hop_length=conf.hop_length,
        n_fft=conf.n_fft,
        fmin=conf.fmin,
        fmax=conf.fmax,
    )
    S_db = librosa.power_to_db(S)
    return S_db.astype(np.float32)


def wav_to_feature(path):
    wav = read_audio(Conf, path)
    mel = audio_to_melspectrogram(Conf, wav)
    return mel.flatten().astype(np.float32)


def process_train_row(args):
    fp, label_str = args
    if not fp.is_file():
        return None
    feature = wav_to_feature(fp)
    labels = label_str.split(",")
    return feature, labels


train_folder_curated = DATA_ROOT / "train_curated"
train_folder_noisy = DATA_ROOT / "train_noisy"

print("Extracting features for training set (curated + noisy) in parallel...")
tasks = []
for _, row in df_train.iterrows():
    fp = train_folder_curated / row.fname
    if not fp.is_file():
        fp = train_folder_noisy / row.fname
    tasks.append((fp, row.labels))

sample_feat = None
for fp, _ in tasks:
    if fp.is_file():
        sample_feat = wav_to_feature(fp)
        break
if sample_feat is None:
    raise RuntimeError("No audio files found for feature size detection.")
feat_dim = sample_feat.size

num_samples = len(tasks)
X = np.empty((num_samples, feat_dim), dtype=np.float32)
labels = []  # will be a list of label lists
max_workers = max(1, (os.cpu_count() or 1) - 1)

with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    for idx, result in enumerate(
        tqdm(
            executor.map(process_train_row, tasks, chunksize=500),
            total=len(tasks),
        )
    ):
        if result is not None:
            feat, lbl = result
            X[idx] = feat  # already float32
            labels.append(lbl)
        else:
            X[idx] = np.zeros(feat_dim, dtype=np.float32)
            labels.append([])

if X.shape[0] == 0:
    raise RuntimeError("No training features were extracted – check data paths.")

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(labels)  # keep default int (0/1) dtype




## === cell 3
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
base_clf = OneVsRestClassifier(LogisticRegression(max_iter=200, n_jobs=5))
base_clf.fit(X_tr, y_tr)
val_probs = base_clf.predict_proba(X_val)
val_lwlrap = label_ranking_average_precision_score(y_val, val_probs)
print(f"Validation LWLRAP: {val_lwlrap:.6f}")




## === cell 4
clf = OneVsRestClassifier(LogisticRegression(max_iter=300, n_jobs=5))
clf.fit(X, y)




## === cell 5
test_folder = DATA_ROOT / "test"


def process_test_fname(fname):
    fp = test_folder / fname
    if not fp.is_file():
        return np.zeros(X.shape[1], dtype=np.float32)
    feat = wav_to_feature(fp)
    return feat.astype(np.float32)


print("Extracting features for test set in parallel...")
num_test = len(test_df)
X_test = np.empty((num_test, X.shape[1]), dtype=np.float32)

with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    for idx, feat in enumerate(
        tqdm(
            executor.map(process_test_fname, test_df.fname, chunksize=500),
            total=num_test,
        )
    ):
        X_test[idx] = feat

test_probs = clf.predict_proba(X_test)




## === cell 6
submission = pd.read_csv(CSV_SUBMISSION)  # header with correct column order
prob_cols = submission.columns[1:]  # all label columns
submission[prob_cols] = test_probs
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv with shape:", submission.shape)
