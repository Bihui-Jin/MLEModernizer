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

3.7

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0

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
import os
import numpy as np
import pandas as pd
import wave
from sklearn import ensemble, metrics, multioutput
from scipy.io import wavfile
import gc
import multiprocessing as mp

gc.enable()

BASE = "/kaggle/data/freesound-audio-tagging-2019"

train = pd.read_csv(os.path.join(BASE, "train_curated.csv"))
trainn = pd.read_csv(os.path.join(BASE, "train_noisy.csv"))
sample_sub = pd.read_csv(os.path.join(BASE, "sample_submission.csv"))

train.shape, trainn.shape, sample_sub.shape



## === cell 1
train["path"] = train["fname"].map(lambda x: os.path.join(BASE, "train_curated", x))
trainn["path"] = trainn["fname"].map(lambda x: os.path.join(BASE, "train_noisy", x))
test = sample_sub[["fname"]].copy()
test["path"] = test["fname"].map(lambda x: os.path.join(BASE, "test", x))

train["noisy"] = 0
trainn["noisy"] = 1
train = pd.concat((train, trainn), sort=False).reset_index(drop=True)

labels = [c for c in sample_sub.columns if c != "fname"]

train["labels_list"] = train["labels"].astype(str).str.split(",")
train = train[
    train["labels_list"].map(lambda ls: any(l in set(labels) for l in ls))
].reset_index(drop=True)

test = test[["path", "fname"]]
train.shape, test.shape



## === cell 2
norm_labels = []
cut = 100
train["primary_label"] = train["labels_list"].map(
    lambda x: x[0] if isinstance(x, list) and len(x) else ""
)
for l in train["primary_label"].unique():
    norm_labels.append(train[train.primary_label == l][:cut])
train = (
    pd.concat(norm_labels, sort=False)
    .sample(frac=1, random_state=0)
    .reset_index(drop=True)
)
train.shape



## === cell 3
waves = """3a5b14ee.wav 404712 423984
7a9cf335.wav 501072 520344
c421d4a2.wav 289080 308352
aa28de21.wav 231264 250536
703ac398.wav 19272 38544
3cbb9c24.wav 57813 77084
7c20368d.wav 616672 635943
c6cb06d9.wav 481775 501046
7f0af3bb.wav 481775 501046
76caa793.wav 385420 404691
767b8f3a.wav 635943 655214
a98c3157.wav 231252 250523
8ddb4c26.wav 0 19271
3e1d0af4.wav 635943 655214
aca0ce49.wav 578130 597401""".split(
    "\n"
)




## === cell 4
def get_nframes_and_short_wave(path, size=1200):
    try:
        with wave.open(path, "rb") as wf:
            nframes = wf.getnframes()
            nch = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            nread = min(size, nframes)
            raw = wf.readframes(nread)

        if nread <= 0 or sampwidth != 2:
            sw = np.zeros(size, dtype=np.int16)
            return int(nframes), sw

        data = np.frombuffer(raw, dtype="<i2")
        if nch > 1:
            data = data.reshape(-1, nch)[:, 0]
        if nframes > size:
            sw = data[:size]
        else:
            rep = int((size / nframes) + 1) if nframes > 0 else 1
            sw = (
                np.repeat(data, rep)[:size]
                if nframes > 0
                else np.zeros(size, dtype=np.int16)
            )

        return int(nframes), sw
    except Exception:
        return 0, np.zeros(size, dtype=np.int16)




## === cell 5
def _worker_get_nf_sw(args):
    p, size = args
    return get_nframes_and_short_wave(p, size=size)


def add_audio_columns(df, size=1200, n_jobs=None, chunksize=64):
    paths = df["path"].values
    n = len(paths)
    nframes = np.empty(n, dtype=np.int64)
    short_waves = np.empty(n, dtype=object)

    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count())

    with mp.get_context("fork").Pool(processes=n_jobs) as pool:
        for i, (nf, sw) in enumerate(
            pool.imap(
                _worker_get_nf_sw, ((p, size) for p in paths), chunksize=chunksize
            )
        ):
            nframes[i] = nf
            short_waves[i] = sw

    df["nframes"] = nframes
    df["short_wave"] = short_waves
    return df


train = add_audio_columns(train, size=1200)
test = add_audio_columns(test, size=1200)




## === cell 6
def _skew_kurtosis_bias_false(A):
    A = A.astype(np.float64, copy=False)
    n = A.shape[1]
    if n < 4:
        skew = np.full(A.shape[0], np.nan, dtype=np.float64)
        kurt = np.full(A.shape[0], np.nan, dtype=np.float64)
        return skew, kurt

    mean = A.mean(axis=1)
    xc = A - mean[:, None]

    s2 = (xc * xc).sum(axis=1) / (n - 1)
    s = np.sqrt(s2)

    with np.errstate(divide="ignore", invalid="ignore"):
        m3 = (xc**3).mean(axis=1)
        m4 = (xc**4).mean(axis=1)

        g1 = m3 / (s**3)  # moment skewness using sample std in denom
        G1 = np.sqrt(n * (n - 1)) / (n - 2) * g1  # bias=False correction

        g2 = m4 / (s**4) - 3.0  # excess kurtosis using sample std in denom
        G2 = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6.0)

    return G1, G2


def features(df, col="short_wave"):
    X = np.stack(df[col].values).astype(np.float64, copy=False)
    Xabs = np.abs(X)

    def add_stats(prefix, A):
        df[prefix + "min"] = A.min(axis=1)
        df[prefix + "max"] = A.max(axis=1)
        df[prefix + "sum"] = A.sum(axis=1)
        df[prefix + "median"] = np.median(A, axis=1)
        df[prefix + "mean"] = A.mean(axis=1)
        df[prefix + "std"] = A.std(axis=1, ddof=1)
        skew, kurt = _skew_kurtosis_bias_false(A)
        df[prefix + "skew"] = skew
        df[prefix + "kurtosis"] = kurt

    add_stats(col, X)
    add_stats(col + "a", Xabs)

    df[col + "max_diff"] = df[col + "max"] - df[col + "mean"]
    df[col + "amax_diff"] = df[col + "amax"] - df[col + "amean"]

    df[col + "min_diff"] = df[col + "mean"] - df[col + "min"]
    df[col + "amin_diff"] = df[col + "amean"] - df[col + "amin"]

    df[col + "max_diff2"] = df[col + "max"] - df[col + "median"]
    df[col + "amax_diff2"] = df[col + "amax"] - df[col + "amedian"]

    df[col + "min_diff2"] = df[col + "median"] - df[col + "min"]
    df[col + "amin_diff2"] = df[col + "amedian"] - df[col + "amin"]
    return df


train = features(train).fillna(-999)
test = features(test).fillna(-999)
print(train.shape, test.shape)



## === cell 7
col = [
    c
    for c in train.columns
    if c
    not in [
        "path",
        "fname",
        "noisy",
        "labels",
        "labels_list",
        "primary_label",
        "short_wave",
    ]
]

Y = np.zeros((len(train), len(labels)), dtype=np.int8)
label_index = {l: i for i, l in enumerate(labels)}
for i, ls in enumerate(train["labels_list"].values):
    for l in ls:
        j = label_index.get(l, None)
        if j is not None:
            Y[i, j] = 1

base1 = ensemble.ExtraTreesClassifier(
    n_jobs=-1, n_estimators=400, max_features=0.9, random_state=10
)
base2 = ensemble.RandomForestClassifier(
    n_jobs=-1, n_estimators=400, max_features=0.9, random_state=9
)

clf1 = multioutput.MultiOutputClassifier(base1, n_jobs=-1)
clf2 = multioutput.MultiOutputClassifier(base2, n_jobs=-1)

split = min(3000, len(train) - 1)

X_all = train[col].to_numpy(dtype=np.float64, copy=False)
X_test = test[col].to_numpy(dtype=np.float64, copy=False)

clf1.fit(X_all[:split], Y[:split])
clf2.fit(X_all[:split], Y[:split])


def LW_LRAP(y_true, y_score):
    return metrics.label_ranking_average_precision_score(y_true, y_score)


def predict_pos_proba(m, X):
    probs = m.predict_proba(X)
    out = np.zeros((X.shape[0], len(probs)), dtype=np.float32)
    for k, pk in enumerate(probs):
        if pk.shape[1] == 2:
            out[:, k] = pk[:, 1]
        else:
            out[:, k] = 0.0
    return out


val_true = Y[split:]
val_pred1 = predict_pos_proba(clf1, X_all[split:])
val_pred2 = predict_pos_proba(clf2, X_all[split:])

print("ETR LW_LRAP", LW_LRAP(val_true, val_pred1))
print("RFR LW_LRAP", LW_LRAP(val_true, val_pred2))

clf1.fit(X_all, Y)
clf2.fit(X_all, Y)

sub1 = predict_pos_proba(clf1, X_test)
sub2 = predict_pos_proba(clf2, X_test)
sub = (sub1 + sub2) / 2.0

sub_df = pd.DataFrame(sub, columns=labels)
sub_df.insert(0, "fname", test["fname"].values)

sub_df = sub_df[["fname"] + labels]
sub_df.to_csv("submission.csv", index=False)
print(sub_df.shape, sub_df.columns[:5].tolist(), "-> wrote submission.csv")
