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

# 5. Target score

0.11683

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, wave, IPython
import numpy as np, pandas as pd
from scipy.io import wavfile
from sklearn import preprocessing, ensemble, metrics
import gc

gc.enable()

BASE = os.path.abspath(os.path.join(".", "data", "freesound-audio-tagging-2019"))
train = pd.read_csv(os.path.join(BASE, "train_curated.csv"))
train_noisy = pd.read_csv(os.path.join(BASE, "train_noisy.csv"))  # use full noisy set

sample_sub = pd.read_csv(os.path.join(BASE, "sample_submission.csv"))
label_cols = [c for c in sample_sub.columns if c != "fname"]

print("Rows:", train.shape[0], train_noisy.shape[0], "Labels:", len(label_cols))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4160462335.py in <cell line: 0>()
     10 BASE = os.path.abspath(os.path.join(".", "data", "freesound-audio-tagging-2019"))
     11 # Load the curated and full noisy training CSVs
---> 12 train = pd.read_csv(os.path.join(BASE, "train_curated.csv"))
     13 train_noisy = pd.read_csv(os.path.join(BASE, "train_noisy.csv"))  # use full noisy set
     14 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/data/freesound-audio-tagging-2019/train_curated.csv'

## === cell 1
train["path"] = train["fname"].map(lambda x: os.path.join(BASE, "train_curated", x))
train_noisy["path"] = train_noisy["fname"].map(
    lambda x: os.path.join(BASE, "train_noisy", x)
)

test = pd.read_csv(os.path.join(BASE, "sample_submission.csv"))
test["path"] = test["fname"].map(lambda x: os.path.join(BASE, "test", x))

train["noisy"] = 0
train_noisy["noisy"] = 1
train = pd.concat((train, train_noisy), sort=False).reset_index(drop=True)

train["labels"] = train["labels"].str.split(",")
train = train.explode("labels").reset_index(drop=True)
train["labels"] = train["labels"].str.strip()

train = train[train["labels"].isin(label_cols)].reset_index(drop=True)
test = test[["path", "fname"]]
print("After cleaning:", train.shape, test.shape)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4174842187.py in <cell line: 0>()
      1 # Build full paths to the audio files
----> 2 train["path"] = train["fname"].map(lambda x: os.path.join(BASE, "train_curated", x))
      3 train_noisy["path"] = train_noisy["fname"].map(
      4     lambda x: os.path.join(BASE, "train_noisy", x)
      5 )

NameError: name 'train' is not defined

## === cell 2
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

wavesc = []
for w in waves:
    w1, c1, c2 = w.split(" ")
    c1, c2 = int(c1), int(c2)
    _, data = wavfile.read(os.path.join(BASE, "train_noisy", w1))
    wavesc.append(data[c1:c2])

wavesc = np.concatenate(wavesc)
wavfile.write("one_step.wav", 44100, wavesc)
IPython.display.display(IPython.display.Audio("one_step.wav"))




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3158511863.py in <cell line: 0>()
     22     w1, c1, c2 = w.split(" ")
     23     c1, c2 = int(c1), int(c2)
---> 24     _, data = wavfile.read(os.path.join(BASE, "train_noisy", w1))
     25     wavesc.append(data[c1:c2])
     26 

/usr/local/lib/python3.11/dist-packages/scipy/io/wavfile.py in read(filename, mmap)
    672         mmap = False
    673     else:
--> 674         fid = open(filename, 'rb')
    675 
    676     try:

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/data/freesound-audio-tagging-2019/train_noisy/3a5b14ee.wav'

## === cell 3
def get_short_wave(w, size=8000):
    rate, data = wavfile.read(w)
    if len(data) > size:
        return data[:size]
    else:
        repeat_factor = int((size / len(data)) + 1)
        return np.repeat(data, repeat_factor)[:size]




## === cell 4
train["nframes"] = train["path"].map(lambda x: wave.open(x).getnframes())
train["short_wave"] = train["path"].map(lambda x: get_short_wave(x))

test["nframes"] = test["path"].map(lambda x: wave.open(x).getnframes())
test["short_wave"] = test["path"].map(lambda x: get_short_wave(x))




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/336072680.py in <cell line: 0>()
      1 # Add simple waveform statistics
----> 2 train["nframes"] = train["path"].map(lambda x: wave.open(x).getnframes())
      3 train["short_wave"] = train["path"].map(lambda x: get_short_wave(x))
      4 
      5 test["nframes"] = test["path"].map(lambda x: wave.open(x).getnframes())

NameError: name 'train' is not defined

## === cell 5
def features(df, col="short_wave"):
    for agg in ["min", "max", "sum", "mean", "std", "skew", "kurtosis"]:
        df[col + agg] = df[col].map(
            lambda x: eval("pd.DataFrame(x)." + agg + "(axis=0)")[0]
        )
        df[col + "a" + agg] = df[col].map(
            lambda x: eval("pd.DataFrame(x).abs()." + agg + "(axis=0)")[0]
        )
    return df


train = features(train).fillna(-999)
test = features(test).fillna(-999)
print("Feature shape:", train.shape, test.shape)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2431494135.py in <cell line: 0>()
     10 
     11 
---> 12 train = features(train).fillna(-999)
     13 test = features(test).fillna(-999)
     14 print("Feature shape:", train.shape, test.shape)

NameError: name 'train' is not defined

## === cell 6
feature_cols = [
    c
    for c in train.columns
    if c not in ["path", "fname", "noisy", "labels", "short_wave"]
]

le = preprocessing.LabelEncoder()
train["labels_enc"] = le.fit_transform(train["labels"])

split_idx = int(0.9 * len(train))

clf1 = ensemble.ExtraTreesClassifier(n_jobs=-1, n_estimators=300, random_state=42)
clf2 = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=300, random_state=42)

clf1.fit(train[feature_cols][:split_idx], train["labels_enc"][:split_idx])
clf2.fit(train[feature_cols][:split_idx], train["labels_enc"][:split_idx])


def lol_wrap(y_true, y_score):
    return metrics.label_ranking_average_precision_score(y_true, y_score)


val_lrap_et = lol_wrap(
    pd.get_dummies(train["labels_enc"][split_idx:]),
    clf1.predict_proba(train[feature_cols][split_idx:]),
)
val_lrap_rf = lol_wrap(
    pd.get_dummies(train["labels_enc"][split_idx:]),
    clf2.predict_proba(train[feature_cols][split_idx:]),
)
print("Validation LRAP – ExtraTrees:", val_lrap_et)
print("Validation LRAP – RandomForest:", val_lrap_rf)

clf1.fit(train[feature_cols], train["labels_enc"])
clf2.fit(train[feature_cols], train["labels_enc"])

test_pred = (
    clf1.predict_proba(test[feature_cols]) + clf2.predict_proba(test[feature_cols])
) / 2

sub = pd.DataFrame(test_pred, columns=le.classes_)
sub["fname"] = test["fname"]

for col in label_cols:
    if col not in sub.columns:
        sub[col] = 0.0
sub = sub[["fname"] + label_cols]

sub.to_csv("submission.csv", index=False)
print("Saved submission.csv with shape:", sub.shape)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1469846815.py in <cell line: 0>()
      2 feature_cols = [
      3     c
----> 4     for c in train.columns
      5     if c not in ["path", "fname", "noisy", "labels", "short_wave"]
      6 ]

NameError: name 'train' is not defined
