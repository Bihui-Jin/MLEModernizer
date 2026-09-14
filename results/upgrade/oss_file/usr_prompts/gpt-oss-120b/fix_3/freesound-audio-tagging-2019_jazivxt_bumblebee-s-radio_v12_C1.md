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
import os, numpy as np, pandas as pd, wave, IPython, gc
from sklearn import *
from scipy.io import wavfile

gc.enable()

BASE_PATH = "/kaggle/input/freesound-audio-tagging-2019"

train = pd.read_csv(os.path.join(BASE_PATH, "train_curated.csv"))
trainn = pd.read_csv(os.path.join(BASE_PATH, "train_noisy.csv"))
sample_sub = pd.read_csv(
    os.path.join(BASE_PATH, "sample_submission.csv")
)  # keep for column order
print(train.shape, trainn.shape, sample_sub.shape)




## === cell 1
train["path"] = train["fname"].map(
    lambda x: os.path.join(BASE_PATH, "train_curated", x)
)
trainn["path"] = trainn["fname"].map(
    lambda x: os.path.join(BASE_PATH, "train_noisy", x)
)
test = sample_sub[["fname"]].copy()
test["path"] = test["fname"].map(lambda x: os.path.join(BASE_PATH, "test", x))

train["noisy"] = 0
trainn["noisy"] = 1
train = pd.concat((train, trainn), sort=False).reset_index(drop=True)

labels = [c for c in sample_sub.columns if c not in ["fname"]]
train = train[train["labels"].isin(labels)].reset_index(drop=True)
print(train.shape, test.shape)




## === cell 2
norm_labels = []
cut = 300
for l in train.labels.unique():
    norm_labels.append(train[train.labels == l][:cut])
train = (
    pd.concat(norm_labels, sort=False)
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)
print(train.shape)




## === cell 3
print("Demo cell executed (no heavy processing).")




## === cell 4
def extract_features(path, size=1200):
    """Read wav file, truncate or repeat to `size` samples and compute all stats."""
    rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    nframes = data.shape[0]

    if len(data) > size:
        short = data[:size]
    else:
        repeat = int((size / len(data)) + 1)
        short = np.repeat(data, repeat)[:size]

    agg = {}
    agg["nframes"] = nframes
    agg["short_wavemin"] = short.min()
    agg["short_wavemax"] = short.max()
    agg["short_wavesum"] = short.sum()
    agg["short_wavemedian"] = np.median(short)
    agg["short_wavemean"] = short.mean()
    agg["short_wavestd"] = short.std()
    agg["short_waveskew"] = pd.Series(short).skew()
    agg["short_wavekurtosis"] = pd.Series(short).kurt()

    abs_short = np.abs(short)
    agg["short_waveamin"] = abs_short.min()
    agg["short_waveamax"] = abs_short.max()
    agg["short_waveasum"] = abs_short.sum()
    agg["short_waveamedian"] = np.median(abs_short)
    agg["short_waveamean"] = abs_short.mean()
    agg["short_waveastd"] = abs_short.std()
    agg["short_waveaskew"] = pd.Series(abs_short).skew()
    agg["short_waveakurtosis"] = pd.Series(abs_short).kurt()

    agg["short_wavemax_diff"] = agg["short_wavemax"] - agg["short_wavemean"]
    agg["short_waveamax_diff"] = agg["short_waveamax"] - agg["short_waveamean"]
    agg["short_wavemin_diff"] = agg["short_wavemean"] - agg["short_wavemin"]
    agg["short_waveamin_diff"] = agg["short_waveamean"] - agg["short_waveamin"]
    agg["short_wavemax_diff2"] = agg["short_wavemax"] - agg["short_wavemedian"]
    agg["short_waveamax_diff2"] = agg["short_waveamax"] - agg["short_waveamedian"]
    agg["short_wavemin_diff2"] = agg["short_wavemedian"] - agg["short_wavemin"]
    agg["short_waveamin_diff2"] = agg["short_waveamedian"] - agg["short_waveamin"]

    return pd.Series(agg)




## === cell 5
train_feats = train["path"].apply(extract_features)
train = pd.concat([train, train_feats], axis=1)

test_feats = test["path"].apply(extract_features)
test = pd.concat([test, test_feats], axis=1)

train = train.fillna(-999)
test = test.fillna(-999)

print(train.shape, test.shape)




## === cell 6
pass




## === cell 7
col = [
    c
    for c in train.columns
    if c not in ["path", "fname", "noisy", "labels", "short_wave"]
]
le = preprocessing.LabelEncoder()
train["labels"] = le.fit_transform(train["labels"])

clf1 = ensemble.ExtraTreesClassifier(
    n_jobs=-1, n_estimators=400, max_features=0.9, random_state=10
)
clf2 = ensemble.RandomForestClassifier(
    n_jobs=-1, n_estimators=400, max_features=0.9, random_state=9
)

split = 3000
clf1.fit(train[col][:split], train["labels"][:split])
clf2.fit(train[col][:split], train["labels"][:split])


def LOL_WRAP(y_true, y_score):
    return metrics.label_ranking_average_precision_score(y_true, y_score)


print(
    "ETR LOL_WRAP",
    LOL_WRAP(
        pd.get_dummies(train["labels"])[split:], clf1.predict_proba(train[col][split:])
    ),
)
print(
    "RFR LOL_WRAP",
    LOL_WRAP(
        pd.get_dummies(train["labels"])[split:], clf2.predict_proba(train[col][split:])
    ),
)

clf1.fit(train[col], train["labels"])
clf2.fit(train[col], train["labels"])

sub_pred = (clf1.predict_proba(test[col]) + clf2.predict_proba(test[col])) / 2
sub = pd.DataFrame(sub_pred, columns=le.classes_)
sub["fname"] = test["fname"]

missing = [c for c in labels if c not in sub.columns]
for c in missing:
    sub[c] = 0.0

final_cols = ["fname"] + [c for c in sample_sub.columns if c != "fname"]
sub = sub[final_cols]

sub.to_csv("submission.csv", index=False)
print("Submission written to submission.csv with shape", sub.shape)
