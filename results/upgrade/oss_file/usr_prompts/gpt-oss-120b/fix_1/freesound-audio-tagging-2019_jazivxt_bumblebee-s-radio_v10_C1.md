# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import wave, IPython
from sklearn import *
from scipy.io import wavfile
import gc; gc.enable()

train = pd.read_csv('../input/train_curated.csv')
trainn = pd.read_csv('../input/train_noisy.csv')[:4000]
test = pd.read_csv('../input/sample_submission.csv')
train.shape, trainn.shape, test.shape


## === cell 1
train['path'] = train['fname'].map(lambda x: '../input/train_curated/'+x)
trainn['path'] = trainn['fname'].map(lambda x: '../input/train_noisy/'+x)
test['path'] = test['fname'].map(lambda x: '../input/test/'+x)

train['noisy'] = 0; trainn['noisy'] = 1
train = pd.concat((train, trainn), sort=False).reset_index(drop=True)

labels = [c for c in test.columns if c not in ['path','fname']]
train = train[train['labels'].isin(labels)].reset_index(drop=True)
test = test[['path', 'fname']]
train.shape, test.shape


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
aca0ce49.wav 578130 597401""".split('\n')
wavesc = []
for w in waves:
    w1, c1, c2 = w.split(' ')
    c1 = int(c1); c2 = int(c2)
    nrate, ndata = wavfile.read('../input/train_noisy/'+w1)
    wavesc.append(ndata[c1:c2])
wavesc = np.concatenate(wavesc)
wavfile.write('one_step.wav', 44100, wavesc)
IPython.display.display(IPython.display.Audio('one_step.wav'))


## === cell 3
def get_short_wave(w, size=8000):
    rate, data = wavfile.read(w)
    if len(data) > size:
        return data[:size]
    else:
        print(int((size / len(data))+1), len(data))
        return np.repeat(data, int((size / len(data))+1))[:size]


## === cell 4
train['nframes'] = train['path'].map(lambda x: wave.open(x).getnframes())
train['short_wave'] = train['path'].map(lambda x: get_short_wave(x))

test['nframes'] = test['path'].map(lambda x: wave.open(x).getnframes())
test['short_wave'] = test['path'].map(lambda x: get_short_wave(x))


## === cell 5
def features(df, col='short_wave'):
    for agg in ['min', 'max', 'sum', 'mean', 'std', 'skew', 'kurtosis']:
        df[col+agg] = df[col].map(lambda x: eval('pd.DataFrame(x).' + agg + '(axis=0)')[0])
        df[col+'a'+agg] = df[col].map(lambda x: eval('pd.DataFrame(x).abs().' + agg + '(axis=0)')[0])
    return df

train = features(train).fillna(-999)
test = features(test).fillna(-999)
print(train.shape, test.shape)


## === cell 6
col = [c for c in train.columns if c not in ['path','fname', 'noisy', 'labels', 'short_wave']]
le = preprocessing.LabelEncoder()
train['labels'] = le.fit_transform(train['labels'])

clf1 = ensemble.ExtraTreesClassifier(n_jobs=-1, n_estimators=300)
clf2 = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=300)

split = 6000
clf1.fit(train[col][:split], train['labels'][:split])
clf2.fit(train[col][:split], train['labels'][:split])
def LOL_WRAP(y_true, y_score): return metrics.label_ranking_average_precision_score(y_true, y_score)

print('ETR LOL_WRAP', LOL_WRAP(pd.get_dummies(train['labels'])[split:], clf1.predict_proba(train[col][split:])))
print('RFR LOL_WRAP', LOL_WRAP(pd.get_dummies(train['labels'])[split:], clf2.predict_proba(train[col][split:])))

clf1.fit(train[col], train['labels'])
clf2.fit(train[col], train['labels'])

sub = clf1.predict_proba(test[col])
sub += clf2.predict_proba(test[col])
sub /= 2

sub = pd.DataFrame(sub, columns=le.classes_)
sub['fname'] = test['fname']
missing = [c for c in labels if c not in sub.columns]
for c in missing:
    sub[c] = 0.0
sub.to_csv('submission.csv', index=False)
