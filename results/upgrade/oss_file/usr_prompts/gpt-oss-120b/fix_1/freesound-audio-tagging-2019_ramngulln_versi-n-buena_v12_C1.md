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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import IPython
import IPython.display
import PIL


import os
print(os.listdir("../input"))



## === cell 4
DATA = Path("../input")
CSV_TRN_CURATED = DATA/"freesound-audio-tagging-2019/train_curated.csv"
CSV_TRN_NOISY = DATA/"freesound-audio-tagging-2019/train_noisy.csv"
CSV_SUBMISSION = DATA/'freesound-audio-tagging-2019/sample_submission.csv'
TRN_CURATED = Path("../input/train-curated-zip")
TRN_NOISY = Path("../input/noisy")
TEST = "../input/freesound-audio-tagging-2019/test.zip"

WORK = Path('work')
IMG_TRN_CURATED = WORK/'image/trn_curated'
IMG_TRN_NOISY = WORK/'image/train_noisy'
IMG_TEST = WORK/'image/test'
for folder in [WORK, IMG_TRN_CURATED, IMG_TRN_NOISY, IMG_TEST]: 
    Path(folder).mkdir(exist_ok=True, parents=True)

df = pd.read_csv(CSV_TRN_CURATED)
test_df = pd.read_csv(CSV_SUBMISSION)

## === cell 5
ruta_extraccion = "datos"

## === cell 6
import zipfile
ruta_zip = TEST
ruta_extraccion = "datos"
password = None
archivo_zip = zipfile.ZipFile(ruta_zip, "r")
try:
    print(archivo_zip.namelist())
    archivo_zip.extractall(pwd=password, path=ruta_extraccion)
except:
    pass
archivo_zip.close()

## === cell 7
df = df.drop(range(235,4970))
    
    

## === cell 9
import librosa
import librosa.display

def read_audio(conf, pathname, trim_long_data):
    y, sr = librosa.load(pathname, sr=conf.sampling_rate)
    if 0 < len(y): # workaround: 0 length causes error
        y, _ = librosa.effects.trim(y) # trim, top_db=default(60)
    if len(y) > conf.samples: # long enough
        if trim_long_data:
            y = y[0:0+conf.samples]
    else: # pad blank
        padding = conf.samples - len(y)    # add padding at both ends
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), 'constant')
    return y

def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(audio, 
                                                 sr=conf.sampling_rate,
                                                 n_mels=conf.n_mels,
                                                 hop_length=conf.hop_length,
                                                 n_fft=conf.n_fft,
                                                 fmin=conf.fmin,
                                                 fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram

def show_melspectrogram(conf, mels, title='Log-frequency power spectrogram'):
    librosa.display.specshow(mels, x_axis='time', y_axis='mel', 
                             sr=conf.sampling_rate, hop_length=conf.hop_length,
                            fmin=conf.fmin, fmax=conf.fmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()

def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):
    x = read_audio(conf, pathname, trim_long_data)
    mels = audio_to_melspectrogram(conf, x)
    if debug_display:
        IPython.display.display(IPython.display.Audio(x, rate=conf.sampling_rate))
        show_melspectrogram(conf, mels)
    return mels


class conf:
    sampling_rate = 44100
    duration = 2
    hop_length = 347*duration # to make time steps 128
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20
    samples = sampling_rate * duration

x = read_as_melspectrogram(conf, TRN_CURATED/'0006ae4e.wav', trim_long_data=False, debug_display=True)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
LibsndfileError                           Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/librosa/core/audio.py in load(path, sr, mono, offset, duration, dtype, res_type)
    175         try:
--> 176             y, sr_native = __soundfile_load(path, offset, duration, dtype)
    177 

/usr/local/lib/python3.11/dist-packages/librosa/core/audio.py in __soundfile_load(path, offset, duration, dtype)
    208         # Otherwise, create the soundfile object
--> 209         context = sf.SoundFile(path)
    210 

/usr/local/lib/python3.11/dist-packages/soundfile.py in __init__(self, file, mode, samplerate, channels, subtype, endian, format, closefd, compression_level, bitrate_mode)
    689                                          format, subtype, endian)
--> 690         self._file = self._open(file, mode_int, closefd)
    691         if set(mode).issuperset('r+') and self.seekable():

/usr/local/lib/python3.11/dist-packages/soundfile.py in _open(self, file, mode_int, closefd)
   1264             err = _snd.sf_error(file_ptr)
-> 1265             raise LibsndfileError(err, prefix="Error opening {0!r}: ".format(self.name))
   1266         if mode_int == _snd.SFM_WRITE:

LibsndfileError: Error opening '../input/train-curated-zip/0006ae4e.wav': System error.

During handling of the above exception, another exception occurred:

FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/579063263.py in <cell line: 0>()
     58 
     59 # example
---> 60 x = read_as_melspectrogram(conf, TRN_CURATED/'0006ae4e.wav', trim_long_data=False, debug_display=True)

/tmp/ipykernel_11/579063263.py in read_as_melspectrogram(conf, pathname, trim_long_data, debug_display)
     38 
     39 def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):
---> 40     x = read_audio(conf, pathname, trim_long_data)
     41     mels = audio_to_melspectrogram(conf, x)
     42     if debug_display:

/tmp/ipykernel_11/579063263.py in read_audio(conf, pathname, trim_long_data)
      3 
      4 def read_audio(conf, pathname, trim_long_data):
----> 5     y, sr = librosa.load(pathname, sr=conf.sampling_rate)
      6     # trim silence
      7     if 0 < len(y): # workaround: 0 length causes error

/usr/local/lib/python3.11/dist-packages/librosa/core/audio.py in load(path, sr, mono, offset, duration, dtype, res_type)
    182                     "PySoundFile failed. Trying audioread instead.", stacklevel=2
    183                 )
--> 184                 y, sr_native = __audioread_load(path, offset, duration, dtype)
    185             else:
    186                 raise exc

<decorator-gen-120> in __audioread_load(path, offset, duration, dtype)

/usr/local/lib/python3.11/dist-packages/librosa/util/decorators.py in __wrapper(func, *args, **kwargs)
     61             stacklevel=3,  # Would be 2, but the decorator adds a level
     62         )
---> 63         return func(*args, **kwargs)
     64 
     65     return decorator(__wrapper)

/usr/local/lib/python3.11/dist-packages/librosa/core/audio.py in __audioread_load(path, offset, duration, dtype)
    238     else:
    239         # If the input was not an audioread object, try to open it
--> 240         reader = audioread.audio_open(path)
    241 
    242     with reader as input_file:

/usr/local/lib/python3.11/dist-packages/audioread/__init__.py in audio_open(path, backends)
    125     for BackendClass in backends:
    126         try:
--> 127             return BackendClass(path)
    128         except DecodeError:
    129             pass

/usr/local/lib/python3.11/dist-packages/audioread/rawread.py in __init__(self, filename)
     57     """
     58     def __init__(self, filename):
---> 59         self._fh = open(filename, 'rb')
     60 
     61         try:

FileNotFoundError: [Errno 2] No such file or directory: '../input/train-curated-zip/0006ae4e.wav'

## === cell 11
def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    X = np.stack([X, X, X], axis=-1)

    mean = mean or X.mean()
    std = std or X.std()
    Xstd = (X - mean) / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V

def convert_wav_to_image(df, source, img_dest):
    X = []
    for i, row in tqdm_notebook(df.iterrows()):
        x = read_as_melspectrogram(conf, source/str(row.fname), trim_long_data=False)
        x_color = mono_to_color(x)
        X.append(x_color)
    return X

X_train = convert_wav_to_image(df, source=TRN_CURATED, img_dest=IMG_TRN_CURATED)
X_test = convert_wav_to_image(test_df, source=Path(ruta_extraccion), img_dest=IMG_TEST)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
LibsndfileError                           Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/librosa/core/audio.py in load(path, sr, mono, offset, duration, dtype, res_type)
    175         try:
--> 176             y, sr_native = __soundfile_load(path, offset, duration, dtype)
    177 

/usr/local/lib/python3.11/dist-packages/librosa/core/audio.py in __soundfile_load(path, offset, duration, dtype)
    208         # Otherwise, create the soundfile object
--> 209         context = sf.SoundFile(path)
    210 

/usr/local/lib/python3.11/dist-packages/soundfile.py in __init__(self, file, mode, samplerate, channels, subtype, endian, format, closefd, compression_level, bitrate_mode)
    689                                          format, subtype, endian)
--> 690         self._file = self._open(file, mode_int, closefd)
    691         if set(mode).issuperset('r+') and self.seekable():

/usr/local/lib/python3.11/dist-packages/soundfile.py in _open(self, file, mode_int, closefd)
   1264             err = _snd.sf_error(file_ptr)
-> 1265             raise LibsndfileError(err, prefix="Error opening {0!r}: ".format(self.name))
   1266         if mode_int == _snd.SFM_WRITE:

LibsndfileError: Error opening '../input/train-curated-zip/0006ae4e.wav': System error.

During handling of the above exception, another exception occurred:

FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/734060780.py in <cell line: 0>()
     30     return X
     31 
---> 32 X_train = convert_wav_to_image(df, source=TRN_CURATED, img_dest=IMG_TRN_CURATED)
     33 X_test = convert_wav_to_image(test_df, source=Path(ruta_extraccion), img_dest=IMG_TEST)

/tmp/ipykernel_11/734060780.py in convert_wav_to_image(df, source, img_dest)
     25     X = []
     26     for i, row in tqdm_notebook(df.iterrows()):
---> 27         x = read_as_melspectrogram(conf, source/str(row.fname), trim_long_data=False)
     28         x_color = mono_to_color(x)
     29         X.append(x_color)

/tmp/ipykernel_11/579063263.py in read_as_melspectrogram(conf, pathname, trim_long_data, debug_display)
     38 
     39 def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):
---> 40     x = read_audio(conf, pathname, trim_long_data)
     41     mels = audio_to_melspectrogram(conf, x)
     42     if debug_display:

/tmp/ipykernel_11/579063263.py in read_audio(conf, pathname, trim_long_data)
      3 
      4 def read_audio(conf, pathname, trim_long_data):
----> 5     y, sr = librosa.load(pathname, sr=conf.sampling_rate)
      6     # trim silence
      7     if 0 < len(y): # workaround: 0 length causes error

/usr/local/lib/python3.11/dist-packages/librosa/core/audio.py in load(path, sr, mono, offset, duration, dtype, res_type)
    182                     "PySoundFile failed. Trying audioread instead.", stacklevel=2
    183                 )
--> 184                 y, sr_native = __audioread_load(path, offset, duration, dtype)
    185             else:
    186                 raise exc

<decorator-gen-120> in __audioread_load(path, offset, duration, dtype)

/usr/local/lib/python3.11/dist-packages/librosa/util/decorators.py in __wrapper(func, *args, **kwargs)
     61             stacklevel=3,  # Would be 2, but the decorator adds a level
     62         )
---> 63         return func(*args, **kwargs)
     64 
     65     return decorator(__wrapper)

/usr/local/lib/python3.11/dist-packages/librosa/core/audio.py in __audioread_load(path, offset, duration, dtype)
    238     else:
    239         # If the input was not an audioread object, try to open it
--> 240         reader = audioread.audio_open(path)
    241 
    242     with reader as input_file:

/usr/local/lib/python3.11/dist-packages/audioread/__init__.py in audio_open(path, backends)
    125     for BackendClass in backends:
    126         try:
--> 127             return BackendClass(path)
    128         except DecodeError:
    129             pass

/usr/local/lib/python3.11/dist-packages/audioread/rawread.py in __init__(self, filename)
     57     """
     58     def __init__(self, filename):
---> 59         self._fh = open(filename, 'rb')
     60 
     61         try:

FileNotFoundError: [Errno 2] No such file or directory: '../input/train-curated-zip/0006ae4e.wav'

## === cell 13
from fastai import *
from fastai.vision import *
from fastai.vision.data import *
import random

CUR_X_FILES, CUR_X = list(df.fname.values), X_train

def open_fat2019_image(fn, convert_mode, after_open)->Image:
    idx = CUR_X_FILES.index(fn.split('/')[-1])
    x = PIL.Image.fromarray(CUR_X[idx])
    time_dim, base_dim = x.size
    crop_x = random.randint(0, time_dim - base_dim)
    x = x.crop([crop_x, 0, crop_x+base_dim, base_dim])    
    return Image(pil2tensor(x, np.float32).div_(255))

vision.data.open_image = open_fat2019_image

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3095431249.py in <cell line: 0>()
      4 import random
      5 
----> 6 CUR_X_FILES, CUR_X = list(df.fname.values), X_train
      7 
      8 def open_fat2019_image(fn, convert_mode, after_open)->Image:

NameError: name 'X_train' is not defined

## === cell 15
def _one_sample_positive_class_precisions(scores, truth):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    retrieved_classes = np.argsort(scores)[::-1]
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits


def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
      truth: np.array of (num_samples, num_classes) giving boolean ground-truth
        of presence of that class in that sample.
      scores: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.

    Returns:
      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: np.array of (num_classes,) giving the prior of each
        class within the truth labels.  Then the overall unbalanced lwlrap is
        simply np.sum(per_class_lwlrap * weight_per_class)
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = (
            _one_sample_positive_class_precisions(scores[sample_num, :],
                                                  truth[sample_num, :]))
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
            precision_at_hits)
    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    return per_class_lwlrap, weight_per_class


def lwlrap(scores, truth, **kwargs):
    score, weight = calculate_per_class_lwlrap(to_np(truth), to_np(scores))
    return torch.Tensor([(score * weight).sum()])


## === cell 16
tfms = get_transforms(do_flip=True, max_rotate=0, max_lighting=0.1, max_zoom=0, max_warp=0.)
src = (ImageList.from_df(df, Path('../../')/CSV_TRN_CURATED, folder='trn_curated')
       .split_by_rand_pct(0.2)
       .label_from_df(label_delim=',')
)
data = (src.transform(tfms, size=128)
        .databunch(bs=64).normalize(imagenet_stats)
)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3486852805.py in <cell line: 0>()
----> 1 tfms = get_transforms(do_flip=True, max_rotate=0, max_lighting=0.1, max_zoom=0, max_warp=0.)
      2 src = (ImageList.from_df(df, Path('../../')/CSV_TRN_CURATED, folder='trn_curated')
      3        .split_by_rand_pct(0.2)
      4        .label_from_df(label_delim=',')
      5 )

NameError: name 'get_transforms' is not defined

## === cell 17
data.show_batch(3)

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4014006935.py in <cell line: 0>()
----> 1 data.show_batch(3)

NameError: name 'data' is not defined

## === cell 18
learn = cnn_learner(data, models.resnet18, pretrained=False, metrics=[lwlrap])
learn.unfreeze()

learn.lr_find(); learn.recorder.plot()

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1411211145.py in <cell line: 0>()
----> 1 learn = cnn_learner(data, models.resnet18, pretrained=False, metrics=[lwlrap])
      2 learn.unfreeze()
      3 
      4 learn.lr_find(); learn.recorder.plot()

NameError: name 'cnn_learner' is not defined

## === cell 19
learn.fit_one_cycle(5, 1e-1)
learn.fit_one_cycle(10, 1e-2)

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3157805616.py in <cell line: 0>()
----> 1 learn.fit_one_cycle(5, 1e-1)
      2 learn.fit_one_cycle(10, 1e-2)

NameError: name 'learn' is not defined

## === cell 20
learn.lr_find(); learn.recorder.plot()

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3116522303.py in <cell line: 0>()
----> 1 learn.lr_find(); learn.recorder.plot()

NameError: name 'learn' is not defined

## === cell 21
learn.fit_one_cycle(20, 3e-3)

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2900721522.py in <cell line: 0>()
----> 1 learn.fit_one_cycle(20, 3e-3)

NameError: name 'learn' is not defined

## === cell 22
learn.lr_find(); learn.recorder.plot()

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3116522303.py in <cell line: 0>()
----> 1 learn.lr_find(); learn.recorder.plot()

NameError: name 'learn' is not defined

## === cell 23
learn.fit_one_cycle(20, 1e-3)

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3872511102.py in <cell line: 0>()
----> 1 learn.fit_one_cycle(20, 1e-3)

NameError: name 'learn' is not defined

## === cell 24
learn.lr_find(); learn.recorder.plot()

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3116522303.py in <cell line: 0>()
----> 1 learn.lr_find(); learn.recorder.plot()

NameError: name 'learn' is not defined

## === cell 25
learn.fit_one_cycle(50, slice(1e-3, 3e-3))

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/560504691.py in <cell line: 0>()
----> 1 learn.fit_one_cycle(50, slice(1e-3, 3e-3))

NameError: name 'learn' is not defined

## === cell 26
learn.fit_one_cycle(10, slice(1e-4, 1e-3))

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1522456141.py in <cell line: 0>()
----> 1 learn.fit_one_cycle(10, slice(1e-4, 1e-3))

NameError: name 'learn' is not defined

## === cell 28
from sklearn.preprocessing import minmax_scale

def visualize_first_layer(learn, save_name=None):
    conv1 = list(learn.model.children())[0][0]
    if isinstance(conv1, torch.nn.modules.container.Sequential):
        conv1 = conv1[0] # for some models, 1 layer inside
    weights = conv1.weight.data.cpu().numpy()
    weights_shape = weights.shape
    weights = minmax_scale(weights.ravel()).reshape(weights_shape)
    fig, axes = plt.subplots(8, 8, figsize=(8,8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.rollaxis(weights[i], 0, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    if save_name:
        fig.savefig(str(save_name))

visualize_first_layer(learn)

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4227798874.py in <cell line: 0>()
     17         fig.savefig(str(save_name))
     18 
---> 19 visualize_first_layer(learn)

NameError: name 'learn' is not defined

## === cell 29
learn.save('fat2019_fastai_cnn2d_stage-2')
learn.export(file = Path("/kaggle/working/export.pkl"))

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1552977016.py in <cell line: 0>()
----> 1 learn.save('fat2019_fastai_cnn2d_stage-2')
      2 learn.export(file = Path("/kaggle/working/export.pkl"))

NameError: name 'learn' is not defined

## === cell 32
CUR_X_FILES, CUR_X = list(test_df.fname.values), X_test

test = ImageList.from_df(test_df, Path('../..')/CSV_SUBMISSION, folder='test')
learn = load_learner("./", test=test)
preds, _ = learn.TTA(ds_type=DatasetType.Test) # <== Simply replacing from learn.get_preds()

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2386429180.py in <cell line: 0>()
----> 1 CUR_X_FILES, CUR_X = list(test_df.fname.values), X_test
      2 
      3 test = ImageList.from_df(test_df, Path('../..')/CSV_SUBMISSION, folder='test')
      4 learn = load_learner("./", test=test)
      5 preds, _ = learn.TTA(ds_type=DatasetType.Test) # <== Simply replacing from learn.get_preds()

NameError: name 'X_test' is not defined

## === cell 33
test_df[learn.data.classes] = preds
test_df.to_csv('submission.csv', index=False)
test_df.head()

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/640963388.py in <cell line: 0>()
----> 1 test_df[learn.data.classes] = preds
      2 test_df.to_csv('submission.csv', index=False)
      3 test_df.head()

NameError: name 'preds' is not defined

## === cell 34
CUR_X_FILES, CUR_X = list(df.fname.values), X_train
learn = cnn_learner(data, models.resnet18, pretrained=False, metrics=[lwlrap])
learn.load('fat2019_fastai_cnn2d_stage-2');

## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3369141294.py in <cell line: 0>()
----> 1 CUR_X_FILES, CUR_X = list(df.fname.values), X_train
      2 learn = cnn_learner(data, models.resnet18, pretrained=False, metrics=[lwlrap])
      3 learn.load('fat2019_fastai_cnn2d_stage-2');

NameError: name 'X_train' is not defined

## === cell 36
from fastai.callbacks.hooks import *

def visualize_cnn_by_cam(learn, data_index):
    x, _y = learn.data.valid_ds[data_index]
    y = _y.data
    if not isinstance(y, (list, np.ndarray)): # single label -> one hot encoding
        y = np.eye(learn.data.valid_ds.c)[y]

    m = learn.model.eval()
    xb,_ = learn.data.one_item(x)
    xb_im = Image(learn.data.denorm(xb)[0])
    xb = xb.cuda()

    def hooked_backward(cat):
        with hook_output(m[0]) as hook_a: 
            with hook_output(m[0], grad=True) as hook_g:
                preds = m(xb)
                preds[0,int(cat)].backward()
        return hook_a,hook_g
    def show_heatmap(img, hm, label):
        _,axs = plt.subplots(1, 2)
        axs[0].set_title(label)
        img.show(axs[0])
        axs[1].set_title(f'CAM of {label}')
        img.show(axs[1])
        axs[1].imshow(hm, alpha=0.6, extent=(0,img.shape[0],img.shape[0],0),
                      interpolation='bilinear', cmap='magma');
        plt.show()

    for y_i in np.where(y > 0)[0]:
        hook_a,hook_g = hooked_backward(cat=y_i)
        acts = hook_a.stored[0].cpu()
        grad = hook_g.stored[0][0].cpu()
        grad_chan = grad.mean(1).mean(1)
        mult = (acts*grad_chan[...,None,None]).mean(0)
        show_heatmap(img=xb_im, hm=mult, label=str(learn.data.valid_ds.y[data_index]))

for idx in range(10):
    visualize_cnn_by_cam(learn, idx)

## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1392089960.py in <cell line: 0>()
      1 # Thanks to https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson6-pets-more.ipynb
----> 2 from fastai.callbacks.hooks import *
      3 
      4 def visualize_cnn_by_cam(learn, data_index):
      5     x, _y = learn.data.valid_ds[data_index]

ModuleNotFoundError: No module named 'fastai.callbacks'

## === cell 37
from shutil import rmtree
rmtree("./datos")
