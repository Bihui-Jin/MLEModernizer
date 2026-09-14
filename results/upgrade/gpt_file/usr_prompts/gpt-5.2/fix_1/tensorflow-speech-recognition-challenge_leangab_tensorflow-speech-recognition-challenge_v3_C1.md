# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Build an algorithm that understands simple spoken commands.

## Metric
Multiclass Accuracy.

There are 12 possible labels for the Test set: `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`, `silence`, `unknown`.

The `unknown` label should be used for a command that is not one one of the first 10 labels or that is not `silence`.

## Submission Format
For audio clip in the test set, you must predict the correct `label`. The submission file should contain a header and have the following format:

```
fname,label
clip_000044442.wav,silence
clip_0000adecb.wav,left
clip_0000d4322.wav,unknown
etc.
```

## Dataset
- **train.7z** - Contains a few informational files and a folder of audio files. The audio folder contains subfolders with 1 second clips of voice commands, with the folder name being the label of the audio clip. There are more labels that should be predicted. The labels you will need to predict in Test are `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`. Everything else should be considered either `unknown` or `silence`. The folder `_background_noise_` contains longer clips of "silence" that you can break up and use as training input.
    
    The files contained in the training audio are not uniquely named across labels, but they are unique if you include the label folder. For example, `00f0204f_nohash_0.wav` is found in 14 folders, but that file is a different speech command in each folder.
    
    The files are named so the first element is the subject id of the person who gave the voice command, and the last element indicated repeated commands. Repeated commands are when the subject repeats the same word multiple times. Subject id is not provided for the test data, and you can assume that the majority of commands in the test data were from subjects not seen in train.
    
    You can expect some inconsistencies in the properties of the training data (e.g., length of the audio).
    
- **test.7z** - Contains an audio folder with 150,000+ files in the format `clip_000044442.wav`. The task is to predict the correct label. Not all of the files are evaluated for the leaderboard score.
- **sample_submission.csv** - A sample submission file in the correct format.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (328 lines)
            sample_submission.csv (6474 lines)
            sample_submission.csv.zip (16.6 kB)
            test.zip (149.1 MB)
            train.zip (1.3 GB)
            tensorflow-speech-recognition-challenge/
                description.md (328 lines)
                sample_submission.csv (6474 lines)
                ... and 3 other files
                tensorflow-speech-recognition-challenge/
                test/
                    audio/
                        clip_00000000.wav (32.0 kB)
                        clip_00000001.wav (32.0 kB)
                        ... and 6471 other files
                    test/
                train/
                    audio/
                        _background_noise_/
                            ... (max depth reached)
                        bed/
                            ... (max depth reached)
                        ... and 29 other folders
                    train/
            test/
                audio/
                    clip_00000000.wav (32.0 kB)
                    clip_00000001.wav (32.0 kB)
                    ... and 6471 other files
                test/
            train/
                audio/
                    _background_noise_/
                        doing_the_dishes.wav (3.0 MB)
                        dude_miaowing.wav (2.0 MB)
                        ... and 3 other files
                    bed/
                        00176480_nohash_0.wav (32.0 kB)
                        004ae714_nohash_0.wav (32.0 kB)
                        ... and 1534 other files
                    ... and 29 other folders
                train/
        input/
            description.md (328 lines)
            sample_submission.csv (6474 lines)
            sample_submission.csv.zip (16.6 kB)
            test.zip (149.1 MB)
            train.zip (1.3 GB)
            tensorflow-speech-recognition-challenge/
                description.md (328 lines)
                sample_submission.csv (6474 lines)
                ... and 3 other files
                tensorflow-speech-recognition-challenge/
                test/
                    audio/
                        clip_00000000.wav (32.0 kB)
                        clip_00000001.wav (32.0 kB)
                        ... and 6471 other files
                    test/
                train/
                    audio/
                        _background_noise_/
                            ... (max depth reached)
                        bed/
                            ... (max depth reached)
                        ... and 29 other folders
                    train/
            test/
                audio/
                    clip_00000000.wav (32.0 kB)
                    clip_00000001.wav (32.0 kB)
                    ... and 6471 other files
                test/
                    audio/
                        clip_00000000.wav (32.0 kB)
                        clip_00000001.wav (32.0 kB)
                        ... and 6471 other files
                    test/
            train/
                audio/
                    _background_noise_/
                        doing_the_dishes.wav (3.0 MB)
                        dude_miaowing.wav (2.0 MB)
                        ... and 3 other files
                    bed/
                        00176480_nohash_0.wav (32.0 kB)
                        004ae714_nohash_0.wav (32.0 kB)
                        ... and 1534 other files
                    ... and 29 other folders
                train/
                    audio/
                        _background_noise_/
                            ... (max depth reached)
                        bed/
                            ... (max depth reached)
                        ... and 29 other folders
                    train/
        working/
            tensorflow-speech-recognition-challenge/
                description.md (328 lines)
                sample_submission.csv (6474 lines)
                ... and 3 other files
                tensorflow-speech-recognition-challenge/
                test/
                    audio/
                        clip_00000000.wav (32.0 kB)
                        clip_00000001.wav (32.0 kB)
                        ... and 6471 other files
                    test/
                train/
                    audio/
                        _background_noise_/
                            ... (max depth reached)
                        bed/
                            ... (max depth reached)
                        ... and 29 other folders
                    train/
```

-> data/sample_submission.csv has 6473 rows and 2 columns.
The columns are: fname, label

-> data/tensorflow-speech-recognition-challenge/sample_submission.csv has 6473 rows and 2 columns.
The columns are: fname, label

-> input/sample_submission.csv has 6473 rows and 2 columns.
The columns are: fname, label

-> input/tensorflow-speech-recognition-challenge/sample_submission.csv has 6473 rows and 2 columns.
The columns are: fname, label

-> working/tensorflow-speech-recognition-challenge/sample_submission.csv has 6473 rows and 2 columns.
The columns are: fname, label

# 5. Target score

0.6199929519558323

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2

!pip install pyunpack
!pip install patool

import os
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
os.system('apt-get install p7zip')

import glob
from pyunpack import Archive
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,activations
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import load_model
import numpy as np 
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa, librosa.display
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import math
import shutil
import pickle
import multiprocessing
import gc

tf.random.set_seed(9)


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
tf.__version__

## === cell 4
from tensorflow.python.client import device_lib

device_lib.list_local_devices()

## === cell 5
root_path = "/kaggle"

## === cell 7

if not os.path.exists(root_path + '/working/train/'):
    os.makedirs(root_path + '/working/train/')
    Archive(root_path + '/input/tensorflow-speech-recognition-challenge/train.7z').extractall(root_path + '/working')
    

train_path = root_path + '/working/train/audio/'

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/28700870.py in <cell line: 0>()
      1 if not os.path.exists(root_path + '/working/train/'):
      2     os.makedirs(root_path + '/working/train/')
----> 3     Archive(root_path + '/input/tensorflow-speech-recognition-challenge/train.7z').extractall(root_path + '/working')
      4 
      5 

/usr/local/lib/python3.11/dist-packages/pyunpack/__init__.py in extractall(self, directory, auto_create_dir, patool_path)
    100         directory = _fullpath(directory)
    101         if not os.path.exists(self.filename):
--> 102             raise ValueError("archive file does not exist:" + str(self.filename))
    103         if not os.path.exists(directory):
    104             if auto_create_dir:

ValueError: archive file does not exist:/kaggle/input/tensorflow-speech-recognition-challenge/train.7z

## === cell 9
train_audio_sample = os.path.join(train_path, "yes/0a7c2a8d_nohash_0.wav")
x,sr = librosa.load(train_audio_sample, sr = 16000)

ipd.Audio(x, rate=sr)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2776708352.py in <cell line: 0>()
----> 1 train_audio_sample = os.path.join(train_path, "yes/0a7c2a8d_nohash_0.wav")
      2 x,sr = librosa.load(train_audio_sample, sr = 16000)
      3 
      4 ipd.Audio(x, rate=sr)

NameError: name 'train_path' is not defined

## === cell 10
hop_length = 256
S = librosa.feature.melspectrogram(x, sr=sr, n_fft=4096, hop_length=hop_length)
logS = librosa.power_to_db(abs(S))

plt.figure(figsize=(14, 9))

plt.figure(1)

plt.subplot(211)
plt.title('Spectrogram')
librosa.display.specshow(logS, sr=sr, hop_length=hop_length, x_axis= None, y_axis='mel')

plt.subplot(212)
plt.title('Audioform')
librosa.display.waveplot(x, sr=sr)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1871670896.py in <cell line: 0>()
      1 hop_length = 256
----> 2 S = librosa.feature.melspectrogram(x, sr=sr, n_fft=4096, hop_length=hop_length)
      3 logS = librosa.power_to_db(abs(S))
      4 
      5 plt.figure(figsize=(14, 9))

NameError: name 'x' is not defined

## === cell 12
mfccs = librosa.feature.mfcc(x, sr=sr,  n_mfcc=40) 
scaler = StandardScaler()
ee= scaler.fit_transform(mfccs.T)

plt.figure(figsize=(14, 5))
librosa.display.specshow(ee.T)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2443703072.py in <cell line: 0>()
----> 1 mfccs = librosa.feature.mfcc(x, sr=sr,  n_mfcc=40)
      2 scaler = StandardScaler()
      3 ee= scaler.fit_transform(mfccs.T)
      4 
      5 plt.figure(figsize=(14, 5))

NameError: name 'x' is not defined

## === cell 14
def pad_audio(samples, L):
    if len(samples) >= L: 
        return samples
    else: 
        return np.pad(samples, pad_width=(L - len(samples), 0), mode='constant', constant_values=(0, 0))
    
    
def chop_audio(samples, L=16000):
    while True:
        beg = np.random.randint(0, len(samples) - L)
        yield samples[beg: beg + L]


def choose_background_generator(sound, backgrounds, max_alpha = 0.7):
    if backgrounds is None:
        return sound
    my_gen = backgrounds[np.random.randint(len(backgrounds))]
    background = next(my_gen) * np.random.uniform(0, max_alpha)
    augmented_data = sound + background
    augmented_data = augmented_data.astype(type(sound[0]))
    return augmented_data 


def random_shift(sound, shift_max = 0.2, sampling_rate = 16000):
    shift = np.random.randint(sampling_rate * shift_max)
    out = np.roll(sound, shift)
    if shift > 0:
        out[:shift] = 0
    else:
        out[shift:] = 0
    return out


def random_change_pitch(x, sr=16000):
    pitch_factor = np.random.randint(1, 4)
    out = librosa.effects.pitch_shift(x, sr, pitch_factor)
    return out


def random_speed_up(x):
    where = ["start", "end"][np.random.randint(0, 1)]
    speed_factor = np.random.uniform(0, 0.5)
    up = librosa.effects.time_stretch(x, 1 + speed_factor)
    up_len = up.shape[0]
    if where == "end":
        up = np.concatenate((up, np.zeros((x.shape[0] - up_len,))))
    else:
        up = np.concatenate((np.zeros((x.shape[0] - up_len,)), up))
    return up


def get_image_list(train_audio_path):
    classes = os.listdir(train_audio_path)
    classes = [thisclass for thisclass in classes if thisclass != '_background_noise_']
    index = [i for i,j in enumerate(classes)]
    outlist = []
    labels = []
    for thisindex,thisclass in zip(index, classes):
        filelist = [f for f in os.listdir(os.path.join(train_audio_path, thisclass)) if f.endswith('.wav')]
        filelist = [os.path.join(train_audio_path, thisclass, x) for x in filelist]
        outlist.append(filelist)
        labels.append(np.full(len(filelist), fill_value= thisindex))   
    return outlist,labels,dict(zip(classes,index))


def split_train_test_stratified_shuffle(images_list, labels, train_size = 0.9):
    classes_size = [len(x) for x in images_list]
    classes_vector = [np.arange(x) for x in classes_size]
    total = np.sum(classes_size)
    total_train = [int(train_size * total * x) for x in classes_size / total]
    train_index = [np.random.choice(x,y,replace=False) for x,y in zip(classes_size,total_train)]
    validation_index = [np.setdiff1d(i,j) for i,j in zip(classes_vector,train_index)]

    train_set = [np.array(x)[idx] for x,idx in zip(images_list,train_index)]
    validation_set = [np.array(x)[idx] for x,idx in zip(images_list,validation_index)]
    train_labels = [np.array(x)[idx] for x,idx in zip(labels,train_index)]
    validation_labels = [np.array(x)[idx] for x,idx in zip(labels,validation_index)]

    train_set = np.array([element for array in train_set for element in array])
    validation_set = np.array([element for array in validation_set for element in array])
    train_labels = np.array([element for array in train_labels for element in array])
    validation_labels = np.array([element for array in validation_labels for element in array])

    train_shuffle = np.random.permutation(len(train_set))
    validation_shuffle =  np.random.permutation(len(validation_set))

    train_set = train_set[train_shuffle]
    validation_set = validation_set[validation_shuffle]
    train_labels = train_labels[train_shuffle]
    validation_labels = validation_labels[validation_shuffle]
    
    return train_set,train_labels,validation_set,validation_labels

        
def preprocess_data(file, background_generator, target_sr = 16000, n_mfcc = 40, threshold = 0.7):
    x,sr = librosa.load(file, sr = target_sr)
    x = pad_audio(x, sr)
    if np.random.uniform(0, 1) > threshold:
        x = choose_background_generator(x, background_generator) # add noinse to 30% of data
    if np.random.uniform(0, 1) > threshold:
        x = random_shift(x) 
    if np.random.uniform(0, 1) > threshold: 
        x = random_change_pitch(x) 
    if np.random.uniform(0, 1) > threshold:
        x = random_speed_up(x) 
    mfccs = librosa.feature.mfcc(x, sr=sr,  n_mfcc=n_mfcc) # transpose for sklearn
    mfccs = np.moveaxis(mfccs, 1, 0)
    scaler = StandardScaler() 
    mfccs_scaled = scaler.fit_transform(mfccs)
    return mfccs_scaled.reshape(mfccs_scaled.shape[0], mfccs_scaled.shape[1], 1) # channels last


class data_generator(Sequence):
    def __init__(self, x_set, y_set, batch_size, background_generator):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.background_generator = background_generator

    def __len__(self):
        return  math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        idx_from = idx * self.batch_size
        idx_to = (idx + 1) * self.batch_size
        batch_x = self.x[idx_from:idx_to]
        batch_y = self.y[idx_from:idx_to]
        x = [preprocess_data(elem, self.background_generator) for elem in batch_x] 
        y = batch_y
        return np.array(x), np.array(y)
    
        

def build_model(n_classes, input_shape):
    model_input = keras.Input(shape=input_shape)
    img_1 = layers.Convolution2D(filters = 32, kernel_size = (3,3), padding = "same", activation=activations.relu)(model_input)
    img_1 = layers.MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = layers.Convolution2D(filters = 64, kernel_size = (3,3), padding = "same", activation=activations.relu)(img_1)
    img_1 = layers.MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = layers.Convolution2D(filters = 128, kernel_size = (3,3), padding = "same", activation=activations.relu)(img_1)
    img_1 = layers.MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = layers.Convolution2D(filters = 256, kernel_size = (3,3), padding = "same", activation=activations.relu)(img_1)
    img_1 = layers.MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = layers.Dropout(rate=0.25)(img_1)
    img_1 = layers.Flatten()(img_1)
    img_1 = layers.Dense(128, activation=activations.relu)(img_1)
    img_1 = layers.Dropout(rate=0.5)(img_1)
    model_output = layers.Dense(n_classes, activation=activations.softmax)(img_1)
    model = keras.Model(model_input, model_output)
    return model


def multiclass_roc(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    all_labels = np.unique(y_test)

    for (idx, c_label) in enumerate(all_labels):
        fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')
    return roc_auc_score(y_test, y_pred, average=average)

## === cell 17

wavfiles = glob.glob(os.path.join(train_path, "_background_noise_/*wav"))
wavfiles = [librosa.load(elem, sr = 16000)[0] for elem in wavfiles]
background_generator = [chop_audio(x) for x in wavfiles]

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/95352028.py in <cell line: 0>()
      1 # Load data with backgrounds
      2 
----> 3 wavfiles = glob.glob(os.path.join(train_path, "_background_noise_/*wav"))
      4 wavfiles = [librosa.load(elem, sr = 16000)[0] for elem in wavfiles]
      5 background_generator = [chop_audio(x) for x in wavfiles]

NameError: name 'train_path' is not defined

## === cell 19

images_list,labels,classes_map =  get_image_list(train_path)

train_set,train_labels,validation_set,validation_labels = split_train_test_stratified_shuffle(images_list,labels)
train_datagen = data_generator(train_set, train_labels, 40, background_generator)
validation_datagen = data_generator(validation_set, validation_labels,40,  None)

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2851391943.py in <cell line: 0>()
      1 # load train
      2 
----> 3 images_list,labels,classes_map =  get_image_list(train_path)
      4 
      5 train_set,train_labels,validation_set,validation_labels = split_train_test_stratified_shuffle(images_list,labels)

NameError: name 'train_path' is not defined

## === cell 22
plt.figure(figsize=(14, 5))
librosa.display.specshow(preprocess_data(train_audio_sample, None).reshape(32, 40).T)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/162428840.py in <cell line: 0>()
      1 plt.figure(figsize=(14, 5))
----> 2 librosa.display.specshow(preprocess_data(train_audio_sample, None).reshape(32, 40).T)

NameError: name 'train_audio_sample' is not defined

## === cell 23
plt.figure(figsize=(14, 5))
librosa.display.specshow(preprocess_data(train_audio_sample, background_generator).reshape(32, 40).T)

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3421576068.py in <cell line: 0>()
      1 plt.figure(figsize=(14, 5))
----> 2 librosa.display.specshow(preprocess_data(train_audio_sample, background_generator).reshape(32, 40).T)

NameError: name 'train_audio_sample' is not defined

## === cell 26
start_random = random_shift(x)
ipd.Audio(start_random , rate=sr)

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1022632954.py in <cell line: 0>()
----> 1 start_random = random_shift(x)
      2 ipd.Audio(start_random , rate=sr)

NameError: name 'x' is not defined

## === cell 28
higher_speed = random_speed_up(x)
ipd.Audio(higher_speed , rate=sr)

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3026982087.py in <cell line: 0>()
----> 1 higher_speed = random_speed_up(x)
      2 ipd.Audio(higher_speed , rate=sr)

NameError: name 'x' is not defined

## === cell 30
pitch_changed = random_change_pitch(x)
ipd.Audio(pitch_changed, rate=sr)

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/570673229.py in <cell line: 0>()
----> 1 pitch_changed = random_change_pitch(x)
      2 ipd.Audio(pitch_changed, rate=sr)

NameError: name 'x' is not defined

## === cell 32
inv_map =  {v: k for k, v in classes_map.items()}
any_present=[i in validation_set for i in train_set]
np.any(any_present)

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2571552187.py in <cell line: 0>()
----> 1 inv_map =  {v: k for k, v in classes_map.items()}
      2 any_present=[i in validation_set for i in train_set]
      3 np.any(any_present)

NameError: name 'classes_map' is not defined

## === cell 34

test1 = np.random.randint(10, 100, 10)

train_set[test1],[inv_map[int(i)] for i in train_labels[test1]]

## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1527153660.py in <cell line: 0>()
      3 test1 = np.random.randint(10, 100, 10)
      4 
----> 5 train_set[test1],[inv_map[int(i)] for i in train_labels[test1]]

NameError: name 'train_set' is not defined

## === cell 35
test1 = np.random.randint(10, 100, 10)
validation_set[test1],[inv_map[int(i)] for i in validation_labels[test1]]

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/989468284.py in <cell line: 0>()
      1 test1 = np.random.randint(10, 100, 10)
----> 2 validation_set[test1],[inv_map[int(i)] for i in validation_labels[test1]]

NameError: name 'validation_set' is not defined

## === cell 37
unique, counts = np.unique(validation_labels, return_counts=True)
x=dict(zip(unique, counts))
out = pd.DataFrame(sorted(x.items(), key=lambda kv: kv[0]))
out.drop(0, inplace = True, axis = 1)
out = out.apply(lambda x: 100 * x/sum(x))

total_labels = [y for x in labels for y in x]
unique, counts = np.unique(total_labels, return_counts=True)
y=dict(zip(unique, counts))
out2 = pd.DataFrame(sorted(y.items(), key=lambda kv: kv[0]))
out2.drop(0, inplace = True, axis = 1)

out2 = out2.apply(lambda x: 100 * x/sum(x))

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3072042159.py in <cell line: 0>()
----> 1 unique, counts = np.unique(validation_labels, return_counts=True)
      2 x=dict(zip(unique, counts))
      3 out = pd.DataFrame(sorted(x.items(), key=lambda kv: kv[0]))
      4 out.drop(0, inplace = True, axis = 1)
      5 out = out.apply(lambda x: 100 * x/sum(x))

NameError: name 'validation_labels' is not defined

## === cell 38
out2.join(out, lsuffix='_left', rsuffix='_right')[:5]

## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2549615791.py in <cell line: 0>()
----> 1 out2.join(out, lsuffix='_left', rsuffix='_right')[:5]

NameError: name 'out2' is not defined

## === cell 39
np.allclose(out.iloc[:,0].values, out2.iloc[:,0].values,  atol=0.01)

## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1682422993.py in <cell line: 0>()
----> 1 np.allclose(out.iloc[:,0].values, out2.iloc[:,0].values,  atol=0.01)

NameError: name 'out' is not defined

## === cell 41
print(keras.backend.image_data_format())

## === cell 43
rows = 32
columns = 40
batch_size = 100
epochs = 50
base_path = root_path + "/working/models"

if not os.path.exists(base_path):
    os.makedirs(base_path)

train_size = train_set.shape[0]
validation_size = validation_set.shape[0]
steps_per_epoch = train_size//batch_size

lr = 1e-3
tensorboard_dir=base_path + "/logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)

checkpoint_filepath = os.path.join(base_path, 'cp-{epoch:04d}.ckpt')
    
    
checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath= checkpoint_filepath,
        save_best_only=True,
        save_weights_only=True,
        monitor='val_acc',
        mode='max',
        verbose=1)

reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                       patience=3, min_lr=1e-5, vebose=1)

    
earlystop_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-3,
        patience=5,
        verbose=1)
    
optimizer = keras.optimizers.Adam(learning_rate = lr)
loss_fn = keras.losses.SparseCategoricalCrossentropy()
acc_metric = keras.metrics.SparseCategoricalAccuracy()
    
model = build_model(len(classes_map), (rows, columns, 1))
model.compile(optimizer = optimizer, loss = loss_fn, metrics= [acc_metric])   
model.summary()

## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4034946813.py in <cell line: 0>()
      8     os.makedirs(base_path)
      9 
---> 10 train_size = train_set.shape[0]
     11 validation_size = validation_set.shape[0]
     12 steps_per_epoch = train_size//batch_size

NameError: name 'train_set' is not defined

## === cell 46

history = model.fit(train_datagen,
                    steps_per_epoch= steps_per_epoch,
                    epochs = epochs,
                    validation_data = validation_datagen,
                    validation_steps = validation_size//batch_size,
                    callbacks=[earlystop_callback, reduce_lr_callback, checkpoint_callback, tensorboard_callback],
                    use_multiprocessing=True)

## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3169538323.py in <cell line: 0>()
      1 # Load the extension and start TensorBoard
      2 
----> 3 history = model.fit(train_datagen,
      4                     steps_per_epoch= steps_per_epoch,
      5                     epochs = epochs,

NameError: name 'model' is not defined

## === cell 47
shutil.rmtree(train_path)
gc.collect()

## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4251964217.py in <cell line: 0>()
----> 1 shutil.rmtree(train_path)
      2 gc.collect()

NameError: name 'train_path' is not defined

## === cell 48
plt.plot(history.history['val_loss'], label = "val_loss")
plt.plot(history.history['loss'], label = "loss")
plt.title('Loss history')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.show()

plt.plot(history.history['sparse_categorical_accuracy'], label = "accuracy")
plt.plot(history.history['val_sparse_categorical_accuracy'], label = "val_accuracy")
plt.title('Accuracy history')
plt.ylabel('Accuracy value (%)')
plt.xlabel('No. epoch')
plt.show()

## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/398561059.py in <cell line: 0>()
      1 # Visualize history
      2 # Plot history: Loss
----> 3 plt.plot(history.history['val_loss'], label = "val_loss")
      4 plt.plot(history.history['loss'], label = "loss")
      5 plt.title('Loss history')

NameError: name 'history' is not defined

## === cell 49
if not os.path.exists(root_path + '/working/test/'):
    os.makedirs(root_path + '/working/test/')
    Archive(root_path + '/input/tensorflow-speech-recognition-challenge/test.7z').extractall(root_path + '/working')

test_path = root_path + '/working/test'

## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/343318441.py in <cell line: 0>()
      1 if not os.path.exists(root_path + '/working/test/'):
      2     os.makedirs(root_path + '/working/test/')
----> 3     Archive(root_path + '/input/tensorflow-speech-recognition-challenge/test.7z').extractall(root_path + '/working')
      4 
      5 test_path = root_path + '/working/test'

/usr/local/lib/python3.11/dist-packages/pyunpack/__init__.py in extractall(self, directory, auto_create_dir, patool_path)
    100         directory = _fullpath(directory)
    101         if not os.path.exists(self.filename):
--> 102             raise ValueError("archive file does not exist:" + str(self.filename))
    103         if not os.path.exists(directory):
    104             if auto_create_dir:

ValueError: archive file does not exist:/kaggle/input/tensorflow-speech-recognition-challenge/test.7z

## === cell 50
test_data,test_labels,_ = get_image_list(test_path) # single folder
test_data = test_data[0]
test_labels = test_labels[0]

test_datagen = data_generator(test_data, test_labels, batch_size,  None)
test_size = len(test_data)
test_steps = np.ceil(test_size / (batch_size))  # steps same than train; https://github.com/keras-team/keras/issues/3477             
y_pred = model.predict_generator(test_datagen, steps = test_steps, verbose=1)

## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2024062938.py in <cell line: 0>()
----> 1 test_data,test_labels,_ = get_image_list(test_path) # single folder
      2 test_data = test_data[0]
      3 test_labels = test_labels[0]
      4 
      5 test_datagen = data_generator(test_data, test_labels, batch_size,  None)

NameError: name 'test_path' is not defined

## === cell 51
y_labs = np.argmax(y_pred, axis=1)

## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2369006302.py in <cell line: 0>()
----> 1 y_labs = np.argmax(y_pred, axis=1)

NameError: name 'y_pred' is not defined

## === cell 53
classes_map

## --- ERROR in cell 53, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1491459961.py in <cell line: 0>()
----> 1 classes_map

NameError: name 'classes_map' is not defined

## === cell 54
inv_map = inv_map = {v: k for k, v in classes_map.items()}

## --- ERROR in cell 54, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3778571998.py in <cell line: 0>()
----> 1 inv_map = inv_map = {v: k for k, v in classes_map.items()}

NameError: name 'classes_map' is not defined

## === cell 55
train_audio_sample =  test_data[16]
x,sr = librosa.load(train_audio_sample, sr = 16000)

ipd.Audio(x, rate=sr)

## --- ERROR in cell 55, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1177314185.py in <cell line: 0>()
----> 1 train_audio_sample =  test_data[16]
      2 x,sr = librosa.load(train_audio_sample, sr = 16000)
      3 
      4 ipd.Audio(x, rate=sr)

NameError: name 'test_data' is not defined

## === cell 56
inv_map[y_labs[16]]

## --- ERROR in cell 56, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1100021917.py in <cell line: 0>()
----> 1 inv_map[y_labs[16]]

NameError: name 'inv_map' is not defined

## === cell 57
train_audio_sample =  test_data[100]
x,sr = librosa.load(train_audio_sample, sr = 16000)

ipd.Audio(x, rate=sr)

## --- ERROR in cell 57, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/474754704.py in <cell line: 0>()
----> 1 train_audio_sample =  test_data[100]
      2 x,sr = librosa.load(train_audio_sample, sr = 16000)
      3 
      4 ipd.Audio(x, rate=sr)

NameError: name 'test_data' is not defined

## === cell 58
inv_map[y_labs[100]]

## --- ERROR in cell 58, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2194673081.py in <cell line: 0>()
----> 1 inv_map[y_labs[100]]

NameError: name 'inv_map' is not defined

## === cell 59
train_audio_sample =  test_data[1001]
x,sr = librosa.load(train_audio_sample, sr = 16000)

ipd.Audio(x, rate=sr)

## --- ERROR in cell 59, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2498110512.py in <cell line: 0>()
----> 1 train_audio_sample =  test_data[1001]
      2 x,sr = librosa.load(train_audio_sample, sr = 16000)
      3 
      4 ipd.Audio(x, rate=sr)

NameError: name 'test_data' is not defined

## === cell 60
inv_map[y_labs[1001]]

## --- ERROR in cell 60, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1929832337.py in <cell line: 0>()
----> 1 inv_map[y_labs[1001]]

NameError: name 'inv_map' is not defined

## === cell 61
shutil.rmtree(test_path)

## --- ERROR in cell 61, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/752103727.py in <cell line: 0>()
----> 1 shutil.rmtree(test_path)

NameError: name 'test_path' is not defined

## === cell 62
my_submission = pd.DataFrame({'fname':  test_data, 'label': [inv_map[x] for x in y_labs]})
my_submission.to_csv('submission.csv', index=False)

## --- ERROR in cell 62, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/354384120.py in <cell line: 0>()
----> 1 my_submission = pd.DataFrame({'fname':  test_data, 'label': [inv_map[x] for x in y_labs]})
      2 my_submission.to_csv('submission.csv', index=False)

NameError: name 'test_data' is not defined
