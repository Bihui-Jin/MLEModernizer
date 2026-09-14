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

## === cell 0
import os, glob, random, math, gc
import numpy as np, pandas as pd
import librosa, librosa.display

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

tf.random.set_seed(9)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
root_path = "/kaggle"
train_path = os.path.join(
    root_path, "input", "tensorflow-speech-recognition-challenge", "train", "audio"
)
test_path = os.path.join(
    root_path, "input", "tensorflow-speech-recognition-challenge", "test", "audio"
)



## === cell 2
ROW_FRAMES = 32  # fixed number of MFCC time frames
N_MFCC = 40  # number of MFCC coefficients


def pad_audio(samples, L):
    if len(samples) >= L:
        return samples
    return np.pad(samples, (L - len(samples), 0), mode="constant")


def chop_audio(samples, L=16000):
    while True:
        beg = np.random.randint(0, len(samples) - L)
        yield samples[beg : beg + L]


def choose_background(sound, backgrounds, max_alpha=0.7):
    if not backgrounds:
        return sound
    gen = backgrounds[np.random.randint(len(backgrounds))]
    bg = next(gen) * np.random.uniform(0, max_alpha)
    out = sound + bg
    return out.astype(sound.dtype)


def random_shift(sound, shift_max=0.2, sr=16000):
    shift = np.random.randint(sr * shift_max)
    out = np.roll(sound, shift)
    if shift > 0:
        out[:shift] = 0
    else:
        out[shift:] = 0
    return out


def random_change_pitch(x, sr=16000):
    factor = np.random.randint(1, 4)
    return librosa.effects.pitch_shift(x, sr, factor)


def random_speed_up(x):
    where = ["start", "end"][np.random.randint(0, 2)]
    factor = np.random.uniform(0, 0.5)
    up = librosa.effects.time_stretch(x, 1 + factor)
    if where == "end":
        up = np.concatenate((up, np.zeros(x.shape[0] - up.shape[0])))
    else:
        up = np.concatenate((np.zeros(x.shape[0] - up.shape[0]), up))
    return up


def get_image_list(base_path):
    classes = [c for c in os.listdir(base_path) if c != "_background_noise_"]
    idx_map = {c: i for i, c in enumerate(classes)}
    files, labels = [], []
    for c in classes:
        folder = os.path.join(base_path, c)
        wavs = [
            os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".wav")
        ]
        files.extend(wavs)
        labels.extend([idx_map[c]] * len(wavs))
    return np.array(files), np.array(labels), idx_map


def split_stratified(files, labels, train_frac=0.9):
    return train_test_split(
        files, labels, train_size=train_frac, stratify=labels, random_state=9
    )


def preprocess_data(
    file, background_generator, target_sr=16000, n_mfcc=N_MFCC, augment_prob=0.7
):
    x, sr = librosa.load(file, sr=target_sr)
    x = pad_audio(x, sr)
    if random.random() > augment_prob:
        x = choose_background(x, background_generator)
    if random.random() > augment_prob:
        x = random_shift(x)
    if random.random() > augment_prob:
        x = random_change_pitch(x)
    if random.random() > augment_prob:
        x = random_speed_up(x)
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.moveaxis(mfcc, 0, 1)  # (time, n_mfcc)
    if mfcc.shape[0] < ROW_FRAMES:
        pad_width = ROW_FRAMES - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode="constant")
    else:
        mfcc = mfcc[:ROW_FRAMES, :]
    mfcc = StandardScaler().fit_transform(mfcc)
    return mfcc[..., np.newaxis]  # (time, n_mfcc, 1)


class DataGenerator(Sequence):
    def __init__(self, files, labels, batch_size, background_generator):
        self.files, self.labels = files, labels
        self.batch_size = batch_size
        self.bg = background_generator

    def __len__(self):
        return math.ceil(len(self.files) / self.batch_size)

    def __getitem__(self, idx):
        batch_files = self.files[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]
        X = [preprocess_data(f, self.bg) for f in batch_files]
        return np.array(X), np.array(batch_labels)


def build_model(n_classes, input_shape):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out)




## === cell 3
bg_files = glob.glob(os.path.join(train_path, "_background_noise_", "*.wav"))
bg_wavs = [librosa.load(f, sr=16000)[0] for f in bg_files]
background_generator = [chop_audio(w) for w in bg_wavs]



## === cell 4
all_files, all_labels, class_map = get_image_list(train_path)

train_files, val_files, train_labels, val_labels = split_stratified(
    all_files, all_labels, train_frac=0.9
)

batch_size = 100
train_gen = DataGenerator(train_files, train_labels, batch_size, background_generator)
val_gen = DataGenerator(val_files, val_labels, batch_size, None)

input_shape = (ROW_FRAMES, N_MFCC, 1)
model = build_model(len(class_map), input_shape)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

model.summary()



## === cell 5
epochs = 30
steps_per_epoch = math.ceil(len(train_files) / batch_size)
val_steps = math.ceil(len(val_files) / batch_size)

model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=epochs,
    verbose=1,
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4115861702.py in <cell line: 0>()
      3 val_steps = math.ceil(len(val_files) / batch_size)
      4 
----> 5 model.fit(
      6     train_gen,
      7     steps_per_epoch=steps_per_epoch,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_55/3377418315.py in __getitem__(self, idx)
    108         batch_files = self.files[idx * self.batch_size : (idx + 1) * self.batch_size]
    109         batch_labels = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]
--> 110         X = [preprocess_data(f, self.bg) for f in batch_files]
    111         return np.array(X), np.array(batch_labels)
    112 

/tmp/ipykernel_55/3377418315.py in <listcomp>(.0)
    108         batch_files = self.files[idx * self.batch_size : (idx + 1) * self.batch_size]
    109         batch_labels = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]
--> 110         X = [preprocess_data(f, self.bg) for f in batch_files]
    111         return np.array(X), np.array(batch_labels)
    112 

/tmp/ipykernel_55/3377418315.py in preprocess_data(file, background_generator, target_sr, n_mfcc, augment_prob)
     80         x = random_shift(x)
     81     if random.random() > augment_prob:
---> 82         x = random_change_pitch(x)
     83     if random.random() > augment_prob:
     84         x = random_speed_up(x)

/tmp/ipykernel_55/3377418315.py in random_change_pitch(x, sr)
     36 def random_change_pitch(x, sr=16000):
     37     factor = np.random.randint(1, 4)
---> 38     return librosa.effects.pitch_shift(x, sr, factor)
     39 
     40 

TypeError: pitch_shift() takes 1 positional argument but 3 were given

## === cell 6
test_files = glob.glob(os.path.join(test_path, "*.wav"))
test_gen = DataGenerator(
    test_files, np.zeros(len(test_files), dtype=int), batch_size, None
)

test_steps = math.ceil(len(test_files) / batch_size)
pred_probs = model.predict(test_gen, steps=test_steps, verbose=1)
pred_labels = np.argmax(pred_probs, axis=1)

inv_map = {v: k for k, v in class_map.items()}
pred_str = [inv_map[i] for i in pred_labels]

submission = pd.DataFrame(
    {"fname": [os.path.basename(p) for p in test_files], "label": pred_str}
)

submission_dir = os.path.join(root_path, "working")
os.makedirs(submission_dir, exist_ok=True)
submission_path = os.path.join(submission_dir, "submission.csv")
submission.to_csv(submission_path, index=False)
print("Submission written to:", submission_path)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2821983263.py in <cell line: 0>()
      6 
      7 test_steps = math.ceil(len(test_files) / batch_size)
----> 8 pred_probs = model.predict(test_gen, steps=test_steps, verbose=1)
      9 pred_labels = np.argmax(pred_probs, axis=1)
     10 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_55/3377418315.py in __getitem__(self, idx)
    108         batch_files = self.files[idx * self.batch_size : (idx + 1) * self.batch_size]
    109         batch_labels = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]
--> 110         X = [preprocess_data(f, self.bg) for f in batch_files]
    111         return np.array(X), np.array(batch_labels)
    112 

/tmp/ipykernel_55/3377418315.py in <listcomp>(.0)
    108         batch_files = self.files[idx * self.batch_size : (idx + 1) * self.batch_size]
    109         batch_labels = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]
--> 110         X = [preprocess_data(f, self.bg) for f in batch_files]
    111         return np.array(X), np.array(batch_labels)
    112 

/tmp/ipykernel_55/3377418315.py in preprocess_data(file, background_generator, target_sr, n_mfcc, augment_prob)
     80         x = random_shift(x)
     81     if random.random() > augment_prob:
---> 82         x = random_change_pitch(x)
     83     if random.random() > augment_prob:
     84         x = random_speed_up(x)

/tmp/ipykernel_55/3377418315.py in random_change_pitch(x, sr)
     36 def random_change_pitch(x, sr=16000):
     37     factor = np.random.randint(1, 4)
---> 38     return librosa.effects.pitch_shift(x, sr, factor)
     39 
     40 

TypeError: pitch_shift() takes 1 positional argument but 3 were given
