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

# 8. Previous improvement plans

- What this solution (achieved 0.29044) has done: 'I change the parallel execution to use a threading backend, which avoids the need to pickle the generator objects used for background augmentation. This simple change fixes the pickling error in cell 4 and allows the model to be trained, after which cell 5 can correctly use the trained classifier to generate the submission file.'
- What this solution (achieved 0.21551) has done: 'I add a “silence” class built from the background‑noise files so the model can learn to recognise silent clips, and use the classifier’s confidence to label very low‑confidence predictions as `unknown`. This requires only a few extra lines in the training cell (to create and append silence features) and a small change in the prediction cell (to apply the confidence‑threshold). The rest of the pipeline, model, and feature extraction stay unchanged.'
- What this solution (achieved 0.34822) has done: 'I adjust the preprocessing so that a single StandardScaler is fit on the whole training set (instead of per‑sample scaling), raise the augmentation probability, use a balanced logistic‑regression classifier, and tighten the unknown‑confidence threshold. These modest changes keep the original model and feature pipeline while expected to raise validation accuracy toward the target score.'

# 9. Code solution

## === cell 0
import os, glob, random, gc
import numpy as np, pandas as pd
import librosa

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from joblib import Parallel, delayed

random.seed(9)
np.random.seed(9)



## === cell 1
root_path = "/kaggle"
train_path = os.path.join(
    root_path, "input", "tensorflow-speech-recognition-challenge", "train", "audio"
)
test_path = os.path.join(
    root_path, "input", "tensorflow-speech-recognition-challenge", "test", "audio"
)



## === cell 2
ROW_FRAMES = 64
N_MFCC = 40


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
    file, background_generator=None, target_sr=16000, n_mfcc=N_MFCC, augment_prob=0.0
):
    """Load a wav, optionally add background, apply simple augmentations,
    compute MFCCs, pad/trim to a fixed number of frames."""
    x, sr = librosa.load(file, sr=target_sr)
    x = pad_audio(x, sr)

    if background_generator and random.random() > (1 - augment_prob):
        x = choose_background(x, background_generator)
    if random.random() > (1 - augment_prob):
        x = random_shift(x)

    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.moveaxis(mfcc, 0, 1).astype(np.float32)  # (time, n_mfcc), float32

    if mfcc.shape[0] < ROW_FRAMES:
        pad_width = ROW_FRAMES - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode="constant")
    else:
        mfcc = mfcc[:ROW_FRAMES, :]

    return mfcc


def preprocess_array(audio, target_sr=16000, n_mfcc=N_MFCC):
    """Same processing as preprocess_data but starts from a raw ndarray."""
    x = pad_audio(audio, target_sr)
    mfcc = librosa.feature.mfcc(y=x, sr=target_sr, n_mfcc=n_mfcc)
    mfcc = np.moveaxis(mfcc, 0, 1).astype(np.float32)
    if mfcc.shape[0] < ROW_FRAMES:
        pad_width = ROW_FRAMES - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode="constant")
    else:
        mfcc = mfcc[:ROW_FRAMES, :]
    return mfcc




## === cell 3
bg_files = glob.glob(os.path.join(train_path, "_background_noise_", "*.wav"))
bg_wavs = [librosa.load(f, sr=16000)[0] for f in bg_files]
background_generator = [chop_audio(w) for w in bg_wavs]

all_files, all_labels, class_map = get_image_list(train_path)

silence_label = "silence"
silence_idx = len(class_map)
class_map[silence_label] = silence_idx

train_files, val_files, train_labels, val_labels = split_stratified(
    all_files, all_labels, train_frac=0.9
)



## === cell 4
X_train_list = Parallel(n_jobs=-1, backend="loky")(
    delayed(preprocess_data)(f, background_generator, augment_prob=0.8)
    for f in train_files
)
X_train = np.stack(X_train_list)
del X_train_list
gc.collect()

X_val_list = Parallel(n_jobs=-1, backend="loky")(
    delayed(preprocess_data)(f, None, augment_prob=0.0) for f in val_files
)
X_val = np.stack(X_val_list)
del X_val_list
gc.collect()

nsamples, nrows, ncols = X_train.shape
X_train_flat = X_train.reshape(nsamples, nrows * ncols)

scaler = StandardScaler(copy=False)
X_train_flat = scaler.fit_transform(X_train_flat)

nsamples_val = X_val.shape[0]
X_val_flat = X_val.reshape(nsamples_val, nrows * ncols)
X_val_flat = scaler.transform(X_val_flat)

silence_features = []
for wav in bg_wavs:
    slice_ = wav[:16000] if len(wav) >= 16000 else wav
    mfcc = preprocess_array(slice_)
    silence_features.append(mfcc)

if silence_features:
    silence_features = np.stack(silence_features)
    n_sil = silence_features.shape[0]
    silence_flat = silence_features.reshape(n_sil, nrows * ncols)
    silence_flat = scaler.transform(silence_flat)
    X_train_flat = np.vstack([X_train_flat, silence_flat])
    silence_labels = np.full(n_sil, silence_idx, dtype=int)
    train_labels = np.concatenate([train_labels, silence_labels])
    del silence_features, silence_flat
    gc.collect()

clf = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=600,
    n_jobs=-1,
    random_state=9,
    class_weight="balanced",
)
clf.fit(X_train_flat, train_labels)

val_acc = clf.score(X_val_flat, val_labels)
print(f"Validation accuracy: {val_acc:.4f}")



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/backend/queues.py", line 159, in _feed
    obj_ = dumps(obj, reducers=reducers)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/backend/reduction.py", line 214, in dumps
    dump(obj, buf, reducers=reducers, protocol=protocol)
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/backend/reduction.py", line 207, in dump
    _LokyPickler(file, reducers=reducers, protocol=protocol).dump(obj)
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/cloudpickle/cloudpickle.py", line 1303, in dump
    return super().dump(obj)
           ^^^^^^^^^^^^^^^^^
TypeError: cannot pickle 'generator' object
"""

The above exception was the direct cause of the following exception:

PicklingError                             Traceback (most recent call last)
/tmp/ipykernel_55/3106732874.py in <cell line: 0>()
      1 # Use process‑based parallelism (loky) for faster CPU utilisation
----> 2 X_train_list = Parallel(n_jobs=-1, backend="loky")(
      3     delayed(preprocess_data)(f, background_generator, augment_prob=0.8)
      4     for f in train_files
      5 )

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in __call__(self, iterable)
   2070         next(output)
   2071 
-> 2072         return output if self.return_generator else list(output)
   2073 
   2074     def __repr__(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _get_outputs(self, iterator, pre_dispatch)
   1680 
   1681             with self._backend.retrieval_context():
-> 1682                 yield from self._retrieve()
   1683 
   1684         except GeneratorExit:

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _retrieve(self)
   1782             # worker traceback.
   1783             if self._aborting:
-> 1784                 self._raise_error_fast()
   1785                 break
   1786 

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _raise_error_fast(self)
   1857         # called directly or if the generator is gc'ed.
   1858         if error_job is not None:
-> 1859             error_job.get_result(self.timeout)
   1860 
   1861     def _warn_exit_early(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in get_result(self, timeout)
    756             # callback thread, and is stored internally. It's just waiting to
    757             # be returned.
--> 758             return self._return_or_raise()
    759 
    760         # For other backends, the main thread needs to run the retrieval step.

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _return_or_raise(self)
    771         try:
    772             if self.status == TASK_ERROR:
--> 773                 raise self._result
    774             return self._result
    775         finally:

PicklingError: Could not pickle the task to send it to the workers.

## === cell 5
X_full_flat = np.vstack([X_train_flat, X_val_flat])
y_full = np.concatenate([train_labels, val_labels])

scaler.fit(X_full_flat)

clf_final = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=600,
    n_jobs=-1,
    random_state=9,
    class_weight="balanced",
)
clf_final.fit(X_full_flat, y_full)

test_files = glob.glob(os.path.join(test_path, "*.wav"))
batch_size = 10000  # larger batch reduces per‑batch overhead

pred_str = []
fname_list = []

inv_map = {v: k for k, v in class_map.items()}
UNKNOWN_THRESH = 0.15

for start in range(0, len(test_files), batch_size):
    batch_files = test_files[start : start + batch_size]
    X_batch_list = Parallel(n_jobs=-1, backend="loky")(
        delayed(preprocess_data)(f, None, augment_prob=0.0) for f in batch_files
    )
    X_batch = np.stack(X_batch_list)
    X_batch_flat = X_batch.reshape(X_batch.shape[0], -1)
    X_batch_flat = scaler.transform(X_batch_flat)

    proba = clf_final.predict_proba(X_batch_flat)
    max_probs = proba.max(axis=1)
    pred_labels_idx = np.argmax(proba, axis=1)

    for idx, conf, fname in zip(pred_labels_idx, max_probs, batch_files):
        if conf < UNKNOWN_THRESH:
            pred_str.append("unknown")
        else:
            pred_str.append(inv_map[idx])
        fname_list.append(os.path.basename(fname))

submission = pd.DataFrame({"fname": fname_list, "label": pred_str})

submission_dir = os.path.join(root_path, "working")
os.makedirs(submission_dir, exist_ok=True)
submission_path = os.path.join(submission_dir, "submission.csv")
submission.to_csv(submission_path, index=False)
print("Submission written to:", submission_path)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2767015164.py in <cell line: 0>()
----> 1 X_full_flat = np.vstack([X_train_flat, X_val_flat])
      2 y_full = np.concatenate([train_labels, val_labels])
      3 
      4 # Re‑fit scaler on the full dataset (same as original logic)
      5 scaler.fit(X_full_flat)

NameError: name 'X_train_flat' is not defined
