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

- What this solution (achieved 0.0) has done: 'I remove the failing archive-extraction/install steps and instead point the pipeline directly at the already-unzipped `/kaggle/input/tensorflow-speech-recognition-challenge/{train,test}/audio` folders, which fixes the missing `.7z` path errors and the protobuf-related import crash. I also fix a few Keras API/typo issues that currently prevent training (`monitor='val_acc'` → `val_sparse_categorical_accuracy`, `vebose` typo, deprecated `predict_generator`). Finally, I ensure the submission uses the required `fname` (basename only, matching `sample_submission.csv`) and maps all non-target words into `unknown` plus background noise into `silence`, matching competition semantics while keeping the same MFCC+CNN core logic.'
- What this solution (achieved 0.25027) has done: 'Main bottlenecks are (1) precomputing MFCCs for *all* validation + test files in Python (double work and heavy TF eager `.numpy()` calls) and (2) per-sample audio decode+MFCC inside `Sequence` with lots of Python overhead. I keep the same augmentations, MFCC extraction, model, and training semantics, but move feature computation into a `tf.data` pipeline with parallel map + prefetch so TensorFlow executes decode+MFCC efficiently and overlaps CPU work with GPU/compute. I also avoid the expensive global `_precompute_feature_cache()` passes and replace caching with `Dataset.cache()` for validation/test only (equivalent: deterministic, no augmentation there). Finally, I ensure thread settings are sensible (don’t force 0) to improve throughput while keeping determinism seeds intact.'
- What this solution (achieved 0.34389) has done: 'The timeout is dominated by per-sample Python audio decoding/augmentation inside the training `tf.data` pipeline (`tf.numpy_function` calling `_load_audio_tf` which itself calls `.numpy()`), plus repeated MFCC/standardization work and unnecessary callbacks/I/O each epoch. I keep the exact model, MFCC computation, augmentation logic, and training semantics, but move training audio decode/resample/pad into pure TensorFlow (`_load_audio_tf_tensor`) and run augmentations in a `tf.numpy_function` that operates on an already-loaded waveform (not file I/O), which is provably equivalent but much faster. I also cache and precompute the validation MFCC tensors once (no augmentation there), and remove the unused Keras `Sequence` generators and device listing that add overhead but don’t affect results. Finally, I reduce checkpoint I/O overhead by tracking best weights in-memory (while still saving best weights to disk exactly as before).'
- What this solution (achieved 0.32535) has done: 'I first fix the TensorFlow import crash caused by forcing the pure-Python protobuf implementation, which is triggering the `MessageFactory.GetPrototype` error in this Kaggle environment. Then I keep the same MFCC+CNN training/inference pipeline but correct a key semantic mismatch: the model is currently trained to predict 30+ folder classes while the competition is scored on 12 labels, so I remap training labels into the 12 competition labels (including `silence` using background noise) and train the same architecture on 12 classes. Finally, I keep the submission formatting identical to `sample_submission.csv` and ensure the output file `submission.csv` is always produced.'

# 9. Code solution

## === cell 0
import os, glob, math, gc
from collections import Counter
import threading

import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONHASHSEED"] = "9"

try:
    import tensorflow as tf
except Exception as e:
    import subprocess, sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
    )
    import importlib

    import google.protobuf  # noqa: F401

    importlib.invalidate_caches()
    import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, activations

import matplotlib.pyplot as plt

tf.random.set_seed(9)
np.random.seed(9)

try:
    cpu = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(min(8, cpu))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TF:", tf.__version__)
print("Keras:", keras.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/826188564.py in <cell line: 0>()
     20 try:
---> 21     import tensorflow as tf
     22 except Exception as e:

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     17 # this software without specific prior written permission.
---> 18 #
     19 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     37 protocol message classes from Descriptor objects at runtime.
---> 38 
     39 Recall that a metaclass is the "type" of a class.

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     28 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
---> 29 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
     30 

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

During handling of the above exception, another exception occurred:

ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/826188564.py in <cell line: 0>()
     33 
     34     importlib.invalidate_caches()
---> 35     import tensorflow as tf
     36 
     37 from tensorflow import keras

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     47 _tf2.enable()
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__
     51 from tensorflow._api.v2 import audio

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator
     10 from tensorflow._api.v2.__internal__ import dispatch

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     19 import threading
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export
     23 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     15 """Utility module that contains APIs usable in the generated code."""
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors
     19 from tensorflow.python.autograph.utils.tensor_list import dynamic_list_append

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     17 import contextlib
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops
     21 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     31 
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2
     35 from tensorflow.core.framework import function_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      3 # source: tensorflow/core/framework/attr_value.proto
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor
      7 from google.protobuf import descriptor_pool as _descriptor_pool

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     40 from google.protobuf.internal import enum_type_wrapper
     41 from google.protobuf import message as _message
---> 42 from google.protobuf import reflection as _reflection
     43 from google.protobuf import symbol_database as _symbol_database
     44 

/usr/local/lib/python3.11/dist-packages/google/protobuf/reflection.py in <module>
     49 
     50 
---> 51 from google.protobuf import message_factory
     52 from google.protobuf import symbol_database
     53 

/usr/local/lib/python3.11/dist-packages/google/protobuf/message_factory.py in <module>
     41 
     42 from google.protobuf.internal import api_implementation
---> 43 from google.protobuf import descriptor_pool
     44 from google.protobuf import message
     45 

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor_pool.py in <module>
     61 import warnings
     62 
---> 63 from google.protobuf import descriptor
     64 from google.protobuf import descriptor_database
     65 from google.protobuf import text_encoding

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     45   import binascii
     46   import os
---> 47   from google.protobuf.pyext import _message
     48   _USE_C_DESCRIPTORS = True
     49 

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

## === cell 1
root_path = "/kaggle"
base_input = os.path.join(root_path, "input", "tensorflow-speech-recognition-challenge")

train_path = os.path.join(base_input, "train", "audio")
test_audio_path = os.path.join(base_input, "test", "audio")
sample_sub_path = os.path.join(base_input, "sample_submission.csv")

assert os.path.isdir(train_path), f"Train path not found: {train_path}"
assert os.path.isdir(test_audio_path), f"Test audio path not found: {test_audio_path}"
assert os.path.isfile(
    sample_sub_path
), f"Sample submission not found: {sample_sub_path}"

sample_sub = pd.read_csv(sample_sub_path)
sample_sub.head(), sample_sub.shape



## === cell 2
train_audio_sample = os.path.join(
    train_path, "yes", os.listdir(os.path.join(train_path, "yes"))[0]
)

wav_bytes = tf.io.read_file(train_audio_sample)
audio, sr = tf.audio.decode_wav(wav_bytes, desired_channels=1)
audio = tf.squeeze(audio, axis=-1)
int(audio.shape[0]), int(sr.numpy())




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2926987447.py in <cell line: 0>()
      3 )
      4 
----> 5 wav_bytes = tf.io.read_file(train_audio_sample)
      6 audio, sr = tf.audio.decode_wav(wav_bytes, desired_channels=1)
      7 audio = tf.squeeze(audio, axis=-1)

NameError: name 'tf' is not defined

## === cell 3
def pad_audio_np(samples, L):
    if len(samples) >= L:
        return samples[:L]
    return np.pad(
        samples,
        pad_width=(L - len(samples), 0),
        mode="constant",
        constant_values=(0, 0),
    )


def chop_audio_np(samples, L=16000):
    if len(samples) <= L:
        while True:
            yield pad_audio_np(samples, L)
    while True:
        beg = np.random.randint(0, len(samples) - L)
        yield samples[beg : beg + L]


def choose_background_generator_np(sound, backgrounds, max_alpha=0.7):
    if backgrounds is None:
        return sound
    my_gen = backgrounds[np.random.randint(len(backgrounds))]
    background = next(my_gen) * np.random.uniform(0, max_alpha)
    augmented_data = sound + background
    augmented_data = augmented_data.astype(sound.dtype, copy=False)
    return augmented_data


def random_shift_np(sound, shift_max=0.2, sampling_rate=16000):
    shift = np.random.randint(int(sampling_rate * shift_max))
    out = np.roll(sound, shift)
    if shift > 0:
        out[:shift] = 0
    else:
        out[shift:] = 0
    return out


def random_change_pitch_np(x, sr=16000):
    pitch_factor = np.random.randint(1, 4)  # mimic original selection
    rate = 1.0 + 0.03 * pitch_factor
    idx = (np.arange(len(x)) * rate).astype(np.int64)
    idx = np.clip(idx, 0, len(x) - 1)
    return x[idx]


def random_speed_up_np(x):
    where = ["start", "end"][np.random.randint(0, 2)]
    speed_factor = np.random.uniform(0, 0.5)
    rate = 1.0 + speed_factor
    idx = (np.arange(int(len(x) / rate)) * rate).astype(np.int64)
    idx = np.clip(idx, 0, len(x) - 1)
    up = x[idx]
    up_len = up.shape[0]
    if up_len >= x.shape[0]:
        return up[: x.shape[0]]
    if where == "end":
        up = np.concatenate((up, np.zeros((x.shape[0] - up_len,), dtype=x.dtype)))
    else:
        up = np.concatenate((np.zeros((x.shape[0] - up_len,), dtype=x.dtype), up))
    return up


def get_image_list(train_audio_path):
    classes = os.listdir(train_audio_path)
    classes = [thisclass for thisclass in classes if thisclass != "_background_noise_"]
    index = [i for i, j in enumerate(classes)]
    outlist = []
    labels = []
    for thisindex, thisclass in zip(index, classes):
        filelist = [
            f
            for f in os.listdir(os.path.join(train_audio_path, thisclass))
            if f.endswith(".wav")
        ]
        filelist = [os.path.join(train_audio_path, thisclass, x) for x in filelist]
        outlist.append(filelist)
        labels.append(np.full(len(filelist), fill_value=thisindex))
    return outlist, labels, dict(zip(classes, index))


def split_train_test_stratified_shuffle(images_list, labels, train_size=0.9):
    classes_size = np.array([len(x) for x in images_list])
    classes_vector = [np.arange(x) for x in classes_size]
    total = np.sum(classes_size)
    total_train = [int(train_size * total * x) for x in (classes_size / total)]
    train_index = [
        np.random.choice(x, y, replace=False) for x, y in zip(classes_size, total_train)
    ]
    validation_index = [np.setdiff1d(i, j) for i, j in zip(classes_vector, train_index)]

    train_set = [np.array(x)[idx] for x, idx in zip(images_list, train_index)]
    validation_set = [np.array(x)[idx] for x, idx in zip(images_list, validation_index)]
    train_labels = [np.array(x)[idx] for x, idx in zip(labels, train_index)]
    validation_labels = [np.array(x)[idx] for x, idx in zip(labels, validation_index)]

    train_set = np.array([element for array in train_set for element in array])
    validation_set = np.array(
        [element for array in validation_set for element in array]
    )
    train_labels = np.array([element for array in train_labels for element in array])
    validation_labels = np.array(
        [element for array in validation_labels for element in array]
    )

    train_shuffle = np.random.permutation(len(train_set))
    validation_shuffle = np.random.permutation(len(validation_set))

    train_set = train_set[train_shuffle]
    validation_set = validation_set[validation_shuffle]
    train_labels = train_labels[train_shuffle]
    validation_labels = validation_labels[validation_shuffle]

    return train_set, train_labels, validation_set, validation_labels


_CACHE_LOCK = threading.Lock()

_MFCC_PARAMS = {
    "sr": 16000,
    "frame_length": 640,
    "frame_step": 320,
    "fft_length": 1024,
    "n_mels": 40,
}
_NUM_SPEC_BINS = _MFCC_PARAMS["fft_length"] // 2 + 1

_MEL_W = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=_MFCC_PARAMS["n_mels"],
    num_spectrogram_bins=_NUM_SPEC_BINS,
    sample_rate=_MFCC_PARAMS["sr"],
    lower_edge_hertz=20.0,
    upper_edge_hertz=_MFCC_PARAMS["sr"] / 2.0,
)

_MFCC_FRAMES = int(math.ceil(_MFCC_PARAMS["sr"] / _MFCC_PARAMS["frame_step"]))  # 50


@tf.function(reduce_retracing=True)
def _mfcc_tf(x_1d_float32, n_mfcc: tf.Tensor):
    stft = tf.signal.stft(
        x_1d_float32,
        frame_length=_MFCC_PARAMS["frame_length"],
        frame_step=_MFCC_PARAMS["frame_step"],
        fft_length=_MFCC_PARAMS["fft_length"],
        window_fn=tf.signal.hann_window,
        pad_end=True,
    )
    spectrogram = tf.abs(stft)
    mel = tf.matmul(tf.square(spectrogram), _MEL_W)
    log_mel = tf.math.log(mel + 1e-6)
    mfcc_all = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)
    mfcc = mfcc_all[..., :n_mfcc]
    mfcc = tf.ensure_shape(mfcc, [_MFCC_FRAMES, None])
    return mfcc


@tf.function(reduce_retracing=True)
def _load_audio_tf_tensor(path, target_sr=tf.constant(16000, tf.int32)):
    wav_bytes = tf.io.read_file(path)
    audio, sr = tf.audio.decode_wav(wav_bytes, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)  # float32
    sr = tf.cast(sr, tf.int32)

    def _resample():
        xlen = tf.shape(audio)[0]
        new_len = tf.cast(
            tf.round(
                tf.cast(xlen, tf.float32)
                * (tf.cast(target_sr, tf.float32) / tf.cast(sr, tf.float32))
            ),
            tf.int32,
        )
        idx_f = tf.linspace(0.0, tf.cast(xlen - 1, tf.float32), new_len)
        idx = tf.cast(tf.round(idx_f), tf.int32)
        idx = tf.clip_by_value(idx, 0, xlen - 1)
        return tf.gather(audio, idx)

    audio = tf.cond(tf.not_equal(sr, target_sr), _resample, lambda: audio)

    L = target_sr
    xlen = tf.shape(audio)[0]
    audio = tf.cond(
        xlen >= L,
        lambda: audio[:L],
        lambda: tf.pad(audio, paddings=[[L - xlen, 0]], constant_values=0.0),
    )
    audio = tf.ensure_shape(audio, [None])
    return audio


def _load_audio_tf(path, target_sr=16000):
    wav_bytes = tf.io.read_file(path)
    audio, sr = tf.audio.decode_wav(wav_bytes, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)  # float32 [-1,1]
    sr_val = int(sr.numpy())
    x = audio.numpy()
    if sr_val != target_sr:
        xlen = len(x)
        new_len = int(round(xlen * (target_sr / sr_val)))
        idx = (np.linspace(0, xlen - 1, new_len)).astype(np.int64)
        x = x[idx]
    x = pad_audio_np(x, target_sr).astype(np.float32, copy=False)
    return x


def build_model(n_classes, input_shape):
    model_input = keras.Input(shape=input_shape)
    img_1 = layers.Convolution2D(
        filters=32, kernel_size=(3, 3), padding="same", activation=activations.relu
    )(model_input)
    img_1 = layers.MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = layers.Convolution2D(
        filters=64, kernel_size=(3, 3), padding="same", activation=activations.relu
    )(img_1)
    img_1 = layers.MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = layers.Convolution2D(
        filters=128, kernel_size=(3, 3), padding="same", activation=activations.relu
    )(img_1)
    img_1 = layers.MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = layers.Convolution2D(
        filters=256, kernel_size=(3, 3), padding="same", activation=activations.relu
    )(img_1)
    img_1 = layers.MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = layers.Dropout(rate=0.25)(img_1)
    img_1 = layers.Flatten()(img_1)
    img_1 = layers.Dense(128, activation=activations.relu)(img_1)
    img_1 = layers.Dropout(rate=0.5)(img_1)
    model_output = layers.Dense(n_classes, activation=activations.softmax)(img_1)
    model = keras.Model(model_input, model_output)
    return model




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2672926995.py in <cell line: 0>()
    128 _NUM_SPEC_BINS = _MFCC_PARAMS["fft_length"] // 2 + 1
    129 
--> 130 _MEL_W = tf.signal.linear_to_mel_weight_matrix(
    131     num_mel_bins=_MFCC_PARAMS["n_mels"],
    132     num_spectrogram_bins=_NUM_SPEC_BINS,

NameError: name 'tf' is not defined

## === cell 4
wavfiles = glob.glob(os.path.join(train_path, "_background_noise_", "*wav"))
wavfiles_np = [_load_audio_tf(elem, target_sr=16000) for elem in wavfiles]
background_generator = [chop_audio_np(x) for x in wavfiles_np]
len(background_generator), [len(w) for w in wavfiles_np[:2]]



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2821906221.py in <cell line: 0>()
      1 wavfiles = glob.glob(os.path.join(train_path, "_background_noise_", "*wav"))
----> 2 wavfiles_np = [_load_audio_tf(elem, target_sr=16000) for elem in wavfiles]
      3 background_generator = [chop_audio_np(x) for x in wavfiles_np]
      4 len(background_generator), [len(w) for w in wavfiles_np[:2]]
      5 

/tmp/ipykernel_55/2821906221.py in <listcomp>(.0)
      1 wavfiles = glob.glob(os.path.join(train_path, "_background_noise_", "*wav"))
----> 2 wavfiles_np = [_load_audio_tf(elem, target_sr=16000) for elem in wavfiles]
      3 background_generator = [chop_audio_np(x) for x in wavfiles_np]
      4 len(background_generator), [len(w) for w in wavfiles_np[:2]]
      5 

NameError: name '_load_audio_tf' is not defined

## === cell 5
images_list, labels, classes_map = get_image_list(train_path)

train_set, train_labels, validation_set, validation_labels = (
    split_train_test_stratified_shuffle(images_list, labels)
)

inv_map = {v: k for k, v in classes_map.items()}
len(classes_map), list(classes_map.items())[:5]



## === cell 6
target_commands = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
]
comp_labels = target_commands + ["silence", "unknown"]
comp_label_to_idx = {lab: i for i, lab in enumerate(comp_labels)}

folder_to_comp = {}
for folder in classes_map.keys():
    if folder in target_commands:
        folder_to_comp[folder] = folder
    elif folder == "_background_noise_":
        folder_to_comp[folder] = "silence"
    else:
        folder_to_comp[folder] = "unknown"


def _paths_to_comp_y(paths):
    out = np.empty((len(paths),), dtype=np.int64)
    for i, p in enumerate(paths):
        folder = os.path.basename(os.path.dirname(p))
        comp = folder_to_comp.get(folder, "unknown")
        out[i] = comp_label_to_idx[comp]
    return out


train_comp_y = _paths_to_comp_y(train_set)
val_comp_y = _paths_to_comp_y(validation_set)

rng = np.random.RandomState(9)
n_silence = int(0.10 * len(train_set))
if len(wavfiles) > 0 and n_silence > 0:
    silence_paths = rng.choice(
        np.array(wavfiles, dtype=object), size=n_silence, replace=True
    )
    train_set = np.concatenate([train_set, silence_paths])
    train_comp_y = np.concatenate(
        [
            train_comp_y,
            np.full((n_silence,), comp_label_to_idx["silence"], dtype=np.int64),
        ]
    )

perm = rng.permutation(len(train_set))
train_set = train_set[perm]
train_comp_y = train_comp_y[perm]

Counter(train_comp_y.tolist()), Counter(val_comp_y.tolist())



## === cell 7
_probe_audio = _load_audio_tf_tensor(tf.constant(train_set[0].encode("utf-8")))
_probe_mfcc = _mfcc_tf(_probe_audio, tf.constant(40, tf.int32))
rows, columns = int(_probe_mfcc.shape[0]), int(_probe_mfcc.shape[1])
del _probe_audio, _probe_mfcc
gc.collect()

batch_size = 100
epochs = 50
base_path = root_path + "/working/models"
os.makedirs(base_path, exist_ok=True)

train_size = train_set.shape[0]
validation_size = validation_set.shape[0]
steps_per_epoch = max(1, train_size // batch_size)

lr = 1e-3

tensorboard_dir = base_path + "/logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)

checkpoint_filepath = os.path.join(base_path, "cp-{epoch:04d}.weights.h5")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_best_only=True,
    save_weights_only=True,
    monitor="val_sparse_categorical_accuracy",
    mode="max",
    verbose=1,
)

BEST_CKPT_PATH = {"path": None, "best": -np.inf}


class BestCkptTracker(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val = logs.get("val_sparse_categorical_accuracy", None)
        if val is None:
            return
        if float(val) > float(BEST_CKPT_PATH["best"]):
            BEST_CKPT_PATH["best"] = float(val)
            BEST_CKPT_PATH["path"] = checkpoint_filepath.format(epoch=epoch + 1)


reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=3, min_lr=1e-5, verbose=1
)

earlystop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", min_delta=1e-3, patience=5, verbose=1
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

model = build_model(len(comp_labels), (rows, columns, 1))
model.compile(optimizer=optimizer, loss=loss_fn, metrics=[acc_metric])
model.summary()

(rows, columns, len(comp_labels), train_size, validation_size, steps_per_epoch)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3790560037.py in <cell line: 0>()
----> 1 _probe_audio = _load_audio_tf_tensor(tf.constant(train_set[0].encode("utf-8")))
      2 _probe_mfcc = _mfcc_tf(_probe_audio, tf.constant(40, tf.int32))
      3 rows, columns = int(_probe_mfcc.shape[0]), int(_probe_mfcc.shape[1])
      4 del _probe_audio, _probe_mfcc
      5 gc.collect()

NameError: name '_load_audio_tf_tensor' is not defined

## === cell 8
AUTO = tf.data.AUTOTUNE


@tf.function(reduce_retracing=True)
def _standardize_like_sklearn_tf(X, eps=tf.constant(1e-12, tf.float32)):
    mu = tf.reduce_mean(X, axis=0, keepdims=True)
    var = tf.reduce_mean(tf.square(X - mu), axis=0, keepdims=True)
    scale = tf.sqrt(var)
    scale = tf.where(scale < eps, tf.ones_like(scale), scale)
    return (X - mu) / scale


def _augment_audio_from_wave_py(x_np):
    x = x_np.astype(np.float32, copy=False)  # length 16000
    threshold = 0.7
    if np.random.uniform(0, 1) > threshold:
        x = choose_background_generator_np(x, background_generator)
    if np.random.uniform(0, 1) > threshold:
        x = random_shift_np(x, sampling_rate=16000)
    if np.random.uniform(0, 1) > threshold:
        x = random_change_pitch_np(x, sr=16000)
    if np.random.uniform(0, 1) > threshold:
        x = random_speed_up_np(x)
    return x.astype(np.float32, copy=False)


@tf.function(reduce_retracing=True)
def _mfcc_from_audio_standardized_tf(x_1d, label, n_mfcc=tf.constant(40, tf.int32)):
    mfcc = _mfcc_tf(x_1d, n_mfcc)
    mfcc = _standardize_like_sklearn_tf(mfcc)
    mfcc = tf.expand_dims(mfcc, axis=-1)
    mfcc = tf.ensure_shape(mfcc, [rows, columns, 1])
    return mfcc, tf.cast(label, tf.int64)


@tf.function(reduce_retracing=True)
def _mfcc_from_path_standardized(path, label, n_mfcc=tf.constant(40, tf.int32)):
    x = _load_audio_tf_tensor(path, target_sr=tf.constant(_MFCC_PARAMS["sr"], tf.int32))
    mfcc = _mfcc_tf(x, n_mfcc)
    mfcc = _standardize_like_sklearn_tf(mfcc)
    mfcc = tf.expand_dims(mfcc, axis=-1)  # (time, n_mfcc, 1)
    mfcc = tf.ensure_shape(mfcc, [rows, columns, 1])
    label = tf.cast(label, tf.int64)
    return mfcc, label


def make_train_dataset(paths, labels, batch_size):
    paths_b = np.asarray(paths, dtype=np.bytes_)
    ds = tf.data.Dataset.from_tensor_slices((paths_b, labels.astype(np.int64)))
    ds = ds.shuffle(buffer_size=min(len(paths_b), 20000), reshuffle_each_iteration=True)

    def _map_train(p, y):
        x = _load_audio_tf_tensor(p, target_sr=tf.constant(16000, tf.int32))
        x = tf.ensure_shape(x, [16000])
        x = tf.numpy_function(_augment_audio_from_wave_py, [x], Tout=tf.float32)
        x = tf.ensure_shape(x, [16000])
        return _mfcc_from_audio_standardized_tf(x, y)

    ds = ds.map(_map_train, num_parallel_calls=AUTO)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


def make_eval_dataset(paths, labels, batch_size, cache=False):
    paths_b = np.asarray(paths, dtype=np.bytes_)
    ds = tf.data.Dataset.from_tensor_slices((paths_b, labels.astype(np.int64)))
    ds = ds.map(_mfcc_from_path_standardized, num_parallel_calls=AUTO)
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


opt = tf.data.Options()
opt.deterministic = True

train_ds = make_train_dataset(
    train_set, train_comp_y, batch_size=batch_size
).with_options(opt)
val_ds = make_eval_dataset(
    validation_set, val_comp_y, batch_size=batch_size, cache=True
).with_options(opt)

history = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_ds,
    validation_steps=max(1, validation_size // batch_size),
    callbacks=[
        earlystop_callback,
        reduce_lr_callback,
        checkpoint_callback,
        BestCkptTracker(),
        tensorboard_callback,
    ],
    verbose=1,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2270845027.py in <cell line: 0>()
----> 1 AUTO = tf.data.AUTOTUNE
      2 
      3 
      4 @tf.function(reduce_retracing=True)
      5 def _standardize_like_sklearn_tf(X, eps=tf.constant(1e-12, tf.float32)):

NameError: name 'tf' is not defined

## === cell 9
plt.plot(history.history.get("val_loss", []), label="val_loss")
plt.plot(history.history.get("loss", []), label="loss")
plt.title("Loss history")
plt.ylabel("Loss value")
plt.xlabel("No. epoch")
plt.legend()
plt.show()

plt.plot(history.history.get("sparse_categorical_accuracy", []), label="accuracy")
plt.plot(
    history.history.get("val_sparse_categorical_accuracy", []), label="val_accuracy"
)
plt.title("Accuracy history")
plt.ylabel("Accuracy value")
plt.xlabel("No. epoch")
plt.legend()
plt.show()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1136939308.py in <cell line: 0>()
----> 1 plt.plot(history.history.get("val_loss", []), label="val_loss")
      2 plt.plot(history.history.get("loss", []), label="loss")
      3 plt.title("Loss history")
      4 plt.ylabel("Loss value")
      5 plt.xlabel("No. epoch")

NameError: name 'plt' is not defined

## === cell 10
best_ckpt = BEST_CKPT_PATH.get("path")
if best_ckpt and os.path.exists(best_ckpt):
    model.load_weights(best_ckpt)
    print("Loaded best weights:", best_ckpt)
else:
    ckpts = sorted(glob.glob(os.path.join(base_path, "cp-*.weights.h5")))
    if ckpts:
        model.load_weights(ckpts[-1])
        print("Loaded last weights:", ckpts[-1])
    else:
        print("No checkpoint found; using current in-memory weights.")

test_df = pd.read_csv(sample_sub_path)
test_fnames = test_df["fname"].tolist()
test_full_paths = np.array([os.path.join(test_audio_path, f) for f in test_fnames])
dummy_labels = np.zeros(len(test_full_paths), dtype=np.int64)

test_ds = make_eval_dataset(
    test_full_paths, dummy_labels, batch_size=batch_size, cache=False
).with_options(opt)
test_steps = int(np.ceil(len(test_full_paths) / batch_size))

y_pred = model.predict(test_ds, steps=test_steps, verbose=1)
y_labs = np.argmax(y_pred, axis=1)
y_labs[:10], y_pred.shape



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/228391438.py in <cell line: 0>()
----> 1 best_ckpt = BEST_CKPT_PATH.get("path")
      2 if best_ckpt and os.path.exists(best_ckpt):
      3     model.load_weights(best_ckpt)
      4     print("Loaded best weights:", best_ckpt)
      5 else:

NameError: name 'BEST_CKPT_PATH' is not defined

## === cell 11
pred_comp = [comp_labels[int(i)] for i in y_labs]

submission = pd.DataFrame({"fname": test_fnames, "label": pred_comp})
submission.to_csv("submission.csv", index=False)

submission.head(), submission.shape



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1073448791.py in <cell line: 0>()
----> 1 pred_comp = [comp_labels[int(i)] for i in y_labs]
      2 
      3 submission = pd.DataFrame({"fname": test_fnames, "label": pred_comp})
      4 submission.to_csv("submission.csv", index=False)
      5 

NameError: name 'y_labs' is not defined

## === cell 12
check = pd.read_csv("submission.csv")
assert list(check.columns) == ["fname", "label"]
assert len(check) == len(sample_sub)
check["label"].value_counts().head()

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2353171461.py in <cell line: 0>()
----> 1 check = pd.read_csv("submission.csv")
      2 assert list(check.columns) == ["fname", "label"]
      3 assert len(check) == len(sample_sub)
      4 check["label"].value_counts().head()

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

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'
