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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.1669806094182826

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.22452) has done: 'I remove the `kaggle_datasets` import that’s triggering the `MessageFactory/GetPrototype` protobuf crash in this environment, since it’s not actually used by your pipeline. Then I fix the missing model file issue by switching from loading a non-existent `../input/newresnet50/NewResNet50.h5` to using a built-in Keras pretrained ResNet50 backbone with a simple Dense multi-label head, which preserves the intended “ResNet50 → sigmoid probs → per-class thresholds → space-delimited labels” core logic and ensures the notebook runs end-to-end. I also fix label encoding (the original `get_dummies(train['labels'])` is incorrect for space-delimited multi-labels) and align prediction columns with the correct 6 classes. Finally, I ensure the submission is written as `submission.csv` with exactly `image,labels` columns and no trailing spaces in the label strings.'
- What this solution (achieved 0.22452) has done: 'The timeout is dominated by running ResNet50 inference at 512×512 over ~3.7k test images, plus input pipeline overhead; the current setup also forces the slow Python protobuf implementation. To keep identical model architecture and prediction semantics, I only optimize the runtime plumbing: switch protobuf back to the compiled backend, enable XLA compilation for the prediction graph, and make the tf.data pipeline more efficient with non-deterministic parallel map ordering (while keeping deterministic results via fixed seeds and op determinism), plus caching decoded/resized test images to avoid repeated decode work. I also remove unused variables/cells work that doesn’t affect outputs and keep batch sizing and thresholds unchanged so results stay the same aside from negligible floating-point variation. Paths and submission formatting remain unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import random, math
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

print(tf.__version__)
print(tf.keras.__version__)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

tf.config.experimental.enable_op_determinism()

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF choose
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = "../input/plant-pathology-2021-fgvc8/"
if not os.path.exists(path):
    path = "/kaggle/input/plant-pathology-2021-fgvc8/"

train = pd.read_csv(os.path.join(path, "train.csv"))
test = pd.read_csv(os.path.join(path, "sample_submission.csv"))
sub = pd.read_csv(os.path.join(path, "sample_submission.csv"))

train.head(), test.head()



## === cell 2
AUTO = tf.data.experimental.AUTOTUNE



## === cell 3
pass



## === cell 5
train_paths = []

test_paths = tf.io.gfile.glob(os.path.join(path, "test_images", "*.jpg"))
test_paths = sorted(test_paths)

len(train_paths), len(test_paths), test_paths[:3]



## === cell 6
ALL_CLASSES = [
    "scab",
    "frog_eye_leaf_spot",
    "complex",
    "rust",
    "powdery_mildew",
    "healthy",
]
ALL_CLASSES




## === cell 7
def encode_labels(label_str: str):
    items = str(label_str).split()
    return [1.0 if c in items else 0.0 for c in ALL_CLASSES]


new_train = train[["image"]].copy()
new_train.head()



## === cell 8
new_train



## === cell 9
IMG_SIZE = (512, 512)


@tf.function
def decode_image(filename, label=None):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(
        bits,
        channels=3,
        dct_method="INTEGER_FAST",  # faster CPU decode; minimal numeric differences
        try_recover_truncated=True,
    )
    image = tf.image.convert_image_dtype(image, tf.float32)  # cast/255.0
    image = tf.image.resize(image, IMG_SIZE)
    image.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    if label is None:
        return image
    else:
        return image, label




## === cell 10
test_paths[:5]



## === cell 11
BATCH_SIZE = 64



## === cell 12
opt = tf.data.Options()
opt.experimental_deterministic = (
    False  # allow overlap/reordering inside input pipeline (labels unaffected)
)
opt.experimental_optimization.map_parallelization = True
opt.experimental_optimization.parallel_batch = True
opt.experimental_optimization.apply_default_optimizations = True

cache_path = "/kaggle/working/test_decode_cache"

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(opt)
    .map(decode_image, num_parallel_calls=AUTO, deterministic=False)
    .cache(cache_path)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)



## === cell 13
import numpy as np
import tensorflow as tf
from tensorflow import keras



## === cell 14
from tensorflow.keras.utils import get_custom_objects

get_custom_objects().update({"swish": keras.layers.Activation(tf.nn.swish)})




## === cell 15
class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)




## === cell 16
base = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)
x = base.output
out = Dense(len(ALL_CLASSES), activation="sigmoid")(x)
model = Model(inputs=base.input, outputs=out)



## === cell 17
predict_kwargs = dict(verbose=1)
try:
    predict_kwargs.update(
        dict(workers=os.cpu_count() or 1, use_multiprocessing=False, max_queue_size=16)
    )
except Exception:
    pass

probs = model.predict(test_dataset, **predict_kwargs)
probs.shape



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/703110942.py in <cell line: 0>()
     10     pass
     11 
---> 12 probs = model.predict(test_dataset, **predict_kwargs)
     13 probs.shape
     14 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.predict() got an unexpected keyword argument 'workers'

## === cell 18
probs.shape



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4130349179.py in <cell line: 0>()
----> 1 probs.shape
      2 

NameError: name 'probs' is not defined

## === cell 19
probs[:2]



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2619598287.py in <cell line: 0>()
----> 1 probs[:2]
      2 

NameError: name 'probs' is not defined

## === cell 20
temp_probs = probs



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1438685205.py in <cell line: 0>()
----> 1 temp_probs = probs
      2 

NameError: name 'probs' is not defined

## === cell 21
temp_probs[:2]



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/442628044.py in <cell line: 0>()
----> 1 temp_probs[:2]
      2 

NameError: name 'temp_probs' is not defined

## === cell 22
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    5: "healthy",
}

threshold = {0: 0.25, 1: 0.50, 2: 0.30, 3: 0.50, 4: 0.50, 5: 0.50}

test_files = [os.path.basename(p) for p in test_paths]

prob_cols = [name[i] for i in range(len(ALL_CLASSES))]
pred_df = pd.DataFrame(temp_probs, columns=prob_cols)
pred_df["image"] = test_files

pred_df = pred_df.set_index("image")
out_df = sub[["image"]].copy()
out_probs = pred_df.reindex(out_df["image"]).fillna(0.0)

probs_np = out_probs[prob_cols].to_numpy(dtype=np.float32, copy=False)
thr_np = np.array([threshold[i] for i in range(len(ALL_CLASSES))], dtype=np.float32)

mask = probs_np > thr_np[None, :]

cols_local = np.array(prob_cols, dtype=object)
row_idxs, col_idxs = np.nonzero(mask)

pred_string = np.full(mask.shape[0], "healthy", dtype=object)
if row_idxs.size:
    order = np.argsort(row_idxs, kind="mergesort")
    row_sorted = row_idxs[order]
    col_sorted = col_idxs[order]

    starts = np.r_[0, 1 + np.flatnonzero(row_sorted[1:] != row_sorted[:-1])]
    ends = np.r_[starts[1:], row_sorted.size]

    for s, e in zip(starts, ends):
        r = row_sorted[s]
        pred_string[r] = " ".join(cols_local[col_sorted[s:e]].tolist())

out_sub = out_df.copy()
out_sub["labels"] = pred_string.tolist()

out_sub.to_csv("submission.csv", index=False)
out_sub.head()

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4112563657.py in <cell line: 0>()
     13 
     14 prob_cols = [name[i] for i in range(len(ALL_CLASSES))]
---> 15 pred_df = pd.DataFrame(temp_probs, columns=prob_cols)
     16 pred_df["image"] = test_files
     17 

NameError: name 'temp_probs' is not defined
