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

0.7559372114496771

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0 --tf_xla_cpu_global_jit=0")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

print("TF version:", tf.__version__)
print("Num GPUs available:", len(tf.config.list_physical_devices("GPU")))

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.experimental.enable_tensor_float_32_execution(False)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def auto_select_accelerator():
    """
    TPU if available, else default strategy.
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except Exception:
        strategy = tf.distribute.get_strategy()
        print("Running on default strategy (CPU/GPU).")
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy




## === cell 2
IMSIZES = (224, 240, 260, 300, 380, 456, 528, 600)
im_size = IMSIZES[7]

load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
train_csv_path = os.path.join(load_dir, "train.csv")
df = pd.read_csv(train_csv_path)

df["labels"] = df["labels"].astype(str)
class_name = df.labels.unique().tolist()
n_labels = len(class_name)

print("Num unique label-strings (as used by this model):", n_labels)
print("Example label-strings:", class_name[:10])



## === cell 3
strategy = auto_select_accelerator()
BATCH_SIZE = strategy.num_replicas_in_sync * 16

test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
sample_sub_path = os.path.join(load_dir, "sample_submission.csv")

sample_sub = pd.read_csv(sample_sub_path)
test_df = sample_sub[["image"]].copy()

try:
    test_images_in_dir = set(tf.io.gfile.listdir(test_dir))
except Exception:
    test_images_in_dir = None

if test_images_in_dir is not None:
    missing = [img for img in test_df["image"].values if img not in test_images_in_dir]
    if missing:
        first_missing = missing[0]
        print(
            "Warning: at least 1 image from sample_submission not found in test_images folder "
            f"(e.g., {first_missing}). Falling back to folder listing."
        )
        test_df = pd.DataFrame({"image": sorted(test_images_in_dir)})
else:
    first_missing = None
    for img in test_df["image"].tolist():
        if not tf.io.gfile.exists(os.path.join(test_dir, img)):
            first_missing = img
            break
    if first_missing is not None:
        print(
            "Warning: at least 1 image from sample_submission not found in test_images folder "
            f"(e.g., {first_missing}). Falling back to folder listing."
        )
        test_df = pd.DataFrame({"image": sorted(tf.io.gfile.listdir(test_dir))})

print("Test images:", len(test_df))



## === cell 4
AUTOTUNE = tf.data.AUTOTUNE


def _read_decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [im_size, im_size], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    return img


def _preprocess_base(img):
    img = img * (1.0 / 255.0)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def _apply_tta_vec(img, idx, tta_round):
    seed_h = tf.stack([tf.cast(tta_round, tf.int32), tf.cast(idx, tf.int32)])
    seed_v = tf.stack([tf.cast(tta_round + 1337, tf.int32), tf.cast(idx, tf.int32)])
    img = tf.image.stateless_random_flip_left_right(img, seed=seed_h)
    img = tf.image.stateless_random_flip_up_down(img, seed=seed_v)
    return img


def make_test_tta_ds(images, tta=5):
    images_t = tf.convert_to_tensor(images, dtype=tf.string)
    n = tf.shape(images_t)[0]
    paths = tf.strings.join([tf.constant(test_dir, tf.string), images_t])

    base_ds = tf.data.Dataset.from_tensor_slices((tf.range(n, dtype=tf.int32), paths))

    opts = tf.data.Options()
    opts.experimental_deterministic = False
    try:
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.autotune = True
        opts.experimental_optimization.map_vectorization.enabled = True
        opts.experimental_optimization.map_parallelization = True
    except Exception:
        pass
    base_ds = base_ds.with_options(opts)

    def _load_one(idx, p):
        img = _read_decode_resize(p)
        img = _preprocess_base(img)
        return idx, img

    base_ds = base_ds.map(_load_one, num_parallel_calls=AUTOTUNE)
    base_ds = base_ds.cache()

    idx_list = []
    img_list = []
    for idx_v, img_v in base_ds:
        idx_list.append(idx_v)
        img_list.append(img_v)
    imgs = tf.stack(img_list, axis=0)  # [n, H, W, 3]

    idx_ds = tf.data.Dataset.from_tensor_slices(tf.range(n, dtype=tf.int32))
    round_ds = tf.data.Dataset.from_tensor_slices(tf.range(tta, dtype=tf.int32))
    pair_ds = idx_ds.repeat(tta).zip(round_ds.repeat(n))  # (idx, round) length n*tta

    def _make_one(idx, r):
        img = tf.gather(imgs, idx)
        img = _apply_tta_vec(img, idx, r)
        return img

    ds = pair_ds.map(_make_one, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_images_list = test_df["image"].tolist()
n_test = len(test_images_list)



## === cell 5
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense

with strategy.scope():
    base = tf.keras.applications.EfficientNetB7(
        weights=None,
        include_top=False,
        input_shape=(im_size, im_size, 3),
    )

    model = Sequential(
        [
            base,
            GlobalMaxPooling2D(),
            Dense(n_labels, activation="softmax"),
        ]
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=4e-4),
        metrics=["accuracy"],
    )

model.summary()



## === cell 6
weights_path = "/kaggle/input/modelplant1/bestmodel_tpu_aug.h5"
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print("Loaded weights:", weights_path)
else:
    print(
        f"Warning: weights file not found at {weights_path}. Proceeding with random weights (score will be poor)."
    )



## === cell 7
TTA = 5

tta_ds = make_test_tta_ds(test_images_list, tta=TTA)

n_pred = n_test * TTA

preds_all = model.predict(tta_ds, verbose=0, batch_size=BATCH_SIZE)

preds_all = preds_all[:n_pred]
if preds_all.shape[0] != n_pred:
    raise RuntimeError(
        f"Prediction count mismatch: got {preds_all.shape[0]}, expected {n_pred}"
    )

preds_all = preds_all.reshape((n_test, TTA, n_labels))
pred = preds_all.mean(axis=1)

argpred = np.argmax(pred, axis=1)

test_df = test_df.copy()
test_df["labels"] = [class_name[i] for i in argpred]

submission = test_df[["image", "labels"]]
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2960143239.py in <cell line: 0>()
      1 TTA = 5
      2 
----> 3 tta_ds = make_test_tta_ds(test_images_list, tta=TTA)
      4 
      5 n_pred = n_test * TTA

/tmp/ipykernel_11/3483503661.py in make_test_tta_ds(images, tta)
     72     idx_ds = tf.data.Dataset.from_tensor_slices(tf.range(n, dtype=tf.int32))
     73     round_ds = tf.data.Dataset.from_tensor_slices(tf.range(tta, dtype=tf.int32))
---> 74     pair_ds = idx_ds.repeat(tta).zip(round_ds.repeat(n))  # (idx, round) length n*tta
     75 
     76     def _make_one(idx, r):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in repeat(self, count, name)
   1375     # pylint: disable=g-import-not-at-top,protected-access,redefined-outer-name
   1376     from tensorflow.python.data.ops import repeat_op
-> 1377     return repeat_op._repeat(self, count, name)
   1378     # pylint: enable=g-import-not-at-top,protected-access,redefined-outer-name
   1379 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/repeat_op.py in _repeat(input_dataset, count, name)
     23 
     24 def _repeat(input_dataset, count, name):  # pylint: disable=unused-private-name
---> 25   return _RepeatDataset(input_dataset, count, name)
     26 
     27 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/repeat_op.py in __init__(self, input_dataset, count, name)
     35       self._count = constant_op.constant(-1, dtype=dtypes.int64, name="count")
     36     else:
---> 37       self._count = ops.convert_to_tensor(
     38           count, dtype=dtypes.int64, name="count")
     39     self._name = name

/usr/local/lib/python3.11/dist-packages/tensorflow/python/profiler/trace.py in wrapped(*args, **kwargs)
    181         with Trace(trace_name, **trace_kwargs):
    182           return func(*args, **kwargs)
--> 183       return func(*args, **kwargs)
    184 
    185     return wrapped

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in convert_to_tensor(value, dtype, name, as_ref, preferred_dtype, dtype_hint, ctx, accepted_result_types)
    730   # TODO(b/142518781): Fix all call-sites and remove redundant arg
    731   preferred_dtype = preferred_dtype or dtype_hint
--> 732   return tensor_conversion_registry.convert(
    733       value, dtype, name, as_ref, preferred_dtype, accepted_result_types
    734   )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor_conversion_registry.py in convert(value, dtype, name, as_ref, preferred_dtype, accepted_result_types)
    207   overload = getattr(value, "__tf_tensor__", None)
    208   if overload is not None:
--> 209     return overload(dtype, name)  #  pylint: disable=not-callable
    210 
    211   for base_type, conversion_func in get(type(value)):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in __tf_tensor__(self, dtype, name)
    625                 name=name))
    626       return graph.capture(self, name=name)
--> 627     return super().__tf_tensor__(dtype, name)
    628 
    629   def _capture_as_const(self, name) -> Optional[tensor_lib.Tensor]:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor.py in __tf_tensor__(self, dtype, name)
    759       ) -> "Tensor":
    760     if dtype is not None and not dtype.is_compatible_with(self.dtype):
--> 761       raise ValueError(
    762           _add_error_prefix(
    763               f"Tensor conversion requested dtype {dtype.name} "

ValueError: count: Tensor conversion requested dtype int64 for Tensor with dtype int32: <tf.Tensor: shape=(), dtype=int32, numpy=3727>
