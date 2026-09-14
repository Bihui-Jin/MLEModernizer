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

0.8004801477377672

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    Dropout,
    GlobalMaxPool2D,
    Conv2D,
    InputLayer,
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/conve01/eff7-e9/epoch-9"
efficientB7 = "../input/efficientb7/effb7"

image_dims = (300, 300, 3)

data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"].fillna("")
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()
NUM_CLASSES = len(dataset_labels)

print("NUM_CLASSES:", NUM_CLASSES)
print("Labels:", dataset_labels)




## === cell 2
def _find_saved_model_dir(path: str):
    """Return a directory containing a valid TF SavedModel (has saved_model.pb or saved_model.pbtxt)."""
    if not path or not os.path.exists(path):
        return None

    def is_savedmodel_dir(d):
        return os.path.isdir(d) and (
            os.path.exists(os.path.join(d, "saved_model.pb"))
            or os.path.exists(os.path.join(d, "saved_model.pbtxt"))
        )

    if is_savedmodel_dir(path):
        return path

    for cand in (
        os.path.join(path, "saved_model"),
        os.path.join(path, "export"),
        os.path.join(path, "1"),
    ):
        if is_savedmodel_dir(cand):
            return cand

    try:
        subdirs = [os.path.join(path, d) for d in os.listdir(path)]
    except Exception:
        subdirs = []
    for d in subdirs:
        if is_savedmodel_dir(d):
            return d

    for root, dirs, files in os.walk(path):
        if "saved_model.pb" in files or "saved_model.pbtxt" in files:
            return root
        rel_depth = os.path.relpath(root, path).count(os.sep)
        if rel_depth >= 1:
            dirs[:] = []
    return None


class MultiLabel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.using_savedmodel = False
        savedmodel_dir = _find_saved_model_dir(efficientB7)

        self.model_backbone = None
        if savedmodel_dir is not None:
            try:
                self.model_backbone = keras.layers.TFSMLayer(
                    savedmodel_dir, call_endpoint="serving_default"
                )
                self.using_savedmodel = True
            except Exception:
                try:
                    self.model_backbone = keras.layers.TFSMLayer(
                        savedmodel_dir, call_endpoint="serve"
                    )
                    self.using_savedmodel = True
                except Exception:
                    self.model_backbone = None
                    self.using_savedmodel = False

        if self.model_backbone is None:
            backbone = keras.applications.EfficientNetB7(
                include_top=False,
                weights="imagenet",
                input_shape=image_dims,
                pooling=None,
            )
            backbone.trainable = False
            self.model_backbone = backbone
            self.using_savedmodel = False

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=image_dims))
        self.model.add(self.model_backbone)
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
        self.model.add(BatchNormalization(momentum=0.7))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
        self.model.add(
            Conv2D(filters=400 * NUM_CLASSES, kernel_size=(1, 1), padding="same")
        )

        self.pred_conv1 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.pred_bn = BatchNormalization()
        self.pred_conv2 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.pred_pool = GlobalMaxPool2D()
        self.pred_dense = Dense(units=1, activation="sigmoid")

    def _apply_pred_head(self, x, training=False):
        x = self.pred_conv1(x)
        x = self.pred_bn(x, training=training)
        x = self.pred_conv2(x)
        x = self.pred_pool(x)
        x = self.pred_dense(x)
        return x

    def call(self, x, training=False, **kwargs):
        y = self.model(x, training=training)

        if isinstance(y, dict):
            if "outputs" in y:
                y = y["outputs"]
            else:
                y = next(iter(y.values()))

        shape = tf.shape(y)
        b, h, w = shape[0], shape[1], shape[2]

        y_per_class = tf.reshape(y, [b, h, w, NUM_CLASSES, 400])  # [B,H,W,C,400]
        y_per_class = tf.transpose(y_per_class, [0, 3, 1, 2, 4])  # [B,C,H,W,400]
        y_per_class = tf.reshape(
            y_per_class, [b * NUM_CLASSES, h, w, 400]
        )  # [B*C,H,W,400]

        k1, b1 = (
            self.pred_conv1.kernel,
            self.pred_conv1.bias,
        )  # [1,1,400*C,1024], [1024]
        k1_g = tf.reshape(k1, [1, 1, 400, NUM_CLASSES, 1024])
        k1_g = tf.transpose(k1_g, [0, 1, 2, 4, 3])  # [1,1,400,1024,C]
        k1_g = tf.reshape(k1_g, [1, 1, 400, 1024 * NUM_CLASSES])

        z = tf.nn.conv2d(
            y_per_class,
            filters=k1_g,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )  # [B*C,H,W,1024*C]
        z = tf.reshape(z, [b * NUM_CLASSES, h, w, 1024, NUM_CLASSES])
        z = tf.linalg.diag_part(z)  # [B*C,H,W,1024]
        z = tf.nn.bias_add(z, b1)
        z = self.pred_bn(z, training=training)
        z = self.pred_conv2(z)
        z = self.pred_pool(z)
        p = self.pred_dense(z)  # [B*C,1]

        p = tf.reshape(p, [b, NUM_CLASSES])  # [B,C]
        return p

    def create_model(self):
        return self.model




## === cell 3
if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)

    try:
        tf.config.experimental.enable_op_determinism(True)
    except Exception:
        pass

    try:
        tf.config.optimizer.set_jit(True)
    except Exception:
        pass

    try:
        tf.config.threading.set_intra_op_parallelism_threads(
            max(1, (os.cpu_count() or 4) // 2)
        )
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except Exception:
        pass

    model = MultiLabel()
    model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])

    weights_path = model_dir
    if os.path.isdir(weights_path):
        candidates = []
        candidates.extend(glob.glob(os.path.join(weights_path, "*.weights.h5")))
        candidates.extend(glob.glob(os.path.join(weights_path, "*.h5")))
        candidates.extend(glob.glob(os.path.join(weights_path, "ckpt*.index")))
        candidates.extend(glob.glob(os.path.join(weights_path, "ckpt*")))
        candidates = [c for c in candidates if os.path.isfile(c)]
        weights_path = max(candidates, key=os.path.getmtime) if candidates else None

    if weights_path is not None and os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print("Loaded weights:", weights_path)
        except Exception:
            ckpt_prefix = weights_path
            if ckpt_prefix.endswith(".index"):
                ckpt_prefix = ckpt_prefix[: -len(".index")]
            model.load_weights(ckpt_prefix)
            print("Loaded checkpoint weights:", ckpt_prefix)
    else:
        print("No external weights found; using backbone default weights (if any).")

    image_files = sorted(
        [
            os.path.join(test_dir, p)
            for p in os.listdir(test_dir)
            if p.lower().endswith(".jpg")
        ]
    )

    AUTOTUNE = tf.data.AUTOTUNE
    BATCH_SIZE = 64

    @tf.function(
        reduce_retracing=True,
        input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)],
    )
    def _load_and_preprocess(path):
        input_img = tf.io.read_file(path)
        image = tf.io.decode_jpeg(input_img, channels=3, dct_method="INTEGER_FAST")
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # [0,1]
        image = tf.image.resize(image, [image_dims[0], image_dims[1]])
        image = image * 255.0
        image.set_shape(image_dims)
        name = tf.strings.split(path, os.sep)[-1]
        return name, image

    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.autotune_buffers = True
        options.experimental_optimization.autotune_cpu_budget = 0
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
    except Exception:
        pass

    files_tf = tf.constant(image_files)
    ds = tf.data.Dataset.from_tensor_slices(files_tf).with_options(options)
    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)

    classes = np.asarray(dataset_labels, dtype=object)

    thr = 0.7

    @tf.function(
        reduce_retracing=True,
        input_signature=[
            tf.TensorSpec(
                shape=[None, image_dims[0], image_dims[1], image_dims[2]],
                dtype=tf.float32,
            )
        ],
        jit_compile=True,
    )
    def _predict(images):
        return model(images, training=False)

    C = NUM_CLASSES
    if C <= 0:
        raise ValueError("NUM_CLASSES computed as 0; check train.csv labels parsing.")

    bit_weights = 1 << np.arange(C, dtype=np.uint16)  # [C]
    lut = np.empty(1 << C, dtype=object)
    for m in range(1 << C):
        if m == 0:
            lut[m] = "healthy"
        else:
            idx = np.flatnonzero((m & bit_weights) != 0)
            lut[m] = " ".join(classes[idx]) if len(idx) else "healthy"

    n = len(image_files)
    out_images = np.empty(n, dtype=object)
    out_labels = np.empty(n, dtype=object)
    pos = 0

    for names_b, images_b in ds:
        probs = _predict(images_b).numpy()  # [B, C] float
        mask = probs > thr  # [B, C] bool
        names_np = names_b.numpy().astype("U")  # bytes -> str

        codes = (mask.astype(np.uint16, copy=False) * bit_weights).sum(axis=1)  # [B]
        rows = lut[codes]  # ndarray object, [B]

        bsz = len(rows)
        out_images[pos : pos + bsz] = names_np
        out_labels[pos : pos + bsz] = rows
        pos += bsz

    if pos != n:
        out_images = out_images[:pos]
        out_labels = out_labels[:pos]

    csv_pd = pd.DataFrame({"image": out_images, "labels": out_labels})
    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)
    print("Wrote submission.csv with shape:", csv_pd.shape, "to", out_path)
    print(csv_pd.head())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3650348393.py in <cell line: 0>()
    126 
    127     for names_b, images_b in ds:
--> 128         probs = _predict(images_b).numpy()  # [B, C] float
    129         mask = probs > thr  # [B, C] bool
    130         names_np = names_b.numpy().astype("U")  # bytes -> str

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_filej3fldfec.py in tf___predict(images)
     13                 except:
     14                     do_return = False
---> 15                     raise
     16                 return fscope.ret(retval_, do_return)
     17         return tf___predict

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    122             raise e.with_traceback(filtered_tb) from None
    123         finally:
--> 124             del filtered_tb
    125 
    126     return error_handler

/tmp/ipykernel_11/4260713701.py in call(self, x, training, **kwargs)
    122         # matching the original head’s behavior on the tiled input.
    123         k1, b1 = (
--> 124             self.pred_conv1.kernel,
    125             self.pred_conv1.bias,
    126         )  # [1,1,400*C,1024], [1024]

AttributeError: in user code:

    File "/tmp/ipykernel_11/3650348393.py", line 107, in _predict  *
        return model(images, training=False)
    File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 122, in error_handler  **
        raise e.with_traceback(filtered_tb) from None
    File "/tmp/ipykernel_11/4260713701.py", line 124, in call
        self.pred_conv1.kernel,

    AttributeError: Exception encountered when calling MultiLabel.call().
    
    You must build the layer before accessing `kernel`.
    
    Arguments received by MultiLabel.call():
      • x=tf.Tensor(shape=(None, 300, 300, 3), dtype=float32)
      • training=False
      • kwargs=<class 'inspect._empty'>
