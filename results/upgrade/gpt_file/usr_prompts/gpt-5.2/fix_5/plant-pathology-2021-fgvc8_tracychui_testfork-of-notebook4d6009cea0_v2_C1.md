# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import glob
import gc
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow.keras.applications.resnet50 as resnet

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

K.set_image_data_format("channels_last")
print("keras:", keras.__version__, "tf:", tf.__version__)

SEED = 123
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## === cell 1
IMG_WIDTH = 300
IMG_HEIGHT = 300
NR_CHANNELS = 3

TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"
TRAIN_CSV = "../input/plant-pathology-2021-fgvc8/train.csv"
SAMPLE_SUB = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"




## === cell 2
imglist_test = sorted(glob.glob(os.path.join(TEST_DIR, "*.jpg")))
print("n_test_images:", len(imglist_test))
assert len(imglist_test) > 0, "No test images found - check TEST_DIR path."




## === cell 3
training_csv = pd.read_csv(TRAIN_CSV)
training_class = np.array([])
for labels in pd.unique(training_csv["labels"]):
    training_class = np.append(training_class, labels.split())
tagnames = np.unique(training_class)
print("n_train:", len(training_csv), "n_tags:", len(tagnames))
print("tags:", tagnames)




## === cell 4
def labels_to_multihot(labels_series, tagnames):
    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer(classes=list(tagnames))
    split = labels_series.fillna("").astype(str).str.split().tolist()
    Y = mlb.fit_transform(split).astype(np.int8, copy=False)
    return Y


Y = labels_to_multihot(training_csv["labels"], tagnames)




## === cell 5
train_paths = [os.path.join(TRAIN_DIR, fn) for fn in training_csv["image"].tolist()]

need_check = True
try:
    if os.path.isdir(TRAIN_DIR) and len(train_paths) == len(training_csv):
        need_check = False
except Exception:
    need_check = True

if need_check:
    exists_mask = np.fromiter(
        (os.path.exists(p) for p in train_paths), dtype=bool, count=len(train_paths)
    )
    if not np.all(exists_mask):
        missing = int(np.sum(~exists_mask))
        print(f"Warning: {missing} training images missing; filtering them out.")
        training_csv = training_csv.loc[exists_mask].reset_index(drop=True)
        train_paths = [p for p, ok in zip(train_paths, exists_mask) if ok]
        Y = Y[exists_mask]
else:
    pass

print("n_train_after_filter:", len(train_paths), "Y shape:", Y.shape)




## === cell 6
base = resnet.ResNet50(
    weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
inp = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base(inp, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
feat_model = keras.Model(inp, x)
feat_dim = feat_model.output_shape[-1]
print("feature dim:", feat_dim)




## === cell 7
def _decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # uint8
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
    img = tf.cast(img, tf.float32)
    img = resnet.preprocess_input(img)
    return img


@tf.function(jit_compile=True)
def _featurize_batch(batch_imgs):
    return feat_model(batch_imgs, training=False)


def extract_features(paths, batch_size=32):
    paths_tf = tf.constant(paths)
    ds = tf.data.Dataset.from_tensor_slices(paths_tf)

    opts = tf.data.Options()
    try:
        opts.experimental_deterministic = True
    except Exception:
        pass
    ds = ds.with_options(opts)

    ds = ds.map(_decode_resize_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    feats = np.empty((len(paths), feat_dim), dtype=np.float32)
    start = 0

    for batch in ds:
        f = _featurize_batch(batch)
        f_np = f.numpy()
        bs = f_np.shape[0]
        feats[start : start + bs] = f_np
        start += bs

    return feats




## === cell 8
Xf_train = extract_features(train_paths, batch_size=64)
print("Xf_train:", Xf_train.shape)




## === cell 9
Xf_test = extract_features(imglist_test, batch_size=64)
print("Xf_test:", Xf_test.shape)

gc.collect()




## === cell 10
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
trainXn = scaler.fit_transform(Xf_train)
testXn_test = scaler.transform(Xf_test)
print("trainXn:", trainXn.shape, "testXn_test:", testXn_test.shape)




## === cell 11
try:
    from joblib import Parallel, delayed

    _HAS_JOBLIB = True
except Exception as e:
    print("joblib not available; falling back to sequential training. Error:", repr(e))
    _HAS_JOBLIB = False


def _fit_one_tag(i, t):
    y = Y[:, i]
    if y.sum() == 0 or y.sum() == len(y):
        return t, None
    lr = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=1,
    )
    lr.fit(trainXn, y)
    return t, lr


if _HAS_JOBLIB:
    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_fit_one_tag)(i, t) for i, t in enumerate(tagnames)
    )
else:
    results = [_fit_one_tag(i, t) for i, t in enumerate(tagnames)]

tagmodels1 = dict(results)

print(
    "trained tag models:",
    sum(m is not None for m in tagmodels1.values()),
    "/",
    len(tagnames),
)




## === cell 12
testKaggle_ppredscore1 = np.zeros((len(testXn_test), len(tagnames)), dtype=np.float32)
for i, t in enumerate(tagnames):
    model = tagmodels1[t]
    if model is None:
        y = Y[:, i]
        const = 1.0 if y.sum() == len(y) else 0.0
        testKaggle_ppredscore1[:, i] = const
    else:
        testKaggle_ppredscore1[:, i] = model.predict_proba(testXn_test)[:, 1]

print("pred score matrix:", testKaggle_ppredscore1.shape)




## === cell 13
def class2tags(classes, tagnames):
    tagnames = np.asarray(tagnames)
    return [" ".join(tagnames[row]) for row in classes]




## === cell 14
test_predclass = testKaggle_ppredscore1 > 0.5
test_predtags = class2tags(test_predclass, tagnames)

for i in range(len(test_predtags)):
    if test_predtags[i] == "":
        test_predtags[i] = "healthy"

print("example preds:", test_predtags[:5])




## === cell 15
df1 = pd.DataFrame({"image": [os.path.basename(p) for p in imglist_test]})
df2 = pd.DataFrame({"labels": test_predtags})
submission = pd.concat([df1, df2], axis=1)

sample = pd.read_csv(SAMPLE_SUB)
assert list(sample.columns) == ["image", "labels"]
assert list(submission.columns) == ["image", "labels"]
assert len(submission) == len(
    sample
), f"Submission rows {len(submission)} != sample rows {len(sample)}"

submission.head()




## === cell 16
submission.to_csv("./submission.csv", index=False)
print("Wrote ./submission.csv with shape:", submission.shape)
print(submission.head())
