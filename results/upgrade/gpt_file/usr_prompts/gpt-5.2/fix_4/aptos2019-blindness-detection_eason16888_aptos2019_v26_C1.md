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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.8452750103900721

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.01069) has done: 'I remove the `pip install tensorflow-addons` (it triggers the protobuf `MessageFactory.GetPrototype` crash) and also remove the unused `tensorflow_addons` import so the notebook can import TensorFlow/Keras cleanly. Then I fix the cell ordering/numbering and ensure `tf`, `np`, and `pd` are defined before they’re used, so the model can be built and inference can run. Finally, I fix test-time preprocessing to match training-time scaling (the current code feeds 0–255 into a model that expects 0–1), and I make the script robust to Kaggle’s input path by auto-detecting whether data lives under `/kaggle/input/...` or `/kaggle/data/...`, ensuring a valid `submission.csv` is always produced.'
- What this solution (achieved 0.0) has done: 'You’re hitting the known TensorFlow/protobuf `MessageFactory.GetPrototype` crash at import-time, so the first fix is to force TensorFlow to use the pure-Python protobuf implementation *before* importing `tensorflow`. Next, your model currently uses `weights=None` and often can’t find the external `.h5` weights, which makes predictions essentially random (hence the negative kappa); the minimal, core-logic-preserving fix is to use ImageNet pretrained weights for the EfficientNet backbone when the external weights aren’t available. Finally, I keep your preprocessing consistent and ensure the script always writes `submission.csv` with the required `id_code,diagnosis` columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")



## === cell 1
import gc

import cv2
import numpy as np
import pandas as pd

try:
    import tensorflow as tf  # noqa: F401
except Exception as e:
    try:
        from google.protobuf.internal import api_implementation

        api_implementation.Type()  # touch
        api_implementation._default_implementation_type = "python"
    except Exception:
        pass

    import importlib

    importlib.invalidate_caches()
    import tensorflow as tf  # noqa: F401

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

print("TF version:", tf.__version__)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
"""
    Config
"""
IMG_SIZE = 224
BATCH_SIZE = 16


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def load_ben_color(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


"""
Preprocessing for ImageDataGenerator since ImageDataGenerator reads images in rgb mode, while opencv in bgr
"""


def preprocessing(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32") / 255.0




## === cell 3
def resolve_comp_dir():
    candidates = [
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
        "../input/aptos2019-blindness-detection",
        "../data/aptos2019-blindness-detection",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for c in candidates:
        if os.path.exists(c):
            maybe = os.path.join(c, "aptos2019-blindness-detection")
            if os.path.exists(maybe):
                return maybe
            if os.path.exists(os.path.join(c, "test_images")) and os.path.exists(
                os.path.join(c, "sample_submission.csv")
            ):
                return c
    raise FileNotFoundError(
        "Could not find aptos2019-blindness-detection directory in expected locations."
    )


COMP_DIR = resolve_comp_dir()
TEST_IMG_DIR = os.path.join(COMP_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(COMP_DIR, "sample_submission.csv")

print("COMP_DIR:", COMP_DIR)
print("TEST_IMG_DIR exists:", os.path.exists(TEST_IMG_DIR))
print("SAMPLE_SUB_PATH exists:", os.path.exists(SAMPLE_SUB_PATH))



## === cell 4
base_model = tf.keras.applications.efficientnet.EfficientNetB2(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

flatten_layer = tf.keras.layers.Flatten()
dense_layer_1 = tf.keras.layers.Dense(4096, activation="relu")
Dropout_1 = tf.keras.layers.Dropout(0.6)
dense_layer_2 = tf.keras.layers.Dense(2048, activation="relu")
Dropout_2 = tf.keras.layers.Dropout(0.5)
dense_layer_3 = tf.keras.layers.Dense(1024, activation="relu")
Dropout_3 = tf.keras.layers.Dropout(0.3)
dense_layer_4 = tf.keras.layers.Dense(512, activation="relu")
prediction_layer = tf.keras.layers.Dense(5, activation="softmax")

model = tf.keras.models.Sequential(
    [
        base_model,
        flatten_layer,
        dense_layer_1,
        Dropout_1,
        dense_layer_2,
        Dropout_2,
        dense_layer_3,
        Dropout_3,
        dense_layer_4,
        prediction_layer,
    ]
)

_ = model(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32))
model.summary()



## === cell 5
WEIGHTS_CANDIDATES = [
    "../input/eff-balance-b1-model-224/eff_balance_b1_model_224.h5",
    "/kaggle/input/eff-balance-b1-model-224/eff_balance_b1_model_224.h5",
]
weights_loaded = False
for wpath in WEIGHTS_CANDIDATES:
    if os.path.exists(wpath):
        model.load_weights(wpath)
        weights_loaded = True
        print("Loaded weights:", wpath)
        break

if not weights_loaded:
    print(
        "WARNING: External pretrained weights not found; using ImageNet backbone weights only."
    )



## === cell 6
test_csv = pd.read_csv(SAMPLE_SUB_PATH)
id_code = test_csv["id_code"].astype(str).tolist()

test_prediction = np.empty(len(id_code), dtype=np.int64)

for i in range(len(id_code)):
    img_path = os.path.join(TEST_IMG_DIR, f"{id_code[i]}.png")
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    img = load_ben_color(img).astype("float32") / 255.0
    X = np.expand_dims(img, axis=0)  # (1, H, W, C)
    pred = model.predict(X, verbose=0)
    test_prediction[i] = int(np.argmax(pred, axis=1)[0])

test_csv["diagnosis"] = test_prediction.astype(np.int64)
test_csv.to_csv("submission.csv", index=False)

unique, counts = np.unique(test_prediction, return_counts=True)
print(dict(zip(unique.tolist(), counts.tolist())))
print("Wrote submission.csv with shape:", test_csv.shape)
print(test_csv.head())
gc.collect()

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3350429677.py in <cell line: 0>()
----> 1 test_csv = pd.read_csv(SAMPLE_SUB_PATH)
      2 id_code = test_csv["id_code"].astype(str).tolist()
      3 
      4 test_prediction = np.empty(len(id_code), dtype=np.int64)
      5 

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection/sample_submission.csv'
