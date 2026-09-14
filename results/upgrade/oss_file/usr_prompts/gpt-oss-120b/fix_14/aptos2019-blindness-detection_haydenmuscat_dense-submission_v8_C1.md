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

3.7

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

0.8675437886331567

# 6. Current score

-0.18487

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.02359) has done: 'The fixes address the early TensorFlow import error, ensure all required variables are defined before use, replace missing keras imports with the TensorFlow‑Keras equivalents, add safe loading of model weights, correct deprecated NumPy dtype usage, and guarantee that the script writes a proper `submission.csv` file. These changes resolve the runtime crashes while keeping the original model and preprocessing logic unchanged, allowing the code to run end‑to‑end and produce a valid submission.'
- What this solution (achieved 0.03768) has done: 'I fix the runtime error caused by the missing ImageNet weights and incorrect fallback image shape, change the model to use a soft‑max output (so predictions are proper class indices instead of a binary‑threshold sum that produced -1 labels), and replace the custom `label_convert` logic with a simple `argmax`. These minimal adjustments keep the original architecture while ensuring valid class predictions, which should raise the quadratic weighted kappa toward the target.'
- What this solution (achieved -0.02256) has done: 'I added an environment fix for the protobuf error, defined the training image directory, and inserted a lightweight training stage (using a Keras `ImageDataGenerator` that applies the same preprocessing as the test pipeline, freezing the DenseNet backbone and training only the final dense layer for a few epochs). This resolves the import crash, allows the model to learn from the provided labels, and keeps the original architecture unchanged while still producing a proper `submission.csv` file. The changes are minimal and focused on fixing the runtime issue and nudging the quadratic weighted kappa score toward the target.'
- What this solution (achieved 0.46752) has done: 'I move the protobuf‑compatibility flag to the very top, replace the faulty Sequential + DenseNet121 wrapper with a proper functional model, and expose the GAP output as the feature extractor. This resolves the import error, fixes the “layer has never been called” issue, and ensures `feature_extractor` exists for inference, allowing the script to run end‑to‑end and produce a valid `submission.csv`.'
- What this solution (achieved 0.61733) has done: 'I added a protobuf compatibility shim before importing TensorFlow to prevent the “MessageFactory has no attribute GetPrototype” error, and I unfroze the DenseNet backbone (setting it trainable) so the model can fine‑tune during training, which should raise the quadratic weighted kappa toward the target. The rest of the pipeline remains unchanged, and the script now writes a proper `submission.csv`.'
- What this solution (achieved -0.18487) has done: 'The changes focus on eliminating the main bottlenecks: image loading and model training.  
1. Increase the thread pool size in `load_image_array` to use more CPU cores for parallel image preprocessing.  
2. Enable multiprocessing data‑augmentation in `model.fit` by adding `workers` and `use_multiprocessing=True`, which lets TensorFlow apply the `ImageDataGenerator` transforms in parallel.  
3. Set a deterministic NumPy seed to keep results reproducible.  
All modifications keep the original model architecture, loss, and training loop unchanged, only improving runtime efficiency.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
try:
    from google.protobuf import message as _message

    if not hasattr(_message.MessageFactory, "GetPrototype"):
        _message.MessageFactory.GetPrototype = _message.MessageFactory.GetMessageClass
except Exception:
    pass

import gc
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix

import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)  # ensure reproducibility of NumPy‑based ops

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from concurrent.futures import ThreadPoolExecutor
from functools import partial  # added for cleaner parallel mapping

IMG_DIM = 256
BATCH_SIZE = 32
CHANNEL_SIZE = 3
NUM_CLASSES = 5

INPUT_FOLDER = "../input/aptos2019-blindness-detection/"
TRAIN_IMAGES_DIR = os.path.join(INPUT_FOLDER, "train_images/")
TEST_IMAGES_DIR = os.path.join(INPUT_FOLDER, "test_images/")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["id_code"] = test_df["id_code"].apply(lambda x: f"{x}.png")




## === cell 2
def crop(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thresh = 5

    rowMaxes = gray.max(axis=1)
    top = 0
    while top < len(rowMaxes) and rowMaxes[top] < thresh:
        top += 1
    bottom = len(rowMaxes) - 1
    while bottom > 0 and rowMaxes[bottom] < thresh:
        bottom -= 1

    middleRow = gray[int((bottom - top) / 2)]
    left = 0
    while left < len(middleRow) and middleRow[left] < thresh:
        left += 1
    right = len(middleRow) - 1
    while right > 0 and middleRow[right] < thresh:
        right -= 1

    height = bottom - top
    width = right - left
    if height < 100 or width < 100:
        return bgr
    return bgr[top:bottom, left:right]


def colourfulEyes(bgr, weight=4, gamma=15):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    return cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)


def processImageBgrToRgb(bgr):
    modified = crop(bgr)
    modified = cv2.resize(modified, (IMG_DIM, IMG_DIM))
    modified = colourfulEyes(modified)
    return cv2.cvtColor(modified, cv2.COLOR_BGR2RGB)




## === cell 3
def dataGenerator(jitter=0.1):
    datagen = image.ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=bool(jitter > 0.01),
        vertical_flip=bool(jitter > 0.01),
        rotation_range=int(800 * jitter),
        brightness_range=[1 - jitter, 1 + jitter],
        channel_shift_range=int(30 * jitter),
        zoom_range=[(1 - jitter), (1 + jitter / 2)],
        fill_mode="reflect",
    )
    return datagen




## === cell 4
def create_model():
    backbone = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_DIM, IMG_DIM, CHANNEL_SIZE),
    )
    backbone.trainable = True

    x = GlobalAveragePooling2D(name="gap")(backbone.output)
    x = Dropout(0.5, name="dropout")(x)
    output = Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    model = Model(inputs=backbone.input, outputs=output, name="densenet_classifier")
    return model


model = create_model()

weights_path = "../input/densenetmulti/dense-multi-second-0.9038.h5"
if os.path.exists(weights_path):
    model.load_weights(weights_path)
else:
    print("Warning: pretrained weights not found; using ImageNet initialization.")

model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

gc.collect()




## === cell 5
train_df = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
train_df["id_code"] = train_df["id_code"].apply(lambda x: f"{x}.png")
train_df["diagnosis"] = train_df["diagnosis"].astype(str)

train_df, val_df = train_test_split(
    train_df, test_size=0.1, stratify=train_df["diagnosis"], random_state=42
)


def _process_path(filename, img_dir):
    img_path = os.path.join(img_dir, filename)
    bgr = cv2.imread(img_path)
    if bgr is not None:
        return processImageBgrToRgb(bgr)
    else:
        return np.full((IMG_DIM, IMG_DIM, 3), 128.0, dtype=np.float32)


def load_image_array(df, img_dir):
    max_workers = min(16, os.cpu_count() or 1)
    filenames = df["id_code"].tolist()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        func = partial(_process_path, img_dir=img_dir)
        results = list(executor.map(func, filenames))
    arr = np.stack(results).astype(np.float32) / 255.0
    return arr


X_train_raw = load_image_array(train_df, TRAIN_IMAGES_DIR)
X_val_raw = load_image_array(val_df, TRAIN_IMAGES_DIR)

y_train = tf.keras.utils.to_categorical(train_df["diagnosis"].astype(int), NUM_CLASSES)
y_val = tf.keras.utils.to_categorical(val_df["diagnosis"].astype(int), NUM_CLASSES)

train_gen = dataGenerator(jitter=0.1).flow(
    X_train_raw, y_train, batch_size=BATCH_SIZE, shuffle=True
)
val_gen = dataGenerator(jitter=0.0).flow(
    X_val_raw, y_val, batch_size=BATCH_SIZE, shuffle=False
)

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
    ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_accuracy"),
]

model.fit(
    train_gen,
    steps_per_epoch=len(X_train_raw) // BATCH_SIZE,
    validation_data=val_gen,
    validation_steps=len(X_val_raw) // BATCH_SIZE,
    epochs=30,
    callbacks=callbacks,
    verbose=2,
    workers=4,  # parallelize data augmentation
    use_multiprocessing=True,
)

gc.collect()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4213914991.py in <cell line: 0>()
     46 ]
     47 
---> 48 model.fit(
     49     train_gen,
     50     steps_per_epoch=len(X_train_raw) // BATCH_SIZE,

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

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'workers'

## === cell 6
X_test = load_image_array(test_df, TEST_IMAGES_DIR)

preds = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
y_pred_list = np.argmax(preds, axis=1)

submission = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
submission["diagnosis"] = y_pred_list
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
