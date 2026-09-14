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

0.7867189737914407

# 6. Current score

0.02225

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0003) has done: 'The fix removes the problematic `tensorflow‑addons` import, consolidates all Keras imports under `tensorflow.keras`, adds a lightweight training pipeline using a pretrained DenseNet121 backbone (so a model is available even when the original checkpoint is missing), and ensures the script reads the data, trains briefly, makes predictions on the test images, and writes a correctly formatted `submission.csv`. All variable names are defined before use, and the code now runs end‑to‑end producing a valid submission file.'
- What this solution (achieved 0.17094) has done: 'I fix the import error by setting the protobuf implementation before importing TensorFlow, convert the diagnosis labels to strings so Keras `flow_from_dataframe` works with `class_mode="categorical"`, add the missing `tqdm` import, and simplify training by removing the mismatched `class_weight` argument. These changes unblock the pipeline, allow proper model training, and enable a valid submission file, moving the score toward the target.'
- What this solution (achieved -0.01645) has done: 'I patch the script to (1) fix the protobuf import error that prevented TensorFlow from loading, (2) remove the invalid “diagnosis” column assignment to the test dataframe, (3) ensure the validation dataframe is correctly created, and (4) modestly increase training epochs to give the model a chance to reach a higher quadratic weighted kappa while keeping the core architecture unchanged. These changes unblock the pipeline, produce a proper submission.csv, and are expected to improve the score toward the target.'
- What this solution (achieved -0.02819) has done: 'I fix the data‑path errors that caused the generators to have zero length by using the absolute Kaggle input directory, make the preprocessing function robust to receive an already‑loaded image, and slightly increase the training epochs (from 5 to 8) to gain a modest improvement in quadratic weighted kappa while keeping the core model unchanged. These changes unblock the pipeline and move the score closer to the target.'
- What this solution (achieved -0.03598) has done: 'The changes add parallel data loading (workers = 4) to the training / validation generators via `model.fit`, and replace the per‑image prediction loop with a batched prediction that loads, preprocesses, and predicts multiple images at once. Both adjustments keep the exact preprocessing, model architecture, and training schedule, but eliminate the costly repeated single‑image I/O and prediction overhead, allowing the whole script to finish well within the 600‑second limit.'
- What this solution (achieved 0.02225) has done: 'The changes speed up data loading by eliminating a redundant second disk read, increase the batch size to halve the number of training steps, and enable parallel data preprocessing with multiple workers. These adjustments keep the model architecture, training regime, and evaluation unchanged, preserving result accuracy while reducing total runtime below the 600‑second limit.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
try:
    import google.protobuf.message_factory as _mf

    if not hasattr(_mf.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, proto):
            return proto.__class__

        _mf.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass  # if protobuf is not available, TensorFlow will raise later

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import gc
from tqdm import tqdm

BASE_DIR = "/kaggle/input/aptos2019-blindness-detection"

IMG_SIZE = 224
BATCH_SIZE = 32  # increased batch size to reduce number of steps per epoch
EPOCHS = 12  # increased modestly for better performance
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)




## === cell 1
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        if img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0] == 0:
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        img = np.stack([img1, img2, img3], axis=-1)
        return img
    return img


def load_ben_color(image, sigmaX=10):
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32") / 255.0




## === cell 2
train_path = os.path.join(BASE_DIR, "train.csv")
test_path = os.path.join(BASE_DIR, "test.csv")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df["diagnosis"] = train_df["diagnosis"].astype(str)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df["diagnosis"],
    random_state=SEED,
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

train_df["filename"] = train_df["id_code"] + ".png"
val_df["filename"] = val_df["id_code"] + ".png"




## === cell 3
def preprocess_path(input_img):
    """
    ImageDataGenerator already loads the image as a NumPy array.
    We simply apply the existing colour‑normalisation pipeline.
    """
    return load_ben_color(input_img)


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_path,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    zoom_range=0.2,
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_path)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=os.path.join(BASE_DIR, "train_images"),
    x_col="filename",
    y_col="diagnosis",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    interpolation="bilinear",
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=os.path.join(BASE_DIR, "train_images"),
    x_col="filename",
    y_col="diagnosis",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED,
    interpolation="bilinear",
)




## === cell 4
base_model = DenseNet121(
    weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

for layer in base_model.layers[:-20]:
    layer.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.5)(x)
output = layers.Dense(5, activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)




## === cell 5
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    verbose=1,
    workers=4,  # parallel data loading
    use_multiprocessing=True,
)

train_generator.close()
val_generator.close()
gc.collect()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1848566153.py in <cell line: 0>()
----> 1 model.fit(
      2     train_generator,
      3     epochs=EPOCHS,
      4     validation_data=val_generator,
      5     verbose=1,

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
test_ids = test_df["id_code"].values
test_images_path = os.path.join(BASE_DIR, "test_images")

test_predictions = np.empty(len(test_ids), dtype="int32")

for start_idx in tqdm(range(0, len(test_ids), BATCH_SIZE), desc="Predicting"):
    end_idx = min(start_idx + BATCH_SIZE, len(test_ids))
    batch_ids = test_ids[start_idx:end_idx]

    batch_imgs = []
    for img_id in batch_ids:
        img_path = os.path.join(test_images_path, f"{img_id}.png")
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype="uint8")
        img = load_ben_color(img)
        batch_imgs.append(img)

    batch_array = np.stack(batch_imgs, axis=0)  # (batch, H, W, C)
    preds = model.predict(batch_array, verbose=0)
    expected = np.dot(preds, np.arange(5))
    pred_classes = np.rint(expected).astype(int)
    pred_classes = np.clip(pred_classes, 0, 4)
    test_predictions[start_idx:end_idx] = pred_classes




## === cell 7
submission_path = os.path.join(BASE_DIR, "sample_submission.csv")
submission = pd.read_csv(submission_path)
submission["diagnosis"] = test_predictions
submission.to_csv("submission.csv", index=False)

unique, counts = np.unique(test_predictions, return_counts=True)
print(dict(zip(unique, counts)))
print("Submission file written to submission.csv")
