# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.9

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

0.8016449638822514

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I replace the failed model‑loading step with a small EfficientNet‑B0 model built directly in the script, fix the TensorFlow/Keras imports, and use the modern `model.predict` API. The script now creates a training generator, trains the model for a few epochs, predicts on the test set, converts predictions to class labels, and writes a proper `submission.csv`. All paths are corrected, the custom dropout and kappa loss are retained, and the code runs end‑to‑end producing a valid submission file.'
- What this solution (achieved 0.0) has done: 'The fix adds a protobuf‑compatible environment setting before importing TensorFlow, converts the `diagnosis` labels to strings so Keras can build a categorical generator, and ensures the generators and model are defined after these corrections. This resolves the import error and the `class_mode="categorical"` type error, allowing the script to run end‑to‑end and produce a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'The update parallel‑loads and preprocesses all images using a multiprocessing pool, which removes the original single‑threaded Python loop that dominated the runtime. Because the preprocessing function itself is unchanged, the resulting image tensors are identical, preserving model inputs and final predictions. The rest of the pipeline (data generators, model definition, training, and inference) remains exactly the same, so accuracy and core logic are untouched.'
- What this solution (achieved 0.0) has done: 'I fix the runtime errors by removing the unsupported `workers` argument from the `ImageDataGenerator.flow` calls and by freezing the EfficientNet backbone to speed up training and improve generalization. I also increase the training epochs slightly to help the model reach a higher quadratic weighted kappa. These minimal changes keep the original pipeline intact while allowing the script to run end‑to‑end and produce a valid `submission.csv` that should achieve a score closer to the target.'
- What this solution (achieved 0.0) has done: 'The changes pre‑compute EfficientNetB0 embeddings once (the backbone is frozen) and then train only the lightweight dropout‑dense head on these fixed features. This removes the expensive forward‑pass through EfficientNet on every epoch, cutting training time dramatically while keeping the exact same architecture, weights, dropout behavior, and loss/metric logic. The preprocessing, data splits, and final prediction steps remain unchanged, preserving result accuracy.'
- What this solution (achieved 0.0) has done: 'I increased the training effort to move the model’s predictions closer to the target quadratic weighted kappa. The head network is now trained for more epochs (20 instead of 5) and class‑weights are computed from the training split and passed to `model.fit` so the under‑represented severity levels influence learning more strongly. These minimal adjustments keep the original architecture and preprocessing unchanged while giving the model a better chance to reach a score near the target.'
- What this solution (achieved 0.0) has done: 'I keep the overall pipeline unchanged but modify the way predictions are turned into class labels. Instead of an argmax (which ignores the ordinal nature of the problem), I compute the expected rating from the soft‑max probabilities and round it to the nearest integer. This simple post‑processing respects the ordering of the classes and typically yields a higher quadratic weighted kappa, moving the score toward the target without altering the model architecture or training logic.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from google.protobuf import message_factory as _mf

    if not hasattr(_mf.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _mf.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass  # If protobuf is not available, let TensorFlow raise its own error later

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
import multiprocessing as mp

tf.random.set_seed(42)
np.random.seed(42)

BASE_PATH = "/kaggle/input/aptos2019-blindness-detection/"
DIM_X, DIM_Y = 256, 256
BATCH_SIZE = 64
NUM_CLASSES = 5

NUM_WORKERS = max(1, mp.cpu_count() - 1)




## === cell 1
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


def circle_crop_v2(img):
    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))
    height, width, depth = img.shape
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    return img


def preprocess_image(image, sigmaX=25, dim_x=DIM_X, dim_y=DIM_Y):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = circle_crop_v2(image)
    image = cv2.resize(image, (dim_x, dim_y))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


class FixedDropout(Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        return tuple(
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        )




## === cell 2
def kappa_keras(y_true, y_pred):
    return tf.constant(0.0)


def create_kappa_loss(bsize, eps=1e-10, N=NUM_CLASSES):
    def kappa_loss(y_true, y_pred):
        return tf.constant(0.0)

    return kappa_loss


KAPPA_LOSS = create_kappa_loss(BATCH_SIZE)




## === cell 3
train_df = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
train_df["diagnosis"] = train_df["diagnosis"].astype(str)
train_df["filename"] = train_df["id_code"].astype(str) + ".png"


def _load_and_preprocess_path(img_path):
    img = cv2.imread(img_path)
    img = preprocess_image(img)  # unchanged heavy preprocessing
    img = img.astype(np.float32) / 255.0  # normalize once
    return img


def load_and_preprocess(df, img_dir):
    paths = [os.path.join(img_dir, fname) for fname in df["filename"]]
    with mp.Pool(NUM_WORKERS) as pool:
        imgs = pool.map(_load_and_preprocess_path, paths)
    return np.stack(imgs, axis=0)


train_img_dir = os.path.join(BASE_PATH, "train_images")
val_size = int(0.1 * len(train_df))

X_all = load_and_preprocess(train_df, train_img_dir)

train_split_df = train_df.iloc[:-val_size].reset_index(drop=True)
val_split_df = train_df.iloc[-val_size:].reset_index(drop=True)

X_train = X_all[: len(train_split_df)]
X_val = X_all[len(train_split_df) :]

y_train = tf.keras.utils.to_categorical(
    train_split_df["diagnosis"].astype(int), NUM_CLASSES
)
y_val = tf.keras.utils.to_categorical(
    val_split_df["diagnosis"].astype(int), NUM_CLASSES
)




## === cell 4
base_model = EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=(DIM_X, DIM_Y, 3), pooling="avg"
)
base_model.trainable = False  # freeze backbone for better generalization

train_embeddings = base_model.predict(X_train, batch_size=BATCH_SIZE, verbose=1)
val_embeddings = base_model.predict(X_val, batch_size=BATCH_SIZE, verbose=1)

embedding_dim = train_embeddings.shape[1]

inputs = tf.keras.Input(shape=(embedding_dim,))
x = FixedDropout(0.3)(inputs)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

class_counts = train_split_df["diagnosis"].astype(int).value_counts().sort_index()
total_samples = len(train_split_df)
class_weight = {
    i: total_samples / (NUM_CLASSES * class_counts.get(i, 1))
    for i in range(NUM_CLASSES)
}

model.fit(
    train_embeddings,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=20,  # longer training for better fit
    validation_data=(val_embeddings, y_val),
    class_weight=class_weight,  # give balanced influence to all classes
    verbose=1,
)




## === cell 5
test_df = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
test_df["filename"] = test_df["id_code"].astype(str) + ".png"

test_img_dir = os.path.join(BASE_PATH, "test_images")
X_test = load_and_preprocess(test_df, test_img_dir)  # reuse the same helper

test_embeddings = base_model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)

pred_probs = model.predict(test_embeddings, batch_size=BATCH_SIZE, verbose=1)

expected_ratings = np.dot(pred_probs, np.arange(NUM_CLASSES))
pred_labels = np.rint(expected_ratings).astype(int)
pred_labels = np.clip(pred_labels, 0, NUM_CLASSES - 1)

submission = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))
submission["diagnosis"] = pred_labels
submission.to_csv("submission.csv", index=False)

print("Submission file saved as submission.csv")
