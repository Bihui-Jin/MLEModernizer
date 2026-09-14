# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf

tf.config.optimizer.set_jit(True)

policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)

try:
    num_threads = min(8, os.cpu_count() or 1)
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
except Exception as e:
    print("TensorFlow threading config skipped:", e)

print("TensorFlow version:", tf.__version__)
print("Input directory listing:", os.listdir("../input"))




## === cell 1
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # image too dark
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        img = np.stack([img1, img2, img3], axis=-1)
        return img
    return img




## === cell 2
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8




## === cell 3
def preprocess_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_image_from_gray(image)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 30), -4, 128)
    return image




## === cell 4
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, PReLU
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50


def getResNet50(input_shape=(IMG_SIZE, IMG_SIZE, 3), classes=5, weights="imagenet"):
    input_layer = Input(shape=input_shape)
    base_model = ResNet50(include_top=False, weights=weights, input_tensor=input_layer)
    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    x = GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = Dense(1024, name="fc1")(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)
    output = Dense(classes, activation="softmax", name="output")(x)
    model = Model(inputs=input_layer, outputs=output)
    return model




## === cell 5
import pandas as pd, numpy as np, cv2, concurrent.futures
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

cv2.setNumThreads(0)

train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
train_ids = train_df["id_code"].values
train_labels = train_df["diagnosis"].values


def load_train_image(idx):
    img_path = os.path.join(
        "../input/aptos2019-blindness-detection/train_images", f"{train_ids[idx]}.png"
    )
    img = preprocess_image(img_path)
    img = img.astype(np.float32) / 255.0
    return img, train_labels[idx]


print("Loading training images in parallel (limited thread pool)...")
MAX_WORKERS = min(8, os.cpu_count() or 1)  # keep a modest number of threads
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    results = list(executor.map(load_train_image, range(len(train_ids))))

X = np.empty((len(train_ids), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
y = np.empty(len(train_ids), dtype=np.int32)
for i, (img, label) in enumerate(results):
    X[i] = img
    y[i] = label

y_cat = to_categorical(y, num_classes=5)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.1, random_state=42, stratify=y
)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = (
    train_ds.shuffle(buffer_size=len(X_train))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

model = getResNet50(weights="imagenet")
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

early_stop = tf.keras.callbacks.EarlyStopping(
    patience=4, restore_best_weights=True, monitor="val_loss"
)
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
    patience=2, factor=0.5, monitor="val_loss"
)

print("Starting training...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, lr_reduce],
    verbose=2,
)




## === cell 6
submit = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
test_ids = submit["id_code"].values


def load_test_image(idx):
    img_path = os.path.join(
        "../input/aptos2019-blindness-detection/test_images", f"{test_ids[idx]}.png"
    )
    img = preprocess_image(img_path)
    img = img.astype(np.float32) / 255.0
    return img


print("Loading test images in parallel (limited thread pool)...")
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    test_imgs = list(executor.map(load_test_image, range(len(test_ids))))

X_test = np.stack(test_imgs)
print("Running batch inference on test set...")
preds = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
ans = [int(np.argmax(p)) for p in preds]

submit["diagnosis"] = ans
submit.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
