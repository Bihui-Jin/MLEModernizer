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
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.28106

# 6. Current score

0.53575

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.68203) has done: 'The timeout is dominated by (1) heavy Python-side image loading/augmentation via `ImageDataGenerator.flow_from_dataframe` and (2) the very expensive `Flatten()` of InceptionResNetV2 feature maps at 299×299, which makes the dense head extremely costly per step. To keep the exact same model architecture and training loop semantics, the main speedups come from eliminating input pipeline overhead (switch to `tf.data` with parallel decode/resize and deterministic seeding) while still applying the same geometric augmentations, and from enabling graph execution/XLA where safe. I also remove expensive per-epoch Python printing overhead from the callback (without changing training behavior) and ensure steps/ordering remain identical. These changes preserve the core model and loss while substantially reducing wall-clock time spent in Python and input I/O.'
- What this solution (achieved 5.68203) has done: 'I fix the two runtime blockers: the protobuf `MessageFactory.GetPrototype` crash caused by an incompatible protobuf implementation (worked around safely by forcing the pure-Python protobuf backend before TensorFlow import), and the missing `tf.image.rotate` API by switching to `tf_keras.preprocessing.image.apply_affine_transform` inside a `tf.numpy_function` while keeping the same augmentation semantics (flip/shift/rotation). These changes are necessary for the notebook to run end-to-end and are score-positive because your previous run effectively trained without the intended augmentation pipeline due to the crash. I also keep your model architecture, loss, optimizer, epochs, and training loop unchanged, and ensure the submission columns/order exactly match `sample_submission.csv`. Finally, I keep deterministic seeding and make sure the pipeline produces `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.5348) has done: 'The timeout is dominated by per-image Python/OpenCV augmentation inside the `tf.data` pipeline (`tf.numpy_function` + `cv2.warpAffine`) repeated every epoch, plus expensive shuffling over the full dataset each epoch. I keep the same model, loss, epochs, and augmentation semantics (rotation/shift/flip with nearest-neighbor + reflect border), but move augmentation fully into TensorFlow graph ops so it can run in parallel efficiently without Python overhead. I also add safe `cache()` on the non-augmented validation/test pipelines (exactly equivalent), and restructure the dataset mapping to avoid recomputing one-hot labels and string hashing more than necessary. These changes preserve training/evaluation semantics while cutting input pipeline overhead dramatically so training fits within 600s.'
- What this solution (achieved 0.55204) has done: 'The crash happens before any variables are defined because `tf_keras` triggers a protobuf/TensorFlow import path that’s incompatible with the environment, so I remove the protobuf workaround and switch to importing TensorFlow normally (and use `tf.keras` APIs) while keeping the exact same model architecture, loss, optimizer, and training loop. I also fix the cell numbering to start at 1 (your notebook starts at cell 0) so the script is consistent, and ensure `tf` is defined everywhere (no `K.tensorflow` indirection). Finally, I keep your `tf.data` pipeline and augmentation logic intact, and ensure the submission columns/order match `sample_submission.csv` and write `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.55204) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which is the direct cause of the `MessageFactory.GetPrototype` error in this environment. I also correct the cell numbering to start at 1 so the script structure matches the required format, without changing the model, loss, optimizer, epochs, or data splits. To move the score down toward the target (lower is better), I ensure the prediction-to-submission class alignment matches `sample_submission.csv` exactly by reindexing predictions to the submission’s breed columns (this is a common hidden logloss killer when class orders differ). All other logic (architecture, training loop, augmentations, and submission writing) remains the same.'
- What this solution (achieved 0.55204) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced pure-Python protobuf backend (it is the direct cause of the `MessageFactory.GetPrototype` AttributeError in this environment) and importing TensorFlow normally. To improve logloss toward your target with minimal semantic impact, I also align the class order to the submission columns by deriving `classes` from `sample_submission.csv` (this prevents silent label/probability misalignment that can severely hurt logloss). Finally, I keep your model architecture, optimizer, epochs, augmentation logic, and training loop intact, and ensure the script always writes a valid `/kaggle/working/submission.csv` with the exact required columns and normalized probabilities.'
- What this solution (achieved 0.53575) has done: 'I fix the runtime crash caused by an incompatible protobuf backend by explicitly forcing the pure-Python protobuf implementation *before* importing TensorFlow (this is the direct root cause of the `MessageFactory.GetPrototype` error). I also make the dataset directory resolution more robust for Kaggle’s `/kaggle/input/...` layout without changing any training logic. Finally, I keep your model/training/augmentation core logic identical, but ensure the submission probabilities are perfectly aligned to `sample_submission.csv` columns and numerically stable (normalization + epsilon clip), which is score-positive for multiclass logloss while being a minimal post-processing fix.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Dense,
    Activation,
    Dropout,
    BatchNormalization,
    Input,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ["TF_DETERMINISTIC_OPS"] = "1"
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

try:
    tf.config.optimizer.set_jit(True)  # XLA compile where possible
except Exception:
    pass

_CANDIDATES = [
    "/kaggle/input/dog-breed-identification",
    "/kaggle/input/dog-breed-identification/dog-breed-identification",
    "../input/dog-breed-identification",
]
INPUT_DIR = None
for c in _CANDIDATES:
    if os.path.isdir(c):
        if os.path.isdir(os.path.join(c, "train")) and os.path.isdir(
            os.path.join(c, "test")
        ):
            if os.path.isfile(os.path.join(c, "labels.csv")) and os.path.isfile(
                os.path.join(c, "sample_submission.csv")
            ):
                INPUT_DIR = c
                break

if INPUT_DIR is None:
    raise FileNotFoundError(
        f"Could not find competition input dir. Tried: {_CANDIDATES}"
    )

TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
LABELS_CSV = os.path.join(INPUT_DIR, "labels.csv")
SAMPLE_SUB_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

WORK_DIR = "/kaggle/working"
NEW_TRAIN_DIR = os.path.join(WORK_DIR, "new_train")
NEW_VALID_DIR = os.path.join(WORK_DIR, "new_valid")
NEW_TEST_DIR = os.path.join(WORK_DIR, "new_test")

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)
print("Resolved INPUT_DIR:", INPUT_DIR)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
print("Train dir exists:", os.path.isdir(TRAIN_DIR))
print("Test dir exists:", os.path.isdir(TEST_DIR))
print("Labels exists:", os.path.isfile(LABELS_CSV))
print("Sample submission exists:", os.path.isfile(SAMPLE_SUB_CSV))
print("Example train files:", sorted(os.listdir(TRAIN_DIR))[:5])



## === cell 2
labels = pd.read_csv(LABELS_CSV)
labels.head(5)



## === cell 3
sample_sub_for_classes = pd.read_csv(SAMPLE_SUB_CSV, nrows=1)
classes = np.array(
    [c for c in sample_sub_for_classes.columns if c != "id"], dtype=object
)
classes_num = classes.size
print("Num classes:", classes_num)
print("First 5 classes:", classes[:5])

unknown = set(labels["breed"].unique()) - set(classes.tolist())
if unknown:
    raise ValueError(
        f"Found breeds in labels not present in sample_submission columns: {sorted(list(unknown))[:10]}"
    )



## === cell 4
images_names = os.listdir(TRAIN_DIR)
images_num = len(images_names)
print(f"Number of train images: {images_num}")



## === cell 5
for d in [NEW_TRAIN_DIR, NEW_VALID_DIR, NEW_TEST_DIR]:
    os.makedirs(d, exist_ok=True)
print("Verified work split directories exist (no copy step):", WORK_DIR)



## === cell 6
labels_jpg = labels.copy(deep=True)
labels_jpg["filename"] = labels_jpg["id"].astype(str) + ".jpg"
labels_jpg["filepath"] = TRAIN_DIR + "/" + labels_jpg["filename"]
labels_jpg.head()



## === cell 7
test_split = 0.1
valid_split = 0.2

rng = np.random.RandomState(SEED)
rnd = rng.rand(len(labels_jpg))
split = np.full(len(labels_jpg), "train", dtype=object)
split[rnd <= test_split] = "test"
split[(rnd > test_split) & (rnd <= (test_split + valid_split))] = "valid"
labels_jpg["split"] = split

filepaths = labels_jpg["filepath"].to_numpy()
exists_mask = np.fromiter(
    (os.path.isfile(p) for p in filepaths), count=len(filepaths), dtype=bool
)
if not exists_mask.all():
    labels_jpg = labels_jpg.loc[exists_mask].reset_index(drop=True)

train_df = labels_jpg[labels_jpg["split"] == "train"][
    ["filename", "breed"]
].reset_index(drop=True)
valid_df = labels_jpg[labels_jpg["split"] == "valid"][
    ["filename", "breed"]
].reset_index(drop=True)
test_df = labels_jpg[labels_jpg["split"] == "test"][["filename", "breed"]].reset_index(
    drop=True
)

print("Split sizes:", len(train_df), len(valid_df), len(test_df))



## === cell 8
width, height, channels_num = 299, 299, 3

images_samples = np.zeros((4, height, width, 3), dtype=float)
samples_labels = []

rnd_indexes = np.random.randint(0, images_num, 4)
for i, rnd_idx in enumerate(rnd_indexes):
    img_filename = images_names[rnd_idx]
    img_id = img_filename[:-4]
    img_bgr = cv2.imread(os.path.join(TRAIN_DIR, img_filename))
    img_rgb = img_bgr[:, :, [2, 1, 0]]
    images_samples[i] = cv2.resize(src=img_rgb, dsize=(width, height)) / 255.0
    img_label = labels.breed[labels.id == img_id].values[0]
    samples_labels.append(img_label)

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
for ax, img, label in zip(axs.ravel(), images_samples, samples_labels):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Class: {label}", size=12)
plt.show()



## === cell 9
transform_params = {
    "featurewise_center": False,
    "featurewise_std_normalization": False,
    "samplewise_center": False,
    "samplewise_std_normalization": False,
    "rotation_range": 30,
    "width_shift_range": 0.15,
    "height_shift_range": 0.15,
    "horizontal_flip": True,
    "rescale": 1 / 255.0,
}

img_gen = ImageDataGenerator(**transform_params)
img_feed = ImageDataGenerator(rescale=1 / 255.0)




## === cell 10
class Plotter(Callback):
    def on_train_begin(self, logs=None):
        self.losses, self.val_losses, self.epochs = [], [], []
        self.acc, self.val_acc = [], []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.acc.append(logs.get("acc", logs.get("accuracy")))
        self.val_acc.append(logs.get("val_acc", logs.get("val_accuracy")))
        self.epochs.append(epoch)
        e = epoch + 1
        tr_acc = (self.acc[-1] or 0.0) * 100.0
        va_acc = (self.val_acc[-1] or 0.0) * 100.0
        print(
            f"Epoch #{e} >> train_acc={tr_acc:.3f}%, train_loss={self.losses[-1]:.5f}"
        )
        print(
            f"Epoch #{e} >> val_acc={va_acc:.3f}%, val_loss={self.val_losses[-1]:.5f}"
        )


plotter = Plotter()



## === cell 11
plateau_reduce = ReduceLROnPlateau(
    monitor="val_loss", factor=0.01, patience=1, min_lr=1e-20
)
e_stop = EarlyStopping(
    monitor="val_loss", patience=15, mode="min", restore_best_weights=True
)
callbacks = [plotter, plateau_reduce, e_stop]




## === cell 12
def dense_block(x, neurons, layer_no):
    x = Dense(
        neurons, kernel_initializer=he_normal(seed=layer_no), name=f"topDense{layer_no}"
    )(x)
    x = Activation("relu", name=f"Relu{layer_no}")(x)
    x = BatchNormalization(name=f"BatchNorm{layer_no}")(x)
    x = Dropout(0.5, name=f"Dropout{layer_no}")(x)
    return x




## === cell 13
def create_model(shape):
    input_layer = Input(shape, name="input_layer")
    incep_res = InceptionResNetV2(
        include_top=False, weights="imagenet", input_tensor=input_layer
    )
    for layer in incep_res.layers:
        layer.trainable = False

    pool = MaxPooling2D(pool_size=[3, 3], strides=[3, 3], padding="same")(
        incep_res.output
    )
    flat1 = Flatten(name="Flatten1")(pool)
    flat1_bn = BatchNormalization(name="BatchNormFlat")(flat1)

    dens1 = dense_block(flat1_bn, neurons=512, layer_no=1)
    dens2 = dense_block(dens1, neurons=512, layer_no=2)
    dens3 = dense_block(dens2, neurons=1024, layer_no=3)

    dens_final = Dense(classes_num, name="Dense4")(dens3)
    output_layer = Activation("softmax", name="Softmax")(dens_final)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    return model




## === cell 14
learning_rate = 0.004
epochs = 15
batch_size = 32

model = create_model((height, width, channels_num))
optimizer = Adam(learning_rate=learning_rate)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["acc"],
    run_eagerly=False,
)
model.summary()



## === cell 15
AUTOTUNE = tf.data.AUTOTUNE
class_to_index = {c: i for i, c in enumerate(classes.tolist())}


def _read_decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


@tf.function
def _augment_tf(img, seed_pair):
    seed_pair = tf.cast(seed_pair, tf.int32)

    flip_r = tf.random.stateless_uniform([], seed=seed_pair, minval=0.0, maxval=1.0)
    img = tf.cond(flip_r < 0.5, lambda: tf.image.flip_left_right(img), lambda: img)

    w = tf.cast(tf.shape(img)[1], tf.float32)
    h = tf.cast(tf.shape(img)[0], tf.float32)

    s1 = seed_pair + tf.constant([1, 0], tf.int32)
    s2 = seed_pair + tf.constant([2, 0], tf.int32)
    s3 = seed_pair + tf.constant([3, 0], tf.int32)

    tx = tf.random.stateless_uniform([], seed=s1, minval=-0.15, maxval=0.15) * w
    ty = tf.random.stateless_uniform([], seed=s2, minval=-0.15, maxval=0.15) * h
    theta = tf.random.stateless_uniform([], seed=s3, minval=-30.0, maxval=30.0) * (
        np.pi / 180.0
    )

    pad = 64
    img_p = tf.pad(img, [[pad, pad], [pad, pad], [0, 0]], mode="REFLECT")

    Hp = tf.cast(tf.shape(img_p)[0], tf.float32)
    Wp = tf.cast(tf.shape(img_p)[1], tf.float32)
    cx = Wp * 0.5
    cy = Hp * 0.5

    cos_t = tf.cos(theta)
    sin_t = tf.sin(theta)

    a0 = cos_t
    a1 = -sin_t
    b0 = sin_t
    b1 = cos_t

    a2 = cx - a0 * cx - a1 * cy + tx
    b2 = cy - b0 * cx - b1 * cy + ty

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[tf.newaxis, :]

    img_t = tf.raw_ops.ImageProjectiveTransformV3(
        images=img_p[tf.newaxis, ...],
        transforms=transform,
        output_shape=tf.shape(img_p)[:2],
        interpolation="NEAREST",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]

    img_out = tf.image.crop_to_bounding_box(img_t, pad, pad, height, width)
    img_out = tf.clip_by_value(img_out, 0.0, 1.0)
    img_out.set_shape([height, width, 3])
    return img_out


def make_dataset(df, training):
    paths_list = [os.path.join(TRAIN_DIR, fn) for fn in df["filename"].tolist()]
    labels_idx_list = [class_to_index[b] for b in df["breed"].tolist()]

    paths = tf.constant(paths_list)
    labels_idx = tf.constant(labels_idx_list, dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels_idx))
    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(p, y):
        img = _read_decode_resize(p)
        if training:
            s = tf.strings.to_hash_bucket_fast(p, 2**31 - 1)
            seed_pair = tf.stack(
                [tf.cast(s, tf.int32), tf.cast(SEED, tf.int32)], axis=0
            )
            img = _augment_tf(img, seed_pair)
        y_oh = tf.one_hot(y, depth=classes_num, dtype=tf.float32)
        return img, y_oh

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(train_df, training=True)
valid_ds = make_dataset(valid_df, training=False).cache()

steps_per_epoch = int(np.ceil(len(train_df) / batch_size))
val_steps = int(np.ceil(len(valid_df) / batch_size))
print("steps_per_epoch:", steps_per_epoch, "val_steps:", val_steps)



## === cell 16
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    callbacks=callbacks,
    verbose=1,
)



## === cell 17
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
breed_cols = [c for c in sample_sub.columns if c != "id"]

sample_ids = sample_sub["id"].astype(str).tolist()
test_filenames = [f"{i}.jpg" for i in sample_ids]

missing = [
    fn for fn in test_filenames if not os.path.isfile(os.path.join(TEST_DIR, fn))
]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images (first 5): {missing[:5]}"
    )

test_df = pd.DataFrame({"id": sample_ids, "filename": test_filenames})
test_ids = test_df["id"].tolist()
print("Num test images:", len(test_ids), "Expected:", len(sample_sub))


def make_test_dataset(df):
    paths_list = [os.path.join(TEST_DIR, fn) for fn in df["filename"].tolist()]
    paths = tf.constant(paths_list)
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map_fn(p):
        img = _read_decode_resize(p)
        return img

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False).cache().prefetch(AUTOTUNE)
    return ds


test_ds = make_test_dataset(test_df)



## === cell 18
preds = model.predict(
    test_ds,
    steps=int(np.ceil(len(test_ids) / batch_size)),
    verbose=1,
)
preds = np.asarray(preds, dtype=np.float64)[: len(test_ids)]

pred_df = pd.DataFrame(preds, columns=list(classes))
pred_df.insert(0, "id", test_ids)

sub = sample_sub[["id"]].merge(pred_df, on="id", how="left")

for c in breed_cols:
    if c not in sub.columns:
        sub[c] = 1.0 / len(breed_cols)

sub = sub[["id"] + breed_cols]

prob = sub[breed_cols].to_numpy(dtype=np.float64)
nan_mask = ~np.isfinite(prob)
if nan_mask.any():
    prob[nan_mask] = 1.0 / len(breed_cols)

row_sums = prob.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
prob = prob / row_sums

eps = 1e-7
prob = np.clip(prob, eps, 1.0 - eps)
prob = prob / prob.sum(axis=1, keepdims=True)

sub.loc[:, breed_cols] = prob

out_path = os.path.join(WORK_DIR, "submission.csv")
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub.head())
print("Submission shape:", sub.shape, "Expected:", sample_sub.shape)
print("Row sums (min/max):", prob.sum(axis=1).min(), prob.sum(axis=1).max())
