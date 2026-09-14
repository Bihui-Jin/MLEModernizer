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

5.06249

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.06376) has done: 'The timeout is dominated by two things: (1) copying ~9k images into 3 directory trees (very slow filesystem I/O), and (2) training/evaluating/predicting at 512×512 with Python-based generators without any input pipelining. To keep the exact same model and training semantics, the main speed fix is to avoid copying by creating the same directory structure but using hardlinks/symlinks (falling back to copy only if linking fails), and to ensure the split is deterministic and done once. Then we speed up the input pipeline by enabling generator multiprocessing/workers in `fit/evaluate/predict` (doesn’t change the algorithm, only parallelizes loading/augmentation). Finally, we remove expensive per-epoch plotting/`clear_output` overhead from the training loop (the training itself is unchanged; only visualization is deferred to the end).'
- What this solution (achieved 5.06378) has done: 'I fix two blockers that prevent the notebook from running end-to-end: the protobuf/Keras import crash in the first cell and the broken train/valid/test directory split that creates links to paths that don’t exist (causing `FileNotFoundError` during `flow_from_directory`). The split bug comes from using only the `.jpg` filename while the actual source directory is organized by breed subfolders, so the code must build correct source paths per breed. These changes are score-neutral by themselves (they restore the intended training data pipeline), but they should also move the logloss substantially toward the target by ensuring the model is actually trained on real images instead of failing mid-epoch. Finally, I keep the model/training logic intact and ensure a valid `submission.csv` is always written in `/kaggle/working/`.'
- What this solution (achieved 5.06377) has done: 'I fix the two root causes preventing the model from training: (1) the protobuf crash caused by forcing the pure-Python protobuf implementation, and (2) incorrect train image paths because this dataset’s `/train/` directory is flat (no breed subfolders), so the split builder must link/copy from `/train/<id>.jpg`. Then I make the split deterministic and stable (per-image RNG derived from the image id) so that you always get non-empty train/valid/test directories and `steps_per_epoch` is positive. These changes keep the same model/training logic intact but should drastically reduce logloss toward the target because the model finally train and produce meaningful probabilities. Finally, I ensure a valid `/kaggle/working/submission.csv` is always written with columns exactly matching `sample_submission.csv`.'
- What this solution (achieved 5.06249) has done: 'I remove the expensive per-breed directory split/linking step and instead build the same train/valid split directly from `labels.csv` using the exact same deterministic CRC32 rule, which preserves the core split semantics while eliminating thousands of filesystem operations. I also remove `tf.py_function` and `ImageDataGenerator` from the `tf.data` pipeline and replace them with equivalent TensorFlow-native augmentation ops (same transforms: rotate, shifts, horizontal flip, rescale), which keeps the model and training loop identical but makes input processing vectorized and parallelizable. Finally, I cache and optimize the validation/test input pipelines (no augmentation) and ensure deterministic behavior via explicit seeds in the stateless random ops.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import shutil
import numpy as np
import pandas as pd
import cv2

import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import clear_output

import tensorflow as tf  # noqa: F401

import tf_keras as keras
from tf_keras import backend as K
from tf_keras.layers import (
    Dense,
    Activation,
    Dropout,
    BatchNormalization,
    Input,
    Flatten,
    MaxPooling2D,
)
from tf_keras.models import Model
from tf_keras.optimizers import Adam
from tf_keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tf_keras.applications.inception_resnet_v2 import InceptionResNetV2
from tf_keras.initializers import he_normal
from tf_keras.preprocessing.image import ImageDataGenerator

np.random.seed(42)
tf.random.set_seed(42)
try:
    keras.utils.set_random_seed(42)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("Imports OK")
print("CPU count:", os.cpu_count())




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_path = "../input/dog-breed-identification/train"
print("Train dir exists:", os.path.isdir(train_path))
print("First 5 files/entries:", sorted(os.listdir(train_path))[:5])




## === cell 2
labels = pd.read_csv("../input/dog-breed-identification/labels.csv")
labels.head(5)




## === cell 3
classes = np.unique(labels.breed)
classes_num = classes.size
classes_num




## === cell 4
train_dir = "../input/dog-breed-identification/train"
images_num = len([f for f in os.listdir(train_dir) if f.lower().endswith(".jpg")])
print(f"Number of images: {images_num}")




## === cell 5
new_train_dir = "/kaggle/working/new_train/"  # training split directory
new_test_dir = "/kaggle/working/new_test/"  # test split directory (used only for evaluation in this notebook)
new_valid_dir = "/kaggle/working/new_valid/"  # validation split directory

for d in [new_train_dir, new_test_dir, new_valid_dir]:
    os.makedirs(d, exist_ok=True)

print(new_train_dir, new_test_dir, new_valid_dir)




## === cell 6
for sub_dir in classes:
    os.makedirs(os.path.join(new_train_dir, sub_dir), exist_ok=True)
    os.makedirs(os.path.join(new_test_dir, sub_dir), exist_ok=True)
    os.makedirs(os.path.join(new_valid_dir, sub_dir), exist_ok=True)

print("Created subdirs:", len(os.listdir(new_train_dir)))




## === cell 7
labels_jpg = labels.copy(deep=True)
labels_jpg["id"] += ".jpg"  # add .jpg to each image id to get its filename

grouped_ids = labels_jpg.groupby("breed")["id"].apply(list).to_dict()
print(classes[0], grouped_ids[classes[0]][:5])




## === cell 8
test_split = 0.1
valid_split = 0.2




## === cell 9
import zlib


def _fast_link_or_copy(src, dst):
    """Create dst as a hardlink/symlink/copy pointing at src."""
    if os.path.exists(dst):
        return
    src_abs = os.path.abspath(src)
    try:
        os.link(src_abs, dst)  # hardlink: fastest, no extra space
    except Exception:
        try:
            os.symlink(src_abs, dst)  # symlink fallback
        except Exception:
            shutil.copy2(src_abs, dst)  # final fallback


def _dir_has_images(base_dir):
    for breed in classes:
        p = os.path.join(base_dir, breed)
        if os.path.isdir(p):
            try:
                with os.scandir(p) as it:
                    for e in it:
                        if e.is_file() and e.name.lower().endswith(".jpg"):
                            return True
            except FileNotFoundError:
                pass
    return False


def _stable_rand01_from_str(s: str) -> float:
    v = zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF
    return v / 4294967296.0


already_split = (
    _dir_has_images(new_train_dir)
    or _dir_has_images(new_valid_dir)
    or _dir_has_images(new_test_dir)
)

train_size = 0
valid_size = 0
test_size = 0

if not already_split:
    for breed, breed_images in grouped_ids.items():
        for img in breed_images:
            rnd_prob = _stable_rand01_from_str(img)
            if rnd_prob <= test_split:
                test_size += 1
            elif rnd_prob <= (test_split + valid_split):
                valid_size += 1
            else:
                train_size += 1
    print("Skipping on-disk split (building datasets from file lists instead).")
else:
    for breed in classes:
        for base, counter in (
            (new_train_dir, "train"),
            (new_valid_dir, "valid"),
            (new_test_dir, "test"),
        ):
            p = os.path.join(base, breed)
            if not os.path.isdir(p):
                continue
            n = 0
            with os.scandir(p) as it:
                for e in it:
                    if e.is_file() and e.name.lower().endswith(".jpg"):
                        n += 1
            if counter == "train":
                train_size += n
            elif counter == "valid":
                valid_size += n
            else:
                test_size += n
    print("Split already exists; using existing directories.")

print("Split sizes:", train_size, valid_size, test_size)

if train_size == 0 or valid_size == 0:
    raise RuntimeError(
        f"Invalid split produced empty directory: train_size={train_size}, valid_size={valid_size}. "
        "Check train_dir path and split logic."
    )




## === cell 10
test_breed = classes[0]
print("Example breed:", test_breed)
if (
    os.path.isdir(os.path.join(new_train_dir, test_breed))
    and len(os.listdir(os.path.join(new_train_dir, test_breed))) > 0
):
    print(
        "First 5 files:",
        sorted(os.listdir(os.path.join(new_train_dir, test_breed)))[:5],
    )
else:
    print(
        "First 5 files in original train dir:",
        sorted([f for f in os.listdir(train_dir) if f.lower().endswith(".jpg")])[:5],
    )




## === cell 11
width, height, channels = 512, 512, 3




## === cell 12
images_samples = np.zeros((4, height, width, 3), dtype=float)
samples_labels = []

rnd_breeds = np.random.choice(classes, size=4, replace=True)
for i, b in enumerate(rnd_breeds):
    breed_ids = grouped_ids[b]
    img_filename = np.random.choice(breed_ids)
    img_path = os.path.join(train_dir, img_filename)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        ok = False
        for _ in range(min(10, len(breed_ids))):
            img_filename = np.random.choice(breed_ids)
            img_path = os.path.join(train_dir, img_filename)
            img_bgr = cv2.imread(img_path)
            if img_bgr is not None:
                ok = True
                break
        if not ok:
            raise RuntimeError(
                f"Failed to read any image for breed '{b}' from {train_dir}"
            )

    img_rgb = img_bgr[:, :, [2, 1, 0]]
    images_samples[i] = cv2.resize(src=img_rgb, dsize=(width, height)) / 255.0
    samples_labels.append(b)




## === cell 13
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
for ax, img, label in zip(axs.ravel(), images_samples, samples_labels):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Class: {label}", size=15)
plt.show()




## === cell 14
norm_factor = 1 / 255

transform_params = {
    "featurewise_center": False,
    "featurewise_std_normalization": False,
    "samplewise_center": False,
    "samplewise_std_normalization": False,
    "rotation_range": 30,
    "width_shift_range": 0.15,
    "height_shift_range": 0.15,
    "horizontal_flip": True,
    "rescale": norm_factor,
}

img_gen = ImageDataGenerator(**transform_params)




## === cell 15
img_feed = ImageDataGenerator(rescale=1 / 255)




## === cell 16
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Augmentation Results", size=32)

for axs_col, img in enumerate(images_samples):
    viz_transoform_params = {
        "theta": np.random.randint(
            -transform_params["rotation_range"], transform_params["rotation_range"]
        ),
        "tx": np.random.uniform(
            -transform_params["width_shift_range"],
            transform_params["width_shift_range"],
        ),
        "ty": np.random.uniform(
            -transform_params["height_shift_range"],
            transform_params["height_shift_range"],
        ),
        "flip_horizontal": np.random.choice([True, False], p=[0.5, 0.5]),
    }
    aug_img = img_gen.apply_transform(img, viz_transoform_params)

    axs[0, axs_col].imshow(img)
    axs[0, axs_col].axis("off")
    axs[0, axs_col].set_title("Original Image", size=15)

    axs[1, axs_col].imshow(aug_img)
    axs[1, axs_col].axis("off")
    axs[1, axs_col].set_title("Augmented Image", size=15)

plt.show()




## === cell 17
class Plotter(Callback):
    def plot(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        ax1.plot(self.epochs, self.losses, label="train_loss")
        ax1.plot(self.epochs, self.val_losses, label="val_loss")

        ax2.plot(self.epochs, self.acc, label="train_acc")
        ax2.plot(self.epochs, self.val_acc, label="val_acc")

        ax1.set_title("Loss vs Epochs")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")

        ax2.set_title("Accuracy vs Epochs")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")

        ax1.legend()
        ax2.legend()
        plt.show()

        if len(self.epochs) > 0:
            print(
                f"Epoch #{self.epochs[-1]+1} >> train_acc={self.acc[-1]*100:.3f}%, train_loss={self.losses[-1]:.5f}"
            )
            print(
                f"Epoch #{self.epochs[-1]+1} >> val_acc={self.val_acc[-1]*100:.3f}%, val_loss={self.val_losses[-1]:.5f}"
            )

    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []
        self.epochs = []
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.acc.append(logs.get("accuracy", logs.get("acc")))
        self.val_acc.append(logs.get("val_accuracy", logs.get("val_acc")))
        self.epochs.append(epoch)

    def on_train_end(self, logs=None):
        self.plot()


plotter = Plotter()




## === cell 18
plateau_reduce = ReduceLROnPlateau(
    monitor="val_loss", factor=0.002, patience=1, min_lr=1e-50
)




## === cell 19
e_stop = EarlyStopping(
    monitor="val_loss", patience=25, mode="min", restore_best_weights=True
)




## === cell 20
callbacks = [plotter, plateau_reduce, e_stop]




## === cell 21
def dense_block(x, neurons, layer_no):
    x = Dense(
        neurons, kernel_initializer=he_normal(layer_no), name=f"topDense{layer_no}"
    )(x)
    x = Activation("relu", name=f"Relu{layer_no}")(x)
    x = BatchNormalization(name=f"BatchNorm{layer_no}")(x)
    x = Dropout(0.5, name=f"Dropout{layer_no}")(x)
    return x




## === cell 22
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
    flat1_bn = BatchNormalization()(flat1)

    dens1 = dense_block(flat1_bn, neurons=512, layer_no=1)
    dens2 = dense_block(dens1, neurons=512, layer_no=2)
    dens3 = dense_block(dens2, neurons=1024, layer_no=3)

    dens_final = Dense(classes_num, name="Dense6")(dens3)
    output_layer = Activation("softmax")(dens_final)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    return model




## === cell 23
height, width, channels_num = 512, 512, 3
learning_rate = 0.001
epochs = 25
batch_size = 32




## === cell 24
model = create_model((height, width, channels_num))
optimizer = Adam(learning_rate)

model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)
model.summary()




## === cell 25
AUTOTUNE = tf.data.AUTOTUNE

one_hot_map = {c: i for i, c in enumerate(classes)}


def _build_split_file_lists_from_labels(labels_df):
    train_files = []
    train_labels = []
    valid_files = []
    valid_labels = []
    test_files_local = []
    test_labels_local = []

    ids = labels_df["id"].to_numpy()
    breeds = labels_df["breed"].to_numpy()
    for img_id, breed in zip(ids, breeds):
        fname = img_id + ".jpg"
        rnd_prob = _stable_rand01_from_str(fname)
        fpath = os.path.join(train_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Expected source image not found: {fpath}")
        y_idx = one_hot_map[breed]
        if rnd_prob <= test_split:
            test_files_local.append(fpath)
            test_labels_local.append(y_idx)
        elif rnd_prob <= (test_split + valid_split):
            valid_files.append(fpath)
            valid_labels.append(y_idx)
        else:
            train_files.append(fpath)
            train_labels.append(y_idx)

    return (
        np.array(train_files, dtype=object),
        np.array(train_labels, dtype=np.int32),
        np.array(valid_files, dtype=object),
        np.array(valid_labels, dtype=np.int32),
        np.array(test_files_local, dtype=object),
        np.array(test_labels_local, dtype=np.int32),
    )


if already_split:
    def _list_files_with_labels(base_dir):
        files = []
        labels_idx = []
        for breed in classes:
            breed_dir = os.path.join(base_dir, breed)
            if not os.path.isdir(breed_dir):
                continue
            with os.scandir(breed_dir) as it:
                for e in it:
                    if e.is_file() and e.name.lower().endswith(".jpg"):
                        files.append(os.path.join(breed_dir, e.name))
                        labels_idx.append(one_hot_map[breed])
        files = np.array(files, dtype=object)
        labels_idx = np.array(labels_idx, dtype=np.int32)
        return files, labels_idx

    train_files, train_labels_idx = _list_files_with_labels(new_train_dir)
    valid_files, valid_labels_idx = _list_files_with_labels(new_valid_dir)
else:
    train_files, train_labels_idx, valid_files, valid_labels_idx, _, _ = (
        _build_split_file_lists_from_labels(labels)
    )

if train_files.size == 0 or valid_files.size == 0:
    raise RuntimeError("No training/validation files found after split.")


def _decode_resize_rescale(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [height, width], method="nearest")
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


_ROT_RAD = float(transform_params["rotation_range"]) * np.pi / 180.0
_WSHIFT = float(transform_params["width_shift_range"])
_HSHIFT = float(transform_params["height_shift_range"])


def _train_augment_tf(img, seed2):
    seed = tf.stack([tf.constant(42, tf.int32), tf.cast(seed2, tf.int32)], axis=0)

    flip_u = tf.random.stateless_uniform([], seed=seed, minval=0.0, maxval=1.0)
    img = tf.cond(flip_u < 0.5, lambda: tf.image.flip_left_right(img), lambda: img)

    angle = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([1, 3], tf.int32), minval=-_ROT_RAD, maxval=_ROT_RAD
    )
    img = tf.image.rotate(img, angle, interpolation="BILINEAR")

    tx_frac = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([5, 7], tf.int32), minval=-_WSHIFT, maxval=_WSHIFT
    )
    ty_frac = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([11, 13], tf.int32), minval=-_HSHIFT, maxval=_HSHIFT
    )
    dx = tf.cast(tf.round(tx_frac * tf.cast(width, tf.float32)), tf.int32)
    dy = tf.cast(tf.round(ty_frac * tf.cast(height, tf.float32)), tf.int32)
    img = tf.roll(img, shift=[dy, dx], axis=[0, 1])

    return img


def _train_map(path, y_idx):
    img = _decode_resize_rescale(path)
    seed2 = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
    img = _train_augment_tf(img, seed2)
    y = tf.one_hot(y_idx, depth=classes_num, dtype=tf.float32)
    return img, y


def _valid_map(path, y_idx):
    img = _decode_resize_rescale(path)
    y = tf.one_hot(y_idx, depth=classes_num, dtype=tf.float32)
    return img, y


train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels_idx))
train_ds = train_ds.shuffle(
    buffer_size=min(4096, int(train_files.size)), seed=42, reshuffle_each_iteration=True
)
train_ds = train_ds.map(_train_map, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

valid_ds = tf.data.Dataset.from_tensor_slices((valid_files, valid_labels_idx))
valid_ds = valid_ds.map(_valid_map, num_parallel_calls=AUTOTUNE).cache()
valid_ds = valid_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

steps_per_epoch = max(1, int(np.ceil(train_files.size / batch_size)))
validation_steps = max(1, int(np.ceil(valid_files.size / batch_size)))

print("train samples:", train_files.size, "valid samples:", valid_files.size)
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4169423003.py in <cell line: 0>()
    139     buffer_size=min(4096, int(train_files.size)), seed=42, reshuffle_each_iteration=True
    140 )
--> 141 train_ds = train_ds.map(_train_map, num_parallel_calls=AUTOTUNE)
    142 train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    143 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in map(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
   2339     from tensorflow.python.data.ops import map_op
   2340 
-> 2341     return map_op._map_v2(
   2342         self,
   2343         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in _map_v2(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
     55           num_parallel_calls,
     56       )
---> 57     return _ParallelMapDataset(
     58         input_dataset,
     59         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in __init__(self, input_dataset, map_func, num_parallel_calls, deterministic, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, use_unbounded_threadpool, name)
    200     self._input_dataset = input_dataset
    201     self._use_inter_op_parallelism = use_inter_op_parallelism
--> 202     self._map_func = structured_function.StructuredFunctionWrapper(
    203         map_func,
    204         self._transformation_name(),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in __init__(self, func, transformation_name, dataset, input_classes, input_shapes, input_types, input_structure, add_to_graph, use_legacy_function, defun_kwargs)
    263         fn_factory = trace_tf_function(defun_kwargs)
    264 
--> 265     self._function = fn_factory()
    266     # There is no graph to add in eager mode.
    267     add_to_graph &= not context.executing_eagerly()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in get_concrete_function(self, *args, **kwargs)
   1249   def get_concrete_function(self, *args, **kwargs):
   1250     # Implements PolymorphicFunction.get_concrete_function.
-> 1251     concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
   1252     concrete._garbage_collector.release()  # pylint: disable=protected-access
   1253     return concrete

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _get_concrete_function_garbage_collected(self, *args, **kwargs)
   1219       if self._variable_creation_config is None:
   1220         initializers = []
-> 1221         self._initialize(args, kwargs, add_initializers_to=initializers)
   1222         self._initialize_uninitialized_variables(initializers)
   1223 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _initialize(self, args, kwds, add_initializers_to)
    694     )
    695     # Force the definition of the function for these arguments
--> 696     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
    697         args, kwds, self._variable_creation_config
    698     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in trace_function(args, kwargs, tracing_options)
    176       kwargs = {}
    177 
--> 178     concrete_function = _maybe_define_function(
    179         args, kwargs, tracing_options
    180     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _maybe_define_function(args, kwargs, tracing_options)
    281         else:
    282           target_func_type = lookup_func_type
--> 283         concrete_function = _create_concrete_function(
    284             target_func_type, lookup_func_context, func_graph, tracing_options
    285         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _create_concrete_function(function_type, type_context, func_graph, tracing_options)
    308       attributes_lib.DISABLE_ACD, False
    309   )
--> 310   traced_func_graph = func_graph_module.func_graph_from_py_func(
    311       tracing_options.name,
    312       tracing_options.python_function,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)
   1057 
   1058     _, original_func = tf_decorator.unwrap(python_func)
-> 1059     func_outputs = python_func(*func_args, **func_kwargs)
   1060 
   1061     # invariant: `func_outputs` contains only Tensors, CompositeTensors,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in wrapped_fn(*args, **kwds)
    597         # the function a weak reference to itself to avoid a reference cycle.
    598         with OptionalXlaContext(compile_with_xla):
--> 599           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    600         return out
    601 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapped_fn(*args)
    229       # Note: wrapper_helper will apply autograph based on context.
    230       def wrapped_fn(*args):  # pylint: disable=missing-docstring
--> 231         ret = wrapper_helper(*args)
    232         ret = structure.to_tensor_list(self._output_structure, ret)
    233         return [ops.convert_to_tensor(t) for t in ret]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapper_helper(*args)
    159       if not _should_unpack(nested_args):
    160         nested_args = (nested_args,)
--> 161       ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
    162       ret = variable_utils.convert_variables_to_tensors(ret)
    163       if _should_pack(ret):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):
--> 693           raise e.ag_error_metadata.to_exception(e)
    694         else:
    695           raise

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    688       try:
    689         with conversion_ctx:
--> 690           return converted_call(f, args, kwargs, options=options)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_fileodywe7ab.py in tf___train_map(path, y_idx)
     10                 img = ag__.converted_call(ag__.ld(_decode_resize_rescale), (ag__.ld(path),), None, fscope)
     11                 seed2 = ag__.converted_call(ag__.ld(tf).strings.to_hash_bucket_fast, (ag__.ld(path), 2 ** 31 - 1), None, fscope)
---> 12                 img = ag__.converted_call(ag__.ld(_train_augment_tf), (ag__.ld(img), ag__.ld(seed2)), None, fscope)
     13                 y = ag__.converted_call(ag__.ld(tf).one_hot, (ag__.ld(y_idx),), dict(depth=ag__.ld(classes_num), dtype=ag__.ld(tf).float32), fscope)
     14                 try:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_fileiv_u3aq7.py in tf___train_augment_tf(img, seed2)
     12                 img = ag__.converted_call(ag__.ld(tf).cond, (ag__.ld(flip_u) < 0.5, ag__.autograph_artifact(lambda: ag__.converted_call(ag__.ld(tf).image.flip_left_right, (ag__.ld(img),), None, fscope)), ag__.autograph_artifact(lambda: ag__.ld(img))), None, fscope)
     13                 angle = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([],), dict(seed=ag__.ld(seed) + ag__.converted_call(ag__.ld(tf).constant, ([1, 3], ag__.ld(tf).int32), None, fscope), minval=-ag__.ld(_ROT_RAD), maxval=ag__.ld(_ROT_RAD)), fscope)
---> 14                 img = ag__.converted_call(ag__.ld(tf).image.rotate, (ag__.ld(img), ag__.ld(angle)), dict(interpolation='BILINEAR'), fscope)
     15                 tx_frac = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([],), dict(seed=ag__.ld(seed) + ag__.converted_call(ag__.ld(tf).constant, ([5, 7], ag__.ld(tf).int32), None, fscope), minval=-ag__.ld(_WSHIFT), maxval=ag__.ld(_WSHIFT)), fscope)
     16                 ty_frac = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([],), dict(seed=ag__.ld(seed) + ag__.converted_call(ag__.ld(tf).constant, ([11, 13], ag__.ld(tf).int32), None, fscope), minval=-ag__.ld(_HSHIFT), maxval=ag__.ld(_HSHIFT)), fscope)

AttributeError: in user code:

    File "/tmp/ipykernel_11/4169423003.py", line 126, in _train_map  *
        img = _train_augment_tf(img, seed2)
    File "/tmp/ipykernel_11/4169423003.py", line 106, in _train_augment_tf  *
        img = tf.image.rotate(img, angle, interpolation="BILINEAR")

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'rotate'


## === cell 26
history = model.fit(
    train_ds,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_ds,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1,
)




## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/608984759.py in <cell line: 0>()
      2     train_ds,
      3     epochs=epochs,
----> 4     steps_per_epoch=steps_per_epoch,
      5     validation_data=valid_ds,
      6     validation_steps=validation_steps,

NameError: name 'steps_per_epoch' is not defined

## === cell 27
print("Skipping new_test_dir evaluation to save time (not needed for submission).")




## === cell 28
one_hot_map = {c: i for i, c in enumerate(classes)}
list(one_hot_map.items())[:5]




## === cell 29
sample_sub_path = "../input/dog-breed-identification/sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)
breed_cols = [c for c in sample_sub.columns if c != "id"]

kaggle_test_dir = "../input/dog-breed-identification/test"
test_files = sorted(
    [f for f in os.listdir(kaggle_test_dir) if f.lower().endswith(".jpg")]
)
sub = sample_sub.copy()
sub_ids = sub["id"].tolist()
id_to_file = {
    os.path.splitext(f)[0]: os.path.join(kaggle_test_dir, f) for f in test_files
}

missing = [i for i in sub_ids if i not in id_to_file]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images referenced by sample_submission. Example: {missing[:3]}"
    )




## === cell 30
test_df = pd.DataFrame({"filename": [id_to_file[i] for i in sub_ids], "id": sub_ids})

test_pred_files = test_df["filename"].to_numpy(dtype=object)


def _pred_map(path):
    return _decode_resize_rescale(path)


pred_ds = tf.data.Dataset.from_tensor_slices(test_pred_files)
pred_ds = pred_ds.map(_pred_map, num_parallel_calls=AUTOTUNE).cache()
pred_ds = pred_ds.batch(batch_size).prefetch(AUTOTUNE)

pred = model.predict(
    pred_ds,
    steps=int(np.ceil(len(test_df) / batch_size)),
    verbose=1,
)

pred = pred[: len(test_df), :]

inv_map = {v: k for k, v in one_hot_map.items()}
model_class_order = [inv_map[i] for i in range(classes_num)]

pred_df = pd.DataFrame(pred, columns=model_class_order)
pred_df = pred_df[breed_cols]  # reorder to match submission template exactly

sub_out = pd.concat(
    [test_df[["id"]].reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1
)

assert sub_out.shape == sample_sub.shape, (sub_out.shape, sample_sub.shape)
assert list(sub_out.columns) == list(
    sample_sub.columns
), "Submission columns do not match sample_submission."

sub_path = "/kaggle/working/submission.csv"
sub_out.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(sub_out.head())
