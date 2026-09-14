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
Given a dataset of images of scanned text that is noisy, remove the noise.

## Metric
Root mean squared error between the cleaned pixel intensities and the actual grayscale pixel intensities.

## Submission Format
Form the submission file by melting each images into a set of pixels, assigning each pixel an id of image_row_col (e.g. 1_2_1 is image 1, row 2, column 1). Intensity values range from 0 (black) to 1 (white). The file should contain a header and have the following format:

```
id,value
1_1_1,1
1_2_1,1
1_3_1,1
etc.
```

## Dataset
You are provided two sets of images, train and test. These images contain various styles of text, to which synthetic noise has been added to simulate real-world, messy artifacts. The training set includes the test without the noise (train_cleaned).

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
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
protobuf==6.33.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        input/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        working/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
```

-> data/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> data/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> working/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

# 5. Target score

0.02953

# 6. Current score

0.28616

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.28616) has done: 'I remove the failing imgaug imports and related augmentation code, add a small validation split and a modest extra convolutional layer (keeping the auto‑encoder architecture essentially the same) to improve generalisation, and adjust early‑stopping to monitor validation loss. This fixes the runtime errors, ensures a valid submission.csv is written, and nudges the RMSE toward the target while preserving the core model logic.'
- What this solution (achieved 0.28616) has done: 'I patch the protobuf incompatibility that prevents TensorFlow from importing by adding a small compatibility shim before the TensorFlow import, and keep the rest of the pipeline unchanged so it runs end‑to‑end and produces a valid `submission.csv`. This fix resolves the runtime error while preserving the original auto‑encoder logic.'
- What this solution (achieved 0.28616) has done: 'I keep the auto‑encoder architecture unchanged and only adjust the training procedure to let the model learn longer and fine‑tune its learning rate. The early‑stopping patience is increased to 100, the maximum epochs are raised to 1000, and a ReduceLROnPlateau scheduler is added so training continues with a smaller step when progress stalls. These modest hyper‑parameter changes are expected to lower validation loss and therefore move the RMSE closer to the target while preserving the core pipeline.'
- What this solution (achieved 0.28616) has done: 'The update keeps the model architecture and training logic unchanged but speeds up inference and submission creation.  
1. It runs the autoencoder on the whole test set in a single batched `predict` call instead of one‑by‑one passes.  
2. It removes the inner pixel‑wise Python loops by generating row/column indices with NumPy and building the `id` strings in a single list comprehension per image.  
3. These changes dramatically cut runtime while preserving the exact predictions and required output format.'
- What this solution (achieved 0.28616) has done: 'The update speeds up the heavy image‑loading steps by reading images directly in grayscale (avoiding an extra color‑to‑gray conversion) and by pre‑allocating NumPy arrays instead of building Python lists that are later converted. This removes unnecessary Python overhead while producing exactly the same normalized tensors, so the model architecture, training loop, and augmentation remain unchanged. All other logic, including callbacks, model definition, and submission creation, is untouched.'
- What this solution (achieved 0.28616) has done: 'The changes switch to a mixed‑precision policy (speed‑up on GPU) and replace the NumPy‑based `model.fit` call with a `tf.data.Dataset` pipeline, eliminating Python‑level per‑epoch overhead while keeping the exact model, loss, augmentation, and early‑stopping logic unchanged.'
- What this solution (achieved 0.28616) has done: 'Implemented targeted optimizations to eliminate the timeout while keeping the model architecture and training logic unchanged.

- In **cell 3**, images are loaded as `float16` instead of `float32`. Mixed‑precision training already uses `float16`, so this reduces memory traffic without affecting numerical results.
- In **cell 8**, both training and validation datasets are cached (`.cache()`) to avoid re‑materializing tensors each epoch, and batch size is increased to 48 to cut the number of gradient steps per epoch. The early‑stopping and learning‑rate callbacks remain unchanged, preserving the original training behavior.'
- What this solution (achieved 0.28616) has done: 'The only change needed is to limit the maximum number of training epochs, because early stopping already stops when validation loss stops improving. Reducing the epoch cap from 1000 to a lower value (e.g., 300) keeps the same training logic, early‑stopping behavior and model architecture while guaranteeing the run finishes well under the 600 s timeout.'
- What this solution (achieved 0.28616) has done: 'The changes conditionally enable mixed‑precision only when a GPU is present (CPU mixed‑precision can be slower) and set a deterministic threading configuration, which speeds up the TF training loop without altering the model, data, or training logic. The rest of the notebook is left unchanged, preserving exact behavior and results.'
- What this solution (achieved 0.28616) has done: 'The changes focus on speeding up the training phase, which is the dominant cost. We increase the batch size (reducing the number of gradient updates per epoch) and lower the maximum epoch count while keeping the early‑stopping logic intact, so training still stops as soon as the validation loss plateaus. All other processing, model architecture, and I/O remain unchanged, preserving the original algorithmic behavior and final accuracy.'
- What this solution (achieved 0.28616) has done: 'The update speeds up training by using a single large batch for both training and validation, eliminating the overhead of many tiny batches while keeping the model architecture and loss unchanged. The batch size is set after the augmented dataset size is known, so all samples are processed in one step per epoch, which dramatically reduces total runtime without affecting correctness. No other logic or hyper‑parameters are altered.'
- What this solution (achieved 0.28616) has done: 'I reduced the batch size to avoid out‑of‑memory crashes, ensured the validation split always contains at least one sample, tightened early‑stopping patience and raised the epoch limit so the auto‑encoder can converge better, and coerced the predicted images to a NumPy float32 type before resizing for the submission. These fixes let the model train end‑to‑end and generate a valid submission file while nudging the RMSE toward the target score.'
- What this solution (achieved 0.28616) has done: 'The changes raise the batch size to use the whole training set in a single batch (reducing the per‑epoch overhead) and set deterministic seeds for reproducibility. The dataset pipeline is kept identical aside from the larger batch, and all other logic—including model architecture, augmentation, and loss—remains unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile, os, cv2, random
from tqdm.auto import tqdm

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks

if tf.config.list_physical_devices("GPU"):
    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)

tf.config.threading.set_inter_op_parallelism_threads(
    tf.config.threading.get_inter_op_parallelism_threads()
)
tf.config.threading.set_intra_op_parallelism_threads(
    tf.config.threading.get_intra_op_parallelism_threads()
)

try:
    import imgaug as ia
    from imgaug import augmenters as iaa

    IMG_AUG_AVAILABLE = True
except Exception:
    IMG_AUG_AVAILABLE = False

sns.set_style("darkgrid")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1286080936.py in <cell line: 0>()
     12 np.random.seed(seed)
     13 random.seed(seed)
---> 14 tf.random.set_seed(seed)
     15 
     16 try:

NameError: name 'tf' is not defined

## === cell 1
path_zip = "/kaggle/input/denoising-dirty-documents/"
path = "/kaggle/working/"

for fname in ["train.zip", "test.zip", "train_cleaned.zip", "sampleSubmission.csv.zip"]:
    with zipfile.ZipFile(os.path.join(path_zip, fname), "r") as zip_ref:
        zip_ref.extractall(path)

train_img = sorted(os.listdir(os.path.join(path, "train")))
train_cleaned_img = sorted(os.listdir(os.path.join(path, "train_cleaned")))
test_img = sorted(os.listdir(os.path.join(path, "test")))




## === cell 2
class config:
    IMG_SIZE = (420, 540)  # height, width




## === cell 3
def load_dataset(file_list, folder):
    """Efficiently load images as float16 tensors of shape (N, H, W, 1)."""
    N = len(file_list)
    H, W = config.IMG_SIZE
    data = np.empty((N, H, W, 1), dtype=np.float16)
    for i, f in enumerate(file_list):
        img = cv2.imread(os.path.join(path, folder, f), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, config.IMG_SIZE[::-1])  # width, height order
        img = img.astype(np.float16) / 255.0
        data[i, :, :, 0] = img
    return data




## === cell 4
train = load_dataset(train_img, "train")
train_cleaned = load_dataset(train_cleaned_img, "train_cleaned")
test = load_dataset(test_img, "test")

print(
    "Shapes -> train:",
    train.shape,
    "cleaned:",
    train_cleaned.shape,
    "test:",
    test.shape,
)




## === cell 5
fig, ax = plt.subplots(4, 2, figsize=(12, 16))
for i in range(4):
    ax[i, 0].imshow(tf.squeeze(train[i]), cmap="gray")
    ax[i, 0].set_title(f"Noisy: {train_img[i]}")
    ax[i, 1].imshow(tf.squeeze(train_cleaned[i]), cmap="gray")
    ax[i, 1].set_title(f"Clean: {train_cleaned_img[i]}")
    for a in ax[i]:
        a.axis("off")
plt.close(fig)  # close to avoid display in non‑interactive env




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3903217097.py in <cell line: 0>()
      1 fig, ax = plt.subplots(4, 2, figsize=(12, 16))
      2 for i in range(4):
----> 3     ax[i, 0].imshow(tf.squeeze(train[i]), cmap="gray")
      4     ax[i, 0].set_title(f"Noisy: {train_img[i]}")
      5     ax[i, 1].imshow(tf.squeeze(train_cleaned[i]), cmap="gray")

NameError: name 'tf' is not defined

## === cell 6
def simple_augment(images):
    """Return original images plus horizontally flipped versions."""
    flipped = np.flip(images, axis=2)  # flip width dimension
    return np.concatenate([images, flipped], axis=0)


if IMG_AUG_AVAILABLE:
    seq = iaa.Sequential([iaa.Fliplr(0.5), iaa.Affine(rotate=(-10, 10))])

    def augment(images):
        return np.concatenate([images, seq.augment_images(images)], axis=0)

else:
    augment = simple_augment

train_aug = augment(train)
train_cleaned_aug = augment(train_cleaned)
print("After augmentation ->", train_aug.shape)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/839261852.py in <cell line: 0>()
      5 
      6 
----> 7 if IMG_AUG_AVAILABLE:
      8     seq = iaa.Sequential([iaa.Fliplr(0.5), iaa.Affine(rotate=(-10, 10))])
      9 

NameError: name 'IMG_AUG_AVAILABLE' is not defined

## === cell 7
class DenoisingAutoencoder(Model):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(*config.IMG_SIZE, 1)),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(
                    256, (3, 3), activation="relu", padding="same"
                ),  # extra layer
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Dropout(0.5),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Conv2D(
                    256, (3, 3), activation="relu", padding="same"
                ),  # match extra encoder layer
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.UpSampling2D((2, 2)),
                layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same"),
            ]
        )

    def call(self, x):
        return self.decoder(self.encoder(x))


autoencoder = DenoisingAutoencoder()
autoencoder.compile(
    optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"]
)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2884395325.py in <cell line: 0>()
----> 1 class DenoisingAutoencoder(Model):
      2     def __init__(self):
      3         super(DenoisingAutoencoder, self).__init__()
      4         self.encoder = tf.keras.Sequential(
      5             [

NameError: name 'Model' is not defined

## === cell 8
total_samples = train_aug.shape[0]
val_fraction = 0.1
val_count = max(1, int(total_samples * val_fraction))  # ensure at least one sample

full_ds = tf.data.Dataset.from_tensor_slices((train_aug, train_cleaned_aug))
full_ds = full_ds.shuffle(
    buffer_size=total_samples, seed=42, reshuffle_each_iteration=False
)

BATCH_SIZE = total_samples

val_ds = full_ds.take(val_count).batch(val_count).cache().prefetch(tf.data.AUTOTUNE)
train_ds = full_ds.skip(val_count).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

es = callbacks.EarlyStopping(
    monitor="val_loss", patience=30, verbose=1, restore_best_weights=True
)

lr_sched = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=10,
    verbose=1,
    min_lr=1e-6,
)

history = autoencoder.fit(
    train_ds,
    validation_data=val_ds,
    callbacks=[es, lr_sched],
    epochs=300,
    verbose=2,
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2378682374.py in <cell line: 0>()
----> 1 total_samples = train_aug.shape[0]
      2 val_fraction = 0.1
      3 val_count = max(1, int(total_samples * val_fraction))  # ensure at least one sample
      4 
      5 full_ds = tf.data.Dataset.from_tensor_slices((train_aug, train_cleaned_aug))

NameError: name 'train_aug' is not defined

## === cell 9
fig, ax = plt.subplots(figsize=(8, 4))
pd.DataFrame(history.history)[["loss", "val_loss"]].plot(ax=ax)
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.close(fig)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4256426084.py in <cell line: 0>()
      1 fig, ax = plt.subplots(figsize=(8, 4))
----> 2 pd.DataFrame(history.history)[["loss", "val_loss"]].plot(ax=ax)
      3 plt.title("Training and Validation Loss")
      4 plt.xlabel("Epoch")
      5 plt.ylabel("MSE")

NameError: name 'history' is not defined

## === cell 10
decoded_batch = autoencoder.predict(test, batch_size=48, verbose=0)  # (N, H, W, 1)

ids = []
vals = []

for i, f in enumerate(test_img):
    file_path = os.path.join(path, "test", f)
    imgid = int(f[:-4])  # numeric id from filename

    original = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    h_orig, w_orig = original.shape

    decoded_img = np.squeeze(decoded_batch[i])  # (config.IMG_SIZE)
    decoded_img = decoded_img.astype(np.float32)

    pred_resized = cv2.resize(decoded_img, (w_orig, h_orig))

    rows = np.repeat(np.arange(1, h_orig + 1), w_orig)
    cols = np.tile(np.arange(1, w_orig + 1), h_orig)
    ids.extend([f"{imgid}_{r}_{c}" for r, c in zip(rows, cols)])

    vals.extend(pred_resized.ravel().tolist())

print("Submission length:", len(ids))
submission = pd.DataFrame({"id": ids, "value": vals})
submission.to_csv("submission.csv", index=False)
print("Results saved to submission.csv!")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1740395049.py in <cell line: 0>()
----> 1 decoded_batch = autoencoder.predict(test, batch_size=48, verbose=0)  # (N, H, W, 1)
      2 
      3 ids = []
      4 vals = []
      5 

NameError: name 'autoencoder' is not defined
