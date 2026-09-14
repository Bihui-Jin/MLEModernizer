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

0.02843

# 6. Current score

0.28616

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28616) has done: 'The crash happens immediately on importing `imgaug`, due to an incompatibility between the installed `protobuf==6.33.0` and `imgaug`’s transitive use of protobuf APIs (it expects `MessageFactory.GetPrototype`, removed in newer protobuf). The smallest safe fix is to avoid importing `imgaug` entirely (it is not used anywhere in the provided cells, and keeping it always crash). This preserves the rest of the notebook’s logic (data loading, model/training, etc.) while unblocking execution deterministically. No other cells are changed.'
- What this solution (achieved 0.28616) has done: 'Diagnosis: The crash happens during TensorFlow import in cell 0 and matches a known incompatibility between TensorFlow 2.18 and protobuf 6.x, where TensorFlow expects an older protobuf API that still provides `MessageFactory.GetPrototype`. This is an environment-level mismatch, not a bug in your model/training logic. The smallest safe fix inside cell 0 is to force TensorFlow to use the pure-Python protobuf implementation (which avoids the missing symbol in the C++ implementation for this combo) by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` **before** importing TensorFlow.

Patch summary: Modify only cell 0 to set the protobuf environment variable before `import tensorflow as tf`, keeping all other imports and logic unchanged.

Updated cells: cell 0 only.

Compatibility notes for cell k+1: No variable names or interfaces change; `tf` and all Keras imports remain available exactly as before, so downstream cells run unchanged.

Assumptions: Setting the env var early is sufficient in this environment (no need to install/downgrade packages), and performance impact is acceptable since this is only to unblock execution.'
- What this solution (achieved 0.28616) has done: 'Diagnosis: The crash happens during the TensorFlow import in cell 0 due to an incompatibility between `tensorflow==2.18.0` and the installed `protobuf==6.33.0`, which triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The line setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is not sufficient to avoid this API mismatch. The minimal deterministic fix is to force TensorFlow to use the C++ protobuf runtime instead (`cpp`) before importing TensorFlow.

Patch summary: Update only the environment variable initialization in cell 0 to set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` to `"cpp"` (using `setdefault` so it won’t override an already-correct setting), leaving the rest of the logic unchanged.

Updated cells: Only cell 0 is modified.

Compatibility notes for cell k+1: No variables, imports, or interfaces used by cell 1 change; TensorFlow simply imports successfully so subsequent cells can run.

Assumptions: The runtime has the C++ protobuf implementation available (standard in Kaggle/TF builds); if not, the original error would persist, but this is the least-invasive fix consistent with the installed package versions.'
- What this solution (achieved 0.28616) has done: 'The crash happens while importing TensorFlow because the notebook forces `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="cpp"`, but the environment’s `google-protobuf` wheel does not provide the required C++ extension module (`google.protobuf.pyext._message`). The minimal fix is to stop forcing the C++ protobuf runtime and instead force the pure-Python implementation before importing TensorFlow. This keeps the rest of the notebook unchanged and only affects the TensorFlow import dependency initialization. No changes are needed for later cells.'
- What this solution (achieved 0.46786) has done: 'The crash happens immediately on importing TensorFlow because the environment pins `protobuf==6.33.0`, which is incompatible with TensorFlow 2.18’s generated protobuf code and triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. Setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"` does not fix this incompatibility for protobuf 6.x. The minimal deterministic fix is to force protobuf to use the upb (C++) implementation instead (or simply not override it), before importing TensorFlow. This keeps all downstream code unchanged and restores TensorFlow import compatibility.'
- What this solution (achieved 0.28616) has done: 'The crash happens during the TensorFlow import in cell 0 due to an incompatibility between `tensorflow==2.18.0` and the installed `protobuf==6.33.0`, which breaks descriptor/message factory APIs (the `GetPrototype` attribute is missing). The minimal fix is to pin protobuf to a TensorFlow-compatible major version (protobuf 4.x) before importing TensorFlow. This must be done inside cell 0 so that later cells can import and run unchanged. No model/training logic is altered; only the environment dependency is corrected so the notebook can execute.'
- What this solution (achieved 0.28616) has done: 'Diagnosis: Cell 8 crashes because `iaa` (and the related `imgaug` namespace) was never imported/defined in any earlier cell, so the first call to `iaa.Rot90(...)` raises a `NameError`. This cell depends on `iaa` existing to construct augmentation objects referenced later by `pipeline` in cell 9.  
Patch summary: Add a minimal, local import of `imgaug` and `imgaug.augmenters as iaa` at the start of cell 8 so the existing augmentation definitions work unchanged. No other logic is modified.  
Updated cells: Only cell 8 is updated.  
Compatibility notes for cell k+1: Cell 9 continue to work because all variables (`rotate90`, `rotate180`, `rotate270`, `hflip`, `vflip`, etc.) are still created with the same names/types.  
Assumptions: The `imgaug` package is available in the runtime (common in Kaggle images); if it is not installed, this would instead raise `ModuleNotFoundError`, but the original code clearly intends to use it.'
- What this solution (achieved 0.28616) has done: 'Diagnosis: Cell 8 fails immediately because it imports `imgaug`, but `imgaug` is not installed in the provided environment (ModuleNotFoundError). The rest of the notebook only needs augmentation objects (`rotate90`, `rotate180`, etc.) to exist so that `pipeline = [...]` in cell 9 can be constructed. To keep the existing augmentation interface unchanged without adding new dependencies, we can provide minimal no-op stand-ins that match the used API (`ia.seed`, `step.augment_images`, and `iaa.*` constructors). This unblocks execution while preserving downstream variable names and call signatures.

Patch summary: Replace the hard `imgaug` imports in cell 8 with a small fallback implementation that is used only when `imgaug` is unavailable. The fallback defines `ia.seed` and the required `iaa` augmenter classes, each returning inputs unchanged via `augment_images`. All augmenter variables (`rotate90`, `seq_rp`, etc.) are still created as before, so cell 9 and later cells can run.

Updated cells: Only cell 8 is modified.

Compatibility notes for cell k+1: Cell 9 expects `rotate90`, `rotate180`, `rotate270`, `hflip`, `vflip` to be defined; this patch keeps those names defined and usable. `augment_pipeline()` in cell 7 calls `ia.seed()` and `step.augment_images(images)`; both are provided by the fallback, matching the original interface.

Assumptions: Using identity/no-op augmentations is acceptable solely to resolve the missing dependency crash, and later code does not rely on `imgaug`-specific behaviors beyond the `augment_images` method existing.'
- What this solution (achieved 0.28616) has done: 'Your current score is far from the target (0.28616 vs 0.02843, lower is better), so we need a real-but-minimal quality improvement without changing the model architecture or training loop structure. The biggest issue is that the augmentation pipeline is currently bugged: it repeatedly augments the *original* images each step instead of chaining augmentations, and it uses `np.append` (slow and can subtly mishandle shapes). I fix `augment_pipeline` to (1) correctly chain augmentations step-by-step and (2) concatenate augmented batches safely; this preserves the same overall approach (augment N ways, then train) but produces a much better-aligned training set. I also remove an unnecessary `.numpy()` roundtrip in inference (no semantic change, just avoids dtype/graph quirks) and clip predictions to `[0,1]` before writing the submission to better match the metric’s expected intensity range.'
- What this solution (achieved 0.28616) has done: 'Your score gap to the target is large (0.28616 vs 0.02843, lower is better), so we need a real quality boost without changing the model architecture or training loop. The biggest low-risk issue is a train/label misalignment risk: you load `train/` and `train_cleaned/` independently and assume sorted filenames match, but any ordering mismatch silently trains on wrong targets and severely hurts RMSE. I enforce pairing by filename and build `train`, `train_cleaned`, and `test` from those aligned lists (core model and training unchanged). I also ensure deterministic ordering in the submission by sorting test filenames numerically and using those filenames consistently in both `test` array creation and the submission loop.'
- What this solution (achieved 0.28616) has done: 'The timeout is dominated by (1) Python-loop image loading/processing and unnecessary “median dimension” full read, (2) extremely slow nested Python loops when building the 5.7M-row submission, and (3) per-image model calls instead of batched prediction. I keep the exact model, loss, and training loop semantics, but speed up by vectorizing submission creation (NumPy meshgrid + ravel), batching inference, and using `tf.data`-equivalent prefetch behavior via `predict` with a batch size. I also make image loading/processing faster and deterministic by using OpenCV grayscale reads directly and preallocating arrays, while leaving resizing/normalization identical. Finally, I avoid re-reading all train images just to compute the median dimensions and gate plotting to prevent unnecessary work during timed runs.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import google.protobuf
from packaging.version import Version

if Version(getattr(google.protobuf, "__version__", "0")) >= Version("5.0.0"):
    import sys, subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])
    import importlib

    importlib.invalidate_caches()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import zipfile, cv2
from tqdm.auto import tqdm
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks, utils


sns.set_style("darkgrid")

SEED = 19
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")




## === cell 1
path_zip = "../input/denoising-dirty-documents/"
path = "/kaggle/working/"

with zipfile.ZipFile(path_zip + "train.zip", "r") as zip_ref:
    zip_ref.extractall(path)

with zipfile.ZipFile(path_zip + "test.zip", "r") as zip_ref:
    zip_ref.extractall(path)

with zipfile.ZipFile(path_zip + "train_cleaned.zip", "r") as zip_ref:
    zip_ref.extractall(path)

with zipfile.ZipFile(path_zip + "sampleSubmission.csv.zip", "r") as zip_ref:
    zip_ref.extractall(path)

train_files = sorted(
    [f for f in os.listdir(path + "/train") if f.lower().endswith(".png")],
    key=lambda x: int(os.path.splitext(x)[0]),
)
train_cleaned_files = sorted(
    [f for f in os.listdir(path + "/train_cleaned") if f.lower().endswith(".png")],
    key=lambda x: int(os.path.splitext(x)[0]),
)
test_files = sorted(
    [f for f in os.listdir(path + "/test") if f.lower().endswith(".png")],
    key=lambda x: int(os.path.splitext(x)[0]),
)

train_set = set(train_files)
train_cleaned_set = set(train_cleaned_files)
common = sorted(
    list(train_set.intersection(train_cleaned_set)),
    key=lambda x: int(os.path.splitext(x)[0]),
)

missing_in_cleaned = sorted(
    list(train_set - train_cleaned_set), key=lambda x: int(os.path.splitext(x)[0])
)
missing_in_train = sorted(
    list(train_cleaned_set - train_set), key=lambda x: int(os.path.splitext(x)[0])
)

if len(missing_in_cleaned) > 0 or len(missing_in_train) > 0:
    print(f"Warning: train/train_cleaned filename mismatch. Using intersection only.")
    print(
        f"Missing in train_cleaned (count={len(missing_in_cleaned)}): {missing_in_cleaned[:10]}"
    )
    print(f"Missing in train (count={len(missing_in_train)}): {missing_in_train[:10]}")

train_img = common
train_cleaned_img = common
test_img = test_files




## === cell 2
class config:
    IMG_SIZE = (420, 540)


_sample_paths = [path + "train/" + f for f in train_img[:8]]
_sample_imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in _sample_paths]
print(
    "Sample Dimensions (H,W):",
    [(im.shape[0], im.shape[1]) for im in _sample_imgs if im is not None][:5],
)
del _sample_paths, _sample_imgs




## === cell 3
def process_image(path_):
    img = cv2.imread(path_, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, config.IMG_SIZE[::-1], interpolation=cv2.INTER_LINEAR)
    img = img.astype("float32") / 255.0
    img = img.reshape((*config.IMG_SIZE, 1))
    return img




## === cell 4
train = np.empty((len(train_img), *config.IMG_SIZE, 1), dtype=np.float32)
train_cleaned = np.empty(
    (len(train_cleaned_img), *config.IMG_SIZE, 1), dtype=np.float32
)
test = np.empty((len(test_img), *config.IMG_SIZE, 1), dtype=np.float32)

for i, f in enumerate(train_img):
    train[i] = process_image(path + "train/" + f)

for i, f in enumerate(train_cleaned_img):
    train_cleaned[i] = process_image(path + "train_cleaned/" + f)

for i, f in enumerate(test_img):
    test[i] = process_image(path + "test/" + f)




## === cell 5
train.shape, train_cleaned.shape, test.shape




## === cell 6
DO_PLOTS = False

if DO_PLOTS:
    fig, ax = plt.subplots(4, 2, figsize=(15, 25))
    for i in range(4):
        ax[i][0].imshow(tf.squeeze(train[i]), cmap="gray")
        ax[i][0].set_title("Noise image: {}".format(train_img[i]))

        ax[i][1].imshow(tf.squeeze(train_cleaned[i]), cmap="gray")
        ax[i][1].set_title("Denoised image: {}".format(train_img[i]))

        ax[i][0].get_xaxis().set_visible(False)
        ax[i][0].get_yaxis().set_visible(False)
        ax[i][1].get_xaxis().set_visible(False)
        ax[i][1].get_yaxis().set_visible(False)




## === cell 7
def augment_pipeline(pipeline, images, seed=19):
    ia.seed(seed)
    processed_images = [images.copy()]
    current = images
    for step in pipeline:
        current = np.array(step.augment_images(current))
        processed_images.append(current)
    return np.concatenate(processed_images, axis=0)




## === cell 8
try:
    import imgaug as ia
    import imgaug.augmenters as iaa
except ModuleNotFoundError:

    class _NoOpAugmenter:
        def __init__(self, *args, **kwargs):
            pass

        def augment_images(self, images):
            return images

    class _NoOpIA:
        @staticmethod
        def seed(seed):
            return None

    class _NoOpIAA:
        Rot90 = _NoOpAugmenter
        PerspectiveTransform = _NoOpAugmenter
        Affine = _NoOpAugmenter
        Crop = _NoOpAugmenter
        Fliplr = _NoOpAugmenter
        Flipud = _NoOpAugmenter
        GaussianBlur = _NoOpAugmenter
        MotionBlur = _NoOpAugmenter

        class Sequential(_NoOpAugmenter):
            def __init__(self, children=None, *args, **kwargs):
                self.children = children or []

            def augment_images(self, images):
                out = images
                for child in self.children:
                    out = child.augment_images(out)
                return out

    ia = _NoOpIA()
    iaa = _NoOpIAA()

rotate90 = iaa.Rot90(1)  # rotate image 90 degrees
rotate180 = iaa.Rot90(2)  # rotate image 180 degrees
rotate270 = iaa.Rot90(3)  # rotate image 270 degrees
random_rotate = iaa.Rot90((1, 3))  # randomly rotate image from 90,180,270 degrees
perc_transform = iaa.PerspectiveTransform(
    scale=(0.02, 0.1)
)  # Skews and transform images without black bg
rotate10 = iaa.Affine(rotate=(10))  # rotate image 10 degrees
rotate10r = iaa.Affine(rotate=(-10))  # rotate image 30 degrees in reverse
crop = iaa.Crop(px=(5, 32))  # Crop between 5 to 32 pixels
hflip = iaa.Fliplr(1)  # horizontal flips for 100% of images
vflip = iaa.Flipud(1)  # vertical flips for 100% of images
gblur = iaa.GaussianBlur(
    sigma=(1, 1.5)
)  # gaussian blur images with a sigma of 1.0 to 1.5
motionblur = iaa.MotionBlur(8)  # motion blur images with a kernel size 8

seq_rp = iaa.Sequential(
    [
        iaa.Rot90((1, 3)),  # randomly rotate image from 90,180,270 degrees
        iaa.PerspectiveTransform(
            scale=(0.02, 0.1)
        ),  # Skews and transform images without black bg
    ]
)

seq_cfg = iaa.Sequential(
    [
        iaa.Crop(
            px=(5, 32)
        ),  # crop images from each side by 5 to 32px (randomly chosen)
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 1.5)),  # blur images with a sigma of 0 to 1.5
    ]
)

seq_fm = iaa.Sequential(
    [
        iaa.Flipud(1),  # vertical flips all the images
        iaa.MotionBlur(k=6),  # motion blur images with a kernel size 6
    ]
)




## === cell 9
pipeline = [rotate90, rotate180, rotate270, hflip, vflip]




## === cell 10
processed_train = augment_pipeline(pipeline, train, seed=SEED)
processed_train_cleaned = augment_pipeline(pipeline, train_cleaned, seed=SEED)

processed_train.shape, processed_train_cleaned.shape




## === cell 11
class DenoisingAutoencoder(Model):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(*config.IMG_SIZE, 1)),
                layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Dropout(0.5),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.UpSampling2D((2, 2)),
                layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same"),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = DenoisingAutoencoder()
autoencoder.compile(
    optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"]
)




## === cell 12
es = callbacks.EarlyStopping(
    monitor="loss", patience=30, verbose=1, restore_best_weights=True
)

history = autoencoder.fit(
    processed_train,
    processed_train_cleaned,
    shuffle=True,
    callbacks=[es],
    epochs=500,
    batch_size=12,
    verbose=1,
)




## === cell 13
if DO_PLOTS:
    fig, ax = plt.subplots(figsize=(20, 6))
    pd.DataFrame(history.history).plot(ax=ax)
del history




## === cell 14
if DO_PLOTS:
    autoencoder.encoder.summary()
    autoencoder.decoder.summary()




## === cell 15
if DO_PLOTS:
    decoded_imgs = autoencoder(train[:4]).numpy()

    fig, ax = plt.subplots(4, 2, figsize=(15, 25))
    for i in range(4):
        ax[i][0].imshow(tf.squeeze(train_cleaned[i]), cmap="gray")
        ax[i][0].set_title("Denoised image: {}".format(train_img[i]))

        ax[i][1].imshow(tf.squeeze(decoded_imgs[i]), cmap="gray")
        ax[i][1].set_title("Predicted image: {}".format(train_img[i]))

        ax[i][0].get_xaxis().set_visible(False)
        ax[i][0].get_yaxis().set_visible(False)
        ax[i][1].get_xaxis().set_visible(False)
        ax[i][1].get_yaxis().set_visible(False)

    del decoded_imgs




## === cell 16
BATCH = 8  # does not change outputs, only throughput

pred_test = autoencoder.predict(test, batch_size=BATCH, verbose=0)  # (N, 420, 540, 1)
pred_test = np.clip(pred_test[..., 0], 0.0, 1.0)  # (N, 420, 540)

ids_chunks = []
vals_chunks = []

for i, f in tqdm(list(enumerate(test_img))):
    file = path + "test/" + f
    imgid = int(f[:-4])

    img0 = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    h0, w0 = img0.shape

    preds_reshaped = cv2.resize(pred_test[i], (w0, h0), interpolation=cv2.INTER_LINEAR)
    preds_reshaped = np.clip(preds_reshaped, 0.0, 1.0)

    rr, cc = np.meshgrid(np.arange(1, h0 + 1), np.arange(1, w0 + 1), indexing="ij")
    ids = (
        np.char.add(
            np.char.add(
                np.char.add(str(imgid), "_"),
                rr.astype(np.int32).astype(str),
            ),
            np.char.add("_", cc.astype(np.int32).astype(str)),
        )
        .ravel()
        .tolist()
    )
    vals = preds_reshaped.ravel().tolist()

    ids_chunks.append(ids)
    vals_chunks.append(vals)

ids = [x for chunk in ids_chunks for x in chunk]
vals = [x for chunk in vals_chunks for x in chunk]

print("Length of IDs: {}".format(len(ids)))
pd.DataFrame({"id": ids, "value": vals}).to_csv("submission.csv", index=False)
print("Results saved to submission.csv!")
