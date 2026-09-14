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

3.7

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
PyYAML==6.0.3
sklearn-pandas==2.2.0
tf_keras==2.18.0

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

0.14197

# 6. Current score

0.08954

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.28608) has done: 'I fix the environment/import crash by switching to `tf_keras` (TensorFlow Keras) instead of standalone `keras==3.x`, which is triggering the protobuf `MessageFactory.GetPrototype` error. I also fix the “inhomogeneous shape” `ValueError` by not forcing variable-sized images into a single NumPy array; instead the generator keep lists and batch only same-shaped images. Next, I update deprecated `fit_generator` to `fit` (same semantics) and ensure all images are consistently normalized to `[0,1]` for both training and inference. Finally, I write a valid `submission.csv` with correct `id,value` rows, iterating through images without trying to `np.array()` them.'
- What this solution (achieved 0.0868) has done: 'I fix the protobuf/Keras crash by forcing TensorFlow’s bundled Keras (`tf_keras`) to be used consistently and by pinning the protobuf Python implementation to avoid the `MessageFactory.GetPrototype` issue in this Kaggle image. Then I fix a real shape bug in `Autoencoder.get_encoder()/get_decoder()` (it currently calls `__init__` and returns `None`, so encoder/decoder creation is broken) while keeping the same architecture and weights. Finally, I make submission writing both faster and safer by writing in the exact `id,value` order from `sampleSubmission.csv` (prevents ordering/alignment mistakes and typically improves RMSE vs. accidental misalignment), still using the same model predictions.'
- What this solution (achieved 0.08984) has done: 'I fix the immediate runtime crash caused by a protobuf/TensorFlow incompatibility that happens during `tf_keras` import by setting the protobuf implementation to `python` before any TensorFlow-related import and by forcing a safe protobuf version range if available in the environment. I also make the imports resilient by falling back between `tf_keras` and `tensorflow.keras` (same API/semantics) to ensure the model builds identically and training/inference run end-to-end. Since your current score (0.0868, lower-is-better) is already significantly better than the target (0.14197), I not make any changes that would intentionally improve performance; all changes are stability-only. The script still write `submission.csv` in the exact `sampleSubmission.csv` order with correct `id,value` columns.'
- What this solution (achieved 0.08521) has done: 'The crash happens before any model code runs: importing `tf_keras`/`tensorflow.keras` triggers a protobuf API mismatch (`MessageFactory.GetPrototype`) in this Kaggle image. I fix this by forcing the pure‑Python protobuf runtime early and (crucially) ensuring TensorFlow is imported before any Keras symbols are imported, which prevents the incompatible fast-protobuf path from being selected. I also remove the standalone `keras==3.x` dependency from the import path entirely (it’s not needed here) and keep the model/training/prediction logic unchanged to avoid moving your score further away from the target. Finally, the script still write `submission.csv` with the exact `id,value` ordering from `sampleSubmission.csv`.'
- What this solution (achieved 0.08954) has done: 'The runtime crash is happening before your model code runs due to a TensorFlow/protobuf incompatibility (`MessageFactory.GetPrototype`) during the TensorFlow import path. I fix this by forcing the pure-Python protobuf runtime and (critically) forcing TensorFlow to use the legacy protobuf API implementation before importing TensorFlow, which avoids the failing codepath in this Kaggle image. I not change the model architecture, data pipeline, training loop, or prediction logic, so the score behavior should remain essentially the same (still better than your target). I also keep the submission writing exactly aligned to `sampleSubmission.csv` and ensure the output filename ends with `.csv`.'

# 9. Code solution

## === cell 0
import os, glob, csv

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TF_USE_LEGACY_PROTOBUF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)

import tensorflow as tf

try:
    import tf_keras as keras
    from tf_keras.layers import (
        Conv2D,
        UpSampling2D,
        MaxPooling2D,
        Input,
        Conv2DTranspose,
    )
    from tf_keras.models import Model
except Exception:
    from tensorflow import keras
    from tensorflow.keras.layers import (
        Conv2D,
        UpSampling2D,
        MaxPooling2D,
        Input,
        Conv2DTranspose,
    )
    from tensorflow.keras.models import Model



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
INPUT_ROOT = "/kaggle/input/denoising-dirty-documents"
TRAIN_DIR = os.path.join(INPUT_ROOT, "train")
TRAIN_CLEAN_DIR = os.path.join(INPUT_ROOT, "train_cleaned")
TEST_DIR = os.path.join(INPUT_ROOT, "test")

train_paths = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.png")))
clean_paths = sorted(glob.glob(os.path.join(TRAIN_CLEAN_DIR, "*.png")))
test_paths = sorted(glob.glob(os.path.join(TEST_DIR, "*.png")))

assert (
    len(train_paths) == len(clean_paths) and len(train_paths) > 0
), "Train and train_cleaned mismatch or empty."
assert len(test_paths) > 0, "Test images not found."


def read_img(path, as_float=True):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if as_float:
        img = img.astype("float32") / 255.0
    return img




## === cell 2
X_vis = [read_img(p, as_float=False) for p in train_paths[:4]]
y_vis = [read_img(p, as_float=False) for p in clean_paths[:4]]

plt.figure(figsize=(16, 8))
for i in range(4):
    plt.subplot(2, 4, i + 1)
    plt.imshow(cv2.cvtColor(X_vis[i], cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.subplot(2, 4, 4 + i + 1)
    plt.imshow(cv2.cvtColor(y_vis[i], cv2.COLOR_BGR2RGB))
    plt.axis("off")
plt.show()



## === cell 3
shapes = np.unique([read_img(p, as_float=False).shape for p in train_paths], axis=0)
print("Unique train shapes:", shapes)




## === cell 4
class Autoencoder:
    def __init__(
        self,
        dimensions_factor=2,
        layers=2,
        k=3,
        filter_size=None,
        pooling_factor=None,
        only_decoder=False,
        only_encoder=False,
        loss="mean_squared_error",
        channels=3,
    ):

        if not (only_decoder or only_encoder):
            self.channels = channels
            self.layers = layers
            self.k = k
            self.filter_size = (
                [(k, k)] * (layers * 2 + 1) if (filter_size is None) else filter_size
            )
            self.pooling_factor = (
                [2] * (layers * 2) if (pooling_factor is None) else pooling_factor
            )
            self.dimensions_factor = dimensions_factor
            self.model = None

        if not only_decoder:
            input_img = Input(shape=(None, None, self.channels))
            x = input_img
            for i in range(self.layers):
                filters = int(
                    self.pooling_factor[i] * int(x.shape[-1]) * self.dimensions_factor
                )
                x = Conv2D(
                    filters,
                    self.filter_size[i],
                    activation="relu",
                    padding="same",
                    name="encoding_conv_" + str(i),
                )(x)
                x = MaxPooling2D(
                    (self.pooling_factor[i], self.pooling_factor[i]),
                    padding="valid",
                    name="encoding_pool_" + str(i),
                )(x)
            self.code_filters = filters
        else:
            input_img = Input(shape=(None, None, self.code_filters))
            x = input_img

        if not only_encoder:
            for i in range(self.layers, 2 * self.layers):
                filters = int(
                    int(x.shape[-1]) / (self.pooling_factor[i]) * self.dimensions_factor
                )
                x = Conv2DTranspose(
                    filters,
                    self.filter_size[i],
                    activation="relu",
                    padding="same",
                    name="decoding_conv_" + str(i),
                )(x)
                x = UpSampling2D(
                    (self.pooling_factor[i], self.pooling_factor[i]),
                    name="decoding_pool_" + str(i),
                )(x)

            x = Conv2D(
                self.channels,
                (self.filter_size[-1]),
                activation="sigmoid",
                padding="same",
                name="decoded",
            )(x)

        autoencoder = Model(input_img, x)
        autoencoder.compile(optimizer="adam", loss=loss, metrics=["mse"])

        if not (only_decoder or only_encoder):
            self.model = autoencoder
        else:
            return autoencoder

    def _build_encoder_model(self):
        input_img = Input(shape=(None, None, self.channels))
        x = input_img
        for i in range(self.layers):
            filters = int(
                self.pooling_factor[i] * int(x.shape[-1]) * self.dimensions_factor
            )
            x = Conv2D(
                filters,
                self.filter_size[i],
                activation="relu",
                padding="same",
                name="encoding_conv_" + str(i),
            )(x)
            x = MaxPooling2D(
                (self.pooling_factor[i], self.pooling_factor[i]),
                padding="valid",
                name="encoding_pool_" + str(i),
            )(x)
        encoder = Model(input_img, x)
        return encoder

    def _build_decoder_model(self):
        input_img = Input(shape=(None, None, self.code_filters))
        x = input_img
        for i in range(self.layers, 2 * self.layers):
            filters = int(
                int(x.shape[-1]) / (self.pooling_factor[i]) * self.dimensions_factor
            )
            x = Conv2DTranspose(
                filters,
                self.filter_size[i],
                activation="relu",
                padding="same",
                name="decoding_conv_" + str(i),
            )(x)
            x = UpSampling2D(
                (self.pooling_factor[i], self.pooling_factor[i]),
                name="decoding_pool_" + str(i),
            )(x)

        x = Conv2D(
            self.channels,
            (self.filter_size[-1]),
            activation="sigmoid",
            padding="same",
            name="decoded",
        )(x)
        decoder = Model(input_img, x)
        return decoder

    def get_encoder(self):
        if self.model is None:
            raise RuntimeError("Model is not created")
        encoder = self._build_encoder_model()
        for layer in encoder.layers:
            if layer.name in [l.name for l in self.model.layers]:
                weights = self.model.get_layer(name=layer.name).get_weights()
                if weights:
                    layer.set_weights(weights)
        return encoder

    def get_decoder(self):
        if self.model is None:
            raise RuntimeError("Model is not created")
        decoder = self._build_decoder_model()
        for layer in decoder.layers:
            if layer.name in [l.name for l in self.model.layers]:
                weights = self.model.get_layer(name=layer.name).get_weights()
                if weights:
                    layer.set_weights(weights)
        return decoder




## === cell 5
class Image_generator:
    """
    Fix: don't convert variable-sized images into a single numpy array.
    Keep lists of images; batch indices are built by grouping images with same shape.
    """

    def __init__(
        self, base_path, val_percentage=0.115, batch_size=4, reduction_factor=4
    ):
        self.base_path = base_path
        self.batch_size = batch_size
        self.reduction_factor = reduction_factor

        self.X = [read_img(p, as_float=True) for p in train_paths]
        self.y = [read_img(p, as_float=True) for p in clean_paths]

        self.shape_to_indices = {}
        for i, img in enumerate(self.X):
            self.shape_to_indices.setdefault(img.shape, []).append(i)

        self.idx_train, self.idx_val = self.split_batches(val_percentage)
        self.train_steps = len(self.idx_train)
        self.val_steps = len(self.idx_val)

        if self.train_steps == 0 or self.val_steps == 0:
            raise RuntimeError(
                "Not enough data to form train/val batches. Try reducing batch_size."
            )

    def split_batches(self, val_percentage):
        batches = []
        for shape, idxs in self.shape_to_indices.items():
            idxs = np.array(idxs, dtype=int)
            np.random.shuffle(idxs)

            n_full = (len(idxs) // self.batch_size) * self.batch_size
            idxs = idxs[:n_full]
            if len(idxs) == 0:
                continue
            shape_batches = idxs.reshape(-1, self.batch_size)
            batches.extend(list(shape_batches))

        batches = np.array(batches, dtype=int)
        np.random.shuffle(batches)

        samples = len(batches)
        val_samples = int(samples * val_percentage)
        val_samples = max(1, val_samples)
        train_samples = samples - val_samples
        train_samples = max(1, train_samples)

        return (
            batches[:train_samples],
            batches[train_samples : train_samples + val_samples],
        )

    def check_size(self, batch_x, batch_y, axis):
        for each_axis in axis:
            while (batch_x.shape[each_axis] % self.reduction_factor) != 0:
                batch_x = np.insert(
                    batch_x, batch_x.shape[each_axis], 1.0, axis=each_axis
                )
                batch_y = np.insert(
                    batch_y, batch_y.shape[each_axis], 1.0, axis=each_axis
                )
        return batch_x, batch_y

    def get_train_batch(self):
        i = 0
        while True:
            idxs = self.idx_train[i]
            batch_x = np.stack([self.X[j] for j in idxs], axis=0)
            batch_y = np.stack([self.y[j] for j in idxs], axis=0)
            batch_x, batch_y = self.check_size(batch_x, batch_y, [-2, -3])
            i = (i + 1) % len(self.idx_train)
            yield batch_x, batch_y

    def get_val_batch(self):
        i = 0
        while True:
            idxs = self.idx_val[i]
            batch_x = np.stack([self.X[j] for j in idxs], axis=0)
            batch_y = np.stack([self.y[j] for j in idxs], axis=0)
            batch_x, batch_y = self.check_size(batch_x, batch_y, [-2, -3])
            i = (i + 1) % len(self.idx_val)
            yield batch_x, batch_y




## === cell 6
data_generator = Image_generator(os.getcwd(), batch_size=4, reduction_factor=4)
autoencoder = Autoencoder(loss="mse")

log = autoencoder.model.fit(
    x=data_generator.get_train_batch(),
    steps_per_epoch=data_generator.train_steps,
    epochs=50,
    shuffle=False,
    validation_data=data_generator.get_val_batch(),
    validation_steps=data_generator.val_steps,
    workers=1,
    use_multiprocessing=False,
    verbose=2,
)



## === cell 7
plt.figure()
for k, v in log.history.items():
    plt.plot(v, label=str(k))
plt.legend()
plt.show()



## === cell 8
SAMPLE_SUB_PATH = os.path.join(INPUT_ROOT, "sampleSubmission.csv")


def predict_one_image_bgr_to_gray_values(path, model, reduction_factor=4):
    x = read_img(path, as_float=True)
    original_h, original_w, _ = x.shape

    while (x.shape[0] % reduction_factor) != 0:
        x = np.insert(x, x.shape[0], 1.0, axis=0)
    while (x.shape[1] % reduction_factor) != 0:
        x = np.insert(x, x.shape[1], 1.0, axis=1)

    x_in = x.reshape((1,) + x.shape)
    pred = model.predict(x_in, verbose=0)[0]
    pred = pred[:original_h, :original_w, :]
    pred_gray = pred.mean(axis=2).astype("float32")
    return pred_gray


test_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_paths]
id_to_path = {tid: p for tid, p in zip(test_ids, test_paths)}

sub = pd.read_csv(SAMPLE_SUB_PATH)
assert list(sub.columns) == ["id", "value"]

parts = sub["id"].str.split("_", expand=True)
sub_img = parts[0]
sub_r = parts[1].astype(np.int32) - 1
sub_c = parts[2].astype(np.int32) - 1

pred_cache = {}
values = np.empty(len(sub), dtype="float32")

unique_imgs = sub_img.unique().tolist()
for n, img_id in enumerate(unique_imgs, start=1):
    p = id_to_path.get(img_id)
    if p is None:
        raise KeyError(
            f"Image id {img_id} from sampleSubmission not found in test set."
        )
    pred_cache[img_id] = predict_one_image_bgr_to_gray_values(
        p, autoencoder.model, reduction_factor=4
    )
    print(f"{n} of {len(unique_imgs)} predictions calculated", end="\r")
print()

for img_id in unique_imgs:
    mask = (sub_img == img_id).to_numpy()
    g = pred_cache[img_id]
    rr = sub_r[mask].to_numpy()
    cc = sub_c[mask].to_numpy()
    values[mask] = g[rr, cc]

sub["value"] = values
out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print(f"Saved {out_path} with shape={sub.shape}")



## === cell 9
encoder = autoencoder.get_encoder()

example = read_img(test_paths[min(10, len(test_paths) - 1)], as_float=True)
example_in = example.reshape((1,) + example.shape)
filters = encoder.predict(example_in, verbose=0)

plt.figure(figsize=(5, 5))
plt.imshow(cv2.cvtColor((example * 255).astype("uint8"), cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()



## === cell 10
import matplotlib.animation as animation
from IPython.display import HTML

fig = plt.figure(figsize=(6, 6))
ims = []
n_filters = min(filters.shape[-1], 16)
for i in range(n_filters):
    im = plt.imshow(filters[0, :, :, i], animated=True, cmap="viridis")
    plt.axis("off")
    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=300, blit=True, repeat_delay=0)



## === cell 11
HTML(ani.to_jshtml())
