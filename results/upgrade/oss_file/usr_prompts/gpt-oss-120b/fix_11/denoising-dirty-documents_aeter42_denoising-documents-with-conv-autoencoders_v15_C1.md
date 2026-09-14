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

0.19042

# 6. Current score

0.09768

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.42814) has done: 'I replace the Keras imports with TensorFlow‑Keras to avoid the protobuf error, drop the unsupported `use_multiprocessing` argument from `model.fit`, and adjust the prediction‑to‑CSV routine so it handles images of different sizes without forcing a homogeneous NumPy array. These fixes eliminate the runtime failures and ensure a proper `submission.csv` is written, keeping the original model architecture unchanged.'
- What this solution (achieved 0.28616) has done: 'I set the protobuf implementation to the pure‑Python version before TensorFlow is imported, change the data generator to use a batch size of 1 (so each batch has a single image and Keras accepts varying image dimensions), and train a few more epochs to lower the RMSE. These minimal fixes remove the runtime errors, allow the model to train on the full dataset, and should move the validation score closer to the target while keeping the original architecture unchanged.'
- What this solution (achieved 0.14069) has done: 'The fixes move the protobuf setting before any imports, replace the image loader with a consistent‑size dataset (padding all training images to the same height/width), and add a tiny blur filter to the predictions to improve denoising performance. These changes resolve the generator shape error, allow the model to train, and nudge the RMSE toward the target while preserving the original architecture.'
- What this solution (achieved 0.13968) has done: 'The fix replaces the TensorFlow‑Keras imports with pure Keras imports to avoid the protobuf `MessageFactory` error that stopped model construction. The same layers and Model class are used, keeping the original architecture unchanged, so training and inference work as before and a proper `submission.csv` is produced.'
- What this solution (achieved 0.28616) has done: 'I replace the pure‑Keras imports with TensorFlow‑Keras imports to fix the protobuf `MessageFactory` error, and lower the number of training epochs from 30 to 10 so the validation RMSE moves from the current 0.14 toward the target ≈ 0.19 without altering the model architecture or other logic. The rest of the pipeline, including data padding, generator, and CSV creation, remains unchanged.'
- What this solution (achieved 0.1415) has done: 'I fix the protobuf import error by importing TensorFlow right after setting the protobuf implementation flag and then raise the training length from 10 to 20 epochs so the model can learn more and lower the RMSE toward the target. No other logic is altered.'
- What this solution (achieved 0.14136) has done: 'I remove the direct TensorFlow import that triggers the protobuf error and keep only the environment flag setup before any other imports. This eliminates the `MessageFactory` issue while preserving the existing Keras‑based autoencoder logic and the rest of the pipeline, ensuring a valid `submission.csv` is produced without altering model performance.'
- What this solution (achieved 0.28616) has done: 'I replace the TensorFlow‑Keras imports with pure Keras imports to fix the protobuf error, lower the training epochs from 20 to 5 (which raises the RMSE toward the target), and remove the post‑prediction blur that was overly smoothing the output. These minimal changes keep the original architecture intact while ensuring the script runs end‑to‑end and produces a valid submission.csv with a slightly higher (worse) score that falls within the target tolerance band.'
- What this solution (achieved 0.09768) has done: 'I fixed the protobuf import error by switching all Keras imports to `tensorflow.keras`, which works correctly after setting the protobuf implementation flag. I also increased the training epochs from 5 to 12 so the model can learn more and lower the RMSE toward the target value while keeping the core architecture unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import glob, csv
import numpy as np
import cv2
import matplotlib.pyplot as plt




## === cell 1
X_sample = [
    cv2.imread(p)
    for p in sorted(glob.glob(os.path.join(os.getcwd(), "../input/train/*.png")))[:4]
]
y_sample = [
    cv2.imread(p)
    for p in sorted(
        glob.glob(os.path.join(os.getcwd(), "../input/train_cleaned/*.png"))
    )[:4]
]




## === cell 2
plt.figure(figsize=(16, 8))
for i in range(len(X_sample)):
    plt.subplot(2, 4, i + 1)
    plt.imshow(cv2.cvtColor(X_sample[i], cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.subplot(2, 4, i + 5)
    plt.imshow(cv2.cvtColor(y_sample[i], cv2.COLOR_BGR2RGB))
    plt.axis("off")
plt.show()




## === cell 3
shapes = np.unique(
    [
        cv2.imread(p).shape
        for p in glob.glob(os.path.join(os.getcwd(), "../input/train/*.png"))
    ],
    axis=0,
)
print("Unique shapes in training data:", shapes)




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
        from tensorflow.keras.layers import (
            Conv2D,
            Conv2DTranspose,
            MaxPooling2D,
            UpSampling2D,
            Input,
        )
        from tensorflow.keras.models import Model

        if not (only_decoder or only_encoder):
            self.channels = channels
            self.layers = layers
            self.k = k
            self.filter_size = (
                [(k, k)] * (layers * 2 + 1) if filter_size is None else filter_size
            )
            self.pooling_factor = (
                [2] * (layers * 2) if pooling_factor is None else pooling_factor
            )
            self.dimensions_factor = dimensions_factor

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
                self.filter_size[-1],
                activation="sigmoid",
                padding="same",
                name="decoded",
            )(x)

        autoencoder = Model(input_img, x)
        autoencoder.compile(optimizer="adam", loss=loss, metrics=["mse"])

        self.model = autoencoder
        self._keras_layers = autoencoder.layers

    def get_encoder(self):
        if not hasattr(self, "model"):
            raise RuntimeError("Model not created yet")
        from tensorflow.keras.layers import Input
        from tensorflow.keras.models import Model

        input_img = Input(shape=(None, None, self.channels))
        x = input_img
        for i in range(self.layers):
            layer = self.model.get_layer(name="encoding_conv_" + str(i))
            x = layer(x)
            pool = self.model.get_layer(name="encoding_pool_" + str(i))
            x = pool(x)
        encoder = Model(input_img, x)
        return encoder

    def get_decoder(self):
        if not hasattr(self, "model"):
            raise RuntimeError("Model not created yet")
        from tensorflow.keras.layers import Input
        from tensorflow.keras.models import Model

        input_img = Input(shape=(None, None, self.code_filters))
        x = input_img
        for i in range(self.layers, 2 * self.layers):
            layer = self.model.get_layer(name="decoding_conv_" + str(i))
            x = layer(x)
            up = self.model.get_layer(name="decoding_pool_" + str(i))
            x = up(x)
        decoded = self.model.get_layer(name="decoded")(x)
        decoder = Model(input_img, decoded)
        return decoder




## === cell 5
class Image_generator:
    def __init__(
        self, base_path, val_percentage=0.115, batch_size=1, reduction_factor=4
    ):
        self.base_path = base_path
        self.batch_size = batch_size
        self.reduction_factor = reduction_factor

        self.y = [
            cv2.imread(p).astype("float32") / 255
            for p in sorted(
                glob.glob(os.path.join(base_path, "../input/train_cleaned/*.png"))
            )
        ]
        self.X = [
            cv2.imread(p).astype("float32") / 255
            for p in sorted(glob.glob(os.path.join(base_path, "../input/train/*.png")))
        ]

        max_h = max(img.shape[0] for img in self.X)
        max_w = max(img.shape[1] for img in self.X)

        def pad_to(img, target_h, target_w):
            pad_h = target_h - img.shape[0]
            pad_w = target_w - img.shape[1]
            padded = np.pad(
                img,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=1.0,
            )
            return padded

        self.X = [pad_to(img, max_h, max_w) for img in self.X]
        self.y = [pad_to(img, max_h, max_w) for img in self.y]

        self.idx_train, self.idx_val = self._split_batches(val_percentage)

        self.train_steps = len(self.idx_train)
        self.val_steps = len(self.idx_val)

    def _split_batches(self, val_percentage):
        batches = []
        shapes = np.unique([img.shape for img in self.X], axis=0)
        for shape in shapes:
            idxs = [i for i, img in enumerate(self.X) if img.shape == tuple(shape)]
            np.random.shuffle(idxs)
            for i in range(0, len(idxs) - len(idxs) % self.batch_size, self.batch_size):
                batches.append(np.array(idxs[i : i + self.batch_size]))
        np.random.shuffle(batches)
        total = len(batches)
        val_cnt = int(total * val_percentage)
        return np.array(batches[:-val_cnt]), np.array(batches[-val_cnt:])

    def _check_size(self, batch_x, batch_y, axes):
        for ax in axes:
            while batch_x.shape[ax] % self.reduction_factor != 0:
                batch_x = np.insert(batch_x, batch_x.shape[ax], 1, axis=ax)
                batch_y = np.insert(batch_y, batch_y.shape[ax], 1, axis=ax)
        return batch_x, batch_y

    def get_train_batch(self):
        while True:
            batch_idx = self.idx_train[0]
            batch_x = np.stack([self.X[i] for i in batch_idx])
            batch_y = np.stack([self.y[i] for i in batch_idx])
            batch_x, batch_y = self._check_size(batch_x, batch_y, [-2, -3])
            self.idx_train = np.roll(self.idx_train, -1, axis=0)
            yield batch_x, batch_y

    def get_val_batch(self):
        while True:
            batch_idx = self.idx_val[0]
            batch_x = np.stack([self.X[i] for i in batch_idx])
            batch_y = np.stack([self.y[i] for i in batch_idx])
            batch_x, batch_y = self._check_size(batch_x, batch_y, [-2, -3])
            self.idx_val = np.roll(self.idx_val, -1, axis=0)
            yield batch_x, batch_y




## === cell 6
np.random.seed(42)
data_generator = Image_generator(os.getcwd())
autoencoder = Autoencoder(loss="mse")




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 7
log = autoencoder.model.fit(
    data_generator.get_train_batch(),
    steps_per_epoch=data_generator.train_steps,
    epochs=12,  # increased epochs to improve RMSE toward target
    validation_data=data_generator.get_val_batch(),
    validation_steps=data_generator.val_steps,
    shuffle=True,
    verbose=1,
)




## === cell 8
plt.figure()
for k, v in log.history.items():
    plt.plot(v, label=str(k))
plt.legend()
plt.show()




## === cell 9
def to_csv(pred_list, ids):
    """Write predictions (list of arrays) to submission.csv."""
    with open("submission.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(
            csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        csvwriter.writerow(("id", "value"))
        for i, each in enumerate(pred_list):
            rows, cols, _ = each.shape
            for row in range(rows):
                for col in range(cols):
                    pixel_id = f"{ids[i]}_{row+1}_{col+1}"
                    value = np.mean(each[row, col, :])
                    csvwriter.writerow([pixel_id, f"{value:.6f}"])


test_paths = sorted(glob.glob(os.path.join(os.getcwd(), "../input/test/*.png")))
X_test = [cv2.imread(p) for p in test_paths]
ids = [os.path.basename(p)[:-4] for p in test_paths]

predictions = []
for x in X_test:
    orig_shape = x.shape
    while x.shape[0] % 4 != 0:
        x = np.insert(x, x.shape[0], 1, axis=0)
    while x.shape[1] % 4 != 0:
        x = np.insert(x, x.shape[1], 1, axis=1)
    x_input = x.astype("float32") / 255
    x_input = x_input.reshape((1,) + x_input.shape)

    pred = autoencoder.model.predict(x_input, verbose=0)
    pred = pred.reshape(pred.shape[1:])  # remove batch dim
    pred = pred[: orig_shape[0], : orig_shape[1], :]  # crop to original size

    predictions.append(pred)
    print(f"{len(predictions)} / {len(X_test)} predictions", end="\r")
print("\nSaving...")
to_csv(predictions, ids)
print("submission.csv saved")




## === cell 10
try:
    encoder = autoencoder.get_encoder()
    example = X_test[0].reshape((1,) + X_test[0].shape)
    encoded = encoder.predict(example)
    plt.figure()
    plt.imshow(example[0])
    plt.title("Original example")
    plt.show()
except Exception as e:
    print("Encoder inspection skipped:", e)
