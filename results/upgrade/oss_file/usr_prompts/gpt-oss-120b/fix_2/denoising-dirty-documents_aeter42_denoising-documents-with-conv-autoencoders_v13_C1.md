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

0.28311

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import glob, os, csv
import numpy as np
import cv2
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    UpSampling2D,
    Input,
)
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_H, IMG_W = 128, 128
CHANNELS = 3


def load_and_resize(path_pattern):
    imgs = []
    for fp in sorted(glob.glob(path_pattern)):
        img = cv2.imread(fp, cv2.IMREAD_COLOR)  # BGR uint8
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # to RGB
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        imgs.append(img.astype("float32") / 255.0)  # normalise to [0,1]
    return np.stack(imgs, axis=0)






## === cell 2
class Autoencoder:
    def __init__(
        self,
        dimensions_factor=2,
        layers=2,
        k=3,
        filter_size=None,
        pooling_factor=None,
        loss="mean_squared_error",
        channels=CHANNELS,
    ):
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

        input_img = Input(shape=(IMG_H, IMG_W, self.channels), name="input")
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
                name=f"encoding_conv_{i}",
            )(x)
            x = MaxPooling2D(
                (self.pooling_factor[i], self.pooling_factor[i]),
                padding="valid",
                name=f"encoding_pool_{i}",
            )(x)
        self.code_filters = filters

        for i in range(self.layers, 2 * self.layers):
            filters = int(
                int(x.shape[-1]) / self.pooling_factor[i] * self.dimensions_factor
            )
            x = Conv2DTranspose(
                filters,
                self.filter_size[i],
                activation="relu",
                padding="same",
                name=f"decoding_conv_{i}",
            )(x)
            x = UpSampling2D(
                (self.pooling_factor[i], self.pooling_factor[i]),
                name=f"decoding_pool_{i}",
            )(x)

        x = Conv2D(
            self.channels,
            self.filter_size[-1],
            activation="sigmoid",
            padding="same",
            name="decoded",
        )(x)

        self.model = Model(inputs=input_img, outputs=x, name="autoencoder")
        self.model.compile(optimizer="adam", loss=loss, metrics=["mse"])

    def get_encoder(self):
        encoder_input = self.model.input
        code_layer = self.model.get_layer(name=f"encoding_pool_{self.layers-1}")
        encoder_output = code_layer.output
        return Model(encoder_input, encoder_output, name="encoder")

    def get_decoder(self):
        code_shape = self.model.get_layer(name="decoded").input_shape[1:]
        code_input = Input(shape=code_shape, name="code_input")
        x = code_input
        for i in range(self.layers, 2 * self.layers):
            filters = int(
                int(x.shape[-1]) / self.pooling_factor[i] * self.dimensions_factor
            )
            x = Conv2DTranspose(
                filters,
                self.filter_size[i],
                activation="relu",
                padding="same",
                name=f"decoding_conv_{i}_clone",
            )(x)
            x = UpSampling2D(
                (self.pooling_factor[i], self.pooling_factor[i]),
                name=f"decoding_pool_{i}_clone",
            )(x)
        x = Conv2D(
            self.channels,
            self.filter_size[-1],
            activation="sigmoid",
            padding="same",
            name="decoded_clone",
        )(x)
        return Model(code_input, x, name="decoder")




## === cell 3
class ImageGenerator:
    def __init__(self, batch_size=8, val_split=0.1):
        self.batch_size = batch_size
        self.X = load_and_resize(os.path.join(os.getcwd(), "../input/train/*.png"))
        self.y = load_and_resize(
            os.path.join(os.getcwd(), "../input/train_cleaned/*.png")
        )
        assert self.X.shape == self.y.shape, "Input and target shapes must match"

        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
        split = int(len(indices) * (1 - val_split))
        self.train_idx = indices[:split]
        self.val_idx = indices[split:]

    def _batch(self, idx):
        sel = np.random.choice(idx, self.batch_size, replace=False)
        return self.X[sel], self.y[sel]

    def train_generator(self):
        while True:
            yield self._batch(self.train_idx)

    def val_generator(self):
        while True:
            yield self._batch(self.val_idx)

    @property
    def steps_per_epoch(self):
        return max(1, len(self.train_idx) // self.batch_size)

    @property
    def validation_steps(self):
        return max(1, len(self.val_idx) // self.batch_size)




## === cell 4
np.random.seed(42)
tf_random = __import__("tensorflow").random
tf_random.set_seed(42)

data_gen = ImageGenerator(batch_size=8, val_split=0.1)
autoencoder = Autoencoder(loss="mse")



## === cell 5
autoencoder.model.fit(
    x=data_gen.train_generator(),
    steps_per_epoch=data_gen.steps_per_epoch,
    epochs=3,
    validation_data=data_gen.val_generator(),
    validation_steps=data_gen.validation_steps,
    verbose=2,
)




## === cell 6
def write_submission(preds, ids, filename="submission.csv"):
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "value"])
        for i, img in enumerate(preds):
            h, w, c = img.shape
            for row in range(h):
                for col in range(w):
                    pixel_id = f"{ids[i]}_{row+1}_{col+1}"
                    value = float(np.mean(img[row, col, :]))
                    writer.writerow([pixel_id, f"{value:.6f}"])


test_paths = sorted(glob.glob(os.path.join(os.getcwd(), "../input/test/*.png")))
test_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_paths]
X_test = load_and_resize(os.path.join(os.getcwd(), "../input/test/*.png"))

predictions = autoencoder.model.predict(X_test, batch_size=8, verbose=1)

write_submission(predictions, test_ids, filename="submission.csv")
print("Submission saved to submission.csv")

## --- ERROR in outputing the csv:
Invalid submission: Expected the submission to have 5789880 rows, but got 475136.
