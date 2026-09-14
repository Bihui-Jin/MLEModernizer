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

0.35886

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.45611) has done: 'I fix the environment/runtime issues by removing notebook-only magics, avoiding the protobuf/Keras import crash, and using `tf_keras` consistently. Then I fix the inhomogeneous-image-shape errors by loading images as lists and batching by shape (keeping your “group-by-shape then train” core idea intact). Finally, I correct the Kaggle input paths, ensure inference uses the same preprocessing (float scaling + padding), and write a valid `submission.csv` with the exact required `id,value` format.'
- What this solution (achieved 0.35886) has done: 'I fix the two runtime errors that prevent training/inference: the protobuf/Keras import crash by switching from `tf_keras` to `tensorflow.keras` (stable in Kaggle), and the `workers=0` argument error by removing `workers`/`use_multiprocessing` for generator training. To move the RMSE score toward the target (lower is better), I make one minimal, score-improving change that preserves your core model/training loop: align padding during training batches with inference by padding each batch to a multiple of 4 using the same `ensure_divisible` logic (instead of `np.insert` on the batch tensor), which reduces edge artifacts and usually improves denoising. I also make submission writing deterministic and ordered by test image id, while keeping the exact required `id,value` format and writing `submission.csv` to the working directory. All other architecture/training semantics (autoencoder structure, loss, epochs, grouped-by-shape batching) remain intact.'

# 9. Code solution

## === cell 0
import os, glob, csv
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Conv2D,
    UpSampling2D,
    MaxPooling2D,
    Input,
    Conv2DTranspose,
)
from tensorflow.keras import Model

import cv2

np.random.seed(42)
tf.random.set_seed(42)

DATA_ROOT = "/kaggle/input/denoising-dirty-documents"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TRAIN_CLEAN_DIR = os.path.join(DATA_ROOT, "train_cleaned")
TEST_DIR = os.path.join(DATA_ROOT, "test")


def imread_rgb(path):
    """Read image as RGB float32 in [0,1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype("float32") / 255.0


def ensure_divisible(img, reduction_factor=4, pad_value=1.0):
    """Pad H and W (bottom/right) with constant so H%rf==0 and W%rf==0."""
    h, w = img.shape[:2]
    pad_h = (reduction_factor - (h % reduction_factor)) % reduction_factor
    pad_w = (reduction_factor - (w % reduction_factor)) % reduction_factor
    if pad_h == 0 and pad_w == 0:
        return img
    return np.pad(
        img,
        pad_width=((0, pad_h), (0, pad_w), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_paths = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.png")))
clean_paths = sorted(glob.glob(os.path.join(TRAIN_CLEAN_DIR, "*.png")))

train_map = {os.path.basename(p): p for p in train_paths}
clean_map = {os.path.basename(p): p for p in clean_paths}
common_names = sorted(set(train_map).intersection(clean_map))

X_vis = [imread_rgb(train_map[n]) for n in common_names[:4]]
y_vis = [imread_rgb(clean_map[n]) for n in common_names[:4]]

plt.figure(figsize=(16, 8))
for i in range(min(4, len(X_vis))):
    plt.subplot(2, 4, 1 + i)
    plt.imshow(X_vis[i])
    plt.axis("off")
    plt.subplot(2, 4, 5 + i)
    plt.imshow(y_vis[i])
    plt.axis("off")
plt.tight_layout()
plt.show()



## === cell 2
shapes = np.unique([imread_rgb(p).shape for p in train_paths], axis=0)
print("Unique train image shapes (H,W,C):")
print(shapes)




## === cell 3
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
                ([(k, k)] * (layers * 2 + 1)) if (filter_size is None) else filter_size
            )
            self.pooling_factor = (
                ([2] * (layers * 2)) if (pooling_factor is None) else pooling_factor
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

    def get_encoder(self):
        if self.model is None:
            raise RuntimeError("Model is not created")
        encoder = self.__init__(only_encoder=True)
        for layer in encoder.layers[1:]:
            weights = self.model.get_layer(name=layer.name).get_weights()
            layer.set_weights(weights)
        return encoder

    def get_decoder(self):
        if self.model is None:
            raise RuntimeError("Model is not created")
        decoder = self.__init__(only_decoder=True)
        for layer in decoder.layers[1:]:
            weights = self.model.get_layer(name=layer.name).get_weights()
            layer.set_weights(weights)
        return decoder




## === cell 4
class Image_generator:
    """
    Keep core approach (group by shape, fixed batch size).
    FIX 2 (score + consistency): pad per-image using the same ensure_divisible()
    as inference (constant white padding), instead of np.insert over the batch tensor.
    This reduces edge artifacts and improves RMSE without changing architecture/training loop.
    """

    def __init__(self, path, val_percentage=0.115, batch_size=4, reduction_factor=4):
        self.base_path = path
        self.batch_size = batch_size
        self.reduction_factor = reduction_factor

        train_paths = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.png")))
        clean_paths = sorted(glob.glob(os.path.join(TRAIN_CLEAN_DIR, "*.png")))

        train_map = {os.path.basename(p): p for p in train_paths}
        clean_map = {os.path.basename(p): p for p in clean_paths}
        self.names = sorted(set(train_map).intersection(clean_map))

        self.X = [imread_rgb(train_map[n]) for n in self.names]
        self.y = [imread_rgb(clean_map[n]) for n in self.names]

        self.idx_train, self.idx_val = self.split_batches(val_percentage)
        self.train_steps = len(self.idx_train)
        self.val_steps = len(self.idx_val)

    def split_batches(self, val_percentage):
        idx_batches = []
        shapes = np.unique([img.shape for img in self.X], axis=0)

        for shape in shapes:
            shape_idx = np.array(
                [i for i, img in enumerate(self.X) if img.shape == tuple(shape)],
                dtype=int,
            )
            np.random.shuffle(shape_idx)

            n_full = (len(shape_idx) // self.batch_size) * self.batch_size
            shape_idx = shape_idx[:n_full]
            if len(shape_idx) == 0:
                continue
            batches = np.split(shape_idx, len(shape_idx) // self.batch_size)
            idx_batches.extend(batches)

        idx_batches = np.array(idx_batches, dtype=object)
        np.random.shuffle(idx_batches)

        samples = len(idx_batches)
        val_samples = max(1, int(samples * val_percentage)) if samples > 1 else 0
        if val_samples == 0:
            return idx_batches, idx_batches[:0]
        return idx_batches[:-val_samples], idx_batches[-val_samples:]

    def _pad_batch_divisible(self, batch_x, batch_y):
        h, w = batch_x.shape[1], batch_x.shape[2]
        pad_h = (
            self.reduction_factor - (h % self.reduction_factor)
        ) % self.reduction_factor
        pad_w = (
            self.reduction_factor - (w % self.reduction_factor)
        ) % self.reduction_factor
        if pad_h == 0 and pad_w == 0:
            return batch_x, batch_y
        batch_x = np.pad(
            batch_x,
            pad_width=((0, 0), (0, pad_h), (0, pad_w), (0, 0)),
            mode="constant",
            constant_values=1.0,
        )
        batch_y = np.pad(
            batch_y,
            pad_width=((0, 0), (0, pad_h), (0, pad_w), (0, 0)),
            mode="constant",
            constant_values=1.0,
        )
        return batch_x, batch_y

    def get_train_batch(self):
        while True:
            batch_idx = list(self.idx_train[0])
            batch_x = np.stack([self.X[i] for i in batch_idx], axis=0)
            batch_y = np.stack([self.y[i] for i in batch_idx], axis=0)
            batch_x, batch_y = self._pad_batch_divisible(batch_x, batch_y)
            self.idx_train = np.roll(self.idx_train, 1, axis=0)
            yield batch_x, batch_y

    def get_val_batch(self):
        while True:
            if len(self.idx_val) == 0:
                yield from self.get_train_batch()
            batch_idx = list(self.idx_val[0])
            batch_x = np.stack([self.X[i] for i in batch_idx], axis=0)
            batch_y = np.stack([self.y[i] for i in batch_idx], axis=0)
            batch_x, batch_y = self._pad_batch_divisible(batch_x, batch_y)
            self.idx_val = np.roll(self.idx_val, 1, axis=0)
            yield batch_x, batch_y




## === cell 5
data_generator = Image_generator(os.getcwd())
autoencoder = Autoencoder(loss="mse")

print(
    f"Train steps: {data_generator.train_steps}, Val steps: {data_generator.val_steps}"
)



## === cell 6
log = autoencoder.model.fit(
    x=data_generator.get_train_batch(),
    steps_per_epoch=data_generator.train_steps,
    epochs=10,
    shuffle=False,
    validation_data=data_generator.get_val_batch(),
    validation_steps=max(1, data_generator.val_steps),
    verbose=2,
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/3446596192.py in <cell line: 0>()
      1 # FIX 3 (runtime): tf.keras Model.fit does not accept workers=0 in this environment.
      2 # Keep generator training; just remove workers/use_multiprocessing.
----> 3 log = autoencoder.model.fit(
      4     x=data_generator.get_train_batch(),
      5     steps_per_epoch=data_generator.train_steps,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

InvalidArgumentError: Graph execution error:

Detected at node PyFunc defined at (most recent call last):
<stack traces unavailable>
Detected at node PyFunc defined at (most recent call last):
<stack traces unavailable>
2 root error(s) found.
  (0) INVALID_ARGUMENT:  TypeError: Generator yielded an element of shape (4, 260, 540, 3) where an element of shape (None, 420, 540, 3) was expected. Your generator provides tensors with variable input dimensions other than the batch size. Make sure that the generator's first two batches do not have the same dimension value wherever there is a variable input dimension.
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 269, in __call__
    ret = func(*args)
          ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_generator_op.py", line 198, in generator_py_func
    values = next(generator_state.get_iterator(iterator_id))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/generator_data_adapter.py", line 53, in get_tf_iterator
    batch = tree.map_structure(
            ^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/keras/src/tree/tree_api.py", line 192, in map_structure
    return tree_impl.map_structure(func, *structures)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/keras/src/tree/optree_impl.py", line 108, in map_structure
    return optree.tree_map(
           ^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/optree/ops.py", line 766, in tree_map
    return treespec.unflatten(map(func, *flat_args))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/keras/src/tree/optree_impl.py", line 104, in func_with_check
    return func(*args)
           ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/generator_data_adapter.py", line 40, in convert_to_tf
    raise TypeError(

TypeError: Generator yielded an element of shape (4, 260, 540, 3) where an element of shape (None, 420, 540, 3) was expected. Your generator provides tensors with variable input dimensions other than the batch size. Make sure that the generator's first two batches do not have the same dimension value wherever there is a variable input dimension.


	 [[{{node PyFunc}}]]
	 [[IteratorGetNext]]
	 [[IteratorGetNext/_4]]
  (1) INVALID_ARGUMENT:  TypeError: Generator yielded an element of shape (4, 260, 540, 3) where an element of shape (None, 420, 540, 3) was expected. Your generator provides tensors with variable input dimensions other than the batch size. Make sure that the generator's first two batches do not have the same dimension value wherever there is a variable input dimension.
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 269, in __call__
    ret = func(*args)
          ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_generator_op.py", line 198, in generator_py_func
    values = next(generator_state.get_iterator(iterator_id))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/generator_data_adapter.py", line 53, in get_tf_iterator
    batch = tree.map_structure(
            ^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/keras/src/tree/tree_api.py", line 192, in map_structure
    return tree_impl.map_structure(func, *structures)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/keras/src/tree/optree_impl.py", line 108, in map_structure
    return optree.tree_map(
           ^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/optree/ops.py", line 766, in tree_map
    return treespec.unflatten(map(func, *flat_args))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/keras/src/tree/optree_impl.py", line 104, in func_with_check
    return func(*args)
           ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/generator_data_adapter.py", line 40, in convert_to_tf
    raise TypeError(

TypeError: Generator yielded an element of shape (4, 260, 540, 3) where an element of shape (None, 420, 540, 3) was expected. Your generator provides tensors with variable input dimensions other than the batch size. Make sure that the generator's first two batches do not have the same dimension value wherever there is a variable input dimension.


	 [[{{node PyFunc}}]]
	 [[IteratorGetNext]]
0 successful operations.
0 derived errors ignored. [Op:__inference_multi_step_on_iterator_2417]

## === cell 7
plt.figure()
for k, v in log.history.items():
    plt.plot(v, label=str(k))
plt.legend()
plt.show()




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3891366262.py in <cell line: 0>()
      1 plt.figure()
----> 2 for k, v in log.history.items():
      3     plt.plot(v, label=str(k))
      4 plt.legend()
      5 plt.show()

NameError: name 'log' is not defined

## === cell 8
def to_csv(pred_list, ids, out_path="submission.csv"):
    with open(out_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(
            csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        csvwriter.writerow(("id", "value"))
        for i, each in enumerate(pred_list):
            rows, cols, _ = each.shape
            for row in range(rows):
                for col in range(cols):
                    id_pixel = str(ids[i]) + "_" + str(row + 1) + "_" + str(col + 1)
                    value_pixel = float(np.mean(each[row, col, :]))
                    csvwriter.writerow([id_pixel, value_pixel])


test_paths = sorted(glob.glob(os.path.join(TEST_DIR, "*.png")))
ids = [os.path.splitext(os.path.basename(p))[0] for p in test_paths]

predictions = []
for j, p in enumerate(test_paths):
    x = imread_rgb(p)
    original_shape = x.shape
    x_pad = ensure_divisible(x, reduction_factor=4, pad_value=1.0)
    x_in = x_pad.reshape((1,) + x_pad.shape)

    pred = autoencoder.model.predict(x_in, verbose=0)[0]
    pred = pred[: original_shape[0], : original_shape[1], :]
    pred = np.clip(pred, 0.0, 1.0)

    predictions.append(pred)
    if (j + 1) % 5 == 0 or (j + 1) == len(test_paths):
        print(f"{j+1} of {len(test_paths)} predictions calculated")

print("Saving...")
to_csv(predictions, ids, out_path="submission.csv")
print("Saved submission.csv")



## === cell 9
try:
    encoder = autoencoder.get_encoder()
    example = imread_rgb(test_paths[min(10, len(test_paths) - 1)])
    example_pad = ensure_divisible(example, reduction_factor=4, pad_value=1.0)
    filters = encoder.predict(example_pad.reshape((1,) + example_pad.shape), verbose=0)

    plt.figure()
    plt.imshow(example)
    plt.axis("off")
    plt.show()
except Exception as e:
    filters = None
    print("Skipping encoder visualization due to:", repr(e))



## === cell 10
try:
    import matplotlib.animation as animation
    from IPython.display import HTML

    fig = plt.figure()
    ims = []
    if filters is not None:
        for i in range(min(filters.shape[-1], 32)):
            im = plt.imshow(filters[0, :, :, i], animated=True, cmap="gray")
            ims.append([im])
        ani = animation.ArtistAnimation(
            fig, ims, interval=200, blit=True, repeat_delay=0
        )
    else:
        ani = None
except Exception as e:
    ani = None
    print("Skipping animation due to:", repr(e))



## === cell 11
try:
    from IPython.display import HTML

    if ani is not None:
        HTML(ani.to_jshtml())
except Exception as e:
    print("Skipping HTML display due to:", repr(e))
