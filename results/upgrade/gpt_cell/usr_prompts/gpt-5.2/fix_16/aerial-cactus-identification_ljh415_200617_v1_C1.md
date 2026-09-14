# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.8

# 2. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


## === cell 1
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==5.28.3"]
)

from glob import glob

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import layers


## === cell 3
from zipfile import ZipFile
with ZipFile('../input/aerial-cactus-identification/test.zip')as test_obj :
  test_obj.extractall()
with ZipFile('../input/aerial-cactus-identification/train.zip')as train_obj :
  train_obj.extractall()


## === cell 4
df = pd.read_csv('../input/aerial-cactus-identification/train.csv')
print(df.head())

file_list = df['id']
has_cactus = df['has_cactus']
print(len(file_list), len(has_cactus))


## === cell 5
test_df = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')
print(test_df.head())

test_fnames = test_df['id']
test_labels = test_df['has_cactus']

print(len(test_fnames), len(test_labels))


## === cell 6
data_paths = glob('train/*.jpg')
test_paths = glob('test/*.jpg')
print(len(data_paths), len(test_paths))


## === cell 7
candidate_train_dirs = [
    "train",
    "../input/aerial-cactus-identification/train",
    "/kaggle/input/aerial-cactus-identification/train",
    "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification/train",
]

pa = None
for d in candidate_train_dirs:
    paths = glob(os.path.join(d, "*.jpg"))
    if paths:
        pa = paths[0]
        break

if pa is None:
    raise FileNotFoundError(
        "Could not find any training images. Checked: "
        + ", ".join([os.path.join(d, "*.jpg") for d in candidate_train_dirs])
    )

pa
g = tf.io.read_file(pa)
im = tf.io.decode_image(g)
print(im.shape)
plt.imshow(im)
plt.show()


## === cell 8
input_shape = (32, 32, 3)
batch_size = 32


## === cell 9
data_dir = 'train'

data_paths = []
for fname, label in zip(file_list, has_cactus) :
  data_paths.append((os.path.join(data_dir, fname), label))


## === cell 10
data_paths[:10]


## === cell 11
def tmp_func (path_name) :
  return path_name


## === cell 12
def read_data(path_name) :
  img_path = path_name[0]
  label = tf.strings.to_number(path_name[1], out_type=tf.int64)

  gfile = tf.io.read_file(img_path)
  image = tf.io.decode_image(gfile)
  
  return image, label


## === cell 13
a = tf.data.Dataset.from_tensor_slices(np.array(data_paths[:2]))
a = a.map(tmp_func)
p = next(iter(a))
p[0], p[1]


## === cell 14
train_ratio = 0.8

train_paths = data_paths[:int(train_ratio*len(data_paths))]
test_paths = data_paths[int(train_ratio*len(data_paths)):]


## === cell 15
train_ds = tf.data.Dataset.from_tensor_slices(np.array(train_paths))
train_ds = train_ds.map(read_data)
train_ds = train_ds.shuffle(len(train_paths))
train_ds = train_ds.batch(batch_size)
train_ds = train_ds.repeat()


## === cell 16
valid_ds = tf.data.Dataset.from_tensor_slices(np.array(test_paths))
valid_ds = valid_ds.map(read_data)
valid_ds = valid_ds.batch(batch_size)
valid_ds = valid_ds.repeat()


## === cell 17
inputs = layers.Input(input_shape)

net = layers.Conv2D(32, 3, 1, 'SAME')(inputs)
net = layers.Activation('relu')(net)
net = layers.Conv2D(32, 3, 1, 'SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D((2, 2))(net)
net = layers.Dropout(0.5)(net)

net = layers.Conv2D(64, 3, 1, 'SAME')(net)
net = layers.Activation('relu')(net)
net = layers.Conv2D(64, 3, 1, 'SAME')(net)
net = layers.Activation('relu')(net)
net = layers.MaxPooling2D((2, 2))(net)
net = layers.Dropout(0.5)(net)

net = layers.Flatten()(net)
net = layers.Dense(512)(net)
net = layers.Activation('relu')(net)
net = layers.Dropout(0.5)(net)
net = layers.Dense(1)(net)
net = layers.Activation('sigmoid')(net)

model = tf.keras.Model(inputs=inputs, outputs=net, name='cactus_cnn')


## === cell 18
model.summary()


## === cell 19
model.compile(loss = tf.keras.losses.binary_crossentropy,
              optimizer = tf.keras.optimizers.Adam(),
              metrics=['accuracy'])


## === cell 20
steps_per_epoch = len(train_paths) // batch_size
validation_steps = len(test_paths) // batch_size


## === cell 21
def _fix_image_shape(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.cond(
        tf.equal(tf.rank(image), 3),
        lambda: tf.expand_dims(image, axis=0),
        lambda: image,
    )
    image = tf.ensure_shape(image, [None, None, None, None])

    channels = tf.shape(image)[-1]

    image = tf.cond(
        tf.equal(channels, 1),
        lambda: tf.image.grayscale_to_rgb(image),
        lambda: tf.cond(
            tf.equal(channels, 4),
            lambda: image[..., :3],  # drop alpha if present
            lambda: image,  # already 3 channels
        ),
    )

    image = tf.image.resize(image, input_shape[:2])
    image.set_shape([None, *input_shape])  # (B, 32, 32, 3)

    label = tf.cast(label, tf.float32)
    return image, label


candidate_train_dirs = [
    "train",  # if extracted into CWD
    "../input/aerial-cactus-identification/train",
    "/kaggle/input/aerial-cactus-identification/train",
    "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification/train",
]
candidate_test_dirs = [
    "test",  # if extracted into CWD
    "../input/aerial-cactus-identification/test",
    "/kaggle/input/aerial-cactus-identification/test",
    "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification/test",
]

train_dir = None
for d in candidate_train_dirs:
    if glob(os.path.join(d, "*.jpg")):
        train_dir = d
        break
if train_dir is None:
    raise FileNotFoundError(
        "Could not find any training images. Checked: "
        + ", ".join([os.path.join(d, "*.jpg") for d in candidate_train_dirs])
    )

test_dir = None
for d in candidate_test_dirs:
    if glob(os.path.join(d, "*.jpg")):
        test_dir = d
        break
if test_dir is None:
    raise FileNotFoundError(
        "Could not find any test images. Checked: "
        + ", ".join([os.path.join(d, "*.jpg") for d in candidate_test_dirs])
    )

data_paths_fixed = [
    (os.path.join(train_dir, fname), label)
    for fname, label in zip(file_list, has_cactus)
]
train_paths_fixed = data_paths_fixed[: int(train_ratio * len(data_paths_fixed))]
test_paths_fixed = data_paths_fixed[int(train_ratio * len(data_paths_fixed)) :]

train_ds = tf.data.Dataset.from_tensor_slices(np.array(train_paths_fixed))
train_ds = train_ds.map(read_data)
train_ds = train_ds.shuffle(len(train_paths_fixed))
train_ds = train_ds.batch(batch_size)
train_ds = train_ds.repeat()

valid_ds = tf.data.Dataset.from_tensor_slices(np.array(test_paths_fixed))
valid_ds = valid_ds.map(read_data)
valid_ds = valid_ds.batch(batch_size)
valid_ds = valid_ds.repeat()

train_ds = train_ds.map(_fix_image_shape, num_parallel_calls=tf.data.AUTOTUNE)
valid_ds = valid_ds.map(_fix_image_shape, num_parallel_calls=tf.data.AUTOTUNE)

hist = model.fit(
    train_ds,
    validation_data=valid_ds,
    validation_steps=validation_steps,
    steps_per_epoch=steps_per_epoch,
    epochs=30,
)


## === cell 22
test_df.head()


## === cell 23
test_fnames[:5]


## === cell 24
test_labels[:5]


## === cell 25
test_dir = 'test'

eval_paths = []
for fname in test_fnames :
  eval_paths.append(os.path.join(test_dir, fname))

print(eval_paths[:5])


## === cell 26
def image_read(path) :
  g = tf.io.read_file(path)
  im = tf.io.decode_image(g)

  return im


## === cell 27
from tqdm import tqdm_notebook


## === cell 28
candidate_test_dirs = []
if "test_dir" in globals() and isinstance(test_dir, str) and test_dir:
    candidate_test_dirs.append(test_dir)

candidate_test_dirs.extend(
    [
        "test",  # if extracted into CWD
        "../input/aerial-cactus-identification/test",
        "/kaggle/input/aerial-cactus-identification/test",
        "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification/test",
    ]
)

resolved_test_dir = None
for d in candidate_test_dirs:
    if glob(os.path.join(d, "*.jpg")):
        resolved_test_dir = d
        break

if resolved_test_dir is None:
    raise FileNotFoundError(
        "Could not find any test images. Checked: "
        + ", ".join([os.path.join(d, "*.jpg") for d in candidate_test_dirs])
    )

eval_paths = [os.path.join(resolved_test_dir, fname) for fname in test_fnames]

test_images = []
for path in tqdm_notebook(eval_paths):
    test_images.append(image_read(path))


## === cell 29
test_ds = tf.data.Dataset.from_tensor_slices(eval_paths)
test_ds = test_ds.map(image_read)
test_ds = test_ds.batch(batch_size)


## === cell 30
pred = model.predict(test_ds)


## === cell 31
pred.shape


## === cell 32
pred = pred.reshape((4000))


## --- ERROR in cell 32, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3685328033.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mpred[0m [0;34m=[0m [0mpred[0m[0;34m.[0m[0mreshape[0m[0;34m([0m[0;34m([0m[0;36m4000[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;31mValueError[0m: cannot reshape array of size 3325 into shape (4000,)

## === cell 33
submit_df = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')
test_fnames = submit_df['id']
test_labels = pred  # 결과가 onehot이 아닌 binary로 담아줘야함

submit_file = pd.DataFrame({'id':test_fnames, 'has_cactus':test_labels}, columns = ['id', 'has_cactus'])
submit_file
submit_file.to_csv('submission.csv', index=False)
