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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.8

# 3. Installed packages

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

# 4. Data file paths

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

# 5. Target score

0.9991

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


## === cell 1
from glob import glob

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import layers


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
pa = glob('train/*.jpg')[0]
pa
g = tf.io.read_file(pa)
im = tf.io.decode_image(g)
print(im.shape)
plt.imshow(im)
plt.show()


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/3666641271.py in <cell line: 0>()
----> 1 pa = glob('train/*.jpg')[0]
      2 pa
      3 g = tf.io.read_file(pa)
      4 im = tf.io.decode_image(g)
      5 print(im.shape)

IndexError: list index out of range

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
hist = model.fit(train_ds,
                 validation_data=valid_ds,
                 validation_steps=validation_steps,
                 steps_per_epoch=steps_per_epoch,
                 epochs = 30)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/648714053.py in <cell line: 0>()
----> 1 hist = model.fit(train_ds,
      2                  validation_data=valid_ds,
      3                  validation_steps=validation_steps,
      4                  steps_per_epoch=steps_per_epoch,
      5                  epochs = 30)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    122             raise e.with_traceback(filtered_tb) from None
    123         finally:
--> 124             del filtered_tb
    125 
    126     return error_handler

ValueError: as_list() is not defined on an unknown TensorShape.

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

test_images = []
for path in tqdm_notebook(eval_paths) :
  test_images.append(image_read(path))



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NotFoundError                             Traceback (most recent call last)
/tmp/ipykernel_11/2360606999.py in <cell line: 0>()
      3 test_images = []
      4 for path in tqdm_notebook(eval_paths) :
----> 5   test_images.append(image_read(path))
      6 
      7 # np.array(test_image).shape

/tmp/ipykernel_11/924591735.py in image_read(path)
      1 def image_read(path) :
----> 2   g = tf.io.read_file(path)
      3   im = tf.io.decode_image(g)
      4 
      5   return im

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/io_ops.py in read_file(filename, name)
    132     A tensor of dtype "string", with the file contents.
    133   """
--> 134   return gen_io_ops.read_file(filename, name)
    135 
    136 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_io_ops.py in read_file(filename, name)
    581       pass
    582     try:
--> 583       return read_file_eager_fallback(
    584           filename, name=name, ctx=_ctx)
    585     except _core._SymbolicException:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_io_ops.py in read_file_eager_fallback(filename, name, ctx)
    604   _inputs_flat = [filename]
    605   _attrs = None
--> 606   _result = _execute.execute(b"ReadFile", 1, inputs=_inputs_flat,
    607                              attrs=_attrs, ctx=ctx, name=name)
    608   if _execute.must_record_gradient():

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     51   try:
     52     ctx.ensure_initialized()
---> 53     tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
     54                                         inputs, attrs, num_outputs)
     55   except core._NotOkStatusException as e:

NotFoundError: {{function_node __wrapped__ReadFile_device_/job:localhost/replica:0/task:0/device:CPU:0}} test/09034a34de0e2015a8a28dfe18f423f6.jpg; No such file or directory [Op:ReadFile]

## === cell 29
test_ds = tf.data.Dataset.from_tensor_slices(eval_paths)
test_ds = test_ds.map(image_read)
test_ds = test_ds.batch(batch_size)


## === cell 30
pred = model.predict(test_ds)


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NotFoundError                             Traceback (most recent call last)
/tmp/ipykernel_11/1577479882.py in <cell line: 0>()
----> 1 pred = model.predict(test_ds)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

NotFoundError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} test/09034a34de0e2015a8a28dfe18f423f6.jpg; No such file or directory
	 [[{{node ReadFile}}]] [Op:IteratorGetNext] name: 

## === cell 31
pred.shape


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1555121774.py in <cell line: 0>()
----> 1 pred.shape

NameError: name 'pred' is not defined

## === cell 32
pred = pred.reshape((4000))


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3685328033.py in <cell line: 0>()
----> 1 pred = pred.reshape((4000))

NameError: name 'pred' is not defined

## === cell 33
submit_df = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')
test_fnames = submit_df['id']
test_labels = pred  # 결과가 onehot이 아닌 binary로 담아줘야함

submit_file = pd.DataFrame({'id':test_fnames, 'has_cactus':test_labels}, columns = ['id', 'has_cactus'])
submit_file
submit_file.to_csv('submission.csv', index=False)


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/34150946.py in <cell line: 0>()
      1 submit_df = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')
      2 test_fnames = submit_df['id']
----> 3 test_labels = pred  # 결과가 onehot이 아닌 binary로 담아줘야함
      4 
      5 submit_file = pd.DataFrame({'id':test_fnames, 'has_cactus':test_labels}, columns = ['id', 'has_cactus'])

NameError: name 'pred' is not defined
