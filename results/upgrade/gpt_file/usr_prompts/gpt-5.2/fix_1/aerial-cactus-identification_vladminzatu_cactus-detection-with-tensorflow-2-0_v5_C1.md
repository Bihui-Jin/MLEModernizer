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

3.7

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.973

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!pip install tensorflow==2.0.0-alpha0

from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import tensorflow as tf

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

from os import listdir


## === cell 2
train_dir = '../input/train/train/'
train_imgs = listdir(train_dir)
nr_train_images = len(train_imgs)
nr_train_images


## === cell 3
train_lables_df = pd.read_csv('../input/train.csv', index_col='id')
print('Total entries: ' + str(train_lables_df.size))
print(train_lables_df.head(10))


## === cell 4
train_lables_df['has_cactus'].value_counts()


## === cell 5
def get_test_image_path(id):
    return train_dir + id

def draw_cactus_image(id, ax):
    path = get_test_image_path(id)
    img = mpimg.imread(path)
    plt.imshow(img)
    ax.set_title('Label: ' + str(train_lables_df.loc[id]['has_cactus']))

fig = plt.figure(figsize=(20,20))
for i in range(12):
    ax = fig.add_subplot(3, 4, i + 1)
    draw_cactus_image(train_imgs[i], ax)


## === cell 6
train_image_paths = [train_dir + ti for ti in train_imgs]
train_image_labels = [train_lables_df.loc[ti]['has_cactus'] for ti in train_imgs]

for i in range(10):
    print(train_image_paths[i], train_image_labels[i])


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'train'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/437488980.py in <cell line: 0>()
      1 train_image_paths = [train_dir + ti for ti in train_imgs]
----> 2 train_image_labels = [train_lables_df.loc[ti]['has_cactus'] for ti in train_imgs]
      3 
      4 for i in range(10):
      5     print(train_image_paths[i], train_image_labels[i])

/tmp/ipykernel_11/437488980.py in <listcomp>(.0)
      1 train_image_paths = [train_dir + ti for ti in train_imgs]
----> 2 train_image_labels = [train_lables_df.loc[ti]['has_cactus'] for ti in train_imgs]
      3 
      4 for i in range(10):
      5     print(train_image_paths[i], train_image_labels[i])

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1189             maybe_callable = com.apply_if_callable(key, self.obj)
   1190             maybe_callable = self._check_deprecated_callable_usage(key, maybe_callable)
-> 1191             return self._getitem_axis(maybe_callable, axis=axis)
   1192 
   1193     def _is_scalar_access(self, key: tuple):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1429         # fall thru to straight lookup
   1430         self._validate_key(key, axis)
-> 1431         return self._get_label(key, axis=axis)
   1432 
   1433     def _get_slice_axis(self, slice_obj: slice, axis: AxisInt):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_label(self, label, axis)
   1379     def _get_label(self, label, axis: AxisInt):
   1380         # GH#5567 this will fail if the label is not present in the axis.
-> 1381         return self.obj.xs(label, axis=axis)
   1382 
   1383     def _handle_lowerdim_multi_index_axis0(self, tup: tuple):

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in xs(self, key, axis, level, drop_level)
   4299                     new_index = index[loc]
   4300         else:
-> 4301             loc = index.get_loc(key)
   4302 
   4303             if isinstance(loc, np.ndarray):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'train'

## === cell 7
def img_to_tensor(img_path):
    img_tensor = tf.cast(tf.image.decode_image(tf.io.read_file(img_path)), tf.float32)
    img_tensor /= 255.0 # normalized to [0,1]
    return img_tensor

img_to_tensor(train_image_paths[0])


## === cell 8
X_train, X_valid, y_train, y_valid = train_test_split(train_image_paths, train_image_labels, test_size=0.2)

def process_image_in_record(path, label):
    return img_to_tensor(path), label

def build_training_dataset(paths, labels, batch_size = 32):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(process_image_in_record)
    ds = ds.shuffle(buffer_size = len(paths))
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def build_validation_dataset(paths, labels, batch_size = 32):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(process_image_in_record)
    ds = ds.batch(batch_size)
    return ds

train_ds = build_training_dataset(X_train, y_train)
validation_ds = build_validation_dataset(X_valid, y_valid)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2980455553.py in <cell line: 0>()
----> 1 X_train, X_valid, y_train, y_valid = train_test_split(train_image_paths, train_image_labels, test_size=0.2)
      2 
      3 def process_image_in_record(path, label):
      4     return img_to_tensor(path), label
      5 

NameError: name 'train_image_labels' is not defined

## === cell 9
mini_train_ds = build_training_dataset(X_train[:5], y_train[:5], batch_size=2)
for images, labels in mini_train_ds.take(1):
    print(images)
    print(labels)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/785498627.py in <cell line: 0>()
----> 1 mini_train_ds = build_training_dataset(X_train[:5], y_train[:5], batch_size=2)
      2 # Fetch and print the first batch of 2 images
      3 for images, labels in mini_train_ds.take(1):
      4     print(images)
      5     print(labels)

NameError: name 'build_training_dataset' is not defined

## === cell 10
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


## === cell 11
history = model.fit(train_ds, epochs=20, steps_per_epoch=400, validation_data=validation_ds)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/419746550.py in <cell line: 0>()
----> 1 history = model.fit(train_ds, epochs=20, steps_per_epoch=400, validation_data=validation_ds)

NameError: name 'train_ds' is not defined

## === cell 12
def plot_accuracies_and_losses(history):
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'validation'], loc='upper left')
    plt.show()
    
    plt.title('Cross-entropy loss')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'], loc='upper left')
    plt.show()

plot_accuracies_and_losses(history)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3180022527.py in <cell line: 0>()
     12     plt.show()
     13 
---> 14 plot_accuracies_and_losses(history)

NameError: name 'history' is not defined

## === cell 13
cnn_model = tf.keras.Sequential()

cnn_model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
cnn_model.add(tf.keras.layers.MaxPooling2D((2,2)))
cnn_model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
cnn_model.add(tf.keras.layers.MaxPooling2D((2,2)))
cnn_model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))

cnn_model.add(tf.keras.layers.Flatten())
cnn_model.add(tf.keras.layers.Dense(64, activation='relu'))
cnn_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


## === cell 14
cnn_model.summary()


## === cell 15
history = cnn_model.fit(train_ds, epochs=20, steps_per_epoch=400, validation_data=validation_ds)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3158337243.py in <cell line: 0>()
----> 1 history = cnn_model.fit(train_ds, epochs=20, steps_per_epoch=400, validation_data=validation_ds)

NameError: name 'train_ds' is not defined

## === cell 16
plot_accuracies_and_losses(history)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3498469180.py in <cell line: 0>()
----> 1 plot_accuracies_and_losses(history)

NameError: name 'history' is not defined

## === cell 17
layer_outputs = [layer.output for layer in cnn_model.layers]
activation_model = tf.keras.Model(inputs=cnn_model.input, outputs=layer_outputs)

def print_example_and_activations(example, col_size, row_size, act_index): 
    example_array = img_to_tensor(example).numpy()
    plt.imshow(example_array)
    activations = activation_model.predict(example_array.reshape(1,32,32,3)) # batch of 1 - just the example array
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray') # image for the first (and only) element in the batch at activation_index
            activation_index += 1


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/180646879.py in <cell line: 0>()
      1 layer_outputs = [layer.output for layer in cnn_model.layers]
----> 2 activation_model = tf.keras.Model(inputs=cnn_model.input, outputs=layer_outputs)
      3 
      4 def print_example_and_activations(example, col_size, row_size, act_index):
      5     example_array = img_to_tensor(example).numpy()

/usr/local/lib/python3.11/dist-packages/keras/src/ops/operation.py in input(self)
    252             Input tensor or list of input tensors.
    253         """
--> 254         return self._get_node_attribute_at_index(0, "input_tensors", "input")
    255 
    256     @property

/usr/local/lib/python3.11/dist-packages/keras/src/ops/operation.py in _get_node_attribute_at_index(self, node_index, attr, attr_name)
    283         """
    284         if not self._inbound_nodes:
--> 285             raise AttributeError(
    286                 f"The layer {self.name} has never been called "
    287                 f"and thus has no defined {attr_name}."

AttributeError: The layer sequential_1 has never been called and thus has no defined input.

## === cell 18
print_example_and_activations(X_train[0], 8, 4, 1)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1555307194.py in <cell line: 0>()
      1 # Visualizing layer 1
----> 2 print_example_and_activations(X_train[0], 8, 4, 1)

NameError: name 'print_example_and_activations' is not defined

## === cell 19
print_example_and_activations(X_train[0], 8, 4, 3)


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1274971156.py in <cell line: 0>()
      1 # Visualizing layer 3
----> 2 print_example_and_activations(X_train[0], 8, 4, 3)

NameError: name 'print_example_and_activations' is not defined

## === cell 20
test_dir = '../input/test/test/'
test_imgs = listdir(test_dir)
print(len(test_imgs))
test_imgs[:5]


## === cell 21
def path_to_numpy_array(path):
    tensor = img_to_tensor(path)
    array = tensor.numpy()
    return array

test_image_paths = [test_dir + ti for ti in test_imgs]
test_instances = np.asarray([path_to_numpy_array(tip) for tip in test_image_paths])

test_instances[:2]


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
FailedPreconditionError                   Traceback (most recent call last)
/tmp/ipykernel_11/2023033881.py in <cell line: 0>()
      5 
      6 test_image_paths = [test_dir + ti for ti in test_imgs]
----> 7 test_instances = np.asarray([path_to_numpy_array(tip) for tip in test_image_paths])
      8 
      9 test_instances[:2]

/tmp/ipykernel_11/2023033881.py in <listcomp>(.0)
      5 
      6 test_image_paths = [test_dir + ti for ti in test_imgs]
----> 7 test_instances = np.asarray([path_to_numpy_array(tip) for tip in test_image_paths])
      8 
      9 test_instances[:2]

/tmp/ipykernel_11/2023033881.py in path_to_numpy_array(path)
      1 def path_to_numpy_array(path):
----> 2     tensor = img_to_tensor(path)
      3     array = tensor.numpy()
      4     return array
      5 

/tmp/ipykernel_11/789652232.py in img_to_tensor(img_path)
      1 def img_to_tensor(img_path):
----> 2     img_tensor = tf.cast(tf.image.decode_image(tf.io.read_file(img_path)), tf.float32)
      3     img_tensor /= 255.0 # normalized to [0,1]
      4     return img_tensor
      5 

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

FailedPreconditionError: {{function_node __wrapped__ReadFile_device_/job:localhost/replica:0/task:0/device:CPU:0}} ../input/test/test/test; Is a directory [Op:ReadFile]

## === cell 22
predictions = cnn_model.predict(test_instances)
print(len(predictions))


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/907658135.py in <cell line: 0>()
----> 1 predictions = cnn_model.predict(test_instances)
      2 print(len(predictions))

NameError: name 'test_instances' is not defined

## === cell 23
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0
predictions[:10]


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2583787901.py in <cell line: 0>()
      1 # try making binary predictions instead of probabilities
----> 2 predictions[predictions > 0.5] = 1
      3 predictions[predictions <= 0.5] = 0
      4 predictions[:10]

NameError: name 'predictions' is not defined

## === cell 24
submission_data = pd.DataFrame({'id': test_imgs, 'has_cactus': predictions.flatten()})
submission_data.head(20)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3888537199.py in <cell line: 0>()
----> 1 submission_data = pd.DataFrame({'id': test_imgs, 'has_cactus': predictions.flatten()})
      2 submission_data.head(20)

NameError: name 'predictions' is not defined

## === cell 25
submission_data.to_csv('submission_binary.csv', index=False)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2951886643.py in <cell line: 0>()
----> 1 submission_data.to_csv('submission_binary.csv', index=False)

NameError: name 'submission_data' is not defined

## === cell 26
!head submission.csv
