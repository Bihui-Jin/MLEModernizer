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

3.7

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
pass


## === cell 1
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
)

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
valid_ids = set(train_lables_df.index.astype(str))
train_imgs = [ti for ti in train_imgs if ti in valid_ids]

train_image_paths = [train_dir + ti for ti in train_imgs]
train_image_labels = [train_lables_df.loc[ti]["has_cactus"] for ti in train_imgs]

for i in range(10):
    print(train_image_paths[i], train_image_labels[i])


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


## === cell 9
mini_train_ds = build_training_dataset(X_train[:5], y_train[:5], batch_size=2)
for images, labels in mini_train_ds.take(1):
    print(images)
    print(labels)


## === cell 10
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


## === cell 11
def img_to_tensor(img_path):
    img_tensor = tf.cast(tf.image.decode_image(tf.io.read_file(img_path)), tf.float32)
    img_tensor.set_shape([32, 32, 3])
    img_tensor /= 255.0  # normalized to [0,1]
    return img_tensor


train_ds = build_training_dataset(X_train, y_train)
validation_ds = build_validation_dataset(X_valid, y_valid)

history = model.fit(
    train_ds, epochs=20, steps_per_epoch=400, validation_data=validation_ds
)


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


## === cell 16
plot_accuracies_and_losses(history)


## === cell 17
layer_outputs = [layer.output for layer in cnn_model.layers]

try:
    _ = cnn_model.inputs  # will raise if not built/called
except Exception:
    cnn_model.build((None, 32, 32, 3))

activation_model = tf.keras.Model(inputs=cnn_model.inputs, outputs=layer_outputs)


def print_example_and_activations(example, col_size, row_size, act_index):
    example_array = img_to_tensor(example).numpy()
    plt.imshow(example_array)
    activations = activation_model.predict(
        example_array.reshape(1, 32, 32, 3)
    )  # batch of 1 - just the example array
    activation = activations[act_index]
    activation_index = 0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size * 2.5, col_size * 1.5))
    for row in range(0, row_size):
        for col in range(0, col_size):
            ax[row][col].imshow(
                activation[0, :, :, activation_index], cmap="gray"
            )  # image for the first (and only) element in the batch at activation_index
            activation_index += 1


## === cell 18
print_example_and_activations(X_train[0], 8, 4, 1)


## === cell 19
print_example_and_activations(X_train[0], 8, 4, 3)


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
[0;31m---------------------------------------------------------------------------[0m
[0;31mFailedPreconditionError[0m                   Traceback (most recent call last)
[0;32m/tmp/ipykernel_10/2023033881.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      5[0m [0;34m[0m[0m
[1;32m      6[0m [0mtest_image_paths[0m [0;34m=[0m [0;34m[[0m[0mtest_dir[0m [0;34m+[0m [0mti[0m [0;32mfor[0m [0mti[0m [0;32min[0m [0mtest_imgs[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 7[0;31m [0mtest_instances[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0masarray[0m[0;34m([0m[0;34m[[0m[0mpath_to_numpy_array[0m[0;34m([0m[0mtip[0m[0;34m)[0m [0;32mfor[0m [0mtip[0m [0;32min[0m [0mtest_image_paths[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      8[0m [0;34m[0m[0m
[1;32m      9[0m [0mtest_instances[0m[0;34m[[0m[0;34m:[0m[0;36m2[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_10/2023033881.py[0m in [0;36m<listcomp>[0;34m(.0)[0m
[1;32m      5[0m [0;34m[0m[0m
[1;32m      6[0m [0mtest_image_paths[0m [0;34m=[0m [0;34m[[0m[0mtest_dir[0m [0;34m+[0m [0mti[0m [0;32mfor[0m [0mti[0m [0;32min[0m [0mtest_imgs[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 7[0;31m [0mtest_instances[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0masarray[0m[0;34m([0m[0;34m[[0m[0mpath_to_numpy_array[0m[0;34m([0m[0mtip[0m[0;34m)[0m [0;32mfor[0m [0mtip[0m [0;32min[0m [0mtest_image_paths[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      8[0m [0;34m[0m[0m
[1;32m      9[0m [0mtest_instances[0m[0;34m[[0m[0;34m:[0m[0;36m2[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_10/2023033881.py[0m in [0;36mpath_to_numpy_array[0;34m(path)[0m
[1;32m      1[0m [0;32mdef[0m [0mpath_to_numpy_array[0m[0;34m([0m[0mpath[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m     [0mtensor[0m [0;34m=[0m [0mimg_to_tensor[0m[0;34m([0m[0mpath[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m     [0marray[0m [0;34m=[0m [0mtensor[0m[0;34m.[0m[0mnumpy[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m     [0;32mreturn[0m [0marray[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_10/259292629.py[0m in [0;36mimg_to_tensor[0;34m(img_path)[0m
[1;32m      3[0m [0;31m# images are 32x32 RGB, so we explicitly set the tensor shape after decoding.[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mdef[0m [0mimg_to_tensor[0m[0;34m([0m[0mimg_path[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m     [0mimg_tensor[0m [0;34m=[0m [0mtf[0m[0;34m.[0m[0mcast[0m[0;34m([0m[0mtf[0m[0;34m.[0m[0mimage[0m[0;34m.[0m[0mdecode_image[0m[0;34m([0m[0mtf[0m[0;34m.[0m[0mio[0m[0;34m.[0m[0mread_file[0m[0;34m([0m[0mimg_path[0m[0;34m)[0m[0;34m)[0m[0;34m,[0m [0mtf[0m[0;34m.[0m[0mfloat32[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m     [0mimg_tensor[0m[0;34m.[0m[0mset_shape[0m[0;34m([0m[0;34m[[0m[0;36m32[0m[0;34m,[0m [0;36m32[0m[0;34m,[0m [0;36m3[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m     [0mimg_tensor[0m [0;34m/=[0m [0;36m255.0[0m  [0;31m# normalized to [0,1][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/io_ops.py[0m in [0;36mread_file[0;34m(filename, name)[0m
[1;32m    132[0m     [0mA[0m [0mtensor[0m [0mof[0m [0mdtype[0m [0;34m"string"[0m[0;34m,[0m [0;32mwith[0m [0mthe[0m [0mfile[0m [0mcontents[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    133[0m   """
[0;32m--> 134[0;31m   [0;32mreturn[0m [0mgen_io_ops[0m[0;34m.[0m[0mread_file[0m[0;34m([0m[0mfilename[0m[0;34m,[0m [0mname[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    135[0m [0;34m[0m[0m
[1;32m    136[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_io_ops.py[0m in [0;36mread_file[0;34m(filename, name)[0m
[1;32m    581[0m       [0;32mpass[0m[0;34m[0m[0;34m[0m[0m
[1;32m    582[0m     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 583[0;31m       return read_file_eager_fallback(
[0m[1;32m    584[0m           filename, name=name, ctx=_ctx)
[1;32m    585[0m     [0;32mexcept[0m [0m_core[0m[0;34m.[0m[0m_SymbolicException[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_io_ops.py[0m in [0;36mread_file_eager_fallback[0;34m(filename, name, ctx)[0m
[1;32m    604[0m   [0m_inputs_flat[0m [0;34m=[0m [0;34m[[0m[0mfilename[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m    605[0m   [0m_attrs[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 606[0;31m   _result = _execute.execute(b"ReadFile", 1, inputs=_inputs_flat,
[0m[1;32m    607[0m                              attrs=_attrs, ctx=ctx, name=name)
[1;32m    608[0m   [0;32mif[0m [0m_execute[0m[0;34m.[0m[0mmust_record_gradient[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py[0m in [0;36mquick_execute[0;34m(op_name, num_outputs, inputs, attrs, ctx, name)[0m
[1;32m     51[0m   [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     52[0m     [0mctx[0m[0;34m.[0m[0mensure_initialized[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 53[0;31m     tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
[0m[1;32m     54[0m                                         inputs, attrs, num_outputs)
[1;32m     55[0m   [0;32mexcept[0m [0mcore[0m[0;34m.[0m[0m_NotOkStatusException[0m [0;32mas[0m [0me[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mFailedPreconditionError[0m: {{function_node __wrapped__ReadFile_device_/job:localhost/replica:0/task:0/device:CPU:0}} ../input/test/test/test; Is a directory [Op:ReadFile]

## === cell 22
predictions = cnn_model.predict(test_instances)
print(len(predictions))
