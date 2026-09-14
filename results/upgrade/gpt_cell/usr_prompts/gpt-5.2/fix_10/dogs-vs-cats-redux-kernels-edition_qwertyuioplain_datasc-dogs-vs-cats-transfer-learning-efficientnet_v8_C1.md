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

3.10

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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

# 3. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 4. Code solution

## === cell 0
import numpy as np
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

import glob

import tensorflow as tf

import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE

import torch


## === cell 1
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


## === cell 2
import zipfile
zip_df = zipfile.ZipFile("/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip", 'r')
zip_df.extractall("/kaggle/working/")
zip_df.close()
zip_df = zipfile.ZipFile("/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip", 'r')
zip_df.extractall("/kaggle/working/")
zip_df.close()


## === cell 3
dataset_path = './dataset'
if not os.path.isdir(dataset_path):
    os.mkdir(dataset_path)

train_path = os.path.join(dataset_path,'train')
val_path = os.path.join(dataset_path,'val')
test_path = os.path.join(dataset_path,'test')
if not os.path.isdir(train_path):
    os.mkdir(train_path)
if not os.path.isdir(val_path):
    os.mkdir(val_path)
if not os.path.isdir(test_path):
    os.mkdir(test_path)


## === cell 4
def get_path(path,name):
    return glob.glob(path+'/*.'+name)
check = lambda x: 1 if x.split('.')[1].split('/')[-1] == 'dog' else 0


## === cell 5
data_list = get_path('./train','jpg')
result = list(map(check,data_list))


## === cell 6
print('dogs:',result.count(1),'cats:',result.count(0))


## === cell 7
dogs_list = [i for i in data_list if check(i)]
cats_list = [i for i in data_list if not check(i)]


## === cell 8
split_ratio = 0.8


## === cell 9
train_data = []
val_data = []
train_label = []
val_label = []

n_pairs = min(len(dogs_list), len(cats_list))

for i in range(n_pairs):
    if i < len(data_list) / 2 * split_ratio:
        train_data.append(dogs_list[i])
        train_data.append(cats_list[i])
    else:
        val_data.append(dogs_list[i])
        val_data.append(cats_list[i])

train_label = list(map(check, train_data))
val_label = list(map(check, val_data))


## === cell 10
class_label = ['dog','cat']


## === cell 11
img_size = 224

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [img_size, img_size])

  return image


## === cell 12
def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)


## === cell 13
train_paths = tf.constant(train_data, dtype=tf.string)
val_paths = tf.constant(val_data, dtype=tf.string)
train_labels = tf.constant(train_label, dtype=tf.int32)
val_labels = tf.constant(val_label, dtype=tf.int32)

ds_train = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
ds_val = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))


def load_and_preprocess_from_path_label(path, label):
    path = tf.cast(path, tf.string)
    label = tf.cast(label, tf.int32)
    return load_and_preprocess_image(path), tf.one_hot(label, 2)


ds_train = ds_train.map(load_and_preprocess_from_path_label)
ds_val = ds_val.map(load_and_preprocess_from_path_label)


## === cell 14
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


## === cell 15
batch_size = 64
dsb_train = ds_train.batch(batch_size=batch_size, drop_remainder=True)
dsb_train = dsb_train.prefetch(tf.data.AUTOTUNE)

dsb_val = ds_val.batch(batch_size=batch_size, drop_remainder=True)


## === cell 16
model = EfficientNetB0(weights="imagenet")


## === cell 17
img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)


## === cell 18
def build_model(num_classes):
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = img_augmentation(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    model.trainable = False

    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


## === cell 19
strategy = tf.distribute.MirroredStrategy()


## === cell 20
steps_per_epoch = max(1, len(train_data) // batch_size)
validation_steps = max(1, len(val_data) // batch_size)

new_model = build_model(num_classes=2)

epochs = 10
hist = new_model.fit(
    dsb_train,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=dsb_val,
    validation_steps=validation_steps,
    verbose=2,
)


## === cell 21
def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


## === cell 22
def plot_hist(hist):
    history = hist.history if hasattr(hist, "history") else (hist or {})
    if not isinstance(history, dict):
        history = {}

    acc_candidates = [
        "accuracy",
        "acc",
        "categorical_accuracy",
        "sparse_categorical_accuracy",
    ]
    val_acc_candidates = [
        "val_accuracy",
        "val_acc",
        "val_categorical_accuracy",
        "val_sparse_categorical_accuracy",
    ]

    acc_key = next((k for k in acc_candidates if k in history), None)
    val_acc_key = next((k for k in val_acc_candidates if k in history), None)

    if acc_key is None:
        print(f"No accuracy metric found in history keys: {list(history.keys())}")
        return

    plt.plot(history[acc_key])
    if val_acc_key is not None:
        plt.plot(history[val_acc_key])
        plt.legend(["train", "validation"], loc="upper left")
    else:
        plt.legend(["train"], loc="upper left")

    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.show()


plot_hist(hist)


## === cell 23
test_list = get_path('./test','jpg')
id_load = lambda x : int(x.split('/')[-1].split('.')[0])
id_list = list(map(id_load,test_list))


## === cell 24
ds_test = tf.data.Dataset.from_tensor_slices((test_list,id_list))

def test(image,id):
    return load_and_preprocess_image(image),id

ds_test = ds_test.map(test)
dsb_test = ds_test.batch(batch_size=100, drop_remainder=True)


## --- ERROR in cell 24, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1252699254.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      4[0m     [0;32mreturn[0m [0mload_and_preprocess_image[0m[0;34m([0m[0mimage[0m[0;34m)[0m[0;34m,[0m[0mid[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;34m[0m[0m
[0;32m----> 6[0;31m [0mds_test[0m [0;34m=[0m [0mds_test[0m[0;34m.[0m[0mmap[0m[0;34m([0m[0mtest[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      7[0m [0mdsb_test[0m [0;34m=[0m [0mds_test[0m[0;34m.[0m[0mbatch[0m[0;34m([0m[0mbatch_size[0m[0;34m=[0m[0;36m100[0m[0;34m,[0m [0mdrop_remainder[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py[0m in [0;36mmap[0;34m(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)[0m
[1;32m   2339[0m     [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mdata[0m[0;34m.[0m[0mops[0m [0;32mimport[0m [0mmap_op[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2340[0m [0;34m[0m[0m
[0;32m-> 2341[0;31m     return map_op._map_v2(
[0m[1;32m   2342[0m         [0mself[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2343[0m         [0mmap_func[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py[0m in [0;36m_map_v2[0;34m(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)[0m
[1;32m     41[0m           [0;34m"`num_parallel_calls` argument is specified."[0m[0;34m[0m[0;34m[0m[0m
[1;32m     42[0m       )
[0;32m---> 43[0;31m     return _MapDataset(
[0m[1;32m     44[0m         [0minput_dataset[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     45[0m         [0mmap_func[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py[0m in [0;36m__init__[0;34m(self, input_dataset, map_func, force_synchronous, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, name)[0m
[1;32m    155[0m     [0mself[0m[0;34m.[0m[0m_use_inter_op_parallelism[0m [0;34m=[0m [0muse_inter_op_parallelism[0m[0;34m[0m[0;34m[0m[0m
[1;32m    156[0m     [0mself[0m[0;34m.[0m[0m_preserve_cardinality[0m [0;34m=[0m [0mpreserve_cardinality[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 157[0;31m     self._map_func = structured_function.StructuredFunctionWrapper(
[0m[1;32m    158[0m         [0mmap_func[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    159[0m         [0mself[0m[0;34m.[0m[0m_transformation_name[0m[0;34m([0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py[0m in [0;36m__init__[0;34m(self, func, transformation_name, dataset, input_classes, input_shapes, input_types, input_structure, add_to_graph, use_legacy_function, defun_kwargs)[0m
[1;32m    263[0m         [0mfn_factory[0m [0;34m=[0m [0mtrace_tf_function[0m[0;34m([0m[0mdefun_kwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    264[0m [0;34m[0m[0m
[0;32m--> 265[0;31m     [0mself[0m[0;34m.[0m[0m_function[0m [0;34m=[0m [0mfn_factory[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    266[0m     [0;31m# There is no graph to add in eager mode.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    267[0m     [0madd_to_graph[0m [0;34m&=[0m [0;32mnot[0m [0mcontext[0m[0;34m.[0m[0mexecuting_eagerly[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py[0m in [0;36mget_concrete_function[0;34m(self, *args, **kwargs)[0m
[1;32m   1249[0m   [0;32mdef[0m [0mget_concrete_function[0m[0;34m([0m[0mself[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1250[0m     [0;31m# Implements PolymorphicFunction.get_concrete_function.[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1251[0;31m     [0mconcrete[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_get_concrete_function_garbage_collected[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1252[0m     [0mconcrete[0m[0;34m.[0m[0m_garbage_collector[0m[0;34m.[0m[0mrelease[0m[0;34m([0m[0;34m)[0m  [0;31m# pylint: disable=protected-access[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1253[0m     [0;32mreturn[0m [0mconcrete[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py[0m in [0;36m_get_concrete_function_garbage_collected[0;34m(self, *args, **kwargs)[0m
[1;32m   1219[0m       [0;32mif[0m [0mself[0m[0;34m.[0m[0m_variable_creation_config[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1220[0m         [0minitializers[0m [0;34m=[0m [0;34m[[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1221[0;31m         [0mself[0m[0;34m.[0m[0m_initialize[0m[0;34m([0m[0margs[0m[0;34m,[0m [0mkwargs[0m[0;34m,[0m [0madd_initializers_to[0m[0;34m=[0m[0minitializers[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1222[0m         [0mself[0m[0;34m.[0m[0m_initialize_uninitialized_variables[0m[0;34m([0m[0minitializers[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1223[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py[0m in [0;36m_initialize[0;34m(self, args, kwds, add_initializers_to)[0m
[1;32m    694[0m     )
[1;32m    695[0m     [0;31m# Force the definition of the function for these arguments[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 696[0;31m     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
[0m[1;32m    697[0m         [0margs[0m[0;34m,[0m [0mkwds[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m_variable_creation_config[0m[0;34m[0m[0;34m[0m[0m
[1;32m    698[0m     )

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py[0m in [0;36mtrace_function[0;34m(args, kwargs, tracing_options)[0m
[1;32m    176[0m       [0mkwargs[0m [0;34m=[0m [0;34m{[0m[0;34m}[0m[0;34m[0m[0;34m[0m[0m
[1;32m    177[0m [0;34m[0m[0m
[0;32m--> 178[0;31m     concrete_function = _maybe_define_function(
[0m[1;32m    179[0m         [0margs[0m[0;34m,[0m [0mkwargs[0m[0;34m,[0m [0mtracing_options[0m[0;34m[0m[0;34m[0m[0m
[1;32m    180[0m     )

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py[0m in [0;36m_maybe_define_function[0;34m(args, kwargs, tracing_options)[0m
[1;32m    281[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    282[0m           [0mtarget_func_type[0m [0;34m=[0m [0mlookup_func_type[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 283[0;31m         concrete_function = _create_concrete_function(
[0m[1;32m    284[0m             [0mtarget_func_type[0m[0;34m,[0m [0mlookup_func_context[0m[0;34m,[0m [0mfunc_graph[0m[0;34m,[0m [0mtracing_options[0m[0;34m[0m[0;34m[0m[0m
[1;32m    285[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py[0m in [0;36m_create_concrete_function[0;34m(function_type, type_context, func_graph, tracing_options)[0m
[1;32m    308[0m       [0mattributes_lib[0m[0;34m.[0m[0mDISABLE_ACD[0m[0;34m,[0m [0;32mFalse[0m[0;34m[0m[0;34m[0m[0m
[1;32m    309[0m   )
[0;32m--> 310[0;31m   traced_func_graph = func_graph_module.func_graph_from_py_func(
[0m[1;32m    311[0m       [0mtracing_options[0m[0;34m.[0m[0mname[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    312[0m       [0mtracing_options[0m[0;34m.[0m[0mpython_function[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py[0m in [0;36mfunc_graph_from_py_func[0;34m(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)[0m
[1;32m   1057[0m [0;34m[0m[0m
[1;32m   1058[0m     [0m_[0m[0;34m,[0m [0moriginal_func[0m [0;34m=[0m [0mtf_decorator[0m[0;34m.[0m[0munwrap[0m[0;34m([0m[0mpython_func[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1059[0;31m     [0mfunc_outputs[0m [0;34m=[0m [0mpython_func[0m[0;34m([0m[0;34m*[0m[0mfunc_args[0m[0;34m,[0m [0;34m**[0m[0mfunc_kwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1060[0m [0;34m[0m[0m
[1;32m   1061[0m     [0;31m# invariant: `func_outputs` contains only Tensors, CompositeTensors,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py[0m in [0;36mwrapped_fn[0;34m(*args, **kwds)[0m
[1;32m    597[0m         [0;31m# the function a weak reference to itself to avoid a reference cycle.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    598[0m         [0;32mwith[0m [0mOptionalXlaContext[0m[0;34m([0m[0mcompile_with_xla[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 599[0;31m           [0mout[0m [0;34m=[0m [0mweak_wrapped_fn[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__wrapped__[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    600[0m         [0;32mreturn[0m [0mout[0m[0;34m[0m[0;34m[0m[0m
[1;32m    601[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py[0m in [0;36mwrapped_fn[0;34m(*args)[0m
[1;32m    229[0m       [0;31m# Note: wrapper_helper will apply autograph based on context.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    230[0m       [0;32mdef[0m [0mwrapped_fn[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m)[0m[0;34m:[0m  [0;31m# pylint: disable=missing-docstring[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 231[0;31m         [0mret[0m [0;34m=[0m [0mwrapper_helper[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    232[0m         [0mret[0m [0;34m=[0m [0mstructure[0m[0;34m.[0m[0mto_tensor_list[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_output_structure[0m[0;34m,[0m [0mret[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    233[0m         [0;32mreturn[0m [0;34m[[0m[0mops[0m[0;34m.[0m[0mconvert_to_tensor[0m[0;34m([0m[0mt[0m[0;34m)[0m [0;32mfor[0m [0mt[0m [0;32min[0m [0mret[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py[0m in [0;36mwrapper_helper[0;34m(*args)[0m
[1;32m    159[0m       [0;32mif[0m [0;32mnot[0m [0m_should_unpack[0m[0;34m([0m[0mnested_args[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    160[0m         [0mnested_args[0m [0;34m=[0m [0;34m([0m[0mnested_args[0m[0;34m,[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 161[0;31m       [0mret[0m [0;34m=[0m [0mautograph[0m[0;34m.[0m[0mtf_convert[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_func[0m[0;34m,[0m [0mag_ctx[0m[0;34m)[0m[0;34m([0m[0;34m*[0m[0mnested_args[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    162[0m       [0mret[0m [0;34m=[0m [0mvariable_utils[0m[0;34m.[0m[0mconvert_variables_to_tensors[0m[0;34m([0m[0mret[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    163[0m       [0;32mif[0m [0m_should_pack[0m[0;34m([0m[0mret[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py[0m in [0;36mwrapper[0;34m(*args, **kwargs)[0m
[1;32m    691[0m       [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me[0m[0;34m:[0m  [0;31m# pylint:disable=broad-except[0m[0;34m[0m[0;34m[0m[0m
[1;32m    692[0m         [0;32mif[0m [0mhasattr[0m[0;34m([0m[0me[0m[0;34m,[0m [0;34m'ag_error_metadata'[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 693[0;31m           [0;32mraise[0m [0me[0m[0;34m.[0m[0mag_error_metadata[0m[0;34m.[0m[0mto_exception[0m[0;34m([0m[0me[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    694[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    695[0m           [0;32mraise[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py[0m in [0;36mwrapper[0;34m(*args, **kwargs)[0m
[1;32m    688[0m       [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    689[0m         [0;32mwith[0m [0mconversion_ctx[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 690[0;31m           [0;32mreturn[0m [0mconverted_call[0m[0;34m([0m[0mf[0m[0;34m,[0m [0margs[0m[0;34m,[0m [0mkwargs[0m[0;34m,[0m [0moptions[0m[0;34m=[0m[0moptions[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    691[0m       [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me[0m[0;34m:[0m  [0;31m# pylint:disable=broad-except[0m[0;34m[0m[0;34m[0m[0m
[1;32m    692[0m         [0;32mif[0m [0mhasattr[0m[0;34m([0m[0me[0m[0;34m,[0m [0;34m'ag_error_metadata'[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py[0m in [0;36mconverted_call[0;34m(f, args, kwargs, caller_fn_scope, options)[0m
[1;32m    437[0m     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    438[0m       [0;32mif[0m [0mkwargs[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 439[0;31m         [0mresult[0m [0;34m=[0m [0mconverted_f[0m[0;34m([0m[0;34m*[0m[0meffective_args[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    440[0m       [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    441[0m         [0mresult[0m [0;34m=[0m [0mconverted_f[0m[0;34m([0m[0;34m*[0m[0meffective_args[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/__autograph_generated_fileqf41yxf3.py[0m in [0;36mtf__test[0;34m(image, id)[0m
[1;32m     10[0m                 [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     11[0m                     [0mdo_return[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 12[0;31m                     [0mretval_[0m [0;34m=[0m [0;34m([0m[0mag__[0m[0;34m.[0m[0mconverted_call[0m[0;34m([0m[0mag__[0m[0;34m.[0m[0mld[0m[0;34m([0m[0mload_and_preprocess_image[0m[0;34m)[0m[0;34m,[0m [0;34m([0m[0mag__[0m[0;34m.[0m[0mld[0m[0;34m([0m[0mimage[0m[0;34m)[0m[0;34m,[0m[0;34m)[0m[0;34m,[0m [0;32mNone[0m[0;34m,[0m [0mfscope[0m[0;34m)[0m[0;34m,[0m [0mag__[0m[0;34m.[0m[0mld[0m[0;34m([0m[0mid[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     13[0m                 [0;32mexcept[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     14[0m                     [0mdo_return[0m [0;34m=[0m [0;32mFalse[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py[0m in [0;36mconverted_call[0;34m(f, args, kwargs, caller_fn_scope, options)[0m
[1;32m    439[0m         [0mresult[0m [0;34m=[0m [0mconverted_f[0m[0;34m([0m[0;34m*[0m[0meffective_args[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    440[0m       [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 441[0;31m         [0mresult[0m [0;34m=[0m [0mconverted_f[0m[0;34m([0m[0;34m*[0m[0meffective_args[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    442[0m     [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    443[0m       [0m_attach_error_metadata[0m[0;34m([0m[0me[0m[0;34m,[0m [0mconverted_f[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/__autograph_generated_filetj6sk2r4.py[0m in [0;36mtf__load_and_preprocess_image[0;34m(path)[0m
[1;32m      8[0m                 [0mdo_return[0m [0;34m=[0m [0;32mFalse[0m[0;34m[0m[0;34m[0m[0m
[1;32m      9[0m                 [0mretval_[0m [0;34m=[0m [0mag__[0m[0;34m.[0m[0mUndefinedReturnValue[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 10[0;31m                 [0mimage[0m [0;34m=[0m [0mag__[0m[0;34m.[0m[0mconverted_call[0m[0;34m([0m[0mag__[0m[0;34m.[0m[0mld[0m[0;34m([0m[0mtf[0m[0;34m)[0m[0;34m.[0m[0mio[0m[0;34m.[0m[0mread_file[0m[0;34m,[0m [0;34m([0m[0mag__[0m[0;34m.[0m[0mld[0m[0;34m([0m[0mpath[0m[0;34m)[0m[0;34m,[0m[0;34m)[0m[0;34m,[0m [0;32mNone[0m[0;34m,[0m [0mfscope[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     11[0m                 [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     12[0m                     [0mdo_return[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py[0m in [0;36mconverted_call[0;34m(f, args, kwargs, caller_fn_scope, options)[0m
[1;32m    329[0m   [0;32mif[0m [0mconversion[0m[0;34m.[0m[0mis_in_allowlist_cache[0m[0;34m([0m[0mf[0m[0;34m,[0m [0moptions[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    330[0m     [0mlogging[0m[0;34m.[0m[0mlog[0m[0;34m([0m[0;36m2[0m[0;34m,[0m [0;34m'Allowlisted %s: from cache'[0m[0;34m,[0m [0mf[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 331[0;31m     [0;32mreturn[0m [0m_call_unconverted[0m[0;34m([0m[0mf[0m[0;34m,[0m [0margs[0m[0;34m,[0m [0mkwargs[0m[0;34m,[0m [0moptions[0m[0;34m,[0m [0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    332[0m [0;34m[0m[0m
[1;32m    333[0m   [0;32mif[0m [0mag_ctx[0m[0;34m.[0m[0mcontrol_status_ctx[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mstatus[0m [0;34m==[0m [0mag_ctx[0m[0;34m.[0m[0mStatus[0m[0;34m.[0m[0mDISABLED[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py[0m in [0;36m_call_unconverted[0;34m(f, args, kwargs, options, update_cache)[0m
[1;32m    458[0m   [0;32mif[0m [0mkwargs[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    459[0m     [0;32mreturn[0m [0mf[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 460[0;31m   [0;32mreturn[0m [0mf[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    461[0m [0;34m[0m[0m
[1;32m    462[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/io_ops.py[0m in [0;36mread_file[0;34m(filename, name)[0m
[1;32m    132[0m     [0mA[0m [0mtensor[0m [0mof[0m [0mdtype[0m [0;34m"string"[0m[0;34m,[0m [0;32mwith[0m [0mthe[0m [0mfile[0m [0mcontents[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    133[0m   """
[0;32m--> 134[0;31m   [0;32mreturn[0m [0mgen_io_ops[0m[0;34m.[0m[0mread_file[0m[0;34m([0m[0mfilename[0m[0;34m,[0m [0mname[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    135[0m [0;34m[0m[0m
[1;32m    136[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_io_ops.py[0m in [0;36mread_file[0;34m(filename, name)[0m
[1;32m    586[0m       [0;32mpass[0m  [0;31m# Add nodes to the TensorFlow graph.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    587[0m   [0;31m# Add nodes to the TensorFlow graph.[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 588[0;31m   _, _, _op, _outputs = _op_def_library._apply_op_helper(
[0m[1;32m    589[0m         "ReadFile", filename=filename, name=name)
[1;32m    590[0m   [0m_result[0m [0;34m=[0m [0m_outputs[0m[0;34m[[0m[0;34m:[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/op_def_library.py[0m in [0;36m_apply_op_helper[0;34m(op_type_name, name, **keywords)[0m
[1;32m    776[0m   [0;32mwith[0m [0mg[0m[0;34m.[0m[0mas_default[0m[0;34m([0m[0;34m)[0m[0;34m,[0m [0mops[0m[0;34m.[0m[0mname_scope[0m[0;34m([0m[0mname[0m[0;34m)[0m [0;32mas[0m [0mscope[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    777[0m     [0;32mif[0m [0mfallback[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 778[0;31m       _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map,
[0m[1;32m    779[0m                              [0mkeywords[0m[0;34m,[0m [0mdefault_type_attr_map[0m[0;34m,[0m [0mattrs[0m[0;34m,[0m [0minputs[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    780[0m                              input_types)

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/op_def_library.py[0m in [0;36m_ExtractInputsAndAttrs[0;34m(op_type_name, op_def, allowed_list_attr_map, keywords, default_type_attr_map, attrs, inputs, input_types)[0m
[1;32m    576[0m                   (input_name, op_type_name, observed))
[1;32m    577[0m         [0;32mif[0m [0minput_arg[0m[0;34m.[0m[0mtype[0m [0;34m!=[0m [0mtypes_pb2[0m[0;34m.[0m[0mDT_INVALID[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 578[0;31m           raise TypeError(f"{prefix} expected type of "
[0m[1;32m    579[0m                           f"{dtypes.as_dtype(input_arg.type).name}.")
[1;32m    580[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: in user code:

    File "/tmp/ipykernel_11/1252699254.py", line 4, in test  *
        return load_and_preprocess_image(image),id
    File "/tmp/ipykernel_11/3718408595.py", line 2, in load_and_preprocess_image  *
        image = tf.io.read_file(path)

    TypeError: Input 'filename' of 'ReadFile' Op has type float32 that does not match expected type of string.


## === cell 25
submission = {'id':[],'label':[]}
dog_prediction = lambda x:x[1]

for batch in dsb_test:
    results = new_model.predict(batch[0])
    id = batch[1].numpy()
    
    submission['id'].extend(id)
    submission['label'].extend(map(dog_prediction,results))
