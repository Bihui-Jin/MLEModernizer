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
    image = tf.image.resize(image, input_shape[:2])
    image = tf.cast(image, tf.float32)
    image.set_shape(input_shape)  # (32, 32, 3)
    return image, label


train_ds = train_ds.map(_fix_image_shape, num_parallel_calls=tf.data.AUTOTUNE)
valid_ds = valid_ds.map(_fix_image_shape, num_parallel_calls=tf.data.AUTOTUNE)

hist = model.fit(
    train_ds,
    validation_data=valid_ds,
    validation_steps=validation_steps,
    steps_per_epoch=steps_per_epoch,
    epochs=30,
)


## --- ERROR in cell 21, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/579297369.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      9[0m [0;34m[0m[0m
[1;32m     10[0m [0;34m[0m[0m
[0;32m---> 11[0;31m [0mtrain_ds[0m [0;34m=[0m [0mtrain_ds[0m[0;34m.[0m[0mmap[0m[0;34m([0m[0m_fix_image_shape[0m[0;34m,[0m [0mnum_parallel_calls[0m[0;34m=[0m[0mtf[0m[0;34m.[0m[0mdata[0m[0;34m.[0m[0mAUTOTUNE[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     12[0m [0mvalid_ds[0m [0;34m=[0m [0mvalid_ds[0m[0;34m.[0m[0mmap[0m[0;34m([0m[0m_fix_image_shape[0m[0;34m,[0m [0mnum_parallel_calls[0m[0;34m=[0m[0mtf[0m[0;34m.[0m[0mdata[0m[0;34m.[0m[0mAUTOTUNE[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     13[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py[0m in [0;36mmap[0;34m(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)[0m
[1;32m   2339[0m     [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mdata[0m[0;34m.[0m[0mops[0m [0;32mimport[0m [0mmap_op[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2340[0m [0;34m[0m[0m
[0;32m-> 2341[0;31m     return map_op._map_v2(
[0m[1;32m   2342[0m         [0mself[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2343[0m         [0mmap_func[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py[0m in [0;36m_map_v2[0;34m(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)[0m
[1;32m     55[0m           [0mnum_parallel_calls[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     56[0m       )
[0;32m---> 57[0;31m     return _ParallelMapDataset(
[0m[1;32m     58[0m         [0minput_dataset[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     59[0m         [0mmap_func[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py[0m in [0;36m__init__[0;34m(self, input_dataset, map_func, num_parallel_calls, deterministic, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, use_unbounded_threadpool, name)[0m
[1;32m    200[0m     [0mself[0m[0;34m.[0m[0m_input_dataset[0m [0;34m=[0m [0minput_dataset[0m[0;34m[0m[0;34m[0m[0m
[1;32m    201[0m     [0mself[0m[0;34m.[0m[0m_use_inter_op_parallelism[0m [0;34m=[0m [0muse_inter_op_parallelism[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 202[0;31m     self._map_func = structured_function.StructuredFunctionWrapper(
[0m[1;32m    203[0m         [0mmap_func[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    204[0m         [0mself[0m[0;34m.[0m[0m_transformation_name[0m[0;34m([0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

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

[0;32m/tmp/__autograph_generated_filek7d85lps.py[0m in [0;36mtf___fix_image_shape[0;34m(image, label)[0m
[1;32m      8[0m                 [0mdo_return[0m [0;34m=[0m [0;32mFalse[0m[0;34m[0m[0;34m[0m[0m
[1;32m      9[0m                 [0mretval_[0m [0;34m=[0m [0mag__[0m[0;34m.[0m[0mUndefinedReturnValue[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 10[0;31m                 [0mimage[0m [0;34m=[0m [0mag__[0m[0;34m.[0m[0mconverted_call[0m[0;34m([0m[0mag__[0m[0;34m.[0m[0mld[0m[0;34m([0m[0mtf[0m[0;34m)[0m[0;34m.[0m[0mimage[0m[0;34m.[0m[0mresize[0m[0;34m,[0m [0;34m([0m[0mag__[0m[0;34m.[0m[0mld[0m[0;34m([0m[0mimage[0m[0;34m)[0m[0;34m,[0m [0mag__[0m[0;34m.[0m[0mld[0m[0;34m([0m[0minput_shape[0m[0;34m)[0m[0;34m[[0m[0;34m:[0m[0;36m2[0m[0;34m][0m[0;34m)[0m[0;34m,[0m [0;32mNone[0m[0;34m,[0m [0mfscope[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     11[0m                 [0mimage[0m [0;34m=[0m [0mag__[0m[0;34m.[0m[0mconverted_call[0m[0;34m([0m[0mag__[0m[0;34m.[0m[0mld[0m[0;34m([0m[0mtf[0m[0;34m)[0m[0;34m.[0m[0mcast[0m[0;34m,[0m [0;34m([0m[0mag__[0m[0;34m.[0m[0mld[0m[0;34m([0m[0mimage[0m[0;34m)[0m[0;34m,[0m [0mag__[0m[0;34m.[0m[0mld[0m[0;34m([0m[0mtf[0m[0;34m)[0m[0;34m.[0m[0mfloat32[0m[0;34m)[0m[0;34m,[0m [0;32mNone[0m[0;34m,[0m [0mfscope[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     12[0m                 [0mag__[0m[0;34m.[0m[0mconverted_call[0m[0;34m([0m[0mag__[0m[0;34m.[0m[0mld[0m[0;34m([0m[0mimage[0m[0;34m)[0m[0;34m.[0m[0mset_shape[0m[0;34m,[0m [0;34m([0m[0mag__[0m[0;34m.[0m[0mld[0m[0;34m([0m[0minput_shape[0m[0;34m)[0m[0;34m,[0m[0;34m)[0m[0;34m,[0m [0;32mNone[0m[0;34m,[0m [0mfscope[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py[0m in [0;36mconverted_call[0;34m(f, args, kwargs, caller_fn_scope, options)[0m
[1;32m    375[0m [0;34m[0m[0m
[1;32m    376[0m   [0;32mif[0m [0;32mnot[0m [0moptions[0m[0;34m.[0m[0muser_requested[0m [0;32mand[0m [0mconversion[0m[0;34m.[0m[0mis_allowlisted[0m[0;34m([0m[0mf[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 377[0;31m     [0;32mreturn[0m [0m_call_unconverted[0m[0;34m([0m[0mf[0m[0;34m,[0m [0margs[0m[0;34m,[0m [0mkwargs[0m[0;34m,[0m [0moptions[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    378[0m [0;34m[0m[0m
[1;32m    379[0m   [0;31m# internal_convert_user_code is for example turned off when issuing a dynamic[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py[0m in [0;36m_call_unconverted[0;34m(f, args, kwargs, options, update_cache)[0m
[1;32m    458[0m   [0;32mif[0m [0mkwargs[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    459[0m     [0;32mreturn[0m [0mf[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 460[0;31m   [0;32mreturn[0m [0mf[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    461[0m [0;34m[0m[0m
[1;32m    462[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py[0m in [0;36merror_handler[0;34m(*args, **kwargs)[0m
[1;32m    151[0m     [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    152[0m       [0mfiltered_tb[0m [0;34m=[0m [0m_process_traceback_frames[0m[0;34m([0m[0me[0m[0;34m.[0m[0m__traceback__[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 153[0;31m       [0;32mraise[0m [0me[0m[0;34m.[0m[0mwith_traceback[0m[0;34m([0m[0mfiltered_tb[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    154[0m     [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    155[0m       [0;32mdel[0m [0mfiltered_tb[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/image_ops_impl.py[0m in [0;36m_resize_images_common[0;34m(images, resizer_fn, size, preserve_aspect_ratio, name, skip_resize_if_same)[0m
[1;32m   1460[0m     [0mimages[0m [0;34m=[0m [0mops[0m[0;34m.[0m[0mconvert_to_tensor[0m[0;34m([0m[0mimages[0m[0;34m,[0m [0mname[0m[0;34m=[0m[0;34m'images'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1461[0m     [0;32mif[0m [0mimages[0m[0;34m.[0m[0mget_shape[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mndims[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1462[0;31m       [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m'\'images\' contains no shape.'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1463[0m     [0;31m# TODO(shlens): Migrate this functionality to the underlying Op's.[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1464[0m     [0mis_batch[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: in user code:

    File "/tmp/ipykernel_11/579297369.py", line 5, in _fix_image_shape  *
        image = tf.image.resize(image, input_shape[:2])

    ValueError: 'images' contains no shape.


## === cell 22
test_df.head()
