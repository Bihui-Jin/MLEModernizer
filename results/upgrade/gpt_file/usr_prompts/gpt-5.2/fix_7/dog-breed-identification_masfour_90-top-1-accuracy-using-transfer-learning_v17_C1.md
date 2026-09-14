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
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.8

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
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.28106

# 6. Current score

5.68203

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 5.68203) has done: 'The timeout is dominated by (1) heavy Python-side image loading/augmentation via `ImageDataGenerator.flow_from_dataframe` and (2) the very expensive `Flatten()` of InceptionResNetV2 feature maps at 299×299, which makes the dense head extremely costly per step. To keep the exact same model architecture and training loop semantics, the main speedups come from eliminating input pipeline overhead (switch to `tf.data` with parallel decode/resize and deterministic seeding) while still applying the same geometric augmentations, and from enabling graph execution/XLA where safe. I also remove expensive per-epoch Python printing overhead from the callback (without changing training behavior) and ensure steps/ordering remain identical. These changes preserve the core model and loss while substantially reducing wall-clock time spent in Python and input I/O.'

# 9. Code solution

## === cell 0
import os
import shutil
import random
import numpy as np
import pandas as pd
import cv2

import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import clear_output

import tensorflow as tf
import tf_keras as keras
from tf_keras import backend as K
from tf_keras.layers import (
    Dense,
    Activation,
    Dropout,
    BatchNormalization,
    Input,
    Flatten,
    MaxPooling2D,
)
from tf_keras.models import Model
from tf_keras.optimizers import Adam
from tf_keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tf_keras.applications.inception_resnet_v2 import InceptionResNetV2
from tf_keras.initializers import he_normal
from tf_keras.preprocessing.image import ImageDataGenerator

os.environ["TF_DETERMINISTIC_OPS"] = "1"
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

try:
    tf.config.optimizer.set_jit(True)  # XLA compile where possible
except Exception as _:
    pass

INPUT_DIR = "../input/dog-breed-identification"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
LABELS_CSV = os.path.join(INPUT_DIR, "labels.csv")
SAMPLE_SUB_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

WORK_DIR = "/kaggle/working"
NEW_TRAIN_DIR = os.path.join(WORK_DIR, "new_train")
NEW_VALID_DIR = os.path.join(WORK_DIR, "new_valid")
NEW_TEST_DIR = os.path.join(WORK_DIR, "new_test")

print("TF version:", tf.__version__)
print("tf_keras version:", keras.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
print("Train dir exists:", os.path.isdir(TRAIN_DIR))
print("Test dir exists:", os.path.isdir(TEST_DIR))
print("Labels exists:", os.path.isfile(LABELS_CSV))
print("Sample submission exists:", os.path.isfile(SAMPLE_SUB_CSV))
print("Example train files:", sorted(os.listdir(TRAIN_DIR))[:5])



## === cell 2
labels = pd.read_csv(LABELS_CSV)
labels.head(5)



## === cell 3
classes = np.unique(labels.breed)
classes_num = classes.size
print("Num classes:", classes_num)
print("First 5 classes:", classes[:5])



## === cell 4
images_names = os.listdir(TRAIN_DIR)
images_num = len(images_names)
print(f"Number of train images: {images_num}")



## === cell 5
for d in [NEW_TRAIN_DIR, NEW_VALID_DIR, NEW_TEST_DIR]:
    os.makedirs(d, exist_ok=True)
print("Verified work split directories exist (no copy step):", WORK_DIR)



## === cell 6
labels_jpg = labels.copy(deep=True)
labels_jpg["filename"] = labels_jpg["id"].astype(str) + ".jpg"
labels_jpg["filepath"] = TRAIN_DIR + "/" + labels_jpg["filename"]
labels_jpg.head()



## === cell 7
test_split = 0.1
valid_split = 0.2

rng = np.random.RandomState(SEED)
rnd = rng.rand(len(labels_jpg))
split = np.full(len(labels_jpg), "train", dtype=object)
split[rnd <= test_split] = "test"
split[(rnd > test_split) & (rnd <= (test_split + valid_split))] = "valid"
labels_jpg["split"] = split

filepaths = labels_jpg["filepath"].to_numpy()
exists_mask = np.array([os.path.isfile(p) for p in filepaths], dtype=bool)
if not exists_mask.all():
    labels_jpg = labels_jpg.loc[exists_mask].reset_index(drop=True)

train_df = labels_jpg[labels_jpg["split"] == "train"][
    ["filename", "breed"]
].reset_index(drop=True)
valid_df = labels_jpg[labels_jpg["split"] == "valid"][
    ["filename", "breed"]
].reset_index(drop=True)
test_df = labels_jpg[labels_jpg["split"] == "test"][["filename", "breed"]].reset_index(
    drop=True
)

print("Split sizes:", len(train_df), len(valid_df), len(test_df))



## === cell 8
width, height, channels_num = 299, 299, 3

images_samples = np.zeros((4, height, width, 3), dtype=float)
samples_labels = []

rnd_indexes = np.random.randint(0, images_num, 4)
for i, rnd_idx in enumerate(rnd_indexes):
    img_filename = images_names[rnd_idx]
    img_id = img_filename[:-4]
    img_bgr = cv2.imread(os.path.join(TRAIN_DIR, img_filename))
    img_rgb = img_bgr[:, :, [2, 1, 0]]
    images_samples[i] = cv2.resize(src=img_rgb, dsize=(width, height)) / 255.0
    img_label = labels.breed[labels.id == img_id].values[0]
    samples_labels.append(img_label)

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
for ax, img, label in zip(axs.ravel(), images_samples, samples_labels):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Class: {label}", size=12)
plt.show()



## === cell 9
transform_params = {
    "featurewise_center": False,
    "featurewise_std_normalization": False,
    "samplewise_center": False,
    "samplewise_std_normalization": False,
    "rotation_range": 30,
    "width_shift_range": 0.15,
    "height_shift_range": 0.15,
    "horizontal_flip": True,
    "rescale": 1 / 255.0,
}

img_gen = ImageDataGenerator(**transform_params)
img_feed = ImageDataGenerator(rescale=1 / 255.0)




## === cell 10
class Plotter(Callback):
    def on_train_begin(self, logs=None):
        self.losses, self.val_losses, self.epochs = [], [], []
        self.acc, self.val_acc = [], []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.acc.append(logs.get("acc", logs.get("accuracy")))
        self.val_acc.append(logs.get("val_acc", logs.get("val_accuracy")))
        self.epochs.append(epoch)
        e = epoch + 1
        tr_acc = (self.acc[-1] or 0.0) * 100.0
        va_acc = (self.val_acc[-1] or 0.0) * 100.0
        print(
            f"Epoch #{e} >> train_acc={tr_acc:.3f}%, train_loss={self.losses[-1]:.5f}"
        )
        print(
            f"Epoch #{e} >> val_acc={va_acc:.3f}%, val_loss={self.val_losses[-1]:.5f}"
        )


plotter = Plotter()



## === cell 11
plateau_reduce = ReduceLROnPlateau(
    monitor="val_loss", factor=0.01, patience=1, min_lr=1e-20
)
e_stop = EarlyStopping(
    monitor="val_loss", patience=15, mode="min", restore_best_weights=True
)
callbacks = [plotter, plateau_reduce, e_stop]




## === cell 12
def dense_block(x, neurons, layer_no):
    x = Dense(
        neurons, kernel_initializer=he_normal(seed=layer_no), name=f"topDense{layer_no}"
    )(x)
    x = Activation("relu", name=f"Relu{layer_no}")(x)
    x = BatchNormalization(name=f"BatchNorm{layer_no}")(x)
    x = Dropout(0.5, name=f"Dropout{layer_no}")(x)
    return x




## === cell 13
def create_model(shape):
    input_layer = Input(shape, name="input_layer")
    incep_res = InceptionResNetV2(
        include_top=False, weights="imagenet", input_tensor=input_layer
    )
    for layer in incep_res.layers:
        layer.trainable = False

    pool = MaxPooling2D(pool_size=[3, 3], strides=[3, 3], padding="same")(
        incep_res.output
    )
    flat1 = Flatten(name="Flatten1")(pool)
    flat1_bn = BatchNormalization(name="BatchNormFlat")(flat1)

    dens1 = dense_block(flat1_bn, neurons=512, layer_no=1)
    dens2 = dense_block(dens1, neurons=512, layer_no=2)
    dens3 = dense_block(dens2, neurons=1024, layer_no=3)

    dens_final = Dense(classes_num, name="Dense4")(dens3)
    output_layer = Activation("softmax", name="Softmax")(dens_final)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    return model




## === cell 14
learning_rate = 0.004
epochs = 15
batch_size = 32

model = create_model((height, width, channels_num))
optimizer = Adam(learning_rate=learning_rate)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["acc"],
    run_eagerly=False,
)
model.summary()



## === cell 15
AUTOTUNE = tf.data.AUTOTUNE

class_to_index = {c: i for i, c in enumerate(classes.tolist())}


def _read_decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


def _augment(img, seed):
    seed = tf.cast(seed, tf.int32)
    img = tf.image.stateless_random_flip_left_right(img, seed=seed)
    seed2 = seed + tf.constant([11, 17], dtype=tf.int32)
    dx = tf.random.stateless_uniform(
        [], seed=seed2, minval=-0.15, maxval=0.15, dtype=tf.float32
    )
    seed3 = seed + tf.constant([23, 29], dtype=tf.int32)
    dy = tf.random.stateless_uniform(
        [], seed=seed3, minval=-0.15, maxval=0.15, dtype=tf.float32
    )
    tx = tf.cast(tf.round(dx * tf.cast(width, tf.float32)), tf.int32)
    ty = tf.cast(tf.round(dy * tf.cast(height, tf.float32)), tf.int32)
    img = tf.roll(img, shift=[ty, tx], axis=[0, 1])
    seed4 = seed + tf.constant([37, 41], dtype=tf.int32)
    angle = tf.random.stateless_uniform(
        [], seed=seed4, minval=-30.0, maxval=30.0, dtype=tf.float32
    ) * (np.pi / 180.0)
    img = tf.image.rotate(
        img, angles=angle, interpolation="NEAREST", fill_mode="REFLECT"
    )
    return img


def make_dataset(df, training):
    paths = tf.constant([os.path.join(TRAIN_DIR, fn) for fn in df["filename"].tolist()])
    labels_idx = tf.constant(
        [class_to_index[b] for b in df["breed"].tolist()], dtype=tf.int32
    )

    ds = tf.data.Dataset.from_tensor_slices((paths, labels_idx))
    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(p, y):
        img = _read_decode_resize(p)
        if training:
            s = tf.strings.to_hash_bucket_fast(p, 2**31 - 1)
            seed = tf.stack([tf.cast(s, tf.int32), tf.cast(SEED, tf.int32)], axis=0)
            img_aug = _augment(img, seed)
            img = img_aug
        y_oh = tf.one_hot(y, depth=classes_num, dtype=tf.float32)
        return img, y_oh

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(train_df, training=True)
valid_ds = make_dataset(valid_df, training=False)

steps_per_epoch = int(np.ceil(len(train_df) / batch_size))
val_steps = int(np.ceil(len(valid_df) / batch_size))
print("steps_per_epoch:", steps_per_epoch, "val_steps:", val_steps)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4108486957.py in <cell line: 0>()
     75 
     76 
---> 77 train_ds = make_dataset(train_df, training=True)
     78 valid_ds = make_dataset(valid_df, training=False)
     79 

/tmp/ipykernel_11/4108486957.py in make_dataset(df, training)
     69         return img, y_oh
     70 
---> 71     ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
     72     ds = ds.batch(batch_size, drop_remainder=False)
     73     ds = ds.prefetch(AUTOTUNE)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in map(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
   2339     from tensorflow.python.data.ops import map_op
   2340 
-> 2341     return map_op._map_v2(
   2342         self,
   2343         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in _map_v2(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
     55           num_parallel_calls,
     56       )
---> 57     return _ParallelMapDataset(
     58         input_dataset,
     59         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in __init__(self, input_dataset, map_func, num_parallel_calls, deterministic, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, use_unbounded_threadpool, name)
    200     self._input_dataset = input_dataset
    201     self._use_inter_op_parallelism = use_inter_op_parallelism
--> 202     self._map_func = structured_function.StructuredFunctionWrapper(
    203         map_func,
    204         self._transformation_name(),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in __init__(self, func, transformation_name, dataset, input_classes, input_shapes, input_types, input_structure, add_to_graph, use_legacy_function, defun_kwargs)
    263         fn_factory = trace_tf_function(defun_kwargs)
    264 
--> 265     self._function = fn_factory()
    266     # There is no graph to add in eager mode.
    267     add_to_graph &= not context.executing_eagerly()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in get_concrete_function(self, *args, **kwargs)
   1249   def get_concrete_function(self, *args, **kwargs):
   1250     # Implements PolymorphicFunction.get_concrete_function.
-> 1251     concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
   1252     concrete._garbage_collector.release()  # pylint: disable=protected-access
   1253     return concrete

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _get_concrete_function_garbage_collected(self, *args, **kwargs)
   1219       if self._variable_creation_config is None:
   1220         initializers = []
-> 1221         self._initialize(args, kwargs, add_initializers_to=initializers)
   1222         self._initialize_uninitialized_variables(initializers)
   1223 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _initialize(self, args, kwds, add_initializers_to)
    694     )
    695     # Force the definition of the function for these arguments
--> 696     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
    697         args, kwds, self._variable_creation_config
    698     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in trace_function(args, kwargs, tracing_options)
    176       kwargs = {}
    177 
--> 178     concrete_function = _maybe_define_function(
    179         args, kwargs, tracing_options
    180     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _maybe_define_function(args, kwargs, tracing_options)
    281         else:
    282           target_func_type = lookup_func_type
--> 283         concrete_function = _create_concrete_function(
    284             target_func_type, lookup_func_context, func_graph, tracing_options
    285         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _create_concrete_function(function_type, type_context, func_graph, tracing_options)
    308       attributes_lib.DISABLE_ACD, False
    309   )
--> 310   traced_func_graph = func_graph_module.func_graph_from_py_func(
    311       tracing_options.name,
    312       tracing_options.python_function,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)
   1057 
   1058     _, original_func = tf_decorator.unwrap(python_func)
-> 1059     func_outputs = python_func(*func_args, **func_kwargs)
   1060 
   1061     # invariant: `func_outputs` contains only Tensors, CompositeTensors,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in wrapped_fn(*args, **kwds)
    597         # the function a weak reference to itself to avoid a reference cycle.
    598         with OptionalXlaContext(compile_with_xla):
--> 599           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    600         return out
    601 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapped_fn(*args)
    229       # Note: wrapper_helper will apply autograph based on context.
    230       def wrapped_fn(*args):  # pylint: disable=missing-docstring
--> 231         ret = wrapper_helper(*args)
    232         ret = structure.to_tensor_list(self._output_structure, ret)
    233         return [ops.convert_to_tensor(t) for t in ret]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapper_helper(*args)
    159       if not _should_unpack(nested_args):
    160         nested_args = (nested_args,)
--> 161       ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
    162       ret = variable_utils.convert_variables_to_tensors(ret)
    163       if _should_pack(ret):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):
--> 693           raise e.ag_error_metadata.to_exception(e)
    694         else:
    695           raise

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    688       try:
    689         with conversion_ctx:
--> 690           return converted_call(f, args, kwargs, options=options)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_filewktfq_s6.py in tf___map_fn(p, y)
     31                 img_aug = ag__.Undefined('img_aug')
     32                 seed = ag__.Undefined('seed')
---> 33                 ag__.if_stmt(ag__.ld(training), if_body, else_body, get_state, set_state, ('img',), 1)
     34                 y_oh = ag__.converted_call(ag__.ld(tf).one_hot, (ag__.ld(y),), dict(depth=ag__.ld(classes_num), dtype=ag__.ld(tf).float32), fscope)
     35                 try:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/operators/control_flow.py in if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
   1215     _tf_if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
   1216   else:
-> 1217     _py_if_stmt(cond, body, orelse)
   1218 
   1219 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/operators/control_flow.py in _py_if_stmt(cond, body, orelse)
   1268 def _py_if_stmt(cond, body, orelse):
   1269   """Overload of if_stmt that executes a Python if statement."""
-> 1270   return body() if cond else orelse()

/tmp/__autograph_generated_filewktfq_s6.py in if_body()
     22                     s = ag__.converted_call(ag__.ld(tf).strings.to_hash_bucket_fast, (ag__.ld(p), 2 ** 31 - 1), None, fscope)
     23                     seed = ag__.converted_call(ag__.ld(tf).stack, ([ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(s), ag__.ld(tf).int32), None, fscope), ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(SEED), ag__.ld(tf).int32), None, fscope)],), dict(axis=0), fscope)
---> 24                     img_aug = ag__.converted_call(ag__.ld(_augment), (ag__.ld(img), ag__.ld(seed)), None, fscope)
     25                     img = ag__.ld(img_aug)
     26 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filerzo4iz4k.py in tf___augment(img, seed)
     19                 seed4 = ag__.ld(seed) + ag__.converted_call(ag__.ld(tf).constant, ([37, 41],), dict(dtype=ag__.ld(tf).int32), fscope)
     20                 angle = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([],), dict(seed=ag__.ld(seed4), minval=-30.0, maxval=30.0, dtype=ag__.ld(tf).float32), fscope) * (ag__.ld(np).pi / 180.0)
---> 21                 img = ag__.converted_call(ag__.ld(tf).image.rotate, (ag__.ld(img),), dict(angles=ag__.ld(angle), interpolation='NEAREST', fill_mode='REFLECT'), fscope)
     22                 try:
     23                     do_return = True

AttributeError: in user code:

    File "/tmp/ipykernel_11/4108486957.py", line 66, in _map_fn  *
        img_aug = _augment(img, seed)
    File "/tmp/ipykernel_11/4108486957.py", line 43, in _augment  *
        img = tf.image.rotate(

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'rotate'


## === cell 16
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    callbacks=callbacks,
    verbose=1,
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1184289778.py in <cell line: 0>()
      1 history = model.fit(
----> 2     train_ds,
      3     validation_data=valid_ds,
      4     epochs=epochs,
      5     steps_per_epoch=steps_per_epoch,

NameError: name 'train_ds' is not defined

## === cell 17
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
breed_cols = [c for c in sample_sub.columns if c != "id"]

sample_ids = sample_sub["id"].astype(str).tolist()
test_filenames = [f"{i}.jpg" for i in sample_ids]

missing = [
    fn for fn in test_filenames if not os.path.isfile(os.path.join(TEST_DIR, fn))
]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images (first 5): {missing[:5]}"
    )

test_df = pd.DataFrame({"id": sample_ids, "filename": test_filenames})
test_ids = test_df["id"].tolist()
print("Num test images:", len(test_ids), "Expected:", len(sample_sub))


def make_test_dataset(df):
    paths = tf.constant([os.path.join(TEST_DIR, fn) for fn in df["filename"].tolist()])
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map_fn(p):
        img = _read_decode_resize(p)
        return img

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


test_ds = make_test_dataset(test_df)



## === cell 18
preds = model.predict(
    test_ds,
    steps=int(np.ceil(len(test_ids) / batch_size)),
    verbose=1,
)
preds = np.asarray(preds, dtype=np.float64)[: len(test_ids)]

pred_df = pd.DataFrame(preds, columns=list(classes))
pred_df.insert(0, "id", test_ids)

sub = sample_sub[["id"]].merge(pred_df, on="id", how="left")

for c in breed_cols:
    if c not in sub.columns:
        sub[c] = 1.0 / len(breed_cols)

sub = sub[["id"] + breed_cols]

prob = sub[breed_cols].to_numpy(dtype=np.float64)
nan_mask = ~np.isfinite(prob)
if nan_mask.any():
    prob[nan_mask] = 1.0 / len(breed_cols)

row_sums = prob.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
prob = prob / row_sums
sub.loc[:, breed_cols] = prob

out_path = os.path.join(WORK_DIR, "submission.csv")
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub.head())
print("Submission shape:", sub.shape, "Expected:", sample_sub.shape)
print("Row sums (min/max):", prob.sum(axis=1).min(), prob.sum(axis=1).max())
