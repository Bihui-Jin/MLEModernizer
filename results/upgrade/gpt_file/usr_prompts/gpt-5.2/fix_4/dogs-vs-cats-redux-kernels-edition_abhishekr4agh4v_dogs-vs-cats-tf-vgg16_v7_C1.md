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
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.10

# 3. Installed packages



# 4. Data file paths

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

# 5. Target score

17.18945

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import shutil

print("TensorFlow:", tf.__version__)

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass
try:
    tf.config.optimizer.set_jit(True)  # XLA
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
import zipfile

train_zip = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
test_zip = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip"


def unzip_to_working(zip_path, dest="/kaggle/working"):
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)


unzip_to_working(train_zip, "/kaggle/working")
unzip_to_working(test_zip, "/kaggle/working")


def first_dir_containing_jpg(root_candidates):
    for root in root_candidates:
        if not root or not os.path.isdir(root):
            continue
        for cur, _, files in os.walk(root):
            for fn in files:
                if fn.lower().endswith(".jpg"):
                    return cur
    return None


candidate_train_roots = [
    "/kaggle/working",
    "/kaggle/working/train",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition",
]
candidate_test_roots = [
    "/kaggle/working",
    "/kaggle/working/test",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition",
]

train_dir = first_dir_containing_jpg(candidate_train_roots)
test_dir = first_dir_containing_jpg(candidate_test_roots)

preferred_train = [
    "/kaggle/working/train/train",
    "/kaggle/working/train",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition/train/train",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition/train",
]
preferred_test = [
    "/kaggle/working/test/test",
    "/kaggle/working/test",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition/test/test",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition/test",
]


def first_existing_with_jpg_recursive(paths):
    for p in paths:
        if os.path.isdir(p):
            for cur, _, files in os.walk(p):
                if any(f.lower().endswith(".jpg") for f in files):
                    return cur
    return None


train_dir = first_existing_with_jpg_recursive(preferred_train) or train_dir
test_dir = first_existing_with_jpg_recursive(preferred_test) or test_dir

print("Resolved train_dir:", train_dir)
print("Resolved test_dir :", test_dir)
assert train_dir is not None, "Could not find extracted train directory with jpgs."
assert test_dir is not None, "Could not find extracted test directory with jpgs."




## === cell 3
def list_jpg_sorted(d):
    out = []
    with os.scandir(d) as it:
        for e in it:
            if e.is_file() and e.name.lower().endswith(".jpg"):
                out.append(e.name)
    out.sort()
    return out


datasets_train = list_jpg_sorted(train_dir)
datasets_test = list_jpg_sorted(test_dir)

print("n_train:", len(datasets_train), "n_test:", len(datasets_test))
print("train sample:", datasets_train[:5])
print("test sample :", datasets_test[:5])



## === cell 4
s = pd.Series(datasets_train, name="imagename")
labels = pd.Series(
    np.where(
        s.str.contains("dog"), "dog", np.where(s.str.contains("cat"), "cat", None)
    ),
    name="labels",
)
dfx = pd.concat([s, labels], axis=1).dropna().reset_index(drop=True)
dftest = pd.DataFrame({"image": datasets_test}).reset_index(drop=True)

print(dfx.head())
print(dftest.head())




## === cell 5
def show_image(imageadd):
    image = cv2.imread(imageadd)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.title(os.path.basename(imageadd))
    plt.imshow(image)
    plt.axis("off")


if len(dfx) > 0:
    show_image(os.path.join(train_dir, dfx.loc[0, "imagename"]))



## === cell 6
bs = 32
IMG_H, IMG_W = 180, 200

class_indices = {"cat": 0, "dog": 1}
print("class_indices:", class_indices)

train_files = (pd.Series(train_dir) + "/" + dfx["imagename"].astype(str)).to_numpy()
train_labels_str = dfx["labels"].astype(str).to_numpy()

label_ids = np.where(train_labels_str == "cat", 0, 1).astype(np.int32)
y_onehot = tf.one_hot(label_ids, depth=2, dtype=tf.float32)

train_ds = tf.data.Dataset.from_tensor_slices((train_files, y_onehot))


def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR)
    return img


def _augment(img):
    img = tf.image.random_flip_left_right(img)

    dx = tf.random.uniform([], -0.1, 0.1) * tf.cast(IMG_W, tf.float32)
    dy = tf.random.uniform([], -0.1, 0.1) * tf.cast(IMG_H, tf.float32)
    dx_i = tf.cast(tf.round(dx), tf.int32)
    dy_i = tf.cast(tf.round(dy), tf.int32)
    img = tf.roll(img, shift=[dy_i, dx_i], axis=[0, 1])

    zoom = tf.random.uniform([], 0.8, 1.2)
    new_h = tf.cast(tf.round(tf.cast(IMG_H, tf.float32) / zoom), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(IMG_W, tf.float32) / zoom), tf.int32)
    new_h = tf.clip_by_value(new_h, 1, IMG_H)
    new_w = tf.clip_by_value(new_w, 1, IMG_W)
    img = tf.image.resize_with_crop_or_pad(img, new_h, new_w)
    img = tf.image.resize(img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR)

    return img


rot_layer = tf.keras.layers.RandomRotation(factor=25.0 / 360.0, fill_mode="nearest")


def _process_train(path, y):
    img = _decode_resize(path)
    img = _augment(img)
    img = rot_layer(img, training=True)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img, y


train_ds = (
    train_ds.shuffle(
        buffer_size=len(train_files), reshuffle_each_iteration=True, seed=42
    )
    .map(_process_train, num_parallel_calls=AUTOTUNE)
    .batch(bs, drop_remainder=False)
    .prefetch(AUTOTUNE)
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2680820204.py in <cell line: 0>()
     11 y_onehot = tf.one_hot(label_ids, depth=2, dtype=tf.float32)
     12 
---> 13 train_ds = tf.data.Dataset.from_tensor_slices((train_files, y_onehot))
     14 
     15 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in from_tensor_slices(tensors, name)
    825     # pylint: disable=g-import-not-at-top,protected-access
    826     from tensorflow.python.data.ops import from_tensor_slices_op
--> 827     return from_tensor_slices_op._from_tensor_slices(tensors, name)
    828     # pylint: enable=g-import-not-at-top,protected-access
    829 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_tensor_slices_op.py in _from_tensor_slices(tensors, name)
     23 
     24 def _from_tensor_slices(tensors, name=None):
---> 25   return _TensorSliceDataset(tensors, name=name)
     26 
     27 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_tensor_slices_op.py in __init__(self, element, is_files, name)
     31   def __init__(self, element, is_files=False, name=None):
     32     """See `Dataset.from_tensor_slices` for details."""
---> 33     element = structure.normalize_element(element)
     34     batched_spec = structure.type_spec_from_value(element)
     35     self._tensors = structure.to_batched_tensor_list(batched_spec, element)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/structure.py in normalize_element(element, element_signature)
    132           dtype = getattr(spec, "dtype", None)
    133           normalized_components.append(
--> 134               ops.convert_to_tensor(t, name="component_%d" % i, dtype=dtype))
    135   return nest.pack_sequence_as(pack_as, normalized_components)
    136 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/profiler/trace.py in wrapped(*args, **kwargs)
    181         with Trace(trace_name, **trace_kwargs):
    182           return func(*args, **kwargs)
--> 183       return func(*args, **kwargs)
    184 
    185     return wrapped

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in convert_to_tensor(value, dtype, name, as_ref, preferred_dtype, dtype_hint, ctx, accepted_result_types)
    730   # TODO(b/142518781): Fix all call-sites and remove redundant arg
    731   preferred_dtype = preferred_dtype or dtype_hint
--> 732   return tensor_conversion_registry.convert(
    733       value, dtype, name, as_ref, preferred_dtype, accepted_result_types
    734   )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor_conversion_registry.py in convert(value, dtype, name, as_ref, preferred_dtype, accepted_result_types)
    232 
    233     if ret is None:
--> 234       ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
    235 
    236     if ret is NotImplemented:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_tensor_conversion.py in _constant_tensor_conversion_function(v, dtype, name, as_ref)
     27 
     28   _ = as_ref
---> 29   return constant_op.constant(v, dtype=dtype, name=name)
     30 
     31 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/weak_tensor_ops.py in wrapper(*args, **kwargs)
    140   def wrapper(*args, **kwargs):
    141     if not ops.is_auto_dtype_conversion_enabled():
--> 142       return op(*args, **kwargs)
    143     bound_arguments = signature.bind(*args, **kwargs)
    144     bound_arguments.apply_defaults()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in constant(value, dtype, shape, name)
    274     ValueError: if called on a symbolic tensor.
    275   """
--> 276   return _constant_impl(value, dtype, shape, name, verify_shape=False,
    277                         allow_broadcast=True)
    278 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in _constant_impl(value, dtype, shape, name, verify_shape, allow_broadcast)
    287       with trace.Trace("tf.constant"):
    288         return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
--> 289     return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
    290 
    291   const_tensor = ops._create_graph_constant(  # pylint: disable=protected-access

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
    299 ) -> ops._EagerTensorBase:
    300   """Creates a constant on the current device."""
--> 301   t = convert_to_eager_tensor(value, ctx, dtype)
    302   if shape is None:
    303     return t

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in convert_to_eager_tensor(value, ctx, dtype)
    106       dtype = dtypes.as_dtype(dtype).as_datatype_enum
    107   ctx.ensure_initialized()
--> 108   return ops.EagerTensor(value, ctx.device_name, dtype)
    109 
    110 

ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type float).

## === cell 7
VGG16transfer = tf.keras.applications.VGG16(
    include_top=False,
    input_shape=(180, 200, 3),
    weights="imagenet",
)
for layer in VGG16transfer.layers:
    layer.trainable = False

flat1 = tf.keras.layers.Flatten()(VGG16transfer.output)
d1 = tf.keras.layers.Dense(32, activation="relu")(flat1)
pred = tf.keras.layers.Dense(2, activation="softmax")(d1)

model1 = tf.keras.Model(inputs=[VGG16transfer.input], outputs=[pred])
model1.summary()



## === cell 8
model1.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.0031415),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=["acc"],
)

history = model1.fit(train_ds, epochs=1)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/775087491.py in <cell line: 0>()
      5 )
      6 
----> 7 history = model1.fit(train_ds, epochs=1)
      8 

NameError: name 'train_ds' is not defined

## === cell 9
test_files = (pd.Series(test_dir) + "/" + dftest["image"].astype(str)).to_numpy()
test_ds = tf.data.Dataset.from_tensor_slices(test_files)


def _process_test(path):
    img = _decode_resize(path)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img


test_ds = (
    test_ds.map(_process_test, num_parallel_calls=AUTOTUNE)
    .batch(bs, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

predict = model1.predict(test_ds, verbose=1)
print("predict shape:", predict.shape)

dog_col = class_indices.get("dog", 1)
dog_prob = predict[:, dog_col].astype(np.float64)

ids = dftest["image"].str.replace(".jpg", "", regex=False).astype(int)
result = pd.DataFrame({"id": ids, "label": dog_prob})
result = result.sort_values("id").reset_index(drop=True)

assert list(result.columns) == ["id", "label"]
assert result["id"].is_monotonic_increasing
assert len(result) == len(dftest)

result.to_csv("submission.csv", index=False)
print(result.head())
print("Wrote submission.csv with", len(result), "rows")

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/757571937.py in <cell line: 0>()
      1 test_files = (pd.Series(test_dir) + "/" + dftest["image"].astype(str)).to_numpy()
----> 2 test_ds = tf.data.Dataset.from_tensor_slices(test_files)
      3 
      4 
      5 def _process_test(path):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in from_tensor_slices(tensors, name)
    825     # pylint: disable=g-import-not-at-top,protected-access
    826     from tensorflow.python.data.ops import from_tensor_slices_op
--> 827     return from_tensor_slices_op._from_tensor_slices(tensors, name)
    828     # pylint: enable=g-import-not-at-top,protected-access
    829 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_tensor_slices_op.py in _from_tensor_slices(tensors, name)
     23 
     24 def _from_tensor_slices(tensors, name=None):
---> 25   return _TensorSliceDataset(tensors, name=name)
     26 
     27 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_tensor_slices_op.py in __init__(self, element, is_files, name)
     31   def __init__(self, element, is_files=False, name=None):
     32     """See `Dataset.from_tensor_slices` for details."""
---> 33     element = structure.normalize_element(element)
     34     batched_spec = structure.type_spec_from_value(element)
     35     self._tensors = structure.to_batched_tensor_list(batched_spec, element)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/structure.py in normalize_element(element, element_signature)
    132           dtype = getattr(spec, "dtype", None)
    133           normalized_components.append(
--> 134               ops.convert_to_tensor(t, name="component_%d" % i, dtype=dtype))
    135   return nest.pack_sequence_as(pack_as, normalized_components)
    136 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/profiler/trace.py in wrapped(*args, **kwargs)
    181         with Trace(trace_name, **trace_kwargs):
    182           return func(*args, **kwargs)
--> 183       return func(*args, **kwargs)
    184 
    185     return wrapped

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in convert_to_tensor(value, dtype, name, as_ref, preferred_dtype, dtype_hint, ctx, accepted_result_types)
    730   # TODO(b/142518781): Fix all call-sites and remove redundant arg
    731   preferred_dtype = preferred_dtype or dtype_hint
--> 732   return tensor_conversion_registry.convert(
    733       value, dtype, name, as_ref, preferred_dtype, accepted_result_types
    734   )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor_conversion_registry.py in convert(value, dtype, name, as_ref, preferred_dtype, accepted_result_types)
    232 
    233     if ret is None:
--> 234       ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
    235 
    236     if ret is NotImplemented:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_tensor_conversion.py in _constant_tensor_conversion_function(v, dtype, name, as_ref)
     27 
     28   _ = as_ref
---> 29   return constant_op.constant(v, dtype=dtype, name=name)
     30 
     31 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/weak_tensor_ops.py in wrapper(*args, **kwargs)
    140   def wrapper(*args, **kwargs):
    141     if not ops.is_auto_dtype_conversion_enabled():
--> 142       return op(*args, **kwargs)
    143     bound_arguments = signature.bind(*args, **kwargs)
    144     bound_arguments.apply_defaults()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in constant(value, dtype, shape, name)
    274     ValueError: if called on a symbolic tensor.
    275   """
--> 276   return _constant_impl(value, dtype, shape, name, verify_shape=False,
    277                         allow_broadcast=True)
    278 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in _constant_impl(value, dtype, shape, name, verify_shape, allow_broadcast)
    287       with trace.Trace("tf.constant"):
    288         return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
--> 289     return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
    290 
    291   const_tensor = ops._create_graph_constant(  # pylint: disable=protected-access

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
    299 ) -> ops._EagerTensorBase:
    300   """Creates a constant on the current device."""
--> 301   t = convert_to_eager_tensor(value, ctx, dtype)
    302   if shape is None:
    303     return t

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in convert_to_eager_tensor(value, ctx, dtype)
    106       dtype = dtypes.as_dtype(dtype).as_datatype_enum
    107   ctx.ensure_initialized()
--> 108   return ops.EagerTensor(value, ctx.device_name, dtype)
    109 
    110 

ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type float).
