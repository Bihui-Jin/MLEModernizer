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

3.9

# 3. Installed packages

No external packages required in the script and installed.

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

5.15761

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, shutil, math, random, collections
import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import keras
from pathlib import Path


def mkdir_if_not_exist(paths):
    """Create directories from a list of path components."""
    dir_path = os.path.join(*paths)
    os.makedirs(dir_path, exist_ok=True)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
src = "/kaggle/input/dog-breed-identification"
dst = "/kaggle/working/dog-breed-identification"
if not os.path.isdir(dst):
    shutil.copytree(src, dst)




## === cell 2
def reorg_train_valid(data_dir, train_dir, input_dir, valid_ratio, idx_label):
    min_n_train_per_label = collections.Counter(idx_label.values()).most_common()[
        :-2:-1
    ][0][1]
    n_valid_per_label = math.floor(min_n_train_per_label * valid_ratio)
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, train_dir)):
        idx = train_file.split(".")[0]
        label = idx_label[idx]
        mkdir_if_not_exist([data_dir, input_dir, "train_valid", label])
        shutil.copy(
            os.path.join(data_dir, train_dir, train_file),
            os.path.join(data_dir, input_dir, "train_valid", label),
        )
        if label not in label_count or label_count[label] < n_valid_per_label:
            mkdir_if_not_exist([data_dir, input_dir, "valid", label])
            shutil.copy(
                os.path.join(data_dir, train_dir, train_file),
                os.path.join(data_dir, input_dir, "valid", label),
            )
            label_count[label] = label_count.get(label, 0) + 1
        else:
            mkdir_if_not_exist([data_dir, input_dir, "train", label])
            shutil.copy(
                os.path.join(data_dir, train_dir, train_file),
                os.path.join(data_dir, input_dir, "train", label),
            )


def reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio):
    with open(os.path.join(data_dir, label_file), "r") as f:
        lines = f.readlines()[1:]  # skip header
        tokens = [l.rstrip().split(",") for l in lines]
        idx_label = {idx: breed for idx, breed in tokens}
    reorg_train_valid(data_dir, train_dir, input_dir, valid_ratio, idx_label)
    mkdir_if_not_exist([data_dir, input_dir, "test", "unknown"])
    for test_file in os.listdir(os.path.join(data_dir, test_dir)):
        shutil.copy(
            os.path.join(data_dir, test_dir, test_file),
            os.path.join(data_dir, input_dir, "test", "unknown"),
        )




## === cell 3
data_dir = "/kaggle/working/dog-breed-identification"
label_file, train_dir, test_dir = "labels.csv", "train", "test"
input_dir, batch_size, valid_ratio = "train_valid_test", 128, 0.1
reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1978689302.py in <cell line: 0>()
      2 label_file, train_dir, test_dir = "labels.csv", "train", "test"
      3 input_dir, batch_size, valid_ratio = "train_valid_test", 128, 0.1
----> 4 reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio)
      5 

/tmp/ipykernel_11/4258561204.py in reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio)
     35         tokens = [l.rstrip().split(",") for l in lines]
     36         idx_label = {idx: breed for idx, breed in tokens}
---> 37     reorg_train_valid(data_dir, train_dir, input_dir, valid_ratio, idx_label)
     38     mkdir_if_not_exist([data_dir, input_dir, "test", "unknown"])
     39     for test_file in os.listdir(os.path.join(data_dir, test_dir)):

/tmp/ipykernel_11/4258561204.py in reorg_train_valid(data_dir, train_dir, input_dir, valid_ratio, idx_label)
      8     for train_file in os.listdir(os.path.join(data_dir, train_dir)):
      9         idx = train_file.split(".")[0]
---> 10         label = idx_label[idx]
     11         # train/valid split directories
     12         mkdir_if_not_exist([data_dir, input_dir, "train_valid", label])

KeyError: 'train'

## === cell 4
data_root = os.path.join(data_dir, input_dir)
train_path = Path(os.path.join(data_root, "train"))
valid_path = Path(os.path.join(data_root, "valid"))
train_valid_path = Path(os.path.join(data_root, "train_valid"))
test_path = Path(os.path.join(data_root, "test"))

label_names = sorted([p.name for p in train_path.glob("*/") if p.is_dir()])
label_to_index = {name: idx for idx, name in enumerate(label_names)}


def list_images(root):
    return [str(p) for p in root.glob("*/*") if p.is_file()]


train_imgs = list_images(train_path)
valid_imgs = list_images(valid_path)
train_valid_imgs = list_images(train_valid_path)
test_imgs = list_images(test_path)

train_labels = [label_to_index[Path(p).parent.name] for p in train_imgs]
valid_labels = [label_to_index[Path(p).parent.name] for p in valid_imgs]
train_valid_labels = [label_to_index[Path(p).parent.name] for p in train_valid_imgs]
test_labels = [-1] * len(test_imgs)  # placeholder

print("Sample train labels:", train_labels[:5])
print("Sample test paths count:", len(test_imgs))




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1877078965.py in <cell line: 0>()
     20 
     21 train_labels = [label_to_index[Path(p).parent.name] for p in train_imgs]
---> 22 valid_labels = [label_to_index[Path(p).parent.name] for p in valid_imgs]
     23 train_valid_labels = [label_to_index[Path(p).parent.name] for p in train_valid_imgs]
     24 test_labels = [-1] * len(test_imgs)  # placeholder

/tmp/ipykernel_11/1877078965.py in <listcomp>(.0)
     20 
     21 train_labels = [label_to_index[Path(p).parent.name] for p in train_imgs]
---> 22 valid_labels = [label_to_index[Path(p).parent.name] for p in valid_imgs]
     23 train_valid_labels = [label_to_index[Path(p).parent.name] for p in train_valid_imgs]
     24 test_labels = [-1] * len(test_imgs)  # placeholder

KeyError: 'english_setter'

## === cell 5
def transform_train(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [400, 400])
    seed = random.randint(8, 100) / 100.0
    new_h = int(seed * tf.shape(img)[0])
    new_w = int(seed * tf.shape(img)[1])
    img = tf.image.random_crop(img, size=[new_h, new_w, 3])
    img = tf.image.resize(img, [224, 224])
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = img / 255.0
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    img = (img - mean) / std
    return img, label


def transform_test(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    img = (img - mean) / std
    return img, label




## === cell 6
train_ds = (
    tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))
    .map(transform_train, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(buffer=len(train_imgs))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices((valid_imgs, valid_labels))
    .map(transform_train, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(buffer=len(valid_imgs))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

train_valid_ds = (
    tf.data.Dataset.from_tensor_slices((train_valid_imgs, train_valid_labels))
    .map(transform_train, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(buffer=len(train_valid_imgs))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((test_imgs, test_labels))
    .map(transform_test, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3064656384.py in <cell line: 0>()
      1 train_ds = (
      2     tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))
----> 3     .map(transform_train, num_parallel_calls=tf.data.AUTOTUNE)
      4     .shuffle(buffer=len(train_imgs))
      5     .batch(batch_size)

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

/tmp/__autograph_generated_file8_1k4v1c.py in tf__transform_train(img_path, label)
     12                 img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), [400, 400]), None, fscope)
     13                 seed = ag__.converted_call(ag__.ld(random).randint, (8, 100), None, fscope) / 100.0
---> 14                 new_h = ag__.converted_call(ag__.ld(int), (ag__.ld(seed) * ag__.converted_call(ag__.ld(tf).shape, (ag__.ld(img),), None, fscope)[0],), None, fscope)
     15                 new_w = ag__.converted_call(ag__.ld(int), (ag__.ld(seed) * ag__.converted_call(ag__.ld(tf).shape, (ag__.ld(img),), None, fscope)[1],), None, fscope)
     16                 img = ag__.converted_call(ag__.ld(tf).image.random_crop, (ag__.ld(img),), dict(size=[ag__.ld(new_h), ag__.ld(new_w), 3]), fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor_util.py in _AssertCompatible(values, dtype)
    405       raise TypeError("Expected any non-tensor type, but got a tensor instead.")
    406     else:
--> 407       raise TypeError(f"Expected {dtype.name}, but got {mismatch} of type "
    408                       f"'{type(mismatch).__name__}'.")
    409 

TypeError: in user code:

    File "/tmp/ipykernel_11/651791258.py", line 6, in transform_train  *
        new_h = int(seed * tf.shape(img)[0])

    TypeError: Expected int32, but got 0.88 of type 'float'.


## === cell 7
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(label_names), activation="softmax"),
    ]
)
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## === cell 8
model.fit(train_ds, epochs=3, validation_data=valid_ds)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2767122012.py in <cell line: 0>()
      1 # Train briefly (more epochs can be added later)
----> 2 model.fit(train_ds, epochs=3, validation_data=valid_ds)
      3 

NameError: name 'train_ds' is not defined

## === cell 9
model.fit(train_valid_ds, epochs=2)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/426211359.py in <cell line: 0>()
      1 # Train on the combined train+valid set for a few more epochs
----> 2 model.fit(train_valid_ds, epochs=2)
      3 

NameError: name 'train_valid_ds' is not defined

## === cell 10
probabilities = model.predict(test_ds)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1980824901.py in <cell line: 0>()
----> 1 probabilities = model.predict(test_ds)
      2 

NameError: name 'test_ds' is not defined

## === cell 11
sample_sub_path = "/kaggle/working/dog-breed-identification/sample_submission.csv"
df = pd.read_csv(sample_sub_path)
for i, col in enumerate(df.columns[1:]):
    df[col] = probabilities[:, i]
df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv with shape:", df.shape)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/437863865.py in <cell line: 0>()
      4 # Ensure column order matches sample submission
      5 for i, col in enumerate(df.columns[1:]):
----> 6     df[col] = probabilities[:, i]
      7 df.to_csv("submission.csv", index=False)
      8 print("Submission saved to submission.csv with shape:", df.shape)

NameError: name 'probabilities' is not defined
